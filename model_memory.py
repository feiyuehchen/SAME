"""
Memory Network with Optimal Transport regularization
Implements memory-augmented supervised learning for deepfake detection with Sinkhorn algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Tuple
import numpy as np

from model_titanet import TitaNetEncoder
from loss import OTMemoryLoss


class OTMemoryTitaNet(pl.LightningModule):
    """
    TitaNet + Memory Network with Optimal Transport
    
    Architecture for supervised binary classification (bonafide vs spoof) with memory reconstruction:
    1. TitaNet encoder: Extract speaker embeddings from raw audio
    2. Memory bank: Learnable prototypes representing bonafide acoustic patterns
    3. Reconstruction: Reconstruct embeddings using memory bank (low error = bonafide)
    4. Sinkhorn OT: Regularize bonafide samples to be evenly distributed across memory slots
    
    Training uses both bonafide and spoof samples:
    - Bonafide: Minimize reconstruction error + OT regularization
    - Spoof: Maximize reconstruction error (hinge loss with margin)
    
    Inference uses reconstruction error as anomaly score (no labels needed)
    """
    
    def __init__(
        self,
        titanet_model: str = "titanet_small",
        embed_dim: int = 192,
        memory_slots: int = 64,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_recon: float = 1.0,
        lambda_ot: float = 0.1,
        margin: float = 1.0,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 0.05,
        logit_scale_init: float = 1.0,
        asv_score_path: Optional[str] = None
    ):
        """
        Args:
            titanet_model: Name of TitaNet model
            embed_dim: Embedding dimension
            memory_slots: Number of memory prototypes
            freeze_encoder: Whether to freeze TitaNet encoder
            lr: Learning rate
            lambda_recon: Weight for reconstruction loss
            lambda_ot: Weight for OT loss
            margin: Hinge loss margin
            sinkhorn_iterations: Number of Sinkhorn iterations
            sinkhorn_epsilon: Entropy regularization parameter
            logit_scale_init: Initial value for learnable logit temperature scaling
            asv_score_path: Path to ASV scores for t-DCF calculation
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.lr = lr
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.asv_score_path = asv_score_path
        
        # TitaNet encoder
        self.encoder = TitaNetEncoder(
            model_name=titanet_model,
            freeze=freeze_encoder,
            embed_dim=embed_dim
        )
        
        # Memory bank: learnable prototypes
        self.memory_bank = nn.Parameter(
            torch.randn(memory_slots, embed_dim)
        )
        # Initialize with normalized vectors
        with torch.no_grad():
            self.memory_bank.data = F.normalize(self.memory_bank.data, p=2, dim=1)
        
        # Learnable temperature scaling for logits (used in OT assignment)
        # This allows the model to control the sharpness of the distribution
        # Keep initial value low (e.g., 1.0) to prevent numerical instability (exp overflow in Sinkhorn)
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale_init)
        
        # Loss function
        self.loss_fn = OTMemoryLoss(
            lambda_recon=lambda_recon,
            lambda_ot=lambda_ot,
            margin=margin
        )
        
        # For validation: collect predictions
        self.validation_step_outputs = []
    
    def sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for optimal transport
        
        Iteratively normalize rows and columns of the cost matrix to get
        doubly-stochastic matrix (soft assignment)
        
        Args:
            logits: Assignment logits, shape (B, K)
        
        Returns:
            Q: Soft assignment matrix, shape (B, K)
        """
        B, K = logits.shape
        
        # Numerical stability: clamp logits before division to prevent overflow in exp
        # Max value chosen to ensure exp((max_logit - min_logit) / epsilon) < 1e30
        logits_clamped = torch.clamp(logits / self.sinkhorn_epsilon, min=-50, max=50)
        
        # Initialize Q with exponential of logits / epsilon
        Q = torch.exp(logits_clamped)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iterations):
            # Normalize rows (sum over columns = 1)
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
            
            # Normalize columns (sum over rows = 1)
            Q = Q / (Q.sum(dim=0, keepdim=True) + eps)
        
        # Final row normalization to ensure each row sums to 1
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        
        return Q
    
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        compute_ot: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            audio: Raw audio waveform, shape (B, T)
            audio_lengths: Audio lengths, shape (B,)
            compute_ot: Whether to compute OT (only during training)
        
        Returns:
            Dictionary containing:
                - z: Normalized embeddings from encoder, shape (B, embed_dim)
                - recon_error: Reconstruction error per sample, shape (B,)
                - logits: OT logits (if compute_ot), shape (B, memory_slots)
                - Q: Sinkhorn assignment (if compute_ot), shape (B, memory_slots)
        """
        # Extract embeddings from TitaNet
        z = self.encoder(audio, audio_lengths)  # (B, embed_dim)
        
        # Normalize memory bank
        memory_normalized = F.normalize(self.memory_bank, p=2, dim=1)
        
        # Compute similarity between embeddings and memory bank
        # Cosine similarity: z @ memory.T
        similarity = torch.matmul(z, memory_normalized.t())  # (B, memory_slots)
        
        # Compute attention weights (softmax over memory slots)
        attn_weights = F.softmax(similarity, dim=1)  # (B, memory_slots)
        
        # Reconstruct embeddings using weighted sum of memory
        z_recon = torch.matmul(attn_weights, memory_normalized)  # (B, embed_dim)
        
        # Compute reconstruction error (L2 distance)
        recon_error = torch.sum((z - z_recon) ** 2, dim=1)  # (B,)
        
        output = {
            'z': z,
            'recon_error': recon_error,
            'attn_weights': attn_weights
        }
        
        # Compute OT assignment if requested (training only)
        if compute_ot:
            # Use scaled similarity as logits for OT assignment
            # This ensures OT loss directly affects memory_bank distribution
            logits = self.logit_scale * similarity  # (B, memory_slots)
            Q = self.sinkhorn(logits)  # (B, memory_slots)
            
            output['logits'] = logits
            output['Q'] = Q
        
        return output
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        audio, labels = batch
        
        # Forward pass
        outputs = self.forward(audio, compute_ot=True)
        
        # Separate bonafide and spoof for OT loss
        is_bonafide = (labels == 0)
        
        if is_bonafide.sum() > 0:
            logits_bonafide = outputs['logits'][is_bonafide]
            Q_bonafide = outputs['Q'][is_bonafide]
        else:
            logits_bonafide = None
            Q_bonafide = None
        
        # Compute loss
        loss_dict = self.loss_fn(
            recon_error=outputs['recon_error'],
            logits=logits_bonafide,
            Q=Q_bonafide,
            labels=labels,
            use_ot=True
        )
        
        # Log metrics (step-based)
        self.log('train/loss', loss_dict['loss'], on_step=True, on_epoch=False, 
                 prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=False,
                 sync_dist=True)
        self.log('train/recon_loss_bonafide', loss_dict['recon_loss_bonafide'], 
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/recon_loss_spoof', loss_dict['recon_loss_spoof'], 
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/ot_loss', loss_dict['ot_loss'], on_step=True, on_epoch=False,
                 sync_dist=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step - collect predictions for EER calculation"""
        # Handle both cases: with or without utt_ids
        if len(batch) == 3:
            audio, labels, utt_ids = batch
        else:
            audio, labels = batch
            utt_ids = None
        
        # Forward pass (no OT during validation)
        outputs = self.forward(audio, compute_ot=False)
        
        # Store predictions
        # Score: -recon_error (higher score = more bonafide)
        # Use negative error directly for better score separation
        scores = -outputs['recon_error']
        
        # Collect for epoch-end evaluation
        self.validation_step_outputs.append({
            'scores': scores.detach().cpu(),
            'labels': labels.detach().cpu(),
            'utt_ids': utt_ids if utt_ids is not None else [],
            'recon_errors': outputs['recon_error'].detach().cpu()
        })
    
    def on_validation_epoch_end(self):
        """Compute EER and t-DCF at end of validation"""
        if len(self.validation_step_outputs) == 0:
            return
        
        # Gather all predictions
        all_scores = torch.cat([x['scores'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_recon_errors = torch.cat([x['recon_errors'] for x in self.validation_step_outputs])
        
        # Convert to numpy
        scores_np = all_scores.numpy()
        labels_np = all_labels.numpy()
        
        # Separate bonafide and spoof
        bonafide_mask = (labels_np == 0)
        spoof_mask = (labels_np == 1)
        
        bonafide_scores = scores_np[bonafide_mask]
        spoof_scores = scores_np[spoof_mask]
        
        bonafide_errors = all_recon_errors[bonafide_mask]
        spoof_errors = all_recon_errors[spoof_mask]
        
        # Check if we have both classes (needed for EER calculation)
        if bonafide_mask.sum() == 0 or spoof_mask.sum() == 0:
            # During sanity check or if only one class present, skip EER
            # Just log reconstruction errors (convert to tensor on current device)
            if bonafide_mask.sum() > 0:
                self.log('val/avg_recon_error_bonafide', bonafide_errors.mean().item(), 
                         on_step=False, on_epoch=True, sync_dist=False)
            if spoof_mask.sum() > 0:
                self.log('val/avg_recon_error_spoof', spoof_errors.mean().item(), 
                         on_step=False, on_epoch=True, sync_dist=False)
            self.validation_step_outputs.clear()
            return
        
        # Compute EER using official ASVspoof 2019 implementation
        try:
            from eval_metrics import compute_eer as compute_eer_official
            
            # Official implementation expects:
            # target_scores = bonafide (higher scores)
            # nontarget_scores = spoof (lower scores)
            eer, threshold = compute_eer_official(bonafide_scores, spoof_scores)
            
            # Log metrics (use sync_dist=False since EER is already computed on all data)
            self.log('val/eer', eer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            self.log('val/eer_threshold', threshold, on_step=False, on_epoch=True, sync_dist=False)
        except Exception as e:
            # If EER calculation fails (e.g., during sanity check), just skip it
            print(f"Warning: Could not compute EER: {e}")
            import traceback
            traceback.print_exc()
        
        # Always log reconstruction errors (convert to Python float to avoid DDP sync issues)
        self.log('val/avg_recon_error_bonafide', bonafide_errors.mean().item(), 
                 on_step=False, on_epoch=True, sync_dist=False)
        self.log('val/avg_recon_error_spoof', spoof_errors.mean().item(), 
                 on_step=False, on_epoch=True, sync_dist=False)
        
        # TODO: Compute t-DCF if ASV scores are available
        # This would require loading ASV scores and computing tandem DCF
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        return optimizer


def test_model():
    """Test model forward pass with dummy data"""
    print("Testing OTMemoryTitaNet...")
    
    # Create dummy data
    batch_size = 4
    audio_length = 64000  # 4 seconds @ 16kHz
    
    dummy_audio = torch.randn(batch_size, audio_length)
    dummy_labels = torch.tensor([0, 0, 1, 1])  # 2 bonafide, 2 spoof
    
    try:
        # Create model
        model = OTMemoryTitaNet(
            titanet_model="titanet_small",
            embed_dim=192,
            memory_slots=64,
            freeze_encoder=False
        )
        
        print("Model created successfully")
        print(f"Memory bank shape: {model.memory_bank.shape}")
        
        # Test forward pass
        outputs = model.forward(dummy_audio, compute_ot=True)
        
        print(f"\nForward pass outputs:")
        print(f"  z shape: {outputs['z'].shape}")
        print(f"  recon_error shape: {outputs['recon_error'].shape}")
        print(f"  logits shape: {outputs['logits'].shape}")
        print(f"  Q shape: {outputs['Q'].shape}")
        
        # Check Sinkhorn properties
        Q = outputs['Q']
        print(f"\nSinkhorn Q properties:")
        print(f"  Row sums (should be ~1): {Q.sum(dim=1)}")
        print(f"  Column sums (should be ~{batch_size/64:.4f}): {Q.sum(dim=0)[:5]}...")
        
        print("\nModel test passed!")
        
    except Exception as e:
        print(f"Model test failed: {e}")
        print("\nNote: If TitaNet model is not available, this is expected.")
        print("Please download the model first to fully test the pipeline.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()

