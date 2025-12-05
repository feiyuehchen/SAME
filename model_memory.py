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
        lambda_oc: float = 0.5,
        lambda_div: float = 0.1,
        top_k: int = 10,
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
            memory_slots: Number of memory prototypes per bank
            freeze_encoder: Whether to freeze TitaNet encoder
            lr: Learning rate
            lambda_recon: Weight for reconstruction loss
            lambda_ot: Weight for OT loss
            lambda_oc: Weight for OC-Softmax loss
            lambda_div: Weight for Diversity loss
            top_k: Top-K slots for sparse attention
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
        self.top_k = top_k
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
        
        # Dual Memory Banks: Real and Spoof
        self.memory_bonafide = nn.Parameter(torch.randn(memory_slots, embed_dim))
        self.memory_spoof = nn.Parameter(torch.randn(memory_slots, embed_dim))
        
        # Initialize with normalized vectors
        with torch.no_grad():
            self.memory_bonafide.data = F.normalize(self.memory_bonafide.data, p=2, dim=1)
            self.memory_spoof.data = F.normalize(self.memory_spoof.data, p=2, dim=1)
        
        # Learnable temperature scaling for logits (used in OT assignment)
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale_init)
        
        # Loss function
        self.loss_fn = OTMemoryLoss(
            lambda_recon=lambda_recon,
            lambda_ot=lambda_ot,
            lambda_oc=lambda_oc,
            lambda_div=lambda_div,
            margin=margin,
            feat_dim=embed_dim
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
    
    def _process_memory_bank(self, z, memory_bank):
        """Helper to process one memory bank with sparse attention"""
        # Normalize memory bank
        memory_normalized = F.normalize(memory_bank, p=2, dim=1)
        
        # Compute similarity: z @ memory.T
        similarity = torch.matmul(z, memory_normalized.t())  # (B, memory_slots)
        
        # Sparse Attention (Top-K)
        topk_values, topk_indices = torch.topk(similarity, k=self.top_k, dim=1)
        topk_softmax = F.softmax(topk_values, dim=1) # (B, K)
        
        # Construct full attention weights (for diversity loss)
        B, K = similarity.shape
        attn_weights = torch.zeros_like(similarity)
        attn_weights.scatter_(1, topk_indices, topk_softmax)
        
        # Reconstruct
        # Retrieve top-k memory vectors
        # topk_indices shape: (B, top_k)
        # memory_normalized shape: (slots, dim)
        # We want (B, top_k, dim)
        
        # Expand memory for gathering
        # Or simply use embedding lookup if we view memory as embedding layer
        # memory_normalized[indices] works if indices is (B, top_k)
        # but we need to flatten or use advanced indexing
        
        # Using F.embedding is cleaner
        selected_memory = F.embedding(topk_indices, memory_normalized) # (B, top_k, dim)
        
        # Weighted sum
        # topk_softmax: (B, top_k) -> (B, top_k, 1)
        z_recon = torch.sum(selected_memory * topk_softmax.unsqueeze(2), dim=1) # (B, dim)
        
        # Compute reconstruction error
        recon_error = torch.sum((z - z_recon) ** 2, dim=1) # (B,)
        
        return z_recon, recon_error, similarity, attn_weights

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        compute_ot: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Dual Memory Banks
        """
        # Extract embeddings from TitaNet
        z = self.encoder(audio, audio_lengths)  # (B, embed_dim)
        
        # Process Bonafide Bank
        z_recon_real, error_real, sim_real, attn_real = \
            self._process_memory_bank(z, self.memory_bonafide)
            
        # Process Spoof Bank
        z_recon_spoof, error_spoof, sim_spoof, attn_spoof = \
            self._process_memory_bank(z, self.memory_spoof)
            
        output = {
            'z': z,
            'error_real': error_real,
            'error_spoof': error_spoof,
            'attn_weights_real': attn_real,
            'attn_weights_spoof': attn_spoof
        }
        
        # Compute OT assignment if requested (training only)
        if compute_ot:
            # Real logits
            logits_real = self.logit_scale * sim_real
            Q_real = self.sinkhorn(logits_real)
            
            # Spoof logits
            logits_spoof = self.logit_scale * sim_spoof
            Q_spoof = self.sinkhorn(logits_spoof)
            
            output['logits_real'] = logits_real
            output['Q_real'] = Q_real
            output['logits_spoof'] = logits_spoof
            output['Q_spoof'] = Q_spoof
        
        return output
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        audio, labels = batch
        
        # Forward pass
        outputs = self.forward(audio, compute_ot=True)
        
        # Prepare OT inputs (only for samples matching the bank)
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        logits_list = [None, None]
        Q_list = [None, None]
        
        if is_bonafide.sum() > 0:
            logits_list[0] = outputs['logits_real'][is_bonafide]
            Q_list[0] = outputs['Q_real'][is_bonafide]
            
        if is_spoof.sum() > 0:
            logits_list[1] = outputs['logits_spoof'][is_spoof]
            Q_list[1] = outputs['Q_spoof'][is_spoof]
        
        # Compute loss
        loss_dict = self.loss_fn(
            error_real=outputs['error_real'],
            error_spoof=outputs['error_spoof'],
            logits=logits_list,
            Q=Q_list,
            labels=labels,
            embeddings=outputs['z'],
            attn_weights=[outputs['attn_weights_real'], outputs['attn_weights_spoof']],
            use_ot=True
        )
        
        # Log metrics
        self.log('train/loss', loss_dict['loss'], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/ot_loss', loss_dict['ot_loss'], on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/oc_loss', loss_dict['oc_loss'], on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/div_loss', loss_dict['div_loss'], on_step=True, on_epoch=False, sync_dist=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step"""
        if len(batch) == 3:
            audio, labels, utt_ids = batch
        else:
            audio, labels = batch
            utt_ids = None
        
        outputs = self.forward(audio, compute_ot=False)
        
        # Score: Error(Spoof) - Error(Real)
        # Bonafide: Low Real Error, High Spoof Error -> High Score
        # Spoof: High Real Error, Low Spoof Error -> Low Score
        scores = outputs['error_spoof'] - outputs['error_real']
        
        self.validation_step_outputs.append({
            'scores': scores.detach().cpu(),
            'labels': labels.detach().cpu(),
            'utt_ids': utt_ids if utt_ids is not None else [],
            'recon_errors': outputs['error_real'].detach().cpu() # Log real error
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
        print(f"  error_real shape: {outputs['error_real'].shape}")
        print(f"  error_spoof shape: {outputs['error_spoof'].shape}")
        print(f"  logits_real shape: {outputs['logits_real'].shape}")
        print(f"  Q_real shape: {outputs['Q_real'].shape}")
        
        # Check Sinkhorn properties
        Q = outputs['Q_real']
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

