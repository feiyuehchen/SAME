"""
Unified Memory Network Model for Audio Deepfake Detection

================================================================================
Detailed System Architecture
================================================================================

【Overall Pipeline】

Input Audio → TitaNet Encoder → Embedding z → Dual Memory Bank Processing → Loss Computation / Score Output
            (192-dim)

================================================================================

【1. TitaNet Encoder】

TitaNet is a speaker recognition model developed by NVIDIA:
- Input: Raw waveform
- Output: 192-dimensional speaker embedding

Why TitaNet?
- Hypothesis: Fake audio cannot perfectly replicate speaker acoustic features
- Experimental goal: Verify if speaker embeddings can capture differences between real and fake audio

Model variants:
- titanet_small: 6M parameters, fast experimentation
- titanet_large: 25M parameters, better performance

================================================================================

【2. Dual Memory Bank】

Structure:
┌─────────────────────────────────────────────────────────────┐
│  Memory Bonafide (M_real)     │  Memory Spoof (M_spoof)     │
│  Shape: (K, D) = (128, 192)   │  Shape: (K, D) = (128, 192) │
│  K = memory_slots             │  D = embed_dim              │
│  Each row is a learnable prototype vector                   │
└─────────────────────────────────────────────────────────────┘

Working principle:
- Bonafide Bank: Learns "typical patterns of real audio"
- Spoof Bank: Learns "typical patterns of fake audio"
- At test time, compare input matching with both banks

================================================================================

【3. Sparse Attention】

Why Top-K?
- Full attention would involve all prototypes in reconstruction
- In practice, only a few prototypes are relevant
- Top-K selects only the most relevant K prototypes, reducing noise

Implementation steps:
1. Compute cosine similarity between z and all prototypes
2. Select Top-K (e.g., K=10) most similar ones
3. Apply softmax to these K to get weights
4. Weighted sum to get reconstruction vector

Code explanation:
```python
similarity = z @ M.T                    # (B, slots) all similarities
topk_values, topk_indices = topk(sim)   # Select Top-K
topk_softmax = softmax(topk_values)     # Normalize weights
selected_M = M[topk_indices]            # Extract corresponding prototypes (B, K, D)
z_recon = sum(selected_M * weights)     # Weighted reconstruction (B, D)
```

================================================================================

【4. Sinkhorn-Knopp OT Algorithm】

Problem: Mode Collapse
- Without regularization, model tends to use only a few memory slots
- Most slots become "dead slots", wasting capacity

Solution: Optimal Transport
- Force uniform distribution of batch samples across all slots
- Use Sinkhorn algorithm to find optimal assignment

Algorithm steps:
```
Input: Similarity matrix S (B × K)
       Epsilon ε (controls "softness" of assignment)
       
1. Q = exp(S / ε)           # Convert to positive matrix
2. For i = 1 to n_iterations:
   a. Q = Q / sum(Q, axis=1)  # Row normalization (each sample's assignment sums to 1)
   b. Q = Q / sum(Q, axis=0)  # Column normalization (each slot receives uniform amount)
3. Q = Q / sum(Q, axis=1)   # Final row normalization
Output: Q (approximate doubly-stochastic matrix)
```

Why "approximate"?
- Pure doubly-stochastic requires row sum=1 and column sum=1
- But B and K may differ, so it's approximate
- Key is to make column sums tend to be uniform

================================================================================

【5. Scoring Mechanism】

Basic Mode: Pure reconstruction error
```
score = error_spoof - error_real
```
- Bonafide: error_real low, error_spoof high → high score
- Spoof: error_real high, error_spoof low → low score

Enhanced Mode: Fused scoring
```
score = w * oc_score + (1-w) * recon_score
```
- oc_score: Maximum similarity to learned prototype centers
- recon_score: Reconstruction error difference (after normalization)
- Both are complementary, more robust

================================================================================

【6. Key Differences from SAMO】

| Aspect | SAMO | Ours |
|--------|------|------|
| Speaker Info | Requires speaker ID | Not needed at all |
| Centers | One per speaker | K learnable |
| Enrollment | Required at test time | Not needed |
| Memory Bank | None | Dual memory bank |
| Augmentation | None | RawBoost + Codec |

================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Tuple
import numpy as np
import math

from models.model_titanet import TitaNetEncoder
from models.loss import UnifiedLoss, AdaptiveMarginScheduler


class OTMemoryTitaNet(pl.LightningModule):
    """
    TitaNet + Memory Network with Optimal Transport
    
    Unified model supporting both basic and enhanced modes:
    - basic: Reconstruction + OT + OC-Softmax + Diversity
    - enhanced: + Multi-Center OC + Contrastive + Adaptive Margin
    """
    
    def __init__(
        self,
        mode: str = "enhanced",  # "basic" or "enhanced"
        titanet_model: str = "titanet_large",
        embed_dim: int = 192,
        memory_slots: int = 128,
        num_oc_centers: int = 20,  # Enhanced mode only
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_ot: float = 0.3,
        lambda_oc: float = 0.5,
        lambda_div: float = 0.1,
        lambda_contrastive: float = 0.3,  # Enhanced only
        # OC-Softmax params
        m_real: float = 0.9,
        m_fake: float = 0.3,
        alpha: float = 20.0,
        # Memory params
        top_k: int = 10,
        margin: float = 1.0,
        # Sinkhorn params
        sinkhorn_iterations: int = 10,
        sinkhorn_epsilon: float = 0.1,
        logit_scale_init: float = 1.0,
        # Enhanced mode: Scoring
        score_fusion: str = "combined",  # "recon", "oc", "combined"
        score_weight: float = 0.5,
        # Enhanced mode: Adaptive margin
        use_adaptive_margin: bool = True,
        max_steps: int = 30000,
        warmup_steps: int = 1000,  # Warm-up steps for learning rate scheduler
        # Other
        asv_score_path: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.mode = mode
        self.embed_dim = embed_dim
        self.use_memory = memory_slots > 0 and top_k > 0  # allow memory-less configs
        self.memory_slots = memory_slots if self.use_memory else 0
        # Keep top_k valid even if user sets larger than available slots
        self.top_k = min(top_k, self.memory_slots) if self.use_memory else 0
        self.lr = lr
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.score_fusion = score_fusion
        self.score_weight = score_weight
        self.use_adaptive_margin = use_adaptive_margin and (mode == "enhanced")
        
        # ===============================
        # TitaNet Encoder
        # ===============================
        self.encoder = TitaNetEncoder(
            model_name=titanet_model,
            freeze=freeze_encoder,
            embed_dim=embed_dim
        )
        
        # ===============================
        # Dual Memory Banks (optional)
        # ===============================
        if self.use_memory:
            # Each bank has memory_slots prototypes, each prototype is embed_dim-dimensional
            self.memory_bonafide = nn.Parameter(torch.randn(self.memory_slots, embed_dim))
            self.memory_spoof = nn.Parameter(torch.randn(self.memory_slots, embed_dim))
            
            # Normalized initialization
            with torch.no_grad():
                self.memory_bonafide.data = F.normalize(self.memory_bonafide.data, p=2, dim=1)
                self.memory_spoof.data = F.normalize(self.memory_spoof.data, p=2, dim=1)
        else:
            # Memory disabled (e.g., TitaNet-only experiment)
            self.memory_bonafide = None
            self.memory_spoof = None
        
        # ===============================
        # Learnable Temperature
        # ===============================
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale_init)
        
        # ===============================
        # Loss Function
        # ===============================
        self.loss_fn = UnifiedLoss(
            mode=mode,
            feat_dim=embed_dim,
            num_centers=num_oc_centers,
            m_real=m_real,
            m_fake=m_fake,
            alpha=alpha,
            lambda_recon=lambda_recon,
            lambda_ot=lambda_ot,
            lambda_oc=lambda_oc,
            lambda_contrastive=lambda_contrastive,
            lambda_diversity=lambda_div,
            margin=margin
        )
        
        # ===============================
        # Adaptive Margin Scheduler (Enhanced only)
        # ===============================
        if self.use_adaptive_margin:
            self.margin_scheduler = AdaptiveMarginScheduler(
                initial_m_real=0.7,
                final_m_real=m_real,
                initial_m_fake=0.3,
                final_m_fake=m_fake,
                warmup_steps=1000,
                total_steps=max_steps
            )
        
        # Validation outputs
        self.validation_step_outputs = []
    
    def sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp Algorithm
        
        Convert similarity matrix to approximate doubly-stochastic assignment matrix
        Ensure uniform distribution of batch samples across all memory slots
        """
        B, K = logits.shape
        
        # Numerical stability: clamp logits range
        logits_clamped = torch.clamp(logits / self.sinkhorn_epsilon, min=-50, max=50)
        
        # Initialize Q
        Q = torch.exp(logits_clamped)
        
        eps = 1e-8
        
        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iterations):
            # Row normalization
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
            # Column normalization
            Q = Q / (Q.sum(dim=0, keepdim=True) + eps)
        
        # Final row normalization
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        
        return Q
    
    def _process_memory_bank(self, z: torch.Tensor, memory_bank: torch.Tensor):
        """
        Process single memory bank with Sparse Attention
        
        Steps:
        1. Compute similarity with all prototypes
        2. Select Top-K most similar ones
        3. Weighted reconstruction
        4. Compute reconstruction error
        """
        # L2 normalize memory
        memory_normalized = F.normalize(memory_bank, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(z, memory_normalized.t())  # (B, memory_slots)
        
        # Sparse Attention: select only Top-K
        topk_values, topk_indices = torch.topk(similarity, k=self.top_k, dim=1)
        topk_softmax = F.softmax(topk_values, dim=1)  # Normalize weights
        
        # Build full attention weights (for diversity loss)
        B, K = similarity.shape
        attn_weights = torch.zeros_like(similarity)
        attn_weights.scatter_(1, topk_indices, topk_softmax)
        
        # Extract corresponding prototype vectors
        selected_memory = F.embedding(topk_indices, memory_normalized)  # (B, top_k, dim)
        
        # Weighted sum to get reconstruction vector
        z_recon = torch.sum(selected_memory * topk_softmax.unsqueeze(2), dim=1)  # (B, dim)
        
        # Compute reconstruction error (MSE)
        recon_error = torch.sum((z - z_recon) ** 2, dim=1)  # (B,)
        
        return z_recon, recon_error, similarity, attn_weights

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        compute_ot: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        1. TitaNet extracts embedding
        2. Interact with both memory banks separately
        3. (During training) Compute OT assignment
        """
        # Extract embedding
        z = self.encoder(audio, audio_lengths)  # (B, embed_dim)
        
        if self.use_memory:
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
            
            # Compute OT assignment (during training)
            if compute_ot:
                logits_real = self.logit_scale * sim_real
                Q_real = self.sinkhorn(logits_real)
                
                logits_spoof = self.logit_scale * sim_spoof
                Q_spoof = self.sinkhorn(logits_spoof)
                
                output['logits_real'] = logits_real
                output['Q_real'] = Q_real
                output['logits_spoof'] = logits_spoof
                output['Q_spoof'] = Q_spoof
        else:
            # Memory-less path: return embeddings only
            output = {
                'z': z,
                'error_real': None,
                'error_spoof': None,
                'attn_weights_real': None,
                'attn_weights_spoof': None
            }
        
        return output
    
    def compute_score(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute final score (for inference)
        
        Supports three modes:
        - "recon": Pure reconstruction error
        - "oc": Pure OC-Softmax similarity
        - "combined": Weighted fusion of both
        """
        recon_score = None
        if self.use_memory and outputs['error_real'] is not None and outputs['error_spoof'] is not None:
            recon_score = outputs['error_spoof'] - outputs['error_real']
        
        # OC-Softmax score (both modes)
        z = outputs['z']
        z_norm = F.normalize(z, p=2, dim=1)
        if self.mode == "enhanced":
            centers_norm = F.normalize(self.loss_fn.oc_loss.centers, p=2, dim=1)
            oc_score = torch.max(z_norm @ centers_norm.t(), dim=1)[0]
        else:
            center_norm = F.normalize(self.loss_fn.oc_loss.center, p=2, dim=1)
            oc_score = torch.matmul(z_norm, center_norm.t()).squeeze(1)
        
        # Score selection
        if not self.use_memory or self.score_fusion == "oc":
            return oc_score
        
        if self.score_fusion == "combined" and recon_score is not None:
            recon_score_norm = (recon_score - recon_score.mean()) / (recon_score.std() + 1e-8)
            oc_score_norm = (oc_score - oc_score.mean()) / (oc_score.std() + 1e-8)
            return self.score_weight * oc_score_norm + (1 - self.score_weight) * recon_score_norm
        
        if recon_score is not None:
            return recon_score
        
        # Fallback: use OC score when reconstruction is unavailable
        return oc_score
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        audio, labels = batch
        
        # Update adaptive margins
        if self.use_adaptive_margin:
            m_real, m_fake = self.margin_scheduler.get_margins(self.global_step)
            self.loss_fn.oc_loss.m_real = m_real
            self.loss_fn.oc_loss.m_fake = m_fake
        
        # Forward
        use_ot = self.use_memory and (self.hparams.lambda_ot > 0)
        outputs = self.forward(audio, compute_ot=use_ot)
        
        # Prepare OT inputs
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        logits_list = [None, None]
        Q_list = [None, None]
        
        # Only access OT-related keys if they exist (i.e., when compute_ot=True)
        if is_bonafide.sum() > 0 and use_ot and 'logits_real' in outputs:
            logits_list[0] = outputs['logits_real'][is_bonafide]
            Q_list[0] = outputs['Q_real'][is_bonafide]
        if is_spoof.sum() > 0 and use_ot and 'logits_spoof' in outputs:
            logits_list[1] = outputs['logits_spoof'][is_spoof]
            Q_list[1] = outputs['Q_spoof'][is_spoof]
        
        # Compute loss
        loss_dict = self.loss_fn(
            embeddings=outputs['z'],
            labels=labels,
            error_real=outputs['error_real'],
            error_spoof=outputs['error_spoof'],
            memory_bonafide=self.memory_bonafide if self.use_memory else None,
            logits=logits_list,
            Q=Q_list,
            attn_weights=[outputs['attn_weights_real'], outputs['attn_weights_spoof']] if self.use_memory else None,
            use_ot=use_ot
        )
        
        # Logging - All losses
        self.log('train/loss', loss_dict['loss'], prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', loss_dict['recon_loss'], sync_dist=True)
        self.log('train/ot_loss', loss_dict['ot_loss'], sync_dist=True)
        self.log('train/oc_loss', loss_dict['oc_loss'], sync_dist=True)
        self.log('train/div_loss', loss_dict['div_loss'], sync_dist=True)
        
        if self.mode == "enhanced":
            self.log('train/contrastive_loss', loss_dict['contrastive_loss'], sync_dist=True)
            self.log('train/proto_div_loss', loss_dict['proto_div_loss'], sync_dist=True)
            if self.use_adaptive_margin:
                self.log('train/m_real', self.loss_fn.oc_loss.m_real, sync_dist=True)
                self.log('train/m_fake', self.loss_fn.oc_loss.m_fake, sync_dist=True)
        
        # Log weighted contributions
        self.log('train/weighted_recon', self.hparams.lambda_recon * loss_dict['recon_loss'], sync_dist=True)
        self.log('train/weighted_ot', self.hparams.lambda_ot * loss_dict['ot_loss'], sync_dist=True)
        self.log('train/weighted_oc', self.hparams.lambda_oc * loss_dict['oc_loss'], sync_dist=True)
        self.log('train/weighted_div', self.hparams.lambda_div * loss_dict['div_loss'], sync_dist=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step"""
        if len(batch) == 3:
            audio, labels, utt_ids = batch
        else:
            audio, labels = batch
            utt_ids = None
        
        outputs = self.forward(audio, compute_ot=False)
        scores = self.compute_score(outputs)
        
        self.validation_step_outputs.append({
            'scores': scores.detach().cpu(),
            'labels': labels.detach().cpu(),
            'utt_ids': utt_ids if utt_ids else [],
            'recon_errors': outputs['error_real'].detach().cpu() if outputs['error_real'] is not None else None
        })
    
    def on_validation_epoch_end(self):
        """Compute EER"""
        if len(self.validation_step_outputs) == 0:
            return
        
        all_scores = torch.cat([x['scores'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        scores_np = all_scores.numpy()
        labels_np = all_labels.numpy()
        
        bonafide_mask = (labels_np == 0)
        spoof_mask = (labels_np == 1)
        
        if bonafide_mask.sum() == 0 or spoof_mask.sum() == 0:
            self.validation_step_outputs.clear()
            return
        
        bonafide_scores = scores_np[bonafide_mask]
        spoof_scores = scores_np[spoof_mask]
        
        try:
            from eval_metrics import compute_eer
            eer, threshold = compute_eer(bonafide_scores, spoof_scores)
            self.log('val/eer', eer, prog_bar=True, sync_dist=False)
            self.log('val/eer_threshold', threshold, sync_dist=False)
        except Exception as e:
            print(f"Warning: Could not compute EER: {e}")
        
        # Score statistics
        self.log('val/bonafide_score_mean', bonafide_scores.mean(), sync_dist=False)
        self.log('val/spoof_score_mean', spoof_scores.mean(), sync_dist=False)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer with warm-up + cosine annealing"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Warm-up + Cosine Annealing
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = self.hparams.warmup_steps
        max_steps = self.hparams.max_steps
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Warm-up: linear increase from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing: smooth decay from 1 to 0
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def test_model():
    """Test model"""
    print("Testing OTMemoryTitaNet...")
    
    batch_size = 4
    audio_length = 64000
    
    dummy_audio = torch.randn(batch_size, audio_length)
    dummy_labels = torch.tensor([0, 0, 1, 1])
    
    try:
        # Test enhanced mode
        model = OTMemoryTitaNet(
            mode="enhanced",
            titanet_model="titanet_small",
            embed_dim=192,
            memory_slots=64,
            num_oc_centers=20,
            freeze_encoder=True
        )
        
        print(f"Model mode: {model.mode}")
        print(f"Memory bonafide shape: {model.memory_bonafide.shape}")
        
        outputs = model.forward(dummy_audio, compute_ot=True)
        print(f"Embedding shape: {outputs['z'].shape}")
        
        scores = model.compute_score(outputs)
        print(f"Scores: {scores}")
        
        print("\n✓ Model test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()
