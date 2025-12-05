"""
Loss functions for OT-Regularized Memory Network
Supervised Binary Classification with Memory Reconstruction for Audio Deepfake Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OTMemoryLoss(nn.Module):
    """
    Combined loss for Supervised Binary Classification with Optimal Transport regularization
    
    This is NOT pure one-class learning - training uses both bonafide and spoof labels.
    
    Components:
    1. Reconstruction Loss: Minimize error for bonafide, maximize for spoof (with margin)
    2. OT Loss: Ensure bonafide samples are evenly distributed across memory slots
    
    During inference, only reconstruction error is used (no labels needed).
    """
    
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_ot: float = 0.1,
        margin: float = 1.0
    ):
        """
        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_ot: Weight for OT regularization loss
            margin: Margin for hinge loss (push spoof samples away)
        """
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_ot = lambda_ot
        self.margin = margin
    
    def compute_reconstruction_loss(
        self,
        recon_error: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss
        
        For bonafide (label=0): Minimize reconstruction error
        For spoof (label=1): Apply hinge loss to push error above margin
        
        Args:
            recon_error: Reconstruction error per sample, shape (B,)
            labels: Binary labels (0=bonafide, 1=spoof), shape (B,)
        
        Returns:
            total_recon_loss: Combined reconstruction loss
            recon_loss_bonafide: Loss for bonafide samples (for logging)
            recon_loss_spoof: Loss for spoof samples (for logging)
        """
        # Create masks
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        # Bonafide loss: minimize reconstruction error
        if is_bonafide.sum() > 0:
            recon_loss_bonafide = recon_error[is_bonafide].mean()
        else:
            recon_loss_bonafide = torch.tensor(0.0, device=recon_error.device)
        
        # Spoof loss: hinge loss to push error above margin
        # We want spoof samples to have high reconstruction error
        # Hinge loss: max(0, margin - error)
        if is_spoof.sum() > 0:
            spoof_errors = recon_error[is_spoof]
            recon_loss_spoof = F.relu(self.margin - spoof_errors).mean()
        else:
            recon_loss_spoof = torch.tensor(0.0, device=recon_error.device)
        
        # Total reconstruction loss
        total_recon_loss = recon_loss_bonafide + recon_loss_spoof
        
        return total_recon_loss, recon_loss_bonafide, recon_loss_spoof
    
    def compute_ot_loss(
        self,
        logits: torch.Tensor,
        Q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OT regularization loss (entropy regularized OT)
        
        This encourages bonafide samples to be evenly distributed across memory slots
        using the Sinkhorn algorithm's soft assignment as the target
        
        Args:
            logits: Logits for memory slot assignment, shape (B, K)
            Q: Target distribution from Sinkhorn algorithm, shape (B, K)
                Must be detached to prevent gradient flow through Sinkhorn iterations
        
        Returns:
            ot_loss: OT regularization loss
        """
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Cross-entropy between Q and log_probs
        # Q is the target from Sinkhorn, log_probs is the prediction
        # CRITICAL: Detach Q to prevent backprop through Sinkhorn algorithm
        # We want to optimize log_probs to match Q, not modify Q to match log_probs
        ot_loss = -torch.sum(Q.detach() * log_probs, dim=1).mean()
        
        return ot_loss
    
    def forward(
        self,
        recon_error: torch.Tensor,
        logits: torch.Tensor,
        Q: torch.Tensor,
        labels: torch.Tensor,
        use_ot: bool = True
    ) -> dict:
        """
        Compute total loss
        
        Args:
            recon_error: Reconstruction error per sample, shape (B,)
            logits: Logits for OT (bonafide only), shape (B_bonafide, K)
            Q: Sinkhorn target distribution (bonafide only), shape (B_bonafide, K)
            labels: Binary labels, shape (B,)
            use_ot: Whether to use OT loss (set False during validation)
        
        Returns:
            Dictionary containing:
                - loss: Total loss
                - recon_loss: Total reconstruction loss
                - recon_loss_bonafide: Bonafide reconstruction loss (for logging)
                - recon_loss_spoof: Spoof reconstruction loss (for logging)
                - ot_loss: OT regularization loss (if use_ot=True)
        """
        # Compute reconstruction loss
        recon_loss, recon_loss_bonafide, recon_loss_spoof = \
            self.compute_reconstruction_loss(recon_error, labels)
        
        # Compute OT loss (only for bonafide samples during training)
        if use_ot and logits is not None and Q is not None:
            ot_loss = self.compute_ot_loss(logits, Q)
        else:
            ot_loss = torch.tensor(0.0, device=recon_error.device)
        
        # Total loss
        total_loss = self.lambda_recon * recon_loss + self.lambda_ot * ot_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'recon_loss_bonafide': recon_loss_bonafide,
            'recon_loss_spoof': recon_loss_spoof,
            'ot_loss': ot_loss
        }


def test_loss():
    """Test loss functions"""
    print("Testing OTMemoryLoss...")
    
    # Create dummy data
    batch_size = 8
    memory_slots = 64
    
    # Reconstruction errors
    recon_error = torch.rand(batch_size)
    
    # Labels (4 bonafide, 4 spoof)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Logits and Q for bonafide samples only
    bonafide_count = (labels == 0).sum().item()
    logits = torch.randn(bonafide_count, memory_slots)
    Q = F.softmax(torch.randn(bonafide_count, memory_slots), dim=1)
    
    # Create loss module
    loss_fn = OTMemoryLoss(lambda_recon=1.0, lambda_ot=0.1, margin=1.0)
    
    # Compute loss
    loss_dict = loss_fn(recon_error, logits, Q, labels, use_ot=True)
    
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  - Bonafide: {loss_dict['recon_loss_bonafide'].item():.4f}")
    print(f"  - Spoof: {loss_dict['recon_loss_spoof'].item():.4f}")
    print(f"OT loss: {loss_dict['ot_loss'].item():.4f}")
    
    print("\nLoss test passed!")


if __name__ == "__main__":
    test_loss()

