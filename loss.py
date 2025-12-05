"""
Loss functions for OT-Regularized Memory Network
Supervised Binary Classification with Memory Reconstruction for Audio Deepfake Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmax(nn.Module):
    """
    One-Class Softmax Loss
    Encourages bonafide samples to cluster tightly and spoof samples to be pushed away.
    Reference: Zhang et al. "One-class Softmax Loss for Anti-spoofing", 2020.
    """
    def __init__(self, feat_dim=192, r_real=0.9, r_fake=0.5, alpha=20.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        # Center for bonafide class
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature embeddings, shape (B, D)
            labels: Binary labels (0=bonafide, 1=spoof), shape (B,)
        """
        # Normalize inputs and center
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        # Compute cosine similarity
        scores = x @ w.t()  # (B, 1)
        scores = scores.squeeze(1)
        
        # Create masks
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss = torch.tensor(0.0, device=x.device)
        
        # Bonafide loss: push score > r_real
        if is_bonafide.sum() > 0:
            loss += self.softplus(self.alpha * (self.r_real - scores[is_bonafide])).mean()
            
        # Spoof loss: push score < r_fake
        if is_spoof.sum() > 0:
            loss += self.softplus(self.alpha * (scores[is_spoof] - self.r_fake)).mean()
            
        return loss


class DiversityLoss(nn.Module):
    """
    Entropy-based Diversity Loss
    Maximizes the entropy of the batch-averaged attention weights to prevent Mode Collapse.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_weights: Attention weights, shape (B, K)
        """
        # Average attention weights over the batch
        # Shape: (K,)
        mean_attn = attn_weights.mean(dim=0)
        
        # Add epsilon for numerical stability
        mean_attn = torch.clamp(mean_attn, min=1e-7)
        
        # Calculate Entropy: H = - sum(p * log(p))
        entropy = -torch.sum(mean_attn * torch.log(mean_attn))
        
        # We want to MAXIMIZE entropy, so MINIMIZE negative entropy
        return -entropy


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
        lambda_oc: float = 0.5,
        lambda_div: float = 0.1,
        margin: float = 1.0,
        feat_dim: int = 192
    ):
        """
        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_ot: Weight for OT regularization loss
            lambda_oc: Weight for OC-Softmax loss
            lambda_div: Weight for Diversity loss
            margin: Margin for hinge loss (push spoof samples away)
            feat_dim: Feature dimension for OC-Softmax
        """
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_ot = lambda_ot
        self.lambda_oc = lambda_oc
        self.lambda_div = lambda_div
        self.margin = margin
        
        # Initialize OC-Softmax
        self.oc_softmax = OCSoftmax(feat_dim=feat_dim)
        
        # Initialize Diversity Loss
        self.diversity_loss = DiversityLoss()
    
    def compute_reconstruction_loss(
        self,
        recon_error: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute single-bank reconstruction loss (Legacy/Single Bank)
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
        if is_spoof.sum() > 0:
            spoof_errors = recon_error[is_spoof]
            recon_loss_spoof = F.relu(self.margin - spoof_errors).mean()
        else:
            recon_loss_spoof = torch.tensor(0.0, device=recon_error.device)
        
        # Total reconstruction loss
        total_recon_loss = recon_loss_bonafide + recon_loss_spoof
        
        return total_recon_loss, recon_loss_bonafide, recon_loss_spoof

    def compute_dual_reconstruction_loss(
        self,
        error_real: torch.Tensor,
        error_spoof: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dual-bank reconstruction loss
        
        Bonafide (0): Minimize error_real, Maximize error_spoof
        Spoof (1): Minimize error_spoof, Maximize error_real
        """
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss_bonafide = torch.tensor(0.0, device=error_real.device)
        loss_spoof = torch.tensor(0.0, device=error_real.device)
        
        # Bonafide samples
        if is_bonafide.sum() > 0:
            # Attract to Real Bank
            loss_bonafide += error_real[is_bonafide].mean()
            # Repel from Spoof Bank
            loss_bonafide += F.relu(self.margin - error_spoof[is_bonafide]).mean()
            
        # Spoof samples
        if is_spoof.sum() > 0:
            # Attract to Spoof Bank
            loss_spoof += error_spoof[is_spoof].mean()
            # Repel from Real Bank
            loss_spoof += F.relu(self.margin - error_real[is_spoof]).mean()
            
        total_loss = loss_bonafide + loss_spoof
        return total_loss, loss_bonafide, loss_spoof

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
        recon_error: torch.Tensor = None,
        error_real: torch.Tensor = None,
        error_spoof: torch.Tensor = None,
        logits: torch.Tensor = None, # Can be list [logits_real, logits_spoof]
        Q: torch.Tensor = None, # Can be list [Q_real, Q_spoof]
        labels: torch.Tensor = None,
        embeddings: torch.Tensor = None,
        attn_weights: torch.Tensor = None, # Can be list
        use_ot: bool = True
    ) -> dict:
        """
        Compute total loss
        """
        # Compute reconstruction loss
        if error_real is not None and error_spoof is not None:
            recon_loss, recon_loss_bonafide, recon_loss_spoof = \
                self.compute_dual_reconstruction_loss(error_real, error_spoof, labels)
        else:
            recon_loss, recon_loss_bonafide, recon_loss_spoof = \
                self.compute_reconstruction_loss(recon_error, labels)
        
        # Compute OT loss
        ot_loss = torch.tensor(0.0, device=labels.device)
        if use_ot:
            # Check if we have lists (Dual Bank)
            if isinstance(logits, list) and isinstance(Q, list):
                # logits[0] = real, logits[1] = spoof
                # Q[0] = real, Q[1] = spoof
                # Apply OT loss on Bonafide samples -> Real Bank
                # Apply OT loss on Spoof samples -> Spoof Bank
                
                is_bonafide = (labels == 0)
                is_spoof = (labels == 1)
                
                if is_bonafide.sum() > 0 and logits[0] is not None:
                     ot_loss += self.compute_ot_loss(logits[0], Q[0])
                
                if is_spoof.sum() > 0 and logits[1] is not None:
                     ot_loss += self.compute_ot_loss(logits[1], Q[1])
            elif logits is not None and Q is not None:
                # Single bank
                ot_loss = self.compute_ot_loss(logits, Q)
            
        # Compute OC-Softmax Loss
        if embeddings is not None:
            oc_loss = self.oc_softmax(embeddings, labels)
        else:
            oc_loss = torch.tensor(0.0, device=labels.device)
            
        # Compute Diversity Loss
        div_loss = torch.tensor(0.0, device=labels.device)
        if attn_weights is not None:
            if isinstance(attn_weights, list):
                for aw in attn_weights:
                    if aw is not None:
                        div_loss += self.diversity_loss(aw)
                div_loss /= len(attn_weights) # Average
            else:
                div_loss = self.diversity_loss(attn_weights)
        
        # Total loss
        total_loss = (self.lambda_recon * recon_loss + 
                      self.lambda_ot * ot_loss + 
                      self.lambda_oc * oc_loss + 
                      self.lambda_div * div_loss)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'recon_loss_bonafide': recon_loss_bonafide,
            'recon_loss_spoof': recon_loss_spoof,
            'ot_loss': ot_loss,
            'oc_loss': oc_loss,
            'div_loss': div_loss
        }


def test_loss():
    """Test loss functions"""
    print("Testing OTMemoryLoss...")
    
    # Create dummy data
    batch_size = 8
    memory_slots = 64
    feat_dim = 192
    
    # Dual Bank errors
    error_real = torch.rand(batch_size)
    error_spoof = torch.rand(batch_size)
    
    # Labels (4 bonafide, 4 spoof)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Embeddings for OC-Softmax
    embeddings = torch.randn(batch_size, feat_dim)
    
    # Attention weights for Diversity Loss (list of two banks)
    attn_weights = [torch.rand(batch_size, memory_slots), torch.rand(batch_size, memory_slots)]
    
    # Logits and Q for Dual Bank
    # Only for samples matching the bank
    logits_real = torch.randn(4, memory_slots) # 4 bonafide
    Q_real = F.softmax(torch.randn(4, memory_slots), dim=1)
    
    logits_spoof = torch.randn(4, memory_slots) # 4 spoof
    Q_spoof = F.softmax(torch.randn(4, memory_slots), dim=1)
    
    logits_list = [logits_real, logits_spoof]
    Q_list = [Q_real, Q_spoof]
    
    # Create loss module
    loss_fn = OTMemoryLoss(
        lambda_recon=1.0, 
        lambda_ot=0.1, 
        lambda_oc=0.5, 
        lambda_div=0.1,
        margin=1.0,
        feat_dim=feat_dim
    )
    
    # Compute loss using keyword arguments
    loss_dict = loss_fn(
        error_real=error_real,
        error_spoof=error_spoof,
        logits=logits_list,
        Q=Q_list,
        labels=labels,
        embeddings=embeddings,
        attn_weights=attn_weights,
        use_ot=True
    )
    
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  - Bonafide: {loss_dict['recon_loss_bonafide'].item():.4f}")
    print(f"  - Spoof: {loss_dict['recon_loss_spoof'].item():.4f}")
    print(f"OT loss: {loss_dict['ot_loss'].item():.4f}")
    print(f"OC loss: {loss_dict['oc_loss'].item():.4f}")
    print(f"Div loss: {loss_dict['div_loss'].item():.4f}")
    
    print("\nLoss test passed!")


if __name__ == "__main__":
    test_loss()

