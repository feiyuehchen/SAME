"""
Unified Loss Functions for TitaNet + Memory Network Anti-Spoofing

Supports two modes:
1. Basic: Reconstruction + OT + OC-Softmax + Diversity
2. Enhanced: + Multi-Center OC-Softmax + Contrastive + Adaptive Margin

================================================================================
Detailed Algorithm Explanation
================================================================================

【1. Dual-Bank Reconstruction Loss】

Principle:
- Build two memory banks: Bonafide Bank (M_real) and Spoof Bank (M_spoof)
- Each memory bank contains K learnable prototype vectors
- Input embedding z attempts to reconstruct itself using prototypes from the memory bank

Mathematical formula:
- Reconstruction vector: z_recon = Σ(w_k * m_k), where w is attention weights
- Reconstruction error: E = ||z - z_recon||²

Training objective:
- Bonafide samples: Minimize E_real (similar to bonafide bank), maximize E_spoof
- Spoof samples: Minimize E_spoof (similar to spoof bank), maximize E_real

At inference:
- Score = E_spoof - E_real
- High score = Bonafide (because E_real is low, E_spoof is high)
- Low score = Spoof (because E_spoof is low, E_real is high)

================================================================================

【2. Sinkhorn Optimal Transport (OT) Regularization】

Problem:
- Traditional attention easily causes "Mode Collapse"
- Most queries only map to a few memory slots
- Causes other slots to be idle (Dead Slots)

Solution: Sinkhorn-Knopp Algorithm
- Force uniform distribution of batch samples across all memory slots
- Generate "doubly stochastic" matrix Q

Algorithm steps:
1. Compute logits: L = scale * CosineSim(z, M)
2. Initialize: Q = exp(L / ε)
3. Iterative normalization (repeat n times):
   - Row normalization: Q = Q / Σ_row(Q)
   - Column normalization: Q = Q / Σ_col(Q)
4. Final: Q becomes approximately uniform distribution

OT Loss:
- L_OT = -Σ Q_detached * log(Softmax(L))
- Q must be detached because we want model predictions to approximate optimal assignment

================================================================================

【3. One-Class Softmax (OC-Softmax)】

Principle:
- Learn a center point C representing "typical pattern of bonafide"
- Real audio should be close to this center
- Fake audio should be far from this center

Mathematical formula:
- Similarity: s = cos(x, C) = (x · C) / (||x|| ||C||)
- Bonafide loss: Softplus(α * (m_real - s))  [hope s > m_real]
- Spoof loss: Softplus(α * (s - m_fake))     [hope s < m_fake]

Parameter meaning:
- m_real (e.g., 0.9): Minimum similarity of Bonafide to center
- m_fake (e.g., 0.3): Maximum similarity of Spoof to center
- alpha (e.g., 20): Controls sharpness of decision boundary

================================================================================

【4. Multi-Center OC-Softmax (Enhanced Mode)】

SAMO approach vs our approach:
- SAMO: One center per speaker (requires speaker ID)
- Ours: K learnable centers (requires no additional information)

Principle:
- Learn K prototype centers to capture diversity of bonafide
- Does not rely on speaker information, more generalizable

Mathematical formula:
- Compute similarity to all centers: S = z @ Centers.T  # (B, K)
- Take maximum similarity: max_sim = max(S, dim=1)
- Loss computation same as OC-Softmax, but use max_sim instead of single center similarity

================================================================================

【5. Contrastive Memory Loss (Enhanced Mode)】

Principle:
- Explicitly establish structure in feature space
- Pull: Make bonafide samples close to memory bank prototypes
- Push: Make spoof samples far from memory bank prototypes

Mathematical formula:
- Pull (InfoNCE-style): L_pull = -log(Σ exp(sim/τ))
- Push (Margin-based): L_push = ReLU(max_sim + margin)

================================================================================

【6. Diversity Loss】

Problem:
- Even with OT, prototypes may become too similar to each other
- Causes redundancy, unable to capture diverse acoustic patterns

Solution:
- Compute average attention distribution within batch
- Maximize entropy of this distribution (more uniform = higher entropy)

Mathematical formula:
- Average attention: w_mean = mean(attention_weights, dim=0)
- Entropy: H = -Σ w_mean * log(w_mean)
- Loss: L_div = -H (minimize negative entropy = maximize entropy)

================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OCSoftmax(nn.Module):
    """
    One-Class Softmax Loss (Single Center Version)
    
    Tightly cluster bonafide around center point, push spoof away from center
    """
    def __init__(self, feat_dim=192, r_real=0.9, r_fake=0.5, alpha=20.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2 normalize
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        # Cosine similarity
        scores = x @ w.t()
        scores = scores.squeeze(1)
        
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss = torch.tensor(0.0, device=x.device)
        
        if is_bonafide.sum() > 0:
            # Bonafide: push score > r_real
            loss += self.softplus(self.alpha * (self.r_real - scores[is_bonafide])).mean()
            
        if is_spoof.sum() > 0:
            # Spoof: push score < r_fake
            loss += self.softplus(self.alpha * (scores[is_spoof] - self.r_fake)).mean()
            
        return loss


class MultiCenterOCSoftmax(nn.Module):
    """
    Multi-Center OC-Softmax Loss (Enhanced Mode)
    
    Does not require speaker ID, learns K generalized bonafide prototypes
    
    Key differences from SAMO:
    - SAMO: 1 center per speaker (requires speaker enrollment)
    - Ours: K learnable centers (completely speaker-agnostic)
    """
    def __init__(
        self, 
        feat_dim: int = 192, 
        num_centers: int = 20,
        m_real: float = 0.9,
        m_fake: float = 0.5,
        alpha: float = 20.0
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        
        # Learnable prototype centers
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        # Orthogonal initialization for better coverage
        nn.init.orthogonal_(self.centers)
        
        self.softplus = nn.Softplus()
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Returns:
            loss: Scalar loss
            max_sim: Maximum similarity scores (for inference)
        """
        # Normalize
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.centers, p=2, dim=1)
        
        # Compute similarity to all centers: (B, K)
        similarities = x @ w.t()
        
        # Max similarity across centers
        max_sim, _ = torch.max(similarities, dim=1)
        
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss = torch.tensor(0.0, device=x.device)
        
        if is_bonafide.sum() > 0:
            bonafide_loss = self.softplus(self.alpha * (self.m_real - max_sim[is_bonafide]))
            loss = loss + bonafide_loss.mean()
        
        if is_spoof.sum() > 0:
            spoof_loss = self.softplus(self.alpha * (max_sim[is_spoof] - self.m_fake))
            loss = loss + spoof_loss.mean()
        
        return loss, max_sim


class ContrastiveMemoryLoss(nn.Module):
    """
    Contrastive Learning on Memory Bank (Enhanced Mode)
    
    Explicitly establish feature space structure:
    - Bonafide close to prototypes (Pull)
    - Spoof far from prototypes (Push)
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, z: torch.Tensor, memory: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, p=2, dim=1)
        memory = F.normalize(memory, p=2, dim=1)
        
        sim = z @ memory.t() / self.temperature
        
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss = torch.tensor(0.0, device=z.device)
        
        # Bonafide: Pull towards prototypes (InfoNCE-style)
        if is_bonafide.sum() > 0:
            bonafide_sim = sim[is_bonafide]
            pull_loss = -torch.logsumexp(bonafide_sim, dim=1).mean()
            loss = loss + pull_loss
        
        # Spoof: Push away from prototypes
        if is_spoof.sum() > 0:
            spoof_sim = sim[is_spoof]
            max_sim = torch.max(spoof_sim, dim=1)[0]
            push_loss = F.relu(max_sim + self.margin).mean()
            loss = loss + push_loss
        
        return loss


class PrototypeDiversityLoss(nn.Module):
    """
    Prototype Diversity Loss
    
    Ensure sufficient diversity between prototypes, prevent collapse into similar vectors
    """
    def __init__(self, min_distance: float = 0.3):
        super().__init__()
        self.min_distance = min_distance
        
    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        p = F.normalize(prototypes, p=2, dim=1)
        sim_matrix = p @ p.t()
        
        K = sim_matrix.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]
        
        max_sim = 1.0 - self.min_distance
        diversity_loss = F.relu(off_diag - max_sim).mean()
        
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Entropy-based Diversity Loss
    
    Maximize entropy of attention distribution, ensure all memory slots are used
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        mean_attn = attn_weights.mean(dim=0)
        mean_attn = torch.clamp(mean_attn, min=1e-7)
        entropy = -torch.sum(mean_attn * torch.log(mean_attn))
        return -entropy  # Minimize negative entropy = maximize entropy


class UnifiedLoss(nn.Module):
    """
    Unified Loss Function
    
    Supports both basic and enhanced modes
    """
    def __init__(
        self,
        mode: str = "enhanced",
        feat_dim: int = 192,
        num_centers: int = 20,
        m_real: float = 0.9,
        m_fake: float = 0.5,
        alpha: float = 20.0,
        lambda_recon: float = 1.0,
        lambda_ot: float = 0.3,
        lambda_oc: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_diversity: float = 0.1,
        margin: float = 1.0
    ):
        super().__init__()
        
        self.mode = mode
        self.lambda_recon = lambda_recon
        self.lambda_ot = lambda_ot
        self.lambda_oc = lambda_oc
        self.lambda_contrastive = lambda_contrastive
        self.lambda_diversity = lambda_diversity
        self.margin = margin
        
        # OC-Softmax (basic mode uses single center)
        if mode == "enhanced":
            self.oc_loss = MultiCenterOCSoftmax(
                feat_dim=feat_dim,
                num_centers=num_centers,
                m_real=m_real,
                m_fake=m_fake,
                alpha=alpha
            )
            self.contrastive_loss = ContrastiveMemoryLoss()
            self.prototype_diversity = PrototypeDiversityLoss()
        else:
            self.oc_loss = OCSoftmax(
                feat_dim=feat_dim,
                r_real=m_real,
                r_fake=m_fake,
                alpha=alpha
            )
        
        self.diversity_loss = DiversityLoss()
    
    def compute_dual_reconstruction_loss(
        self,
        error_real: torch.Tensor,
        error_spoof: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Dual-Bank Reconstruction Loss
        
        Bonafide: Minimize error_real, maximize error_spoof
        Spoof: Minimize error_spoof, maximize error_real
        """
        is_bonafide = (labels == 0)
        is_spoof = (labels == 1)
        
        loss_bonafide = torch.tensor(0.0, device=error_real.device)
        loss_spoof = torch.tensor(0.0, device=error_real.device)
        
        if is_bonafide.sum() > 0:
            # Attract to Real Bank
            loss_bonafide += error_real[is_bonafide].mean()
            # Repel from Spoof Bank (with margin hinge)
            loss_bonafide += F.relu(self.margin - error_spoof[is_bonafide]).mean()
            
        if is_spoof.sum() > 0:
            # Attract to Spoof Bank
            loss_spoof += error_spoof[is_spoof].mean()
            # Repel from Real Bank
            loss_spoof += F.relu(self.margin - error_real[is_spoof]).mean()
            
        return loss_bonafide + loss_spoof, loss_bonafide, loss_spoof

    def compute_ot_loss(self, logits: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Optimal Transport Loss
        
        Make model prediction distribution approximate optimal assignment computed by Sinkhorn
        """
        if logits is None or Q is None:
            return torch.tensor(0.0)
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(Q.detach() * log_probs, dim=1).mean()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        error_real: torch.Tensor = None,
        error_spoof: torch.Tensor = None,
        memory_bonafide: torch.Tensor = None,
        logits: list = None,
        Q: list = None,
        attn_weights: list = None,
        use_ot: bool = True
    ) -> dict:
        """
        Forward pass computing all losses
        """
        device = embeddings.device
        
        # 1. Reconstruction Loss
        if error_real is not None and error_spoof is not None:
            recon_loss, recon_bonafide, recon_spoof = self.compute_dual_reconstruction_loss(
                error_real, error_spoof, labels
            )
        else:
            recon_loss = torch.tensor(0.0, device=device)
        
        # 2. OT Loss
        ot_loss = torch.tensor(0.0, device=device)
        if use_ot and logits is not None and Q is not None:
            if isinstance(logits, list):
                for l, q in zip(logits, Q):
                    if l is not None and q is not None:
                        ot_loss = ot_loss + self.compute_ot_loss(l, q)
            else:
                ot_loss = self.compute_ot_loss(logits, Q)
        
        # 3. OC-Softmax Loss
        if self.mode == "enhanced":
            oc_loss, oc_scores = self.oc_loss(embeddings, labels)
        else:
            oc_loss = self.oc_loss(embeddings, labels)
            oc_scores = None
        
        # 4. Diversity Loss
        div_loss = torch.tensor(0.0, device=device)
        if attn_weights is not None:
            if isinstance(attn_weights, list):
                for aw in attn_weights:
                    if aw is not None:
                        div_loss += self.diversity_loss(aw)
                div_loss /= len(attn_weights)
            else:
                div_loss = self.diversity_loss(attn_weights)
        
        # 5. Enhanced mode: Contrastive + Prototype Diversity
        contrastive_loss = torch.tensor(0.0, device=device)
        proto_div_loss = torch.tensor(0.0, device=device)
        
        if self.mode == "enhanced":
            if memory_bonafide is not None:
                contrastive_loss = self.contrastive_loss(embeddings, memory_bonafide, labels)
            proto_div_loss = self.prototype_diversity(self.oc_loss.centers)
        
        # Total Loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_ot * ot_loss +
            self.lambda_oc * oc_loss +
            self.lambda_diversity * div_loss
        )
        
        if self.mode == "enhanced":
            total_loss = total_loss + self.lambda_contrastive * contrastive_loss
            total_loss = total_loss + self.lambda_diversity * proto_div_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'ot_loss': ot_loss,
            'oc_loss': oc_loss,
            'div_loss': div_loss,
            'contrastive_loss': contrastive_loss,
            'proto_div_loss': proto_div_loss,
            'oc_scores': oc_scores,
        }


# Backward compatibility
OTMemoryLoss = UnifiedLoss


class AdaptiveMarginScheduler:
    """
    Adaptive Margin Scheduler (Curriculum Learning)
    
    Principle:
    - Use relaxed margin in early training, let model learn basic discrimination first
    - Gradually tighten margin as training progresses, force model to learn finer features
    
    Benefits:
    - Avoid convergence failure in early training due to being too strict
    - Ultimately achieve better classification performance
    """
    def __init__(
        self,
        initial_m_real: float = 0.7,
        final_m_real: float = 0.95,
        initial_m_fake: float = 0.3,
        final_m_fake: float = 0.1,
        warmup_steps: int = 1000,
        total_steps: int = 20000
    ):
        self.initial_m_real = initial_m_real
        self.final_m_real = final_m_real
        self.initial_m_fake = initial_m_fake
        self.final_m_fake = final_m_fake
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def get_margins(self, step: int) -> tuple:
        if step < self.warmup_steps:
            progress = 0.0
        else:
            progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
        m_real = self.initial_m_real + progress * (self.final_m_real - self.initial_m_real)
        m_fake = self.initial_m_fake + progress * (self.final_m_fake - self.initial_m_fake)
        
        return m_real, m_fake


def test_loss():
    """Test loss functions"""
    print("Testing UnifiedLoss...")
    
    batch_size = 8
    feat_dim = 192
    memory_slots = 64
    
    # Create dummy data
    embeddings = torch.randn(batch_size, feat_dim)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    error_real = torch.rand(batch_size)
    error_spoof = torch.rand(batch_size)
    memory = torch.randn(memory_slots, feat_dim)
    attn = [torch.rand(batch_size, memory_slots), torch.rand(batch_size, memory_slots)]
    logits = [torch.randn(4, memory_slots), torch.randn(4, memory_slots)]
    Q = [F.softmax(torch.randn(4, memory_slots), dim=1), F.softmax(torch.randn(4, memory_slots), dim=1)]
    
    # Test enhanced mode
    loss_fn = UnifiedLoss(mode="enhanced", feat_dim=feat_dim, num_centers=20)
    loss_dict = loss_fn(
        embeddings=embeddings,
        labels=labels,
        error_real=error_real,
        error_spoof=error_spoof,
        memory_bonafide=memory,
        logits=logits,
        Q=Q,
        attn_weights=attn
    )
    
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"OT loss: {loss_dict['ot_loss'].item():.4f}")
    print(f"OC loss: {loss_dict['oc_loss'].item():.4f}")
    print(f"Contrastive loss: {loss_dict['contrastive_loss'].item():.4f}")
    
    print("\n✓ Loss test passed!")


if __name__ == "__main__":
    test_loss()
