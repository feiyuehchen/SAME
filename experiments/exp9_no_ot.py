"""
Experiment 9: No OT (Memory + Reconstruction only)

Goal: Test performance without OT regularization while keeping memory banks
and reconstruction loss. This isolates the contribution of OT to the baseline.
Baseline: Working config (Reconstruction + OT + Memory)
Change: Remove OT (lambda_ot = 0.0), keep Memory and Reconstruction
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp9Config(BaseConfig):
    """Experiment 9: Memory + Reconstruction (No OT)"""
    
    # Experiment metadata
    experiment_name = "exp9_no_ot"
    experiment_version = "exp9"
    
    # Mode: basic (same as baseline)
    mode = "basic"
    
    # Remove OT regularization
    lambda_ot = 0.0
    
    # Disable Sinkhorn algorithm (set iterations to 0)
    sinkhorn_iterations = 0  # Disable Sinkhorn algorithm execution
    sinkhorn_epsilon = 0.05  # Not used when iterations = 0, but keep for consistency
    
    # Keep Memory and Reconstruction
    memory_slots = 64  # Keep memory banks
    top_k = 10
    lambda_recon = 1.0  # Keep reconstruction loss
    
    # Keep baseline OC-Softmax settings (disabled)
    lambda_oc = 0.0
    lambda_contrastive = 0.0
    
    # Keep other baseline settings
    # titanet_model = "titanet_small"
    # freeze_encoder = False
    # lr = 1e-4


Config = Exp9Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 9: No OT (Memory + Reconstruction only)")
    print("="*70)
    print("\nChanges from baseline:")
    print("  - lambda_ot = 0.0 (was 0.2, remove OT regularization)")
    print("  - sinkhorn_iterations = 0 (was 3, disable Sinkhorn algorithm)")
    print("  + Keep memory_slots = 64, top_k = 10")
    print("  + Keep lambda_recon = 1.0")
    print("\nExpected:")
    print("  Performance may degrade without OT regularization")
    print("  This isolates OT's contribution to baseline performance")
    print("="*70)
    
    Exp9Config.print_config()

