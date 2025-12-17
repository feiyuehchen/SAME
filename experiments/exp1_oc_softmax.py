"""
Experiment 1: Add OC-Softmax Loss

Goal: Test if single-center OC-Softmax improves performance
Baseline: Working config (Reconstruction + OT only, lambda_oc = 0.0)
Change: Add OC-Softmax with single center (lambda_oc = 0.5)
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp1Config(BaseConfig):
    """Experiment 1: Baseline + OC-Softmax (Single Center)"""
    
    # Experiment metadata
    experiment_name = "exp1_oc_softmax"
    experiment_version = "exp1"
    
    # Mode: Basic mode with OC-Softmax (single center)
    mode = "basic"
    
    # OC-Softmax parameters (single center in basic mode)
    lambda_oc = 0.5  # Enable OC-Softmax loss
    m_real = 0.9
    m_fake = 0.3
    alpha = 20.0
    
    # Disable other enhanced features
    lambda_contrastive = 0.0  # Not used in basic mode
    use_adaptive_margin = False
    
    # Note: In basic mode, OCSoftmax uses single center automatically
    # num_oc_centers is only used in enhanced mode
    
    # Keep baseline settings
    # titanet_model = "titanet_small"
    # freeze_encoder = False
    # memory_slots = 64
    # lr = 1e-4
    # lambda_recon = 1.0
    # lambda_ot = 0.2
    # sinkhorn_iterations = 3
    # sinkhorn_epsilon = 0.05


# For training script
Config = Exp1Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 1: Add OC-Softmax")
    print("="*70)
    print("\nChanges from baseline:")
    print("  + OC-Softmax loss (λ=0.5, was 0.0)")
    print("  + Single center (automatic in basic mode)")
    print("\nExpected:")
    print("  EER should improve compared to baseline")
    print("  Target: < 0.1%")
    print("="*70)
    
    Exp1Config.print_config()

