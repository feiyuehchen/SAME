"""
Experiment 2: Multi-Center OC-Softmax

Goal: Test if multiple centers capture bonafide diversity better
Baseline: Exp 1 (single-center OC-Softmax)
Change: Use 20 learnable centers instead of 1, switch to enhanced mode
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp2Config(BaseConfig):
    """Experiment 2: Multi-Center OC-Softmax (Enhanced Mode)"""
    
    # Experiment metadata
    experiment_name = "exp2_multi_center"
    experiment_version = "exp2"
    
    # Mode: Enhanced for multi-center OC
    mode = "enhanced"
    
    # Multi-center OC-Softmax
    num_oc_centers = 20  # Multiple learnable centers (vs 1 in basic mode)
    lambda_oc = 0.5  # Same weight as Exp1
    m_real = 0.9
    m_fake = 0.3
    alpha = 20.0
    
    # Disable other enhanced features for now (test multi-center alone)
    lambda_contrastive = 0.0  # Test multi-center OC alone first
    lambda_diversity = 0.1  # Keep minimal diversity
    use_adaptive_margin = False
    score_fusion = "recon"  # Use reconstruction score (same as baseline)
    
    # Keep baseline settings
    # titanet_model = "titanet_small"
    # freeze_encoder = False
    # memory_slots = 64
    # lr = 1e-4


Config = Exp2Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 2: Multi-Center OC-Softmax")
    print("="*70)
    print("\nChanges from Exp1:")
    print("  + Multi-center OC-Softmax (20 centers, was 1)")
    print("  + Enhanced mode (was basic)")
    print("  + Keep contrastive disabled for now")
    print("\nExpected:")
    print("  Better than single center (captures more diversity)")
    print("  Target: < 0.08%")
    print("="*70)
    
    Exp2Config.print_config()

