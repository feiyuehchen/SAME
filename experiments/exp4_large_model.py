"""
Experiment 4: Larger Model

Goal: Test if more capacity helps
Baseline: Best config from Exp 1-3
Change: TitaNet large (25M params)
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp4Config(BaseConfig):
    """Experiment 4: Larger Model"""
    
    # Experiment metadata
    experiment_name = "exp4_large_model"
    experiment_version = "exp4"
    
    # NEW: Larger model
    titanet_model = "titanet_large"  # 25M params vs 6M
    freeze_encoder = False  # Still trainable!
    
    
    # Use best settings from previous experiments
    mode = "enhanced"
    num_oc_centers = 20
    lambda_oc = 0.5
    lambda_contrastive = 0.3
    lambda_diversity = 0.1
    
    # May need more memory slots for larger model
    memory_slots = 128  # Increased from 64
    
    # Other settings
    use_adaptive_margin = False
    score_fusion = "recon"  # Use reconstruction score (test fusion in Exp6)
    score_weight = 0.5  # Not used when score_fusion = "recon"


Config = Exp4Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 4: Larger Model")
    print("="*70)
    print("\nChanges from previous:")
    print("  + TitaNet large (25M params, was small)")
    print("  + Memory slots: 128 (from 64)")
    print("  + LR: 3e-5 (from 1e-4)")
    print("\nExpected:")
    print("  Better performance, slower training")
    print("  Target: < 0.05%")
    print("\n⚠️ Warning: Requires more GPU memory")
    print("="*70)
    
    Exp4Config.print_config()

