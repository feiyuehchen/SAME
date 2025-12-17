"""
Experiment 6: Score Fusion Tuning

Goal: Test different score fusion strategies and weights
Baseline: Exp 3/4/5 (using reconstruction score only)
Change: Test combined score fusion with different weights
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp6Config(BaseConfig):
    """Experiment 6: Score Fusion Tuning"""
    
    # Experiment metadata
    experiment_name = "exp6_score_fusion"
    experiment_version = "exp6"
    
    # Mode: Enhanced
    mode = "enhanced"
    
    # Use best settings from previous experiments (same as exp5)
    titanet_model = "titanet_large"
    num_oc_centers = 20
    lambda_oc = 0.5
    lambda_contrastive = 0.3
    lambda_diversity = 0.1
    
    # Match exp4/5 memory slots for large model
    memory_slots = 128
    
    # Adaptive margin (from exp5)
    use_adaptive_margin = True
    
    # NEW: Test combined score fusion (only change from exp5)
    score_fusion = "combined"  # Combine OC + reconstruction scores
    score_weight = 0.7  # Give more weight to OC score
    
    # Alternative strategies to test:
    # score_fusion = "oc"  # Test OC score only
    # score_weight = 0.5  # Equal weight (can test this too)


Config = Exp6Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 6: Score Fusion Tuning")
    print("="*70)
    print("\nChanges from previous:")
    print("  + Score fusion: combined (was recon only)")
    print("  + Score weight: 0.7 (more weight to OC score)")
    print("  + Can test different fusion strategies")
    print("\nExpected:")
    print("  Better balance between OC and reconstruction scores")
    print("  Target: < 0.03%")
    print("="*70)
    
    Exp6Config.print_config()

