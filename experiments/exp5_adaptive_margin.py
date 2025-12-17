"""
Experiment 5: Adaptive Margin (Curriculum Learning)

Goal: Test if adaptive margin scheduling improves convergence
Baseline: Best config from Exp 1-4
Change: Enable adaptive margin scheduler for OC-Softmax
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp5Config(BaseConfig):
    """Experiment 5: Adaptive Margin"""
    
    # Experiment metadata
    experiment_name = "exp5_adaptive_margin"
    experiment_version = "exp5"
    
    # Mode: Enhanced
    mode = "enhanced"
    
    # Use best settings from previous experiments (same as exp4)
    titanet_model = "titanet_large"
    num_oc_centers = 20
    lambda_oc = 0.5
    lambda_contrastive = 0.3
    lambda_diversity = 0.1
    
    # Match exp4 memory slots for large model
    memory_slots = 128
    
    # NEW: Enable adaptive margin (only change from exp4)
    use_adaptive_margin = True
    
    # Keep same scoring as exp4 (test fusion in exp6)
    score_fusion = "recon"
    score_weight = 0.5


Config = Exp5Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 5: Adaptive Margin")
    print("="*70)
    print("\nChanges from previous:")
    print("  + Adaptive margin scheduler (curriculum learning)")
    print("  + Margins gradually tighten during training")
    print("\nExpected:")
    print("  Better convergence, more stable training")
    print("  Target: < 0.05%")
    print("="*70)
    
    Exp5Config.print_config()

