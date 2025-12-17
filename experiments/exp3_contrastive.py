"""
Experiment 3: Add Contrastive Loss

Goal: Explicitly structure feature space
Baseline: Exp 2 (if successful)
Change: Add contrastive memory loss
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp3Config(BaseConfig):
    """Experiment 3: + Contrastive Loss"""
    
    # Experiment metadata
    experiment_name = "exp3_contrastive"
    experiment_version = "exp3"
    
    # Mode: Enhanced
    mode = "enhanced"
    
    # Multi-center OC-Softmax (from Exp 2)
    num_oc_centers = 20
    lambda_oc = 0.5
    
    # NEW: Contrastive loss
    lambda_contrastive = 0.3
    
    # Diversity
    lambda_diversity = 0.1
    
    # Other settings
    use_adaptive_margin = False
    score_fusion = "recon"  # Use reconstruction score (same as Exp2)
    score_weight = 0.5  # Not used when score_fusion = "recon"


Config = Exp3Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 3: Add Contrastive Loss")
    print("="*70)
    print("\nChanges from Exp 2:")
    print("  + Contrastive loss (λ=0.3)")
    print("  + Keep reconstruction score (test fusion in Exp6)")
    print("\nExpected:")
    print("  Better feature space structure")
    print("  Target: < 0.08%")
    print("="*70)
    
    Exp3Config.print_config()

