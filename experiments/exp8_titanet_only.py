"""
Experiment 8: TitaNet only (no OT, no Memory)

Goal: Establish encoder-only baseline by removing both OT regularization
and memory banks; rely solely on OC-Softmax over embeddings.
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp8Config(BaseConfig):
    """Experiment 8: TitaNet encoder only"""
    
    # Experiment metadata
    experiment_name = "exp8_titanet_only"
    experiment_version = "exp8"
    
    # Mode: basic (single-center OC for scoring)
    mode = "basic"
    
    # Disable memory + OT
    memory_slots = 0
    top_k = 0
    lambda_recon = 0.0
    lambda_ot = 0.0
    lambda_diversity = 0.0
    
    # Use OC-Softmax on embeddings for detection
    lambda_oc = 0.5
    score_fusion = "oc"
    
    # No contrastive when memory is absent
    lambda_contrastive = 0.0


Config = Exp8Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 8: TitaNet Only (No OT, No Memory)")
    print("="*70)
    print("\nChanges:")
    print("  - memory_slots = 0, top_k = 0 (disable memory banks)")
    print("  - lambda_ot = 0.0, lambda_recon = 0.0")
    print("  - score_fusion = 'oc' (OC-Softmax only)")
    print("\nExpected:")
    print("  - Encoder-only baseline to quantify memory/OT gains")
    print("="*70)
    
    Exp8Config.print_config()

