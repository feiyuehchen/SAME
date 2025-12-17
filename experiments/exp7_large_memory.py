"""
Experiment 7: Baseline settings, larger memory bank

Goal: Keep everything aligned with the working baseline config, only
increase memory capacity and retrieval k to test if a bigger memory
improves reconstruction/OT performance.
"""
import sys
sys.path.append('..')

from configs.config_working import Config as BaseConfig


class Exp7Config(BaseConfig):
    """Experiment 7: Baseline + larger memory"""
    
    # Experiment metadata
    experiment_name = "exp7_larger_memory"
    experiment_version = "exp7"
    
    # Keep everything at baseline, only enlarge memory
    memory_slots = 128
    top_k = 20


Config = Exp7Config


if __name__ == "__main__":
    print("="*70)
    print("Experiment 7: Baseline with larger memory")
    print("="*70)
    print("\nChanges:")
    print("  - Use baseline settings from config_working")
    print("  - Increase memory_slots to 128")
    print("  - Increase top_k to 20")
    print("\nExpected:")
    print("  - Check if larger memory improves baseline performance")
    print("="*70)
    
    Exp7Config.print_config()

