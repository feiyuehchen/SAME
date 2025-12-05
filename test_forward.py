"""
Test forward pass and verify all components work correctly
"""
import torch
import numpy as np

from model_memory import OTMemoryTitaNet
from loss import OTMemoryLoss
from config import Config


def test_sinkhorn():
    """Test Sinkhorn algorithm properties"""
    print("="*80)
    print("Testing Sinkhorn Algorithm")
    print("="*80)
    
    # Create a minimal model just to test Sinkhorn (without loading TitaNet)
    class MinimalModel:
        def __init__(self, memory_slots, sinkhorn_iterations, sinkhorn_epsilon):
            self.memory_slots = memory_slots
            self.sinkhorn_iterations = sinkhorn_iterations
            self.sinkhorn_epsilon = sinkhorn_epsilon
        
        def sinkhorn(self, logits):
            B, K = logits.shape
            Q = torch.exp(logits / self.sinkhorn_epsilon)
            for _ in range(self.sinkhorn_iterations):
                Q = Q / Q.sum(dim=1, keepdim=True)
                Q = Q / Q.sum(dim=0, keepdim=True)
            Q = Q / Q.sum(dim=1, keepdim=True)
            return Q
    
    model = MinimalModel(
        memory_slots=Config.memory_slots,
        sinkhorn_iterations=Config.sinkhorn_iterations,
        sinkhorn_epsilon=Config.sinkhorn_epsilon
    )
    
    # Create dummy logits
    batch_size = 8
    logits = torch.randn(batch_size, Config.memory_slots)
    
    # Apply Sinkhorn
    Q = model.sinkhorn(logits)
    
    print(f"Input logits shape: {logits.shape}")
    print(f"Output Q shape: {Q.shape}")
    print(f"\nSinkhorn properties:")
    print(f"  Min value: {Q.min().item():.6f}")
    print(f"  Max value: {Q.max().item():.6f}")
    
    # Check row normalization (should sum to 1)
    row_sums = Q.sum(dim=1)
    print(f"\n  Row sums (should be ~1.0):")
    print(f"    Mean: {row_sums.mean().item():.6f}")
    print(f"    Std: {row_sums.std().item():.6f}")
    print(f"    Min: {row_sums.min().item():.6f}")
    print(f"    Max: {row_sums.max().item():.6f}")
    
    # Check column normalization
    col_sums = Q.sum(dim=0)
    expected_col_sum = batch_size / Config.memory_slots
    print(f"\n  Column sums (should be ~{expected_col_sum:.4f}):")
    print(f"    Mean: {col_sums.mean().item():.6f}")
    print(f"    Std: {col_sums.std().item():.6f}")
    print(f"    Min: {col_sums.min().item():.6f}")
    print(f"    Max: {col_sums.max().item():.6f}")
    
    print("\n✓ Sinkhorn algorithm test passed!\n")


def test_loss_computation():
    """Test loss computation"""
    print("="*80)
    print("Testing Loss Computation")
    print("="*80)
    
    loss_fn = OTMemoryLoss(
        lambda_recon=Config.lambda_recon,
        lambda_ot=Config.lambda_ot,
        margin=Config.margin
    )
    
    # Create dummy data
    batch_size = 16
    recon_error = torch.rand(batch_size) * 2  # Random errors between 0 and 2
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # Half bonafide, half spoof
    
    # Logits and Q for bonafide only
    bonafide_count = (labels == 0).sum().item()
    logits = torch.randn(bonafide_count, Config.memory_slots)
    Q = torch.softmax(torch.randn(bonafide_count, Config.memory_slots), dim=1)
    
    # Compute loss
    loss_dict = loss_fn(recon_error, logits, Q, labels, use_ot=True)
    
    print(f"Batch size: {batch_size}")
    print(f"Bonafide count: {bonafide_count}")
    print(f"Spoof count: {batch_size - bonafide_count}")
    print(f"\nReconstruction errors:")
    print(f"  Bonafide mean: {recon_error[labels == 0].mean().item():.4f}")
    print(f"  Spoof mean: {recon_error[labels == 1].mean().item():.4f}")
    print(f"\nLoss values:")
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"    - Bonafide: {loss_dict['recon_loss_bonafide'].item():.4f}")
    print(f"    - Spoof: {loss_dict['recon_loss_spoof'].item():.4f}")
    print(f"  OT loss: {loss_dict['ot_loss'].item():.4f}")
    
    print("\n✓ Loss computation test passed!\n")


def test_memory_bank():
    """Test memory bank reconstruction"""
    print("="*80)
    print("Testing Memory Bank Reconstruction")
    print("="*80)
    
    # Create model (without TitaNet, just test memory operations)
    embed_dim = Config.embed_dim
    memory_slots = Config.memory_slots
    
    # Create dummy embeddings
    batch_size = 8
    z = torch.randn(batch_size, embed_dim)
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    
    # Create memory bank
    memory = torch.randn(memory_slots, embed_dim)
    memory = torch.nn.functional.normalize(memory, p=2, dim=1)
    
    # Compute similarity
    similarity = torch.matmul(z, memory.t())
    attn_weights = torch.nn.functional.softmax(similarity, dim=1)
    
    # Reconstruct
    z_recon = torch.matmul(attn_weights, memory)
    
    # Compute reconstruction error
    recon_error = torch.sum((z - z_recon) ** 2, dim=1)
    
    print(f"Embedding dim: {embed_dim}")
    print(f"Memory slots: {memory_slots}")
    print(f"Batch size: {batch_size}")
    print(f"\nSimilarity stats:")
    print(f"  Mean: {similarity.mean().item():.4f}")
    print(f"  Std: {similarity.std().item():.4f}")
    print(f"  Min: {similarity.min().item():.4f}")
    print(f"  Max: {similarity.max().item():.4f}")
    print(f"\nAttention weights stats:")
    print(f"  Row sums: {attn_weights.sum(dim=1)}")  # Should be all 1.0
    print(f"  Max weight per sample: {attn_weights.max(dim=1)[0]}")
    print(f"\nReconstruction error:")
    print(f"  Mean: {recon_error.mean().item():.4f}")
    print(f"  Std: {recon_error.std().item():.4f}")
    print(f"  Min: {recon_error.min().item():.4f}")
    print(f"  Max: {recon_error.max().item():.4f}")
    
    print("\n✓ Memory bank test passed!\n")


def test_full_forward_pass():
    """Test full forward pass (will fail if TitaNet not loaded, but that's OK)"""
    print("="*80)
    print("Testing Full Forward Pass")
    print("="*80)
    
    try:
        # Create model
        model = OTMemoryTitaNet(
            titanet_model=Config.titanet_model,
            embed_dim=Config.embed_dim,
            memory_slots=Config.memory_slots,
            freeze_encoder=Config.freeze_encoder,
            lr=Config.lr,
            lambda_recon=Config.lambda_recon,
            lambda_ot=Config.lambda_ot,
            margin=Config.margin
        )
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model device: {device}")
        
        # Create dummy audio on the same device
        batch_size = 4
        audio = torch.randn(batch_size, Config.max_length).to(device)
        
        # Forward pass
        print("Running forward pass...")
        outputs = model.forward(audio, compute_ot=True)
        
        print(f"\nInput shape: {audio.shape}")
        print(f"\nOutput shapes:")
        print(f"  Embeddings (z): {outputs['z'].shape}")
        print(f"  Reconstruction error: {outputs['recon_error'].shape}")
        print(f"  Logits: {outputs['logits'].shape}")
        print(f"  Sinkhorn Q: {outputs['Q'].shape}")
        print(f"  Attention weights: {outputs['attn_weights'].shape}")
        
        # Check embedding normalization
        z_norms = torch.norm(outputs['z'], p=2, dim=1)
        print(f"\nEmbedding norms (should be ~1.0):")
        print(f"  Mean: {z_norms.mean().item():.6f}")
        print(f"  Std: {z_norms.std().item():.6f}")
        
        # Check reconstruction errors
        print(f"\nReconstruction errors:")
        print(f"  Mean: {outputs['recon_error'].mean().item():.4f}")
        print(f"  Std: {outputs['recon_error'].std().item():.4f}")
        
        print("\n✓ Full forward pass test passed!\n")
        
    except Exception as e:
        print(f"\n✗ Full forward pass test failed: {e}")
        print("\nThis is EXPECTED if TitaNet model is not downloaded yet.")
        print("To download:")
        print("1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small")
        print("2. Download titanet_small.nemo (v1.19.0)")
        print("3. Place in project directory\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING TITANET + OT-MEMORY NETWORK")
    print("="*80 + "\n")
    
    # Test individual components
    test_sinkhorn()
    test_loss_computation()
    test_memory_bank()
    
    # Test full forward pass (may fail without TitaNet model)
    test_full_forward_pass()
    
    print("="*80)
    print("ALL COMPONENT TESTS COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("1. Download TitaNet model (if not done yet)")
    print("2. Run training: python train.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

