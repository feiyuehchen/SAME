"""
TitaNet Encoder wrapper for NeMo pretrained model
Loads TitaNet-small from NVIDIA NeMo and provides embedding extraction
"""
import torch
import torch.nn as nn
from typing import Optional
import os
import logging

# Suppress NeMo verbose logging (sw_xxxx speaker IDs, etc.)
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
nemo_logger = logging.getLogger('nemo_logger.collections')
nemo_logger.setLevel(logging.ERROR)


class TitaNetEncoder(nn.Module):
    """
    Wrapper for NeMo TitaNet-small pretrained model
    
    TitaNet expects raw audio waveforms as input and outputs speaker embeddings
    """
    
    def __init__(
        self,
        model_name: str = "titanet_small",
        freeze: bool = False,
        embed_dim: int = 192
    ):
        """
        Args:
            model_name: Name of the NeMo model to load
            freeze: If True, freeze encoder weights
            embed_dim: Expected embedding dimension (192 for titanet_small)
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.embed_dim = embed_dim
        
        # Load NeMo model
        self.model = self._load_nemo_model()
        
        # Freeze if requested
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def _load_nemo_model(self):
        """Load pretrained NeMo TitaNet model"""
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            
            # Try to load from NGC
            # User should download the .nemo file manually if not available
            model_path = f"{self.model_name}.nemo"
            
            if os.path.exists(model_path):
                print(f"Loading TitaNet from local file: {model_path}")
                model = EncDecSpeakerLabelModel.restore_from(model_path)
            else:
                print(f"Attempting to load TitaNet from NGC: {self.model_name}")
                # Try to load from NGC (requires NGC API key and proper setup)
                model = EncDecSpeakerLabelModel.from_pretrained(
                    model_name=f"nvidia/nemo/{self.model_name}"
                )
            
            # Set model to eval mode for inference
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading NeMo model: {e}")
            print("\nPlease download the TitaNet model manually:")
            print("1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small")
            print("2. Download the .nemo file")
            print(f"3. Place it in the current directory as '{self.model_name}.nemo'")
            print("\nOr install NGC CLI and use:")
            print(f"ngc registry model download-version nvidia/nemo/{self.model_name}:1.19.0")
            raise RuntimeError("Failed to load NeMo TitaNet model")
    
    def forward(self, audio: torch.Tensor, audio_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through TitaNet encoder
        
        Args:
            audio: Raw audio waveform tensor
                   Shape: (B, T) or (B, 1, T) where T is time samples
            audio_lengths: Optional tensor of audio lengths for each sample in batch
                          Shape: (B,)
        
        Returns:
            embeddings: L2-normalized speaker embeddings
                       Shape: (B, embed_dim)
        """
        # Ensure audio is in the right shape (B, T)
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension if present
        
        # If audio_lengths not provided, assume all samples are full length
        if audio_lengths is None:
            audio_lengths = torch.tensor([audio.shape[1]] * audio.shape[0], 
                                        device=audio.device, dtype=torch.long)
        
        # Get embeddings from NeMo model
        # NeMo models have different methods depending on version
        # Try the most common approaches
        try:
            with torch.set_grad_enabled(not self.freeze):
                # Method 1: Direct forward call
                if hasattr(self.model, 'forward'):
                    # Some NeMo models return (logits, embeddings)
                    output = self.model(input_signal=audio, input_signal_length=audio_lengths)
                    if isinstance(output, tuple):
                        embeddings = output[-1]  # Usually embeddings are last
                    else:
                        embeddings = output
                
                # Method 2: Use get_embedding method if available
                elif hasattr(self.model, 'get_embedding'):
                    embeddings = self.model.get_embedding(audio, audio_lengths)
                
                # Method 3: Use encoder directly
                elif hasattr(self.model, 'encoder'):
                    processed_signal, processed_lengths = self.model.preprocessor(
                        input_signal=audio, length=audio_lengths
                    )
                    encoded, encoded_lengths = self.model.encoder(
                        audio_signal=processed_signal, length=processed_lengths
                    )
                    # Apply pooling if available
                    if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'pooling'):
                        embeddings = self.model.decoder.pooling(encoded, encoded_lengths)
                    else:
                        # Simple mean pooling
                        embeddings = torch.mean(encoded, dim=1)
                
                else:
                    raise AttributeError("Cannot find suitable method to extract embeddings")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise
        
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def train(self, mode: bool = True):
        """Override train method to respect freeze setting"""
        if not self.freeze:
            super().train(mode)
        return self
    
    def eval(self):
        """Set to evaluation mode"""
        super().train(False)
        return self


def test_titanet():
    """Test TitaNet encoder with dummy data"""
    print("Testing TitaNet Encoder...")
    
    # Create dummy audio (batch_size=2, length=64000 samples = 4 seconds @ 16kHz)
    dummy_audio = torch.randn(2, 64000)
    
    try:
        # Initialize encoder
        encoder = TitaNetEncoder(model_name="titanet_small", freeze=False)
        
        # Forward pass
        embeddings = encoder(dummy_audio)
        
        print(f"Input shape: {dummy_audio.shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Expected embedding dim: {encoder.embed_dim}")
        
        # Check if embeddings are normalized
        norms = torch.norm(embeddings, p=2, dim=1)
        print(f"Embedding norms (should be ~1.0): {norms}")
        
        print("TitaNet test passed!")
        
    except Exception as e:
        print(f"TitaNet test failed: {e}")
        print("\nThis is expected if the .nemo model file is not downloaded yet.")
        print("Please follow the instructions above to download the model.")


if __name__ == "__main__":
    test_titanet()

