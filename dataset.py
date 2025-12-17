"""
ASVspoof 2019 LA Dataset
Loads audio files and returns raw waveforms for TitaNet encoder
"""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Tuple, Optional
import numpy as np
import io

# Import RawBoost augmentation
try:
    from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
except ImportError:
    print("Warning: RawBoost not found. Data augmentation will be disabled.")


class CodecAugment:
    """
    Apply Codec Augmentation (MP3, AAC, etc.)
    Simulates audio compression artifacts
    """
    def __init__(self, sample_rate=16000, formats=['mp3', 'aac']):
        self.sample_rate = sample_rate
        self.formats = formats
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random codec compression
        Args:
            waveform: Audio waveform, shape (T,)
        Returns:
            Augmented waveform
        """
        # Select random format
        fmt = np.random.choice(self.formats)
        
        # Prepare for saving (ensure 2D: C x T)
        if waveform.dim() == 1:
            waveform_save = waveform.unsqueeze(0)
        else:
            waveform_save = waveform
            
        # Buffer for in-memory file
        out_file = io.BytesIO()
        
        try:
            # Map simple format names to torchaudio parameters if needed
            # For AAC, we often use MP4 container
            save_fmt = fmt
            if fmt == 'aac':
                save_fmt = 'mp4'
            
            # Save to buffer
            # Note: This requires ffmpeg to be available for mp3/mp4
            torchaudio.save(out_file, waveform_save, self.sample_rate, format=save_fmt)
            
            # Read back
            out_file.seek(0)
            aug_waveform, sr = torchaudio.load(out_file, format=save_fmt)
            
            # Resample back if sample rate changed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                aug_waveform = resampler(aug_waveform)
                
            # Ensure length matches original (padding or cropping)
            orig_len = waveform.shape[0]
            aug_len = aug_waveform.shape[1]
            
            if aug_len < orig_len:
                padding = orig_len - aug_len
                aug_waveform = torch.nn.functional.pad(aug_waveform, (0, padding))
            elif aug_len > orig_len:
                aug_waveform = aug_waveform[:, :orig_len]
                
            return aug_waveform.squeeze(0)
            
        except Exception as e:
            # If augmentation fails (e.g. missing ffmpeg), return original
            # Don't spam logs, but maybe print once?
            # print(f"Codec augmentation ({fmt}) failed: {e}")
            return waveform


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 Dataset that returns raw audio waveforms"""
    
    def __init__(
        self,
        protocol_path: str,
        audio_dir: str,
        max_length: int = 64000,
        sample_rate: int = 16000,
        return_utt_id: bool = False,
        use_augmentation: bool = False,
        augmentation_algo: list = [1, 2, 3],
        use_codec_aug: bool = False,
        codec_aug_formats: list = ['mp3', 'aac']
    ):
        """
        Args:
            protocol_path: Path to protocol file (e.g., ASVspoof2019.LA.cm.train.trn.txt)
            audio_dir: Path to directory containing audio files
            max_length: Maximum audio length in samples (will pad/truncate)
            sample_rate: Target sample rate
            return_utt_id: If True, also return utterance ID for evaluation
            use_augmentation: If True, apply RawBoost augmentation on-the-fly
            augmentation_algo: List of RawBoost algorithms to use [1,2,3]
            use_codec_aug: If True, apply Codec augmentation
            codec_aug_formats: List of formats for Codec augmentation
        """
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.return_utt_id = return_utt_id
        self.use_augmentation = use_augmentation
        self.augmentation_algo = augmentation_algo
        self.use_codec_aug = use_codec_aug
        
        # Initialize Codec Augmentation
        if use_codec_aug:
            self.codec_augment = CodecAugment(sample_rate, codec_aug_formats)
        else:
            self.codec_augment = None
        
        # Parse protocol file
        self.data = self._parse_protocol(protocol_path)
        
    def _parse_protocol(self, protocol_path: str):
        """Parse ASVspoof protocol file"""
        data = []
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    utt_id = parts[1]
                    
                    # Auto-detect label column (compatible with 2019 and 2021)
                    label = None
                    if parts[4] in ['bonafide', 'spoof']:
                        label = parts[4]
                    elif len(parts) > 5 and parts[5] in ['bonafide', 'spoof']:
                        label = parts[5]
                    
                    if label is not None:
                        # Convert label to binary: 0 = bonafide, 1 = spoof
                        label_int = 0 if label == 'bonafide' else 1
                        
                        data.append({
                            'speaker_id': speaker_id,
                            'utt_id': utt_id,
                            'label': label_int,
                            'label_str': label
                        })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _load_audio(self, utt_id: str) -> torch.Tensor:
        """Load audio file and preprocess"""
        # Audio files are in .flac format
        audio_path = os.path.join(self.audio_dir, f"{utt_id}.flac")
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Handle corrupted or unreadable audio files
            print(f"Warning: Failed to load audio {audio_path}: {e}")
            # Return zero tensor as placeholder
            return torch.zeros(self.max_length)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate to max_length
        if waveform.shape[1] < self.max_length:
            # Pad with zeros
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.max_length:
            # Random crop during training, center crop during eval
            # For now, center crop
            start = (waveform.shape[1] - self.max_length) // 2
            waveform = waveform[:, start:start + self.max_length]
        
        return waveform.squeeze(0)  # Return shape (T,)
    
    def _apply_rawboost(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply RawBoost augmentation on-the-fly
        Default parameters from: https://github.com/TakHemlata/RawBoost-antispoofing
        """
        # Convert to numpy for RawBoost
        audio_np = waveform.numpy()
        
        # Randomly select one augmentation algorithm from the list
        algo_id = np.random.choice(self.augmentation_algo)
        
        # RawBoost default parameters (optimized for LA)
        # Linear & Non-linear Convolutive Noise (LnL)
        N_f = 5  # Order of non-linearity
        nBands = 5  # Number of notch filters
        minF, maxF = 20, 8000  # Frequency range [Hz]
        minBW, maxBW = 100, 1000  # Bandwidth range [Hz]
        minCoeff, maxCoeff = 10, 100  # Filter coefficients
        minG, maxG = 0, 0  # Gain factor
        minBiasLinNonLin, maxBiasLinNonLin = 5, 20  # Linear/non-linear bias
        
        # Impulsive Signal-Dependent (ISD) Noise
        P = 10  # Max uniformly distributed samples [%]
        g_sd = 2  # Gain parameter
        
        # Stationary Signal-Independent (SSI) Noise
        SNRmin, SNRmax = 10, 40  # SNR range [dB]
        
        # Apply selected augmentation
        if algo_id == 1:
            # Algorithm 1: Linear and non-linear convolutive noise
            audio_np = LnL_convolutive_noise(
                audio_np, N_f, nBands, minF, maxF, minBW, maxBW,
                minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                maxBiasLinNonLin, self.sample_rate
            )
        elif algo_id == 2:
            # Algorithm 2: Impulsive signal-dependent additive noise
            audio_np = ISD_additive_noise(audio_np, P, g_sd)
        elif algo_id == 3:
            # Algorithm 3: Stationary signal-independent additive noise
            audio_np = SSI_additive_noise(
                audio_np, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                maxBW, minCoeff, maxCoeff, minG, maxG, self.sample_rate
            )
        
        # Normalize
        audio_np = normWav(audio_np, 0)
        
        # Convert back to tensor
        return torch.from_numpy(audio_np).float()
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample"""
        item = self.data[idx]
        
        # Load audio
        waveform = self._load_audio(item['utt_id'])
        
        # Apply data augmentation if enabled (training only)
        if self.use_augmentation:
            try:
                waveform = self._apply_rawboost(waveform)
            except Exception as e:
                # If augmentation fails, use original waveform
                print(f"Warning: RawBoost augmentation failed for {item['utt_id']}: {e}")
                pass
        
        # Apply Codec augmentation if enabled (training only)
        if self.use_codec_aug and self.codec_augment is not None:
            # Apply with 50% probability if not combined with RawBoost, 
            # or sequentially. Let's apply it sequentially but randomly.
            # Here we apply it always if enabled, or maybe random choice?
            # Usually augmentation is applied with some probability.
            # Let's assume if enabled, we apply it. 
            # But we might want to mix original, rawboost, and codec.
            # For now, apply on top of potentially RawBoosted audio.
            waveform = self.codec_augment(waveform)
        
        label = item['label']
        
        if self.return_utt_id:
            return waveform, label, item['utt_id']
        else:
            return waveform, label


class ASVspoofDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ASVspoof 2019"""
    
    def __init__(
        self,
        train_protocol: str,
        dev_protocol: str,
        train_audio_dir: str,
        dev_audio_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        max_length: int = 64000,
        sample_rate: int = 16000,
        use_augmentation: bool = False,
        augmentation_algo: list = [1, 2, 3],
        use_codec_aug: bool = False,
        codec_aug_formats: list = ['mp3', 'aac']
    ):
        super().__init__()
        self.train_protocol = train_protocol
        self.dev_protocol = dev_protocol
        self.train_audio_dir = train_audio_dir
        self.dev_audio_dir = dev_audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.use_augmentation = use_augmentation
        self.augmentation_algo = augmentation_algo
        self.use_codec_aug = use_codec_aug
        self.codec_aug_formats = codec_aug_formats
        
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets"""
        if stage == "fit" or stage is None:
            # Training set (with augmentation if enabled)
            self.train_dataset = ASVspoofDataset(
                protocol_path=self.train_protocol,
                audio_dir=self.train_audio_dir,
                max_length=self.max_length,
                sample_rate=self.sample_rate,
                return_utt_id=False,
                use_augmentation=self.use_augmentation,
                augmentation_algo=self.augmentation_algo,
                use_codec_aug=self.use_codec_aug,
                codec_aug_formats=self.codec_aug_formats
            )
            
            # Development set for validation (official split, NO augmentation)
            # This is the correct usage - Dev set is designed for model selection
            # Final results should be reported on Eval set
            self.val_dataset = ASVspoofDataset(
                protocol_path=self.dev_protocol,
                audio_dir=self.dev_audio_dir,
                max_length=self.max_length,
                sample_rate=self.sample_rate,
                return_utt_id=True,  # Return utt_id for EER calculation
                use_augmentation=False,  # Never augment validation data
                use_codec_aug=False
            )
            
            print(f"Using official ASVspoof 2019 LA splits:")
            print(f"  Train: {len(self.train_dataset)} samples" + 
                  (f" (with RawBoost augmentation: {self.augmentation_algo})" if self.use_augmentation else "") +
                  (f" (with Codec augmentation: {self.codec_aug_formats})" if self.use_codec_aug else ""))
            print(f"  Dev (validation): {len(self.val_dataset)} samples (no augmentation)")
            print(f"  → Final evaluation should be done on Eval set")
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2,
        )


def test_dataset():
    """Test dataset loading"""
    from config import Config
    
    print("Testing ASVspoofDataset...")
    
    # Test training dataset
    train_dataset = ASVspoofDataset(
        protocol_path=Config.get_train_protocol_path(),
        audio_dir=Config.get_train_audio_path(),
        max_length=Config.max_length,
        sample_rate=Config.sample_rate
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Load a sample
    waveform, label = train_dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Label: {label} ({'bonafide' if label == 0 else 'spoof'})")
    
    # Test dataloader
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch_waveform, batch_labels = next(iter(dataloader))
    print(f"Batch waveform shape: {batch_waveform.shape}")
    print(f"Batch labels: {batch_labels}")
    
    print("Dataset test passed!")


if __name__ == "__main__":
    test_dataset()

