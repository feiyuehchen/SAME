"""
Configuration file for TitaNet + OT-Memory ASVspoof 2019 training
"""
import os
from pytorch_lightning.strategies import DDPStrategy


class Config:
    # Data paths
    base_path = "/home/feiyueh/hw/LA"
    train_protocol = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    train_audio_dir = "ASVspoof2019_LA_train/flac"
    dev_audio_dir = "ASVspoof2019_LA_dev/flac"
    asv_score_file = "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    
    # Audio processing
    sample_rate = 16000
    max_length = 64000  
    
    # Model architecture
    titanet_model = "titanet_large"
    embed_dim = 192
    memory_slots = 64
    freeze_encoder = False  # Set to True to freeze TitaNet encoder (RECOMMENDED to prevent overfitting)
    
    # Training hyperparameters
    batch_size = 128  
    lr = 1e-4  # Reduced from 1e-4 to prevent overfitting
    weight_decay = 2e-4  # L2 regularization
    max_steps = 99*200  # Adjusted for larger batch size (50 batches per GPU * 50 = 2500 steps)
    val_check_interval = 99  # Validate every 50 steps (adjusted for DDP and larger batch)
    log_every_n_steps = 10
    num_workers = 24
    
    # Data Augmentation (RawBoost)
    # Following "ASV spoofing and deepfake detection using wav2vec 2.0 and data augmentation" (Tak et al.)
    use_data_augmentation = False  # Enable/disable on-the-fly data augmentation
    # For LA database: use algorithms [1,2,3] (convolutive + impulsive noise)
    # Algorithms: 1=LnL_convolutive_noise, 2=ISD_additive_noise, 3=SSI_additive_noise
    rawboost_algo = [1, 2, 3]  # List of augmentation algorithms (paper uses combination for LA)
    
    # Loss hyperparameters
    lambda_recon = 1.0
    lambda_ot = 0.3  # Increased from 0.1 for stronger regularization
    margin = 1.0  # Hinge loss margin for spoof samples
    
    # Sinkhorn algorithm
    sinkhorn_iterations = 3
    sinkhorn_epsilon = 0.05
    logit_scale_init = 1.0  # Initial value for learnable logit scaling (temperature)
    
    # Training infrastructure
    accelerator = "gpu"
    devices = 2  # Use all available GPUs
    strategy = DDPStrategy(find_unused_parameters=True)
    precision = 32
    
    # Experiment versioning
    # Change this for different runs: v1, v2, v3, freeze_encoder, etc.
    experiment_version = "titanet_ot_memory"
    
    # Checkpointing
    # Structure: checkpoints/<version>/titanet-ot-XXXXXX-0.XXXX.ckpt
    checkpoint_dir = f"/home/feiyueh/hw/titanet_asvspoof2019/checkpoints/{experiment_version}"
    save_top_k = 3  # Save top 3 checkpoints by EER
    
    # Logging
    # Structure: logs/<version>/version_0/events.out.tfevents...
    log_dir = "/home/feiyueh/hw/titanet_asvspoof2019/logs"
    
    @classmethod
    def get_train_protocol_path(cls):
        return os.path.join(cls.base_path, cls.train_protocol)
    
    @classmethod
    def get_dev_protocol_path(cls):
        return os.path.join(cls.base_path, cls.dev_protocol)
    
    @classmethod
    def get_train_audio_path(cls):
        return os.path.join(cls.base_path, cls.train_audio_dir)
    
    @classmethod
    def get_dev_audio_path(cls):
        return os.path.join(cls.base_path, cls.dev_audio_dir)
    
    @classmethod
    def get_asv_score_path(cls):
        return os.path.join(cls.base_path, cls.asv_score_file)

