"""
Working Configuration - Based on Old Model that Achieved EER 0.08%

This config reproduces the settings from titanet_ot_memory/version_1
which achieved EER 0.0008 (0.08%)

Key differences from current config:
1. freeze_encoder = False (MOST IMPORTANT!)
2. titanet_small instead of large
3. Lower learning rate (5e-5 vs 1e-4)
4. Tighter Sinkhorn (3 iters, ε=0.05)
5. Simple mode: lambda_oc = 0.0, lambda_contrastive = 0.0 (pure Reconstruction + OT)
"""
import os
from pytorch_lightning.strategies import DDPStrategy


class Config:
    """
    Working configuration that achieved EER 0.08%
    """
    
    # ===========================================
    # Mode Selection
    # ===========================================
    mode = "basic"  # Start with basic mode that worked
    
    # ===========================================
    # Data Paths
    # ===========================================
    base_path = "/home/feiyueh/hw/LA"
    train_protocol = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    train_audio_dir = "ASVspoof2019_LA_train/flac"
    dev_audio_dir = "ASVspoof2019_LA_dev/flac"
    asv_score_file = "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    
    # ===========================================
    # Audio Processing
    # ===========================================
    sample_rate = 16000
    max_length = 64600  # ~4 seconds
    
    # ===========================================
    # Model Architecture - MATCHING OLD CONFIG
    # ===========================================
    titanet_model = "titanet_small"  
    embed_dim = 192
    memory_slots = 64  
    freeze_encoder = False  
    top_k = 10
    
    # Multi-center OC-Softmax (used in enhanced mode with num_oc_centers > 1)
    num_oc_centers = 20
    
    # ===========================================
    # Training Hyperparameters - MATCHING OLD CONFIG
    # ===========================================
    batch_size = 64
    lr = 1e-4
    weight_decay = 2e-3
    max_steps = 5000
    warmup_steps = 500  # Warm-up steps for learning rate scheduler
    val_check_interval = 100
    log_every_n_steps = 10
    num_workers = 2
    
    # ===========================================
    # Data Augmentation
    # ===========================================
    use_data_augmentation = True
    rawboost_algo = [1, 2, 3]  # LnL + ISD + SSI
    use_codec_aug = True
    codec_aug_formats = ['mp3', 'aac']
    
    # ===========================================
    # Loss Hyperparameters - MATCHING OLD CONFIG
    # ===========================================
    lambda_recon = 1.0
    lambda_ot = 0.2  
    lambda_oc = 0.0  # Baseline: No OC-Softmax (pure Reconstruction + OT)
    lambda_diversity = 0.1
    lambda_contrastive = 0.0  # Baseline: No contrastive loss
    margin = 1.0
    
    # OC-Softmax parameters (used when lambda_oc > 0)
    m_real = 0.9
    m_fake = 0.3
    alpha = 20.0
    
    # ===========================================
    # Sinkhorn OT Parameters - MATCHING OLD CONFIG
    # ===========================================
    sinkhorn_iterations = 3  # ⭐ Changed from 10
    sinkhorn_epsilon = 0.05  # ⭐ Changed from 0.1
    logit_scale_init = 1.0
    
    # ===========================================
    # Enhanced Mode Settings (not used in basic)
    # ===========================================
    score_fusion = "recon"  # Use simple reconstruction score
    score_weight = 0.5
    use_adaptive_margin = False  # Disable for basic mode
    
    # ===========================================
    # Training Infrastructure
    # ===========================================
    accelerator = "gpu"
    devices = 2
    strategy = DDPStrategy(find_unused_parameters=True)
    precision = 32
    
    # ===========================================
    # Experiment Versioning
    # ===========================================
    experiment_version = "working_v1"  # New experiment name
    checkpoint_dir = f"checkpoints/{experiment_version}"
    save_top_k = 3
    log_dir = "logs"
    
    # ASVspoof 2021
    base_path_2021 = "/home/feiyueh/dataset/ASVspoof2021_LA_eval"
    eval_protocol_2021 = "keys/LA/CM/trial_metadata.txt"
    eval_audio_dir_2021 = "flac"
    
    # ===========================================
    # Path Helpers
    # ===========================================
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
    
    @classmethod
    def get_eval_protocol_2021_path(cls):
        return os.path.join(cls.base_path_2021, cls.eval_protocol_2021)
    
    @classmethod
    def get_eval_audio_2021_path(cls):
        return os.path.join(cls.base_path_2021, cls.eval_audio_dir_2021)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print(f"Configuration (Mode: {cls.mode.upper()})")
        print("="*60)
        print(f"TitaNet: {cls.titanet_model}")
        print(f"Freeze Encoder: {cls.freeze_encoder}")
        print(f"Memory Slots: {cls.memory_slots}")
        print(f"Learning Rate: {cls.lr}")
        print(f"Lambda OT: {cls.lambda_ot}")
        print(f"Sinkhorn: iters={cls.sinkhorn_iterations}, ε={cls.sinkhorn_epsilon}")
        if cls.mode == "enhanced":
            print(f"OC Centers: {cls.num_oc_centers}")
            print(f"Score Fusion: {cls.score_fusion}")
            print(f"Adaptive Margin: {cls.use_adaptive_margin}")
        print(f"Batch Size: {cls.batch_size}")
        print(f"Data Aug: {cls.use_data_augmentation}")
        print("="*60)

