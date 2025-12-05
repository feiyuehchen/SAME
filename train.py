"""
Training script for TitaNet + OT-Memory Network
Uses PyTorch Lightning for distributed training with DDP
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

from config import Config
from dataset import ASVspoofDataModule
from model_memory import OTMemoryTitaNet


def main(args):
    """Main training function"""
    
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create data module
    print("Setting up data module...")
    data_module = ASVspoofDataModule(
        train_protocol=Config.get_train_protocol_path(),
        dev_protocol=Config.get_dev_protocol_path(),
        train_audio_dir=Config.get_train_audio_path(),
        dev_audio_dir=Config.get_dev_audio_path(),
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        max_length=Config.max_length,
        sample_rate=Config.sample_rate,
        use_augmentation=Config.use_data_augmentation,
        augmentation_algo=Config.rawboost_algo
    )
    
    # Create model
    print("Creating model...")
    model = OTMemoryTitaNet(
        titanet_model=Config.titanet_model,
        embed_dim=Config.embed_dim,
        memory_slots=Config.memory_slots,
        freeze_encoder=Config.freeze_encoder,
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        lambda_recon=Config.lambda_recon,
        lambda_ot=Config.lambda_ot,
        margin=Config.margin,
        sinkhorn_iterations=Config.sinkhorn_iterations,
        sinkhorn_epsilon=Config.sinkhorn_epsilon,
        logit_scale_init=Config.logit_scale_init,
        asv_score_path=Config.get_asv_score_path()
    )
    
    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    
    # Logger - use version as subdirectory
    # This creates: logs/<experiment_version>/version_X/
    logger = TensorBoardLogger(
        save_dir=Config.log_dir,
        name=Config.experiment_version,
        default_hp_metric=False
    )
    
    # Callbacks
    # Don't specify dirpath - will use logger's log_dir automatically
    # This creates: checkpoints/<experiment_version>/version_X/
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(Config.checkpoint_dir, f"version_{logger.version}"),
        filename='titanet-ot-{step:06d}-{val/eer:.4f}',
        monitor='val/eer',
        mode='min',
        save_top_k=Config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer
    print("Setting up trainer...")
    trainer = pl.Trainer(
        accelerator=Config.accelerator,
        devices=Config.devices,
        strategy=Config.strategy if torch.cuda.device_count() > 1 else 'auto',
        precision=Config.precision,
        max_steps=Config.max_steps,
        val_check_interval=Config.val_check_interval,
        log_every_n_steps=Config.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=False,  # Set to True for reproducibility (slower)
        enable_progress_bar=True,
    )
    
    # Print configuration
    print("\n" + "="*80)
    print("Training Configuration:")
    print("="*80)
    print(f"TitaNet Model: {Config.titanet_model}")
    print(f"Embedding Dim: {Config.embed_dim}")
    print(f"Memory Slots: {Config.memory_slots}")
    print(f"Freeze Encoder: {Config.freeze_encoder}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Learning Rate: {Config.lr}")
    print(f"Max Steps: {Config.max_steps}")
    print(f"Validation Interval: {Config.val_check_interval} steps")
    print(f"Data Augmentation: {'Enabled' if Config.use_data_augmentation else 'Disabled'}")
    if Config.use_data_augmentation:
        algo_names = {1: 'LnL_convolutive', 2: 'ISD_additive', 3: 'SSI_additive'}
        algo_str = ', '.join([algo_names.get(a, f'Unknown({a})') for a in Config.rawboost_algo])
        print(f"  RawBoost Algorithms: {algo_str}")
    print(f"Lambda Recon: {Config.lambda_recon}")
    print(f"Lambda OT: {Config.lambda_ot}")
    print(f"Margin: {Config.margin}")
    print(f"Devices: {Config.devices}")
    print(f"Strategy: {Config.strategy}")
    print("="*80 + "\n")
    
    # Train
    print("Starting training...")
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        
        if "TitaNet" in str(e) or "nemo" in str(e).lower():
            print("\n" + "="*80)
            print("ERROR: Failed to load TitaNet model")
            print("="*80)
            print("Please download the TitaNet model file:")
            print("1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small")
            print("2. Download the .nemo file (version 1.19.0)")
            print("3. Place it in the project directory as 'titanet_small.nemo'")
            print("\nAlternatively, use NGC CLI:")
            print("  ngc registry model download-version nvidia/nemo/titanet_small:1.19.0")
            print("="*80 + "\n")
        raise
    
    print("\nTraining completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best EER: {checkpoint_callback.best_model_score:.4f}")


def test_training_setup():
    """Test if training setup works (without actually training)"""
    print("Testing training setup...")
    
    # Create data module
    data_module = ASVspoofDataModule(
        train_protocol=Config.get_train_protocol_path(),
        dev_protocol=Config.get_dev_protocol_path(),
        train_audio_dir=Config.get_train_audio_path(),
        dev_audio_dir=Config.get_dev_audio_path(),
        batch_size=4,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        max_length=Config.max_length,
        sample_rate=Config.sample_rate
    )
    
    # Setup data
    data_module.setup('fit')
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    
    # Test dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Train batch audio shape: {batch[0].shape}")
    print(f"Train batch labels: {batch[1]}")
    
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    print(f"Val batch audio shape: {batch[0].shape}")
    print(f"Val batch labels: {batch[1]}")
    print(f"Val batch has {len(batch[2])} utterance IDs")
    
    print("\nTraining setup test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TitaNet + OT-Memory Network')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--test', action='store_true',
                       help='Test training setup without actually training')
    
    args = parser.parse_args()
    
    if args.test:
        test_training_setup()
    else:
        main(args)

