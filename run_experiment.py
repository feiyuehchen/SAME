"""
Unified Experiment Runner

Usage:
    python run_experiment.py baseline
    python run_experiment.py exp1
    python run_experiment.py exp2
    python run_experiment.py exp3
    python run_experiment.py exp4
    python run_experiment.py exp5
    python run_experiment.py exp6
    python run_experiment.py exp7
    python run_experiment.py exp8
    python run_experiment.py exp9
"""
import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import ASVspoofDataModule
from models.model_memory import OTMemoryTitaNet


def load_config(exp_name):
    """Load configuration for experiment"""
    if exp_name == "baseline":
        from configs.config_working import Config
        print("\n🏁 Running BASELINE experiment")
    elif exp_name == "exp1":
        from experiments.exp1_oc_softmax import Config
        print("\n🧪 Running EXPERIMENT 1: OC-Softmax")
    elif exp_name == "exp2":
        from experiments.exp2_multi_center import Config
        print("\n🧪 Running EXPERIMENT 2: Multi-Center OC")
    elif exp_name == "exp3":
        from experiments.exp3_contrastive import Config
        print("\n🧪 Running EXPERIMENT 3: Contrastive Loss")
    elif exp_name == "exp4":
        from experiments.exp4_large_model import Config
        print("\n🧪 Running EXPERIMENT 4: Large Model")
    elif exp_name == "exp5":
        from experiments.exp5_adaptive_margin import Config
        print("\n🧪 Running EXPERIMENT 5: Adaptive Margin")
    elif exp_name == "exp6":
        from experiments.exp6_score_fusion import Config
        print("\n🧪 Running EXPERIMENT 6: Score Fusion Tuning")
    elif exp_name == "exp7":
        from experiments.exp7_large_memory import Config
        print("\n🧪 Running EXPERIMENT 7: Large Memory (TitaNet + Memory)")
    elif exp_name == "exp8":
        from experiments.exp8_titanet_only import Config
        print("\n🧪 Running EXPERIMENT 8: TitaNet Only (No Memory/OT)")
    elif exp_name == "exp9":
        from experiments.exp9_no_ot import Config
        print("\n🧪 Running EXPERIMENT 9: No OT (Memory + Reconstruction only)")
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")
    
    return Config


def main(args):
    """Main training function"""
    
    # Load config
    Config = load_config(args.experiment)
    
    # Set seed
    pl.seed_everything(42, workers=True)
    
    # Print header
    print("="*70)
    print(f"Experiment: {args.experiment.upper()}")
    print("="*70)
    Config.print_config()
    print("="*70 + "\n")
    
    # Data module
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
        augmentation_algo=Config.rawboost_algo,
        use_codec_aug=Config.use_codec_aug,
        codec_aug_formats=Config.codec_aug_formats
    )
    
    # Model
    print("Creating model...")
    model = OTMemoryTitaNet(
        mode=Config.mode,
        titanet_model=Config.titanet_model,
        embed_dim=Config.embed_dim,
        memory_slots=Config.memory_slots,
        num_oc_centers=Config.num_oc_centers,
        freeze_encoder=Config.freeze_encoder,
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        # Loss weights
        lambda_recon=Config.lambda_recon,
        lambda_ot=Config.lambda_ot,
        lambda_oc=Config.lambda_oc,
        lambda_div=Config.lambda_diversity,
        lambda_contrastive=Config.lambda_contrastive,
        # OC-Softmax
        m_real=Config.m_real,
        m_fake=Config.m_fake,
        alpha=Config.alpha,
        # Memory
        top_k=Config.top_k,
        margin=Config.margin,
        # Sinkhorn
        sinkhorn_iterations=Config.sinkhorn_iterations,
        sinkhorn_epsilon=Config.sinkhorn_epsilon,
        logit_scale_init=Config.logit_scale_init,
        # Enhanced mode settings
        score_fusion=Config.score_fusion,
        score_weight=Config.score_weight,
        use_adaptive_margin=Config.use_adaptive_margin,
        max_steps=Config.max_steps,
        warmup_steps=Config.warmup_steps,
    )
    
    # Directories
    experiment_name = f"{args.experiment}_{Config.mode}"
    checkpoint_dir = f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=Config.log_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, f"version_{logger.version}"),
        filename=f'{args.experiment}-{{step:06d}}-{{val/eer:.4f}}',
        monitor='val/eer',
        mode='min',
        save_top_k=Config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    early_stopping = EarlyStopping(
        monitor='val/eer',
        patience=30,
        mode='min',
        verbose=True
    )
    
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
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True,
    )
    
    # Train
    print("Starting training...")
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Results
    print("\n" + "="*70)
    print(f"Experiment {args.experiment.upper()} Complete!")
    print("="*70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best EER: {checkpoint_callback.best_model_score:.4f}")
    
    # Save results
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/{args.experiment}_results.txt", "w") as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Best EER: {checkpoint_callback.best_model_score:.4f}\n")
        f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
        f.write(f"Config: {Config.experiment_name if hasattr(Config, 'experiment_name') else 'baseline'}\n")
    
    print(f"\nResults saved to: {results_dir}/{args.experiment}_results.txt")
    print("\nNext steps:")
    print(f"  1. Check TensorBoard: tensorboard --logdir logs/")
    print(f"  2. Evaluate: python evaluate.py -c {checkpoint_callback.best_model_path}")
    print(f"  3. Compare with baseline")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('experiment', type=str, 
                       choices=['baseline', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9'],
                       help='Experiment to run')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)

