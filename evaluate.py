"""
Evaluation script for TitaNet + OT-Memory Network
Loads checkpoint and evaluates on ASVspoof 2019 LA dev/eval sets
"""
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from model_memory import OTMemoryTitaNet
from dataset import ASVspoofDataset
from config import Config
from utils import compute_eer, compute_tdcf, load_asv_scores, save_scores


def load_target_trials(asv_protocol_file):
    """
    Load target trial utterance IDs from ASV protocol
    
    For target-only evaluation (SAMO comparison):
    - Include bonafide samples with 'target' label
    - Include ALL spoof samples (labeled as 'spoof')
    
    Args:
        asv_protocol_file: Path to ASV protocol file
        
    Returns:
        set of target utterance IDs
    """
    target_ids = set()
    
    try:
        with open(asv_protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    utt_id = parts[1]
                    trial_type = parts[3]  # 'target', 'nontarget', or 'spoof'
                    
                    # Include target bonafide trials and ALL spoof trials
                    if trial_type in ['target', 'spoof']:
                        target_ids.add(utt_id)
    except FileNotFoundError:
        print(f"Warning: ASV protocol file not found: {asv_protocol_file}")
        return None
    
    return target_ids


def evaluate_model(model, dataset, device='cuda', desc='Evaluating', target_only_ids=None):
    """
    Evaluate model on a dataset and return scores
    
    Args:
        model: OTMemoryTitaNet model
        dataset: ASVspoofDataset
        device: Device to run on
        desc: Progress bar description
        target_only_ids: Optional set of utterance IDs to filter (for target-only evaluation)
    
    Returns:
        scores: Numpy array of scores (higher = more bonafide)
        labels: Numpy array of labels (0=bonafide, 1=spoof)
        utt_ids: List of utterance IDs
    """
    model.eval()
    model.to(device)
    
    all_scores = []
    all_labels = []
    all_utt_ids = []
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            audio, labels, utt_ids = batch
            audio = audio.to(device)
            
            # Forward pass
            outputs = model.forward(audio, compute_ot=False)
            
            # Score calculation for Dual Bank
            # Higher score = more bonafide
            # Bonafide: Low Real Error, High Spoof Error -> High Score
            # Spoof: High Real Error, Low Spoof Error -> Low Score
            scores = outputs['error_spoof'] - outputs['error_real']
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
            all_utt_ids.extend(utt_ids)
    
    # Concatenate all batches
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    # Filter by target-only if requested
    if target_only_ids is not None:
        mask = np.array([utt_id in target_only_ids for utt_id in all_utt_ids])
        scores = scores[mask]
        labels = labels[mask]
        all_utt_ids = [utt_id for utt_id, m in zip(all_utt_ids, mask) if m]
    
    return scores, labels, all_utt_ids


def compute_metrics(scores, labels, bonafide_scores=None, spoof_scores=None):
    """
    Compute evaluation metrics
    
    Args:
        scores: All scores
        labels: All labels
        bonafide_scores: Optional separate bonafide scores
        spoof_scores: Optional separate spoof scores
    
    Returns:
        Dictionary of metrics
    """
    # Separate by label if not provided
    if bonafide_scores is None:
        bonafide_mask = (labels == 0)
        spoof_mask = (labels == 1)
        bonafide_scores = scores[bonafide_mask]
        spoof_scores = scores[spoof_mask]
    
    # Compute EER
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    
    return {
        'eer': eer,
        'eer_threshold': threshold,
        'num_bonafide': len(bonafide_scores),
        'num_spoof': len(spoof_scores),
        'bonafide_score_mean': np.mean(bonafide_scores),
        'bonafide_score_std': np.std(bonafide_scores),
        'spoof_score_mean': np.mean(spoof_scores),
        'spoof_score_std': np.std(spoof_scores),
    }


def evaluate_with_tdcf(scores, labels, asv_score_file):
    """
    Compute t-DCF metric using ASV scores
    
    Args:
        scores: CM scores
        labels: CM labels
        asv_score_file: Path to ASV score file
    
    Returns:
        min_tdcf: Minimum t-DCF value
    """
    try:
        # Load ASV scores
        tar_asv, non_asv, spoof_asv, asv_threshold = load_asv_scores(asv_score_file)
        
        # Separate bonafide and spoof
        bonafide_mask = (labels == 0)
        spoof_mask = (labels == 1)
        bonafide_scores = scores[bonafide_mask]
        spoof_scores = scores[spoof_mask]
        
        # Compute t-DCF
        min_tdcf = compute_tdcf(
            bonafide_scores,
            spoof_scores,
            tar_asv,
            non_asv,
            spoof_asv,
            asv_threshold
        )
        
        return min_tdcf
    
    except Exception as e:
        print(f"Warning: Could not compute t-DCF: {e}")
        return None


def main(args):
    """Main evaluation function"""
    
    print("="*80)
    print("TitaNet + OT-Memory Network Evaluation")
    print("="*80)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    try:
        # Load model from checkpoint
        model = OTMemoryTitaNet.load_from_checkpoint(args.checkpoint)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Evaluate on development set
    if args.eval_dev:
        print("-"*80)
        print("Evaluating on Development Set")
        print("(Used for validation during training - checkpoint selection)")
        print("-"*80)
        
        dev_dataset = ASVspoofDataset(
            protocol_path=Config.get_dev_protocol_path(),
            audio_dir=Config.get_dev_audio_path(),
            max_length=Config.max_length,
            sample_rate=Config.sample_rate,
            return_utt_id=True
        )
        
        # Load target trials for target-only evaluation
        target_ids = None
        if args.target_only or args.compare_both:
            asv_protocol = os.path.join(
                Config.base_path,
                "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt"
            )
            target_ids = load_target_trials(asv_protocol)
            if target_ids:
                print(f"Loaded {len(target_ids)} target trial IDs for target-only evaluation")
        
        # Complete evaluation (all samples)
        if not args.target_only or args.compare_both:
            print("\n" + "="*60)
            print("COMPLETE EVALUATION (All samples)")
            print("="*60)
            
            dev_scores, dev_labels, dev_utt_ids = evaluate_model(
                model, dev_dataset, device, desc='Dev set (complete)'
            )
            
            # Compute metrics
            dev_metrics = compute_metrics(dev_scores, dev_labels)
            
            print(f"\nDevelopment Set Results (Complete):")
            print(f"  EER: {dev_metrics['eer']*100:.4f}%")
            print(f"  EER Threshold: {dev_metrics['eer_threshold']:.6f}")
            print(f"  # Bonafide: {dev_metrics['num_bonafide']}")
            print(f"  # Spoof: {dev_metrics['num_spoof']}")
            print(f"  Bonafide score: {dev_metrics['bonafide_score_mean']:.6f} ± {dev_metrics['bonafide_score_std']:.6f}")
            print(f"  Spoof score: {dev_metrics['spoof_score_mean']:.6f} ± {dev_metrics['spoof_score_std']:.6f}")
            
            # Compute t-DCF if ASV scores available
            if Config.asv_score_file:
                asv_score_path = Config.get_asv_score_path()
                if os.path.exists(asv_score_path):
                    min_tdcf = evaluate_with_tdcf(dev_scores, dev_labels, asv_score_path)
                    if min_tdcf is not None:
                        print(f"  min t-DCF: {min_tdcf:.4f}")
            
            # Save complete scores
            if args.save_scores:
                score_file = os.path.join(args.output_dir, 'dev_scores_complete.txt')
                save_scores(dev_scores, dev_labels, dev_utt_ids, score_file)
                print(f"  ✓ Scores saved to: {score_file}")
        
        # Target-only evaluation (for fair comparison with SAMO)
        if args.target_only or args.compare_both:
            if target_ids:
                print("\n" + "="*60)
                print("TARGET-ONLY EVALUATION (For SAMO comparison)")
                print("="*60)
                
                dev_scores_target, dev_labels_target, dev_utt_ids_target = evaluate_model(
                    model, dev_dataset, device, desc='Dev set (target-only)', 
                    target_only_ids=target_ids
                )
                
                # Compute metrics
                dev_metrics_target = compute_metrics(dev_scores_target, dev_labels_target)
                
                print(f"\nDevelopment Set Results (Target-only):")
                print(f"  EER: {dev_metrics_target['eer']*100:.4f}%")
                print(f"  EER Threshold: {dev_metrics_target['eer_threshold']:.6f}")
                print(f"  # Bonafide: {dev_metrics_target['num_bonafide']}")
                print(f"  # Spoof: {dev_metrics_target['num_spoof']}")
                print(f"  Bonafide score: {dev_metrics_target['bonafide_score_mean']:.6f} ± {dev_metrics_target['bonafide_score_std']:.6f}")
                print(f"  Spoof score: {dev_metrics_target['spoof_score_mean']:.6f} ± {dev_metrics_target['spoof_score_std']:.6f}")
                
                # Compute t-DCF if ASV scores available
                if Config.asv_score_file:
                    asv_score_path = Config.get_asv_score_path()
                    if os.path.exists(asv_score_path):
                        min_tdcf_target = evaluate_with_tdcf(dev_scores_target, dev_labels_target, asv_score_path)
                        if min_tdcf_target is not None:
                            print(f"  min t-DCF: {min_tdcf_target:.4f}")
                
                # Save target-only scores
                if args.save_scores:
                    score_file = os.path.join(args.output_dir, 'dev_scores_target_only.txt')
                    save_scores(dev_scores_target, dev_labels_target, dev_utt_ids_target, score_file)
                    print(f"  ✓ Scores saved to: {score_file}")
        
        print()
    
    # Evaluate on evaluation set (if available)
    if args.eval_eval:
        print("-"*80)
        print("Evaluating on ASVspoof 2019 LA Evaluation Set")
        print("(FINAL TEST - Report this result in papers)")
        print("-"*80)
        
        # Check if eval protocol exists
        eval_protocol = os.path.join(
            Config.base_path,
            "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        )
        eval_audio_dir = os.path.join(
            Config.base_path,
            "ASVspoof2019_LA_eval/flac"
        )
        
        if not os.path.exists(eval_protocol):
            print(f"✗ Evaluation protocol not found: {eval_protocol}")
            print("  Skipping evaluation set")
        else:
            eval_dataset = ASVspoofDataset(
                protocol_path=eval_protocol,
                audio_dir=eval_audio_dir,
                max_length=Config.max_length,
                sample_rate=Config.sample_rate,
                return_utt_id=True
            )
            
            # Load target trials for target-only evaluation
            eval_target_ids = None
            if args.target_only or args.compare_both:
                eval_asv_protocol = os.path.join(
                    Config.base_path,
                    "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
                )
                eval_target_ids = load_target_trials(eval_asv_protocol)
                if eval_target_ids:
                    print(f"Loaded {len(eval_target_ids)} target trial IDs for target-only evaluation")
            
            # Complete evaluation (all samples)
            if not args.target_only or args.compare_both:
                print("\n" + "="*60)
                print("COMPLETE EVALUATION 2019 (All samples)")
                print("="*60)
                
                eval_scores, eval_labels, eval_utt_ids = evaluate_model(
                    model, eval_dataset, device, desc='Eval set 2019 (complete)'
                )
                
                # Compute metrics
                eval_metrics = compute_metrics(eval_scores, eval_labels)
                
                print(f"\nEvaluation Set 2019 Results (Complete):")
                print(f"  EER: {eval_metrics['eer']*100:.4f}%")
                print(f"  EER Threshold: {eval_metrics['eer_threshold']:.6f}")
                print(f"  # Bonafide: {eval_metrics['num_bonafide']}")
                print(f"  # Spoof: {eval_metrics['num_spoof']}")
                print(f"  Bonafide score: {eval_metrics['bonafide_score_mean']:.6f} ± {eval_metrics['bonafide_score_std']:.6f}")
                print(f"  Spoof score: {eval_metrics['spoof_score_mean']:.6f} ± {eval_metrics['spoof_score_std']:.6f}")
                
                # Compute t-DCF if ASV scores available
                eval_asv_score = os.path.join(
                    Config.base_path,
                    "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
                )
                if os.path.exists(eval_asv_score):
                    min_tdcf_eval = evaluate_with_tdcf(eval_scores, eval_labels, eval_asv_score)
                    if min_tdcf_eval is not None:
                        print(f"  min t-DCF: {min_tdcf_eval:.4f}")
                
                # Save complete scores
                if args.save_scores:
                    score_file = os.path.join(args.output_dir, 'eval_scores_2019_complete.txt')
                    save_scores(eval_scores, eval_labels, eval_utt_ids, score_file)
                    print(f"  ✓ Scores saved to: {score_file}")
            
            # Target-only evaluation (for fair comparison with SAMO)
            if args.target_only or args.compare_both:
                if eval_target_ids:
                    print("\n" + "="*60)
                    print("TARGET-ONLY EVALUATION 2019 (For SAMO comparison)")
                    print("="*60)
                    
                    eval_scores_target, eval_labels_target, eval_utt_ids_target = evaluate_model(
                        model, eval_dataset, device, desc='Eval set 2019 (target-only)', 
                        target_only_ids=eval_target_ids
                    )
                    
                    # Compute metrics
                    eval_metrics_target = compute_metrics(eval_scores_target, eval_labels_target)
                    
                    print(f"\nEvaluation Set 2019 Results (Target-only):")
                    print(f"  EER: {eval_metrics_target['eer']*100:.4f}%")
                    print(f"  EER Threshold: {eval_metrics_target['eer_threshold']:.6f}")
                    print(f"  # Bonafide: {eval_metrics_target['num_bonafide']}")
                    print(f"  # Spoof: {eval_metrics_target['num_spoof']}")
                    print(f"  Bonafide score: {eval_metrics_target['bonafide_score_mean']:.6f} ± {eval_metrics_target['bonafide_score_std']:.6f}")
                    print(f"  Spoof score: {eval_metrics_target['spoof_score_mean']:.6f} ± {eval_metrics_target['spoof_score_std']:.6f}")
                    
                    # Compute t-DCF if ASV scores available
                    eval_asv_score = os.path.join(
                        Config.base_path,
                        "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
                    )
                    if os.path.exists(eval_asv_score):
                        min_tdcf_eval_target = evaluate_with_tdcf(eval_scores_target, eval_labels_target, eval_asv_score)
                        if min_tdcf_eval_target is not None:
                            print(f"  min t-DCF: {min_tdcf_eval_target:.4f}")
                    
                    # Save target-only scores
                    if args.save_scores:
                        score_file = os.path.join(args.output_dir, 'eval_scores_2019_target_only.txt')
                        save_scores(eval_scores_target, eval_labels_target, eval_utt_ids_target, score_file)
                        print(f"  ✓ Scores saved to: {score_file}")
        
        print()
    
    # Evaluate on ASVspoof 2021 LA (if requested)
    if args.eval_2021:
        print("-"*80)
        print("Evaluating on ASVspoof 2021 LA Evaluation Set")
        print("(Additional robustness test)")
        print("-"*80)
        
        eval_protocol_2021 = Config.get_eval_protocol_2021_path()
        eval_audio_dir_2021 = Config.get_eval_audio_2021_path()
        
        if not os.path.exists(eval_protocol_2021):
            print(f"✗ 2021 Evaluation protocol not found: {eval_protocol_2021}")
            print("  Skipping 2021 evaluation set")
        else:
            eval_dataset_2021 = ASVspoofDataset(
                protocol_path=eval_protocol_2021,
                audio_dir=eval_audio_dir_2021,
                max_length=Config.max_length,
                sample_rate=Config.sample_rate,
                return_utt_id=True
            )
            
            print("\n" + "="*60)
            print("COMPLETE EVALUATION 2021 (All samples)")
            print("="*60)
            
            eval_scores_2021, eval_labels_2021, eval_utt_ids_2021 = evaluate_model(
                model, eval_dataset_2021, device, desc='Eval set 2021'
            )
            
            # Compute metrics
            eval_metrics_2021 = compute_metrics(eval_scores_2021, eval_labels_2021)
            
            print(f"\nEvaluation Set 2021 Results (Complete):")
            print(f"  EER: {eval_metrics_2021['eer']*100:.4f}%")
            print(f"  EER Threshold: {eval_metrics_2021['eer_threshold']:.6f}")
            print(f"  # Bonafide: {eval_metrics_2021['num_bonafide']}")
            print(f"  # Spoof: {eval_metrics_2021['num_spoof']}")
            print(f"  Bonafide score: {eval_metrics_2021['bonafide_score_mean']:.6f} ± {eval_metrics_2021['bonafide_score_std']:.6f}")
            print(f"  Spoof score: {eval_metrics_2021['spoof_score_mean']:.6f} ± {eval_metrics_2021['spoof_score_std']:.6f}")
            
            # Save scores
            if args.save_scores:
                score_file = os.path.join(args.output_dir, 'eval_scores_2021.txt')
                save_scores(eval_scores_2021, eval_labels_2021, eval_utt_ids_2021, score_file)
                print(f"  ✓ Scores saved to: {score_file}")
        
        print()

    print("="*80)
    print("Evaluation Complete")
    print("="*80)
    
    # Comparison with SAMO
    if args.compare_samo:
        print()
        print("="*80)
        print("Comparison with SAMO (arXiv:2211.02718)")
        print("="*80)
        print()
        print("ASVspoof 2019 LA Official Splits:")
        print("  • Train: 25,380 samples (for training)")
        print("  • Dev: 24,844 samples (for validation/checkpoint selection)")
        print("  • Eval: 71,237 samples (for final testing/reporting)")
        print()
        print("SAMO Results on ASVspoof2019 LA:")
        print()
        print("  Dev Set (Target-only, 1,484 bonafide + 22,296 spoof):")
        print("    EER: ~1.09% (average)")
        print()
        print("  Eval Set (Target-only, 5,370 bonafide + 63,882 spoof):")
        print("    Method: SAMO with enrollment")
        print("    EER: 0.88% (best), 1.08% (average)")
        print("    min t-DCF: 0.0291 (best), 0.0356 (average)")
        print()
        print("Evaluation Protocol Differences:")
        print("  • SAMO: Target-only trials (1,484 dev / 5,370 eval bonafide)")
        print("  • Complete: All trials (2,548 dev / 7,355 eval bonafide)")
        print("  • Use --target-only for fair SAMO comparison")
        print()
        print("Method Differences:")
        print("  • SAMO: Speaker attractors (one per speaker)")
        print("  • Ours: Memory bank (speaker-agnostic prototypes)")
        print("  • SAMO: Requires speaker ID during training")
        print("  • Ours: No speaker ID required")
        print()
        print("NOTE: Report EVAL SET results in papers, not DEV SET")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate TitaNet + OT-Memory Network'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    
    parser.add_argument(
        '--eval-dev',
        action='store_true',
        help='Evaluate on development set'
    )
    
    parser.add_argument(
        '--eval-eval',
        action='store_true',
        default=True,
        help='Evaluate on evaluation set (if available)'
    )
    
    parser.add_argument(
        '--eval-2021',
        action='store_true',
        default=True,
        help='Evaluate on ASVspoof 2021 LA evaluation set'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    parser.add_argument(
        '--save-scores',
        action='store_true',
        help='Save scores to file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--target-only',
        action='store_true',
        help='Evaluate only on target trials (for fair SAMO comparison)'
    )
    
    parser.add_argument(
        '--compare-both',
        action='store_true',
        default=True,
        help='Show both complete and target-only results (default)'
    )
    
    parser.add_argument(
        '--compare-samo',
        action='store_true',
        default=True,
        help='Show comparison with SAMO paper'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.save_scores:
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

