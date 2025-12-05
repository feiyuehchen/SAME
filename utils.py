"""
Utility functions for ASVspoof evaluation
"""
import numpy as np
from eval_metrics import compute_eer as compute_eer_official


def compute_eer(bonafide_scores, spoof_scores):
    """
    Compute Equal Error Rate (EER) using official ASVspoof 2019 implementation
    
    Args:
        bonafide_scores: Scores for bonafide samples (higher = more bonafide)
        spoof_scores: Scores for spoof samples (higher = more bonafide)
    
    Returns:
        eer: Equal Error Rate
        threshold: EER threshold
    
    Note:
        Uses official eval_metrics.compute_eer() for consistency with
        ASVspoof 2019 evaluation protocol.
    """
    # Use official implementation
    # target_scores = bonafide (positive class)
    # nontarget_scores = spoof (negative class)
    eer, threshold = compute_eer_official(bonafide_scores, spoof_scores)
    
    return eer, threshold


def load_asv_scores(asv_score_file):
    """
    Load ASV scores from file
    
    Format: SPEAKER_ID KEY SCORE
    where KEY is 'target', 'nontarget', or 'spoof'
    
    Args:
        asv_score_file: Path to ASV score file
    
    Returns:
        tar_asv: Target scores
        non_asv: Non-target scores
        spoof_asv: Spoof scores
        asv_threshold: EER threshold for ASV
    """
    from eval_metrics import compute_eer as compute_eer_asv
    
    # Load ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)
    
    # Extract scores by type
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']
    
    # Compute ASV EER threshold
    eer_asv, asv_threshold = compute_eer_asv(tar_asv, non_asv)
    
    return tar_asv, non_asv, spoof_asv, asv_threshold


def compute_tdcf(
    bonafide_scores,
    spoof_scores,
    tar_asv,
    non_asv,
    spoof_asv,
    asv_threshold
):
    """
    Compute tandem Detection Cost Function (t-DCF)
    
    Args:
        bonafide_scores: CM scores for bonafide samples
        spoof_scores: CM scores for spoof samples
        tar_asv: ASV target scores
        non_asv: ASV non-target scores
        spoof_asv: ASV spoof scores
        asv_threshold: ASV EER threshold
    
    Returns:
        min_tdcf: Minimum normalized t-DCF
    """
    from eval_metrics import obtain_asv_error_rates, compute_tDCF
    
    # ASV error rates
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )
    
    # Cost model (from ASVspoof 2019)
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }
    
    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(
        bonafide_scores,
        spoof_scores,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        cost_model,
        print_cost=False
    )
    
    # Get minimum t-DCF
    min_tdcf = np.min(tDCF_curve)
    
    return min_tdcf


def save_scores(scores, labels, utt_ids, output_file):
    """
    Save scores to file in ASVspoof format
    
    Format: UTT_ID LABEL SCORE
    
    Args:
        scores: Array of scores
        labels: Array of labels (0=bonafide, 1=spoof)
        utt_ids: List of utterance IDs
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for utt_id, label, score in zip(utt_ids, labels, scores):
            label_str = 'bonafide' if label == 0 else 'spoof'
            f.write(f"{utt_id} {label_str} {score:.6f}\n")


def load_scores(score_file):
    """
    Load scores from file
    
    Format: UTT_ID LABEL SCORE
    
    Args:
        score_file: Path to score file
    
    Returns:
        scores: Array of scores
        labels: Array of labels (0=bonafide, 1=spoof)
        utt_ids: List of utterance IDs
    """
    data = []
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            label_str = parts[1]
            score = float(parts[2])
            
            label = 0 if label_str == 'bonafide' else 1
            data.append((utt_id, label, score))
    
    utt_ids = [x[0] for x in data]
    labels = np.array([x[1] for x in data])
    scores = np.array([x[2] for x in data])
    
    return scores, labels, utt_ids

