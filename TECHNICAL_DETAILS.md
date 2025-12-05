# TitaNet + Optimal Transport Memory Network Technical Report

## 1. Project Overview

### 1.1 Research Goal
This project develops a robust **Memory-Augmented Supervised Learning System** for audio deepfake detection. The system combines:
- **TitaNet**: Speaker verification model by NVIDIA (as a feature extractor).
- **Memory Network**: Learnable memory bank to model the distribution of bona fide speech.
- **Optimal Transport**: Sinkhorn algorithm to ensure diverse coverage of the memory bank.

### 1.2 Theoretical Basis
**Supervised Learning Paradigm & Memory Reconstruction Bottleneck**:
- **Training**: Uses bona fide and spoof labels (not pure one-class learning).
    - Bona fide: Minimize reconstruction error + OT regularization.
    - Spoof: Maximize reconstruction error (hinge loss).
- **Inference**: Uses only reconstruction error as the anomaly score (no labels required).
- **Goal**: Compress bona fide speech near the memory bank while pushing away spoof speech.

**Difference from SAMO Paper**:
- **SAMO**: Uses speaker labels, builds one attractor per speaker.
- **This System**: Does not require speaker labels, uses speaker-agnostic memory prototypes.

---

## 2. Core Concepts

### 2.1 Memory Bank
**Concept**: A set of **learned templates** representing "what bona fide speech looks like".
- **Structure**: $M = [m_1, m_2, ..., m_{64}]$, where each $m_i$ is a 192-dim vector.
- **Mechanism**:
    1. Extract feature $z$ from audio via TitaNet.
    2. Calculate cosine similarity with all memory slots.
    3. Reconstruct $z_{recon}$ using attention-weighted sum of memory slots.
    4. Calculate reconstruction error: $||z - z_{recon}||^2$.
- **Intuition**:
    - Low error $\rightarrow$ Can be reconstructed by memory $\rightarrow$ Bona fide.
    - High error $\rightarrow$ Cannot be reconstructed $\rightarrow$ Spoof.

### 2.2 Optimal Transport (OT)
**Problem**: **Mode Collapse**. Without regularization, the model might use only a few memory slots, leaving others idle.
**Solution**: **Sinkhorn Algorithm**.
- Treats memory slots as "warehouses" and batch samples as "clients".
- Enforces a balanced transport plan where every memory slot receives an equal amount of "mass".
- **Effect**: Ensures diverse coverage of the acoustic space and better generalization.

---

## 3. Model Architecture

```
Raw Audio (64,600 samples, ~4s)
    ↓
┌─────────────────────────────────────────────────────────┐
│  TitaNet Encoder (pre-trained, titanet_small)           │
│  - Input: Raw waveform                                  │
│  - Output: 192-dim embedding (z)                        │
└─────────────────────────────────────────────────────────┘
    ↓
  z ∈ R^192 (normalized)
    ↓
┌─────────────────────────────────────────────────────────┐
│  Memory Network                                         │
│  - Learnable memory bank: M ∈ R^(64×192)                │
└─────────────────────────────────────────────────────────┘
    ↓
┌──────────────────┬──────────────────────────────────────┐
│ Reconstruction   │  Optimal Transport (Training only)   │
│ Path             │  Path                                │
├──────────────────┼──────────────────────────────────────┤
│ 1. Cosine Sim    │  1. Scaled similarity (logits)       │
│ 2. Attention     │  2. Sinkhorn Algorithm               │
│ 3. Reconstruct   │     → Q (soft assignment)            │
│ 4. Error         │  3. Consistency Loss                 │
└──────────────────┴──────────────────────────────────────┘
    ↓
Loss Function (Recon Loss + OT Loss)
```

### 3.1 TitaNet Encoder
- **Source**: NVIDIA NeMo (`titanet_small` v1.19.0).
- **Input**: Raw waveform (16kHz, mono).
- **Architecture**: 1D Conv + MegaBlocks (SE-ResNet style) + Attentive Statistics Pooling.
- **Strategy**: Can be frozen (transfer learning) or unfrozen (fine-tuning).

### 3.2 Memory Network Implementation
- **Update Mechanism**: Updated at **every training step** via backpropagation (unlike SAMO which updates every M epochs).
- **Gradient Sources**:
    1. **Reconstruction Path**: Main driver. Bona fide pulls memory closer; Spoof pushes memory away.
    2. **OT Path**: Indirectly affects learning via embedding alignment.

### 3.3 Optimal Transport Implementation
- **Parameter Coupling**: Logits are derived directly from `logit_scale * similarity`, not a separate projection layer.
- **Stability**:
    - `Q.detach()`: The target distribution $Q$ is detached from the computation graph.
    - **Numerical Stability**: Clamping and epsilon protection in Sinkhorn to prevent overflow/division by zero.

---

## 4. Loss Functions

Total Loss:
$$L_{total} = \lambda_{recon} \cdot L_{recon} + \lambda_{OT} \cdot L_{OT}$$

### 4.1 Reconstruction Loss ($L_{recon}$)
- **Bona fide**: $L_{recon\_bonafide} = \text{mean}(||z - z_{recon}||^2)$
- **Spoof (Hinge)**: $L_{recon\_spoof} = \text{mean}(\text{ReLU}(margin - ||z - z_{recon}||^2))$
- **Margin**: 1.0.

### 4.2 Optimal Transport Loss ($L_{OT}$)
- Computed only on **bona fide** samples.
- $L_{OT} = -\sum (Q_{detached} \cdot \log\_softmax(logits))$
- Enforces the model's similarity distribution to match the optimal transport plan $Q$.

---

## 5. Data Augmentation (RawBoost)
Implements the RawBoost technique (Tak et al., 2022) for on-the-fly augmentation.
- **Algorithm 1**: LnL Convolutive Noise (simulating channel distortion).
- **Algorithm 2**: ISD Additive Noise (impulsive noise).
- **Algorithm 3**: SSI Additive Noise (stationary noise).
- **Strategy**: Applied only during training. Randomly selects one algorithm per batch.

---

## 6. Evaluation Metrics

### 6.1 Equal Error Rate (EER)
- The point where False Acceptance Rate (FAR) = False Rejection Rate (FRR).
- **Score**: `score = -recon_error` (Higher score = More likely bona fide).

### 6.2 Minimum Tandem Detection Cost Function (min t-DCF)
- **Integrated Metric**: Evaluates the CM (Countermeasure) in tandem with an ASV (Automatic Speaker Verification) system.
- **Implementation**: Ported from ASVspoof 2019 official MATLAB/Python scripts.
- **ASV Scores**: Uses official ASV scores provided by the challenge.
- **Formula**: $t\text{-DCF} = P_{miss}C_{miss} + P_{fa}C_{fa} + P_{spoof}P_{fa\_spoof}C_{fa\_spoof}$

### 6.3 Evaluation Modes
1.  **Complete Evaluation**: Uses **all** trials in the dev/eval sets.
    - Dev: 2,548 bona fide + 22,296 spoof.
2.  **Target-Only Evaluation** (For SAMO comparison):
    - Uses only bona fide trials from **target speakers**.
    - Dev: 1,484 bona fide + 22,296 spoof.
    - **Bug Fix**: Previous implementation incorrectly filtered out spoof trials in this mode. Fixed to include all spoof trials.

---

## 7. Comparison with SAMO

| Feature | SAMO (ICASSP 2023) | This System (Ours) |
| :--- | :--- | :--- |
| **Backbone** | AASIST | TitaNet-Small |
| **Center Type** | Speaker Attractors | Memory Prototypes |
| **Speaker ID** | **Required** | **Not Required** (Speaker-agnostic) |
| **Center Count** | # of Speakers | Fixed (e.g., 64) |
| **Update Freq** | Every M epochs | Every Step |
| **Regularization** | Multi-center OC-Softmax | Optimal Transport |
| **Inference** | Enrollment-dependent | Unified Reconstruction Score |

**Performance Note**:
- SAMO (Target-only): ~0.88% EER (Eval).
- Ours: Designed to achieve competitive results (1-3% EER) without requiring speaker labels or enrollment.

---

## 8. Bug Fixes & Improvements

### 8.1 Target-Only Evaluation Fix
- **Issue**: `NaN` errors because spoof samples were accidentally filtered out when selecting "target-only" trials.
- **Fix**: Logic updated to `if trial_type in ['target', 'spoof']`.

### 8.2 NeMo Logging
- **Issue**: Excessive `sw_xxxx` logs from NeMo's speaker verification modules.
- **Fix**: Suppressed `nemo_logger` level to ERROR.

### 8.3 Numerical Stability
- **Sinkhorn**: Added clamping `[-50, 50]` and epsilon `1e-8` to prevent `NaN` during exp/div operations.
- **Logit Scale**: Initialized to 1.0 (was 10.0) to prevent gradient explosion.

