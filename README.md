# SAME: Speaker-Agnostic Memory-Enhanced Anti-Spoofing

Audio anti-spoofing using TitaNet embeddings with dual memory banks and optimal transport.

Name for ECE 477 final report: OTM-Titanet: Leveraging Pre-trained Speaker Embeddings with Optimal Transport Memory for Audio Anti-Spoofing

## Quick Start

### 1. Setup
```bash
# Download ASVspoof 2019 LA Dataset
# Download from: https://datashare.ed.ac.uk/handle/10283/3336
# Extract to: ../LA/
# Expected structure:
#   LA/
#   ├── ASVspoof2019_LA_train/
#   ├── ASVspoof2019_LA_dev/
#   ├── ASVspoof2019_LA_cm_protocols/
#   └── ASVspoof2019_LA_asv_scores/

# Download ASVspoof 2021 LA Evaluation Dataset
# Download from: https://www.asvspoof.org/index2021.html
# Extract to: ../../dataset/ASVspoof2021_LA_eval/
# Expected structure:
#   ASVspoof2021_LA_eval/
#   ├── flac/
#   └── keys/LA/CM/trial_metadata.txt

# Download TitaNet model from nvidia nemo webpage or hugginface
# small
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small
# rename as titanet_small.nemo

# Conda Environment
conda create -n <name> python=3.10
conda activate <name>
pip install -r requirements.txt


# Check configuration
python -c "from configs.config_working import Config; Config.print_config()"

```

### 2. Train Baseline
```bash
# Start training
python run_experiment.py baseline

# Monitor
tensorboard --logdir logs/
```

### 3. Run Experiments
```bash
# After baseline succeeds
# n: 1-9
python run_experiment.py exp{n}

```


-

## 📚 Documentation

- **Quick Start**: `docs/QUICKSTART.md`
- **Experiments**: `experiments/README.md`
- **Full Structure**: `docs/PROJECT_STRUCTURE.md`
- **Config Details**: `docs/FINAL_CONFIG_SUMMARY.md`

---

## 🏗️ Model Architecture

```
Input Audio (waveform)
    ↓
[TitaNet Encoder]
    ↓
192-dim Embedding (z)
    ↓
    ├─→ [Memory Bonafide Bank] ──→ Top-K Attention ──→ Reconstruction Error (error_real)
    │   (K=64 prototypes, 192-dim each)
    │
    └─→ [Memory Spoof Bank] ──→ Top-K Attention ──→ Reconstruction Error (error_spoof)
        (K=64 prototypes, 192-dim each)
    ↓
[Sinkhorn Optimal Transport] ──→ Uniform Slot Usage (prevent mode collapse)
    ↓
[Loss Computation]
    ├─→ Reconstruction Loss (L2 between z and reconstructed z)
    ├─→ OT Loss (uniform distribution constraint)
    ├─→ OC-Softmax Loss (angular margin classification)
    └─→ Diversity Loss (encourage slot diversity)
    ↓
Final Score = error_spoof - error_real
```

**Key Components:**
- **TitaNet Encoder**: Extracts 192-dim speaker embeddings from raw audio
- **Dual Memory Banks**: Learnable prototypes for bonafide and spoof patterns
- **Top-K Sparse Attention**: Selects most relevant K prototypes for reconstruction
- **Sinkhorn OT**: Ensures uniform usage of memory slots to prevent collapse
- **Multi-Loss Training**: Combines reconstruction, OT, OC-Softmax, and diversity losses

---

## 📁 Project Structure

```
SAME/
├── 🚀 Training Scripts
│   └── run_experiment.py     ⭐ Unified training & experiments
│
├── 📦 Core Modules
│   ├── configs/              Configuration files
│   │   ├── config_working.py ⭐ Working config
│   │   └── config.py         (legacy)
│   │
│   ├── models/               Model implementations
│   │   ├── model_memory.py   Main model (OTMemoryTitaNet)
│   │   ├── model_titanet.py  TitaNet wrapper
│   │   └── loss.py           Loss functions
│   │
│   ├── dataset.py            Data loading
│   ├── evaluate.py           Evaluation script
│   └── utils.py              Helper functions
│
├── 🧪 experiments/           Incremental experiments
│   ├── README.md             Experiment roadmap
│   ├── exp1_oc_softmax.py   + OC-Softmax
│   ├── exp2_multi_center.py + Multi-center
│   ├── exp3_contrastive.py  + Contrastive
│   ├── exp4_large_model.py  + Large model
│   ├── exp5_adaptive_margin.py + Adaptive margin scheduler
│   ├── exp6_score_fusion.py + Score fusion tuning
│   ├── exp7_large_memory.py + Larger memory bank
│   ├── exp8_titanet_only.py + TitaNet encoder only
│   └── exp9_no_ot.py        + Memory without OT
│
├── 📚 docs/                  Documentation
│   ├── QUICKSTART.md         ⭐ Start here
│   ├── PROJECT_STRUCTURE.md  Full structure
│   ├── REGRESSION_ANALYSIS.md Why old model was better
│   └── ...                   Analysis docs
│
├── 💾 checkpoints/           Model checkpoints
├── 📈 logs/                  TensorBoard logs
└── 📦 archive/               Old files
```

---
