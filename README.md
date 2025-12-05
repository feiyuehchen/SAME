# TitaNet + OT-Regularized Memory Network for ASVspoof 2019

This project implements a deepfake audio detection system using **TitaNet** (as a feature extractor) and an **Optimal Transport (OT) Regularized Memory Network**. It is designed for the ASVspoof 2019 Logical Access (LA) challenge.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create environment
conda create -n asv python=3.10 -y
conda activate asv

# Install dependencies
cd /home/feiyueh/hw/SAME  # (Or your project root)
pip install -r requirements.txt
```

### 2. Download TitaNet Model
You need the pre-trained `titanet_small.nemo` model.
```bash
python download_titanet.py
```
*Alternatively, download manually from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small) and place it in the root directory.*

### 3. Run Tests
Verify that the custom layers (Sinkhorn, Memory) and data loading are working:
```bash
python test_forward.py
python train.py --test
```

### 4. Start Training
```bash
python train.py
```
*Training logs will be saved to `logs/titanet_ot_memory`.*

### 5. Monitor Training
```bash
tensorboard --logdir logs/titanet_ot_memory
```
Access at http://localhost:6006.

---

## 📊 Evaluation

Once trained, you can evaluate the model using `evaluate.py`.

### Basic Evaluation (Complete Set)
Evaluates on the full development set (all 2,548 bona fide + 22,296 spoof samples).
```bash
python evaluate.py --checkpoint checkpoints/titanet_ot_memory/best.ckpt
```

### Comparison with SAMO (Target-Only)
To compare with the SAMO paper, use the `--target-only` flag, which restricts bona fide samples to target speakers only (matches SAMO protocol).
```bash
python evaluate.py --checkpoint checkpoints/titanet_ot_memory/best.ckpt --target-only
```

### Compare Both Modes
```bash
python evaluate.py --checkpoint checkpoints/titanet_ot_memory/best.ckpt --compare-both
```

### Full Evaluation with Score Saving
```bash
python evaluate.py \
  --checkpoint checkpoints/titanet_ot_memory/best.ckpt \
  --eval-eval \
  --compare-both \
  --save-scores \
  --output-dir evaluation_results
```

---

## 📁 Project Structure

```
.
├── config.py           # Hyperparameter configuration
├── dataset.py          # ASVspoof 2019 LA data loader
├── model_titanet.py    # TitaNet encoder wrapper
├── model_memory.py     # Memory Network + Sinkhorn implementation
├── loss.py             # Reconstruction + OT Loss
├── train.py            # Training script (PyTorch Lightning)
├── evaluate.py         # Evaluation script
├── eval_metrics.py     # EER & t-DCF metric calculation
├── utils.py            # Helper functions
├── test_forward.py     # Unit tests
├── download_titanet.py # Model downloader
└── TECHNICAL_DETAILS.md # Detailed architectural & theoretical documentation
```

## ⚙️ Configuration (`config.py`)

Key parameters to adjust:
- `batch_size`: Default 128. Lower if OOM.
- `memory_slots`: Default 64.
- `lambda_ot`: Weight for OT loss (default 0.3).
- `freeze_encoder`: Set `True` to freeze TitaNet, `False` to fine-tune.

## 📈 Performance Goals
- **EER**: < 5% (Target < 3%)
- **min t-DCF**: < 0.1

## 📝 Citation
If you use this code, please cite the ASVspoof 2019 challenge and relevant TitaNet/OT papers.

See `TECHNICAL_DETAILS.md` for in-depth explanation of the architecture, loss functions, and experimental setup.
