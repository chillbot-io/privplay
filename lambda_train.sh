#!/bin/bash
# =============================================================================
# Privplay Lambda Training Script
# =============================================================================
# Run this on a Lambda GPU instance (A10/A100 recommended)
#
# Usage:
#   chmod +x lambda_train.sh
#   ./lambda_train.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Privplay Training Pipeline - Lambda Setup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------------------------
echo "[1/7] Setting up environment..."

# Create working directory
mkdir -p ~/privplay_training
cd ~/privplay_training

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "✓ Virtual environment ready"
echo ""

# -----------------------------------------------------------------------------
# 2. Install Dependencies
# -----------------------------------------------------------------------------
echo "[2/7] Installing dependencies..."

# Core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers and training
pip install transformers datasets accelerate

# Privplay dependencies
pip install typer rich faker spacy
pip install presidio-analyzer presidio-anonymizer

# Download spaCy model
python -m spacy download en_core_web_lg

echo "✓ Dependencies installed"
echo ""

# -----------------------------------------------------------------------------
# 3. Clone/Setup Privplay
# -----------------------------------------------------------------------------
echo "[3/7] Setting up Privplay..."

# Option A: Clone from git (uncomment and set your repo)
# git clone https://github.com/YOUR_USERNAME/privplay.git
# cd privplay
# pip install -e .

# Option B: If you're uploading via scp, the code should be in ~/privplay
if [ -d ~/privplay ]; then
    cd ~/privplay
    pip install -e .
    echo "✓ Privplay installed from ~/privplay"
else
    echo "ERROR: Privplay not found at ~/privplay"
    echo "Please either:"
    echo "  1. Clone your repo (edit this script)"
    echo "  2. Upload via: scp -r /mnt/d/privplay user@lambda-ip:~/"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# 4. Verify Clean Model Download
# -----------------------------------------------------------------------------
echo "[4/8] Verifying clean Stanford model..."

python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import shutil

MODEL_NAME = "StanfordAIMI/stanford-deidentifier-base"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Clear any cached version to ensure fresh download
model_cache_pattern = "models--StanfordAIMI--stanford-deidentifier-base"
for item in os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else []:
    if model_cache_pattern in item:
        path = os.path.join(CACHE_DIR, item)
        print(f"Clearing cached model: {path}")
        shutil.rmtree(path)

# Fresh download
print(f"Downloading fresh model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

print(f"✓ Model downloaded: {model.num_labels} labels")
print(f"✓ Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
EOF

echo ""

# -----------------------------------------------------------------------------
# 5. Run BASELINE Benchmarks (Before Training)
# -----------------------------------------------------------------------------
echo "[5/8] Running BASELINE benchmarks (before fine-tuning)..."
echo ""

echo "Baseline on Synthetic PHI:"
phi-train benchmark run synthetic_phi -n 500 --no-verify
BASELINE_SYNTH_ID=$(phi-train benchmark history -n 1 | grep -oP '[a-f0-9-]{36}' | head -1)

echo ""
echo "Baseline on AI4Privacy (small sample):"
phi-train benchmark run ai4privacy -n 500 --no-verify
BASELINE_AI4P_ID=$(phi-train benchmark history -n 1 | grep -oP '[a-f0-9-]{36}' | head -1)

echo ""
echo "Baseline IDs saved for comparison"
echo "  Synthetic: $BASELINE_SYNTH_ID"
echo "  AI4Privacy: $BASELINE_AI4P_ID"
echo ""

# -----------------------------------------------------------------------------
# 6. Run FULL Benchmark with Training Capture (10k samples)
# -----------------------------------------------------------------------------
echo "[6/7] Running AI4Privacy benchmark (10k samples) and capturing training data..."
echo "      This will take 15-30 minutes..."
echo ""

# Clear any existing training data to start fresh
phi-train reset --yes 2>/dev/null || true

# Run benchmark and capture errors as training data
phi-train benchmark run ai4privacy -n 10000 --capture-errors --no-verify

# Show what we captured
echo ""
echo "Captured training data:"
phi-train stats

echo ""

# -----------------------------------------------------------------------------
# 7. Fine-tune Model
# -----------------------------------------------------------------------------
echo "[7/7] Fine-tuning model..."
echo "      This will take 10-20 minutes on GPU..."
echo ""

# Fine-tune with conservative settings to avoid catastrophic forgetting
phi-train finetune \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --output ~/.privplay/models/ai4privacy_finetuned

echo ""

# -----------------------------------------------------------------------------
# 8. Re-benchmark to Measure Improvement
# -----------------------------------------------------------------------------
echo "[8/8] Re-running benchmarks to measure improvement..."
echo ""

# Point to the new model
export PRIVPLAY_MODEL_PATH=~/.privplay/models/ai4privacy_finetuned/final

echo "--- AFTER FINE-TUNING ---"
echo ""

# Run on Synthetic PHI (should NOT regress)
echo "Testing on Synthetic PHI (original domain - checking for regression):"
phi-train benchmark run synthetic_phi -n 500 --no-verify
AFTER_SYNTH_ID=$(phi-train benchmark history -n 1 | grep -oP '[a-f0-9-]{36}' | head -1)

echo ""

# Run on AI4Privacy (should improve significantly)
echo "Testing on AI4Privacy (training domain):"
phi-train benchmark run ai4privacy -n 1000 --no-verify
AFTER_AI4P_ID=$(phi-train benchmark history -n 1 | grep -oP '[a-f0-9-]{36}' | head -1)

echo ""

# -----------------------------------------------------------------------------
# Summary and Comparison
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Model saved to: ~/.privplay/models/ai4privacy_finetuned/final"
echo ""

echo "--- COMPARISON: SYNTHETIC PHI (checking for regression) ---"
if [ -n "$BASELINE_SYNTH_ID" ] && [ -n "$AFTER_SYNTH_ID" ]; then
    phi-train benchmark compare "$BASELINE_SYNTH_ID" "$AFTER_SYNTH_ID" 2>/dev/null || echo "  (run 'phi-train benchmark compare' manually)"
fi
echo ""

echo "--- COMPARISON: AI4PRIVACY (should improve) ---"
if [ -n "$BASELINE_AI4P_ID" ] && [ -n "$AFTER_AI4P_ID" ]; then
    phi-train benchmark compare "$BASELINE_AI4P_ID" "$AFTER_AI4P_ID" 2>/dev/null || echo "  (run 'phi-train benchmark compare' manually)"
fi
echo ""

echo "Full benchmark history:"
phi-train benchmark history -n 6
echo ""

echo "=============================================="
echo "To download the model to your local machine:"
echo "  scp -r ubuntu@<lambda-ip>:~/.privplay/models/ai4privacy_finetuned /mnt/d/privplay/models/"
echo ""
echo "To use in Privplay:"
echo "  export PRIVPLAY_MODEL_PATH=/path/to/ai4privacy_finetuned/final"
echo "=============================================="
