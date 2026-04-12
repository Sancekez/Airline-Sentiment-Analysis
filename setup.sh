#!/usr/bin/env bash
# ============================================================
# Airline Sentiment Analysis — Full Setup & Run
# Works on clean Ubuntu 20.04 / 22.04 / 24.04 + Python 3.10+
# ============================================================
set -e

echo "============================================"
echo "  ✈️  Airline Sentiment Analysis — Setup"
echo "============================================"
echo ""

# ── 1. System packages ────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3-venv python3-pip python3-dev > /dev/null 2>&1
echo "  ✅ System packages installed"

# ── 2. Virtual environment ────────────────────────────
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✅ Virtual environment created"
else
    echo "  ✅ Virtual environment already exists"
fi
source venv/bin/activate

# ── 3. Python dependencies ────────────────────────────
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✅ Dependencies installed"

# ── 4. Generate dataset ───────────────────────────────
echo "[4/6] Generating dataset..."
python scripts/generate_data.py

# ── 5. Run tests ──────────────────────────────────────
echo "[5/6] Running tests..."
python -m pytest tests/ -v
echo ""

# ── 6. Train baseline ────────────────────────────────
echo "[6/6] Training baseline model..."
python scripts/train.py --baseline-only

echo ""
echo "============================================"
echo "  ✅ SETUP COMPLETE!"
echo "============================================"
echo ""
echo "  To activate environment:"
echo "    source venv/bin/activate"
echo ""
echo "  To start API (with baseline model):"
echo "    ./run.sh"
echo ""
echo "  To train BERT model (recommended!):"
echo "    chmod +x scripts/train_bert.sh && ./scripts/train_bert.sh"
echo ""
echo "  API docs:"
echo "    http://localhost:8000/docs"
echo ""
