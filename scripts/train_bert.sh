#!/usr/bin/env bash
# ============================================================
# Train DistilBERT model — run this on your machine
# Works on CPU (~40 min) or GPU (~10 min)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "============================================"
echo "  🧠 DistilBERT Training"
echo "============================================"

# Check torch
python3 -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch {torch.__version__}')
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Estimated time: ~10 minutes')
else:
    print(f'No GPU detected — training on CPU')
    print(f'Estimated time: ~30-40 minutes')
    print(f'Tip: Use Google Colab for faster training')
print()
"

# Check if data exists
if [ ! -f "data/twitter_airline_sentiment.csv" ]; then
    echo "[INFO] Generating dataset..."
    python3 scripts/generate_data.py
fi

# Train
echo ""
echo "[INFO] Starting DistilBERT training..."
echo "       This will download distilbert-base-uncased (~260 MB) on first run."
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from src.data import prepare_dataset, split_dataset
from src.bert_model import BERTTrainer, print_test_report
from src.baseline import BaselineModel, evaluate_on_test
import json

# ── Data ──
print('='*60)
print('STEP 1: Preparing data')
print('='*60)
df = prepare_dataset()
train_df, val_df, test_df = split_dataset(df)

# ── BERT ──
print()
print('='*60)
print('STEP 2: Training DistilBERT')
print('='*60)
trainer = BERTTrainer()
history = trainer.train(train_df, val_df, epochs=5)

# ── Test ──
print()
print('='*60)
print('STEP 3: Evaluating on test set')
print('='*60)
print_test_report(trainer, test_df)
bert_metrics, _ = trainer.evaluate(test_df)

# ── Baseline comparison ──
print()
print('='*60)
print('STEP 4: Comparing with Baseline')
print('='*60)
baseline = BaselineModel()
baseline.train(train_df, val_df)
baseline_metrics = evaluate_on_test(baseline, test_df)
baseline.save()

# ── Summary ──
print()
print('='*60)
print('FINAL COMPARISON: Baseline vs BERT')
print('='*60)
print(f'{\"Task\":<15} {\"Model\":<12} {\"Accuracy\":<12} {\"F1-macro\":<12}')
print('-' * 50)
for task in ['sentiment', 'category', 'criticality']:
    bl = baseline_metrics.get(task, {})
    bt = bert_metrics.get(task, {})
    print(f'{task:<15} {\"Baseline\":<12} {bl.get(\"accuracy\",0):<12.4f} {bl.get(\"f1_macro\",0):<12.4f}')
    print(f'{\"\":<15} {\"BERT\":<12} {bt.get(\"accuracy\",0):<12.4f} {bt.get(\"f1_macro\",0):<12.4f}')
    delta = bt.get('f1_macro',0) - bl.get('f1_macro',0)
    print(f'{\"\":<15} {\"Δ BERT\":<12} {\"\":<12} {delta:+.4f}')
    print()

# ── Save report ──
report = {
    'baseline': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in baseline_metrics.items()},
    'bert': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in bert_metrics.items()},
}
with open('reports/metrics.json', 'w') as f:
    json.dump(report, f, indent=2)
print('[INFO] Metrics saved to reports/metrics.json')

# ── Demo predictions ──
print()
print('='*60)
print('DEMO PREDICTIONS (BERT)')
print('='*60)
examples = [
    'The flight was fast and there were no delays.',
    'My flight was delayed 6 hours and nobody helped us!',
    'Amazing crew, comfortable seats, arrived early!',
    'Lost my luggage and customer service hung up on me. Want a refund!',
    'Average flight. Nothing special but okay for the price.',
    'Not a single problem. Boarding was smooth and crew was friendly.',
    'Check-in took forever and they lost my reservation. Unacceptable!',
]
for text in examples:
    r = trainer.predict_single(text)
    s, c, cr = r['sentiment'], r['category'], r['criticality']
    print(f'  {s[\"label\"]:>10} | {c[\"label\"]:>18} | {cr[\"label\"]:>7} ← {text}')

print()
print('✅ BERT training complete! Model saved to models/bert/')
print('   Run API: ./run.sh')
"

echo ""
echo "============================================"
echo "  ✅ Done! BERT model trained and saved."
echo "============================================"
