#!/usr/bin/env python3
"""Main training pipeline: data → baseline → BERT → evaluation → report."""
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import prepare_dataset, split_dataset
from src.baseline import BaselineModel, evaluate_on_test
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR


def train_baseline(train_df, val_df, test_df):
    """Train and evaluate baseline model."""
    print("\n" + "=" * 60)
    print("STAGE 1: BASELINE (TF-IDF + LogisticRegression)")
    print("=" * 60)
    
    model = BaselineModel()
    val_results = model.train(train_df, val_df)
    test_results = evaluate_on_test(model, test_df)
    model.save()
    
    return test_results


def train_bert(train_df, val_df, test_df, epochs=5):
    """Train and evaluate BERT model."""
    print("\n" + "=" * 60)
    print("STAGE 2: DistilBERT FINE-TUNING")
    print("=" * 60)
    
    from src.bert_model import BERTTrainer, print_test_report
    
    trainer = BERTTrainer()
    history = trainer.train(train_df, val_df, epochs=epochs)
    
    # Test evaluation
    test_metrics, _ = trainer.evaluate(test_df)
    print_test_report(trainer, test_df)
    
    return test_metrics, trainer


def save_report(baseline_results, bert_results):
    """Save comparison report as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "baseline": {k: {kk: round(vv, 4) for kk, vv in v.items()} 
                     for k, v in baseline_results.items()},
    }
    if bert_results:
        report["bert"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} 
                          for k, v in bert_results.items()}
    
    report_path = REPORTS_DIR / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Metrics report saved to {report_path}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Model':<12} {'Accuracy':<12} {'F1-macro':<12}")
    print("-" * 50)
    for task in ["sentiment", "category", "criticality"]:
        bl = baseline_results.get(task, {})
        print(f"{task:<15} {'Baseline':<12} {bl.get('accuracy', 0):<12.4f} {bl.get('f1_macro', 0):<12.4f}")
        if bert_results and task in bert_results:
            bt = bert_results[task]
            print(f"{'':<15} {'BERT':<12} {bt.get('accuracy', 0):<12.4f} {bt.get('f1_macro', 0):<12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Airline Sentiment Analysis Pipeline")
    parser.add_argument("--baseline-only", action="store_true", help="Train only baseline model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of BERT training epochs")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT training")
    args = parser.parse_args()
    
    # Step 1: Prepare data
    print("[STEP 1] Preparing dataset...")
    df = prepare_dataset()
    train_df, val_df, test_df = split_dataset(df)
    
    # Save splits
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    
    # Step 2: Baseline
    print("\n[STEP 2] Training baseline...")
    baseline_results = train_baseline(train_df, val_df, test_df)
    
    # Step 3: BERT (optional)
    bert_results = None
    if not args.baseline_only and not args.skip_bert:
        try:
            print("\n[STEP 3] Training BERT...")
            bert_results, trainer = train_bert(train_df, val_df, test_df, epochs=args.epochs)
        except Exception as e:
            print(f"[WARN] BERT training failed: {e}")
            print("[INFO] Proceeding with baseline results only.")
    
    # Step 4: Report
    print("\n[STEP 4] Generating report...")
    save_report(baseline_results, bert_results)
    
    print("\n[DONE] Pipeline complete!")


if __name__ == "__main__":
    main()
