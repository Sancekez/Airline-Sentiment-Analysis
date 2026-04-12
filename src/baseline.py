"""Baseline models: TF-IDF + LogisticRegression / SVM for all 3 tasks."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR, SENTIMENT_LABELS, CATEGORY_LABELS, CRITICALITY_LABELS
)


class BaselineModel:
    """TF-IDF + LogisticRegression baseline for all 3 classification tasks."""
    
    def __init__(self):
        self.pipelines = {}
        self.tasks = {
            "sentiment": {"target": "sentiment_id", "labels": SENTIMENT_LABELS},
            "category": {"target": "category_id", "labels": CATEGORY_LABELS},
            "criticality": {"target": "criticality_id", "labels": CRITICALITY_LABELS},
        }
    
    def train(self, train_df, val_df=None):
        """Train baseline models for all tasks."""
        results = {}
        
        for task_name, task_cfg in self.tasks.items():
            print(f"\n{'='*50}")
            print(f"Training baseline: {task_name}")
            print(f"{'='*50}")
            
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                )),
            ])
            
            X_train = train_df["text"].values
            y_train = train_df[task_cfg["target"]].values
            
            pipeline.fit(X_train, y_train)
            self.pipelines[task_name] = pipeline
            
            # Train metrics
            y_pred_train = pipeline.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_train)
            print(f"Train Accuracy: {train_acc:.4f}")
            
            # Validation metrics
            if val_df is not None:
                X_val = val_df["text"].values
                y_val = val_df[task_cfg["target"]].values
                y_pred_val = pipeline.predict(X_val)
                
                val_acc = accuracy_score(y_val, y_pred_val)
                val_f1 = f1_score(y_val, y_pred_val, average="macro")
                
                print(f"Val Accuracy: {val_acc:.4f}")
                print(f"Val F1-macro: {val_f1:.4f}")
                print(f"\nClassification Report ({task_name}):")
                print(classification_report(
                    y_val, y_pred_val,
                    target_names=task_cfg["labels"],
                    digits=4
                ))
                
                results[task_name] = {
                    "accuracy": val_acc,
                    "f1_macro": val_f1,
                }
        
        return results
    
    def predict(self, texts):
        """Predict all tasks for a list of texts."""
        predictions = {}
        for task_name, pipeline in self.pipelines.items():
            preds = pipeline.predict(texts)
            labels = self.tasks[task_name]["labels"]
            predictions[task_name] = [labels[p] for p in preds]
        return predictions
    
    def predict_single(self, text: str) -> dict:
        """Predict all tasks for a single text."""
        result = {}
        for task_name, pipeline in self.pipelines.items():
            pred_id = pipeline.predict([text])[0]
            proba = pipeline.predict_proba([text])[0] if hasattr(pipeline["clf"], "predict_proba") else None
            labels = self.tasks[task_name]["labels"]
            result[task_name] = {
                "label": labels[pred_id],
                "confidence": float(max(proba)) if proba is not None else None,
                "probabilities": {labels[i]: float(p) for i, p in enumerate(proba)} if proba is not None else None,
            }
        return result
    
    def save(self, path: Path = None):
        """Save all pipelines."""
        path = path or MODELS_DIR / "baseline"
        path.mkdir(parents=True, exist_ok=True)
        for task_name, pipeline in self.pipelines.items():
            joblib.dump(pipeline, path / f"{task_name}_pipeline.joblib")
        print(f"[INFO] Baseline models saved to {path}")
    
    def load(self, path: Path = None):
        """Load all pipelines."""
        path = path or MODELS_DIR / "baseline"
        for task_name in self.tasks:
            fpath = path / f"{task_name}_pipeline.joblib"
            if fpath.exists():
                self.pipelines[task_name] = joblib.load(fpath)
        print(f"[INFO] Baseline models loaded from {path}")


def evaluate_on_test(model: BaselineModel, test_df):
    """Evaluate baseline on test set and print final metrics."""
    print(f"\n{'='*60}")
    print("BASELINE — TEST SET EVALUATION")
    print(f"{'='*60}")
    
    results = {}
    for task_name, task_cfg in model.tasks.items():
        X_test = test_df["text"].values
        y_test = test_df[task_cfg["target"]].values
        y_pred = model.pipelines[task_name].predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        print(f"\n--- {task_name.upper()} ---")
        print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=task_cfg["labels"],
            digits=4
        ))
        results[task_name] = {"accuracy": acc, "f1_macro": f1}
    
    return results
