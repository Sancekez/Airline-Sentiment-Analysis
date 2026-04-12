"""DistilBERT fine-tuning for multi-task airline sentiment analysis."""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path

from src.config import (
    BERT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SENTIMENT_LABELS, CATEGORY_LABELS, CRITICALITY_LABELS, MODELS_DIR
)


# ── Dataset ────────────────────────────────────────────

class AirlineDataset(Dataset):
    """PyTorch dataset for airline reviews with multi-task labels."""
    
    def __init__(self, texts, sentiment_ids, category_ids, criticality_ids, tokenizer, max_length):
        self.texts = texts
        self.sentiment_ids = sentiment_ids
        self.category_ids = category_ids
        self.criticality_ids = criticality_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "sentiment": torch.tensor(self.sentiment_ids[idx], dtype=torch.long),
            "category": torch.tensor(self.category_ids[idx], dtype=torch.long),
            "criticality": torch.tensor(self.criticality_ids[idx], dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────

class MultiTaskBERT(nn.Module):
    """DistilBERT with 3 classification heads: sentiment, category, criticality."""
    
    def __init__(self, n_sentiment=3, n_category=7, n_criticality=3, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768 for distilbert
        
        self.dropout = nn.Dropout(dropout)
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_sentiment)
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_category)
        )
        
        self.criticality_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_criticality)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        
        return {
            "sentiment": self.sentiment_head(pooled),
            "category": self.category_head(pooled),
            "criticality": self.criticality_head(pooled),
        }


# ── Training ───────────────────────────────────────────

class BERTTrainer:
    """Training and evaluation loop for MultiTaskBERT."""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = MultiTaskBERT().to(self.device)
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    
    def _make_loader(self, df, shuffle=False):
        dataset = AirlineDataset(
            texts=df["text"].tolist(),
            sentiment_ids=df["sentiment_id"].tolist(),
            category_ids=df["category_id"].tolist(),
            criticality_ids=df["criticality_id"].tolist(),
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH,
        )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    
    def train(self, train_df, val_df, epochs=EPOCHS, lr=LEARNING_RATE):
        """Full training loop with validation."""
        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader = self._make_loader(val_df)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0
        
        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                loss = (
                    criterion(outputs["sentiment"], batch["sentiment"].to(self.device)) +
                    criterion(outputs["category"], batch["category"].to(self.device)) * 0.5 +
                    criterion(outputs["criticality"], batch["criticality"].to(self.device)) * 0.3
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)
            
            # ── Validate ──
            val_metrics, avg_val_loss = self.evaluate(val_df, val_loader)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_metrics"].append(val_metrics)
            
            sent_f1 = val_metrics["sentiment"]["f1_macro"]
            print(f"\nEpoch {epoch+1}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} | "
                  f"sentiment_f1={sent_f1:.4f}")
            
            # Save best model
            if sent_f1 > best_f1:
                best_f1 = sent_f1
                self.save()
                print(f"  → New best model saved (F1={best_f1:.4f})")
        
        return self.history
    
    def evaluate(self, df=None, loader=None):
        """Evaluate on a dataset, return per-task metrics."""
        if loader is None:
            loader = self._make_loader(df)
        
        self.model.eval()
        all_preds = {"sentiment": [], "category": [], "criticality": []}
        all_labels = {"sentiment": [], "category": [], "criticality": []}
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                loss = (
                    criterion(outputs["sentiment"], batch["sentiment"].to(self.device)) +
                    criterion(outputs["category"], batch["category"].to(self.device)) * 0.5 +
                    criterion(outputs["criticality"], batch["criticality"].to(self.device)) * 0.3
                )
                total_loss += loss.item()
                
                for task in ["sentiment", "category", "criticality"]:
                    preds = outputs[task].argmax(dim=1).cpu().numpy()
                    labels = batch[task].numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)
        
        metrics = {}
        label_names = {
            "sentiment": SENTIMENT_LABELS,
            "category": CATEGORY_LABELS,
            "criticality": CRITICALITY_LABELS,
        }
        
        for task in ["sentiment", "category", "criticality"]:
            acc = accuracy_score(all_labels[task], all_preds[task])
            f1 = f1_score(all_labels[task], all_preds[task], average="macro")
            metrics[task] = {"accuracy": acc, "f1_macro": f1}
        
        avg_loss = total_loss / len(loader)
        return metrics, avg_loss
    
    def predict_single(self, text: str) -> dict:
        """Predict all tasks for a single text string."""
        self.model.eval()
        
        encoding = self.tokenizer(
            text, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.model(input_ids, attention_mask)
        
        result = {}
        label_names = {
            "sentiment": SENTIMENT_LABELS,
            "category": CATEGORY_LABELS,
            "criticality": CRITICALITY_LABELS,
        }
        
        for task, labels in label_names.items():
            logits = outputs[task][0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            pred_id = int(probs.argmax())
            result[task] = {
                "label": labels[pred_id],
                "confidence": float(probs[pred_id]),
                "probabilities": {labels[i]: float(p) for i, p in enumerate(probs)},
            }
        
        return result
    
    def save(self, path: Path = None):
        """Save model, tokenizer, and training history."""
        path = path or MODELS_DIR / "bert"
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(str(path / "tokenizer"))
        
        with open(path / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"[INFO] BERT model saved to {path}")
    
    def load(self, path: Path = None):
        """Load saved model."""
        path = path or MODELS_DIR / "bert"
        
        state_dict = torch.load(path / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        tokenizer_path = path / "tokenizer"
        if tokenizer_path.exists():
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
        
        print(f"[INFO] BERT model loaded from {path}")


def print_test_report(trainer: BERTTrainer, test_df):
    """Print full classification report on test set."""
    loader = trainer._make_loader(test_df)
    
    trainer.model.eval()
    all_preds = {"sentiment": [], "category": [], "criticality": []}
    all_labels = {"sentiment": [], "category": [], "criticality": []}
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            outputs = trainer.model(input_ids, attention_mask)
            
            for task in all_preds:
                all_preds[task].extend(outputs[task].argmax(dim=1).cpu().numpy())
                all_labels[task].extend(batch[task].numpy())
    
    label_names = {
        "sentiment": SENTIMENT_LABELS,
        "category": CATEGORY_LABELS,
        "criticality": CRITICALITY_LABELS,
    }
    
    print(f"\n{'='*60}")
    print("BERT — TEST SET EVALUATION")
    print(f"{'='*60}")
    
    for task, labels in label_names.items():
        print(f"\n--- {task.upper()} ---")
        print(classification_report(all_labels[task], all_preds[task], target_names=labels, digits=4))
