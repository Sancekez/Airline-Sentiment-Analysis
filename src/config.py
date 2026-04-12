"""Project configuration and constants."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Dataset ────────────────────────────────────────────
DATASET_URL = "https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/AirlineReviews.csv"
TWITTER_DATASET = "twitter_airline_sentiment"  # loaded via script

# ── Labels ─────────────────────────────────────────────
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
CATEGORY_LABELS = ["baggage", "booking", "delay", "in-flight", "check-in", "customer_service", "other"]
CRITICALITY_LABELS = ["low", "medium", "high"]

SENTIMENT2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
ID2SENTIMENT = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

CATEGORY2ID = {l: i for i, l in enumerate(CATEGORY_LABELS)}
ID2CATEGORY = {i: l for i, l in enumerate(CATEGORY_LABELS)}

CRITICALITY2ID = {l: i for i, l in enumerate(CRITICALITY_LABELS)}
ID2CRITICALITY = {i: l for i, l in enumerate(CRITICALITY_LABELS)}

# ── Model ──────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# ── Category keywords for rule-based labeling ──────────
CATEGORY_KEYWORDS = {
    "baggage": ["bag", "baggage", "luggage", "suitcase", "lost bag", "damaged bag", "carry-on", "checked bag"],
    "booking": ["book", "booking", "reservation", "ticket", "refund", "cancel", "reschedule", "price", "fare"],
    "delay": ["delay", "delayed", "late", "waiting", "hour late", "cancelled flight", "cancel", "on time"],
    "in-flight": ["seat", "food", "meal", "entertainment", "legroom", "cabin", "crew", "comfort", "wifi", "blanket"],
    "check-in": ["check-in", "checkin", "check in", "boarding", "gate", "queue", "line", "kiosk", "counter"],
    "customer_service": ["service", "staff", "rude", "helpful", "agent", "representative", "call", "phone", "support"],
}
