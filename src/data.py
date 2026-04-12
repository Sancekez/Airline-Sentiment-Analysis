"""Data loading, preprocessing, and automatic labeling."""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from src.config import (
    DATA_DIR, SENTIMENT2ID, CATEGORY_KEYWORDS, CATEGORY2ID,
    CRITICALITY2ID, RANDOM_SEED
)


# ── Loading ────────────────────────────────────────────

def load_twitter_airline_data() -> pd.DataFrame:
    """Load Twitter US Airline Sentiment dataset.
    
    Downloads from GitHub if not cached locally.
    Dataset: ~14,640 tweets about US airlines with sentiment labels.
    """
    csv_path = DATA_DIR / "twitter_airline_sentiment.csv"
    
    if csv_path.exists():
        print(f"[INFO] Loading cached data from {csv_path}")
        return pd.read_csv(csv_path)
    
    print("[INFO] Downloading Twitter Airline Sentiment dataset...")
    url = "https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/AirlineTweets.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception:
        print("[WARN] GitHub download failed, generating synthetic dataset...")
        df = _generate_synthetic_dataset()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved {len(df)} records to {csv_path}")
    return df


def _generate_synthetic_dataset(n: int = 5000) -> pd.DataFrame:
    """Generate a synthetic airline review dataset for development."""
    np.random.seed(RANDOM_SEED)
    
    negative_templates = [
        "My flight was delayed by {hours} hours and nobody told us anything.",
        "Lost my luggage again! This is the {n}th time with this airline.",
        "The worst customer service I've ever experienced. Rude staff at the counter.",
        "Seat was broken, no entertainment system, terrible food. Never again.",
        "Check-in process took over 2 hours. Absolutely unacceptable.",
        "They cancelled my flight without any notification. Had to rebook at twice the price.",
        "Baggage was damaged and they refused to compensate.",
        "Called customer support 5 times, kept getting transferred. No resolution.",
        "The boarding process was chaotic. No organization at the gate.",
        "Flight attendants were dismissive and unhelpful throughout the flight.",
        "Paid for premium seat but got a regular economy seat. No refund offered.",
        "My connecting flight was missed due to their delay. No hotel provided.",
        "The online booking system is terrible. Charged twice for one ticket.",
        "Waited 45 minutes at baggage claim. Several bags came damaged.",
        "The cabin was dirty and the seats were uncomfortable for a 6-hour flight.",
    ]
    
    positive_templates = [
        "Great flight experience! Crew was friendly and professional.",
        "Smooth check-in, on-time departure, comfortable seats. Will fly again!",
        "Best airline food I've had. The entertainment selection was excellent.",
        "Customer service resolved my issue quickly and gave me a voucher.",
        "Boarding was efficient, flight was on time, luggage arrived fast.",
        "Upgraded to business class! Amazing service and comfortable seats.",
        "The crew went above and beyond to help with my special meal request.",
        "Easy online booking, great price, and the flight was perfect.",
        "Very comfortable legroom in economy. Will definitely choose them again.",
        "Quick response from customer service. My refund was processed in 2 days.",
    ]
    
    neutral_templates = [
        "Flight was okay. Nothing special but nothing terrible either.",
        "Average experience. The flight was on time but food was mediocre.",
        "Standard airline service. Check-in was normal, flight was uneventful.",
        "The plane was a bit old but everything worked fine.",
        "Decent flight for the price. Would consider flying with them again.",
        "Normal boarding process. Seat was standard economy size.",
        "Flight arrived 10 minutes late. Not a big deal.",
        "Basic service, no complaints but no standout moments either.",
    ]
    
    records = []
    sentiments = ["negative"] * int(n * 0.45) + ["positive"] * int(n * 0.30) + ["neutral"] * int(n * 0.25)
    np.random.shuffle(sentiments)
    
    airlines = ["United", "Delta", "American", "Southwest", "JetBlue", "Spirit"]
    
    for i, sent in enumerate(sentiments):
        if sent == "negative":
            text = np.random.choice(negative_templates)
            text = text.format(hours=np.random.randint(2, 12), n=np.random.randint(2, 5))
        elif sent == "positive":
            text = np.random.choice(positive_templates)
        else:
            text = np.random.choice(neutral_templates)
        
        records.append({
            "tweet_id": 100000 + i,
            "airline_sentiment": sent,
            "airline": np.random.choice(airlines),
            "text": text,
        })
    
    return pd.DataFrame(records)


# ── Preprocessing ──────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize text for NLP."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"http\S+|www\S+", "", text) # remove URLs
    text = re.sub(r"#(\w+)", r"\1", text)      # remove # but keep word
    text = re.sub(r"[^a-zA-Z0-9\s!?.,']", "", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()   # normalize whitespace
    
    # Handle negation: join "no/not/never" with next word so TF-IDF treats as one token
    text = re.sub(r"\b(no|not|never|dont|doesn't|didn't|wasn't|weren't|isn't|aren't|haven't|hasn't|wouldn't|couldn't)\s+(\w+)", r"\1_\2", text)
    
    return text


def assign_category(text: str) -> str:
    """Rule-based category assignment from text content."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score
    
    if not scores:
        return "other"
    return max(scores, key=scores.get)


def assign_criticality(sentiment: str, text: str) -> str:
    """Assign criticality level based on sentiment and text signals."""
    text_lower = text.lower()
    
    # High criticality: strong negative sentiment + urgent keywords
    urgent_words = ["never", "worst", "terrible", "unacceptable", "horrible",
                    "lawsuit", "sue", "complaint", "compensation", "refund",
                    "dangerous", "unsafe", "emergency", "discriminat"]
    
    if sentiment == "negative":
        urgent_count = sum(1 for w in urgent_words if w in text_lower)
        if urgent_count >= 2 or any(w in text_lower for w in ["lawsuit", "sue", "dangerous", "unsafe"]):
            return "high"
        elif urgent_count >= 1:
            return "medium"
        else:
            return "medium"
    elif sentiment == "neutral":
        return "low"
    else:  # positive
        return "low"


# ── Pipeline ───────────────────────────────────────────

def prepare_dataset() -> pd.DataFrame:
    """Full data preparation pipeline: load → clean → label → split."""
    df = load_twitter_airline_data()
    
    # Identify text column
    text_col = None
    for col in ["text", "Text", "review", "content", "tweet_text"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[-1]  # fallback to last column
    
    # Identify sentiment column
    sent_col = None
    for col in ["airline_sentiment", "sentiment", "Sentiment", "label"]:
        if col in df.columns:
            sent_col = col
            break
    
    # Build clean dataframe
    result = pd.DataFrame()
    result["text_raw"] = df[text_col].astype(str)
    result["text"] = result["text_raw"].apply(clean_text)
    
    # Sentiment
    if sent_col:
        result["sentiment"] = df[sent_col].str.lower().str.strip()
        result["sentiment"] = result["sentiment"].map(
            lambda x: x if x in SENTIMENT2ID else "neutral"
        )
    else:
        result["sentiment"] = "neutral"
    
    result["sentiment_id"] = result["sentiment"].map(SENTIMENT2ID)
    
    # Category (auto-labeled)
    result["category"] = result["text"].apply(assign_category)
    result["category_id"] = result["category"].map(CATEGORY2ID)
    
    # Criticality (auto-labeled)
    result["criticality"] = result.apply(
        lambda row: assign_criticality(row["sentiment"], row["text"]), axis=1
    )
    result["criticality_id"] = result["criticality"].map(CRITICALITY2ID)
    
    # Drop empty texts
    result = result[result["text"].str.len() > 5].reset_index(drop=True)
    
    print(f"[INFO] Dataset prepared: {len(result)} samples")
    print(f"  Sentiment distribution: {result['sentiment'].value_counts().to_dict()}")
    print(f"  Category distribution: {result['category'].value_counts().to_dict()}")
    print(f"  Criticality distribution: {result['criticality'].value_counts().to_dict()}")
    
    return result


def split_dataset(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15) -> Tuple:
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED, stratify=df["sentiment_id"]
    )
    
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=RANDOM_SEED, stratify=train_df["sentiment_id"]
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"[INFO] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = prepare_dataset()
    train, val, test = split_dataset(df)
    
    # Save splits
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    val.to_csv(DATA_DIR / "val.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print("[INFO] Saved train/val/test splits to data/")
