"""FastAPI REST API for airline sentiment prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from src.config import MODELS_DIR

app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="NLP-система анализа тональности и классификации обращений авиапассажиров",
    version="1.0.0",
)

# Global model reference
_model = None
_model_type = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, 
                      examples=["My flight was delayed 3 hours and nobody helped"])


class TaskResult(BaseModel):
    label: str
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None


class PredictResponse(BaseModel):
    sentiment: TaskResult
    category: TaskResult
    criticality: TaskResult
    model_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]


def load_model():
    """Load the best available model."""
    global _model, _model_type
    
    bert_path = MODELS_DIR / "bert" / "model.pt"
    baseline_path = MODELS_DIR / "baseline" / "sentiment_pipeline.joblib"
    
    if bert_path.exists():
        try:
            from src.bert_model import BERTTrainer
            trainer = BERTTrainer()
            trainer.load()
            _model = trainer
            _model_type = "bert"
            print("[INFO] BERT model loaded")
            return
        except Exception as e:
            print(f"[WARN] BERT loading failed: {e}")
    
    if baseline_path.exists():
        from src.baseline import BaselineModel
        model = BaselineModel()
        model.load()
        _model = model
        _model_type = "baseline"
        print("[INFO] Baseline model loaded")
        return
    
    print("[WARN] No trained model found. Train first with: python scripts/train.py")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_type=_model_type
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    
    from src.data import clean_text
    cleaned = clean_text(request.text)
    
    result = _model.predict_single(cleaned)
    
    return PredictResponse(
        sentiment=TaskResult(**result["sentiment"]),
        category=TaskResult(**result["category"]),
        criticality=TaskResult(**result["criticality"]),
        model_type=_model_type,
    )


@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    from src.data import clean_text
    results = []
    for text in texts[:50]:  # limit to 50
        cleaned = clean_text(text)
        pred = _model.predict_single(cleaned)
        pred["text"] = text
        results.append(pred)
    
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
