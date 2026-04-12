.PHONY: setup train train-baseline test api clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -r requirements.txt

train:  ## Run full pipeline (baseline + BERT)
	python scripts/train.py

train-baseline:  ## Run baseline only (fast, no GPU needed)
	python scripts/train.py --baseline-only

train-bert:  ## Run BERT fine-tuning
	python scripts/train.py --skip-bert=false --epochs 5

test:  ## Run tests
	python -m pytest tests/ -v

api:  ## Start FastAPI server
	python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

clean:  ## Remove generated files
	rm -rf data/*.csv models/baseline models/bert reports/*.json __pycache__ src/__pycache__
