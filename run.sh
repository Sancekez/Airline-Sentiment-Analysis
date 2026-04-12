#!/usr/bin/env bash
# Start the API server
set -e

source venv/bin/activate
echo "Starting Airline Sentiment API on http://0.0.0.0:8000"
echo "Swagger docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo ""
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
