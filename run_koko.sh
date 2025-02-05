#!/bin/bash

# Environment setup
#export PYTHONPATH=/app:/app/models
export PYTHONUNBUFFERED=1
export API_HOST="localhost"
export API_PORT=8880
export GRADIO_WATCH=1

# Error handling
set -e

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    exit 1
fi

# Start FastAPI service in background
echo "Starting FastAPI service..."
cd "$(dirname "$0")"
uv run python -m uvicorn api.src.main:app --host "0.0.0.0" --port $API_PORT --log-level debug &
API_PID=$!

# Wait for FastAPI to start
sleep 2

# Check if FastAPI started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "Error: FastAPI failed to start"
    exit 1
fi

# Start Gradio UI
echo "Starting Gradio UI..."
cd ui
uv run python app.py

# Wait for all background processes
wait

