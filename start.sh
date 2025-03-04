#!/bin/bash

# Navigate to the project directory
cd /root/projects_python/florence2_api

# Activate virtual environment (if applicable)
source venv/bin/activate  # Uncomment if using a virtual environment

# Start FastAPI with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 9090 --workers 4
