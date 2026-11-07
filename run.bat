@echo off
echo Starting Crop Yield Prediction API server on http://localhost:8000 ...
py -3.14 -m uvicorn app:app --reload --port 8000
pause
