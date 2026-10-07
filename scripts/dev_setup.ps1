# BhashaSense Dev Setup for Windows PowerShell

Write-Host "=== Setting up BhashaSense Development Environment ===" -ForegroundColor Cyan

# 1. Backend Setup
Write-Host "`n[1/2] Configuring Python Backend..." -ForegroundColor Yellow
Set-Location -Path "$PSScriptRoot\..\backend"

if (-Not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment in backend/.venv..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Installing backend requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Frontend Setup
Write-Host "`n[2/2] Configuring React Frontend..." -ForegroundColor Yellow
Set-Location -Path "$PSScriptRoot\..\frontend"

Write-Host "Installing npm packages..."
npm install

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "To run backend: cd backend && .venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"
Write-Host "To run frontend: cd frontend && npm run dev"
