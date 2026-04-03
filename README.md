# Neural Linguistic Suite

A full-stack, serverless NLP platform built with React and FastAPI. This project integrates directly with Hugging Face's cloud infrastructure to provide high-speed, multilingual translation and text summarization using state-of-the-art Transformer models.

## Architecture

- **Frontend**: React, Vite, TailwindCSS
- **Backend**: FastAPI, Python
- **Cloud Inference**: Hugging Face API (Serverless)
- **Models**:
  - **Translation**: Helsinki-NLP/opus-mt (English, Spanish, Hindi)
  - **Summarization**: sshleifer/distilbart-cnn-12-6

## Engineering Highlights

- **Serverless Integration**: Zero local GPU requirement. Offloads heavy tensor computation to cloud inference endpoints, reducing repository size and local compute overhead.
- **Cold-Start Handling**: Gracefully catches Hugging Face API timeouts and provides precise, human-readable wait times to the frontend UI.
- **Auto-Script Detection**: Automatically routes reverse translations (Hindi to English vs. Spanish to English) via Unicode block detection (Devanagari) in the FastAPI backend.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/virajchoudhary/Neural-Linguistic-Suite.git
cd Neural-Linguistic-Suite
```

### 2. Setup the Backend

```bash
cd backend
# Create a .env file from the example
cp .env.example .env
# Note: Add your Hugging Face API key to the .env file
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Setup the Frontend (in a new terminal)

```bash
cd frontend
npm install
npm run dev
```

## Live Demo
