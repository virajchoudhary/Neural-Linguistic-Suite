# Neural Linguistic Suite
**A serverless NLP API featuring dynamic model routing and zero-GPU overhead.**

https://github.com/user-attachments/assets/9f0a9154-0a2b-4d7c-a2b4-c393a9db3a38

## Key Features
* **Serverless Architecture:** Eliminates local GPU constraints by offloading heavy inference to Hugging Face’s infrastructure.
* **Dynamic Model Routing:** Abstracts the complexity of Transformer models (MarianMT, DistilBART) behind a unified FastAPI interface.
* **Robust Error Handling:** Implements specific exception handling for cloud gateway timeouts and API rate limits.

## Engineering Challenge: API Deprecation
This project demonstrates real-world debugging of third-party infrastructure. 

During the development cycle, Hugging Face deprecated their legacy inference domain (api-inference.huggingface.co), which resulted in critical 500 and 504 gateway timeouts. I diagnosed the gateway failure through server tracebacks and updated the routing logic to target the modern router.huggingface.co/hf-inference endpoint, ensuring the application remained stable without requiring local code changes.

## Tech Stack
* **Backend:** FastAPI (Python)
* **Models:** Helsinki-NLP MarianMT, DistilBART CNN 12-6
* **Infrastructure:** Render, Hugging Face Inference API

## API Endpoints

### 1. Multilingual Translation
`POST /translate`
```json
{
  "text": "The rapid advancement of AI is reshaping the economy.",
  "target_lang": "es"
}
```

### 2. Abstractive Summarization
`POST /summarize`
```json
{
  "text": "[Extensive article text here...]"
}
```

## Quick Start

### Local Setup
```bash
git clone https://github.com/virajchoudhary/Neural-Linguistic-Suite.git
cd neural-linguistic-suite

python -m venv .venv
# On Windows use: .venv\Scripts\activate
# On Mac/Linux use: source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration
Create a .env file in the root directory:
```env
HF_TOKEN=hf_your_actual_token_here
```

### Run the Server
```bash
uvicorn backend.main:app --reload
```
