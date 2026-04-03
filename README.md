# Neural Linguistic Suite

A lightweight, cloud-first API for multilingual translation and abstractive text summarization.

This suite acts as a streamlined bridge to Hugging Face's infrastructure. It dynamically routes requests through production-grade Transformer models to process natural language instantly, delivering high-quality NLP features with zero local GPU overhead.

## Live Demo
https://github.com/user-attachments/assets/9f0a9154-0a2b-4d7c-a2b4-c393a9db3a38

## Architecture & Stack
The system utilizes an autoregressive Encoder-Decoder processing pipeline, bypassing local resource constraints via serverless inference.

* **Translation:** Helsinki-NLP MarianMT 
* **Summarization:** DistilBART CNN 12-6 
* **Backend:** FastAPI
* **Infrastructure:** Render, Hugging Face Inference API

<img width="1195" height="869" alt="Image" src="https://github.com/user-attachments/assets/0aee4b9c-a503-450e-a411-b4123856b8c5" />

## Quick Start

### 1. Local Setup
```bash
git clone https://github.com/virajchoudhary/Neural-Linguistic-Suite.git
cd neural-linguistic-suite

python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory for local development:
```env
HF_TOKEN=hf_your_actual_token_here
```

### 3. Run the Server
```bash
uvicorn backend.main:app --reload
```

---

**Developer Notes: Infrastructure & API Routing**

During development, Hugging Face deprecated their legacy inference domain (`api-inference.huggingface.co`), which resulted in 500 and 504 gateway timeouts. 

To maintain stability, the backend routing logic was patched to target the modern `router.huggingface.co/hf-inference` endpoint. When deploying this application to production environments (such as Render), ensure the `HF_TOKEN` environment variable is explicitly configured in the deployment dashboard to authenticate the router requests.
