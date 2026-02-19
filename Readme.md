# Fraud Detection System

A production-ready fraud detection system with automated model training, selection, and real-time inference via REST API.

## Overview

End-to-end fraud detection pipeline using tree-based models (Random Forest, XGBoost, Extra Trees, Decision Tree). Features automated model selection, artifact management, and FastAPI deployment with Docker containerization.

**Key Features:**
- Automated training pipeline (preprocessing → training → model selection)
- Real-time prediction API with input validation
- Reproducible deployments via Docker
- Training-serving consistency through metadata-driven feature ordering
- Multiple model comparison with automatic best-model selection

## Why Tree-Based Models?

Tree models are ideal for tabular fraud data:
- **No feature scaling required** — handle raw structured data directly
- **Interpretable** — feature importance aids business decisions
- **Robust** — handle non-linear relationships and categorical variables
- **Fast inference** — sub-millisecond predictions for real-time compliance

## Dataset

**Loan Prediction Dataset** — Binary classification (fraud/non-fraud)
- Features: Income, Age, Experience, Marital Status, House/Car Ownership, Profession, Location
- Train/Test Split: 70/30 (stratified)
- Preprocessing: Label encoding for categorical features

## System Architecture

Data → Preprocessing (Label Encoding) → Train/Test Split (70/30, stratified) → Train 4 models (RF, XGBoost, ExtraTrees, DT) → Select Best (by accuracy) → Save artifacts (model + encoders + metadata) → FastAPI API → Real-time predictions

## Pipelines

**Training** (`src/fraud/pipeline/train_pipeline.py`): Load data → encode → train all models → select best → save artifacts

**Inference** (`src/fraud/pipeline/predict_pipeline.py`): Load artifacts → validate input → apply encoders → reorder features → predict

Key: Metadata ensures consistent feature ordering between training and inference.

## API

**Prediction Endpoint**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Income": 500000, "Age": 45, ..., "Married_Single": "single"}'
```

Response: `{"fraud_prediction": 0}` (0 = no fraud, 1 = fraudulent)

Auto-generated docs at `http://localhost:8000/docs`

## Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt . && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=src
EXPOSE 8000
CMD ["uvicorn", "fraud.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Run:**
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Key Design Decisions

- **Automated Model Selection**: Trains all 4 models, picks best automatically—no manual bias
- **Artifact Management**: Models, encoders, metadata versioned together for reproducibility
- **Feature Consistency**: Metadata stores feature order, prevents train-serve skew
- **Stateless API**: No state management, scales horizontally

## Future Work

- Model monitoring and drift detection
- Batch prediction endpoint
- Hyperparameter tuning (Optuna)
- Class-weighted models for fraud cost optimization
- Feature importance API endpoint
- Cloud deployment (AWS Fargate, GKE)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Setup
```bash
# Clone repository
git clone <repo-url>
cd fraud-detection

# Create virtual environment
python -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Model
```bash
python src/fraud/pipeline/train_pipeline.py
```
Generates: `artifacts/model.pkl`, `artifacts/encoders.pkl`, `artifacts/metadata.json`

### Run API
```bash
uvicorn fraud.api.app:app --reload
```
Visit: `http://localhost:8000/docs` for interactive API documentation

### Docker
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

---

## Project Structure

```
fraud-detection/
├── src/fraud/
│   ├── api/              # FastAPI application & request schemas
│   ├── models/           # Model training & evaluation logic
│   ├── features/         # Data loading & preprocessing
│   ├── pipeline/         # Training & inference orchestration
│   └── utils/            # Shared utilities
├── data/raw/             # Input CSV data
├── artifacts/            # Trained models & metadata
├── notebooks/            # Exploratory analysis
├── tests/                # Unit tests
├── Dockerfile            # Container specification
├── requirements.txt      # Python dependencies
└── README.md
```


