# Fraud Detection System

A production-ready fraud detection system built with tree-based machine learning models, automated training pipelines, and real-time inference via REST API.

---

## Overview

This project implements an **end-to-end fraud detection pipeline** that processes structured tabular data, automatically trains and selects the best-performing model, and exposes predictions through a FastAPI endpoint. The system is containerized with Docker for reproducible deployments across environments.

**Key Business Value:**
- Real-time fraud risk assessment
- Automated model retraining and selection
- Reproducible deployments with versioned artifacts
- API-based integration for downstream systems

---

## Problem Statement

Fraud detection in financial systems requires:
- **Speed**: Decisions must be made in milliseconds at scale
- **Robustness**: Models must generalize across diverse customer profiles
- **Interpretability**: Tree-based models provide feature importance insights
- **Reliability**: System must consistently identify risky loans without false negatives

This system addresses these needs with a production-grade pipeline that eliminates manual model selection and ensures feature consistency between training and inference.

---

## Key Features

### Engineering Excellence
- **End-to-End Pipeline**: Automated data loading → preprocessing → training → model selection → artifact storage
- **Reproducible Deployments**: Docker containerization ensures consistency across dev, staging, and production
- **Artifact Management**: Versioned models, encoders, and metadata for full reproducibility
- **Feature Consistency**: Metadata-driven feature ordering prevents training-serving skew
- **Automated Model Selection**: Trains multiple algorithms and selects the best performer
- **REST API**: FastAPI endpoint for real-time predictions with schema validation

---

## Machine Learning Approach

### Why Tree-Based Models?
- **Structured Data**: Tree models excel on tabular features without feature scaling
- **Interpretability**: Feature importance outputs aid business decisions and regulatory requirements
- **Robustness**: Handles non-linear relationships and categorical variables naturally
- **Speed**: Fast inference—ideal for real-time fraud checks

### Model Comparison
The system trains multiple candidates:
| Model | Strengths |
|-------|-----------|
| **Random Forest** | Ensemble stability, handles high-dimensional data well |
| **Extra Trees** | Randomized splits reduce variance, faster training |
| **XGBoost** | Industry-standard for tabular data, optimized loss functions |
| **Decision Tree** | Baseline for interpretability and feature importance |

Best model is **selected automatically** based on validation accuracy. This prevents manual bias and ensures optimal performance.

### Tradeoffs
- **Performance vs. Interpretability**: XGBoost offers high accuracy; Decision Tree offers clear decision rules
- **Speed vs. Accuracy**: All models achieve sub-millisecond inference—no production bottleneck
- **Complexity vs. Maintenance**: Automated selection reduces manual tuning overhead

---

## Dataset

The system uses the **Loan Prediction Dataset** with the following structure:
- **Features**: Income, Age, Experience, Marital Status, House/Car Ownership, Profession, Location, Employment History
- **Target**: Risk Flag (binary: fraud/non-fraud)
- **Split**: 70% training, 30% validation (stratified for class balance)
- **Preprocessing**: Label encoding for categorical features, no scaling required (tree models)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw CSV Data                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Data Loading & Validation                      │
│           (pandas, missing value handling)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Train/Test Split (Stratified)                     │
│              70/30 split, random_state=42                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Feature Encoding (Categorical → Numeric)            │
│    LabelEncoder for all object-type columns                 │
│    Encoders saved for consistent inference                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
   ┌──────────┐              ┌──────────┐
   │ X_train  │              │ X_test   │
   │ y_train  │              │ y_test   │
   └────┬─────┘              └────┬─────┘
        │                         │
        ▼                         │
┌──────────────────────────┐     │
│    Train 4 Models        │     │
│ • Random Forest          │     │
│ • Extra Trees            │     │
│ • XGBoost                │     │
│ • Decision Tree          │     │
└────┬─────────────────────┘     │
     │                           │
     │    ┌─────────────────────┘
     │    │ Evaluate each model
     │    │ on validation set
     │    │
     ▼    ▼
┌──────────────────────────┐
│    Select Best Model     │
│  (Highest Accuracy)      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Save Artifacts                   │
│  • model.pkl (trained classifier)        │
│  • encoders.pkl (label encoders)         │
│  • metadata.json (feature order)         │
└────┬─────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│       FastAPI Service                    │
│  - Load artifacts                        │
│  - Accept prediction requests            │
│  - Preprocess input (encode features)    │
│  - Run inference                         │
│  - Return risk prediction                │
└──────────────────────────────────────────┘
```

---

## Training and Inference Pipelines

### Training Pipeline (`src/fraud/pipeline/train_pipeline.py`)
```
1. Load raw CSV data
2. Split into train/test with stratification
3. Encode categorical features (fit on train, apply to test)
4. Save feature order before training (critical for reproducibility)
5. Train all 4 candidate models
6. Evaluate on validation set
7. Select best model by accuracy
8. Persist artifacts: model + encoders + metadata
```

**Why this design?**
- Feature order saved before training prevents accidental reordering in inference
- Encoders fit only on training data prevents data leakage
- Metadata enables the predict pipeline to use identical preprocessing

### Inference Pipeline (`src/fraud/pipeline/predict_pipeline.py`)
```
1. Load model, encoders, and metadata from artifacts
2. Receive prediction request (JSON payload)
3. Validate required columns
4. Apply saved encoders to input features
5. Reorder features to match training order
6. Run model.predict()
7. Return fraud/non-fraud prediction
```

**Why this design?**
- Stateless API—no data dependency or state management
- Metadata-driven ensures training-serving consistency
- Validation catches data quality issues early

---

## API and Deployment

### Endpoint Specification

**Base URL**: `http://localhost:8000`

**Health Check**
```bash
curl http://localhost:8000/
```

**Fraud Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Income": 500000,
    "Age": 45,
    "Experience": 10,
    "Married_Single": "single",
    "House_Ownership": "rented",
    "Car_Ownership": "no",
    "Profession": "Engineer",
    "CITY": "Mumbai",
    "STATE": "Maharashtra",
    "CURRENT_JOB_YRS": 5,
    "CURRENT_HOUSE_YRS": 3
  }'
```

**Response**
```json
{
  "fraud_prediction": 0
}
```
(0 = No fraud, 1 = Fraudulent)

### Why FastAPI?
- **Type Safety**: Automatic validation via Pydantic schemas
- **Performance**: Async support for high-throughput inference
- **Documentation**: Auto-generated OpenAPI docs at `/docs`
- **Deployment**: ASGI-compliant for production servers (Gunicorn, hypercorn)

---

## Docker and Reproducibility

### Multi-stage Build Strategy
```dockerfile
FROM python:3.10-slim
```
- Lightweight base image
- Minimal attack surface for production
- Compatible with most cloud runtimes

### Key Design Decisions
1. **Requirements cached layer**: Dependencies install before code copy → faster local rebuilds
2. **PYTHONPATH configuration**: Ensures `fraud` package discoverable at runtime
3. **Port exposure**: Explicit 8000 declaration for clear port binding
4. **No extra dependencies**: Only essentials (FastAPI, pandas, scikit-learn)

**Build and Run Locally**
```bash
docker build -t fraud-detection:latest .
docker run -p 8000:8000 fraud-detection:latest
```

---

## Results and Insights

### Model Selection
The automated pipeline evaluates all 4 candidates and selects the best by validation accuracy. This ensures:
- No manual bias in model choice
- Reproducibility across runs
- Easy to extend with new algorithms

### Production Robustness
- **Feature validation**: Inference rejects missing categorical values
- **Encoding consistency**: Metadata ensures identical preprocessing in training and serving
- **Stratified split**: Maintains class distribution → realistic performance estimates
- **Binary output**: Simple fraud/non-fraud decision for interpretation

### Why Accuracy-Based Selection?
- Fraud detection prioritizes recall (catching all frauds) and precision (minimizing false alarms)
- Consider extending to F1-score or custom loss functions for cost-sensitive scenarios

---

## Limitations and Future Work

### Current Limitations
1. **Threshold selection**: Binary classification at 0.5 probability cutoff—may not be optimal for fraud costs
2. **Model interpretability**: Feature importance available but not exposed via API
3. **Monitoring**: No runtime performance tracking or model drift detection
4. **Imbalanced data**: No explicit handling of class imbalance (SMOTE, class weights)
5. **Hyperparameter tuning**: Models use defaults—grid search could improve performance

### Recommended Improvements
- **Cloud Deployment**: Containerized system ready for AWS Fargate, GKE, or AzureML
- **Performance Monitoring**: Add logging for predictions, latency, and model performance tracking
- **Model Versioning**: Implement model registry for A/B testing and rollback
- **API Enhancements**: 
  - Batch prediction endpoint for bulk fraud checks
  - Feature importance endpoint for business insights
  - Prediction confidence scores
- **Advanced Tuning**:
  - Hyperparameter optimization (Optuna, Ray Tune)
  - Class weight balancing for fraud cost optimization
  - Cost-sensitive scoring (false positive vs. false negative costs)
- **Production Safeguards**:
  - Automated retraining pipeline on schedule
  - Data quality checks and anomaly detection
  - Prediction explainability (SHAP values)

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


