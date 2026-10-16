# Hybrid-Guard

Hybrid-Guard is a phishing URL risk analyzer that combines machine learning predictions with external checks (DNS, SSL, domain age, and HTTP/content signals).

## Main Features

- Multi-model support:
  - RF (Random Forest)
  - DT (Decision Tree)
  - DL (Deep Learning: char n-gram + neural network)
  - LR (Logistic Regression with StandardScaler)
  - SVM (Support Vector Machine: Linear/RBF with probability calibration)
- Hybrid risk scoring pipeline:
  - Base model probability
  - Policy/risk adjustment
  - External enrichment checks
- Web UI + REST API for interactive analysis + QR Code / Quishing Scanner
- Benchmark script for model evaluation (accuracy, precision, recall, F1, TP/TN/FP/FN)

## Project Structure

- `url_ml/`: feature engineering and DL model components
- `webapp/`: FastAPI backend + Jinja UI
- `train_url_model.py`: train RF/DT/DL/LR/SVM model
- `train_dl_model.py`: convenience entrypoint for DL mode
- `debug_url_model.py`: inspect one or multiple URLs with explainable output
- `benchmark_url_detector.py`: benchmark model or hybrid mode
- `requirements.txt`: Python dependencies

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Train RF:

```bash
python train_url_model.py --data Dataset.csv --model rf --output-dir artifacts/rf_final
```

Train DT:

```bash
python train_url_model.py --data Dataset.csv --model dt --max-depth 10 --output-dir artifacts/dt_final
```

Train DL:

```bash
python train_url_model.py --data Dataset.csv --model dl --output-dir artifacts/dl_final
```

Train LR (Logistic Regression):

```bash
# Cách 1: Chạy script bash riêng
bash train_lr_model.sh
# hoặc: ./train_lr_model.sh Dataset.csv artifacts/lr_final

# Cách 2: Chạy script Python riêng
python train_lr_model.py --data Dataset.csv

# Cách 3: Chạy qua train_url_model
python train_url_model.py --data Dataset.csv --model lr --output-dir artifacts/lr_final
```

Train SVM (Support Vector Machine):

```bash
# Cách 1: Chạy script bash riêng (mặc định Linear SVM nhanh & calibrated)
bash train_svm_model.sh
# hoặc kernel RBF với giới hạn mẫu:
bash train_svm_model.sh Dataset.csv artifacts/svm_rbf_final rbf --sample-limit 25000

# Cách 2: Chạy script Python riêng
python train_svm_model.py --data Dataset.csv

# Cách 3: Chạy qua train_url_model
python train_url_model.py --data Dataset.csv --model svm --svm-kernel linear --output-dir artifacts/svm_final
```

## Debug / Analyze URL

Single URL (RF model artifact):

```bash
python debug_url_model.py --url "github.com" --model-path artifacts/rf_final/url_phishing_model.joblib --timeout 8 --json
```

Fresh check without relying on previous cache:

```bash
python debug_url_model.py --url "https://example.com" --model-path artifacts/rf_final/url_phishing_model.joblib --force-enrichment --cache-file artifacts/tmp_cache_run.json --json
```

## Run API + Web UI

```bash
uvicorn webapp.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000

## Benchmark

Model-only benchmark:

```bash
python benchmark_url_detector.py --data Dataset.csv --mode ml --model-path artifacts/rf_final/url_phishing_model.joblib --output-dir artifacts/benchmark_rf
```

Hybrid benchmark:

```bash
python benchmark_url_detector.py --data Dataset.csv --mode hybrid --sample-size 500 --output-dir artifacts/benchmark_hybrid
```

## Notes for This Repository

This repository is source-focused:

- Included: source code, configs, scripts
- Excluded by `.gitignore`:
  - local dataset file (`Dataset.csv`)
  - generated artifacts/reports/benchmarks (`artifacts/`)
  - virtual environment and caches

If you clone this repo on another machine, provide your own dataset and train models before running full analysis.
