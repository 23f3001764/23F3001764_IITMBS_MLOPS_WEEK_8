# Week 8 — IRIS Data Poisoning Experiments

This README covers the Week 8 assignment for IRIS data poisoning using MLflow.

## 📁 Files and Utilities
This folder contains the following files:

- **iris_poisoning_mlflow_debug.py** — Main experiment script that performs poisoning and logs results to MLflow.
- **myscript.txt** — Helper notes or commands you created (content can be embedded if needed).
- **week8_metrics.csv** — Exported metrics summary for all MLflow runs.
- **artifacts_flat/** — Contains confusion matrices, debug JSON files, and model artifacts collected from MLflow runs.
- **README.md / README-week8.md** — Documentation explaining the Week 8 workflow and results.

## 🧪 How to Run the Experiments
Make sure the MLflow server is running on your VM and accessible:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python iris_poisoning_mlflow_debug.py --poison-levels 0.0 0.05 0.10 0.50 --noise-std 1.0
```

## 📄 Included Script Placeholder
The content of **myscript.txt** can be inserted here if you upload or paste it.

