#!/usr/bin/env python3
"""
Final iris_poisoning_mlflow_debug.py
- Ensures MLflow client uses server (default http://127.0.0.1:5000)
- Logs n_poisoned_samples, pre/post stats, confusion matrix, debug JSON, model.joblib
- Converts numpy objects to native Python types before JSON dumping
"""
import os
import json
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def ensure_tracking_uri(default_uri="http://127.0.0.1:5000"):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = default_uri
        mlflow.set_tracking_uri(default_uri)
    return mlflow.get_tracking_uri()

def poison_data_debug(X, y, poison_fraction, noise_std=1.0, seed=None):
    rng = np.random.RandomState(seed)
    Xp = X.copy()
    n = Xp.shape[0]
    k = int(round(n * poison_fraction))
    poisoned_idx = []
    if k > 0:
        poisoned_idx = rng.choice(n, k, replace=False)
        noise = rng.normal(loc=0.0, scale=noise_std, size=Xp[poisoned_idx].shape)
        Xp[poisoned_idx] += noise
    return Xp, y, poisoned_idx

def plot_and_save_cm(y_true, y_pred, outdir, label):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation='nearest')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_title(f"Confusion Matrix: {label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    outpath = os.path.join(outdir, f"confusion_{label}.png")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

def to_py(x):
    """Convert numpy types/arrays to native Python types for JSON serialization."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_py(i) for i in x]
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    return x

def run_experiment(poison_fraction, seed=42, noise_std=1.0, n_estimators=200, experiment_name="iris_poisoning_experiment_debug"):
    mlflow.set_experiment(experiment_name)
    run_name = f"poison_{int(poison_fraction*100)}pct_seed{seed}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        # Log params
        mlflow.log_param("poison_fraction", float(poison_fraction))
        mlflow.log_param("noise_std", float(noise_std))
        mlflow.log_param("seed", int(seed))
        mlflow.log_param("n_estimators", int(n_estimators))

        # Load data and split
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        # Stats before poisoning
        stats_before = {"mean": to_py(X_train.mean(axis=0)), "std": to_py(X_train.std(axis=0)), "shape": X_train.shape}

        # Poison training set only
        Xp_train, yp_train, poisoned_idx = poison_data_debug(X_train, y_train, poison_fraction, noise_std=noise_std, seed=seed)

        # Stats after poisoning
        stats_after = {"mean": to_py(Xp_train.mean(axis=0)), "std": to_py(Xp_train.std(axis=0)), "shape": Xp_train.shape}

        # Log debug metrics
        mlflow.log_metric("n_poisoned_samples", int(np.size(poisoned_idx)))
        # example scalar metrics from pre/post
        mlflow.log_metric("train_mean_feature0_before", float(stats_before["mean"][0]))
        mlflow.log_metric("train_mean_feature0_after", float(stats_after["mean"][0]))

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        model.fit(Xp_train, yp_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision_macro", float(prec))
        mlflow.log_metric("recall_macro", float(rec))
        mlflow.log_metric("f1_macro", float(f1))

        # Log model (server stores artifact)
        mlflow.sklearn.log_model(model, "model")

        # Local artifact folder for additional artifacts (keeps copies)
        local_run_dir = os.path.join("mlruns", run.info.experiment_id, run_id, "artifacts")
        os.makedirs(local_run_dir, exist_ok=True)

        # Save and log confusion matrix
        cm_path = plot_and_save_cm(y_test, y_pred, local_run_dir, f"{int(poison_fraction*100)}pct")
        mlflow.log_artifact(cm_path)

        # Debug JSON (convert numpy -> lists)
        debug_info = {
            "run_id": run_id,
            "poison_fraction": to_py(poison_fraction),
            "n_poisoned_samples": int(np.size(poisoned_idx)),
            "poisoned_indices_sample": to_py(poisoned_idx)[:100] if poisoned_idx is not None else [],
            "stats_before": to_py(stats_before),
            "stats_after": to_py(stats_after),
            "metrics": {"accuracy": to_py(acc), "precision_macro": to_py(prec), "recall_macro": to_py(rec), "f1_macro": to_py(f1)}
        }
        debug_path = os.path.join(local_run_dir, f"debug_{int(poison_fraction*100)}pct.json")
        with open(debug_path, "w") as f:
            json.dump(debug_info, f, indent=2)
        mlflow.log_artifact(debug_path)

        # Save model.joblib locally and log it
        model_joblib = os.path.join(local_run_dir, "model.joblib")
        joblib.dump(model, model_joblib)
        mlflow.log_artifact(model_joblib)

        # Print quick links for convenience (works if server is local)
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            exp_id = exp.experiment_id
            print(f"[run {run_id}] poison={poison_fraction} n_poisoned={int(np.size(poisoned_idx))} acc={acc:.4f}")
            print(f"🏃 View run at: {mlflow.get_tracking_uri()}/#/experiments/{exp_id}/runs/{run_id}")
            print(f"🧪 View experiment at: {mlflow.get_tracking_uri()}/#/experiments/{exp_id}")
        except Exception:
            print(f"[run {run_id}] poison={poison_fraction} n_poisoned={int(np.size(poisoned_idx))} acc={acc:.4f}")

def main():
    # Ensure script uses the MLflow server (unless MLFLOW_TRACKING_URI is set externally)
    uri = ensure_tracking_uri()
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison-levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.50],
                        help="List of poison fractions (e.g. 0.05 for 5%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--experiment-name", type=str, default="iris_poisoning_experiment_debug")
    args = parser.parse_args()

    print("Using MLflow tracking URI:", mlflow.get_tracking_uri())
    for p in args.poison_levels:
        run_experiment(poison_fraction=p, seed=args.seed, noise_std=args.noise_std,
                       n_estimators=args.n_estimators, experiment_name=args.experiment_name)

if __name__ == "__main__":
    main()
