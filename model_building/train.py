# Train XGBoost model with MLflow tracking; register best model on Hugging Face Hub
import math
import pandas as pd
import os
import sys
import certifi
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def _fix_ssl_certs():
    # Clear broken SSL env vars and point to a valid CA bundle
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT"):
        path = os.environ.get(key, "")
        if path and not os.path.isfile(path):
            os.environ.pop(key, None)
    ca = certifi.where()
    if os.path.isfile(ca):
        os.environ["SSL_CERT_FILE"] = ca
        os.environ["REQUESTS_CA_BUNDLE"] = ca
        return ca
    candidate = f"/opt/homebrew/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/certifi/cacert.pem"
    if os.path.isfile(candidate):
        os.environ["SSL_CERT_FILE"] = candidate
        os.environ["REQUESTS_CA_BUNDLE"] = candidate
        return candidate
    return None


_fix_ssl_certs()

# Disable git metadata logging in CI (avoids GitHub auth prompts in Actions)
os.environ.setdefault("MLFLOW_GIT_DISABLE", "1")

# MLflow tracking for production and CI/CD runs
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-experiment")

# Hugging Face dataset and model repositories
DATASET_REPO = "noormd100/tourism-package-data"
MODEL_REPO = "noormd100/tourism-package-model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Load train and test data from Hugging Face Hub
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze("columns")
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze("columns")
print("Train and test data loaded successfully from Hugging Face Hub.")

feature_cols = Xtrain.columns.tolist()

# Step 2: Define model, pipeline, and hyperparameter grid
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), feature_cols),
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__colsample_bytree": [0.4, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
print("XGBoost model and hyperparameter grid defined.")

# Step 3: Tune model with GridSearchCV and log all tuned parameters in MLflow
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=1
    )
    grid_search.fit(Xtrain, ytrain)
    print("Hyperparameter tuning complete.")

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        mean_score = results["mean_test_score"][i]
        if math.isnan(mean_score):
            continue
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    print(f"Best parameters: {grid_search.best_params_}")

    # Step 4: Evaluate model performance on train and test sets
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train = (
        best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold
    ).astype(int)
    y_pred_test = (
        best_model.predict_proba(Xtest)[:, 1] >= classification_threshold
    ).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1-score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"],
        }
    )

    print("\nTrain Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # Step 5: Save best model locally and register on Hugging Face model hub
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    print(f"Model saved locally as {MODEL_PATH}")

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO}' not found. Creating new model repo...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print(f"Model repo '{MODEL_REPO}' created.")

    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded to {MODEL_REPO}")

print("Model building complete.")
