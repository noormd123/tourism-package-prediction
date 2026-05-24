# Load, clean, encode, and split tourism data; upload splits to Hugging Face Hub
import pandas as pd
import sklearn
import os
import sys
import certifi
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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

# Hugging Face dataset repository
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/noormd100/tourism-package-data/tourism.csv"
REPO_ID = "noormd100/tourism-package-data"

# Step 1: Load dataset from Hugging Face Hub
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Step 2: Clean data — drop unnecessary columns and fix data quality issues
drop_cols = ["CustomerID"]
if "Unnamed: 0" in df.columns:
    drop_cols.append("Unnamed: 0")
df.drop(columns=drop_cols, inplace=True)
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
print(f"Cleaned dataset shape: {df.shape}")

# Step 3: Encode categorical columns using LabelEncoder
categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

target_col = "ProdTaken"

# Step 4: Split into features (X) and target (y), then train-test split (80/20)
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(Xtrain)}, Test size: {len(Xtest)}")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Step 5: Save train and test splits locally
Xtrain.to_csv(DATA_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(DATA_DIR / "Xtest.csv", index=False)
ytrain.to_csv(DATA_DIR / "ytrain.csv", index=False)
ytest.to_csv(DATA_DIR / "ytest.csv", index=False)
print(f"Saved split files to {DATA_DIR}/.")

# Step 6: Upload split files back to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for filename in files:
    file_path = DATA_DIR / filename
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to {REPO_ID}")

print("Data preparation complete.")
