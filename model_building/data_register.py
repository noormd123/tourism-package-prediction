# Register raw dataset files on Hugging Face Hub
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
import sys
import certifi
from pathlib import Path

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
repo_id = "noormd100/tourism-package-data"
repo_type = "dataset"

# Authenticate with HF_TOKEN from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Create dataset repo if it does not exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload local data folder to Hugging Face Hub
api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded {DATA_DIR} to {repo_id}")
