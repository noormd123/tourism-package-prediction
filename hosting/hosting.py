# Upload deployment files to Hugging Face Space
from huggingface_hub import HfApi
import os
from pathlib import Path

# Resolve deployment folder relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_FOLDER = PROJECT_ROOT / "deployment"

# Hugging Face Space repository
SPACE_REPO = "noormd100/tourism-package-prediction"

# Authenticate with HF_TOKEN and upload deployment folder
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=str(DEPLOYMENT_FOLDER),
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files uploaded to {SPACE_REPO}")
