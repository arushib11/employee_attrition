import os
import yaml

def ensure_directory(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dvc_data_md5(dvc_pointer_path: str) -> str | None:
    """
    Return the md5 hash stored in a `.dvc` pointer file.

    Used as a lightweight "data version" label for MLflow logging.
    """
    try:
        with open(dvc_pointer_path, "r") as f:
            dvc_meta = yaml.safe_load(f) or {}
        outs = dvc_meta.get("outs") or []
        if not outs or not isinstance(outs, list):
            return None
        md5 = outs[0].get("md5")
        return md5 if isinstance(md5, str) else None
    except Exception:
        return None