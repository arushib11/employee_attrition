import os

def ensure_directory(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))