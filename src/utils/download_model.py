import yaml

from pathlib import Path
from huggingface_hub import snapshot_download

from logger import get_logger


logger = get_logger(__name__)

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load the configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_file = project_root / config_path
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def download_model(model_id: str, local_dir: str) -> None:
    """
    Download a model from HuggingFace to a local directory.
    
    Args:
        model_id (str): HuggingFace model ID
        local_dir (str): Local directory to save the model
    """
    logger.info(f"Downloading model '{model_id}' to '{local_dir}'")
    
    # Resolve `local_dir` relative to `project_root` if it's not absolute
    local_path = Path(local_dir)
    if not local_path.is_absolute():
        local_path = project_root / local_dir
    
    # Create the local directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded '{model_id}' to '{local_path.resolve()}'")
    except Exception as e:
        logger.error(f"Failed to download model '{model_id}': {e}")
        raise
