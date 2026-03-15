from pathlib import Path
from huggingface_hub import snapshot_download

from logger import get_logger
from config import load_config


logger = get_logger(__name__)


def download_model(model_id: str, local_dir: str) -> None:
    """
    Download a model from HuggingFace to a local directory.
    
    Args:
        model_id: HuggingFace model ID
        local_dir: Local directory to save the model
    """
    logger.info(f"Downloading model '{model_id}' to '{local_dir}'")
    
    config = load_config()
    local_path = Path(local_dir)
    if not local_path.is_absolute():
        local_path = config["pipeline"]["local_model_dir"] / local_dir
    
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
