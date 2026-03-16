import warnings
from pathlib import Path
from huggingface_hub import snapshot_download

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from utils.logger import get_logger


logger = get_logger(__name__)


def download_model(model_id: str, local_dir: str) -> None:
    """
    Download a model from HuggingFace to a local directory.
    
    Args:
        model_id: HuggingFace model ID
        local_dir: Local directory to save the model
    """
    logger.info(f"Downloading model '{model_id}' to '{local_dir}'")
    
    local_path = Path(local_dir)
    if not local_path.is_absolute():
        from utils.config import PROJECT_ROOT
        local_path = PROJECT_ROOT / local_dir
    
    # Create the local directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    console = Console()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        with Live(
            Spinner("dots", text=f"[bold blue]Pulling model {model_id}...[/bold blue]"),
            refresh_per_second=10,
            console=console,
            transient=True
        ):
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_path)
                )
                logger.info(f"Successfully downloaded '{model_id}' to '{local_path.resolve()}'")
            except Exception as e:
                logger.error(f"Failed to download model '{model_id}': {e}")
                console.print(f"[red]Failed to pull {model_id}: {e}[/red]")
                raise
