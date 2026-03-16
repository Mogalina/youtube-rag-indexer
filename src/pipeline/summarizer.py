from pathlib import Path

import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, logging

from utils.logger import get_logger
from utils.config import load_config


# Suppress Hugging Face output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.set_verbosity_error()
logging.disable_progress_bar()

logger = get_logger(__name__)


class Summarizer:
    """
    Local text summarizer using summarization transformer.

    Long texts are split into chunks (by word count), each chunk is summarized,
    and the results are concatenated into a final summary.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        config = load_config(config_path)
        summary_config = config["summary"]

        # Model parameters
        self.max_tokens = summary_config["max_tokens"]
        self.summary_max_tokens = summary_config["summary_max_tokens"]

        # Model loading
        local_dir = Path(summary_config["local_model_dir"])
        model_id = summary_config["model_id"]
        model_source = str(local_dir) if local_dir.exists() and any(local_dir.iterdir()) else model_id

        # Model initialization
        logger.info(f"Loading model from: {model_source}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = T5ForConditionalGeneration.from_pretrained(model_source)
        self.model.eval()

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"Device: {self.device}")

    def _chunk_words(
        self, 
        text: str, 
        max_words: int | None = None
    ) -> list[str]:
        """
        Split text into word-based chunks safe for the model's token limit.

        Args:
            text: Input text to summarize
            max_words: Maximum number of words per chunk

        Returns:
            List of word-based chunks
        """
        if max_words is None:
            max_words = self.max_tokens
            
        words = text.split()
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)

        return chunks

    def summarize(self, text: str) -> str:
        """
        Summarize a (potentially long) text.

        Args:
            text: Input text to summarize

        Returns:
            Summarized string
        """
        chunks = self._chunk_words(text)
        summaries = []

        for chunk in chunks:
            prompt = f"summarize: {chunk}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.summary_max_tokens,
                    num_beams=4,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return " ".join(summaries)
