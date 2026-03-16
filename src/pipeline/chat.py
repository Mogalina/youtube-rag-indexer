import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from utils.logger import get_logger
from utils.config import load_config


# Suppress Hugging Face output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.set_verbosity_error()

logger = get_logger(__name__)


class ChatEngine:
    """
    Local Question Answering engine using question answering small language model 
    with context.

    Uses local embedding model to embed the question and then uses the embedding
    to search for the most relevant chunks in the FAISS index.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ChatEngine.

        Args:
            config_path: Path to the configuration file
        """
        config = load_config(config_path)
        chat_config = config["chat"]

        self.max_new_tokens = chat_config["max_new_tokens"]
        
        # Model loading
        local_dir = Path(chat_config["local_model_dir"])
        model_id = chat_config["model_id"]
        model_source = str(local_dir) if local_dir.exists() and any(local_dir.iterdir()) else model_id

        logger.info(f"Loading chat model from: {model_source}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model initialization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source, 
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(torch.float16)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"Chat device: {self.device}")

    def answer(self, question: str, context: str) -> str:
        """
        Generate an answer based on the provided context.

        Args:
            question: The question to answer
            context: The context to use for answering the question

        Returns:
            The answer to the question
        """
        # Prompt engineering template
        prompt = (
            f"Instruct: Use the following context to answer the question. "
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Output:"
        )
        
        # Tokenize the input prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=False
        ).to(self.device)
        
        # Generate the answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated tokens back to text
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the output part
        if "Output:" in full_text:
            return full_text.split("Output:")[-1].strip()
        
        return full_text.strip()
