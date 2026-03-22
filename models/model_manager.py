"""[DEPRECATED] HuggingFace-backed model manager.

Superseded by ``core/llm_provider.py`` (LiteLLM + Ollama) in Phase 2.
This file is kept for reference only and will be removed in Phase 3.
Do NOT use in new code.
"""

import warnings

warnings.warn(
    "ModelManager is deprecated and will be removed in Phase 3. "
    "Use core.llm_provider.LLMProvider instead.",
    DeprecationWarning,
    stacklevel=2,
)

from config import MODEL_ID, EMBEDDING_MODEL_NAME, MAX_LENGTH, REPETITION_PENALTY  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self.llm = None
        self.embedding_model = None

    def initialize_models(self) -> bool:
        """Load the LLM pipeline and embedding model.

        Returns:
            True on success, False on failure.
        """
        try:
            # Lazy imports keep test environments lightweight
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float32,
            )

            device = 0 if torch.cuda.is_available() else -1
            logger.info("Using device: %s", "GPU" if device == 0 else "CPU")

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info("Models initialised successfully")
            return True

        except Exception:
            logger.exception("Error initialising models")
            return False
