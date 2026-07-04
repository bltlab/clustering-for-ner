from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from openai import OpenAI

from ..data import Mention


@dataclass
class EmbeddingClient:
    """Holds an OpenAI client and endpoint
    metadata to facilitate embedding mentions."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        prompt: Optional[str],
        with_context: bool,
    ):
        logger.info(f"Configuring Embedding Client for {model_id}")
        self.model_id = model_id
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=5,
            timeout=1200.0,
        )
        self.prompt = prompt
        self.with_context = with_context

    def embed(self, mentions: list[Mention]) -> list[NDArray[np.float32]]:
        """
        Generate embeddings for a list of mentions using the vLLM hosted model.

        Args:
            mentions: List of strings to embed. Will be
                concatenated with the provided or default prompt.

        Return:
            list[NDArray[np.float32]]: List of embeddings
        """

        logger.info(f"Generating embeddings for {len(mentions)} mentions")

        if self.prompt:
            queries = [f"{self._gen_prompt(mention)}" for mention in mentions]
        else:
            queries = [f"{' '.join(mention.tokens)}" for mention in mentions]

        embedding_response = self.client.embeddings.create(
            model=self.model_id,
            input=queries,
            encoding_format="float",
        )

        logger.info(f"Generated embeddings for {len(mentions)} mentions")

        return [
            np.array(embedding.embedding, dtype=np.float32)
            for embedding in embedding_response.data
        ]

    def _gen_prompt(self, mention: Mention) -> str:
        prompt = self.prompt if self.prompt else ""
        context = (
            f"""### Context
{mention.text}

### Mention
"""
            if self.with_context
            else ""
        )
        return prompt + context + f"{' '.join(mention.tokens)}"
