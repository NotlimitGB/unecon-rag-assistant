"""Sentence Transformers adapter; model loading is deferred until needed."""

from typing import Protocol

import numpy as np


class DenseEmbedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vector, dtype=np.float32)
