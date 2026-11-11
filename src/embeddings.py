"""Passage & Query Dense Embeddings using SentenceTransformers.

Course: CS 328 — Introduction to Data Science
Module: Sentence-Transformers (all-MiniLM-L6-v2) with L2-normalized embeddings.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config

logger = logging.getLogger(__name__)

EMB_CACHE_DIR = config.CACHE_DIR / "embeddings"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingModel:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading SentenceTransformer: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = config.MAX_SEQ_LENGTH
        self.model.max_seq_length = self.max_seq_length

    def count_tokens(self, text: str) -> int:
        """Counts wordpiece tokens using the model's tokenizer."""
        return len(self.tokenizer.tokenize(text))

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encodes texts into L2-normalized dense embeddings (dim=384)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # Crucial for inner product = cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_passages_with_cache(
        self,
        passage_ids: List[str],
        passage_texts: List[str],
        cache_name: str = "passages",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Encodes passage texts with disk caching and monitors token truncation."""
        content_hash = hashlib.sha256(
            (self.model_name + "".join(passage_ids)).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = EMB_CACHE_DIR / f"{cache_name}_{content_hash}.npy"
        meta_path = EMB_CACHE_DIR / f"{cache_name}_{content_hash}_meta.npz"

        if cache_path.exists() and meta_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}...")
            embeddings = np.load(cache_path)
            meta = np.load(meta_path)
            stats = {
                "total_passages": int(meta["total_passages"]),
                "truncated_count": int(meta["truncated_count"]),
                "truncated_pct": float(meta["truncated_pct"]),
            }
            return embeddings, stats

        # Monitor token truncation
        truncated_count = 0
        for text in passage_texts:
            if self.count_tokens(text) > self.max_seq_length:
                truncated_count += 1
        truncated_pct = (truncated_count / len(passage_texts) * 100) if passage_texts else 0.0
        stats = {
            "total_passages": len(passage_texts),
            "truncated_count": truncated_count,
            "truncated_pct": truncated_pct,
        }
        logger.info(
            f"Passage truncation check: {truncated_count}/{len(passage_texts)} ({truncated_pct:.2f}%) exceed {self.max_seq_length} tokens"
        )

        logger.info(f"Encoding {len(passage_texts)} passages...")
        embeddings = self.encode(passage_texts, show_progress_bar=True)

        np.save(cache_path, embeddings)
        np.savez(
            meta_path,
            total_passages=stats["total_passages"],
            truncated_count=stats["truncated_count"],
            truncated_pct=stats["truncated_pct"],
        )
        return embeddings, stats


_DEFAULT_EMBEDDER: Optional[EmbeddingModel] = None

def get_embedder() -> EmbeddingModel:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = EmbeddingModel()
    return _DEFAULT_EMBEDDER
