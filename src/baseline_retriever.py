"""Dense Vector Retrieval Baseline using FAISS IndexFlatIP.

Course: CS 328 — Introduction to Data Science
Theory: Inner product on L2-normalized unit spheres is mathematically equivalent
to exact cosine similarity: u^T v = cos(theta) (Lecture 11).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.data_loader import DatasetSplit
from src.embeddings import EmbeddingModel, get_embedder

logger = logging.getLogger(__name__)


class DenseBaselineRetriever:
    """Exact Cosine Inner-Product Search over L2-normalized passage embeddings."""

    def __init__(self, embedder: Optional[EmbeddingModel] = None):
        self.embedder = embedder or get_embedder()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.passage_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.passage_embeddings: Optional[np.ndarray] = None
        self.truncation_stats: Dict[str, float] = {}

    def build_index(self, split: DatasetSplit, cache_prefix: str = "eval") -> None:
        """Encodes all passages in the split's corpus and indexes them in FAISS IndexFlatIP."""
        self.passage_ids = sorted(list(split.corpus.keys()))
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.passage_ids)}
        passage_texts = [split.corpus[pid].full_text for pid in self.passage_ids]

        self.passage_embeddings, self.truncation_stats = self.embedder.encode_passages_with_cache(
            passage_ids=self.passage_ids,
            passage_texts=passage_texts,
            cache_name=f"{cache_prefix}_passages",
        )

        dim = self.passage_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.passage_embeddings)
        logger.info(f"FAISS IndexFlatIP built with {self.index.ntotal} vectors of dimension {dim}.")

    def retrieve(self, query: str, top_k: int = config.K_PRIMARY) -> Tuple[List[str], Dict[str, float]]:
        """Retrieves top_k passages for a single query."""
        if self.index is None:
            raise RuntimeError("FAISS index has not been built yet. Call build_index() first.")

        query_vec = self.embedder.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(query_vec, top_k)

        retrieved_ids: List[str] = []
        scores: Dict[str, float] = {}
        for rank, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.passage_ids):
                pid = self.passage_ids[idx]
                score = float(distances[0][rank])
                retrieved_ids.append(pid)
                scores[pid] = score

        return retrieved_ids, scores

    def retrieve_all_scores(self, query: str) -> Dict[str, float]:
        """Computes cosine similarity of query against all passages in corpus (used for tie-breaking)."""
        if self.index is None or self.passage_embeddings is None:
            raise RuntimeError("FAISS index has not been built yet.")

        query_vec = self.embedder.encode([query], show_progress_bar=False)
        # Full inner product
        sims = (query_vec @ self.passage_embeddings.T)[0]
        return {pid: float(sims[idx]) for idx, pid in enumerate(self.passage_ids)}

    def evaluate_split(
        self, split: DatasetSplit, k_values: Tuple[int, ...] = config.K_VALUES
    ) -> Dict[str, dict]:
        """Runs dense retrieval across all questions in the split and evaluates recall."""
        if self.index is None:
            self.build_index(split, cache_prefix=split.name)

        results: Dict[str, dict] = {}
        logger.info(f"Evaluating {len(split.questions)} questions on Dense Baseline...")

        # Batch encode queries
        query_texts = [q.question for q in split.questions]
        query_embeddings = self.embedder.encode(query_texts, show_progress_bar=True)

        max_k = max(k_values)
        distances, indices = self.index.search(query_embeddings, max_k)

        for q_idx, q in enumerate(split.questions):
            gold = split.ground_truth[q.id]
            retrieved_max = [self.passage_ids[idx] for idx in indices[q_idx] if idx >= 0]
            scores_max = {
                self.passage_ids[idx]: float(distances[q_idx][r])
                for r, idx in enumerate(indices[q_idx])
                if idx >= 0
            }

            q_result = {
                "question_id": q.id,
                "type": q.type,
                "question": q.question,
                "retrieved": retrieved_max,
                "scores": scores_max,
                "gold": sorted(list(gold)),
            }

            for k in k_values:
                top_k_ids = set(retrieved_max[:k])
                hits = len(top_k_ids.intersection(gold))
                recall_k = hits / len(gold) if gold else 0.0
                all_recall_k = 1.0 if gold.issubset(top_k_ids) else 0.0
                q_result[f"recall@{k}"] = recall_k
                q_result[f"all_recall@{k}"] = all_recall_k

            results[q.id] = q_result

        return results


def run_dense_baseline(split: DatasetSplit) -> Dict[str, dict]:
    retriever = DenseBaselineRetriever()
    retriever.build_index(split, cache_prefix=split.name)
    return retriever.evaluate_split(split)


if __name__ == "__main__":
    from src.data_loader import get_or_create_splits

    eval_split, _ = get_or_create_splits()
    retriever = DenseBaselineRetriever()
    results = retriever.evaluate_split(eval_split)

    r5_list = [v["recall@5"] for v in results.values()]
    r2_list = [v["recall@2"] for v in results.values()]
    ar5_list = [v["all_recall@5"] for v in results.values()]

    print("\n" + "=" * 60)
    print("DENSE BASELINE RETRIEVAL RESULTS (N=200)")
    print("=" * 60)
    print(f"Mean recall@5:     {np.mean(r5_list) * 100:.2f}%")
    print(f"Mean recall@2:     {np.mean(r2_list) * 100:.2f}%")
    print(f"Mean all-recall@5: {np.mean(ar5_list) * 100:.2f}%")
    print(f"Passage truncation: {retriever.truncation_stats.get('truncated_pct', 0.0):.2f}%")

