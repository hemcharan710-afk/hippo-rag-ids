"""Unit Tests for Phase 2: Embeddings & Dense Baseline Retriever.

Course: CS 328 — Introduction to Data Science
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import get_or_create_splits
from src.embeddings import get_embedder
from src.baseline_retriever import DenseBaselineRetriever


class TestDenseBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_split, _ = get_or_create_splits(force_recreate=False)
        cls.embedder = get_embedder()
        cls.retriever = DenseBaselineRetriever(embedder=cls.embedder)
        cls.retriever.build_index(cls.eval_split, cache_prefix="eval")

    def test_embedding_normalization(self):
        """Verify embeddings lie on the unit hypersphere: ||v||_2 = 1.0 (Lecture 11)."""
        sample_texts = ["First passage text", "Second query text"]
        vecs = self.embedder.encode(sample_texts)
        self.assertEqual(vecs.shape, (2, config.EMBEDDING_DIM))
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_faiss_index_integrity(self):
        """Verify FAISS index has exact passage count and dimensionality."""
        self.assertEqual(self.retriever.index.ntotal, len(self.eval_split.corpus))
        self.assertEqual(self.retriever.index.d, config.EMBEDDING_DIM)

    def test_single_retrieval(self):
        """Verify single query retrieval returns top_k passages with valid cosine scores."""
        query = "Who directed the film Polish-Russian War?"
        retrieved_ids, scores = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(len(retrieved_ids), 5)
        self.assertEqual(len(scores), 5)
        # Cosine scores should be within [-1.0, 1.0]
        for pid, score in scores.items():
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0001)

    def test_recall_metric_bounds(self):
        """Verify recall scores are strictly bounded in [0.0, 1.0]."""
        results = self.retriever.evaluate_split(self.eval_split)
        for qid, res in results.items():
            self.assertGreaterEqual(res["recall@5"], 0.0)
            self.assertLessEqual(res["recall@5"], 1.0)
            self.assertGreaterEqual(res["recall@2"], 0.0)
            self.assertLessEqual(res["recall@2"], 1.0)
            self.assertIn(res["all_recall@5"], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()

