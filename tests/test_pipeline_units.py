"""Unit tests for Entity Linker, PPR Retriever, Evaluation Metrics, and Statistical Tests.

Course: CS 328 — Introduction to Data Science
"""

import unittest
import numpy as np
import networkx as nx

from src.entity_linker import EntityLinker, HyperplaneLSH
from src.evaluation import (
    compute_exact_match,
    compute_token_f1,
    compute_recall_at_k,
    compute_all_recall_at_k,
    normalize_answer,
    evaluate_retrieval_metrics,
)
from src.statistical_tests import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    paired_sign_flip_permutation_test,
    paired_mcnemar_test,
    holm_bonferroni,
    evaluate_preregistered_verdict,
)


class TestHyperplaneLSH(unittest.TestCase):
    def test_lsh_insertion_and_query(self):
        # l = 16, r = 4, b = 4 (l == r * b)
        lsh = HyperplaneLSH(l=16, r=4, b=4, seed=42)
        dim = 16
        vec1 = np.ones((1, dim), dtype=np.float32)
        vec1 /= np.linalg.norm(vec1)
        vec2 = vec1.copy()  # Identical vector
        vec3 = -vec1.copy() # Opposite vector
        embeddings = np.vstack([vec1, vec2, vec3])
        entities = ["e1", "e2", "e3"]

        lsh.fit(entities, embeddings)

        query_sig = (vec1 @ lsh.hyperplanes.T > 0).astype(np.uint8)[0]
        candidates = lsh.get_candidates(query_sig)

        # e1 and e2 should be among candidates (indices 0 and 1)
        self.assertIn(0, candidates)
        self.assertIn(1, candidates)


class TestEvaluationMetrics(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(normalize_answer("The United States of America."), "united states of america")
        self.assertEqual(normalize_answer("   An Apple, A Day!  "), "apple day")
        self.assertEqual(normalize_answer("yes"), "yes")

    def test_exact_match(self):
        self.assertEqual(compute_exact_match("United States", "the united states."), 1.0)
        self.assertEqual(compute_exact_match("Paris", "London"), 0.0)

    def test_token_f1(self):
        # Exact match -> F1 = 1.0
        self.assertAlmostEqual(compute_token_f1("New York City", "new york city"), 1.0)
        # 2 out of 3 tokens match -> F1 = 0.8
        self.assertAlmostEqual(compute_token_f1("New York", "new york city"), 0.8)
        # Disjoint -> 0.0
        self.assertAlmostEqual(compute_token_f1("Paris", "London"), 0.0)

    def test_recall_at_k(self):
        gold = {"p1", "p2"}
        # Both found in top 2
        self.assertEqual(compute_recall_at_k(["p1", "p2", "p3"], gold, k=2), 1.0)
        # 1 found in top 2
        self.assertEqual(compute_recall_at_k(["p1", "p3", "p2"], gold, k=2), 0.5)
        # None found
        self.assertEqual(compute_recall_at_k(["p4", "p5"], gold, k=2), 0.0)

    def test_all_recall_at_k(self):
        gold = {"p1", "p2"}
        self.assertEqual(compute_all_recall_at_k(["p1", "p2", "p3"], gold, k=2), 1.0)
        self.assertEqual(compute_all_recall_at_k(["p1", "p3", "p2"], gold, k=2), 0.0)


class TestStatisticalTests(unittest.TestCase):
    def test_bootstrap_diff_ci(self):
        a = np.array([1.0, 1.0, 0.8, 0.9, 1.0] * 20)
        b = np.array([0.2, 0.3, 0.1, 0.4, 0.2] * 20)
        ci_low, ci_high = bootstrap_diff_ci(a, b, B=500, seed=42)
        # Mean diff is ~0.7, CI should be well above 0
        self.assertGreater(ci_low, 0.5)
        self.assertLess(ci_high, 0.9)

    def test_sign_flip_permutation_test(self):
        a = np.array([1.0] * 50)
        b = np.array([0.0] * 50)
        stat, p_val = paired_sign_flip_permutation_test(a, b, n_perm=1000, seed=42)
        self.assertLess(p_val, 0.005)

    def test_holm_bonferroni(self):
        p_vals = {
            "test1": 0.001,
            "test2": 0.015,
            "test3": 0.040,
            "test4": 0.250,
        }
        results = holm_bonferroni(p_vals, alpha=0.05)
        # Sorted: 0.001 (alpha/4=0.0125, sig), 0.015 (alpha/3=0.0167, sig), 0.040 (alpha/2=0.025, not sig)
        self.assertTrue(results["test1"]["significant"])
        self.assertTrue(results["test2"]["significant"])
        self.assertFalse(results["test3"]["significant"])
        self.assertFalse(results["test4"]["significant"])

    def test_preregistered_verdict(self):
        # Superior
        v1 = evaluate_preregistered_verdict(0.10, 0.02, 0.18, 0.001)
        self.assertIn("BRANCH 1: REJECT H0 IN FAVOR OF H1", v1)
        # Inferior
        v2 = evaluate_preregistered_verdict(-0.10, -0.18, -0.02, 0.001)
        self.assertIn("BRANCH 2: REJECT H0 IN FAVOR OF VECTOR SEARCH", v2)
        # Inconclusive
        v3 = evaluate_preregistered_verdict(0.02, -0.05, 0.09, 0.35)
        self.assertIn("BRANCH 3: FAIL TO REJECT H0", v3)


if __name__ == "__main__":
    unittest.main()
