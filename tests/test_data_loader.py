"""Unit Tests for Phase 1 Data Loader & Preprocessing.

Course: CS 328 — Introduction to Data Science
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import (
    Passage,
    QuestionItem,
    DatasetSplit,
    compute_strata_sizes,
    get_or_create_splits,
    normalize_answer,
    normalize_text,
)


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_split, cls.tune_split = get_or_create_splits(force_recreate=False)

    def test_text_normalization(self):
        """Verify deterministic text normalization."""
        self.assertEqual(normalize_text("  Albert Einstein  "), "albert einstein")
        self.assertEqual(normalize_text("Polish-Russian\n\nWar  (film)"), "polish-russian war (film)")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("Mixed CASE"), "mixed case")

    def test_answer_normalization(self):
        """Verify QA answer normalization strips articles and punctuation."""
        self.assertEqual(normalize_answer("The United States of America!"), "united states of america")
        self.assertEqual(normalize_answer("A Dog, a Cat, and An Elephant"), "dog cat and elephant")
        self.assertEqual(normalize_answer("Yes."), "yes")
        self.assertEqual(normalize_answer("  1985  "), "1985")

    def test_split_sizes(self):
        """Verify sample sizes match pre-registered parameters exactly."""
        self.assertEqual(len(self.eval_split.questions), config.N_EVAL)
        self.assertEqual(len(self.tune_split.questions), config.N_TUNE)

    def test_zero_overlap_guarantee(self):
        """Verify evaluation and tuning splits are strictly disjoint (no data snooping)."""
        eval_ids = self.eval_split.question_ids
        tune_ids = self.tune_split.question_ids
        overlap = eval_ids.intersection(tune_ids)
        self.assertEqual(len(overlap), 0, f"Detected overlapping question IDs: {overlap}")

    def test_question_types_stratification(self):
        """Verify all 4 pre-registered question types are present with correct strata."""
        eval_counts = {qtype: len(self.eval_split.filter_by_type(qtype)) for qtype in config.QUESTION_TYPES}
        self.assertEqual(sum(eval_counts.values()), config.N_EVAL)
        self.assertEqual(eval_counts["compositional"], 83)
        self.assertEqual(eval_counts["comparison"], 48)
        self.assertEqual(eval_counts["bridge_comparison"], 44)
        self.assertEqual(eval_counts["inference"], 25)

        tune_counts = {qtype: len(self.tune_split.filter_by_type(qtype)) for qtype in config.QUESTION_TYPES}
        self.assertEqual(sum(tune_counts.values()), config.N_TUNE)
        self.assertEqual(tune_counts["compositional"], 42)
        self.assertEqual(tune_counts["comparison"], 24)
        self.assertEqual(tune_counts["bridge_comparison"], 22)
        self.assertEqual(tune_counts["inference"], 12)

    def test_passage_representation_symmetry(self):
        """Verify passages follow the unified 'Title: {title}\\nText: {text}' format."""
        for pid, passage in list(self.eval_split.corpus.items())[:50]:
            self.assertTrue(passage.full_text.startswith(f"Title: {passage.title}\nText: "))
            self.assertEqual(passage.id, normalize_text(passage.title))

    def test_gold_passage_coverage(self):
        """Verify 100% of gold supporting passages are present in the split's corpus."""
        for split in (self.eval_split, self.tune_split):
            for qid, gold_titles in split.ground_truth.items():
                self.assertGreater(len(gold_titles), 0, f"Question {qid} has empty gold titles")
                for gold in gold_titles:
                    self.assertIn(
                        gold,
                        split.corpus,
                        f"Gold title '{gold}' for question {qid} not found in {split.name} corpus!",
                    )

    def test_serialization_roundtrip(self):
        """Verify serialization and deserialization preserve split objects identically."""
        q0 = self.eval_split.questions[0]
        reloaded_q = self.eval_split.get_question(q0.id)
        self.assertIsNotNone(reloaded_q)
        self.assertEqual(q0.id, reloaded_q.id)
        self.assertEqual(q0.question, reloaded_q.question)
        self.assertEqual(q0.answer, reloaded_q.answer)
        self.assertEqual(q0.gold_titles, reloaded_q.gold_titles)


if __name__ == "__main__":
    unittest.main()
