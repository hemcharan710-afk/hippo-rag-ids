"""Data Loading, Preprocessing, and Stratified Partitioning for 2WikiMultiHopQA.

Course: CS 328 — Introduction to Data Science
Project: Graph-Based Retrieval vs Vector Search on Multi-Hop Questions

This module implements Phase 1 of the implementation plan:
1. Downloading and loading the 2WikiMultiHopQA development set (`dev.json`).
2. Canonical string normalization for passage titles, entities, and answers.
3. Constructing symmetric passage representations: "Title: {title}\nText: {text}".
4. Disjoint stratified partitioning into:
   - Evaluation Split (N=200, seed=42) proportional to dev set question types.
   - Tuning Split (N=100, seed=101) completely disjoint from evaluation.
5. Extracting ground-truth supporting passage mappings for exact recall evaluation.
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests
from tqdm import tqdm

from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Official / verified mirror URL for 2WikiMultiHopQA dev.json
DEV_DATASET_URL = (
    "https://huggingface.co/datasets/RAGLAB/data/resolve/main/eval_datasets/2WikiMultiHopQA/dev.json"
)


# ==============================================================================
# 1. String Normalization Functions
# ==============================================================================
def normalize_text(text: str) -> str:
    """Canonical normalization for passage titles and entity strings.

    Lowercases text, collapses multiple whitespace characters into single spaces,
    and strips leading and trailing whitespace. Preserves internal punctuation
    (e.g., 'Polish-Russian War (film)') to keep entity names distinguishable.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_answer(text: str) -> str:
    """Canonical answer normalization for QA evaluation (Exact Match & F1).

    Lowercases text, strips standard English punctuation, removes English
    articles ('a', 'an', 'the'), and collapses internal whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    # Strip articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Strip punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# 2. Data Structures
# ==============================================================================
@dataclass(frozen=True)
class Passage:
    """Represents a standardized context passage.

    Both dense vector search and graph-based retrieval operate on the exact
    same textual representation: `full_text = f"Title: {title}\nText: {text}"`.
    """

    id: str  # Normalized title
    title: str  # Raw original title
    text: str  # Joined body text
    full_text: str  # Unified symmetric representation
    sentences: Tuple[str, ...]  # Original individual sentences

    @classmethod
    def from_raw(cls, raw_title: str, raw_sentences: List[str]) -> Passage:
        norm_title = normalize_text(raw_title)
        body = " ".join(raw_sentences).strip()
        full = f"Title: {raw_title}\nText: {body}"
        return cls(
            id=norm_title,
            title=raw_title,
            text=body,
            full_text=full,
            sentences=tuple(raw_sentences),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "full_text": self.full_text,
            "sentences": list(self.sentences),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Passage:
        return cls(
            id=data["id"],
            title=data["title"],
            text=data["text"],
            full_text=data["full_text"],
            sentences=tuple(data.get("sentences", [])),
        )


@dataclass
class QuestionItem:
    """Represents a multi-hop question with ground-truth and local context."""

    id: str
    question: str
    type: str  # 'compositional', 'inference', 'comparison', 'bridge_comparison'
    answer: str
    gold_titles: Set[str]  # Normalized titles of gold supporting passages
    supporting_facts: List[Tuple[str, int]]  # Raw (title, sent_idx) pairs
    context_titles: Set[str]  # Normalized titles present in question's context

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "answer": self.answer,
            "gold_titles": sorted(list(self.gold_titles)),
            "supporting_facts": [list(f) for f in self.supporting_facts],
            "context_titles": sorted(list(self.context_titles)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuestionItem:
        return cls(
            id=data["id"],
            question=data["question"],
            type=data["type"],
            answer=data["answer"],
            gold_titles=set(data["gold_titles"]),
            supporting_facts=[(f[0], int(f[1])) for f in data["supporting_facts"]],
            context_titles=set(data["context_titles"]),
        )


@dataclass
class DatasetSplit:
    """Encapsulates a self-contained split (questions + deduplicated corpus)."""

    name: str  # 'eval' or 'tune'
    questions: List[QuestionItem]
    corpus: Dict[str, Passage]  # normalized title -> Passage
    ground_truth: Dict[str, Set[str]]  # question_id -> set of normalized gold titles

    @property
    def question_ids(self) -> Set[str]:
        return {q.id for q in self.questions}

    def get_question(self, qid: str) -> Optional[QuestionItem]:
        for q in self.questions:
            if q.id == qid:
                return q
        return None

    def get_passage(self, title: str) -> Optional[Passage]:
        return self.corpus.get(normalize_text(title))

    def filter_by_type(self, qtype: str) -> List[QuestionItem]:
        return [q for q in self.questions if q.type == qtype]

    def summary(self) -> str:
        type_counts = Counter(q.type for q in self.questions)
        lines = [
            f"=== DatasetSplit: {self.name.upper()} ===",
            f"Total Questions: {len(self.questions)}",
            f"Unique Passages in Corpus: {len(self.corpus)}",
            "Stratification Breakdown:",
        ]
        for qtype in config.QUESTION_TYPES:
            count = type_counts.get(qtype, 0)
            pct = (count / len(self.questions) * 100) if self.questions else 0.0
            lines.append(f"  - {qtype:<18}: {count:>3} ({pct:5.1f}%)")

        gold_lens = [len(gt) for gt in self.ground_truth.values()]
        avg_gold = sum(gold_lens) / len(gold_lens) if gold_lens else 0.0
        lines.append(f"Avg Gold Passages/Question: {avg_gold:.2f}")
        return "\n".join(lines)


# ==============================================================================
# 3. Download & Raw Loading
# ==============================================================================
def download_dataset_if_needed(destination: Path = config.DEV_DATA_FILE) -> Path:
    """Downloads 2WikiMultiHopQA dev.json if not present locally."""
    if destination.exists() and destination.stat().st_size > 10_000_000:
        logger.info(f"Dataset already exists at {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading 2WikiMultiHopQA dev.json from {DEV_DATASET_URL}...")
    response = requests.get(DEV_DATASET_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(destination, "wb") as f, tqdm(
        desc="dev.json",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"Downloaded dev.json successfully to {destination}")
    return destination


def load_raw_data(data_path: Path = config.DEV_DATA_FILE) -> List[dict]:
    """Loads raw JSON items from 2WikiMultiHopQA dev.json."""
    if not data_path.exists():
        download_dataset_if_needed(data_path)

    logger.info(f"Loading raw dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    logger.info(f"Loaded {len(raw_items)} questions from dev set.")
    return raw_items


# ==============================================================================
# 4. Stratified Split Creation
# ==============================================================================
def compute_strata_sizes(target_total: int, raw_items: List[dict]) -> Dict[str, int]:
    """Computes strata sample sizes matching dev set proportions.

    Rounding discrepancies are added to the largest stratum ('compositional')
    to ensure the exact target total is reached.
    """
    total_items = len(raw_items)
    type_counts = Counter(item["type"] for item in raw_items)
    sizes = {
        qtype: round(target_total * (type_counts.get(qtype, 0) / total_items))
        for qtype in config.QUESTION_TYPES
    }
    # Fix rounding discrepancy if any
    discrepancy = target_total - sum(sizes.values())
    if discrepancy != 0:
        sizes["compositional"] += discrepancy
    return sizes


def build_split_from_raw_items(name: str, sampled_items: List[dict]) -> DatasetSplit:
    """Converts raw JSON items into a DatasetSplit with deduplicated corpus."""
    questions: List[QuestionItem] = []
    corpus: Dict[str, Passage] = {}
    ground_truth: Dict[str, Set[str]] = {}

    for raw in sampled_items:
        qid = str(raw["_id"])
        qtype = str(raw["type"])
        question_text = str(raw["question"]).strip()
        answer = str(raw.get("answer", "")).strip()

        # Extract context passages
        context_titles: Set[str] = set()
        for title, sentences in raw.get("context", []):
            passage = Passage.from_raw(title, sentences)
            context_titles.add(passage.id)
            if passage.id not in corpus:
                corpus[passage.id] = passage

        # Extract gold supporting passage titles
        raw_facts = raw.get("supporting_facts", [])
        gold_titles = {normalize_text(fact[0]) for fact in raw_facts if fact}

        # Verify gold passages are present in context
        for gold in gold_titles:
            if gold not in corpus:
                # Fallback: if passage was not in context list, create minimal passage
                corpus[gold] = Passage(
                    id=gold,
                    title=gold,
                    text="",
                    full_text=f"Title: {gold}\nText: ",
                    sentences=(),
                )

        q_item = QuestionItem(
            id=qid,
            question=question_text,
            type=qtype,
            answer=answer,
            gold_titles=gold_titles,
            supporting_facts=[(f[0], int(f[1])) for f in raw_facts if len(f) >= 2],
            context_titles=context_titles,
        )
        questions.append(q_item)
        ground_truth[qid] = gold_titles

    return DatasetSplit(
        name=name,
        questions=questions,
        corpus=corpus,
        ground_truth=ground_truth,
    )


def create_disjoint_splits(
    raw_items: Optional[List[dict]] = None,
    n_eval: int = config.N_EVAL,
    seed_eval: int = config.SEED_EVAL,
    n_tune: int = config.N_TUNE,
    seed_tune: int = config.SEED_TUNE,
) -> Tuple[DatasetSplit, DatasetSplit]:
    """Partitions dev.json into disjoint evaluation (N=200) and tuning (N=100) splits.

    Guarantees:
    1. Deterministic sampling: Questions sorted by ID prior to pseudo-random sampling.
    2. Zero overlap: `eval_split.question_ids & tune_split.question_ids == set()`.
    3. Stratification: Both splits mirror the 4-way question type dev distribution.
    4. Independent corpora: Each split has its own deduplicated passage corpus.
    """
    if raw_items is None:
        raw_items = load_raw_data()

    # Group questions by type
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for item in raw_items:
        by_type[item["type"]].append(item)

    # 1. Sample Evaluation Split (N_eval, seed_eval)
    eval_strata = compute_strata_sizes(n_eval, raw_items)
    rng_eval = random.Random(seed_eval)
    eval_raw: List[dict] = []

    for qtype in sorted(eval_strata.keys()):
        size = eval_strata[qtype]
        candidates = sorted(by_type[qtype], key=lambda x: str(x["_id"]))
        eval_raw.extend(rng_eval.sample(candidates, size))

    eval_ids = {str(item["_id"]) for item in eval_raw}

    # 2. Candidate pool for Tuning Split: Strictly exclude all eval IDs
    remaining_by_type: Dict[str, List[dict]] = defaultdict(list)
    for qtype, items in by_type.items():
        for item in items:
            if str(item["_id"]) not in eval_ids:
                remaining_by_type[qtype].append(item)

    # 3. Sample Tuning Split (N_tune, seed_tune) from disjoint pool
    tune_strata = compute_strata_sizes(n_tune, raw_items)
    rng_tune = random.Random(seed_tune)
    tune_raw: List[dict] = []

    for qtype in sorted(tune_strata.keys()):
        size = tune_strata[qtype]
        candidates = sorted(remaining_by_type[qtype], key=lambda x: str(x["_id"]))
        tune_raw.extend(rng_tune.sample(candidates, size))

    tune_ids = {str(item["_id"]) for item in tune_raw}

    # Invariant verification: Zero overlap
    overlap = eval_ids.intersection(tune_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} questions overlap between eval and tune!"
    assert len(eval_raw) == n_eval, f"Expected {n_eval} eval items, got {len(eval_raw)}"
    assert len(tune_raw) == n_tune, f"Expected {n_tune} tune items, got {len(tune_raw)}"

    eval_split = build_split_from_raw_items("eval", eval_raw)
    tune_split = build_split_from_raw_items("tune", tune_raw)

    return eval_split, tune_split


# ==============================================================================
# 5. Serialization & Storage
# ==============================================================================
def save_split_to_json(split: DatasetSplit, filepath: Path) -> None:
    """Serializes a DatasetSplit to a standalone JSON file."""
    data = {
        "name": split.name,
        "questions": [q.to_dict() for q in split.questions],
        "corpus": {pid: p.to_dict() for pid, p in split.corpus.items()},
        "ground_truth": {qid: sorted(list(golds)) for qid, golds in split.ground_truth.items()},
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {split.name} split ({len(split.questions)} Qs, {len(split.corpus)} passages) to {filepath}")


def load_split_from_json(filepath: Path) -> DatasetSplit:
    """Loads a previously saved DatasetSplit from JSON."""
    logger.info(f"Loading split from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [QuestionItem.from_dict(q) for q in data["questions"]]
    corpus = {pid: Passage.from_dict(p) for pid, p in data["corpus"].items()}
    ground_truth = {qid: set(golds) for qid, golds in data["ground_truth"].items()}

    return DatasetSplit(
        name=data["name"],
        questions=questions,
        corpus=corpus,
        ground_truth=ground_truth,
    )


def get_or_create_splits(force_recreate: bool = False) -> Tuple[DatasetSplit, DatasetSplit]:
    """Loads existing pre-partitioned splits or creates and persists them."""
    if not force_recreate and config.EVAL_SPLIT_FILE.exists() and config.TUNE_SPLIT_FILE.exists():
        logger.info("Found existing serialized splits. Loading directly...")
        eval_split = load_split_from_json(config.EVAL_SPLIT_FILE)
        tune_split = load_split_from_json(config.TUNE_SPLIT_FILE)
        return eval_split, tune_split

    logger.info("Creating fresh stratified splits from dev.json...")
    raw_items = load_raw_data()
    eval_split, tune_split = create_disjoint_splits(raw_items)

    save_split_to_json(eval_split, config.EVAL_SPLIT_FILE)
    save_split_to_json(tune_split, config.TUNE_SPLIT_FILE)

    return eval_split, tune_split


# ==============================================================================
# 6. Self-Verification Execution
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION: 2WikiMultiHopQA Data Loader & Partitioning")
    print("=" * 70)

    eval_split, tune_split = get_or_create_splits(force_recreate=True)

    print("\n" + eval_split.summary())
    print("\n" + tune_split.summary())

    overlap = eval_split.question_ids.intersection(tune_split.question_ids)
    print(f"\n[Verification] Eval & Tune question ID overlap: {len(overlap)} (Must be 0)")
    assert len(overlap) == 0, "Question overlap detected!"

    # Sample Inspection
    sample_q = eval_split.questions[0]
    print("\n" + "-" * 70)
    print("Sample Evaluation Question:")
    print(f"ID:              {sample_q.id}")
    print(f"Question Type:   {sample_q.type}")
    print(f"Question:        {sample_q.question}")
    print(f"Gold Answer:     {sample_q.answer}")
    print(f"Gold Passages:   {sample_q.gold_titles}")
    print(f"Supporting Facts Count: {len(sample_q.supporting_facts)}")

    sample_gold_title = next(iter(sample_q.gold_titles))
    sample_passage = eval_split.get_passage(sample_gold_title)
    print("\nSample Gold Passage Unified Representation:")
    if sample_passage:
        print(f"Passage ID (Normalized Title): '{sample_passage.id}'")
        print("Passage Full Text:")
        print("--- BEGIN PASSAGE ---")
        print(sample_passage.full_text[:300] + ("..." if len(sample_passage.full_text) > 300 else ""))
        print("--- END PASSAGE ---")

    print("\n[Verification] All Phase 1 assertions passed successfully!")
