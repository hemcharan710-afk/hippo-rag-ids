"""Evaluation Metrics and Pre-Specified Hypothesis Contrasts.

Course: CS 328 — Introduction to Data Science
Implements retrieval recall, all-recall, exact match (EM), token F1,
and confirmatory reasoning chain contrasts.
"""

from __future__ import annotations

import collections
import logging
import re
import string
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.data_loader import normalize_answer

logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Computes recall@k for a single question."""
    if not gold:
        return 0.0
    top_k_set = set(retrieved[:k])
    return float(len(top_k_set.intersection(gold)) / len(gold))


def compute_all_recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Computes all-recall@k (1.0 if all gold passages present in top-k, else 0.0)."""
    if not gold:
        return 0.0
    top_k_set = set(retrieved[:k])
    return 1.0 if gold.issubset(top_k_set) else 0.0


def compute_exact_match(pred: str, gold: str) -> float:
    """Computes exact match binary score between normalized prediction and gold answer."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_token_f1(pred: str, gold: str) -> float:
    """Computes token-level F1 score between normalized prediction and gold answer."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def evaluate_retrieval_metrics(
    results_graph: Dict[str, dict],
    results_vector: Dict[str, dict],
    k_values: Tuple[int, ...] = config.K_VALUES,
) -> Dict[str, dict]:
    """Computes overall and stratified retrieval metric summaries."""
    question_ids = sorted(list(results_graph.keys()))
    metrics_summary: Dict[str, dict] = {}

    for k in k_values:
        r_g = np.array([results_graph[qid][f"recall@{k}"] for qid in question_ids])
        r_v = np.array([results_vector[qid][f"recall@{k}"] for qid in question_ids])
        ar_g = np.array([results_graph[qid][f"all_recall@{k}"] for qid in question_ids])
        ar_v = np.array([results_vector[qid][f"all_recall@{k}"] for qid in question_ids])

        diff_r = r_g - r_v
        diff_ar = ar_g - ar_v

        metrics_summary[f"recall@{k}"] = {
            "graph_mean": float(np.mean(r_g)),
            "vector_mean": float(np.mean(r_v)),
            "delta_mean": float(np.mean(diff_r)),
            "graph_scores": r_g,
            "vector_scores": r_v,
            "diffs": diff_r,
        }
        metrics_summary[f"all_recall@{k}"] = {
            "graph_mean": float(np.mean(ar_g)),
            "vector_mean": float(np.mean(ar_v)),
            "delta_mean": float(np.mean(diff_ar)),
            "graph_scores": ar_g,
            "vector_scores": ar_v,
            "diffs": diff_ar,
        }

    # Stratified by question type
    strata: Dict[str, dict] = defaultdict(dict)
    for qtype in config.QUESTION_TYPES:
        q_ids = [qid for qid in question_ids if results_graph[qid]["type"] == qtype]
        if not q_ids:
            continue
        r5_g = np.array([results_graph[qid]["recall@5"] for qid in q_ids])
        r5_v = np.array([results_vector[qid]["recall@5"] for qid in q_ids])
        strata[qtype] = {
            "count": len(q_ids),
            "graph_recall@5": float(np.mean(r5_g)),
            "vector_recall@5": float(np.mean(r5_v)),
            "delta_recall@5": float(np.mean(r5_g - r5_v)),
        }
    metrics_summary["strata"] = dict(strata)

    # Confirmatory H1 sub-claim contrast:
    # Relational chain (compositional + inference) vs Comparison
    chain_qids = [
        qid for qid in question_ids if results_graph[qid]["type"] in ("compositional", "inference")
    ]
    comp_qids = [
        qid for qid in question_ids if results_graph[qid]["type"] == "comparison"
    ]

    delta_chain = float(
        np.mean([results_graph[qid]["recall@5"] - results_vector[qid]["recall@5"] for qid in chain_qids])
    )
    delta_comp = float(
        np.mean([results_graph[qid]["recall@5"] - results_vector[qid]["recall@5"] for qid in comp_qids])
    )
    metrics_summary["confirmatory_contrast"] = {
        "delta_chain": delta_chain,
        "delta_comp": delta_comp,
        "contrast_gain": delta_chain - delta_comp,
    }

    return metrics_summary


def evaluate_reader_metrics(reader_results: Dict[str, dict]) -> Dict[str, dict]:
    """Computes Exact Match and F1 scores across the 4 experimental reader arms."""
    qids = sorted(list(reader_results.keys()))
    arms = ["vector", "graph", "oracle", "closed"]
    summary: Dict[str, dict] = {}

    for arm in arms:
        ems = []
        f1s = []
        for qid in qids:
            res = reader_results[qid]
            gold = res["gold_answer"]
            pred = res[f"pred_{arm}"]
            ems.append(compute_exact_match(pred, gold))
            f1s.append(compute_token_f1(pred, gold))

        summary[arm] = {
            "exact_match": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
            "ems": np.array(ems),
            "f1s": np.array(f1s),
        }

    return summary

