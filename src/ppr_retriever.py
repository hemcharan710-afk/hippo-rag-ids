"""Personalized PageRank (PPR) Retrieval on the Entity Knowledge Graph.

Course: CS 328 — Introduction to Data Science
Lectures:
- PageRank Lecture: Teleporting random walks, power iteration, stationary distribution pi
- Lecture 10: Eigenvalues, dominant eigenvectors, power iteration convergence
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.baseline_retriever import DenseBaselineRetriever
from src.data_loader import DatasetSplit
from src.entity_linker import EntityLinker

logger = logging.getLogger(__name__)


class PPRRetriever:
    """Personalized PageRank multi-hop retrieval with specificity passage scoring."""

    def __init__(
        self,
        graph: nx.Graph,
        entity_to_passages: Dict[str, Set[str]],
        passage_to_entities: Dict[str, Set[str]],
        entity_linker: EntityLinker,
        dense_retriever: Optional[DenseBaselineRetriever] = None,
        alpha: float = config.ALPHA_PRIMARY,
    ):
        self.G = graph
        self.entity_to_passages = entity_to_passages
        self.passage_to_entities = passage_to_entities
        self.linker = entity_linker
        self.dense_retriever = dense_retriever
        self.alpha = alpha

        # Compute document frequency df(e) for every entity
        self.df: Dict[str, int] = {
            e: max(len(pids), 1) for e, pids in entity_to_passages.items()
        }
        logger.info(
            f"Initialized PPRRetriever (alpha={self.alpha}, tol={config.PPR_TOL}, nodes={self.G.number_of_nodes()})"
        )

    def run_ppr(self, teleport_dict: Dict[str, float], alpha: Optional[float] = None) -> Dict[str, float]:
        """Runs power-iteration Personalized PageRank with teleport vector v_S."""
        damping = self.alpha if alpha is None else alpha

        # Arm 3: No-Walk Control (alpha=0.0): mass stays exactly on seed nodes
        if damping <= 1e-6:
            return {node: prob for node, prob in teleport_dict.items()}

        # Filter teleport dict to nodes existing in G
        valid_teleport = {n: w for n, w in teleport_dict.items() if self.G.has_node(n)}
        if not valid_teleport:
            return {}

        total_w = sum(valid_teleport.values())
        norm_teleport = {n: w / total_w for n, w in valid_teleport.items()}

        try:
            ppr_scores = nx.pagerank(
                self.G,
                alpha=damping,
                personalization=norm_teleport,
                nstart=norm_teleport,
                tol=config.PPR_TOL,
                max_iter=config.PPR_MAX_ITER,
            )
        except Exception as e:
            logger.warning(f"nx.pagerank failed: {e}. Falling back to teleport seeds.")
            ppr_scores = norm_teleport

        return ppr_scores

    def score_passages(
        self, ppr_scores: Dict[str, float], all_passage_ids: List[str]
    ) -> Dict[str, float]:
        """Passage Scoring (Formulation A): Specificity-weighted sum Score(P) = sum_{e in P} PPR(e) / df(e)."""
        passage_scores: Dict[str, float] = defaultdict(float)

        for entity, score in ppr_scores.items():
            if score <= 0.0:
                continue
            passages = self.entity_to_passages.get(entity, set())
            weight = score / self.df.get(entity, 1)
            for pid in passages:
                passage_scores[pid] += weight

        # Ensure all corpus passages exist in dict
        for pid in all_passage_ids:
            if pid not in passage_scores:
                passage_scores[pid] = 0.0

        return dict(passage_scores)

    def retrieve(
        self,
        question: str,
        all_passage_ids: List[str],
        dense_scores: Optional[Dict[str, float]] = None,
        top_k: int = config.K_PRIMARY,
        alpha: Optional[float] = None,
        cached_teleport: Optional[Tuple[Dict[str, float], bool]] = None,
    ) -> Tuple[List[str], Dict[str, float], bool, Tuple[Dict[str, float], bool]]:
        """Retrieves top_k passages via PPR with lexicographical dense tie-breaking."""
        if cached_teleport is not None:
            teleport_dict, fallback_used = cached_teleport
        else:
            teleport_dict, fallback_used = self.linker.link_query(question)

        ppr_node_scores = self.run_ppr(teleport_dict, alpha=alpha)
        passage_ppr_scores = self.score_passages(ppr_node_scores, all_passage_ids)

        if dense_scores is None:
            dense_scores = {pid: 0.0 for pid in all_passage_ids}

        # Lexicographical sort: (Score_PPR, Score_Dense)
        # Transparent tie-breaking with zero artificial distortion
        ranked_passages = sorted(
            all_passage_ids,
            key=lambda pid: (passage_ppr_scores.get(pid, 0.0), dense_scores.get(pid, 0.0)),
            reverse=True,
        )

        top_retrieved = ranked_passages[:top_k]
        top_scores = {pid: passage_ppr_scores.get(pid, 0.0) for pid in top_retrieved}

        return top_retrieved, top_scores, fallback_used, (teleport_dict, fallback_used)

    def evaluate_split(
        self,
        split: DatasetSplit,
        dense_results: Optional[Dict[str, dict]] = None,
        k_values: Tuple[int, ...] = config.K_VALUES,
        alpha: Optional[float] = None,
        teleport_cache: Optional[Dict[str, Tuple[Dict[str, float], bool]]] = None,
    ) -> Tuple[Dict[str, dict], Dict[str, Tuple[Dict[str, float], bool]]]:
        """Evaluates PPR retrieval across all questions in the split."""
        all_pids = sorted(list(split.corpus.keys()))
        results: Dict[str, dict] = {}
        out_teleport_cache: Dict[str, Tuple[Dict[str, float], bool]] = (
            dict(teleport_cache) if teleport_cache is not None else {}
        )
        fallback_count = 0
        max_k = max(k_values)

        target_alpha = self.alpha if alpha is None else alpha
        arm_label = f"alpha={target_alpha}" if target_alpha > 0 else "No-Walk Control (alpha=0)"
        logger.info(f"Evaluating {len(split.questions)} questions on Graph + PPR ({arm_label})...")

        for q in split.questions:
            dense_scores = None
            if dense_results and q.id in dense_results:
                dense_scores = dense_results[q.id].get("scores", None)

            cached_tp = out_teleport_cache.get(q.id)
            retrieved_max, scores_max, fb_used, tp_info = self.retrieve(
                question=q.question,
                all_passage_ids=all_pids,
                dense_scores=dense_scores,
                top_k=max_k,
                alpha=target_alpha,
                cached_teleport=cached_tp,
            )
            out_teleport_cache[q.id] = tp_info

            if fb_used:
                fallback_count += 1

            gold = split.ground_truth[q.id]
            q_result = {
                "question_id": q.id,
                "type": q.type,
                "question": q.question,
                "retrieved": retrieved_max,
                "scores": scores_max,
                "gold": sorted(list(gold)),
                "fallback_used": fb_used,
            }

            for k in k_values:
                top_k_ids = set(retrieved_max[:k])
                hits = len(top_k_ids.intersection(gold))
                q_result[f"recall@{k}"] = hits / len(gold) if gold else 0.0
                q_result[f"all_recall@{k}"] = 1.0 if gold.issubset(top_k_ids) else 0.0

            results[q.id] = q_result

        fb_pct = fallback_count / len(split.questions) * 100
        logger.info(f"Evaluation finished for {arm_label}. Fallback triggered on {fallback_count}/{len(split.questions)} ({fb_pct:.1f}%) queries.")
        return results, out_teleport_cache

