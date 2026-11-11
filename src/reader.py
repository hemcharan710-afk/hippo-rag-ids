"""LLM Reader and Grounding Bounds (RQ2: Exact Match & F1).

Course: CS 328 — Introduction to Data Science
Evaluates question answering across 4 experimental arms in a unified structured query:
1. Dense Vector Reader Arm (Top-5 vector passages)
2. Graph + PPR Reader Arm (Top-5 graph passages)
3. Oracle Upper Bound (Gold supporting passages)
4. Closed-Book Lower Bound (Parametric memory floor, no passages)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.data_loader import DatasetSplit, Passage
from src.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

UNIFIED_READER_SYSTEM_PROMPT = (
    "You are an exact reading comprehension system. You will receive a multi-hop question and three contexts: "
    "Context 1 (Vector Top-5), Context 2 (Graph Top-5), Context 3 (Oracle Gold). "
    "For each context, output ONLY the short canonical answer (an entity name, year, number, or 'yes'/'no'). "
    "Also output the answer without any context (closed-book parametric memory floor). "
    "Never write full sentences, explanations, or conversational filler. Output valid JSON."
)

UNIFIED_READER_USER_PROMPT = """Question: {question}

--- CONTEXT 1 (Vector Baseline Top-5) ---
{context_vector}

--- CONTEXT 2 (Graph + PPR Top-5) ---
{context_graph}

--- CONTEXT 3 (Oracle Gold Passages) ---
{context_oracle}

Output ONLY valid JSON in this exact schema:
{{"pred_vector": "<short answer>", "pred_graph": "<short answer>", "pred_oracle": "<short answer>", "pred_closed": "<short answer>"}}
"""


class LLMReader:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self._last_call_time = 0.0

    def format_passages(self, passage_ids: List[str], corpus: Dict[str, Passage]) -> str:
        """Formats passages with unified symmetry for the reader context."""
        parts = []
        for i, pid in enumerate(passage_ids, 1):
            passage = corpus.get(pid)
            if passage:
                parts.append(f"[{i}] {passage.full_text}")
        return "\n\n".join(parts) if parts else "(No passages available)"

    def run_reader_evaluations(
        self,
        split: DatasetSplit,
        vector_results: Dict[str, dict],
        graph_results: Dict[str, dict],
        k: int = config.K_PRIMARY,
        max_questions: Optional[int] = None,
    ) -> Dict[str, dict]:
        """Runs the LLM Reader across all 4 experimental arms in a single unified prompt per question."""
        questions = split.questions
        if max_questions is not None:
            questions = questions[:max_questions]

        reader_results: Dict[str, dict] = {}
        logger.info(f"Running LLM Reader evaluations across 4 arms for {len(questions)} questions...")

        for q in tqdm(questions, desc="LLM Reader Evaluation"):
            gold = split.ground_truth[q.id]
            vector_top_k = vector_results[q.id]["retrieved"][:k]
            graph_top_k = graph_results[q.id]["retrieved"][:k]
            oracle_passages = list(gold)

            ctx_vector = self.format_passages(vector_top_k, split.corpus)
            ctx_graph = self.format_passages(graph_top_k, split.corpus)
            ctx_oracle = self.format_passages(oracle_passages, split.corpus)

            prompt = UNIFIED_READER_USER_PROMPT.format(
                question=q.question,
                context_vector=ctx_vector,
                context_graph=ctx_graph,
                context_oracle=ctx_oracle,
            )

            # Safe rate limit pause
            elapsed = time.time() - self._last_call_time
            if elapsed < 4.2:
                time.sleep(4.2 - elapsed)

            ans_vector = ""
            ans_graph = ""
            ans_oracle = ""
            ans_closed = ""

            try:
                response_text = self.llm.query(
                    prompt=prompt,
                    system_prompt=UNIFIED_READER_SYSTEM_PROMPT,
                    json_mode=True,
                    temperature=0.0,
                )
                self._last_call_time = time.time()
                data = json.loads(response_text)
                ans_vector = str(data.get("pred_vector", "")).strip()
                ans_graph = str(data.get("pred_graph", "")).strip()
                ans_oracle = str(data.get("pred_oracle", "")).strip()
                ans_closed = str(data.get("pred_closed", "")).strip()
            except Exception as e:
                logger.warning(f"Unified reader failed for question {q.id}: {e}")
                self._last_call_time = time.time()

            reader_results[q.id] = {
                "question_id": q.id,
                "type": q.type,
                "question": q.question,
                "gold_answer": q.answer,
                "pred_vector": ans_vector,
                "pred_graph": ans_graph,
                "pred_oracle": ans_oracle,
                "pred_closed": ans_closed,
            }

        return reader_results
