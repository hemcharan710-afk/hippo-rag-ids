"""Entity Linking and Intra-Graph Synonym Augmentation via Hyperplane LSH.

Course: CS 328 — Introduction to Data Science
Lectures:
- Lectures 6-7: Hashing, MinHash, Banding, S-curves (inflection p* = (1/b)^(1/r) = 0.7071)
- Lecture 11: Random Hyperplane LSH (angle-to-probability mapping: p = 1 - theta / pi)
"""

from __future__ import annotations

import json
import logging
import re
import time
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
from src.embeddings import EmbeddingModel, get_embedder
from src.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

QUERY_NER_SYSTEM_PROMPT = (
    "Extract the key named entities (people, places, works, organizations, concepts) "
    "from this question. Output ONLY a valid JSON list of lowercase strings."
)
QUERY_NER_PROMPT_TEMPLATE = """Question: {question}
Return ONLY a valid JSON array of entity strings, e.g.: ["entity 1", "entity 2"]
"""


def is_number_or_date(text: str) -> bool:
    """Detects if string is purely numeric, a year, or simple date to prevent false synonym edges."""
    clean = text.strip()
    if re.fullmatch(r"\d+", clean):
        return True
    if re.fullmatch(r"\d{1,2}\s+[a-z]+\s+\d{4}", clean.lower()):
        return True
    if re.fullmatch(r"[a-z]+\s+\d{1,2},?\s+\d{4}", clean.lower()):
        return True
    return False


class HyperplaneLSH:
    """NumPy-based Random Hyperplane LSH with Signature Banding (Lectures 6-7 & 11)."""

    def __init__(
        self,
        l: int = config.LSH_L,
        r: int = config.LSH_R,
        b: int = config.LSH_B,
        seed: int = config.SEED_LSH,
    ):
        assert l == r * b, f"Total hyperplanes {l} must equal r * b ({r} * {b})"
        self.l = l
        self.r = r
        self.b = b
        self.seed = seed
        self.hyperplanes: Optional[np.ndarray] = None
        self.band_buckets: List[Dict[Tuple[int, ...], List[int]]] = []
        self.entity_list: List[str] = []
        self.entity_embeddings: Optional[np.ndarray] = None

    def fit(self, entities: List[str], embeddings: np.ndarray) -> None:
        """Draws random hyperplanes and indexes entities into signature bands."""
        self.entity_list = entities
        self.entity_embeddings = embeddings
        d = embeddings.shape[1]

        rng = np.random.default_rng(self.seed)
        # Random normal hyperplanes: (l, d)
        self.hyperplanes = rng.standard_normal((self.l, d)).astype(np.float32)

        # Bit signatures: (N, l) binary array
        signatures = (embeddings @ self.hyperplanes.T > 0).astype(np.uint8)

        # Signature banding tables
        self.band_buckets = [defaultdict(list) for _ in range(self.b)]
        for idx in range(len(entities)):
            sig = signatures[idx]
            for band_idx in range(self.b):
                band_bits = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
                self.band_buckets[band_idx][band_bits].append(idx)

        logger.info(
            f"Hyperplane LSH indexed {len(entities)} entities across {self.b} bands (r={self.r}, l={self.l})."
        )

    def get_candidates(self, query_signature: np.ndarray) -> Set[int]:
        """Finds entity candidate indices matching across any of the b bands."""
        candidates: Set[int] = set()
        for band_idx in range(self.b):
            band_bits = tuple(query_signature[band_idx * self.r : (band_idx + 1) * self.r])
            bucket = self.band_buckets[band_idx].get(band_bits, [])
            candidates.update(bucket)
        return candidates

    def find_synonym_pairs(
        self, threshold: float = config.TAU_SYNONYM
    ) -> List[Tuple[str, str, float]]:
        """Finds intra-graph synonym pairs using LSH candidates with exact cosine verification."""
        if self.hyperplanes is None or self.entity_embeddings is None:
            raise RuntimeError("LSH index is not fit yet.")

        candidate_pairs: Set[Tuple[int, int]] = set()
        for band_idx in range(self.b):
            for bucket in self.band_buckets[band_idx].values():
                if len(bucket) > 1 and len(bucket) < 500:  # Ignore giant degenerate buckets
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            u, v = bucket[i], bucket[j]
                            if u > v:
                                u, v = v, u
                            candidate_pairs.add((u, v))

        logger.info(f"Evaluating {len(candidate_pairs)} candidate pairs from LSH buckets...")
        synonym_edges: List[Tuple[str, str, float]] = []

        for u_idx, v_idx in candidate_pairs:
            u_name = self.entity_list[u_idx]
            v_name = self.entity_list[v_idx]

            # Safeguards: ignore strings <= 2 chars, identical names, or pure numbers/dates
            if len(u_name) <= 2 or len(v_name) <= 2 or u_name == v_name:
                continue
            if is_number_or_date(u_name) or is_number_or_date(v_name):
                continue

            # Exact cosine verification
            cos_sim = float(np.dot(self.entity_embeddings[u_idx], self.entity_embeddings[v_idx]))
            if cos_sim >= threshold:
                synonym_edges.append((u_name, v_name, cos_sim))

        logger.info(f"Discovered {len(synonym_edges)} intra-graph synonym edges with cosine >= {threshold:.2f}")
        return synonym_edges


QUERY_CACHE_FILE = config.CACHE_DIR / "query_entities.json"


class EntityLinker:
    """Extracts query entities and links them to graph nodes using LSH + Softmax weights."""

    def __init__(
        self,
        graph: nx.Graph,
        embedder: Optional[EmbeddingModel] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.G = graph
        self.embedder = embedder or get_embedder()
        self.llm = llm_client or get_llm_client()
        self.lsh = HyperplaneLSH()
        self._last_call_time = 0.0

        # Load query entities cache
        self.query_cache: Dict[str, List[str]] = {}
        if QUERY_CACHE_FILE.exists():
            try:
                with open(QUERY_CACHE_FILE, "r", encoding="utf-8") as f:
                    self.query_cache = json.load(f)
            except Exception:
                pass

        # Build LSH index over all graph nodes
        self.nodes = sorted(list(self.G.nodes()))
        logger.info(f"Embedding {len(self.nodes)} graph nodes for EntityLinker LSH...")
        self.node_embeddings = self.embedder.encode(self.nodes, show_progress_bar=False)
        self.lsh.fit(self.nodes, self.node_embeddings)
        self.node_to_idx = {name: idx for idx, name in enumerate(self.nodes)}

    def augment_graph_with_synonyms(self, threshold: float = config.TAU_SYNONYM) -> int:
        """Finds synonym pairs via LSH and adds them as edges to graph G."""
        pairs = self.lsh.find_synonym_pairs(threshold=threshold)
        added = 0
        for u, v, sim in pairs:
            if not self.G.has_edge(u, v):
                self.G.add_edge(u, v, relation="synonym", weight=sim)
                added += 1
        logger.info(f"Added {added} new synonym edges to graph G.")
        return added

    def precompute_query_entities(self, questions: List[str], batch_size: int = 10) -> None:
        """Batch-extracts named entities for questions in groups of 10 to minimize latency and API calls."""
        uncached = [q.strip() for q in questions if q.strip() not in self.query_cache]
        if not uncached:
            logger.info("All query entities already cached.")
            return

        logger.info(f"Precomputing query entities for {len(uncached)} uncached questions (batches of {batch_size})...")
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            prompt = (
                "Extract the key named entities (people, places, works, organizations) from each question below.\n"
                "Return ONLY a valid JSON object matching this schema:\n"
                '{"results": [{"idx": 0, "entities": ["entity 1", "entity 2"]}]}\n\n'
            )
            for b_idx, q in enumerate(batch):
                prompt += f"Question {b_idx}: {q}\n"

            # Rate limiting
            elapsed = time.time() - self._last_call_time
            if elapsed < 4.0:
                time.sleep(4.0 - elapsed)

            try:
                res = self.llm.query(prompt, QUERY_NER_SYSTEM_PROMPT, json_mode=True, temperature=0.0)
                self._last_call_time = time.time()
                data = json.loads(res)
                res_list = data.get("results", [])
                for item in res_list:
                    idx = int(item.get("idx", -1))
                    ents = [str(e).strip().lower() for e in item.get("entities", []) if e]
                    if 0 <= idx < len(batch):
                        self.query_cache[batch[idx]] = ents
            except Exception as e:
                logger.warning(f"Batch query NER failed: {e}. Falling back to regex.")
                self._last_call_time = time.time()

            # Ensure all in batch are assigned
            for q in batch:
                if q not in self.query_cache:
                    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", q)
                    self.query_cache[q] = [m.lower() for m in matches]

            # Save incrementally
            try:
                with open(QUERY_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.query_cache, f, ensure_ascii=False)
            except Exception:
                pass
        logger.info(f"Finished precomputing query entities ({len(self.query_cache)} total cached).")

    def extract_query_entities(self, question: str) -> List[str]:
        """Extracts named entities from question text using LLM with local caching and regex fallback."""
        norm_q = question.strip()
        if norm_q in self.query_cache:
            return self.query_cache[norm_q]

        # Rate limit throttling (safe 4.0s interval between query NER calls)
        elapsed = time.time() - self._last_call_time
        if elapsed < 4.0:
            time.sleep(4.0 - elapsed)

        prompt = QUERY_NER_PROMPT_TEMPLATE.format(question=question)
        entities = []
        try:
            res = self.llm.query(prompt, QUERY_NER_SYSTEM_PROMPT, json_mode=True, temperature=0.0)
            self._last_call_time = time.time()
            parsed = json.loads(res)
            if isinstance(parsed, list):
                entities = [str(e).strip().lower() for e in parsed if e]
            elif isinstance(parsed, dict) and "entities" in parsed:
                entities = [str(e).strip().lower() for e in parsed["entities"] if e]
        except Exception as e:
            logger.warning(f"Query entity extraction LLM call failed for '{question[:50]}...': {e}. Using regex fallback.")
            self._last_call_time = time.time()

        if not entities:
            # Fallback regex for capitalized noun phrases
            matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question)
            entities = [m.lower() for m in matches]

        self.query_cache[norm_q] = entities
        try:
            with open(QUERY_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.query_cache, f, ensure_ascii=False)
        except Exception:
            pass

        return entities

    def link_query(
        self, question: str
    ) -> Tuple[Dict[str, float], bool]:
        """Links query entities to graph nodes with temperature-sharpened softmax weights.

        Returns:
            teleport_dict: Dict[node_name, probability] summing to 1.0
            fallback_triggered: True if full question embedding fallback was used
        """
        entities = self.extract_query_entities(question)
        seed_matches: Dict[str, float] = {}  # node -> cosine similarity

        if entities:
            ent_embeddings = self.embedder.encode(entities, show_progress_bar=False)
            ent_sigs = (ent_embeddings @ self.lsh.hyperplanes.T > 0).astype(np.uint8)

            for i, ent_name in enumerate(entities):
                # Exact match check first
                if ent_name in self.node_to_idx:
                    seed_matches[ent_name] = max(seed_matches.get(ent_name, 0.0), 1.0)
                    continue

                candidates = self.lsh.get_candidates(ent_sigs[i])
                if not candidates:
                    continue

                cand_indices = list(candidates)
                cand_sims = (self.node_embeddings[cand_indices] @ ent_embeddings[i])

                # Check >= 0.80 matches
                top_matches = []
                for c_idx, sim in zip(cand_indices, cand_sims):
                    if sim >= config.QUERY_MATCH_THRESHOLD:
                        top_matches.append((self.nodes[c_idx], float(sim)))

                if top_matches:
                    top_matches.sort(key=lambda x: x[1], reverse=True)
                    for node, sim in top_matches[: config.MAX_CANDIDATES_PER_ENTITY]:
                        seed_matches[node] = max(seed_matches.get(node, 0.0), sim)
                else:
                    # Check floor >= 0.40
                    best_local_idx = int(np.argmax(cand_sims))
                    best_sim = float(cand_sims[best_local_idx])
                    if best_sim >= config.QUERY_MATCH_FLOOR:
                        best_node = self.nodes[cand_indices[best_local_idx]]
                        seed_matches[best_node] = max(seed_matches.get(best_node, 0.0), best_sim)

        fallback_triggered = False
        # Fallback: if no seeds meet floor, embed full question string
        if not seed_matches:
            fallback_triggered = True
            q_emb = self.embedder.encode([question], show_progress_bar=False)[0]
            # Exact cosine against all graph nodes
            all_sims = self.node_embeddings @ q_emb
            top_k_indices = np.argsort(all_sims)[::-1][: config.MAX_CANDIDATES_PER_ENTITY]
            for idx in top_k_indices:
                sim = float(all_sims[idx])
                if sim >= config.QUERY_FALLBACK_FLOOR:
                    seed_matches[self.nodes[idx]] = sim

        # If still empty (e.g. all nodes < 0.35), link to top-1 node unconditionally
        if not seed_matches:
            fallback_triggered = True
            q_emb = self.embedder.encode([question], show_progress_bar=False)[0]
            all_sims = self.node_embeddings @ q_emb
            best_idx = int(np.argmax(all_sims))
            seed_matches[self.nodes[best_idx]] = float(all_sims[best_idx])

        # Temperature-sharpened softmax: w_i \propto \exp(sim_i / T)
        T = config.QUERY_SEED_TEMPERATURE
        nodes = list(seed_matches.keys())
        sims = np.array([seed_matches[n] for n in nodes])
        # Subtract max for numerical stability
        exp_weights = np.exp((sims - np.max(sims)) / T)
        probs = exp_weights / np.sum(exp_weights)

        teleport_dict = {nodes[i]: float(probs[i]) for i in range(len(nodes))}
        return teleport_dict, fallback_triggered

