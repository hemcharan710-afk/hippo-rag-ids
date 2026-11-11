"""Entity Graph Construction via LLM OpenIE Triple Extraction.

Course: CS 328 — Introduction to Data Science (Lecture 1-2: Data as Graphs).
Builds an undirected NetworkX knowledge graph from passages with:
1. LLM triple extraction & coreference/pronoun resolution to canonical names.
2. Intra-passage grounding: connects passage title to all internal entities.
3. Strict self-loop avoidance (u != v).
4. Regex noun-phrase fallback for empty extractions.
5. Batch extraction (8 passages/call) + per-passage persistent caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from tqdm import tqdm

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.data_loader import DatasetSplit, Passage, normalize_text
from src.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

GRAPH_CACHE_DIR = config.CACHE_DIR / "graphs"
GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRIPLES_CACHE_DIR = config.CACHE_DIR / "triples"
TRIPLES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10
MIN_CALL_INTERVAL = 4.0  # Safe 15 RPM margin for Gemini free tier


def _passage_cache_path(passage_id: str) -> Path:
    h = hashlib.sha256(f"triple_v2_{passage_id}".encode("utf-8")).hexdigest()[:16]
    return TRIPLES_CACHE_DIR / f"{h}.json"


def extract_fallback_entities(text: str, title: str) -> List[str]:
    """Regex fallback to extract capitalized noun phrases if LLM returns 0 triples."""
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    entities = set()
    norm_title = normalize_text(title)
    for m in matches:
        norm_m = normalize_text(m)
        if norm_m and norm_m != norm_title and len(norm_m) > 2:
            entities.add(norm_m)
    return list(entities)


class GraphBuilder:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self._last_call_time = 0.0

    def _rate_limited_query(self, prompt: str, system_prompt: str = "") -> str:
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        res = self.llm.query(prompt, system_prompt, json_mode=True, temperature=0.0)
        self._last_call_time = time.time()
        return res

    def extract_triples_batch(self, passages: List[Passage]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Extracts triples for a batch of passages in a single structured LLM call."""
        # Check cache first
        results: Dict[str, List[Tuple[str, str, str]]] = {}
        uncached: List[Passage] = []

        for p in passages:
            cpath = _passage_cache_path(p.id)
            if cpath.exists():
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                        results[p.id] = [tuple(t) for t in saved["triples"]]
                except Exception:
                    uncached.append(p)
            else:
                uncached.append(p)

        if not uncached:
            return results

        # Build batched prompt
        prompt = (
            "Extract all factual (subject, relation, object) triples for each passage below.\n"
            "Requirements:\n"
            "- Resolve all pronouns (he, she, it, they, the film, the city, etc.) to canonical entity names.\n"
            "- Output valid JSON matching this schema:\n"
            '{"results": [{"id": "<passage_id>", "triples": [["subject", "relation", "object"]]}]}\n\n'
        )
        for p in uncached:
            prompt += f"Passage ID: {p.id}\nTitle: {p.title}\nText: {p.text}\n---\n"

        try:
            response_text = self._rate_limited_query(prompt)
            data = json.loads(response_text)
            returned_results = data.get("results", [])
            for item in returned_results:
                pid = item.get("id", "")
                raw_triples = item.get("triples", [])
                parsed = []
                for t in raw_triples:
                    if isinstance(t, list) and len(t) >= 3:
                        s = normalize_text(str(t[0]))
                        r = normalize_text(str(t[1]))
                        o = normalize_text(str(t[2]))
                        if s and o and s != o:
                            parsed.append((s, r, o))
                if pid:
                    results[pid] = parsed
                    # Save to cache
                    with open(_passage_cache_path(pid), "w", encoding="utf-8") as f:
                        json.dump({"passage_id": pid, "triples": parsed}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}. Falling back to individual processing.")
            for p in uncached:
                if p.id not in results:
                    fb = extract_fallback_entities(p.text, p.title)
                    results[p.id] = [(p.id, "mentions", ent) for ent in fb]

        # Ensure all uncached passages have an entry
        for p in uncached:
            if p.id not in results:
                fb = extract_fallback_entities(p.text, p.title)
                results[p.id] = [(p.id, "mentions", ent) for ent in fb]
                with open(_passage_cache_path(p.id), "w", encoding="utf-8") as f:
                    json.dump({"passage_id": p.id, "triples": results[p.id]}, f, ensure_ascii=False)

        return results

    def build_graph(
        self,
        split: DatasetSplit,
        cache_name: str = "eval",
        max_passages: Optional[int] = None,
    ) -> Tuple[nx.Graph, Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Builds an undirected entity graph from all passages in the split corpus."""
        cache_file = GRAPH_CACHE_DIR / f"{cache_name}_graph.json"

        passages = list(split.corpus.values())
        if max_passages is not None:
            passages = passages[:max_passages]

        # Check for full cached graph
        if cache_file.exists() and max_passages is None:
            logger.info(f"Loading cached graph from {cache_file}...")
            with open(cache_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            G = nx.node_link_graph(saved["graph"], edges="links")
            entity_to_passages = {k: set(v) for k, v in saved["entity_to_passages"].items()}
            passage_to_entities = {k: set(v) for k, v in saved["passage_to_entities"].items()}
            logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G, entity_to_passages, passage_to_entities

        G = nx.Graph()
        entity_to_passages: Dict[str, Set[str]] = defaultdict(set)
        passage_to_entities: Dict[str, Set[str]] = defaultdict(set)

        # Batch extract triples
        all_triples: Dict[str, List[Tuple[str, str, str]]] = {}
        logger.info(f"Extracting triples in batches of {BATCH_SIZE} for {len(passages)} passages...")

        for i in tqdm(range(0, len(passages), BATCH_SIZE), desc="Extracting Triples"):
            batch = passages[i : i + BATCH_SIZE]
            batch_triples = self.extract_triples_batch(batch)
            all_triples.update(batch_triples)

        logger.info("Constructing NetworkX graph structure...")
        for passage in passages:
            title_node = passage.id
            G.add_node(title_node, type="passage_title", title=passage.title)
            entity_to_passages[title_node].add(passage.id)
            passage_to_entities[passage.id].add(title_node)

            triples = all_triples.get(passage.id, [])
            extracted_entities: Set[str] = set()

            for sub, rel, obj in triples:
                extracted_entities.add(sub)
                extracted_entities.add(obj)
                G.add_node(sub, type="entity")
                G.add_node(obj, type="entity")
                G.add_edge(sub, obj, relation=rel)

            # Intra-passage grounding: connect passage title to all entities in passage
            for entity in extracted_entities:
                if entity != title_node:
                    G.add_edge(title_node, entity, relation="passage_mention")
                    entity_to_passages[entity].add(passage.id)
                    passage_to_entities[passage.id].add(entity)

        logger.info(
            f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{nx.number_connected_components(G)} connected components."
        )

        if max_passages is None:
            data_to_save = {
                "graph": nx.node_link_data(G, edges="links"),
                "entity_to_passages": {k: sorted(list(v)) for k, v in entity_to_passages.items()},
                "passage_to_entities": {k: sorted(list(v)) for k, v in passage_to_entities.items()},
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False)
            logger.info(f"Saved graph to {cache_file}")

        return G, entity_to_passages, passage_to_entities
