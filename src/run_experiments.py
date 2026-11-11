"""End-to-End Experiment Execution Pipeline.

Course: CS 328 — Introduction to Data Science
Orchestrates:
- Phase 2: Dense Vector Baseline (FAISS IndexFlatIP)
- Phase 3: Entity Graph Construction
- Phase 4: LSH Intra-Graph Synonyms & Query Entity Linking
- Phase 5: Personalized PageRank (alpha=0.5), No-Walk Control (alpha=0.0), and Sensitivity (alpha=0.85)
- Phase 6: Downstream LLM Reader (Vector, Graph, Oracle, Closed-Book)
- Phase 7: Evaluation Metrics & Pre-Specified Contrasts
- Phase 8: Statistical Hypothesis Tests & Pre-Registered Decision Protocol
- Phase 10: Publication Figures & Results Summaries
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.baseline_retriever import DenseBaselineRetriever
from src.data_loader import DatasetSplit, get_or_create_splits
from src.entity_linker import EntityLinker
from src.evaluation import (
    compute_exact_match,
    compute_token_f1,
    evaluate_reader_metrics,
    evaluate_retrieval_metrics,
)
from src.graph_builder import GraphBuilder
from src.ppr_retriever import PPRRetriever
from src.reader import LLMReader
from src.statistical_tests import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    evaluate_preregistered_verdict,
    holm_bonferroni,
    paired_mcnemar_test,
    paired_sign_flip_permutation_test,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_full_pipeline(max_reader_questions: Optional[int] = 50) -> Dict[str, dict]:
    """Executes the complete experimental pipeline and saves all results."""
    print("=" * 80)
    print("CS 328 RESEARCH STUDY: GRAPH-BASED RETRIEVAL VS DENSE VECTOR SEARCH")
    print("=" * 80)

    # 1. Load Data
    eval_split, tune_split = get_or_create_splits()
    print(f"\n[Phase 1] Loaded Evaluation Split: {len(eval_split.questions)} questions, {len(eval_split.corpus)} passages.")

    # 2. Phase 2: Dense Vector Baseline
    print("\n" + "-" * 60)
    print("[Phase 2] Running Dense Vector Baseline (FAISS IndexFlatIP)...")
    dense_retriever = DenseBaselineRetriever()
    dense_retriever.build_index(eval_split, cache_prefix="eval")
    dense_results = dense_retriever.evaluate_split(eval_split)

    # 3. Phase 3: Graph Construction
    print("\n" + "-" * 60)
    print("[Phase 3] Constructing / Loading Entity Knowledge Graph...")
    graph_builder = GraphBuilder()
    G, e2p, p2e = graph_builder.build_graph(eval_split, cache_name="eval")

    # 4. Phase 4: Entity Linker & Intra-Graph LSH Synonyms
    print("\n" + "-" * 60)
    print("[Phase 4] Fitting Hyperplane LSH & Linking Entities...")
    linker = EntityLinker(graph=G, embedder=dense_retriever.embedder)
    synonyms_added = linker.augment_graph_with_synonyms(threshold=config.TAU_SYNONYM)
    print(f"Intra-graph synonym augmentation: added {synonyms_added} edges (tau={config.TAU_SYNONYM}).")
    linker.precompute_query_entities([q.question for q in eval_split.questions], batch_size=10)

    # 5. Phase 5: Personalized PageRank Retrieval
    print("\n" + "-" * 60)
    print("[Phase 5] Running Personalized PageRank Retrieval Arms...")
    ppr_retriever = PPRRetriever(
        graph=G,
        entity_to_passages=e2p,
        passage_to_entities=p2e,
        entity_linker=linker,
        dense_retriever=dense_retriever,
        alpha=config.ALPHA_PRIMARY,
    )

    # Arm 2: Primary Graph + PPR (alpha=0.5)
    graph_results, teleport_cache = ppr_retriever.evaluate_split(
        split=eval_split,
        dense_results=dense_results,
        alpha=config.ALPHA_PRIMARY,
    )

    # Arm 3: No-Walk Control Arm (Ablation: alpha=0.0)
    print("\nRunning Arm 3: No-Walk Control (Ablation: alpha=0.0)...")
    nowalk_results, _ = ppr_retriever.evaluate_split(
        split=eval_split,
        dense_results=dense_results,
        alpha=config.ALPHA_NO_WALK,
        teleport_cache=teleport_cache,
    )

    # Sensitivity Check: alpha=0.85
    print("\nRunning Sensitivity Check: alpha=0.85...")
    sens_results, _ = ppr_retriever.evaluate_split(
        split=eval_split,
        dense_results=dense_results,
        alpha=config.ALPHA_SENSITIVITY,
        teleport_cache=teleport_cache,
    )

    # 6. Phase 6: Downstream LLM Reader (RQ2)
    print("\n" + "-" * 60)
    print(f"[Phase 6] Running LLM Reader Across 4 Arms (Evaluating {max_reader_questions or len(eval_split.questions)} questions)...")
    reader = LLMReader()
    reader_results = reader.run_reader_evaluations(
        split=eval_split,
        vector_results=dense_results,
        graph_results=graph_results,
        k=config.K_PRIMARY,
        max_questions=max_reader_questions,
    )
    reader_metrics = evaluate_reader_metrics(reader_results)

    # 7. Phase 7: Evaluation Metrics & Contrasts
    print("\n" + "-" * 60)
    print("[Phase 7] Computing Comparative Retrieval Metrics & Hypothesis Contrasts...")
    retrieval_metrics = evaluate_retrieval_metrics(graph_results, dense_results)

    # 8. Phase 8: Rigorous Statistical Tests & Pre-Registered Protocol
    print("\n" + "-" * 60)
    print("[Phase 8] Running Hypothesis Testing & Pre-Registered Protocol...")

    # Primary Metric: recall@5
    r5_graph = retrieval_metrics["recall@5"]["graph_scores"]
    r5_vector = retrieval_metrics["recall@5"]["vector_scores"]
    delta_r5 = retrieval_metrics["recall@5"]["delta_mean"]

    ci_r5_lower, ci_r5_upper = bootstrap_diff_ci(r5_graph, r5_vector)
    _, p_perm_r5 = paired_sign_flip_permutation_test(r5_graph, r5_vector)

    # Individual method CIs
    ci_graph_r5 = bootstrap_mean_ci(r5_graph)
    ci_vector_r5 = bootstrap_mean_ci(r5_vector)

    # Secondary Tests (for Holm-Bonferroni FWER control)
    r2_graph = retrieval_metrics["recall@2"]["graph_scores"]
    r2_vector = retrieval_metrics["recall@2"]["vector_scores"]
    _, p_perm_r2 = paired_sign_flip_permutation_test(r2_graph, r2_vector)

    ar5_graph = retrieval_metrics["all_recall@5"]["graph_scores"]
    ar5_vector = retrieval_metrics["all_recall@5"]["vector_scores"]
    _, p_mcnemar_ar5 = paired_mcnemar_test(ar5_graph, ar5_vector)

    # No-Walk Control test: Graph+PPR (alpha=0.5) vs Seeds Only (alpha=0.0)
    r5_nowalk = np.array([nowalk_results[q.id]["recall@5"] for q in eval_split.questions])
    _, p_perm_nowalk = paired_sign_flip_permutation_test(r5_graph, r5_nowalk)

    # Damping Sensitivity test: alpha=0.5 vs alpha=0.85
    r5_sens = np.array([sens_results[q.id]["recall@5"] for q in eval_split.questions])
    _, p_perm_sens = paired_sign_flip_permutation_test(r5_graph, r5_sens)

    # Secondary p-values dictionary
    secondary_p_vals = {
        "recall@2_gain": p_perm_r2,
        "all-recall@5_mcnemar": p_mcnemar_ar5,
        "no_walk_control": p_perm_nowalk,
        "damping_sensitivity": p_perm_sens,
    }
    holm_results = holm_bonferroni(secondary_p_vals, alpha=config.SIGNIFICANCE_ALPHA)

    # Pre-registered primary verdict
    verdict_text = evaluate_preregistered_verdict(
        delta_r5=delta_r5,
        ci_lower=ci_r5_lower,
        ci_upper=ci_r5_upper,
        perm_p=p_perm_r5,
    )

    # Print Report
    print("\n" + "=" * 80)
    print("EXPERIMENTAL SUMMARY & PRE-REGISTERED VERDICT")
    print("=" * 80)
    print(f"Vector Baseline recall@5: {np.mean(r5_vector)*100:.2f}% (95% CI: [{ci_vector_r5[0]*100:.2f}%, {ci_vector_r5[1]*100:.2f}%])")
    print(f"Graph + PPR     recall@5: {np.mean(r5_graph)*100:.2f}% (95% CI: [{ci_graph_r5[0]*100:.2f}%, {ci_graph_r5[1]*100:.2f}%])")
    print(f"Observed Delta (Gain):    {delta_r5*100:+.2f}%")
    print(f"95% Paired Bootstrap CI:  [{ci_r5_lower*100:.2f}%, {ci_r5_upper*100:.2f}%]")
    print(f"Sign-Flip Permutation p:  {p_perm_r5:.5f}")
    print("\n--- PRE-REGISTERED DECISION ---")
    print(verdict_text)

    print("\n--- SECONDARY HYPOTHESIS TESTS (Holm-Bonferroni Adjusted) ---")
    for name, res in holm_results.items():
        sig_str = "SIGNIFICANT" if res["significant"] else "NOT SIGNIFICANT"
        print(f"  - {name:<22}: Raw p = {res['raw_p']:.5f}, Threshold = {res['threshold']:.5f} -> {sig_str}")

    print("\n--- REASONING CHAIN CONTRAST (H1 Sub-Claim) ---")
    contrast = retrieval_metrics["confirmatory_contrast"]
    print(f"  - Relational Chains Gain (Delta_chain): {contrast['delta_chain']*100:+.2f}%")
    print(f"  - Direct Comparisons Gain (Delta_comp): {contrast['delta_comp']*100:+.2f}%")
    print(f"  - Contrast (Delta_chain - Delta_comp):  {contrast['contrast_gain']*100:+.2f}%")

    print("\n--- DOWNSTREAM QA ACCURACY (RQ2) ---")
    for arm in ("closed", "vector", "graph", "oracle"):
        em = reader_metrics[arm]["exact_match"] * 100
        f1 = reader_metrics[arm]["f1"] * 100
        print(f"  - {arm.upper():<8}: Exact Match = {em:5.1f}%, F1 = {f1:5.1f}%")

    # 9. Phase 10: Generate Publication Figures & Export Data
    print("\n" + "-" * 60)
    print("[Phase 10] Generating Figures and Saving Results...")
    _generate_figures(
        eval_split=eval_split,
        retrieval_metrics=retrieval_metrics,
        r5_graph=r5_graph,
        r5_vector=r5_vector,
        ci_graph=ci_graph_r5,
        ci_vector=ci_vector_r5,
        G=G,
    )

    # Save summary dictionary
    full_output = {
        "verdict": verdict_text,
        "delta_r5": delta_r5,
        "ci_r5": [ci_r5_lower, ci_r5_upper],
        "p_perm_r5": p_perm_r5,
        "retrieval_metrics": {
            k: {
                m: v
                for m, v in val.items()
                if not isinstance(v, np.ndarray)
            }
            for k, val in retrieval_metrics.items()
            if isinstance(val, dict)
        },
        "secondary_tests": holm_results,
        "reader_metrics": {
            k: {"exact_match": v["exact_match"], "f1": v["f1"]}
            for k, v in reader_metrics.items()
        },
    }
    with open(config.RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nAll experimental results saved to {config.RESULTS_DIR / 'experiment_results.json'}!")
    return full_output


def _generate_figures(
    eval_split: DatasetSplit,
    retrieval_metrics: Dict[str, dict],
    r5_graph: np.ndarray,
    r5_vector: np.ndarray,
    ci_graph: Tuple[float, float],
    ci_vector: Tuple[float, float],
    G: nx.Graph,
) -> None:
    """Generates the three core figures required by Section 10.5 of the Implementation Plan."""
    # 1. Recall Comparison Bar Chart with 95% Bootstrap CIs
    plt.figure(figsize=(9, 5))
    methods = ["Vector Baseline", "Graph + PPR (alpha=0.5)"]
    means = [np.mean(r5_vector) * 100, np.mean(r5_graph) * 100]
    yerr_lower = [means[0] - ci_vector[0] * 100, means[1] - ci_graph[0] * 100]
    yerr_upper = [ci_vector[1] * 100 - means[0], ci_graph[1] * 100 - means[1]]

    colors = ["#4C72B0", "#55A868"]
    bars = plt.bar(methods, means, yerr=[yerr_lower, yerr_upper], capsize=6, color=colors, alpha=0.85, width=0.5)
    plt.ylabel("Recall@5 (%)", fontsize=12)
    plt.title("Retrieval Recall@5 Comparison with 95% Bootstrap CIs (N=200)", fontsize=13, fontweight="bold")
    plt.ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h / 2.0, f"{h:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "recall_comparison.png", dpi=200)
    plt.close()

    # 2. Per-Question 2D Scatter Plot
    plt.figure(figsize=(7, 7))
    types = [q.type for q in eval_split.questions]
    palette = {"compositional": "#4C72B0", "comparison": "#DD8452", "bridge_comparison": "#8172B3", "inference": "#55A868"}

    # Add small jitter to reveal overlapping points
    rng = np.random.default_rng(config.SEED_EVAL)
    jitter_v = r5_vector * 100 + rng.uniform(-1.5, 1.5, size=len(r5_vector))
    jitter_g = r5_graph * 100 + rng.uniform(-1.5, 1.5, size=len(r5_graph))

    df_scatter = pd.DataFrame({"Vector_Recall": jitter_v, "Graph_Recall": jitter_g, "Type": types})
    sns.scatterplot(data=df_scatter, x="Vector_Recall", y="Graph_Recall", hue="Type", palette=palette, alpha=0.8, s=40)
    plt.plot([0, 105], [0, 105], "--", color="gray", label="y = x (Parity)")
    plt.xlabel("Vector Baseline Recall@5 (%)", fontsize=11)
    plt.ylabel("Graph + PPR Recall@5 (%)", fontsize=11)
    plt.title("Per-Question Recall@5: Graph vs Vector Baseline (Points above diagonal = Graph wins)", fontsize=11, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "per_question_scatter.png", dpi=200)
    plt.close()

    # 3. Subgraph Walk Visualization
    plt.figure(figsize=(10, 8))
    # Pick an ego-graph around an informative passage title node
    ego_center = None
    for n, d in G.degree():
        if d >= 3 and G.nodes[n].get("type") == "passage_title":
            ego_center = n
            break
    if ego_center is None:
        ego_center = list(G.nodes())[0]
    sub_nodes = list(nx.single_source_shortest_path_length(G, ego_center, cutoff=2).keys())[:16]
    subG = G.subgraph(sub_nodes)
    pos = nx.spring_layout(subG, seed=config.SEED_EVAL)
    node_colors = ["#55A868" if subG.nodes[n].get("type") == "passage_title" else "#4C72B0" for n in subG.nodes()]
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=600, alpha=0.85)
    nx.draw_networkx_edges(subG, pos, alpha=0.5, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(subG, pos, font_size=8, font_family="sans-serif")
    plt.title(f"Sample Multi-Hop Subgraph Walk around '{ego_center}'\n(Green = Passage Titles, Blue = Entities)", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "subgraph_walk.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    run_full_pipeline(max_reader_questions=25)

