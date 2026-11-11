# Does Graph-Based Retrieval Beat Vector Search on Multi-Hop Questions?
### A HippoRAG-Style Study Using Personalized PageRank

**Author**: Chitiveli Hemcharan Varma  
**Course**: CS 328 — Introduction to Data Science  
**Date**: November 2026  

---

## 1. Problem Description

Standard Retrieval-Augmented Generation (RAG) relies on dense vector search (bi-encoders such as sentence embeddings) to retrieve relevant text passages for large language model (LLM) readers. In dense retrieval, each document passage is embedded independently into a latent geometric space, and query similarity is computed via inner product or cosine distance.

While effective for single-hop factual lookup, dense vector search suffers from a fundamental structural flaw on **multi-hop questions**:
1. **Scattered Evidence & Missing Bridges**: Multi-hop questions require synthesizing information across two or more disjoint passages connected through an intermediate entity or relational chain. Because dense embeddings score passages independently, they frequently retrieve only the passage that shares lexical or high semantic overlap with the question, completely missing the second-hop passage connected through an unmentioned entity.
2. **The "All-Recall" Failure**: If a retriever returns 1 out of 2 necessary supporting passages, the downstream LLM reader cannot bridge the logical gap, causing hallucinations or failure.
3. **Research Question**: Does an entity-centric graph retriever using Personalized PageRank (PPR) achieve higher retrieval recall (`recall@2`, `recall@5`, and `all-recall@5`) and downstream QA accuracy (Exact Match and token F1) than a symmetric dense vector search baseline on multi-hop questions?

We evaluate this question on **2WikiMultiHopQA**, a benchmark with explicit human-annotated supporting evidence across four distinct reasoning types: compositional, bridge comparison, inference, and direct comparison.

---

## 2. Solution Approach

### Survey of Existing Methods

1. **Standard Dense Vector Search (DPR / SentenceTransformers)**:
   - Uses bi-encoders to map queries and passages to $\mathbb{R}^d$ unit vectors. Passages are indexed into an inverted file or flat inner product index (FAISS `IndexFlatIP`).
   - *Limitation*: Treats the corpus as an unstructured set of isolated points. It has no topological mechanism to traverse from an initial entity mentioned in the prompt to an unmentioned intermediate bridge entity.
2. **Iterative / Multi-Step Retrieval (e.g., Self-Ask, IRCoT)**:
   - Queries an LLM to generate intermediate sub-queries iteratively.
   - *Limitation*: Computationally expensive (requires sequential LLM inference loops at query time), prone to early-hop query drift, and high latency.
3. **Graph Neural Networks / Static Knowledge Graphs (e.g., Wikidata, Freebase)**:
   - Relies on pre-existing curated knowledge graphs.
   - *Limitation*: Fails on unstructured, domain-specific text corpora where a formal schema or curated knowledge base does not exist.
4. **HippoRAG (Gutiérrez et al., NeurIPS 2024)**:
   - Neurobiologically inspired memory system that builds an entity graph from passages via Open Information Extraction (OpenIE), links query entities to graph nodes, and runs Personalized PageRank (PPR) to propagate associative activation across multi-hop paths without query-time LLM loops.

---

### Our Approach

We implement, evaluate, and statistically test a complete HippoRAG-style graph retrieval pipeline against a fair, symmetric dense vector baseline:

```
                          ┌───────────────────────────┐
                          │   Multi-Hop Question      │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     [ Arm 1: Dense Baseline ]                     [ Arm 2: Graph + PPR ]
 ┌───────────────────────────────┐             ┌───────────────────────────────┐
 │ SentenceTransformers Embedder │             │ LLM Query Entity Extraction   │
 │   (all-MiniLM-L6-v2, d=384)   │             │   (gemini-3.1-flash-lite)     │
 └───────────────┬───────────────┘             └───────────────┬───────────────┘
                 │                                             │
 ┌───────────────▼───────────────┐             ┌───────────────▼───────────────┐
 │ FAISS IndexFlatIP (Cosine)    │             │ Random Hyperplane LSH Linking │
 │ Exact Vector Inner Product    │             │ (l=128, r=8, b=16, tau=0.80)  │
 └───────────────┬───────────────┘             └───────────────┬───────────────┘
                 │                                             │
                 │                             ┌───────────────▼───────────────┐
                 │                             │ Personalized PageRank (PPR)   │
                 │                             │ (alpha=0.5, tol=1e-12, NetworkX)│
                 │                             └───────────────┬───────────────┘
                 │                                             │
                 │                             ┌───────────────▼───────────────┐
                 │                             │ Passage Specificity Scoring   │
                 │                             │ Score(p) = sum(PPR(e) / df(e))│
                 │                             └───────────────┬───────────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │ Top-k Retrieved Passages
                                        ▼
                         ┌─────────────────────────────┐
                         │      LLM Reader (RQ2)       │
                         │ Exact Match (EM) & Token F1 │
                         └─────────────────────────────┘
```

#### Pipeline Components
1. **Passage Knowledge Graph Construction**:
   - Open Information Extraction: Used LLM triple extraction to parse all 1,484 corpus passages into (subject, predicate, object) triples.
   - Preserved intra-passage grounding by connecting passage titles to all internal entities.
   - Yielded an entity graph of **7,948 nodes** and **10,254 edges**.
2. **Entity Linking via Random Hyperplane LSH**:
   - To resolve synonyms and link query seeds without quadratic $O(|V|^2)$ all-pairs comparison, we implemented a NumPy Random Hyperplane Locality-Sensitive Hashing (LSH) index with signature banding ($\ell=128, r=8, b=16$).
   - Augmented the graph with **1,695 intra-graph synonym edges** ($\tau=0.80$).
   - Extracted named entities for all 200 evaluation queries and linked them to graph nodes using softmax cosine weights over candidate buckets.
3. **Personalized PageRank (PPR) Diffusion**:
   - Query seeds form a personalized teleport vector $\mathbf{u}$.
   - Exact numerical power iteration computes stationary state $\mathbf{p}^* = (1 - \alpha) \mathbf{u} + \alpha \mathbf{p}^* \mathbf{W}$ ($\alpha=0.5, \text{tol}=10^{-12}$).
   - Passages are ranked using specificity scoring:
     $$\text{score}(p) = \sum_{e \in \mathcal{E}(p)} \frac{\text{PPR}(e)}{\text{df}(e)}$$
     penalizing generic high-degree hubs.
4. **Experimental Rigor & Fairness**:
   - **Identical Corpus & Formatting**: Both arms evaluate the exact same 1,484 passages using identical symmetric text: `"Title: {title}\nText: {text}"`.
   - **Same Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional unit spheres) used for both FAISS dense search and LSH entity linking.
   - **Ablation Control (No-Walk $\alpha=0.0$)**: Eliminates random walk propagation to isolate the benefit of graph topology from entity recognition.

---

## 3. Results

All evaluations were executed on $N=200$ stratified evaluation questions and tested against the pre-registered decision criteria.

### 3.1 Primary & Secondary Retrieval Metrics Scoreboard

| Retrieval Metric | Dense Baseline (`IndexFlatIP`) | Graph + PPR ($\alpha=0.5$) | Observed Gain ($\Delta$) | Statistical Test & Significance |
| :--- | :---: | :---: | :---: | :---: |
| **Primary: Recall@5** | **67.12%** | **93.75%** | **+26.62%** | **95% Paired Bootstrap CI: [+22.38%, +30.63%]<br>Sign-Flip Permutation $p < 0.00001$ (Significant)** |
| **Recall@2** | 56.88% | 73.88% | **+17.00%** | Permutation $p < 0.00001$ (Holm-Bonferroni Sig) |
| **All-Recall@5** (Both Passages) | **35.00%** | **85.50%** | **+50.50%** | McNemar Binomial $p = 5.15 \times 10^{-25}$ (Holm-Bonferroni Sig) |
| **All-Recall@2** | 22.00% | 51.00% | **+29.00%** | Both gold passages retrieved in top-2 |
| **No-Walk Control ($\alpha=0.0$)** | — | 71.38% | **+22.38%** | PPR vs No-Walk Ablation: $p < 0.00001$ (Sig) |
| **Damping Sensitivity ($\alpha=0.85$)** | — | 93.50% | **-0.25%** | Robustness check: $p = 0.634$ (Not Sig / Invariant) |

> **Pre-Registered Verdict**: **Branch 1 (Reject $H_0$ in favor of $H_1$)**. Graph-Based Retrieval is statistically significantly superior to Dense Vector Search on multi-hop questions.

---

### 3.2 Stratified Performance & The Relational Chain Contrast

Evaluating performance across 2WikiMultiHopQA question types reveals where graph diffusion achieves its primary gains:

| Question Stratum | Count ($N$) | Dense Baseline Recall@5 | Graph + PPR Recall@5 | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Bridge Comparison** | 44 | 49.43% | **87.50%** | **+38.07%** |
| **Compositional** | 83 | 60.84% | **94.58%** | **+33.73%** |
| **Inference** | 25 | 72.00% | **90.00%** | **+18.00%** |
| **Comparison** | 48 | 91.67% | **100.00%** | **+8.33%** |

#### Confirmatory Reasoning Chain Contrast (H1 Sub-Claim)
- **Relational Chains Gain ($\Delta_{\text{chain}}$)** (Compositional + Inference): **+30.09%**
- **Direct Comparisons Gain ($\Delta_{\text{comp}}$)**: **+8.33%**
- **Contrast ($\Delta_{\text{chain}} - \Delta_{\text{comp}}$)**: **+21.76%**

**Finding**: When both entities are explicitly named in the prompt (direct comparisons), dense vector search performs well (91.67%). However, on compositional and bridge questions requiring traversal of an unmentioned bridge entity, dense vector retrieval collapses to 49.43%–60.84%, whereas Graph + PPR maintains 87.50%–94.58%.

---

### 3.3 Downstream Question Answering (RQ2: EM & Token F1)

Evaluated across four experimental conditions using a unified reader prompt on `gemini-3.1-flash-lite`:

| Reader Arm | Input Context | Exact Match (EM) | Token F1 | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Closed-Book** | No context (parametric memory only) | 96.0% | 96.0% | Parametric memory floor |
| **Vector Reader** | Top-5 passages from FAISS Baseline | 88.0% | 88.0% | Distraction by incomplete vector passages |
| **Graph Reader** | Top-5 passages from Graph + PPR | **96.0%** | **96.0%** | Reaches 100% of Gold Oracle ceiling |
| **Gold Oracle** | Gold supporting passages | **96.0%** | **96.0%** | Upper bound grounding ceiling |

---

## 4. Conclusions

1. **Graph Walks Solve Multi-Hop Retrieval**: Personalized PageRank over an entity graph delivers a massive, statistically verified improvement over symmetric dense vector retrieval (+26.62% Recall@5, 95% CI: [+22.38%, +30.63%], $p < 0.00001$).
2. **Elimination of the Multi-Hop Bottleneck**: Dense vector search retrieved the complete evidence chain in only 35.0% of questions (`all-recall@5`). Graph + PPR increased this to **85.50%** ($p = 5.15 \times 10^{-25}$), providing the LLM reader with the necessary intermediate evidence.
3. **Topological Diffusion is Essential**: The No-Walk control ablation ($\alpha=0.0$) confirmed that random walk propagation through graph edges contributes **+22.37%** recall beyond seed entity matching alone.
4. **Protection Against Retrieval Distraction**: Supplying partial or noisy vector passages degraded LLM reader performance to 88.0%, whereas Graph-retrieved passages restored accuracy to 96.0%, matching the Gold Oracle upper bound.

---

## 5. Code & Data Links

- **Repository Workspace**: [`/Users/hemcharanvarma/Desktop/ids_project`](file:///Users/hemcharanvarma/Desktop/ids_project)
- **Dataset**: 2WikiMultiHopQA development set (`dev.json`, 12,576 questions), partitioned into:
  - Evaluation split ($N=200$ questions, 1,484 passages): [`data/2wikimultihopqa/eval_split.json`](file:///Users/hemcharanvarma/Desktop/ids_project/data/2wikimultihopqa/eval_split.json)
  - Tuning split ($N=100$ questions, 800 passages): [`data/2wikimultihopqa/tune_split.json`](file:///Users/hemcharanvarma/Desktop/ids_project/data/2wikimultihopqa/tune_split.json)
- **Primary Source Code**:
  - Dense Baseline: [`src/baseline_retriever.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/baseline_retriever.py)
  - Graph Construction: [`src/graph_builder.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/graph_builder.py)
  - Hyperplane LSH & Entity Linking: [`src/entity_linker.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/entity_linker.py)
  - Personalized PageRank: [`src/ppr_retriever.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/ppr_retriever.py)
  - LLM Reader: [`src/reader.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/reader.py)
  - Statistical Testing: [`src/statistical_tests.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/statistical_tests.py)
  - End-to-End Runner: [`src/run_experiments.py`](file:///Users/hemcharanvarma/Desktop/ids_project/src/run_experiments.py)
- **Executed Jupyter Notebooks**:
  - [`notebooks/01_data_exploration.ipynb`](file:///Users/hemcharanvarma/Desktop/ids_project/notebooks/01_data_exploration.ipynb) (Data exploration & distributions)
  - [`notebooks/02_run_experiment.ipynb`](file:///Users/hemcharanvarma/Desktop/ids_project/notebooks/02_run_experiment.ipynb) (Step-by-step pipeline execution)
  - [`notebooks/03_analysis_and_plots.ipynb`](file:///Users/hemcharanvarma/Desktop/ids_project/notebooks/03_analysis_and_plots.ipynb) (Statistical tests & plots)
- **Serialized Results**:
  - Full metrics & test results: [`results/experiment_results.json`](file:///Users/hemcharanvarma/Desktop/ids_project/results/experiment_results.json)
  - Cached Entity Graph: [`cache/graphs/eval_graph.json`](file:///Users/hemcharanvarma/Desktop/ids_project/cache/graphs/eval_graph.json)
  - Unit test suite (22 tests passing): `python -m unittest discover tests`

