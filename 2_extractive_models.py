"""
Script 2: Extractive Summarization Models
==========================================
Implements 4 extractive summarization models on 1000 preprocessed test samples.

  Model 1 – Lead-3    : Baseline — first 3 sentences of the article
  Model 2 – TextRank  : TF-IDF vectorization → cosine similarity graph → PageRank
  Model 3 – LSA       : TF-IDF → TruncatedSVD → sentence scores from topic weights
  Model 4 – LexRank   : IDF-modified cosine similarity → stochastic graph → PageRank

Evaluation: ROUGE-1/2/L (Precision, Recall, F1) for each model
Output:  outputs/extractive_results.pkl
         outputs/extractive_results.csv
         outputs/extractive_distributions.png
"""

import os
import math
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ─── Config ───────────────────────────────────────────────────────────
OUTPUTS_DIR  = "outputs"
EVAL_SAMPLES = 1000
TOP_N        = 3

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Load Preprocessed Data ───────────────────────────────────────────
print("=" * 62)
print("  NLP CA-3 | Step 2: Extractive Summarization (4 Models)")
print("=" * 62)
pkl_path = os.path.join(OUTPUTS_DIR, "preprocessed_data.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError("Run 1_preprocessing.py first.")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
data = data[:EVAL_SAMPLES]
print(f"\n[INFO] Loaded {len(data)} preprocessed samples.")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 1: Lead-3
# ═══════════════════════════════════════════════════════════════════════
def lead_n_summarize(sentences: list, n: int = TOP_N) -> str:
    """Return the first n sentences as the summary."""
    return " ".join(sentences[:n]) if sentences else ""


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 2: TextRank
# ═══════════════════════════════════════════════════════════════════════
def textrank_summarize(sentences: list, n: int = TOP_N) -> str:
    """
    TextRank Algorithm:
      1. TF-IDF vectorize sentences
      2. Build cosine-similarity adjacency matrix
      3. PageRank to score sentences
      4. Return top-n in original order
    """
    if len(sentences) <= n:
        return " ".join(sentences)
    try:
        tfidf  = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(sentences)
        sim    = cosine_similarity(matrix)
        np.fill_diagonal(sim, 0)
        graph  = nx.from_numpy_array(sim)
        scores = nx.pagerank(graph, max_iter=300, tol=1e-5)
    except Exception:
        return lead_n_summarize(sentences, n)
    top_idx = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    return " ".join(sentences[i] for i in top_idx)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 3: LSA (Latent Semantic Analysis)
# ═══════════════════════════════════════════════════════════════════════
def lsa_summarize(sentences: list, n: int = TOP_N, n_components: int = 5) -> str:
    """
    LSA Algorithm:
      1. TF-IDF vectorize sentences  (sentences × terms)
      2. TruncatedSVD to discover k latent topics
      3. Score each sentence by L2 norm of its topic vector
      4. Return top-n sentences in original order
    """
    if len(sentences) <= n:
        return " ".join(sentences)
    try:
        tfidf      = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix     = tfidf.fit_transform(sentences)
        k          = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if k < 1:
            return lead_n_summarize(sentences, n)
        svd        = TruncatedSVD(n_components=k, random_state=42)
        topic_mat  = svd.fit_transform(matrix)          # (n_sentences, k)
        scores     = np.sqrt((topic_mat ** 2).sum(axis=1))
    except Exception:
        return lead_n_summarize(sentences, n)
    top_idx = sorted(np.argsort(scores)[::-1][:n])
    return " ".join(sentences[i] for i in top_idx)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 4: LexRank
# ═══════════════════════════════════════════════════════════════════════
def _compute_idf(sentences_tokens: list) -> dict:
    """Compute IDF for each unique token across all sentence token lists."""
    N  = len(sentences_tokens)
    df = {}
    for tokens in sentences_tokens:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1
    return {w: math.log((N + 1) / (cnt + 1)) + 1.0 for w, cnt in df.items()}


def _idf_cosine(tok_a: list, tok_b: list, idf: dict) -> float:
    """IDF-weighted cosine similarity between two token lists."""
    vocab  = list(set(tok_a) | set(tok_b))
    vec_a  = np.array([tok_a.count(w) * idf.get(w, 1.0) for w in vocab])
    vec_b  = np.array([tok_b.count(w) * idf.get(w, 1.0) for w in vocab])
    norms  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / norms) if norms > 0 else 0.0


def lexrank_summarize(sentences: list, processed_tokens: list,
                      n: int = TOP_N, threshold: float = 0.1) -> str:
    """
    LexRank Algorithm:
      1. Compute IDF over all sentences
      2. Build IDF-modified cosine similarity matrix
      3. Threshold low similarities → stochastic row-normalize → graph
      4. PageRank to score sentences
      5. Return top-n sentences in original order
    """
    if len(sentences) <= n:
        return " ".join(sentences)
    try:
        idf     = _compute_idf(processed_tokens)
        m       = len(sentences)
        sim_mat = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                s = _idf_cosine(processed_tokens[i], processed_tokens[j], idf)
                sim_mat[i][j] = sim_mat[j][i] = s if s >= threshold else 0.0

        row_sums              = sim_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1.0
        stochastic            = sim_mat / row_sums

        graph  = nx.from_numpy_array(stochastic)
        scores = nx.pagerank(graph, max_iter=300, tol=1e-5)
    except Exception:
        return lead_n_summarize(sentences, n)
    top_idx = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    return " ".join(sentences[i] for i in top_idx)


# ═══════════════════════════════════════════════════════════════════════
#  Run All 4 Models
# ═══════════════════════════════════════════════════════════════════════
MODEL_NAMES = ["Lead-3", "TextRank", "LSA", "LexRank"]
print(f"\n[INFO] Running {len(MODEL_NAMES)} extractive models on {EVAL_SAMPLES} samples ...\n")

rows = []
for sample in tqdm(data, desc="Extractive Models"):
    sents    = sample["sentences"]
    proc     = sample["processed"]
    ref      = sample["reference"]

    summaries = {
        "Lead-3"   : lead_n_summarize(sents),
        "TextRank" : textrank_summarize(sents),
        "LSA"      : lsa_summarize(sents),
        "LexRank"  : lexrank_summarize(sents, proc),
    }

    row = {"id": sample["id"], "reference": ref}
    for name, summary in summaries.items():
        sc = scorer.score(ref, summary)
        row[f"{name}_summary"] = summary
        row[f"{name}_r1_p"]    = round(sc["rouge1"].precision,  4)
        row[f"{name}_r1_r"]    = round(sc["rouge1"].recall,     4)
        row[f"{name}_r1_f"]    = round(sc["rouge1"].fmeasure,   4)
        row[f"{name}_r2_p"]    = round(sc["rouge2"].precision,  4)
        row[f"{name}_r2_r"]    = round(sc["rouge2"].recall,     4)
        row[f"{name}_r2_f"]    = round(sc["rouge2"].fmeasure,   4)
        row[f"{name}_rL_f"]    = round(sc["rougeL"].fmeasure,   4)
    rows.append(row)

df = pd.DataFrame(rows)

# ─── Aggregate & Print ────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  Extractive Models — Aggregate ROUGE Scores")
print("=" * 68)
header = f"{'Model':<12} {'R1-Prec':>9} {'R1-Rec':>9} {'R1-F1':>9} {'R2-F1':>9} {'RL-F1':>9}"
print(header)
print("─" * 68)

agg = {}
for name in MODEL_NAMES:
    r1p = df[f"{name}_r1_p"].mean()
    r1r = df[f"{name}_r1_r"].mean()
    r1f = df[f"{name}_r1_f"].mean()
    r2f = df[f"{name}_r2_f"].mean()
    rLf = df[f"{name}_rL_f"].mean()
    agg[name] = {"R1-Prec": r1p, "R1-Rec": r1r, "R1-F1": r1f, "R2-F1": r2f, "RL-F1": rLf}
    print(f"{name:<12} {r1p:>9.4f} {r1r:>9.4f} {r1f:>9.4f} {r2f:>9.4f} {rLf:>9.4f}")

# ─── Qualitative Example ──────────────────────────────────────────────
print("\n\n📰 Qualitative Example (Sample 1):")
print("─" * 68)
row0 = rows[0]
print(f"Article   : {data[0]['article'][:300]}...\n")
print(f"Reference : {row0['reference']}\n")
for name in MODEL_NAMES:
    print(f"[{name}] {row0[name+'_summary'][:200]}")
    print(f"         ROUGE-1 F1={row0[name+'_r1_f']}  "
          f"ROUGE-2 F1={row0[name+'_r2_f']}  ROUGE-L F1={row0[name+'_rL_f']}\n")

# ─── Save ─────────────────────────────────────────────────────────────
out_pkl = os.path.join(OUTPUTS_DIR, "extractive_results.pkl")
out_csv = os.path.join(OUTPUTS_DIR, "extractive_results.csv")
with open(out_pkl, "wb") as f:
    pickle.dump({"df": df, "agg": agg, "models": MODEL_NAMES}, f)
df.to_csv(out_csv, index=False, encoding="utf-8")

# ─── ROUGE-1 F1 Distribution Plot ────────────────────────────────────
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Extractive Models – ROUGE-1 F1 Score Distributions",
             fontsize=13, fontweight="bold")
for ax, name, color in zip(axes, MODEL_NAMES, colors):
    col = f"{name}_r1_f"
    ax.hist(df[col], bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("ROUGE-1 F1")
    ax.set_ylabel("Frequency")
    ax.axvline(df[col].mean(), color="red", linestyle="--",
               label=f"Mean: {df[col].mean():.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
dist_path = os.path.join(OUTPUTS_DIR, "extractive_distributions.png")
plt.savefig(dist_path, dpi=150)
plt.show()

print(f"\n✅ Results PKL   → {out_pkl}")
print(f"✅ Results CSV   → {out_csv}")
print(f"✅ Distribution  → {dist_path}")
print("\n✅ Extractive models complete! Run 3_abstractive_models.py next.")
