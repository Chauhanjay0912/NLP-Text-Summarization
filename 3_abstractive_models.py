"""
Script 3: Abstractive Summarization Models
==========================================
Implements 2 abstractive models using HuggingFace Transformers:

  Model 5 – T5-small         : "summarize:" prefix → seq2seq generation
  Model 6 – BART-large-cnn   : beam search, fine-tuned on CNN/DailyMail

  - Auto-detects CUDA GPU (falls back to CPU)
  - Batched inference for efficiency
  - ROUGE-1/2/L (Precision, Recall, F1) evaluation

Output: outputs/abstractive_results.pkl
        outputs/abstractive_results.csv
        outputs/abstractive_distributions.png
"""

import os
import pickle

# ── Force PyTorch backend (prevents Keras 3 / TensorFlow conflict) ────
os.environ["TRANSFORMERS_NO_TF"]  = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from rouge_score import rouge_scorer

# ─── Config ───────────────────────────────────────────────────────────
OUTPUTS_DIR  = "outputs"
EVAL_SAMPLES = 200    # manageable on CPU; increase if CUDA available
BATCH_SIZE   = 4
MAX_WORDS    = 600    # truncate articles before sending to model

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Device ───────────────────────────────────────────────────────────
device      = 0 if torch.cuda.is_available() else -1
device_name = f"CUDA ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU"

print("=" * 62)
print("  NLP CA-3 | Step 3: Abstractive Summarization (2 Models)")
print("=" * 62)
print(f"\n[INFO] Device       : {device_name}")
print(f"[INFO] Eval samples : {EVAL_SAMPLES}")
print(f"[INFO] Batch size   : {BATCH_SIZE}")

# ─── Load Preprocessed Data ───────────────────────────────────────────
pkl_path = os.path.join(OUTPUTS_DIR, "preprocessed_data.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError("Run 1_preprocessing.py first.")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
data = data[:EVAL_SAMPLES]
print(f"[INFO] Loaded {len(data)} preprocessed samples.\n")

# ─── Load Models ──────────────────────────────────────────────────────
print("[INFO] Loading T5-small (~240 MB, downloads on first run) ...")
t5_pipe = pipeline(
    "summarization", model="t5-small",
    framework="pt", device=device, truncation=True
)

print("[INFO] Loading BART-large-cnn (~1.6 GB, downloads on first run) ...")
bart_pipe = pipeline(
    "summarization", model="facebook/bart-large-cnn",
    framework="pt", device=device, truncation=True
)
print("[INFO] Both models loaded.\n")

# ─── Helper Functions ─────────────────────────────────────────────────
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def run_model(pipe, inputs: list, label: str, **kwargs) -> list:
    """Run inference in batches; return list of summary strings."""
    summaries = []
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc=label):
        batch = inputs[i : i + BATCH_SIZE]
        try:
            outs = pipe(batch, **kwargs)
            summaries.extend(o["summary_text"] for o in outs)
        except Exception as e:
            print(f"\n[WARN] Batch {i//BATCH_SIZE} failed: {e}")
            summaries.extend("" for _ in batch)
    return summaries


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 5: T5-small
# ═══════════════════════════════════════════════════════════════════════
articles_raw = [truncate(d["article"]) for d in data]
references   = [d["reference"]         for d in data]
ids          = [d["id"]                for d in data]

# T5 requires the "summarize: " task prefix
t5_inputs = ["summarize: " + a for a in articles_raw]

print("=" * 62)
print("  Model 5: T5-small")
print("=" * 62)
t5_summaries = run_model(
    t5_pipe, t5_inputs, "T5-small",
    max_length=150, min_length=40,
    num_beams=4, early_stopping=True,
    truncation=True
)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 6: BART-large-cnn
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  Model 6: BART-large-cnn")
print("=" * 62)
bart_summaries = run_model(
    bart_pipe, articles_raw, "BART",
    max_length=142, min_length=56,
    num_beams=4, length_penalty=2.0,
    early_stopping=True, truncation=True
)


# ─── ROUGE Evaluation ─────────────────────────────────────────────────
print("\n[INFO] Computing ROUGE scores ...")
rows = []
for sid, ref, t5_sum, bart_sum in zip(ids, references, t5_summaries, bart_summaries):
    t5sc   = scorer.score(ref, t5_sum)
    bartsc = scorer.score(ref, bart_sum)
    rows.append({
        "id"              : sid,
        "reference"       : ref,
        # T5
        "T5-small_summary": t5_sum,
        "T5-small_r1_p"   : round(t5sc["rouge1"].precision,  4),
        "T5-small_r1_r"   : round(t5sc["rouge1"].recall,     4),
        "T5-small_r1_f"   : round(t5sc["rouge1"].fmeasure,   4),
        "T5-small_r2_p"   : round(t5sc["rouge2"].precision,  4),
        "T5-small_r2_r"   : round(t5sc["rouge2"].recall,     4),
        "T5-small_r2_f"   : round(t5sc["rouge2"].fmeasure,   4),
        "T5-small_rL_f"   : round(t5sc["rougeL"].fmeasure,   4),
        # BART
        "BART_summary"    : bart_sum,
        "BART_r1_p"       : round(bartsc["rouge1"].precision, 4),
        "BART_r1_r"       : round(bartsc["rouge1"].recall,    4),
        "BART_r1_f"       : round(bartsc["rouge1"].fmeasure,  4),
        "BART_r2_p"       : round(bartsc["rouge2"].precision, 4),
        "BART_r2_r"       : round(bartsc["rouge2"].recall,    4),
        "BART_r2_f"       : round(bartsc["rouge2"].fmeasure,  4),
        "BART_rL_f"       : round(bartsc["rougeL"].fmeasure,  4),
    })

df = pd.DataFrame(rows)

# ─── Aggregate & Print ────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  Abstractive Models — Aggregate ROUGE Scores")
print("=" * 68)
print(f"{'Model':<12} {'R1-Prec':>9} {'R1-Rec':>9} {'R1-F1':>9} {'R2-F1':>9} {'RL-F1':>9}")
print("─" * 68)

agg = {}
for name in ["T5-small", "BART"]:
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
print(f"Article   : {data[0]['article'][:300]}...\n")
print(f"Reference : {rows[0]['reference']}\n")
print(f"[T5-small] {rows[0]['T5-small_summary']}")
print(f"           ROUGE-1 F1={rows[0]['T5-small_r1_f']}  ROUGE-2 F1={rows[0]['T5-small_r2_f']}\n")
print(f"[BART]     {rows[0]['BART_summary']}")
print(f"           ROUGE-1 F1={rows[0]['BART_r1_f']}  ROUGE-2 F1={rows[0]['BART_r2_f']}")

# ─── Save ─────────────────────────────────────────────────────────────
out_pkl = os.path.join(OUTPUTS_DIR, "abstractive_results.pkl")
out_csv = os.path.join(OUTPUTS_DIR, "abstractive_results.csv")
with open(out_pkl, "wb") as f:
    pickle.dump({"df": df, "agg": agg, "models": ["T5-small", "BART"]}, f)
df.to_csv(out_csv, index=False, encoding="utf-8")

# ─── Distribution Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Abstractive Models – ROUGE-1 F1 Score Distributions",
             fontsize=13, fontweight="bold")
for ax, name, color in zip(axes, ["T5-small", "BART"], ["#CCB974", "#64B5CD"]):
    col = f"{name}_r1_f"
    ax.hist(df[col], bins=25, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("ROUGE-1 F1")
    ax.set_ylabel("Frequency")
    ax.axvline(df[col].mean(), color="red", linestyle="--",
               label=f"Mean: {df[col].mean():.3f}")
    ax.legend()

plt.tight_layout()
dist_path = os.path.join(OUTPUTS_DIR, "abstractive_distributions.png")
plt.savefig(dist_path, dpi=150)
plt.show()

print(f"\n✅ Results PKL   → {out_pkl}")
print(f"✅ Results CSV   → {out_csv}")
print(f"✅ Distribution  → {dist_path}")
print("\n✅ Abstractive models complete! Run 4_evaluation.py next.")
