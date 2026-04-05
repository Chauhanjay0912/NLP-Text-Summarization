"""
Script 1: Data Preprocessing
=============================
Builds a reusable preprocessing pipeline on 1000 CNN/DailyMail test samples:
  Step 1 – Cleaning       : remove HTML tags, CNN prefix, special characters
  Step 2 – Sentence Tok.  : NLTK sent_tokenize
  Step 3 – Word Tok.      : NLTK word_tokenize
  Step 4 – Stop-word Rem. : NLTK English stopword list
  Step 5 – Normalization  : lowercase + lemmatization (WordNetLemmatizer)

Output: outputs/preprocessed_data.pkl  (reused by all subsequent scripts)
"""

import os
import re
import pickle

import nltk
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── NLTK Downloads ───────────────────────────────────────────────────
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# ─── Config ───────────────────────────────────────────────────────────
DATASET_VERSION = "3.0.0"
OUTPUTS_DIR     = "outputs"
NUM_SAMPLES     = 1000

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Preprocessing Utilities ──────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Step 1: Remove HTML tags, CNN byline prefix, special chars, extra whitespace."""
    text = re.sub(r'<[^>]+>',              '',  text)   # HTML tags
    text = re.sub(r'\(CNN\)\s*--?\s*',     '',  text)   # CNN byline
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]',' ', text)  # special chars (keep punct)
    text = re.sub(r'\s+',                  ' ', text)   # extra whitespace
    return text.strip()

def preprocess(article: str) -> dict:
    """Full preprocessing pipeline for one article."""
    # Step 1: Clean
    cleaned = clean_text(article)

    # Step 2: Sentence tokenization (on cleaned text)
    sentences = sent_tokenize(cleaned)

    # Step 3: Word tokenization per sentence
    tokenized = [word_tokenize(s) for s in sentences]

    # Steps 4 & 5: Stop-word removal + lowercase + lemmatization
    processed = [
        [lemmatizer.lemmatize(tok.lower())
         for tok in sent_tokens
         if tok.isalpha() and tok.lower() not in STOP_WORDS]
        for sent_tokens in tokenized
    ]

    return {
        "cleaned"   : cleaned,
        "sentences" : sentences,   # original sentences (used by models for output)
        "tokenized" : tokenized,   # word tokens (raw)
        "processed" : processed,   # normalized, no stopwords (used by LexRank/LSA)
    }

# ─── 1. Load Dataset ──────────────────────────────────────────────────
print("=" * 62)
print("  NLP CA-3 | Step 1: Data Preprocessing")
print("=" * 62)
print(f"\n[INFO] Loading CNN/DailyMail {DATASET_VERSION} (test split) ...")
dataset   = load_dataset("ccdv/cnn_dailymail", DATASET_VERSION)
test_data = dataset["test"].select(range(NUM_SAMPLES))
print(f"[INFO] Selected {NUM_SAMPLES} test samples for evaluation.\n")

# ─── 2. Run Pipeline ──────────────────────────────────────────────────
print("[INFO] Running preprocessing pipeline ...")
records = []
for example in tqdm(test_data, desc="Preprocessing"):
    result = preprocess(example["article"])
    records.append({
        "id"        : example["id"],
        "article"   : example["article"],
        "reference" : example["highlights"],
        "cleaned"   : result["cleaned"],
        "sentences" : result["sentences"],
        "tokenized" : result["tokenized"],
        "processed" : result["processed"],
    })

# ─── 3. Before / After Examples ──────────────────────────────────────
print("\n" + "=" * 62)
print("  Before ↔ After Preprocessing (2 samples)")
print("=" * 62)

for i in range(2):
    s = records[i]
    orig_words = s["article"].split()
    proc_words = [tok for sent in s["processed"] for tok in sent]

    print(f"\n{'─'*62}")
    print(f"  Sample {i+1}")
    print(f"{'─'*62}")
    print(f"  ORIGINAL       : {s['article'][:280]}...")
    print(f"\n  CLEANED        : {s['cleaned'][:280]}...")
    print(f"\n  SENTENCES      : {len(s['sentences'])} sentences")
    print(f"  SAMPLE SENT    : {s['sentences'][0][:120] if s['sentences'] else 'N/A'}")
    print(f"\n  PROCESSED TOKS : {s['processed'][0][:12] if s['processed'] else []}")
    print(f"\n  Word count     : {len(orig_words):,} → {len(proc_words):,} "
          f"({(1 - len(proc_words)/max(len(orig_words),1))*100:.1f}% reduction)")

# ─── 4. Statistics ────────────────────────────────────────────────────
stats_rows = []
for r in records:
    all_proc = [t for s in r["processed"] for t in s]
    stats_rows.append({
        "n_sentences"    : len(r["sentences"]),
        "words_original" : len(r["article"].split()),
        "words_cleaned"  : len(r["cleaned"].split()),
        "words_processed": len(all_proc),
        "words_reference": len(r["reference"].split()),
    })

df_stats = pd.DataFrame(stats_rows)
print("\n\n📊 Preprocessing Statistics (word counts):")
print("-" * 62)
print(df_stats.describe().round(1).to_string())

# ─── 5. Pipeline Visualization ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("CNN/DailyMail – Preprocessing Word-Count Distributions",
             fontsize=13, fontweight="bold")

plot_info = [
    ("words_original",  "#4C72B0", "Original Article"),
    ("words_cleaned",   "#55A868", "After Cleaning"),
    ("words_processed", "#C44E52", "After Stop-word Removal\n& Lemmatization"),
]
for ax, (col, color, label) in zip(axes, plot_info):
    ax.hist(df_stats[col], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(label)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.axvline(df_stats[col].mean(), color="black", linestyle="--",
               label=f"Mean: {df_stats[col].mean():.0f}")
    ax.legend()

plt.tight_layout()
plot_path = os.path.join(OUTPUTS_DIR, "preprocessing_wordcount.png")
plt.savefig(plot_path, dpi=150)
plt.show()

# ─── 6. Save ──────────────────────────────────────────────────────────
pkl_path   = os.path.join(OUTPUTS_DIR, "preprocessed_data.pkl")
stats_path = os.path.join(OUTPUTS_DIR, "preprocessing_stats.csv")

with open(pkl_path, "wb") as f:
    pickle.dump(records, f)
df_stats.describe().round(1).to_csv(stats_path)

print(f"\n✅ Preprocessed data  → {pkl_path}")
print(f"✅ Stats CSV          → {stats_path}")
print(f"✅ Word-count plot    → {plot_path}")
print(f"\n  Total samples saved: {len(records)}")
print("\n✅ Preprocessing complete! Run 2_extractive_models.py next.")
