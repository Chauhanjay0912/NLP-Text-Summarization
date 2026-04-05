"""
Script 5: Interactive Demo
===========================
Interactive CLI — paste any news article to get summaries from all 6 models:

  Extractive  : Lead-3  |  TextRank  |  LSA  |  LexRank
  Abstractive : T5-small  |  BART-large-cnn

Options:
  [1] Run ALL 6 models
  [2] Choose specific models
  [3] Try a built-in sample article
  [Q] Quit

If you paste a reference summary, ROUGE-1/2/L (Precision, Recall, F1) are shown.
"""

import os
import math
import textwrap

# ── Force PyTorch backend (prevents Keras 3 / TensorFlow conflict) ────
os.environ["TRANSFORMERS_NO_TF"]  = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import nltk
import torch
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from rouge_score import rouge_scorer

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# ─── Config ───────────────────────────────────────────────────────────
TOP_N      = 3
MAX_WORDS  = 600
WRAP_WIDTH = 78
LEMMA      = WordNetLemmatizer()
STOPS      = set(stopwords.words('english'))
ROUGE      = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

SAMPLE_ARTICLE = (
    "(CNN) -- NASA scientists confirmed the discovery of seven Earth-sized planets "
    "orbiting TRAPPIST-1, a dwarf star located about 40 light-years from Earth. "
    "Three of the planets are in the habitable zone, where liquid water could exist "
    "on the surface. The discovery, published in the journal Nature, was made using "
    "the Spitzer Space Telescope and ground-based observatories. Scientists say this "
    "system offers the best chance yet of finding signs of life beyond our solar system. "
    "All seven planets are rocky and similar in size to Earth. Researchers plan to "
    "use the James Webb Space Telescope to study the atmospheres of the three "
    "potentially habitable planets. 'We've made a giant step forward in the search "
    "for life,' said lead researcher Michaël Gillon of the University of Liège. "
    "The findings were independently confirmed by multiple international teams."
)

# ─── Preprocessing ────────────────────────────────────────────────────
def preprocess(article: str):
    sents = sent_tokenize(article)
    tok   = [word_tokenize(s) for s in sents]
    proc  = [[LEMMA.lemmatize(w.lower()) for w in s
              if w.isalpha() and w.lower() not in STOPS] for s in tok]
    return sents, proc

def truncate(text, n=MAX_WORDS):
    words = text.split()
    return " ".join(words[:n]) if len(words) > n else text

# ─── Extractive Models ────────────────────────────────────────────────
def lead3(sents, _proc, n=TOP_N):
    return " ".join(sents[:n])

def textrank(sents, _proc, n=TOP_N):
    if len(sents) <= n: return " ".join(sents)
    try:
        mat  = TfidfVectorizer(stop_words="english").fit_transform(sents)
        sim  = cosine_similarity(mat); np.fill_diagonal(sim, 0)
        sc   = nx.pagerank(nx.from_numpy_array(sim), max_iter=300)
    except Exception: return lead3(sents, _proc, n)
    return " ".join(sents[i] for i in sorted(sorted(sc, key=sc.get, reverse=True)[:n]))

def lsa(sents, _proc, n=TOP_N, k=5):
    if len(sents) <= n: return " ".join(sents)
    try:
        mat  = TfidfVectorizer(stop_words="english").fit_transform(sents)
        kk   = min(k, mat.shape[0]-1, mat.shape[1]-1)
        if kk < 1: return lead3(sents, _proc, n)
        sc   = np.sqrt((TruncatedSVD(kk, random_state=42).fit_transform(mat)**2).sum(1))
    except Exception: return lead3(sents, _proc, n)
    return " ".join(sents[i] for i in sorted(np.argsort(sc)[::-1][:n]))

def lexrank(sents, proc, n=TOP_N, thr=0.1):
    if len(sents) <= n: return " ".join(sents)
    try:
        N = len(proc)
        df_map = {}
        for tokens in proc:
            for t in set(tokens): df_map[t] = df_map.get(t, 0) + 1
        idf = {w: math.log((N+1)/(c+1))+1 for w, c in df_map.items()}

        def idf_cos(a, b):
            vocab = list(set(a)|set(b))
            va = np.array([a.count(w)*idf.get(w,1.) for w in vocab])
            vb = np.array([b.count(w)*idf.get(w,1.) for w in vocab])
            n_ = np.linalg.norm(va)*np.linalg.norm(vb)
            return float(np.dot(va,vb)/n_) if n_ > 0 else 0.

        sim = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1, N):
                s = idf_cos(proc[i], proc[j])
                sim[i][j] = sim[j][i] = s if s >= thr else 0.
        rs = sim.sum(1, keepdims=True); rs[rs==0] = 1
        sc = nx.pagerank(nx.from_numpy_array(sim/rs), max_iter=300)
    except Exception: return lead3(sents, proc, n)
    return " ".join(sents[i] for i in sorted(sorted(sc, key=sc.get, reverse=True)[:n]))

EXTRACTIVE_MODELS = {
    "Lead-3"  : lead3,
    "TextRank": textrank,
    "LSA"     : lsa,
    "LexRank" : lexrank,
}

# ─── Abstractive Models (loaded once) ────────────────────────────────
device      = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

def _box(title, text, code="34"):
    bar = "─" * (WRAP_WIDTH + 2)
    print(f"\n\033[{code}m╔{bar}╗\n║  {title:<{WRAP_WIDTH}}║\n╠{bar}╣\033[0m")
    for line in textwrap.fill(text, WRAP_WIDTH).split("\n"):
        print(f"  {line}")
    print(f"\033[{code}m╚{bar}╝\033[0m")

def _rouge_report(ref, gen, label):
    sc = ROUGE.score(ref, gen)
    print(f"\n  📊 ROUGE ({label}):")
    for key, name in [("rouge1","ROUGE-1"), ("rouge2","ROUGE-2"), ("rougeL","ROUGE-L")]:
        s = sc[key]
        print(f"     {name}: P={s.precision:.4f}  R={s.recall:.4f}  F1={s.fmeasure:.4f}")

# ─── Main Loop ────────────────────────────────────────────────────────
print("\n" + "="*64)
print("  📰  News Summarization Demo  —  All 6 Models")
print(f"  Device: {device_name}")
print("="*64)

t5_pipe = bart_pipe = None

def load_abstractive():
    global t5_pipe, bart_pipe
    if t5_pipe is None:
        print("\n⏳ Loading T5-small ...")
        t5_pipe   = pipeline("summarization", model="t5-small",
                              framework="pt", device=device, truncation=True)
    if bart_pipe is None:
        print("⏳ Loading BART-large-cnn ...")
        bart_pipe = pipeline("summarization", model="facebook/bart-large-cnn",
                              framework="pt", device=device, truncation=True)

MODEL_MENU = {
    "1": "Lead-3", "2": "TextRank", "3": "LSA",
    "4": "LexRank", "5": "T5-small", "6": "BART"
}

while True:
    print("\n" + "━"*64)
    print("  [1] Summarize my article  [2] Built-in sample  [Q] Quit")
    print("━"*64)
    choice = input("\nChoice: ").strip().lower()

    if choice in ("q", "quit", "exit"):
        print("\n👋 Bye!\n"); break

    # ── Get article ───────────────────────────────────────────────────
    if choice == "2":
        article   = SAMPLE_ARTICLE
        reference = None
        print("\n[Using built-in sample: TRAPPIST-1 exoplanet article]")
    elif choice == "1":
        print("\nPaste your article. Type END on a new line when done:\n")
        lines = []
        while True:
            l = input()
            if l.strip().upper() == "END": break
            lines.append(l)
        article = " ".join(lines).strip()
        if len(article.split()) < 10:
            print("[WARN] Too short. Please try again."); continue
        print("\nReference summary (press Enter to skip):")
        reference = input().strip() or None
    else:
        print("[WARN] Invalid choice."); continue

    # ── Model selection ────────────────────────────────────────────────
    print("\nSelect models (e.g. 1 2 3 4 5 6 for all, or press Enter for all):")
    for k, v in MODEL_MENU.items(): print(f"  [{k}] {v}")
    sel = input("Models: ").strip()
    chosen = [MODEL_MENU[k] for k in sel.split() if k in MODEL_MENU] if sel else list(MODEL_MENU.values())

    # Load abstractive if needed
    if "T5-small" in chosen or "BART" in chosen:
        load_abstractive()

    sents, proc = preprocess(article)

    # ── Run models ────────────────────────────────────────────────────
    print()
    for name in chosen:
        if name in EXTRACTIVE_MODELS:
            summary = EXTRACTIVE_MODELS[name](sents, proc)
            _box(f"📌 [{name}]  Extractive", summary, "34")
        elif name == "T5-small":
            out     = t5_pipe("summarize: " + truncate(article),
                               max_length=150, min_length=40,
                               num_beams=4, early_stopping=True, truncation=True)
            summary = out[0]["summary_text"]
            _box(f"🤖 [T5-small]  Abstractive", summary, "33")
        elif name == "BART":
            out     = bart_pipe(truncate(article),
                                max_length=142, min_length=56,
                                num_beams=4, length_penalty=2.0,
                                early_stopping=True, truncation=True)
            summary = out[0]["summary_text"]
            _box(f"🤖 [BART]  Abstractive", summary, "33")

        if reference:
            _rouge_report(reference, summary, name)

    if reference:
        _box("📎 Reference Summary", reference, "32")
