# NLP CA-3 Mini Project: Text Summarization of News Articles

## Overview

Implements and compares **6 summarization models** on the CNN/DailyMail dataset,
covering all 5 CA-3 task requirements: dataset selection, preprocessing,
≥5 models, evaluation (Precision/Recall/F1), and comparative analysis.

| Type | Models |
|------|--------|
| Extractive | Lead-3 (baseline), TextRank, LSA, LexRank |
| Abstractive | PEGASUS-CNN, BART-large-cnn |

**Evaluation Metrics**: ROUGE-1/2/L — Precision, Recall, F1 + BERTScore (semantic similarity)

---

## Task: Text Summarization

### What it does
Takes a long news article as input and produces a short, coherent summary as output.

### Input → Output

```
Input  : Full news article (500–800 words)
           ↓
      [Preprocessing]  →  cleaned text, tokenized sentences
           ↓
      [Summarization Model]
           ↓
Output : Short summary (3 sentences / ~50–150 words)
           ↓
      [ROUGE Evaluation]  →  compared against human-written reference summary
```

### Example

| | Text |
|---|---|
| **Input** | Full CNN/DailyMail news article (~600 words) |
| **Reference** | Human-written highlights by CNN editors |
| **Lead-3 output** | First 3 sentences of the article |
| **TextRank output** | 3 most important sentences selected by graph ranking |
| **PEGASUS output** | Newly generated summary (~30–200 words) |
| **BART output** | Newly generated summary (~30–200 words) |

- **Extractive models** → select sentences directly from the input
- **Abstractive models** → generate new text not necessarily in the original article

---

## Dataset
- **[CNN / DailyMail 3.0.0](https://huggingface.co/datasets/ccdv/cnn_dailymail)** — ~300k English news articles
- **Fields**: `article` (full text) · `highlights` (reference summary)
- **Eval subset**: 1000 test samples (extractive) / 200 test samples (abstractive)

---

## Project Structure

```
NLP_CA_3_miniproject/
├── requirements.txt              # All pip dependencies
├── README.md                     # This file
├── 1_preprocessing.py            # Cleaning, tokenization, stop-word removal, lemmatization
├── 2_extractive_models.py        # Lead-3 | TextRank | LSA | LexRank
├── 3_abstractive_models.py       # PEGASUS-CNN | BART-large-cnn
├── 4_evaluation.py               # Comparison table, bar chart, box plot, win-rate
├── 5_demo.py                     # Interactive CLI demo (all 6 models)
├── outputs/                      # All generated plots, CSVs, PKL files
└── samples/                      # Saved sample articles
```

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run.

---

## How to Run (in order)

```bash
# Step 1 – Preprocessing
python 1_preprocessing.py

# Step 2 – Extractive Models (Lead-3, TextRank, LSA, LexRank)
python 2_extractive_models.py

# Step 3 – Abstractive Models (PEGASUS-CNN, BART) — downloads models on first run
python 3_abstractive_models.py

# Step 4 – Evaluation & Comparative Analysis
python 4_evaluation.py

# Step 5 – Interactive Demo
python 5_demo.py
```

---

## Preprocessing Pipeline (Script 1)

| Step | Operation | Tool |
|------|-----------|------|
| 1 | Cleaning (HTML, CNN byline, special chars) | `re` |
| 2 | Sentence tokenization | `nltk.sent_tokenize` |
| 3 | Word tokenization | `nltk.word_tokenize` |
| 4 | Stop-word removal | `nltk.corpus.stopwords` |
| 5 | Normalization (lowercase + lemmatization) | `WordNetLemmatizer` |

---

## Model Details

### Extractive (Script 2)

| Model | Algorithm |
|-------|-----------|
| **Lead-3** | Return the first 3 sentences (inverted-pyramid baseline) |
| **TextRank** | TF-IDF → cosine similarity graph → PageRank sentence scoring |
| **LSA** | TF-IDF → TruncatedSVD (k topics) → L2 norm topic scores |
| **LexRank** | IDF-weighted cosine → threshold → stochastic graph → PageRank |

### Abstractive (Script 3)

| Model | Details |
|-------|---------|
| **PEGASUS-CNN** | `google/pegasus-cnn_dailymail`, ~2.3 GB, fine-tuned on CNN/DailyMail, beam search (k=8) |
| **BART** | `facebook/bart-large-cnn`, ~1.6 GB, fine-tuned on CNN/DailyMail, beam search (k=6) |

> Both models auto-detect CUDA GPU and fall back to CPU.

---

## Interactive Demo (Script 5)

Run `python 5_demo.py` to summarize any article interactively:

```
Options:
  [1] Paste your own article
  [2] Use built-in sample (TRAPPIST-1 exoplanet article)
  [Q] Quit

Then select which models to run (1–6 or Enter for all).
Optionally provide a reference summary to get live ROUGE scores.
```

---

## Generated Outputs

| File | Description |
|------|-------------|
| `outputs/preprocessed_data.pkl` | Preprocessed samples (reused by all scripts) |
| `outputs/extractive_results.pkl/csv` | Per-sample scores for 4 extractive models |
| `outputs/abstractive_results.pkl/csv` | Per-sample scores for 2 abstractive models |
| `outputs/model_comparison.csv` | Aggregated ROUGE table for all 6 models |
| `outputs/bertscore_results.csv` | BERTScore (P/R/F1) for all 6 models |
| `outputs/rouge_comparison.png` | Grouped bar chart (R1/R2/RL F1) |
| `outputs/rouge_boxplot.png` | Box plot of ROUGE-1 F1 distributions |
| `outputs/extractive_distributions.png` | Histogram of per-model ROUGE-1 F1 |
| `outputs/preprocessing_wordcount.png` | Word count before/after preprocessing |
| `outputs/win_rate.csv` | Win-rate of each model vs Lead-3 baseline |

---

## Results

### Extractive Models — Results (1000 test samples)

| Model | R1-Prec | R1-Rec | R1-F1 | R2-F1 | RL-F1 |
|-------|---------|--------|-------|-------|-------|
| Lead-3 | 22.72% | 46.84% | 29.47% | 11.21% | 19.96% |
| TextRank | 20.55% | 44.98% | 27.21% | 9.32% | 18.38% |
| LSA | 17.97% | 25.67% | 19.97% | 4.50% | 13.56% |
| LexRank | 21.49% | 42.29% | 27.42% | 8.80% | 18.28% |

### Abstractive Models — Results (200 test samples)

| Model | R1-Prec | R1-Rec | R1-F1 | R2-F1 | RL-F1 |
|-------|---------|--------|-------|-------|-------|
| PEGASUS-CNN | 34.32% | 35.27% | 33.82% | 14.37% | 25.42% |
| BART | 34.45% | 38.31% | 35.37% | 15.48% | 26.69% |

### BERTScore — All Models (semantic similarity)

| Model | BERT-P | BERT-R | BERT-F1 |
|-------|--------|--------|---------|
| Lead-3 | 75.69% | 80.45% | 77.95% |
| TextRank | 75.91% | 80.23% | 77.97% |
| LSA | 73.48% | 75.28% | 74.32% |
| LexRank | 75.77% | 80.11% | 77.84% |
| PEGASUS-CNN | 77.65% | 80.16% | 78.85% |
| BART | **80.02%** | **80.94%** | **80.44%** |

> Lead-3 scores highest among extractive models due to CNN/DailyMail's inverted-pyramid writing style.
> BART outperforms PEGASUS as it was fine-tuned on this exact dataset.
> BERTScore reveals abstractive models are semantically stronger than ROUGE alone suggests.

---

## Key Insights

1. **Lead-3** is a surprisingly strong baseline because CNN/DailyMail articles place the most important facts in the first few sentences.
2. **LexRank** outperforms TextRank and LSA among extractive methods due to IDF-weighted similarity capturing term importance more accurately.
3. **BART** consistently achieves the highest ROUGE and BERTScore because it was fine-tuned directly on CNN/DailyMail.
4. **PEGASUS-CNN** (33.82% R1-F1, 78.85% BERT-F1) significantly outperforms the old T5-small (31.38% R1-F1) as it is also fine-tuned on this dataset.
5. **BERTScore** shows abstractive models are semantically stronger than ROUGE alone suggests — PEGASUS nearly matches Lead-3 on ROUGE but beats it by ~1% on BERT-F1.
6. **TextRank** improved after switching to preprocessed (lemmatized, stopword-free) tokens for TF-IDF similarity.

---

## References
- Lewis et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training.
- Zhang et al. (2020). PEGASUS: Pre-training with Extracted Gap-Sentences for Abstractive Summarization.
- Erkan & Radev (2004). LexRank: Graph-based Lexical Centrality.
- Mihalcea & Tarau (2004). TextRank: Bringing Order into Texts.
- Gong & Liu (2001). Generic Text Summarization Using Relevance Measure and Latent Semantic Analysis.
- Zhang et al. (2019). BERTScore: Evaluating Text Generation with BERT.
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/ccdv/cnn_dailymail)

## GitHub Repository

[https://github.com/Chauhanjay0912/NLP-Text-Summarization](https://github.com/Chauhanjay0912/NLP-Text-Summarization)
