# NLP Text Summarization — Comparative Study

A comparative analysis of **6 text summarization models** (4 extractive + 2 abstractive) on the CNN/DailyMail dataset, with full ROUGE evaluation and an IEEE-format research report.

> M.Tech NLP — CA-3 Mini Project

---

## Results

| Model | Type | R1-F1 | R2-F1 | RL-F1 |
|-------|------|-------|-------|-------|
| Lead-3 | Extractive | 29.47% | 11.21% | 19.96% |
| TextRank | Extractive | 27.28% | 9.24% | 18.42% |
| LSA | Extractive | 22.50% | 5.91% | 15.28% |
| LexRank | Extractive | 27.42% | 8.80% | 18.28% |
| T5-small | Abstractive | 31.38% | 11.90% | 22.53% |
| **BART-large-cnn** | **Abstractive** | **34.55%** | **14.51%** | **24.69%** |

BART wins against Lead-3 on **69%** of individual samples. Abstractive models outperform extractive by **6.30%** on average ROUGE-1 F1.

---

## Models

| # | Model | Approach |
|---|-------|----------|
| 1 | Lead-3 | First 3 sentences (baseline) |
| 2 | TextRank | TF-IDF → cosine similarity graph → PageRank |
| 3 | LSA | TF-IDF → TruncatedSVD → L2 norm scoring |
| 4 | LexRank | IDF-weighted cosine → stochastic graph → PageRank |
| 5 | T5-small | `t5-small`, beam search, ~240 MB |
| 6 | BART-large-cnn | `facebook/bart-large-cnn`, fine-tuned on CNN/DM, ~1.6 GB |

---

## Dataset

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/ccdv/cnn_dailymail) — ~300k English news articles with human-written reference summaries.

- Extractive evaluation: **1,000** test samples
- Abstractive evaluation: **200** test samples

---

## Project Structure

```
NLP-Text-Summarization/
├── 1_preprocessing.py        # Cleaning, tokenization, stopword removal, lemmatization
├── 2_extractive_models.py    # Lead-3 | TextRank | LSA | LexRank
├── 3_abstractive_models.py   # T5-small | BART-large-cnn
├── 4_evaluation.py           # ROUGE comparison, bar chart, box plot, win-rate
├── 5_demo.py                 # Interactive CLI demo (all 6 models)
├── requirements.txt
├── report.md                 # Full research report (Markdown)
├── report.tex                # Full research report (IEEE LaTeX)
└── outputs/
    ├── rouge_comparison.png
    ├── rouge_boxplot.png
    ├── extractive_distributions.png
    ├── preprocessing_wordcount.png
    ├── model_comparison.csv
    ├── win_rate.csv
    ├── extractive_results.csv
    └── preprocessing_stats.csv
```

---

## Setup

```bash
git clone https://github.com/<your-username>/NLP-Text-Summarization.git
cd NLP-Text-Summarization
pip install -r requirements.txt
```

NLTK data downloads automatically on first run.

---

## How to Run

```bash
# Step 1 — Preprocessing (saves preprocessed_data.pkl)
python 1_preprocessing.py

# Step 2 — Extractive models
python 2_extractive_models.py

# Step 3 — Abstractive models (downloads T5 ~240MB + BART ~1.6GB on first run)
python 3_abstractive_models.py

# Step 4 — Full evaluation & plots
python 4_evaluation.py

# Step 5 — Interactive demo
python 5_demo.py
```

---

## Output Plots

### ROUGE F1 Comparison
![ROUGE Comparison](outputs/rouge_comparison.png)

### ROUGE-1 F1 Distribution (Box Plot)
![Box Plot](outputs/rouge_boxplot.png)

### Extractive Model Distributions
![Extractive Distributions](outputs/extractive_distributions.png)

### Preprocessing Word Count
![Preprocessing](outputs/preprocessing_wordcount.png)

---

## Preprocessing Pipeline

| Step | Operation | Tool |
|------|-----------|------|
| 1 | HTML cleaning, CNN byline removal | `re` |
| 2 | Sentence tokenization | `nltk.sent_tokenize` |
| 3 | Word tokenization | `nltk.word_tokenize` |
| 4 | Stop-word removal | `nltk.corpus.stopwords` |
| 5 | Lowercase + lemmatization | `WordNetLemmatizer` |

Average word count reduced from **626.3 → 339.4** (45.8% reduction).

---

## Report

A full IEEE-format research report is included:
- [`report.md`](report.md) — Markdown version
- [`report.tex`](report.tex) — LaTeX version (compile with `pdflatex report.tex`)

---

## References

- Lewis et al. (2020). BART. *ACL 2020.*
- Raffel et al. (2020). T5. *JMLR.*
- Erkan & Radev (2004). LexRank. *JAIR.*
- Mihalcea & Tarau (2004). TextRank. *EMNLP.*
- Gong & Liu (2001). LSA Summarization. *SIGIR.*
- Lin (2004). ROUGE. *ACL Workshop.*
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/ccdv/cnn_dailymail)
