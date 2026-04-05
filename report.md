# Comparative Analysis of Extractive and Abstractive Text Summarization Models on News Articles

**Authors:** [Your Name], [Co-author if any]
**Institution:** [Your University / Department]
**Course:** M.Tech — Natural Language Processing (CA-3 Mini Project)
**Date:** 2025

---

## Abstract

Automatic text summarization is a fundamental task in Natural Language Processing (NLP) that aims to condense long documents into shorter, coherent representations while preserving key information. This paper presents a comparative study of six summarization models — four extractive (Lead-3, TextRank, LSA, LexRank) and two abstractive (T5-small, BART-large-cnn) — evaluated on the CNN/DailyMail 3.0.0 benchmark dataset. A five-stage preprocessing pipeline was applied to 1,000 test samples, covering HTML cleaning, sentence tokenization, word tokenization, stop-word removal, and lemmatization. Models were evaluated using ROUGE-1, ROUGE-2, and ROUGE-L metrics (Precision, Recall, F1). Results show that BART-large-cnn achieves the highest overall performance (ROUGE-1 F1: 34.55%, ROUGE-2 F1: 14.51%, ROUGE-L F1: 24.69%), while the simple Lead-3 baseline remains highly competitive among extractive methods (ROUGE-1 F1: 29.47%) due to the inverted-pyramid writing style of CNN/DailyMail articles. Abstractive models outperform extractive models by an average of 6.30% on ROUGE-1 F1. This work provides a reproducible end-to-end pipeline and a practical comparison to guide model selection for news summarization tasks.

---

## Keywords

Text Summarization, Extractive Summarization, Abstractive Summarization, ROUGE Evaluation, TextRank, LSA, LexRank, T5, BART, CNN/DailyMail, Natural Language Processing, Deep Learning

---

## I. Introduction

The exponential growth of digital text — news articles, research papers, social media posts, and reports — has made it increasingly difficult for users to consume information efficiently. Automatic text summarization addresses this challenge by generating concise summaries that retain the most important content from a source document, reducing reading time while preserving informational value.

Text summarization approaches broadly fall into two categories. **Extractive summarization** selects and concatenates the most relevant sentences directly from the source document. These methods are computationally efficient and produce grammatically correct output since they reuse original sentences. **Abstractive summarization**, on the other hand, generates new text that may not appear verbatim in the source, similar to how a human would paraphrase a document. While abstractive methods can produce more fluent and concise summaries, they require significantly more computational resources and are prone to factual hallucinations.

This project implements and compares six models spanning both paradigms on the CNN/DailyMail dataset — one of the most widely used benchmarks for news summarization. The specific contributions of this work are:

1. A reproducible five-stage preprocessing pipeline for the CNN/DailyMail dataset.
2. From-scratch implementations of four extractive models: Lead-3, TextRank, LSA, and LexRank.
3. Evaluation of two state-of-the-art transformer-based abstractive models: T5-small and BART-large-cnn.
4. A comprehensive comparative analysis using ROUGE-1/2/L (Precision, Recall, F1) across all six models.
5. Per-sample win-rate analysis against the Lead-3 baseline.

The remainder of this paper is organized as follows: Section II reviews related work; Section III describes the methodology including dataset, preprocessing, model implementations, and evaluation; Section IV presents conclusions and future directions.

---

## II. Literature Review

### A. Selecting a Template

Early work in automatic text summarization dates to Luhn (1958), who proposed selecting sentences based on word frequency. Edmundson (1969) extended this with positional and cue-word features. These rule-based extractive approaches laid the foundation for graph-based methods.

**TextRank** (Mihalcea & Tarau, 2004) adapted Google's PageRank algorithm to sentence graphs, where nodes represent sentences and edges represent cosine similarity between TF-IDF vectors. Sentences with the highest PageRank scores are selected as the summary. This unsupervised approach requires no training data and generalizes well across domains.

**LexRank** (Erkan & Radev, 2004) introduced IDF-modified cosine similarity into the graph construction, giving higher weight to rare but informative terms. A threshold is applied to remove weak edges, and the resulting stochastic matrix is used for PageRank scoring. LexRank consistently outperforms TextRank on news datasets due to its better handling of term importance.

**Latent Semantic Analysis (LSA)** for summarization (Gong & Liu, 2001) applies Singular Value Decomposition (SVD) to the TF-IDF sentence-term matrix to discover latent topics. Sentences are scored by the L2 norm of their topic vectors, and the top-scoring sentences are selected. LSA captures semantic relationships beyond surface-level word overlap but is sensitive to the number of latent components chosen.

The **Lead-3 baseline** — selecting the first three sentences of a news article — is deceptively strong on CNN/DailyMail due to the inverted-pyramid journalistic style, where the most newsworthy information is placed at the beginning. Nallapati et al. (2016) demonstrated that Lead-3 outperforms many sophisticated extractive models on this dataset.

### B. Maintaining the Integrity of the Specifications

The shift to neural abstractive summarization began with sequence-to-sequence models (Sutskever et al., 2014; Bahdanau et al., 2015). Rush et al. (2015) applied attention-based encoder-decoder models to headline generation. See et al. (2017) introduced the Pointer-Generator Network with a coverage mechanism to reduce repetition and enable copying from the source.

**T5 (Text-to-Text Transfer Transformer)** (Raffel et al., 2020) reframes all NLP tasks as text-to-text problems. The model is pre-trained on the Colossal Clean Crawled Corpus (C4) with a masked span prediction objective. For summarization, the input is prefixed with "summarize:" and the model generates the summary autoregressively. T5-small (~60M parameters, ~240 MB) is the smallest variant, making it suitable for resource-constrained environments.

**BART (Bidirectional and Auto-Regressive Transformers)** (Lewis et al., 2020) combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT). It is pre-trained by corrupting text with various noise functions and learning to reconstruct the original. The `facebook/bart-large-cnn` checkpoint (~400M parameters, ~1.6 GB) is fine-tuned directly on the CNN/DailyMail dataset, giving it a significant advantage on this benchmark. BART has become the de facto standard for news summarization and consistently achieves state-of-the-art ROUGE scores.

Recent advances include PEGASUS (Zhang et al., 2020), which uses gap-sentence generation as a pre-training objective specifically designed for summarization, and PRIMERA (Xiao et al., 2022) for multi-document summarization. However, BART-large-cnn remains a strong and widely reproduced baseline for single-document news summarization.

---

## III. Methodology

### A. Abbreviations and Acronyms

The following abbreviations are used throughout this paper:

| Abbreviation | Full Form |
|---|---|
| NLP | Natural Language Processing |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| SVD | Singular Value Decomposition |
| LSA | Latent Semantic Analysis |
| T5 | Text-to-Text Transfer Transformer |
| BART | Bidirectional and Auto-Regressive Transformers |
| CNN/DM | CNN/DailyMail Dataset |
| F1 | F1-Score (harmonic mean of Precision and Recall) |
| GPU | Graphics Processing Unit |
| CPU | Central Processing Unit |

**Dataset.** The CNN/DailyMail 3.0.0 dataset (Hermann et al., 2015; Nallapati et al., 2016) contains approximately 300,000 English news articles paired with human-written bullet-point summaries ("highlights"). It is hosted on HuggingFace (`ccdv/cnn_dailymail`). For this study, 1,000 samples from the test split were used for extractive evaluation and 200 samples for abstractive evaluation (due to CPU inference time constraints).

**Dataset Statistics (1,000 test samples):**

| Statistic | Sentences/Article | Words (Original) | Words (Processed) | Words (Reference) |
|---|---|---|---|---|
| Mean | 33.4 | 626.3 | 339.4 | 34.6 |
| Std | 19.8 | 352.0 | 188.5 | 9.7 |
| Min | 4 | 73 | 43 | 9 |
| Max | 112 | 1,750 | 1,326 | 71 |

**Preprocessing Pipeline.** A five-stage pipeline was applied to all articles:

1. **Cleaning** — HTML tags, CNN byline prefixes (e.g., "(CNN) --"), and special characters were removed using regular expressions. Extra whitespace was normalized.
2. **Sentence Tokenization** — `nltk.sent_tokenize` was applied to the cleaned text to split articles into individual sentences.
3. **Word Tokenization** — `nltk.word_tokenize` was applied per sentence to produce word-level tokens.
4. **Stop-word Removal** — NLTK's English stop-word list was used to filter non-content words.
5. **Normalization** — All tokens were lowercased and lemmatized using `WordNetLemmatizer`. Non-alphabetic tokens were discarded.

Preprocessing reduced the average word count from 626.3 to 339.4 words per article (a 45.8% reduction), retaining only content-bearing terms for model input.

### B. Authors and Affiliations

**Model Implementations.**

**Lead-3 (Baseline):** Returns the first three sentences of the cleaned article. No training or computation required. Exploits the inverted-pyramid structure of news writing.

**TextRank:** Sentences are vectorized using TF-IDF. A cosine similarity matrix is computed between all sentence pairs, and self-similarity is set to zero. A weighted undirected graph is constructed using NetworkX, and PageRank (max_iter=300, tol=1e-5) is applied to score sentences. The top-3 sentences by score are returned in their original order.

**LSA:** Sentences are vectorized using TF-IDF (max_features=5,000). TruncatedSVD is applied with k = min(5, n_sentences−1, n_terms−1) latent components. Each sentence is scored by the L2 norm of its topic vector. The top-3 sentences are returned in original order.

**LexRank:** IDF values are computed over all sentences in the article. An IDF-weighted cosine similarity matrix is built between all sentence pairs. Similarities below a threshold (0.1) are set to zero. The matrix is row-normalized to form a stochastic matrix. PageRank is applied to score sentences, and the top-3 are returned in original order.

**T5-small:** The HuggingFace `transformers` pipeline is used with model `t5-small`. Each article is truncated to 600 words and prefixed with "summarize: ". Beam search (num_beams=4) is used with max_length=150, min_length=40. Inference runs in batches of 4 on CPU.

**BART-large-cnn:** The HuggingFace `transformers` pipeline is used with model `facebook/bart-large-cnn`. Articles are truncated to 600 words. Beam search (num_beams=4) is used with max_length=142, min_length=56, length_penalty=2.0. Inference runs in batches of 4 on CPU.

**Evaluation Metric.** ROUGE (Lin, 2004) is the standard metric for summarization evaluation. It measures n-gram overlap between the generated summary and the human reference:

- **ROUGE-1**: Unigram overlap between generated and reference summary.
- **ROUGE-2**: Bigram overlap between generated and reference summary.
- **ROUGE-L**: Longest Common Subsequence (LCS) between generated and reference summary.

Each metric reports Precision, Recall, and F1. The `rouge-score` library with `use_stemmer=True` was used for all evaluations.

### C. Figures and Tables

**Table 1: Full ROUGE Score Comparison — All 6 Models**

| Model | Type | R1-Prec | R1-Rec | R1-F1 | R2-F1 | RL-F1 |
|---|---|---|---|---|---|---|
| Lead-3 | Extractive | 22.72% | 46.84% | 29.47% | 11.21% | 19.96% |
| TextRank | Extractive | 20.74% | 44.57% | 27.28% | 9.24% | 18.42% |
| LSA | Extractive | 19.54% | 30.45% | 22.50% | 5.91% | 15.28% |
| LexRank | Extractive | 21.49% | 42.29% | 27.42% | 8.80% | 18.28% |
| T5-small | Abstractive | 29.31% | 35.35% | 31.38% | 11.90% | 22.53% |
| **BART** | **Abstractive** | **29.06%** | **44.90%** | **34.55%** | **14.51%** | **24.69%** |

*Extractive models evaluated on 1,000 test samples; Abstractive models on 200 test samples.*

**Table 2: Win-Rate vs Lead-3 Baseline (ROUGE-1 F1, per-sample, n=200)**

| Model | Wins | Losses | Ties | Win% |
|---|---|---|---|---|
| TextRank | 91 | 105 | 4 | 45.5% |
| LSA | 67 | 133 | 0 | 33.5% |
| LexRank | 89 | 109 | 2 | 44.5% |
| T5-small | 113 | 87 | 0 | 56.5% |
| BART | **138** | 61 | 1 | **69.0%** |

**Figure 1: ROUGE F1 Score Comparison (Grouped Bar Chart)**
*(See `outputs/rouge_comparison.png`)*

The grouped bar chart shows ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for all six models side by side. BART consistently leads across all three metrics. Among extractive models, Lead-3 achieves the highest scores, followed closely by LexRank and TextRank. LSA performs the weakest overall.

**Figure 2: ROUGE-1 F1 Score Distribution (Box Plot)**
*(See `outputs/rouge_boxplot.png`)*

The box plot reveals the per-sample score distributions. BART shows the highest median and a relatively tight distribution, indicating consistent performance. Lead-3 has high recall-driven scores but wider variance. LSA shows the lowest median and the narrowest spread, reflecting its limited ability to select informative sentences.

**Figure 3: Preprocessing Word Count Distributions**
*(See `outputs/preprocessing_wordcount.png`)*

Three histograms show word count distributions at each preprocessing stage: original article, after cleaning, and after stop-word removal and lemmatization. The mean word count drops from 626.3 (original) to 638.8 (cleaned, slightly higher due to byline removal exposing more text) to 339.4 (processed), confirming effective noise reduction.

---

## IV. Conclusion

This paper presented a comprehensive comparison of six text summarization models on the CNN/DailyMail benchmark. The key findings are:

1. **BART-large-cnn is the best overall model**, achieving ROUGE-1 F1 of 34.55%, ROUGE-2 F1 of 14.51%, and ROUGE-L F1 of 24.69%. It wins against Lead-3 on 69% of individual samples, confirming the value of fine-tuning on domain-specific data.

2. **Lead-3 is a surprisingly strong baseline**, achieving ROUGE-1 F1 of 29.47% — higher than TextRank, LSA, and LexRank — due to the inverted-pyramid structure of CNN/DailyMail articles. This highlights the importance of domain-aware baselines.

3. **Abstractive models outperform extractive models on average** by 6.30% on ROUGE-1 F1 (32.97% vs 26.67%), demonstrating the advantage of generative approaches for this task.

4. **T5-small (31.38% R1-F1) beats Lead-3 (29.47%)** despite being a general-purpose model not fine-tuned on CNN/DailyMail, showing the power of large-scale pre-training.

5. **LSA is the weakest extractive model** (22.50% R1-F1), as its topic-based scoring does not align well with the positional bias of this dataset.

6. **LexRank marginally outperforms TextRank** (27.42% vs 27.28% R1-F1) due to IDF-weighted similarity better capturing term importance.

**Limitations.** Abstractive models were evaluated on only 200 samples due to CPU inference time (~27 minutes for BART on 200 samples). Evaluation on the full 1,000 samples may yield slightly different results. Additionally, ROUGE measures lexical overlap and does not capture semantic correctness or factual accuracy.

**Future Work.** Future directions include: (1) evaluating PEGASUS and PRIMERA for comparison; (2) fine-tuning T5-small on CNN/DailyMail to close the gap with BART; (3) incorporating BERTScore for semantic evaluation; (4) extending to multi-document summarization.

---

## GitHub

Project repository: `https://github.com/<your-username>/NLP_CA_3_miniproject`

The repository contains all scripts, requirements, and generated outputs required to fully reproduce this study.

---

## References

1. Luhn, H. P. (1958). The automatic creation of literature abstracts. *IBM Journal of Research and Development*, 2(2), 159–165.

2. Edmundson, H. P. (1969). New methods in automatic extracting. *Journal of the ACM*, 16(2), 264–285.

3. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into texts. *Proceedings of EMNLP 2004*, 404–411.

4. Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. *Journal of Artificial Intelligence Research*, 22, 457–479.

5. Gong, Y., & Liu, X. (2001). Generic text summarization using relevance measure and latent semantic analysis. *Proceedings of SIGIR 2001*, 19–25.

6. Hermann, K. M., Kocisky, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. *Advances in Neural Information Processing Systems (NeurIPS)*, 28.

7. Nallapati, R., Zhou, B., dos Santos, C., Gulcehre, C., & Xiang, B. (2016). Abstractive text summarization using sequence-to-sequence RNNs and beyond. *Proceedings of CoNLL 2016*, 280–290.

8. See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. *Proceedings of ACL 2017*, 1073–1083.

9. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.

10. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *Proceedings of ACL 2020*, 7871–7880.

11. Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74–81.

12. Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization. *Proceedings of ICML 2020*, 11328–11339.
