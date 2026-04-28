# Resume screening and job-posting match ranking

End-to-end pipeline: download labeled resume text from Kaggle, train a **supervised** text classifier (**Multinomial Naive Bayes** on **TF-IDF** features), evaluate on a held-out test set, then—optionally—**rank the top-k resumes** for each plain-text **job posting** using predicted category plus **cosine similarity** in the same vector space.

---

## Project scope

| In scope | Out of scope (limitations / future work) |
|----------|------------------------------------------|
| Multiclass resume classification by job category | Hiring decisions, bias/fairness audits, legal compliance |
| Screening-style ranking: posting → predicted category → top similar resumes in that category | Training on job postings as labels (we do not have relevance labels posting↔resume) |
| Reproducible scripts, saved figures and ranking text files | Live API, database, or real ATS integration |
| English text; fixed vocabulary from training resumes | Multilingual postings, embeddings (BERT, etc.) |

---

## Dataset

| Item | Detail |
|------|--------|
| **Source** | [Kaggle: Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (`snehaanbhawal/resume-dataset`), fetched via Kaggle API into `data/`. |
| **Size** | **2,484** rows after filtering to the categories listed in `src/data_load.py` (full public release of this dataset). |
| **Categories** | **24** job categories (e.g. `INFORMATION-TECHNOLOGY`, `HEALTHCARE`, `FINANCE`, …). Counts are roughly balanced across many categories (~120 per large class) with a **long tail** (smaller classes in the ~20–60 range). |
| **Columns used** | `Resume_str` (full resume text), `Category` (string label). |

Other columns, if present in a future CSV revision, are ignored; only these two drive training and ranking.

---

## Repository layout

```
resume-classifier/
├── .env                      # KAGGLE_USERNAME, KAGGLE_KEY (not committed)
├── data/                     # Downloaded CSV + unzip (gitignored except placeholders)
├── job_postings/             # Your .txt job postings (optional); see below
├── results/                  # confusion_matrix.png, ranking_*.txt, etc.
├── src/
│   ├── data_load.py          # Kaggle download + category filter
│   ├── preprocess.py         # clean_text(), TF-IDF (fit on train split only)
│   ├── model.py              # MultinomialNB train/predict
│   ├── evaluate.py           # accuracy, classification report, confusion matrix
│   └── rank.py               # job posting → predict_proba → cosine top-k
├── main.py                   # Full pipeline
├── requirements.txt
└── README.md
```

---

## Setup

1. Create a `.env` file in `resume-classifier/` with:
   ```bash
   KAGGLE_USERNAME=yourusername
   KAGGLE_KEY=yourapikey
   ```
2. Obtain your API key from [Kaggle → Settings → API](https://www.kaggle.com/settings).
3. Install dependencies: `pip install -r requirements.txt`
4. Run from the `resume-classifier` directory: `python main.py`

---

## How to run

```bash
cd resume-classifier
python main.py
```

## Inference (predict / match without retraining)

After you’ve run `main.py` at least once, the trained artifacts are saved under `results/models/`.
You can then run inference on new text without retraining:

```bash
cd resume-classifier

# Predict category for a new job posting
.venv/bin/python infer.py predict-posting --text-file job_postings/SomePosting.txt

# Predict category for a new resume
.venv/bin/python infer.py predict-resume --text-file path/to/resume.txt

# Match a posting to the top resume(s) using cosine similarity (default top-k=5)
.venv/bin/python infer.py match-posting --text-file job_postings/SomePosting.txt --top-k 1
```

**Job postings (optional):** Add one or more UTF-8 files named `*.txt` under `job_postings/`. Each file is one posting; the filename (without `.txt`) labels the run. If no `.txt` files are present, ranking is skipped and the rest of the pipeline still runs.

**Outputs (typical):**

- Console: accuracy and sklearn `classification_report`
- `results/confusion_matrix.png`
- For each `job_postings/<name>.txt`: `results/ranking_<name>.txt` (predicted category, top-5 class probabilities, top-k resume indices, cosine scores, short snippets)

---

## Experiments (model comparison + cross-validation)

To address the “spiral approach” / model exploration expectation, this repo includes a lightweight CV runner that compares common text baselines on the same TF‑IDF features.

Run:

```bash
cd resume-classifier
python3 experiments.py
```

Optional (try bigrams):

```bash
python3 experiments.py --bigrams
```

**5-fold CV results (macro-F1 prioritized):**

| Model | CV macro-F1 (mean ± std) | CV accuracy (mean ± std) |
|---|---:|---:|
| LinearSVC | 0.6464 ± 0.0268 | 0.6952 ± 0.0173 |
| LogisticRegression | 0.6100 ± 0.0078 | 0.6687 ± 0.0111 |
| MultinomialNB (alpha=0.1) | 0.5209 ± 0.0076 | 0.5789 ± 0.0078 |
| MultinomialNB (alpha=0.5) | 0.5059 ± 0.0120 | 0.5753 ± 0.0082 |
| MultinomialNB (alpha=1.0) | 0.4967 ± 0.0151 | 0.5725 ± 0.0131 |
| MultinomialNB (alpha=2.0) | 0.4809 ± 0.0089 | 0.5644 ± 0.0111 |

**Takeaway:** on this dataset + feature setup, **Linear SVM (LinearSVC)** is the strongest baseline by macro‑F1, while **Naive Bayes** is simpler/faster but less accurate.

---

## Final project checklist (mapped to this repo)

### 1. Frame the problem and look at the big picture

**a. Objective (business terms)**  
Help recruiters or hiring systems **shortlist candidates** from a resume pool when a **new job description** arrives: infer which **job category** the posting resembles, then surface the **most textually similar** resumes in that category as a first-pass screen (not a hire/no-hire decision).

**b. How the solution is used**  
Run offline: train/evaluate on the labeled corpus; drop new postings as `.txt` files and read ranked lists from `results/`. Suitable as a **prototype** for exploratory screening or coursework demos.

**c. Assumptions**

- Resume **category labels** are a reasonable proxy for “who belongs in that bucket” when matching a posting classified to the same label.
- **English** text; same preprocessing for resumes and postings.
- **TF-IDF vocabulary** is built from **training split only** (see preprocessing), so deployment postings use the same word universe as training.
- Job postings are **plain text**; formatting loss is acceptable.
- Similarity in bag-of-words space approximates **relevance** well enough for a student project (not validated against human HR judgments).

---

### 2. Get the data and explore the data

**a. Workspace**  
Data land in `data/` after Kaggle download; `results/` holds outputs.

**b. Copy / sample**  
Exploration can use the **full filtered frame** in memory (~2.5k rows); for heavier EDA you can sample in a notebook while keeping `main.py` on the full data. The pipeline does not subsample for training beyond the 80/20 split.

**c. Attributes**

| Attribute | Meaning | Type | Missing | Quality / noise | Usefulness |
|-----------|---------|------|---------|-----------------|------------|
| `Resume_str` | Full resume body | Text (string) | Dropped with `dropna` in loader | HTML-like noise, typos, varied length | **Primary input** for features |
| `Category` | Job domain label | Categorical (string) | Dropped if missing | Consistent labels after normalization | **Target** for supervised learning |

**d. Supervised vs unsupervised; algorithms**  
**Supervised learning:** predict `Category` from text → **Multinomial Naive Bayes** on **TF-IDF** vectors (standard for sparse word counts).  
**Ranking step** is not a second trained model: it uses **cosine similarity** between the posting vector and resume vectors **within** the predicted category—an **information retrieval**-style decision rule on top of the classifier.

---

### 3. Prepare the data

**a. Copies**  
`preprocess_data` operates on a **copy** of the dataframe for cleaning.

**b. Cleaning** (`clean_text` in `src/preprocess.py`)

- Lowercase, strip URLs, remove non-letters, collapse spaces, **NLTK English stopwords**.
- Rows with missing `Resume_str` or `Category` are **dropped** in `data_load.py` (not imputed).
- **Outliers:** extremely short/long text is not clipped; empty strings after cleaning become degenerate documents (limitation).

**c. Feature selection (*)**  
**TF-IDF** with `max_features=1000` caps vocabulary size and acts as **feature selection**.

**d. Feature engineering (*)**  
Word unigrams via TF-IDF; no extra binning or numeric features. N-grams or embeddings would be optional extensions.

**e. Scaling**  
TF-IDF applies **L2 normalization** per document by default in scikit-learn’s implementation (`norm="l2"`), which is appropriate for cosine similarity and NB inputs.

---

### 4. Explore many models (spiral approach)

**Current codebase:** multiple baseline classifiers are supported via `experiments.py` (NB, Logistic Regression, Linear SVM) on the same TF‑IDF features.  
**Comparison / iteration:** `experiments.py` runs **Stratified 5-fold cross-validation** and prints a copy/paste markdown table of macro‑F1 + accuracy, plus a tiny NB `alpha` sweep.

**Significant “variables” for text models:** interpret via **high TF-IDF weight** terms per class (e.g. `feature_names_out` with NB `feature_log_prob_`)—not wired as a script here, but standard for reports.

---

### 5. Fine-tune and combine

**Hyperparameters / CV:** `max_features`, `alpha`, `test_size`, and `random_state` are **fixed constants** in code; there is **no** `GridSearchCV` or cross-validation loop yet. Treating `max_features` or `alpha` as tunables with CV is a documented improvement path.  
**Ensembles:** not implemented; a simple future option is combining **NB class-probability** with **cosine rank** as a weighted score.

---

### 6. Present the solution

**a–b. Storyline**  
Lead with the **business goal** (shortlist from postings), then **data → TF-IDF → NB → optional ranking**.  
**c. What worked / limitations**  
Worked: end-to-end automation, interpretable probabilities, cosine ranking in the same space. Limits: no gold “relevance” labels for posting–resume pairs; small classes hurt recall; bag-of-words misses semantics.  
**d. Visuals**  
Confusion matrix in `results/`; for slides add category distribution or top terms per class from your own EDA.

---

## Remarks (documentation & automation)

- **Documentation:** This README plus docstrings on `clean_text`, `preprocess_data`, `evaluate_and_save`, and `rank_postings`.  
- **Automation:** `main.py` runs load → preprocess → train → evaluate → rank in one command.  
- **Functions:** All cleaning and vectorization live in `src/preprocess.py`; evaluation in `src/evaluate.py`; ranking in `src/rank.py`—so the same transforms apply to **new resumes** or **new postings** once the vectorizer and model are fitted in process (for production you would persist `vectorizer` + model with `joblib`).

---

## License / data credit

Resume data: use per Kaggle dataset license; job postings you add locally are your responsibility to cite if copied from third-party listings.
