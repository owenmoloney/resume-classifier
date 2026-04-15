# Final Project Workflow Revision

## 1. Frame the problem and look at the big picture

### a. Business objective
- Build an automated resume screening pipeline that helps recruiters identify resumes most likely to match a job posting.
- The solution should reduce manual resume review effort by predicting a job category and ranking candidate resumes by textual similarity.

### b. How the solution is used
- A recruiter provides a job posting as a `.txt` file.
- The system predicts the most likely job category for the posting.
- It ranks resumes in the predicted category by cosine similarity and produces a top candidate list.
- Output is written to `results/ranking_<posting>.txt`.

### c. Assumptions made so far
- The dataset categories are meaningful buckets for resume-job matching.
- Text similarity within the predicted category is a valid proxy for good matches.
- Resume quality, work history, and soft skills are not directly modeled.
- The job posting and resume texts are preprocessed for bag-of-words representation.

## 2. Get the data and explore the data to gain insights

### a. Workspace and data storage
- The working code lives in `resume-classifier/`.
- Raw data is stored under `data/data/data/` and `data/Resume/Resume.csv`.
- The same data tree is mirrored inside `resume-classifier/data/`.

### b. Copy data for exploration
- Use local copies of the dataset when experimenting.
- Sampling can be done by creating smaller subsets from the dataset folders if needed.

### c. Attribute study
- The main input features are text fields from resumes and job postings.
- Labels are categorical job classes like `ACCOUNTANT`, `ENGINEERING`, `HR`, etc.

#### i. What each attribute represents
- Resume text: full text of the applicant's resume.
- Category label: job bucket assigned to that resume.
- Job posting text: target document for which matching resumes are ranked.

#### ii. Attribute types
- Text fields: unstructured data (string).
- Labels: categorical.
- Index/ID fields: integer.

#### iii. Missing values
- The current pipeline assumes non-empty text fields.
- Missing or malformed entries should be cleaned in preprocessing.

#### iv. Data quality and noisiness
- Text data may contain noise such as punctuation, line breaks, and formatting artifacts.
- The pipeline uses preprocessing to normalize text and remove noise.
- Outliers are primarily resumes with extreme length or strange formatting.

#### v. Usefulness for the task
- Resume text and posting text are essential.
- Categories are required for supervised classification.
- Additional metadata is not currently used.

### d. Problem framing and algorithm choice
- This is a supervised learning problem for category prediction.
- The pipeline uses TF-IDF feature extraction and Multinomial Naive Bayes for classification.
- Ranking is performed using cosine similarity over the same vector space.

## 3. Prepare the data

### a. Work on copies of the data
- The feature pipeline operates on a copy of the processed text rather than mutating raw files.
- Keep original raw data intact.

### b. Data cleaning
- Preprocess text by lowercasing, stripping whitespace, and normalizing line breaks.
- Remove or handle missing text entries before vectorization.
- Outliers can be filtered by extreme document length if necessary.

#### i. Fix or remove outliers
- Identify resumes with very short or very long text and examine them manually.
- Remove only if they degrade model quality.

#### ii. Handle missing values
- Drop rows with missing text if they cannot be repaired.
- Alternatively, fill missing entries with an empty string during preprocessing.

### c. Feature selection
- The model limits TF-IDF vocabulary size (e.g. `max_features=1000`) to select the most informative terms.

### d. Feature engineering
- Primary feature engineering is text preprocessing and TF-IDF vectorization.
- Future improvements could include n-grams or keyword extraction.

### e. Feature scaling
- TF-IDF output is already normalized as part of vectorization.
- Cosine similarity uses normalized vectors.

## 4. Explore many different models and short-list the best ones

### a. Train many models
- Start with Naive Bayes because it is fast and well-suited for text.
- Compare with alternative classifiers such as Logistic Regression or SVM if time permits.

### b. Measure performance
- Use accuracy and classification reports to evaluate the classifier.
- Evaluate ranking quality by inspecting ranked output for job postings.

### c. Analyze significant variables
- In text classification, the most significant features are high-weight TF-IDF terms.
- Track which words strongly influence category predictions.

### d. Iterate feature selection and engineering
- Revisit TF-IDF vocabulary size, tokenization, and stopword handling.
- Explore removing noisy resumes or low-information documents.

### e. Iterations
- Perform multiple cycles: train, evaluate, adjust preprocessing, and retrain.

### f. Short-list promising models
- Prefer models with different error behavior.
- Example: Naive Bayes for baseline, Logistic Regression for a stronger alternative.

## 5. Fine-tune models and combine them

### a. Hyperparameter tuning
- Treat preprocessing choices as hyperparameters.
- Examples: TF-IDF max features, n-gram range, vocabulary pruning.
- Use cross-validation to compare settings.

#### i. Data transformation choices
- Test different missing-value strategies: drop vs. fill.
- Evaluate whether to use raw text or cleaned text variants.

#### ii. Avoid overfitting
- Use validation holdout or cross-validation to confirm performance.
- Keep the model simple when text data is sparse.

### b. Ensemble methods
- A future improvement is to combine multiple classifiers or ranking heuristics.
- Ensembles could improve robustness across different job categories.

## 6. Present the solution

### a. Presentation structure
- Start with the business objective and the problem statement.
- Show how the solution maps to the recruiter workflow.

### b. Why the solution works
- Explain category prediction plus similarity ranking.
- Emphasize that the system reduces manual screening by focusing on likely matches.

### c. Interesting findings
- Document what worked and what did not.
- Note assumptions and limitations.

#### i. What worked
- TF-IDF + Naive Bayes is effective for category prediction on labeled resumes.
- Cosine similarity ranking gives readable candidate lists.

#### ii. What did not work / limitations
- The pipeline does not model hiring quality.
- It relies on text data only and ignores resume structure, experience, and skills beyond keywords.

### d. Visualizations and communication
- Use example ranking outputs and category prediction results.
- Keep findings concise and supported by logs, sample files, and charts if available.

## Remarks
- Document each major step and keep the process reproducible.
- Automate preprocessing and feature transformation through functions.
- A reusable pipeline makes it easier to apply the model to a fresh dataset later.
- Treat data cleaning and transformation as part of the model workflow.

## Current Project Status
- The repository already includes a working classification and ranking pipeline.
- The current output format has been improved to show full resume text with better formatting.
- The next task is to link this workflow to a final presentation and document the experiments.
