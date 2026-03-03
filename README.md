# Sentiment-Free-Nutrient-Food

# Sentiment Analysis of Indonesian Tweets (Technical Assessment)

This repository contains a Jupyter Notebook for a **sentiment analysis technical assessment** (Data Science Intern / Data Analyst challenge).  
The notebook builds an end-to-end mini pipeline to **clean, normalize, and label Indonesian tweet texts**, then **visualize** and **extract insights** from public sentiment around the *“Makan Siang Gratis (MBG)”* program.

> Notebook: `MBG_Sentiment_Analysis.ipynb`

---

## Project Goals

- Ingest a tweet-text dataset (CSV).
- Perform Indonesian-specific text preprocessing:
  - basic cleaning (mentions, hashtags, URLs, punctuation, numbers)
  - normalization of slang/alay words into formal Indonesian
  - stopword removal (including custom domain words)
  - stemming (Bahasa Indonesia)
- Automatically label sentiment using a pre-trained transformer model.
- Visualize sentiment and common terms (word clouds + top frequent words).
- Summarize key themes from positive vs. negative discourse.

---

## Data & Files

The notebook expects these local files (placed in the same working directory as the notebook):

- `Dataset MBG.csv` — main dataset containing at least a `text` column (tweet texts).
- `Colloquial-indonesian-lexicon.csv` — slang → formal lexicon used for normalization.
- `Stopwords-indonesia.txt` — Indonesian stopwords list (one word per line).

---

## Tech Stack

- **Language**: Python 3
- **Environment**: Jupyter Notebook / Google Colab compatible

### Main Libraries Used

- **Data processing**: `pandas`, `numpy`
- **Progress display**: `tqdm`
- **Visualization**: `matplotlib`, `wordcloud`
- **Indonesian NLP**:
  - `PySastrawi` (Sastrawi stemmer)
  - custom slang normalization using `Colloquial-indonesian-lexicon.csv`
- **Sentiment model (Transformers)**:
  - `transformers` (`AutoTokenizer`, `AutoModelForSequenceClassification`, `pipeline`)
  - Pretrained model: `mdhugol/indonesia-bert-sentiment-classification` (Hugging Face)

---

## Methodology (Notebook Flow)

### 1) Data Ingestion
- Load the dataset from `Dataset MBG.csv` into a DataFrame.

### 2) Text Preprocessing
- **Cleaning**: remove mentions, hashtags, RT tokens, links, numbers, punctuation; lowercase and trim spaces.
- **Normalization**: replace slang/alay words with formal equivalents based on the lexicon file.
- **Stopword removal**:
  - remove Indonesian stopwords from `Stopwords-indonesia.txt`
  - remove additional custom “domain words” (e.g., highly frequent topic words such as *makan, siang, gratis, prabowo, program*, etc.) to sharpen the theme extraction.
- **Stemming**: Indonesian stemming using Sastrawi.

Generated columns in the DataFrame:
- `text_cleaning`
- `normalized_alay`
- `no_stopwords_text`
- `stemmed_text`

### 3) Sentiment Labelling
- Sample `n=3000` rows from the dataset.
- Run transformer sentiment classification on `stemmed_text`.
- Map model labels to human-readable labels:
  - `LABEL_0 → positive`
  - `LABEL_1 → neutral`
  - `LABEL_2 → negative`
- Keep **high-confidence** predictions with `sentiment_score >= 0.95` for analysis (`top_sentiment`).

### 4) Visualization
- Sentiment distribution (counts & percentages).
- Word clouds for positive and negative sentiment.
- Top frequent words per sentiment label.

### 5) Insights
- Narrative interpretation of:
  - sentiment distribution
  - recurring themes in negative vs. positive sentiment

---

## How to Run

### Option A — Google Colab
1. Upload the notebook and required data files (`Dataset MBG.csv`, `Colloquial-indonesian-lexicon.csv`, `Stopwords-indonesia.txt`) to Colab.
2. Open and run the notebook cells from top to bottom.
3. The notebook installs required packages via `pip`:
   ```bash
   pip install transformers PySastrawi wordcloud
   ```

### Option B — Local (Jupyter)
1. Create/activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install pandas numpy tqdm matplotlib wordcloud transformers PySastrawi
   ```
3. Ensure the required dataset/supporting files are in the same folder as the notebook.
4. Run:
   ```bash
   jupyter notebook
   ```

---

## Outputs

Running the notebook produces:
- A labelled dataset sample (`train`) with:
  - `sentiment_label` and `sentiment_score`
- A filtered high-confidence subset (`top_sentiment`)
- Plots:
  - sentiment distribution
  - word clouds per sentiment
  - top words per sentiment
- Written insights summarizing public discourse themes.

---

## Notes & Assumptions

- The notebook performs sentiment inference on a sample (3,000 rows) for practicality.
- Filtering by high confidence (`>= 0.95`) is used to reduce noisy predictions before visualization/insight extraction.
- Custom stopword deletions are topic-specific and can be adjusted depending on the analysis goal.

---

## License

This project is provided for assessment/demo purposes. Add a license if you plan to redistribute.
