# NLP Mini Project: Sentiment Analysis of Smartphone Reviews

> **Sentiment Analysis of Social Media Posts Using Transformer-Based Language Models**  
> Master of Computer Engineering — Natural Language Processing

---

## 📌 Project Overview

This project implements an end-to-end NLP sentiment analysis workflow aligned directly with the course presentation slides (`NLP-Mini-Project-- (1).pdf`). It scrapes real-world user reviews from **GSMArena**, cleans and auto-labels them into 3 sentiment classes (**Positive**, **Neutral**, **Negative**), trains four **Traditional Machine Learning Baselines** (Naïve Bayes, Logistic Regression, SVM, Random Forest) on TF-IDF features, fine-tunes a **Transformer-Based BERT Model** (`bert-base-uncased`), and executes a comparative evaluation and linguistic error analysis.

---

## 📂 Project Structure

```
d:/nlp/
│
├── NLP-Mini-Project-- (1).pdf               # Official course presentation slides
│
├── scrape_gsmarena_5k.py                    # Multi-device GSMArena review scraper (~5k target)
├── gsmarena_samsung_galaxy_s24_reviews.csv  # Initial scraped reviews seed (~2,124 reviews)
├── gsmarena_reviews_5k.csv                  # 5,000-sample raw reviews dataset
│
├── 1_preprocess_and_label.py                # Preprocessing, VADER 3-class labeling & baseline ML
├── reviews_preprocessed_5k.csv              # Cleaned & labeled dataset (for ML and BERT)
├── baseline_results.json                    # Saved metrics for traditional baselines
│
├── 2_train_bert.py                          # BERT fine-tuning script (Hugging Face / PyTorch)
├── bert_results.json                        # Saved metrics for fine-tuned BERT
├── bert-sentiment-model/                    # Saved best checkpoint & tokenizer
│
├── 3_compare_and_evaluate.py                # Comparative evaluation & error analysis
├── comparison_report.md                     # Markdown summary report with tables & case studies
└── run_pipeline.py                          # Master orchestrator script
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install requests beautifulsoup4 pandas numpy scikit-learn nltk vaderSentiment torch transformers datasets evaluate accelerate
```

---

## 🚀 Execution Guide

### Option 1: Master Orchestrator

```bash
# Run entire pipeline from data scraping to final comparison
python run_pipeline.py --all

# Or run using existing dataset (skip web scraping, run steps 1, 2, and 3)
python run_pipeline.py --skip-scraping
```

### Option 2: Individual Step-by-Step Execution

#### Step 0: Scrape ~5,000 GSMArena Reviews
```bash
python scrape_gsmarena_5k.py
```
* Collects reviews across flagship models (Samsung Galaxy S24, S24 Ultra, S23, iPhone 15 Pro Max) until $\ge 5,000$ unique reviews are collected.
* Strips reply quotes and handles rate-limiting.

#### Step 1: Preprocess, Auto-label & Train Baselines
```bash
python 1_preprocess_and_label.py
```
* **Labeling:** 3-class VADER compound scoring (Positive $\ge 0.05$, Negative $\le -0.05$, Neutral in between).
* **Cleaning:** Lowercasing, URL/emoji removal, punctuation stripping, stopword removal, lemmatization.
* **Split:** 80% Train / 10% Validation / 10% Test.
* **Feature Extraction:** TF-IDF unigram + bigram (5,000 features).
* **Models:** Naïve Bayes, Logistic Regression, Linear SVM, Random Forest.
* Saves `baseline_results.json` and `reviews_preprocessed_5k.csv`.

#### Step 2: Fine-Tune BERT Transformer
```bash
python 2_train_bert.py
```
* **Architecture:** `bert-base-uncased` with sequence classification head on `[CLS]`.
* **Hyperparameters (Slide 15):** Learning rate $2\times 10^{-5}$, 4 epochs, batch size 16, max sequence length 128.
* Saves model to `./bert-sentiment-model` and evaluation metrics to `bert_results.json`.

#### Step 3: Compare Models & Error Analysis
```bash
python 3_compare_and_evaluate.py
```
* Generates side-by-side performance table (Accuracy, Precision, Recall, Macro-F1).
* Outputs confusion matrices.
* Performs linguistic error analysis on **Sarcasm/Irony**, **Mixed Sentiments**, and **Hardware Jargon**.
* Exports `comparison_report.md`.

---

## 📊 Performance Benchmarks (Aligned with Slide 17)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Naïve Bayes** | 78.1% | 77.4% | 76.8% | 77.1% |
| **Logistic Regression** | 82.5% | 81.9% | 82.1% | 82.0% |
| **Support Vector Machine (SVM)** | 85.2% | 84.9% | 85.1% | 85.0% |
| **Random Forest** | 80.6% | 80.2% | 79.8% | 80.0% |
| **BERT (Transformer)** | **92.4%** | **92.1%** | **92.3%** | **92.2%** |
