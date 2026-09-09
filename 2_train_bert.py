"""
STEP 3 — Fine-tune BERT for 3-Class Sentiment Classification
============================================================
Matches the "Transformer-Based Model" slide of the NLP Mini Project:
- Model: Fine-tuned 'bert-base-uncased' (bidirectional contextual embeddings)
- Classification head on [CLS] token
- Hyperparameters:
    - Learning rate: 2e-5
    - Epochs: 4 (within 3-5 range specified in slides)
    - Batch size: 16
    - Max sequence length: 128
    - Output classes: Negative (0), Neutral (1), Positive (2)
- Train / Validation / Test split: 80% / 10% / 10% (matching Step 1 seed 42)
- Saves model to './bert-sentiment-model'
- Exports 'bert_results.json' and appends 'pred_BERT' to 'test_predictions_all.csv'

Install dependencies:
    pip install torch transformers datasets evaluate scikit-learn pandas numpy accelerate

Usage:
    python 2_train_bert.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_CSV = "reviews_preprocessed_5k.csv"
FALLBACK_CSV = "reviews_preprocessed_labeled.csv"
TEXT_COLUMN = "review_text"     # BERT requires full raw context and capitalization
LABEL_COLUMN = "sentiment"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
BATCH_SIZE = 16
RANDOM_STATE = 42
OUTPUT_DIR = "./bert-sentiment-model"
RESULTS_JSON = "bert_results.json"
TEST_PREDICTIONS_CSV = "test_predictions_all.csv"

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing BERT fine-tuning on device: {device.upper()}")
    if device == "cpu":
        print("Note: Running on CPU. Training will take longer; for faster CPU runs, batch size is adjusted automatically.")

    # 1. Load dataset
    target_csv = INPUT_CSV if os.path.exists(INPUT_CSV) else FALLBACK_CSV
    if not os.path.exists(target_csv):
        raise FileNotFoundError(
            f"Could not find '{target_csv}'. Please run 'python 1_preprocess_and_label.py' first!"
        )

    print(f"Loading preprocessed dataset from '{target_csv}'...")
    df = pd.read_csv(target_csv)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df["label"] = df[LABEL_COLUMN].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} total labeled reviews.")
    print("Class distribution:")
    for sentiment, count in df[LABEL_COLUMN].value_counts().items():
        print(f"  {sentiment:<10}: {count} ({count/len(df):.1%})")

    # 2. 80 / 10 / 10 Stratified Split (matching Step 1)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["label"]
    )
    print(f"\nSplit Sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = Dataset.from_pandas(train_df[[TEXT_COLUMN, "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[[TEXT_COLUMN, "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[[TEXT_COLUMN, "label"]].reset_index(drop=True))

    # 3. Sub-word Tokenization
    print(f"\nLoading Tokenizer '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    print("Tokenizing train, val, and test splits...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Model Setup
    print(f"Loading pre-trained '{MODEL_NAME}' with sequence classification head...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Effective batch size for CPU vs GPU
    batch_size = BATCH_SIZE if device == "cuda" else 8
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Training
    print("\n" + "=" * 60)
    print("STARTING BERT FINE-TUNING")
    print("=" * 60)
    trainer.train()

    # 6. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 60)
    print("EVALUATING BERT ON HELD-OUT TEST SET (10%)")
    print("=" * 60)
    test_eval = trainer.evaluate(test_ds)
    print("Test Evaluation Metrics:", test_eval)

    preds_output = trainer.predict(test_ds)
    y_pred_ids = np.argmax(preds_output.predictions, axis=-1)
    y_true_ids = preds_output.label_ids
    y_pred_labels = [ID2LABEL[idx] for idx in y_pred_ids]
    y_true_labels = [ID2LABEL[idx] for idx in y_true_ids]

    acc = accuracy_score(y_true_ids, y_pred_ids)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_ids, y_pred_ids, average="macro", zero_division=0
    )
    labels_order = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels_order).tolist()

    bert_results = {
        "model": "BERT (bert-base-uncased)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "labels": labels_order,
    }

    print("\nBERT Test Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(bert_results, f, indent=2)
    print(f"Saved BERT evaluation results to '{RESULTS_JSON}'.")

    # 7. Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned BERT model checkpoint to '{OUTPUT_DIR}'.")

    # 8. Merge test predictions for comparative and error analysis
    if os.path.exists("test_baseline_predictions.csv"):
        test_df_merge = pd.read_csv("test_baseline_predictions.csv")
        test_df_merge["pred_BERT"] = y_pred_labels
        test_df_merge.to_csv(TEST_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
        print(f"Merged baseline and BERT predictions into '{TEST_PREDICTIONS_CSV}'.")


if __name__ == "__main__":
    main()
