"""
evaluate_transformer.py

Loads your fine-tuned transformer NER model (saved by train_transformer.py
into ./transformer_ner_model) and produces the same paper-ready evaluation
artifacts as the CRF script: classification_report (CSV + LaTeX), a
confusion matrix, and an F1 bar chart -- written to ./eval_transformer/

Run this on the same machine/environment where you trained the model
(needs `transformers`, `torch`, `datasets`).

Run:
    python3 evaluate_transformer.py
"""
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data_prep import load_sentences
from evaluation_utils import full_evaluation_report

MODEL_DIR = "./transformer_ner_nlpaueb_legal-bert-base-uncased"


def build_test_split():
    docs = load_sentences("data.csv")
    random.Random(42).shuffle(docs)  # must match the seed used in train_transformer.py

    n = len(docs)
    n_train = int(n * 0.7)
    n_dev = int(n * 0.15)
    test_docs = docs[n_train + n_dev:]

    sents = []
    for doc in test_docs:
        sents.extend(doc)
    return sents


def predict_sentence(tokens, tokenizer, model, id2label, device):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits

    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # take the prediction of only the FIRST subword of each original word,
    # matching how labels were aligned during training
    aligned_preds = []
    prev_word_id = None
    for pred, word_id in zip(preds, word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            aligned_preds.append(id2label[pred])
        prev_word_id = word_id

    # safety: pad/truncate in the rare case lengths don't line up
    # (can happen if a document got truncated at max_length)
    if len(aligned_preds) < len(tokens):
        aligned_preds += ["O"] * (len(tokens) - len(aligned_preds))
    else:
        aligned_preds = aligned_preds[: len(tokens)]

    return aligned_preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    id2label = model.config.id2label

    test_sents = build_test_split()
    print(f"Test sequences: {len(test_sents)}")

    y_true, y_pred = [], []
    for sent in test_sents:
        tokens = [tok for tok, _ in sent]
        true_tags = [tag for _, tag in sent]
        pred_tags = predict_sentence(tokens, tokenizer, model, id2label, device)
        y_true.append(true_tags)
        y_pred.append(pred_tags)

    print("\nGenerating paper-ready evaluation artifacts...")
    full_evaluation_report(y_true, y_pred, out_dir="eval_transformer_ner_nlpaueb_legal-bert-base-uncased", title="law-ai/InLegalBERT")


if __name__ == "__main__":
    main()