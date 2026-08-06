"""
data_prep.py
Loads the corrected BIO-tagged CSV (id, token, bio_tag) and converts it into
per-sentence lists of (token, tag) pairs, split by document id so that no
document leaks across train/dev/test.
"""
import pandas as pd
import random


def load_sentences(csv_path, encoding="utf-8-sig"):
    """
    Returns a list of documents, each document is a list of sentences,
    each sentence is a list of (token, tag) tuples.
    Blank rows (id and token both NaN) mark sentence boundaries within a document.
    A change in `id` marks a new document.
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    documents = []
    current_doc = []
    current_sent = []
    current_id = None

    for _, row in df.iterrows():
        tok, tag, doc_id = row["token"], row["bio_tag"], row["id"]

        # blank separator row -> end of sentence
        if pd.isna(tok) and pd.isna(doc_id):
            if current_sent:
                current_doc.append(current_sent)
                current_sent = []
            continue

        if pd.isna(tok) or pd.isna(tag):
            continue

        # new document id -> flush
        if current_id is not None and doc_id != current_id:
            if current_sent:
                current_doc.append(current_sent)
                current_sent = []
            if current_doc:
                documents.append(current_doc)
                current_doc = []

        current_id = doc_id
        current_sent.append((str(tok), str(tag)))

    if current_sent:
        current_doc.append(current_sent)
    if current_doc:
        documents.append(current_doc)

    return documents


def split_documents(documents, train_frac=0.7, dev_frac=0.15, seed=42):
    """Splits at the DOCUMENT level to avoid leakage between train/dev/test."""
    docs = documents[:]
    random.Random(seed).shuffle(docs)

    n = len(docs)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    train_docs = docs[:n_train]
    dev_docs = docs[n_train:n_train + n_dev]
    test_docs = docs[n_train + n_dev:]

    def flatten(doc_list):
        sents = []
        for doc in doc_list:
            sents.extend(doc)
        return sents

    return flatten(train_docs), flatten(dev_docs), flatten(test_docs)


if __name__ == "__main__":
    docs = load_sentences("data.csv")
    print(f"Documents: {len(docs)}")
    n_sents = sum(len(d) for d in docs)
    n_tokens = sum(len(s) for d in docs for s in d)
    print(f"Sentences: {n_sents}")
    print(f"Tokens: {n_tokens}")

    train, dev, test = split_documents(docs)
    print(f"Train sentences: {len(train)}  Dev: {len(dev)}  Test: {len(test)}")
    print("\nSample sentence:")
    print(train[0])
