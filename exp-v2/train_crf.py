"""
train_crf.py
Trains a CRF (Conditional Random Field) baseline NER model on the legal
BIO-tagged dataset, and evaluates it with entity-level precision/recall/F1
(via seqeval) -- not token accuracy, which is misleadingly high given the
'O' tag dominates the label distribution.

Splits at the DOCUMENT level (not sentence/row level) to avoid leakage.

Run: python3 train_crf.py
"""
import random
import joblib
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score

from data_prep import load_sentences


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def word_features(sent, i):
    word = sent[i][0]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.has_digit()": any(c.isdigit() for c in word),
        "word.has_hyphen()": "-" in word,
        "word.has_slash()": "/" in word,
        "word.len": len(word),
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True

    if i > 1:
        features["-2:word.lower()"] = sent[i - 2][0].lower()

    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True

    if i < len(sent) - 2:
        features["+2:word.lower()"] = sent[i + 2][0].lower()

    return features


def sent_to_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent):
    return [tag for _, tag in sent]


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def main():
    docs = load_sentences("data.csv")  # list of documents; each doc = list of sentences
    random.Random(42).shuffle(docs)

    n = len(docs)
    n_train = int(n * 0.7)
    n_dev = int(n * 0.15)
    train_docs = docs[:n_train]
    dev_docs = docs[n_train:n_train + n_dev]
    test_docs = docs[n_train + n_dev:]

    def docs_to_sents(doc_list):
        sents = []
        for doc in doc_list:
            sents.extend(doc)
        return sents

    train_sents = docs_to_sents(train_docs)
    dev_sents = docs_to_sents(dev_docs)
    test_sents = docs_to_sents(test_docs)

    print(f"Train documents: {len(train_docs)} ({len(train_sents)} sequences)")
    print(f"Dev documents:   {len(dev_docs)} ({len(dev_sents)} sequences)")
    print(f"Test documents:  {len(test_docs)} ({len(test_sents)} sequences)")

    X_train = [sent_to_features(s) for s in train_sents]
    y_train = [sent_to_labels(s) for s in train_sents]
    X_dev = [sent_to_features(s) for s in dev_sents]
    y_dev = [sent_to_labels(s) for s in dev_sents]
    X_test = [sent_to_features(s) for s in test_sents]
    y_test = [sent_to_labels(s) for s in test_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    print("\nTraining CRF...")
    crf.fit(X_train, y_train)

    print("\n=== Dev set results (entity-level, seqeval) ===")
    y_dev_pred = crf.predict(X_dev)
    print(classification_report(y_dev, y_dev_pred))
    print("Dev micro F1:", f1_score(y_dev, y_dev_pred))

    print("\n=== Test set results (entity-level, seqeval) ===")
    y_test_pred = crf.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("Test micro F1:", f1_score(y_test, y_test_pred))

    joblib.dump(crf, "crf_model.joblib")
    print("\nModel saved to crf_model.joblib")


if __name__ == "__main__":
    main()
