"""
train_transformer.py

Fine-tunes a transformer (default: InLegalBERT, a BERT model pretrained on
Indian legal text -- a good match for this dataset) for token classification
on the legal NER dataset.

NOTE: this script needs internet access to the Hugging Face Hub to download
the pretrained model/tokenizer. Run it in Colab, or any machine with normal
internet access -- it will NOT run in a network-restricted sandbox.

Install once:
    pip install transformers datasets seqeval accelerate torch

Run:
    python3 train_transformer.py
"""
import random
import inspect
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

from data_prep import load_sentences

# ---------------------------------------------------------------------------
# Config -- swap MODEL_NAME to try a different base model, e.g.:
#   "bert-base-uncased"                (general English BERT, baseline)
#   "law-ai/InLegalBERT"                (Indian-legal-domain pretrained BERT)
#   "nlpaueb/legal-bert-base-uncased"   (general legal-domain BERT)
# ---------------------------------------------------------------------------
MODEL_NAME = "law-ai/InLegalBERT"
OUTPUT_DIR = "./transformer_ner_law-ai_InLegalBERT"
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 3e-5


def build_dataset():
    docs = load_sentences("data.csv")
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

    def to_hf_format(sents):
        return {
            "tokens": [[tok for tok, _ in s] for s in sents],
            "ner_tags": [[tag for _, tag in s] for s in sents],
        }

    train_data = to_hf_format(docs_to_sents(train_docs))
    dev_data = to_hf_format(docs_to_sents(dev_docs))
    test_data = to_hf_format(docs_to_sents(test_docs))

    # build label list from the full dataset so all splits share the same mapping
    all_tags = sorted({tag for doc in docs for sent in doc for _, tag in sent})
    label2id = {l: i for i, l in enumerate(all_tags)}
    id2label = {i: l for l, i in label2id.items()}

    def encode_labels(data):
        data["ner_tags"] = [[label2id[t] for t in tags] for tags in data["ner_tags"]]
        return data

    ds = DatasetDict({
        "train": Dataset.from_dict(encode_labels(train_data)),
        "validation": Dataset.from_dict(encode_labels(dev_data)),
        "test": Dataset.from_dict(encode_labels(test_data)),
    })
    return ds, label2id, id2label


def main():
    import transformers
    print(f"transformers version: {transformers.__version__}")

    ds, label2id, id2label = build_dataset()
    print(ds)
    print(f"Num labels: {len(label2id)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)  # only label the first subword
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    seqeval_metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p_] for (p_, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p_, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # `evaluation_strategy` was renamed to `eval_strategy` in newer
    # `transformers` versions -- detect which one this install expects.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **{strategy_kwarg: "epoch"},
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
    )

    # `tokenizer=` was renamed to `processing_class=` in newer `transformers`
    # versions -- detect which one this install's Trainer expects.
    trainer_params = inspect.signature(Trainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **{tokenizer_kwarg: tokenizer},
    )

    print("\nTraining...")
    trainer.train()

    print("\n=== Test set results ===")
    test_results = trainer.evaluate(tokenized_ds["test"])
    print(test_results)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()