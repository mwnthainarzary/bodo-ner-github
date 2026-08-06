"""
evaluation_utils.py

Shared evaluation utilities producing paper-ready artifacts from a set of
true/predicted BIO tag sequences:

  1. classification_report.csv   -- per-entity precision/recall/F1/support
  2. classification_report.tex   -- the same, as a LaTeX table
  3. confusion_matrix.png        -- entity-TYPE-level confusion matrix
                                     (B-/I- prefixes stripped; 'O' included)
  4. f1_barchart.png             -- per-entity F1 bar chart, sorted

Works with any model's output as long as you can produce
y_true / y_pred as: List[List[str]]  (one list of BIO tags per sequence).

Usage:
    from evaluation_utils import full_evaluation_report
    full_evaluation_report(y_true, y_pred, out_dir="eval_crf", title="CRF")
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_report
from seqeval.scheme import IOB2


def _strip_prefix(tag):
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1] if "-" in tag else tag


def save_classification_report(y_true, y_pred, out_dir):
    """Entity-level (seqeval) precision/recall/F1/support -> CSV + LaTeX."""
    report_dict = seqeval_report(y_true, y_pred, output_dict=True, mode=None)

    rows = []
    for label, scores in report_dict.items():
        if label in ("micro avg", "macro avg", "weighted avg"):
            continue
        rows.append({
            "entity": label,
            "precision": round(scores["precision"], 3),
            "recall": round(scores["recall"], 3),
            "f1-score": round(scores["f1-score"], 3),
            "support": int(scores["support"]),
        })
    rows.sort(key=lambda r: -r["support"])

    for avg_name in ("micro avg", "macro avg", "weighted avg"):
        if avg_name in report_dict:
            scores = report_dict[avg_name]
            rows.append({
                "entity": avg_name,
                "precision": round(scores["precision"], 3),
                "recall": round(scores["recall"], 3),
                "f1-score": round(scores["f1-score"], 3),
                "support": int(scores["support"]),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "classification_report.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(out_dir, "classification_report.tex")
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, caption="Entity-level precision/recall/F1",
                             label="tab:ner_results", escape=True, float_format="%.3f"))

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    return df


def save_confusion_matrix(y_true, y_pred, out_dir, title="Confusion Matrix"):
    """
    Entity-TYPE-level confusion matrix (B-/I- prefixes stripped, so e.g.
    B-PERSON and I-PERSON both count as PERSON). This is the standard way
    to visualize *which* entity types get confused for which -- a token-level
    confusion matrix that keeps B-/I- separate is much less readable and
    rarely used in papers.
    """
    flat_true = [_strip_prefix(t) for seq in y_true for t in seq]
    flat_pred = [_strip_prefix(t) for seq in y_pred for t in seq]

    labels = sorted(set(flat_true) | set(flat_pred))
    cm = confusion_matrix(flat_true, flat_pred, labels=labels)

    # normalize by row (true label) so the matrix shows recall-style rates
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Row-normalized rate")

    # annotate cells with raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="black" if cm_norm[i, j] < 0.5 else "white")

    fig.tight_layout()
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {png_path}")

    # also save the raw counts as CSV for the appendix / supplementary material
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv = os.path.join(out_dir, "confusion_matrix_counts.csv")
    cm_df.to_csv(cm_csv)
    print(f"Saved: {cm_csv}")


def save_f1_barchart(df_report, out_dir, title="Per-entity F1"):
    """Bar chart of per-entity F1, sorted descending, excluding avg rows."""
    plot_df = df_report[~df_report["entity"].isin(
        ["micro avg", "macro avg", "weighted avg"]
    )].sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["entity"], plot_df["f1-score"], color="#4C72B0")
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    for i, (f1, sup) in enumerate(zip(plot_df["f1-score"], plot_df["support"])):
        ax.text(f1 + 0.01, i, f"n={sup}", va="center", fontsize=8)
    fig.tight_layout()

    png_path = os.path.join(out_dir, "f1_barchart.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {png_path}")


def full_evaluation_report(y_true, y_pred, out_dir="eval_output", title="Model"):
    """Runs all of the above and writes every artifact into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    df_report = save_classification_report(y_true, y_pred, out_dir)
    save_confusion_matrix(y_true, y_pred, out_dir, title=f"{title} - Confusion Matrix")
    save_f1_barchart(df_report, out_dir, title=f"{title} - Per-entity F1")
    print(f"\nAll evaluation artifacts written to: {out_dir}/")
    return df_report


def compare_models(reports: dict, out_dir="eval_output"):
    """
    reports: {"CRF": df_report_crf, "InLegalBERT": df_report_transformer, ...}
    Produces a combined comparison table + grouped bar chart of micro-F1 and
    macro-F1 across models -- the table/plot you'd put in a paper to show
    the improvement from one method to another.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for model_name, df in reports.items():
        micro = df[df["entity"] == "micro avg"].iloc[0]
        macro = df[df["entity"] == "macro avg"].iloc[0]
        rows.append({
            "model": model_name,
            "micro_precision": micro["precision"], "micro_recall": micro["recall"], "micro_f1": micro["f1-score"],
            "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1-score"],
        })
    comp_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "model_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    tex_path = os.path.join(out_dir, "model_comparison.tex")
    with open(tex_path, "w") as f:
        f.write(comp_df.to_latex(index=False, caption="Model comparison (micro/macro F1)",
                                  label="tab:model_comparison", float_format="%.3f"))
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width / 2, comp_df["micro_f1"], width, label="Micro F1")
    ax.bar(x + width / 2, comp_df["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    png_path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {png_path}")
    return comp_df