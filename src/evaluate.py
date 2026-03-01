# src/evaluate.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    label_names,
    out_dir: Path,
    model_name: str,
    train_time_sec: float,
):
    """
    Evaluates model and saves:
      - metrics_<model>.txt
      - confusion_matrix_<model>.png (RAW COUNTS with readable numbers)

    Returns:
      y_pred, summary_dict
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Prediction timing
    # -------------------------
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time_sec = time.perf_counter() - t0

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_test, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    (out_dir / f"metrics_{model_name}.txt").write_text(report)

    # -------------------------
    # Confusion Matrix (RAW COUNTS)
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)
    n_classes = len(label_names)

    # Dynamic figure size
    fig_w = min(24, max(12, 0.9 * n_classes))
    fig_h = min(24, max(10, 0.8 * n_classes))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")


    ax.set_title(f"{model_name} Confusion Matrix (Counts)", fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Add readable numbers inside cells ----
    thresh = cm.max() / 2.0
    font_size = 8 if n_classes <= 12 else 7 if n_classes <= 18 else 6

    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]
            ax.text(
                j,
                i,
                f"{value:,}",   # comma formatting (e.g., 45,123)
                ha="center",
                va="center",
                fontsize=font_size,
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    out_path = out_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # -------------------------
    # Summary
    # -------------------------
    summary = {
        "Model": model_name,
        "Train_Time_s": round(float(train_time_sec), 4),
        "Predict_Time_s": round(float(pred_time_sec), 4),
        "Accuracy": round(float(acc), 6),
        "Precision_Macro": round(float(prec_macro), 6),
        "Recall_Macro": round(float(rec_macro), 6),
        "F1_Macro": round(float(f1_macro), 6),
        "Precision_Weighted": round(float(prec_weighted), 6),
        "Recall_Weighted": round(float(rec_weighted), 6),
        "F1_Weighted": round(float(f1_weighted), 6),
        "Num_Test_Samples": int(len(y_test)),
        "Num_Classes": int(len(label_names)),
    }

    return y_pred, summary
