# src/main.py
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from load_csvs import load_all_csvs
from preprocess import preprocess
from train_models import train_lightgbm, train_xgboost
from evaluate import evaluate_model
from threat_modeling import map_threat_attributes, save_threat_table


# Resolve paths relative to project root (one level above src/)
BASE_INPUT = Path(__file__).parent.parent / "Input_Folder"
BASE_OUTPUT = Path(__file__).parent.parent / "Output_Folder"


def main():
    print("\n=== Threat Modeling Classification (LightGBM vs XGBoost) ===\n")

    dataset_folder = input("Enter dataset folder name (inside Input_Folder/): ").strip()
    run_name = input("Enter output run folder name: ").strip()
    llm_model = input("Enter Ollama model name for threat lookup [default: llama3.2]: ").strip() or "llama3.2"

    dataset_dir = BASE_INPUT / dataset_folder
    run_dir = BASE_OUTPUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # STEP 1: Load CSV dataset
    # -------------------------
    print("[STEP 1] Loading CSV files...")
    df = load_all_csvs(dataset_dir)

    merged_path = run_dir / "merged_dataset.csv"
    df.to_csv(merged_path, index=False)
    print(f"[INFO] Saved merged dataset: {merged_path}")

    # -------------------------
    # STEP 2: Preprocess
    # -------------------------
    print("[STEP 2] Preprocessing...")
    X_train, X_test, y_train, y_test, le, feature_cols = preprocess(df, target_col="Label")

    label_names = list(le.classes_)
    num_classes = len(label_names)

    (run_dir / "run_info.txt").write_text(
        f"Dataset folder: {dataset_folder}\n"
        f"Run name: {run_name}\n"
        f"Rows: {len(df)}\n"
        f"Features used (numeric): {len(feature_cols)}\n"
        f"Classes: {num_classes}\n"
        f"Class labels: {label_names}\n"
    )

    # -------------------------
    # STEP 3: Train LightGBM + Evaluate
    # -------------------------
    print("[STEP 3] Training LightGBM...")
    t0 = time.perf_counter()
    lgbm = train_lightgbm(X_train, y_train)
    lgbm_train_time = time.perf_counter() - t0

    lgbm_pred, lgbm_summary = evaluate_model(
        model=lgbm,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        out_dir=run_dir,
        model_name="LightGBM",
        train_time_sec=lgbm_train_time,
    )

    # -------------------------
    # STEP 4: Train XGBoost + Evaluate
    # -------------------------
    print("[STEP 4] Training XGBoost...")
    t0 = time.perf_counter()
    xgb = train_xgboost(X_train, y_train, num_classes=num_classes)
    xgb_train_time = time.perf_counter() - t0

    xgb_pred, xgb_summary = evaluate_model(
        model=xgb,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        out_dir=run_dir,
        model_name="XGBoost",
        train_time_sec=xgb_train_time,
    )

    # -------------------------
    # STEP 5: Save predictions + Threat Modeling Report
    # -------------------------
    true_labels = le.inverse_transform(y_test)
    pred_lgbm_labels = le.inverse_transform(lgbm_pred.astype(int))
    pred_xgb_labels = le.inverse_transform(xgb_pred.astype(int))

    # Threat attribute mapping based on XGBoost prediction (primary model)
    threat_attrs_df = map_threat_attributes(pred_xgb_labels, model=llm_model)

    final_report = pd.DataFrame({
        "True_Label": true_labels,
        "Pred_LightGBM": pred_lgbm_labels,
        "Pred_XGBoost": pred_xgb_labels,
    })
    final_report = pd.concat([final_report, threat_attrs_df], axis=1)

    threat_report_path = run_dir / "threat_modeling_report.csv"
    final_report.to_csv(threat_report_path, index=False)
    print(f"[INFO] Threat modeling report saved: {threat_report_path}")

    save_threat_table(final_report, run_dir)

    # Also save raw predictions file
    pred_path = run_dir / "predictions_test.csv"
    final_report[["True_Label", "Pred_LightGBM", "Pred_XGBoost"]].to_csv(pred_path, index=False)

    # -------------------------
    # STEP 6: Save Comparison Summary
    # -------------------------
    comparison_df = pd.DataFrame([lgbm_summary, xgb_summary])
    comp_csv = run_dir / "comparison_summary.csv"
    comparison_df.to_csv(comp_csv, index=False)

    # Decide winner using weighted F1 (best for imbalanced datasets)
    primary_metric = "F1_Weighted"
    best_row = comparison_df.sort_values(by=primary_metric, ascending=False).iloc[0]

    summary_text = "\n".join([
        "=== Model Comparison Summary ===",
        f"Primary metric: {primary_metric}",
        f"Winner: {best_row['Model']}  ({primary_metric} = {best_row[primary_metric]})",
        "",
        "Saved files:",
        " - merged_dataset.csv",
        " - run_info.txt",
        " - metrics_LightGBM.txt",
        " - metrics_XGBoost.txt",
        " - confusion_matrix_LightGBM.png",
        " - confusion_matrix_XGBoost.png",
        " - predictions_test.csv",
        " - threat_modeling_report.csv",
        " - threat_modeling_table.png",
        " - comparison_summary.csv",
        " - comparison_summary.txt",
    ])

    (run_dir / "comparison_summary.txt").write_text(summary_text)

    print("\n✅ DONE — Outputs saved in:")
    print(run_dir.resolve())
    print(f"\nWinner by {primary_metric}: {best_row['Model']}")


if __name__ == "__main__":
    main()
