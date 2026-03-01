from pathlib import Path
import pandas as pd


def load_all_csvs(dataset_dir: Path) -> pd.DataFrame:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    dfs = []
    for f in csv_files:
        print(f"[LOAD] {f.name}")
        df = pd.read_csv(f, low_memory=False)

        # ✅ normalize headers (removes leading/trailing spaces)
        df.columns = [c.strip() for c in df.columns]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
