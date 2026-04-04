# Threat Modeling Classification

A machine learning pipeline that classifies network traffic as benign or attack types using the **CIC-IDS-2017** dataset, then enriches each prediction with structured threat modeling attributes powered by a **local LLM (Ollama / llama3.2)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How It Works — End to End](#2-how-it-works--end-to-end)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
   - [main.py](#51-mainpy--pipeline-orchestrator)
   - [load_csvs.py](#52-load_csvspy--data-loading)
   - [preprocess.py](#53-preprocesspy--data-preprocessing)
   - [train_models.py](#54-train_modelspy--model-training)
   - [evaluate.py](#55-evaluatepy--model-evaluation)
   - [threat_modeling.py](#56-threat_modelingpy--llm-threat-enrichment)
6. [LLM Integration (Ollama)](#6-llm-integration-ollama)
7. [Output Files](#7-output-files)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Project](#9-running-the-project)
10. [Configuration & Hyperparameters](#10-configuration--hyperparameters)
11. [Attack Classes in CIC-IDS-2017](#11-attack-classes-in-cic-ids-2017)

---

## 1. Project Overview

This project answers the question:

> *Given raw network traffic features, can a machine learning model correctly identify the type of attack — and then automatically describe its threat vector, impact, controls, actor, and risk level?*

It does this in two stages:

**Stage 1 — Classification:**
Two gradient boosting models (LightGBM and XGBoost) are trained to classify each network flow as one of 15 traffic classes (BENIGN + 14 attack types). Their performance is compared using precision, recall, F1 score, and confusion matrices.

**Stage 2 — Threat Modeling:**
The XGBoost predictions are passed to a local LLM (llama3.2 via Ollama) which enriches each predicted attack class with five structured threat attributes:
- Threat Vector
- Threat Impact
- Suggested Control
- Threat Actor
- Risk Ranking

The final output is a colour-coded summary table saved as a PNG — ready for use in security reports.

---

## 2. How It Works — End to End

```
Input CSVs (8 daily traffic files)
         │
         ▼
[STEP 1] Load & Merge all CSVs into one DataFrame
         │
         ▼
[STEP 2] Preprocess
         • Strip column whitespace
         • Replace Infinity / NaN with 0
         • Encode attack labels as integers (LabelEncoder)
         • Stratified 80/20 train/test split
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
[STEP 3] Train LightGBM              [STEP 4] Train XGBoost
         • 300 trees, lr=0.05                  • 300 trees, lr=0.05
         • num_leaves=63                        • max_depth=6
         │                                      │
         ▼                                      ▼
      Evaluate                              Evaluate
      • Accuracy, Precision,               • Accuracy, Precision,
        Recall, F1 (macro +                  Recall, F1 (macro +
        weighted)                            weighted)
      • Confusion matrix PNG               • Confusion matrix PNG
      • metrics_LightGBM.txt               • metrics_XGBoost.txt
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
[STEP 5] Threat Modeling (XGBoost predictions used)
         • Known labels  ──► hardcoded threat map (instant)
         • Unknown labels ──► Ollama llama3.2 (LLM, once per unique label, cached)
         • Produces: Threat Vector, Impact, Control, Actor, Risk
         │
         ▼
      threat_modeling_report.csv   (per-row, all test samples)
      threat_modeling_table.png    (aggregated, colour-coded summary)
         │
         ▼
[STEP 6] Model Comparison Summary
         • Winner selected by Weighted F1
         • comparison_summary.csv / .txt
```

---

## 3. Project Structure

```
Threat Modeling Classification/
│
├── Input_Folder/
│   └── CIC/                         ← 8 CIC-IDS-2017 CSV files go here
│       ├── Monday-WorkingHours.pcap_ISCX.csv
│       ├── Tuesday-WorkingHours.pcap_ISCX.csv
│       ├── Wednesday-workingHours.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│       └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│
├── Output_Folder/
│   └── <run_name>/                  ← Created automatically per run
│       ├── merged_dataset.csv
│       ├── run_info.txt
│       ├── metrics_LightGBM.txt
│       ├── metrics_XGBoost.txt
│       ├── confusion_matrix_LightGBM.png
│       ├── confusion_matrix_XGBoost.png
│       ├── predictions_test.csv
│       ├── threat_modeling_report.csv
│       ├── threat_modeling_table.png
│       ├── comparison_summary.csv
│       └── comparison_summary.txt
│
├── src/
│   ├── main.py                      ← Entry point / pipeline orchestrator
│   ├── load_csvs.py                 ← Load and merge CSV files
│   ├── preprocess.py                ← Clean, encode, split data
│   ├── train_models.py              ← Train LightGBM and XGBoost
│   ├── evaluate.py                  ← Compute metrics and plot confusion matrix
│   └── threat_modeling.py           ← LLM-based threat attribute enrichment
│
├── requirements.txt
└── Readme.md
```

---

## 4. Dataset

**CIC-IDS-2017** (Canadian Institute for Cybersecurity Intrusion Detection System 2017)

- 8 CSV files representing Monday–Friday network traffic captures
- ~2.8 million network flow records
- 78 features per flow (packet lengths, flow durations, flag counts, etc.)
- 1 label column: `Label`

### Attack Classes

| Label | Description |
|-------|-------------|
| BENIGN | Normal traffic |
| DDoS | Distributed Denial of Service |
| PortScan | Network reconnaissance |
| DoS Hulk | HTTP flood DoS |
| DoS GoldenEye | HTTP DoS variant |
| DoS slowloris | Slow-rate DoS |
| DoS Slowhttptest | Slow HTTP DoS |
| FTP-Patator | FTP brute force |
| SSH-Patator | SSH brute force |
| Web Attack – Brute Force | Web login brute force |
| Web Attack – XSS | Cross-site scripting |
| Web Attack – Sql Injection | SQL injection |
| Bot | Botnet traffic |
| Infiltration | Advanced persistent threat |
| Heartbleed | OpenSSL Heartbleed exploit |

---

## 5. Module-by-Module Breakdown

### 5.1 `main.py` — Pipeline Orchestrator

**What it does:**
Acts as the single entry point that ties all modules together in sequence. It prompts the user for inputs, coordinates all steps, and writes the final summary.

**Key responsibilities:**
- Accepts user input for dataset folder name and output run name
- Runs a pre-flight check to verify Ollama is accessible before starting training
- Calls each step in order: load → preprocess → train → evaluate → threat model → summarise
- Uses `time.perf_counter()` to measure training and inference time for each model
- Selects the winning model by **Weighted F1 score** (chosen over accuracy because the dataset is heavily imbalanced — BENIGN traffic vastly outnumbers attack traffic)

**Path resolution:**
```python
BASE_INPUT  = Path(__file__).parent.parent / "Input_Folder"
BASE_OUTPUT = Path(__file__).parent.parent / "Output_Folder"
```
Paths are resolved relative to the script file itself, so the pipeline works regardless of which directory you run it from.

---

### 5.2 `load_csvs.py` — Data Loading

**What it does:**
Scans the given input directory for all `.csv` files, loads them individually, and concatenates them into a single DataFrame.

**Key detail — column normalisation:**
```python
df.columns = [c.strip() for c in df.columns]
```
CIC-IDS-2017 CSVs have leading/trailing spaces in their column headers (e.g., `" Flow Duration"`). This strip removes them so column lookups work correctly downstream.

**Why merge all files?**
Each day's CSV contains different attack types. Monday has only BENIGN traffic; Friday afternoon has DDoS and PortScan. Merging all files ensures the model sees every attack class during training.

---

### 5.3 `preprocess.py` — Data Preprocessing

**What it does:**
Cleans the merged DataFrame and prepares it for model training.

**Step-by-step:**

1. **Strip column whitespace** (second pass safety)

2. **Replace bad values:**
   ```python
   df.replace(["Infinity", "inf", "INF", "NaN", "nan"], np.nan, inplace=True)
   df.replace([np.inf, -np.inf], np.nan, inplace=True)
   ```
   CIC-IDS-2017 contains literal string `"Infinity"` and `numpy.inf` values in features like `Flow Bytes/s`. These are converted to `NaN` first.

3. **Drop rows with missing labels** — rows with no `Label` value are removed.

4. **Separate features and target:**
   - `X` = all columns except `Label`
   - `y_raw` = the `Label` column as strings

5. **Convert all feature columns to numeric:**
   ```python
   X[col] = pd.to_numeric(X[col], errors="coerce")
   ```
   Any column that cannot be converted becomes `NaN`, then filled with `0`.

6. **Label encoding:**
   ```python
   le = LabelEncoder()
   y = le.fit_transform(y_raw)
   ```
   Converts string labels like `"DDoS"` to integers (e.g., `3`). The `LabelEncoder` object is returned so predictions can be decoded back to strings later.

7. **Stratified train/test split (80/20):**
   ```python
   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   ```
   `stratify=y` ensures each attack class is proportionally represented in both train and test sets — critical given the heavy class imbalance in this dataset.

**Returns:** `X_train, X_test, y_train, y_test, le, feature_cols`

---

### 5.4 `train_models.py` — Model Training

**What it does:**
Trains two gradient boosting classifiers on the training data.

#### LightGBM

```python
LGBMClassifier(
    boosting_type="gbdt",   # Gradient Boosted Decision Trees
    n_estimators=300,       # 300 trees
    learning_rate=0.05,     # Small step size to prevent overfitting
    num_leaves=63,          # Max leaves per tree (2^6 - 1, balanced depth)
    random_state=42
)
```

LightGBM uses a **leaf-wise** tree growth strategy, making it faster and often more accurate than traditional level-wise approaches on tabular data.

#### XGBoost

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,            # Max tree depth
    subsample=0.9,          # 90% of rows sampled per tree (reduces overfitting)
    colsample_bytree=0.9,   # 90% of features sampled per tree
    objective="multi:softprob",   # Multiclass probability output
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42
)
```

XGBoost uses a **level-wise** (depth-first) strategy. The `subsample` and `colsample_bytree` parameters introduce randomness to reduce overfitting.

**Why both models?**
Both are industry-standard for tabular classification. Running both allows direct comparison on the same data under identical conditions, with the winner selected automatically by Weighted F1.

---

### 5.5 `evaluate.py` — Model Evaluation

**What it does:**
Runs inference on the test set, computes multiple performance metrics, saves a classification report, and generates a confusion matrix PNG.

#### Metrics computed

| Metric | Average type | Why it's included |
|--------|-------------|-------------------|
| Accuracy | — | Overall correctness |
| Precision | Macro | Equal weight per class |
| Recall | Macro | Equal weight per class |
| F1 | Macro | Equal weight per class |
| Precision | Weighted | Accounts for class frequency |
| Recall | Weighted | Accounts for class frequency |
| F1 | **Weighted** | **Primary winner metric** — best for imbalanced data |

**Macro vs Weighted:**
- *Macro* treats every class equally regardless of how many samples it has. A class with 2 samples counts as much as BENIGN with 454,000.
- *Weighted* weights each class by its sample count. More representative of real-world performance on this heavily imbalanced dataset.

#### Confusion Matrix

The confusion matrix is rendered as a publication-quality PNG:
- Each cell shows the raw **count** (e.g., `45,123`) with comma formatting
- Cell text is white on dark cells, black on light cells (threshold at `max/2`)
- Figure size scales dynamically with number of classes
- Saved at 300 DPI

#### Inference timing

```python
t0 = time.perf_counter()
y_pred = model.predict(X_test)
pred_time_sec = time.perf_counter() - t0
```

Prediction time is measured separately from training time and included in the comparison summary — relevant for real-time IDS deployment scenarios.

**Returns:** `y_pred` (integer array), `summary` (dict of all metrics)

---

### 5.6 `threat_modeling.py` — LLM Threat Enrichment

**What it does:**
Takes the array of XGBoost-predicted attack labels and enriches each with five threat modeling attributes, producing a structured threat report.

This module has three layers of resolution for each label:

```
Label received
      │
      ├─ Is it in _KNOWN_THREAT_MAP?  ──► Return instantly (no LLM call)
      │
      ├─ Is it in _LLM_CACHE?         ──► Return cached result (no repeat API call)
      │
      └─ Unknown label                ──► Call Ollama (llama3.2), cache result
```

#### Layer 1: Known Threat Map

Five well-known labels are hardcoded with expert-defined attributes:

| Label | Vector | Impact | Control | Actor | Risk |
|-------|--------|--------|---------|-------|------|
| BENIGN | None | None | None | None | Low |
| DDoS | Network Flooding | Service Disruption | Rate Limiting / Firewall | External Botnet | Critical |
| PortScan | Reconnaissance | Information Disclosure | IDS / Network Monitoring | External Attacker | Medium |
| Infiltration | Unauthorized Access | System Compromise | Access Control / Patch Mgmt | APT | High |
| Web Attack | Application Exploit | Data Breach | Web Application Firewall | Skilled Attacker | High |

This handles the most common labels instantly with zero LLM cost.

#### Layer 2: LLM Cache

```python
_LLM_CACHE: dict[str, dict] = {}
```

Once a label is resolved by the LLM, the result is stored in this in-memory dictionary. Since the same attack class appears thousands of times in the test set (e.g., `DoS Hulk` appears 46,283 times), the LLM is called only **once per unique label** — not once per row. This makes the LLM integration efficient regardless of dataset size.

#### Layer 3: Ollama LLM Call

For labels not in the known map (e.g., `DoS Hulk`, `FTP-Patator`, `Heartbleed`), the model is queried with a structured prompt:

```
You are a cybersecurity expert. Given the network attack class name below,
return ONLY a JSON object with exactly these five keys:
  "Vector"  — the attack vector
  "Impact"  — the primary impact
  "Control" — the recommended security control
  "Actor"   — the likely threat actor
  "Risk"    — one of: Critical, High, Medium, Low

Attack class: DoS Hulk

Respond with ONLY the raw JSON object, no explanation, no markdown fences.
```

The response is parsed with a regex that extracts the first `{...}` block, tolerating any markdown fences the model may add. The `Risk` value is validated against known levels and defaults to `"Medium"` if unrecognised.

**Temperature is set to 0** — deterministic responses, no randomness.

#### Pre-flight Check

```python
check_ollama("llama3.2")
```

Called at the very start of `main()` before any training begins. Sends a minimal ping request to verify Ollama is running and the model is loaded. If it fails, the pipeline raises a clear `RuntimeError` immediately rather than silently producing Unknown values hours later after training completes.

#### Threat Table Visualisation

`save_threat_table()` aggregates the per-row report into one row per unique attack class + attributes, adds a `Count` column, and renders it as a styled matplotlib table PNG:

- **Dark blue header** row with white bold text
- **Row colours by Risk level:**
  - Critical → red `#ff4c4c`
  - High → orange `#ff9900`
  - Medium → yellow `#ffe066`
  - Low → green `#c6efce`
  - None → light grey `#f2f2f2`
- Rows sorted by Count descending
- Figure size scales dynamically with number of rows
- Saved at 300 DPI

---

## 6. LLM Integration (Ollama)

### Why Ollama?

Ollama runs LLMs locally on your machine with no internet connection, no API key, and no data leaving your system. This is important in security contexts where network traffic data may be sensitive.

### How the integration works

1. The `ollama` Python package communicates with the Ollama server running on `localhost:11434`
2. `ollama.chat()` sends a message to the model and receives a response dict
3. The response text is parsed for a JSON object
4. Results are cached per unique label to avoid redundant calls

### Model used: llama3.2

`llama3.2` is a small (~2GB) but capable instruction-following model. It reliably follows the structured JSON prompt format when `temperature=0`.

### Caching efficiency

With 15 unique attack classes in CIC-IDS-2017:
- 5 are resolved from the hardcoded map (0 LLM calls)
- 10 are resolved via LLM (exactly 10 calls total, regardless of dataset size)
- The test set has ~560,000 rows — without caching this would be 560,000 LLM calls

---

## 7. Output Files

All files are written to `Output_Folder/<run_name>/`:

| File | Description |
|------|-------------|
| `merged_dataset.csv` | All 8 input CSVs merged into one file |
| `run_info.txt` | Dataset name, row count, feature count, class list |
| `metrics_LightGBM.txt` | Full sklearn classification report for LightGBM |
| `metrics_XGBoost.txt` | Full sklearn classification report for XGBoost |
| `confusion_matrix_LightGBM.png` | Colour-coded confusion matrix with raw counts |
| `confusion_matrix_XGBoost.png` | Colour-coded confusion matrix with raw counts |
| `predictions_test.csv` | True label vs LightGBM prediction vs XGBoost prediction per test row |
| `threat_modeling_report.csv` | Full per-row report including all 5 threat attributes |
| `threat_modeling_table.png` | Aggregated colour-coded threat summary table |
| `comparison_summary.csv` | Side-by-side metric table for both models |
| `comparison_summary.txt` | Human-readable summary with winner announcement |

---

## 8. Setup & Installation

### Prerequisites

- Python 3.10+ (Anaconda recommended)
- [Ollama](https://ollama.com) installed and running

### Install Ollama and pull the model

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull the llama3.2 model (~2GB)
ollama pull llama3.2
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
numpy
scikit-learn
lightgbm
xgboost
matplotlib
ollama
```

### Dataset

Download the **CIC-IDS-2017** dataset from the Canadian Institute for Cybersecurity and place the 8 CSV files into `Input_Folder/CIC/`.

---

## 9. Running the Project

```bash
cd "/path/to/Threat Modeling Classification/src"
python main.py
```

You will be prompted for:

```
Enter dataset folder name (inside Input_Folder/): CIC
Enter output run folder name: My_Run_01
```

The pipeline will then:

1. Verify Ollama is running
2. Load and merge the 8 CSV files
3. Preprocess the data
4. Train LightGBM (approx. 2–5 minutes)
5. Train XGBoost (approx. 3–7 minutes)
6. Evaluate both models and save confusion matrices
7. Query the LLM for each unique unknown attack class (10 calls, ~10 seconds)
8. Save all output files to `Output_Folder/My_Run_01/`

---

## 10. Configuration & Hyperparameters

### Model hyperparameters

| Parameter | LightGBM | XGBoost | Effect |
|-----------|----------|---------|--------|
| `n_estimators` | 300 | 300 | Number of trees. More = better fit, slower training |
| `learning_rate` | 0.05 | 0.05 | Step size. Smaller = more trees needed but more stable |
| `num_leaves` | 63 | — | Max leaves per tree. 63 = deep, expressive trees |
| `max_depth` | — | 6 | Max tree depth. Limits model complexity |
| `subsample` | — | 0.9 | Row sampling per tree. Reduces overfitting |
| `colsample_bytree` | — | 0.9 | Feature sampling per tree. Reduces overfitting |
| `random_state` | 42 | 42 | Fixed seed for full reproducibility |

### Train/test split

- 80% training / 20% testing
- Stratified by class label
- `random_state=42` — results are fully reproducible across runs

### LLM settings

- Model: `llama3.2`
- Temperature: `0` (deterministic)
- Caching: in-memory, per unique label, per pipeline run

---

## 11. Attack Classes in CIC-IDS-2017

The dataset contains 15 traffic classes across 5 attack categories:

**DoS / DDoS (5 types)**
- `DDoS` — High-volume botnet-driven flooding
- `DoS Hulk` — HTTP GET flood targeting web servers
- `DoS GoldenEye` — HTTP DoS targeting keep-alive connections
- `DoS slowloris` — Slow-rate HTTP header attack
- `DoS Slowhttptest` — Slow HTTP body attack

**Brute Force (2 types)**
- `FTP-Patator` — Automated FTP credential guessing
- `SSH-Patator` — Automated SSH credential guessing

**Web Attacks (3 types)**
- `Web Attack – Brute Force` — Web application login brute force
- `Web Attack – XSS` — Cross-site scripting injection
- `Web Attack – Sql Injection` — SQL injection via web forms

**Infiltration / APT (1 type)**
- `Infiltration` — Multi-stage attack simulating an APT

**Other Threats (2 types)**
- `Bot` — Botnet command-and-control traffic
- `Heartbleed` — OpenSSL memory leak exploit (CVE-2014-0160)

**Benign (1 type)**
- `BENIGN` — Normal background traffic (~80% of dataset)
