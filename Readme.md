# Threat Modeling Classification

A machine learning pipeline that classifies network traffic into attack categories using **LightGBM** and **XGBoost**, then enriches predictions with structured threat intelligence via a local **LLM (llama3.2 via Ollama)**.

---

## Requirements

- Python 3.9 or higher
- [Ollama](https://ollama.com) installed and running locally
- CIC-IDS dataset CSV files (CIC or CIC_2 format)

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "Threat Modeling Classification"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Set Up Ollama

Ollama runs the local LLM used for threat attribute enrichment.

**Step 1 — Download Ollama:**

Go to [https://ollama.com/download](https://ollama.com/download) and install for your OS.

**Step 2 — Start the Ollama server:**

```bash
ollama serve
```

Keep this running in a separate terminal.

**Step 3 — Pull the llama3.2 model:**

```bash
ollama pull llama3.2
```

This downloads the model (~2 GB). Only needed once.

**Step 4 — Verify it works:**

```bash
ollama run llama3.2 "ping"
```

You should get a response from the model.

---

## Troubleshooting

**`Ollama not reachable` error**
- Make sure `ollama serve` is running in a separate terminal
- Make sure you have pulled the model: `ollama pull llama3.2`

**`Target column 'Label' not found` error**
- Your CSV column may have a leading space. The pipeline strips spaces automatically, but verify your CSV has a column literally named `Label`.

**`No CSV files found` error**
- Check that your dataset folder name matches exactly what you typed (case-sensitive on macOS/Linux)

**Out of memory during training**
- Try a smaller dataset subset or reduce `n_estimators` in [train_models.py](src/train_models.py)
