from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Known threat map (instant lookup, no LLM call needed)
# ---------------------------------------------------------------------------
_KNOWN_THREAT_MAP: dict[str, dict] = {
    "BENIGN": {
        "Vector": "None",
        "Impact": "None",
        "Control": "None",
        "Actor": "None",
        "Risk": "Low",
    },
    "DDoS": {
        "Vector": "Network Flooding",
        "Impact": "Service Disruption",
        "Control": "Rate Limiting / Firewall",
        "Actor": "External Botnet",
        "Risk": "Critical",
    },
    "PortScan": {
        "Vector": "Reconnaissance",
        "Impact": "Information Disclosure",
        "Control": "IDS / Network Monitoring",
        "Actor": "External Attacker",
        "Risk": "Medium",
    },
    "Infiltration": {
        "Vector": "Unauthorized Access",
        "Impact": "System Compromise",
        "Control": "Access Control / Patch Management",
        "Actor": "Advanced Persistent Threat",
        "Risk": "High",
    },
    "Web Attack": {
        "Vector": "Application Exploit",
        "Impact": "Data Breach",
        "Control": "Web Application Firewall",
        "Actor": "Skilled Attacker",
        "Risk": "High",
    },
}

# Risk level → row background colour (used by save_threat_table)
_RISK_COLORS = {
    "Critical": "#ff4c4c",
    "High":     "#ff9900",
    "Medium":   "#ffe066",
    "Low":      "#c6efce",
    "None":     "#f2f2f2",
}

_FALLBACK_ENTRY = {
    "Vector": "Unknown",
    "Impact": "Unknown",
    "Control": "Manual Review",
    "Actor": "Unknown",
    "Risk": "Medium",
}

# ---------------------------------------------------------------------------
# LLM-based lookup via Claude API
# ---------------------------------------------------------------------------
_LLM_CACHE: dict[str, dict] = {}

_PROMPT_TEMPLATE = """\
You are a cybersecurity expert. Given the network attack class name below, \
return ONLY a JSON object with exactly these five keys:
  "Vector"  — the attack vector (e.g. "Network Flooding", "SQL Injection")
  "Impact"  — the primary impact (e.g. "Service Disruption", "Data Breach")
  "Control" — the recommended security control (e.g. "Rate Limiting / Firewall")
  "Actor"   — the likely threat actor (e.g. "External Botnet", "Insider Threat")
  "Risk"    — one of: Critical, High, Medium, Low

Attack class: {label}

Respond with ONLY the raw JSON object, no explanation, no markdown fences."""


def _call_claude(label: str) -> dict:
    """Call Claude API and parse JSON response."""
    try:
        import anthropic
    except ImportError:
        print("[WARN] anthropic package not installed. Run: pip install anthropic")
        return _FALLBACK_ENTRY.copy()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    prompt = _PROMPT_TEMPLATE.format(label=label)
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON even if the model wraps it in markdown fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        parsed = json.loads(json_match.group())

        entry = {
            "Vector":  str(parsed.get("Vector",  "Unknown")),
            "Impact":  str(parsed.get("Impact",  "Unknown")),
            "Control": str(parsed.get("Control", "Manual Review")),
            "Actor":   str(parsed.get("Actor",   "Unknown")),
            "Risk":    str(parsed.get("Risk",    "Medium")),
        }
        if entry["Risk"] not in _RISK_COLORS:
            entry["Risk"] = "Medium"
        return entry

    except Exception as exc:
        print(f"[WARN] Claude API lookup failed for '{label}': {exc}")
        return _FALLBACK_ENTRY.copy()


def _resolve_label(label: str) -> dict:
    """Return threat entry from known map, Claude API cache, or fresh API call."""
    label_clean = label.strip()

    if label_clean in _KNOWN_THREAT_MAP:
        return _KNOWN_THREAT_MAP[label_clean]

    if label_clean in _LLM_CACHE:
        return _LLM_CACHE[label_clean]

    print(f"[Claude] Querying threat attributes for: '{label_clean}'")
    entry = _call_claude(label_clean)
    _LLM_CACHE[label_clean] = entry
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_threat_attributes(label_series) -> pd.DataFrame:
    """
    Maps attack labels to structured threat modeling attributes.
    Known labels are resolved instantly; unknown labels are sent to the
    Claude API (called once per unique label, then cached).
    Requires ANTHROPIC_API_KEY environment variable.
    """
    vectors, impacts, controls, actors, risks = [], [], [], [], []

    for label in label_series:
        entry = _resolve_label(str(label))
        vectors.append(entry["Vector"])
        impacts.append(entry["Impact"])
        controls.append(entry["Control"])
        actors.append(entry["Actor"])
        risks.append(entry["Risk"])

    return pd.DataFrame({
        "Threat_Vector":     vectors,
        "Threat_Impact":     impacts,
        "Suggested_Control": controls,
        "Threat_Actor":      actors,
        "Risk_Ranking":      risks,
    })


def save_threat_table(final_report: pd.DataFrame, out_dir: Path) -> None:
    """
    Builds an aggregated threat summary table and saves it as a PNG,
    styled the same way as the confusion-matrix figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = (
        final_report
        .groupby(
            ["Pred_XGBoost", "Threat_Vector", "Threat_Impact",
             "Suggested_Control", "Threat_Actor", "Risk_Ranking"],
            sort=False,
        )
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    col_labels = [
        "Attack Class", "Count", "Threat Vector", "Threat Impact",
        "Suggested Control", "Threat Actor", "Risk Ranking",
    ]
    cell_data = agg[
        ["Pred_XGBoost", "Count", "Threat_Vector", "Threat_Impact",
         "Suggested_Control", "Threat_Actor", "Risk_Ranking"]
    ].values.tolist()

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig_w = min(26, max(16, 0.9 * n_cols * 2.5))
    fig_h = min(30, max(4, 0.45 * n_rows + 1.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Threat Modeling Summary Table", fontsize=16, pad=14)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))

    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#1f3864")
        cell.set_text_props(color="white", fontweight="bold")

    for row_idx, row in enumerate(cell_data, start=1):
        risk = str(row[-1]).strip()
        bg = _RISK_COLORS.get(risk, "#ffffff")
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")

    fig.tight_layout()
    out_path = out_dir / "threat_modeling_table.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Threat modeling table saved: {out_path}")
