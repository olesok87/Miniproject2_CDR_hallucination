# ----------------------------
# Vertical plotting from scored CSV with exclusion option
# ----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def short_label(name):
    m = re.search(r"design(\d+)_n(\d+)", name)
    if m:
        return f"d{m.group(1)}.n{m.group(2)}"
    m2 = re.search(r"d(?:esign)?[_-]?(\d+)[^A-Za-z0-9]*n[_-]?(\d+)", name)
    if m2:
        return f"d{m2.group(1)}.n{m2.group(2)}"
    return (name[:20] + "...") if len(name) > 23 else name

def plot_from_scored_csv_vertical(csv_path,
                                  out_png,
                                  figsize=(14,7),
                                  xlabel="Model",
                                  ylabel="Composite score (higher = better)",
                                  exclude_patterns=None):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(p)
    if not {"description","composite_score","color"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: description, composite_score, color")

    # apply exclusions (list of regex strings or substrings)
    if exclude_patterns:
        mask = pd.Series(False, index=df.index)
        for pat in exclude_patterns:
            mask = mask | df["description"].astype(str).str.contains(pat, regex=True)
        df = df[~mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left to plot after applying exclusions")

    df["short"] = df["description"].astype(str).map(short_label)
    plot_df = df[["short","composite_score","color"]].copy()
    plot_df = plot_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    labels = plot_df["short"].tolist()
    scores = plot_df["composite_score"].astype(float).to_numpy()
    colors = plot_df["color"].astype(str).tolist()
    colors = [c if c.startswith("#") else ("#" + c.lstrip()) for c in colors]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, scores, color=colors, edgecolor="#222222", linewidth=0.6)

    # labels and rotation
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)

    # annotate numeric score above bars
    y_off_unit = 0.01 * max(1.0, np.nanmax(np.abs(scores)))
    for rect, v in zip(bars, scores):
        if np.isfinite(v):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + y_off_unit,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#111111")

    # highlight top model label style
    if len(plot_df) > 0:
        ax.get_xticklabels()[0].set_fontweight("bold")
        try:
            ax.get_xticklabels()[0].set_color("#2ca02c")
        except Exception:
            pass
        winner_desc = plot_df.loc[0, "short"]
    else:
        winner_desc = None

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.title("Model ranking (from CSV)", fontsize=13)
    plt.tight_layout()
    outpath = Path(out_png)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved vertical plot to: {outpath} — winner: {winner_desc}")
    plt.show()

# ----------------------------
# Call the function with your paths and exclude pattern for the 6Y6C row
# ----------------------------
csv_path = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics_scored.csv"
out_png = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\composite_scores_from_csv.png"

# exclude any description containing '6Y6C' or the full WT pattern; adjust pattern if needed
plot_from_scored_csv_vertical(csv_path=csv_path, out_png=out_png, figsize=(7,6),
                              exclude_patterns=[r"6Y6C", r"relaxed_6Y6C"])