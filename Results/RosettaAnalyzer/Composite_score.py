import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pathlib import Path

def short_label(name: str) -> str:
    """
    Robustly shorten model names to 'dX.nY' style when possible.
    Examples:
      'd1_n1_relaxed' -> 'd1.n1'
      'prefix_d0n2_suffix' -> 'd0.n2'
      'relaxed_WT' -> 'WT'
      'some_long_name' -> 'some.long.name' (fallback)
    """
    s = str(name)

    # 1) find d<number> and n<number> anywhere (allow separators _, -, ., nothing)
    m = re.search(r"(d\d+)", s, flags=re.IGNORECASE)
    n = re.search(r"(n\d+)", s, flags=re.IGNORECASE)

    if m and n:
        # return canonical lowercase 'dX.nY'
        dpart = m.group(1).lower()
        npart = n.group(1).lower()
        return f"{dpart}.{npart}"
    elif m:
        return m.group(1).lower()
    elif n:
        return n.group(1).lower()

    # 2) fallback: remove common suffixes and compress underscores to dots
    # remove trailing '_relaxed' or similar common suffixes
    s2 = re.sub(r"(_relaxed|\.relaxed|-relaxed)$", "", s, flags=re.IGNORECASE)
    s2 = re.sub(r"\.(pdb|pdbqt|ent)$", "", s2, flags=re.IGNORECASE)
    # if it contains 'WT' or 'wildtype', normalize to 'WT'
    if re.search(r"\bWT\b", s2, flags=re.IGNORECASE) or re.search(r"wild.?type", s2, flags=re.IGNORECASE):
        return "WT"

    # replace underscores and hyphens with dots, collapse repeated dots
    s2 = re.sub(r"[_\-]+", ".", s2)
    s2 = re.sub(r"\.+", ".", s2).strip(".")
    # if still long, return only the first two chunks separated by dots
    parts = s2.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2])
    return s2 or s

def plot_from_scored_csv_vertical(
    csv_path,
    out_png,
    figsize=(14, 7),
    xlabel="Model",
    ylabel="Composite score (higher = better)",
    exclude_patterns=None,
):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(p)
    if not {"description", "composite_score", "color"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: description, composite_score, color")

    # apply exclusions
    if exclude_patterns:
        mask = pd.Series(False, index=df.index)
        for pat in exclude_patterns:
            mask |= df["description"].astype(str).str.contains(pat, regex=True)
        df = df[~mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left to plot after exclusions")

    # create short labels using the robust short_label function
    df["short"] = df["description"].astype(str).map(short_label)

    # sort descending by composite_score
    plot_df = df[["short", "composite_score"]].copy()
    plot_df = plot_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    labels = plot_df["short"].tolist()
    scores = plot_df["composite_score"].astype(float).to_numpy()

    # build colormap red -> yellow -> green based on normalized scores
    cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["#d62728", "#ffcc00", "#2ca02c"])
    # normalize between min and max of scores to color scale
    norm = mpl.colors.Normalize(vmin=np.nanmin(scores), vmax=np.nanmax(scores))
    bar_colors = [cmap(norm(s)) for s in scores]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, scores, color=bar_colors, edgecolor="#222222", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)

    # annotate numeric score above bars (closer and lighter)
    y_off_unit = 0.003 * max(1.0, np.nanmax(np.abs(scores)))
    for rect, v in zip(bars, scores):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_off_unit,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="normal",
                color="#222222",
            )

    # highlight top model label
    if len(plot_df) > 0:
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.get_xticklabels()[0].set_color("#2ca02c")
        winner_desc = plot_df.loc[0, "short"]
    else:
        winner_desc = None

    # grid and colorbar
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Quality (red = worst → green = best)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # title
    plt.title("Rosetta Interface Analyzer Composite Score", fontsize=14, weight="semibold")
    plt.tight_layout()

    outpath = Path(out_png)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"✅ Saved vertical plot to: {outpath} — winner: {winner_desc}")
    plt.show()


# ----------------------------
# Call the function with your paths and exclude pattern for the 6Y6C row
# ----------------------------
csv_path = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics_scored.csv"
out_png = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\composite_scores_from_csv.png"

# exclude any description containing '6Y6C' or the full WT pattern; adjust pattern if needed
plot_from_scored_csv_vertical(csv_path=csv_path, out_png=out_png, figsize=(9,8),
                              exclude_patterns=[r"6Y6C", r"relaxed_6Y6C"])