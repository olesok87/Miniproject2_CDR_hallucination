# ----------------------------
# Vertical plotting from scored CSV with Rosetta + PISA weights
# ----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def short_label(name):
    """Short label for plotting."""
    m = re.search(r"design(\d+)_n(\d+)", name)
    if m:
        return f"d{m.group(1)}.n{m.group(2)}"
    m2 = re.search(r"d(?:esign)?[_-]?(\d+)[^A-Za-z0-9]*n[_-]?(\d+)", name)
    if m2:
        return f"d{m2.group(1)}.n{m2.group(2)}"
    return (name[:20] + "...") if len(name) > 23 else name

def normalize_series(series, invert=False):
    """Min-max normalization of a pandas Series."""
    s = series.astype(float)
    if invert:
        s = -s  # lower = better
    return (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s))

def plot_from_rosetta_pisa(csv_path_rosetta, csv_path_pisa,
                            out_png, figsize=(14,7),
                            xlabel="Model", ylabel="Composite score (higher = better)",
                            exclude_patterns=None,
                            rosetta_weight=0.7, pisa_weight=0.3):
    # --- Load CSVs ---
    df_rosetta = pd.read_csv(csv_path_rosetta)
    df_pisa = pd.read_csv(csv_path_pisa)

    if not {"description","composite_score","color"}.issubset(df_rosetta.columns):
        raise ValueError("Rosetta CSV must contain columns: description, composite_score, color")
    if not {"filename","stab_en"}.issubset(df_pisa.columns):
        raise ValueError("PISA CSV must contain columns: filename, stab_en")

    # --- Normalize ---
    df_rosetta["norm_rosetta"] = normalize_series(df_rosetta["composite_score"], invert=True)
    df_pisa["norm_pisa"] = normalize_series(df_pisa["stab_en"], invert=True)

    # --- Match names between Rosetta and PISA ---
    def match_name(rname):
        # relaxed_design0_n0_0001_0001 -> d0_n0_relaxed
        m = re.search(r"design(\d+)_n(\d+)", rname)
        if m:
            return f"d{m.group(1)}_n{m.group(2)}_relaxed"
        if rname.startswith("relaxed_WT"):
            return "relaxed_WT"
        return rname

    df_rosetta["pisa_match"] = df_rosetta["description"].map(match_name)
    df_merged = df_rosetta.merge(df_pisa[["filename","norm_pisa"]],
                                 left_on="pisa_match", right_on="filename", how="left")
    df_merged["norm_pisa"] = df_merged["norm_pisa"].fillna(0.0)

    # --- Composite weighted score ---
    df_merged["composite_score"] = rosetta_weight * df_merged["norm_rosetta"] + \
                                   pisa_weight * df_merged["norm_pisa"]

    # --- Apply exclusions ---
    if exclude_patterns:
        mask = pd.Series(False, index=df_merged.index)
        for pat in exclude_patterns:
            mask |= df_merged["description"].astype(str).str.contains(pat, regex=True)
        df_merged = df_merged[~mask].reset_index(drop=True)

    # --- Prepare for plotting ---
    df_merged["short"] = df_merged["description"].map(short_label)
    plot_df = df_merged[["short","composite_score","color"]].copy()
    plot_df = plot_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    labels = plot_df["short"].tolist()
    scores = plot_df["composite_score"].astype(float).to_numpy()
    colors = plot_df["color"].astype(str).tolist()
    colors = [c if c.startswith("#") else ("#" + c.lstrip()) for c in colors]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, scores, color=colors, edgecolor="#222222", linewidth=0.6)

    # annotate numeric score above bars
    y_off_unit = 0.01 * max(1.0, np.nanmax(np.abs(scores)))
    for rect, v in zip(bars, scores):
        if np.isfinite(v):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + y_off_unit,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#111111")

    # highlight top model
    if len(plot_df) > 0:
        ax.get_xticklabels()[0].set_fontweight("bold")
        try:
            ax.get_xticklabels()[0].set_color("#2ca02c")
        except Exception:
            pass
        winner_desc = plot_df.loc[0, "short"]
    else:
        winner_desc = None

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.title("Rosetta + PISA composite score ranking", fontsize=13)
    plt.tight_layout()
    outpath = Path(out_png)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {outpath} — winner: {winner_desc}")
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    csv_path_rosetta = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics_scored.csv"
    csv_path_pisa = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\PISA\pisa_interfaces_summary.csv"
    out_png = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\composite_rosetta_pisa.png"

    plot_from_rosetta_pisa(csv_path_rosetta, csv_path_pisa, out_png,
                           exclude_patterns=[r"6Y6C", r"relaxed_6Y6C"])
