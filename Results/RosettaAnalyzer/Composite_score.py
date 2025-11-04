# ----------------------------
# Composite scoring from Rosetta and PISA metrics
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ----------------------------
# Helper functions
# ----------------------------
def normalize_minmax(series):
    """Standard min-max normalization to [0,1]"""
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)  # avoid div by zero
    return (series - series.min()) / (series.max() - series.min())

def normalize_inverted(series):
    """Inverted min-max normalization: lower is better"""
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series.max() - series) / (series.max() - series.min())

def short_label(name):
    if pd.isna(name):
        return "unknown"
    name = str(name)  # ensure it’s a string
    m = re.search(r"design(\d+)_n(\d+)", name)
    if m:
        return f"d{m.group(1)}.n{m.group(2)}"
    m2 = re.search(r"d(?:esign)?[_-]?(\d+)[^A-Za-z0-9]*n[_-]?(\d+)", name)
    if m2:
        return f"d{m2.group(1)}.n{m2.group(2)}"
    return (name[:20] + "...") if len(name) > 23 else name



# ----------------------------
# Main function
# ----------------------------
def combine_rosetta_pisa(rosetta_csv, pisa_csv, out_csv, out_png, exclude_patterns=None):
    # --- Load CSVs ---
    df_r = pd.read_csv(rosetta_csv)
    df_p = pd.read_csv(pisa_csv)

    # --- Keep only necessary columns for scoring ---
    # Rosetta
    df_r_sub = df_r[['description','dG_separated','dSASA_int','hbonds_int']].copy()
    # PISA
    df_p_sub = df_p[['filename','int_area','stab_en','n_hbonds','pvalue']].copy()

    # --- Normalize Rosetta metrics ---
    df_r_sub['dG_norm'] = normalize_inverted(df_r_sub['dG_separated'])
    df_r_sub['area_norm'] = normalize_minmax(df_r_sub['dSASA_int'])
    df_r_sub['hbonds_norm'] = normalize_minmax(df_r_sub['hbonds_int'])

    # --- Rosetta score (weights rescaled to sum=1) ---
    df_r_sub['Rosetta_score'] = (
            0.444 * df_r_sub['dG_norm'] +
            0.333 * df_r_sub['area_norm'] +
            0.222 * df_r_sub['hbonds_norm']
    )

    # --- Normalize PISA metrics ---
    df_p_sub['area_norm'] = normalize_minmax(df_p_sub['int_area'])
    df_p_sub['stab_norm'] = normalize_inverted(df_p_sub['stab_en'])
    df_p_sub['hbonds_norm'] = normalize_minmax(df_p_sub['n_hbonds'])
    df_p_sub['pval_norm'] = normalize_inverted(df_p_sub['pvalue'])

    # --- PISA score ---
    df_p_sub['PISA_score'] = (
        0.4*df_p_sub['area_norm'] +
        0.3*df_p_sub['stab_norm'] +
        0.2*df_p_sub['hbonds_norm'] +
        0.1*df_p_sub['pval_norm']
    )

    # --- Merge Rosetta and PISA scores ---
    # Align names (basic heuristic: remove 'relaxed_' prefix from Rosetta description)
    df_r_sub['merge_name'] = df_r_sub['description'].str.replace(r"^relaxed_", "", regex=True).str.replace(r"_\d{4}_\d{4}$", "", regex=True)
    df_p_sub['merge_name'] = df_p_sub['filename'].str.replace(r"_relaxed$", "", regex=True)

    df_merged = pd.merge(df_r_sub, df_p_sub, on='merge_name', how='outer', suffixes=('_R','_P'))

    # Fill missing PISA_score with 0
    df_merged['PISA_score'] = df_merged['PISA_score'].fillna(0.0)

    # --- Normalize Rosetta_score and PISA_score across all models ---
    df_merged['Rosetta_score_norm'] = normalize_minmax(df_merged['Rosetta_score'])
    df_merged['PISA_score_norm'] = normalize_minmax(df_merged['PISA_score'])

    # --- Composite score ---
    df_merged['composite_score'] = 0.7*df_merged['Rosetta_score_norm'] + 0.3*df_merged['PISA_score_norm']

    # --- Optional: apply exclusions ---
    if exclude_patterns:
        mask = pd.Series(False, index=df_merged.index)
        for pat in exclude_patterns:
            mask = mask | df_merged['description'].astype(str).str.contains(pat, regex=True)
        df_merged = df_merged[~mask]

    # --- Save CSV ---
    df_merged.to_csv(out_csv, index=False)
    print(f"Saved combined scores to: {out_csv}")

    # --- Plot vertical bar ---
    labels = df_merged['description'].astype(str).map(short_label).tolist()
    scores = df_merged['composite_score'].to_numpy()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14,7))
    bars = ax.bar(x, scores, color="#1f77b4", edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Composite score (higher = better)", fontsize=12)
    ax.set_xlabel("Model", fontsize=11)

    # annotate
    y_off = 0.01*max(1.0, np.nanmax(np.abs(scores)))
    for rect, v in zip(bars, scores):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+y_off, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#111111")

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.title("Composite Scores from Rosetta + PISA", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {out_png}")

    return df_merged

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rosetta_csv = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics.csv"
    pisa_csv = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\PISA\pisa_interfaces_summary.csv"
    out_csv = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\combined_scores.csv"
    out_png = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\combined_scores.png"

    combine_rosetta_pisa(rosetta_csv, pisa_csv, out_csv, out_png, exclude_patterns=[r"6Y6C"])
