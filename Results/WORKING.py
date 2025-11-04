# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import re

# === 1. User-configurable paths ===
PISA_CSV = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\PISA\pisa_interfaces_summary.csv"
ROSETTA_CSV = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics.csv"

PISA_PNG = Path(PISA_CSV).parent / "pisa_ranking.png"
ROSETTA_PNG = Path(ROSETTA_CSV).parent / "rosetta_ranking.png"


# === 2. Shorten labels function ===
def short_label(desc, remove_relaxed=False):
    s = str(desc).strip()
    if not s:
        return ""
    # split by common delimiters and take the last meaningful segment
    parts = re.split(r"[\\\/\|\:,]+", s)
    core = parts[-1].strip()
    # strip common file extensions
    core = re.sub(r"\.(pdb|pdb\.gz|cif|mmcif|txt|csv|gz)$", "", core, flags=re.I)
    # remove unwanted numeric token appearing in some names
    core = re.sub(r"0001_0001", "", core)
    tmp = re.sub(r"[\(\)]", "", core).strip().lower()

    # remove _relaxed if requested (for PISA)
    if remove_relaxed:
        core = re.sub(r"_relaxed", "", core, flags=re.I)

    # if token contains both 'relaxed' and '6y6c', normalize whole label to '6Y6C'
    if "relaxed" in tmp and "6y6c" in tmp:
        core = "6Y6C"
    else:
        core = re.sub(r"(?i)\b6y6c\b", "6Y6C", core)

    # normalize design0 naming patterns to canonical form `d0.nX`
    core = re.sub(r"(?i)(?:relaxed[_\-]*)?design0[_\-]*n0*([0-9]+)", r"d0.n\1", core)
    core = re.sub(r"(?i)\bd0[_\-]*n0*([0-9]+)\b", r"d0.n\1", core)

    # normalize design1 naming patterns to canonical form `d1.nX`
    core = re.sub(r"(?i)(?:relaxed[_\-]*)?design1[_\-]*n0*([0-9]+)", r"d1.n\1", core)
    core = re.sub(r"(?i)\bd1[_\-]*n0*([0-9]+)\b", r"d1.n\1", core)

    # remove trailing underscore if present (for Rosetta)
    core = re.sub(r"_$", "", core)

    return core

# === 3. Read CSVs ===
df = pd.read_csv(PISA_CSV)        # PISA CSV
df_r = pd.read_csv(ROSETTA_CSV)   # Rosetta CSV

# === PISA labels ===
df["label"] = df["filename"].map(lambda x: short_label(x, remove_relaxed=True))

# === Rosetta labels ===
df_r["label"] = df_r["description"].map(lambda x: short_label(x, remove_relaxed=False))



# === 4. Normalize PISA metrics ===
def normalize(col):
    col = col.astype(float)
    rng = col.max() - col.min()
    if rng == 0 or np.isnan(rng):
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - col.min()) / rng

df["Area_norm"] = normalize(df["int_area"])
df["Stability_norm"] = normalize(-df["stab_en"])
df["H-bonds_norm"] = normalize(df["n_hbonds"])
df["p-value_norm"] = 1 - normalize(df["pvalue"])

df["PISA_score"] = (
    0.4 * df["Area_norm"] +
    0.3 * df["Stability_norm"] +
    0.2 * df["H-bonds_norm"] +
    0.1 * df["p-value_norm"]
)

# Short labels for x-axis
df["label"] = df["filename"].map(short_label)

# === 5. Normalize Rosetta scores ===
# Normalize each Rosetta component (0 = worst, 1 = best)
df_r["dG_sep_norm"] = normalize(-df_r["dG_separated"])      # lower (more negative) is better
df_r["dSASA_norm"] = normalize(df_r["dSASA_int"])           # higher = better
df_r["hbonds_norm"] = normalize(df_r["hbonds_int"])         # higher = better

# Compute weighted Rosetta score (like PISA)
df_r["Rosetta_score"] = (
    0.4 * df_r["dG_sep_norm"] +
    0.4 * df_r["dSASA_norm"] +
    0.2 * df_r["hbonds_norm"]
)

# Use this directly as normalized score (already 0–1)
df_r["Rosetta_norm"] = df_r["Rosetta_score"]


# === 6. Colormap ===
cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["#d62728", "#ffcc00", "#2ca02c"])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# === 7. Plot PISA ranking ===
labels_pisa = df["label"].tolist()
scores_pisa = df["PISA_score"].tolist()
bar_colors_pisa = [cmap(norm(s)) for s in scores_pisa]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels_pisa, scores_pisa, color=bar_colors_pisa, edgecolor="k", linewidth=0.6)
ax.set_title("Normalized PISA-based ranking", fontsize=14)
ax.set_ylabel("PISA score (0–1)", fontsize=12)
ax.set_xticklabels(labels_pisa, rotation=35, ha="right", fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

# Numeric labels
for bar, score in zip(bars, scores_pisa):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{score:.2f}",
            ha="center", va="bottom", fontsize=9, weight="normal", color="black")

# Colorbar
plt.colorbar(sm, ax=ax, pad=0.02, label="Quality (red = worst → green = best)")

plt.tight_layout()
plt.savefig(PISA_PNG, dpi=300, bbox_inches="tight")
plt.show()

# === 8. Plot Rosetta ranking ===
labels_ros = df_r["label"].tolist()
scores_ros = df_r["Rosetta_norm"].tolist()
bar_colors_ros = [cmap(norm(s)) for s in scores_ros]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels_ros, scores_ros, color=bar_colors_ros, edgecolor="k", linewidth=0.6)
ax.set_title("Normalized Rosetta ranking", fontsize=14)
ax.set_ylabel("Rosetta normalized score", fontsize=12)
ax.set_xticklabels(labels_ros, rotation=35, ha="right", fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

# Numeric labels
for bar, score in zip(bars, scores_ros):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{score:.2f}",
            ha="center", va="bottom", fontsize=9, weight="normal", color="black")

# Colorbar
plt.colorbar(sm, ax=ax, pad=0.02, label="Quality (red = worst → green = best)")

plt.tight_layout()
plt.savefig(ROSETTA_PNG, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Saved PISA ranking → {PISA_PNG}")
print(f"✅ Saved Rosetta ranking → {ROSETTA_PNG}")
