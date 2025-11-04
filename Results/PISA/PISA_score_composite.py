import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# === 1. Read CSV ===
df = pd.read_csv(r"/Results/PISA/pisa_interfaces_summary.csv")


# === 2. Normalize metrics (0–1 scale, 1 = best) ===
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

df["Area_norm"] = normalize(df["int_area"])
df["Stability_norm"] = normalize(-df["stab_en"])     # lower = better
df["H-bonds_norm"] = normalize(df["n_hbonds"])
df["p-value_norm"] = 1 - normalize(df["pvalue"])     # smaller = better

# === 3. Compute PISA_score ===
df["PISA_score"] = (
    0.4 * df["Area_norm"]
    + 0.3 * df["Stability_norm"]
    + 0.2 * df["H-bonds_norm"]
    + 0.1 * df["p-value_norm"]
)

# === 4. Filter out WT ===
df_plot = df[~df["filename"].str.contains("WT", case=False)].copy()

# === 5. Format labels and sort ===
df_plot["label"] = (
    df_plot["filename"]
    .str.replace("_relaxed", "", regex=False)
    .str.replace("_", ".")
)
df_plot = df_plot.sort_values(by="PISA_score", ascending=False)

# === 6. Build colormap (red → yellow → green) ===
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "red_yellow_green", ["#d62728", "#ffcc00", "#2ca02c"]
)
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# === 7. Assign colors ===
bar_colors = [cmap(norm(v)) for v in df_plot["PISA_score"]]

# === 8. Plot ===
plt.figure(figsize=(8,5))
bars = plt.bar(df_plot["label"], df_plot["PISA_score"], color=bar_colors,
               edgecolor="k", linewidth=0.6)

plt.xlabel("Design variant", fontsize=12)
plt.ylabel("PISA score (normalized)", fontsize=12)
plt.title("Normalized PISA-based ranking", fontsize=14)
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

# === 9. Add numeric labels above each bar ===
for bar, score in zip(bars, df_plot["PISA_score"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.005,             # closer to the bar (was +0.02)
        f"{score:.2f}",             # 2 decimal precision
        ha="center", va="bottom",
        fontsize=9,
        weight="normal",            # lighter font weight
        color="black"               # optional for readability
    )

# === 10. Add colorbar ===
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label("Quality (red = worst → green = best)", fontsize=10)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["worst", "0.25", "0.5", "0.75", "best"])

plt.tight_layout()
plt.savefig(r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\PISA\pisa_scores_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# === 11. Save normalized data ===
df.to_csv("pisa_scores_normalized_sorted.csv", index=False)
print("✅ Saved normalized data → pisa_scores_normalized_sorted.csv")

