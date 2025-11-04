#!/usr/bin/env python3
# compare_interfaces_colormap_with_custom_labels.py
"""
Parse a Rosetta score file and plot 5 small comparison graphs (dG_cross, hbonds_int,
hbond_E_fraction, delta_unsatHbonds, sc_value) with a unified red->green colorbar.
Replace the model descriptions (x-axis labels) with custom names:
  WR (6y6C), d0n0, d1n0, d1n1, d1.n2, d1.n3
If the number of names doesn't match the number of rows in the score file the script
will use the first N names or append numeric labels as needed.

Designed to run in PyCharm. Edit SCORE_PATH and OUT_PNG as required.
"""

import re
import math
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import csv
from pathlib import Path


sns.set(style="whitegrid", context="notebook")

# ===== USER CONFIG =====
SCORE_PATH = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\score.sc"
OUT_PNG = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\Interface_Analyzer_Rosetta.png"
# Custom model labels (the user requested six names). The script will map these onto the rows.
CUSTOM_LABELS = ["WT (6y6C)", "d0.n0", "d1.n0", "d1.n1", "d1.n2", "d1.n3"]
# Increase overall figure/text size
FIGSIZE = (16, 9)
TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 10
AX_TITLE_FONTSIZE = 11
XTICK_FONTSIZE = 10
YTICK_FONTSIZE = 12
LABEL_FONTWEIGHT = "bold"
# =======================

# Selected criteria (packstat removed), preference (True = lower is better), and full names
CRITERIA = [
    "complex_normalized",
    "dSASA_int",
    "hbonds_int",
    "hbond_E_fraction",
    "delta_unsatHbonds",
    "sc_value",
]
PREFER_LOWER = {
    "complex_normalized": True,
    "dSASA_int": False,
    "hbonds_int": False,
    "hbond_E_fraction": False,
    "delta_unsatHbonds": True,
    "sc_value": False,
}
FULL_NAMES = {
    "complex_normalized": "Predicted binding free energy normalized",
    "dSASA_int": "Buried interface area (dSASA_int, Å²)",
    "hbonds_int": "Number of interface hydrogen bonds (hbonds_int)",
    "hbond_E_fraction": "Fraction of interface energy from H-bonds (hbond_E_fraction)",
    "delta_unsatHbonds": "Buried unsatisfied hydrogen bonds (delta_unsatHbonds)",
    "sc_value": "Shape complementarity (sc_value)",
}


def find_score_header(lines: List[str]) -> Tuple[int, str]:
    for i, L in enumerate(lines):
        if L.strip().startswith("SCORE:") and ("description" in L or "dG_cross" in L or "complex_normalized" in L):
            return i, L
    for i, L in enumerate(lines):
        if L.strip().startswith("SCORE:"):
            return i, L
    raise RuntimeError("Could not find SCORE: header in file.")


def parse_scorefile(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read().splitlines()
    h_idx, header_line = find_score_header(raw)
    header = re.sub(r"^SCORE:\s*", "", header_line.strip())
    col_names = re.split(r"\s+", header)
    if "description" not in col_names:
        col_names[-1] = "description"
    rows = []
    for L in raw[h_idx + 1 :]:
        if not L.strip():
            continue
        if not L.strip().startswith("SCORE:"):
            continue
        line = re.sub(r"^SCORE:\s*", "", L).rstrip()
        parts = re.split(r"\s+", line)
        if len(parts) > len(col_names):
            parts = parts[: len(col_names) - 1] + [" ".join(parts[len(col_names) - 1 :])]
        if len(parts) < len(col_names):
            parts = parts + [np.nan] * (len(col_names) - len(parts))
        rows.append(parts)
    df = pd.DataFrame(rows, columns=col_names)
    return df


def parse_cell_val(cell: Any):
    if pd.isna(cell):
        return None
    s = str(cell).strip().replace(",", " ")
    tokens = re.split(r"\s+", s)
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except Exception:
            break
        if len(nums) >= 2:
            break
    if len(nums) == 0:
        return None
    if len(nums) == 1:
        return nums[0]
    return nums[:2]


def prepare_data(df: pd.DataFrame):
    if "description" not in df.columns:
        df["description"] = [f"row{i}" for i in range(len(df))]
    records = []
    for i, row in df.iterrows():
        rec = {"description": row.get("description", f"row{i}")}
        for c in CRITERIA:
            rec[c] = parse_cell_val(row[c]) if c in df.columns else None
        records.append(rec)
    tdf = pd.DataFrame(records)

    is_pair = {}
    numeric_data = {}
    for c in CRITERIA:
        has_pair = tdf[c].apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).any()
        is_pair[c] = bool(has_pair)
        if is_pair[c]:
            v1, v2 = [], []
            for v in tdf[c]:
                if isinstance(v, (list, tuple, np.ndarray)):
                    if len(v) >= 2:
                        v1.append(float(v[0])); v2.append(float(v[1]))
                    elif len(v) == 1:
                        v1.append(float(v[0])); v2.append(np.nan)
                    else:
                        v1.append(np.nan); v2.append(np.nan)
                elif v is None:
                    v1.append(np.nan); v2.append(np.nan)
                else:
                    v1.append(float(v)); v2.append(np.nan)
            numeric_data[c + "_1"] = np.array(v1, dtype=float)
            numeric_data[c + "_2"] = np.array(v2, dtype=float)
        else:
            vals = []
            for v in tdf[c]:
                if v is None:
                    vals.append(np.nan)
                else:
                    vals.append(float(v))
            numeric_data[c] = np.array(vals, dtype=float)
    return tdf, is_pair, numeric_data

def save_parsed_metrics_csv(raw_df: pd.DataFrame,
                            tdf: pd.DataFrame,
                            is_pair: dict,
                            numeric_data: dict,
                            quality: dict,
                            out_csv: str):
    """
    Save a tidy CSV directly from parsed score.sc content.

    Columns:
      - description
      - for each metric M: M (representative), M_1, M_2
      - for each metric M: M_quality (0..1 normalized quality)
    raw_df: original DataFrame parsed from score.sc (the one returned by parse_scorefile)
    tdf: tidy parsed table produced by prepare_data (contains descriptions)
    is_pair, numeric_data, quality: as produced by prepare_data/compute_quality_arrays
    out_csv: output path
    """
    import csv

    metrics = list(is_pair.keys())
    n = len(tdf)

    rows = []
    for i in range(n):
        row = {}
        # description from tdf (more reliable)
        row["description"] = str(tdf.loc[i, "description"]) if "description" in tdf.columns else f"row{i+1}"
        # include raw_df scalar columns if raw_df length matches and column not duplicate metric/description
        if raw_df is not None and len(raw_df) == n:
            for col in raw_df.columns:
                if col in metrics or col == "description":
                    continue
                try:
                    val = raw_df.iloc[i][col]
                    row[col] = "" if pd.isna(val) else val
                except Exception:
                    row[col] = ""
        # per metric values (pairs and representative)
        for m in metrics:
            if is_pair.get(m, False):
                a = numeric_data.get(m + "_1", np.full(n, np.nan))[i]
                b = numeric_data.get(m + "_2", np.full(n, np.nan))[i]
                # representative: mean of available numeric values
                rep = np.nanmean([a, b])
                row[f"{m}_1"] = "" if np.isnan(a) else a
                row[f"{m}_2"] = "" if np.isnan(b) else b
                row[f"{m}"] = "" if np.isnan(rep) else rep
            else:
                arr = numeric_data.get(m, np.full(n, np.nan))
                v = arr[i]
                row[f"{m}"] = "" if np.isnan(v) else v
                row[f"{m}_1"] = ""
                row[f"{m}_2"] = ""
            # quality value
            qarr = quality.get(m, np.full(n, np.nan))
            qv = qarr[i] if i < len(qarr) else np.nan
            row[f"{m}_quality"] = "" if np.isnan(qv) else qv
        rows.append(row)

    # build header ordering: description, metrics, metric_1, metric_2, metric_quality, then remaining raw_df cols
    header = ["description"]
    for m in metrics:
        header.append(m)
    for m in metrics:
        header.append(f"{m}_1")
        header.append(f"{m}_2")
    for m in metrics:
        header.append(f"{m}_quality")
    if raw_df is not None and len(raw_df) == n:
        for col in raw_df.columns:
            if col not in header:
                header.append(col)

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out_r = {k: (r.get(k, "")) for k in header}
            writer.writerow(out_r)

    print(f"Saved parsed metrics CSV: {out_csv}")

def compute_quality_arrays(tdf, is_pair, numeric_data):
    quality = {}
    for c in CRITERIA:
        if is_pair[c]:
            a = numeric_data[c + "_1"]
            b = numeric_data[c + "_2"]
            avg = np.nanmean(np.vstack([a, b]), axis=0)
            vals = avg
        else:
            vals = numeric_data[c]
        valid = ~np.isnan(vals)
        q = np.full_like(vals, np.nan, dtype=float)
        if valid.sum() > 0:
            subset = vals[valid]
            if PREFER_LOWER.get(c, True):
                best = np.nanmin(subset)
                worst = np.nanmax(subset)
                denom = worst - best if not math.isclose(worst, best) else 1.0
                q[valid] = 1.0 - (subset - best) / denom
            else:
                best = np.nanmax(subset)
                worst = np.nanmin(subset)
                denom = best - worst if not math.isclose(best, worst) else 1.0
                q[valid] = (subset - worst) / denom
            q = np.clip(q, 0.0, 1.0)
        quality[c] = q
    return quality


def assign_custom_labels(n_rows: int, custom_labels: List[str]) -> List[str]:
    """
    Return a label list of length n_rows using custom_labels.
    If custom_labels has more entries than n_rows, use the first n_rows.
    If fewer, append numeric suffix labels to reach n_rows.
    """
    if len(custom_labels) >= n_rows:
        return custom_labels[:n_rows]
    else:
        labels = list(custom_labels)
        # append numeric fallback labels
        for i in range(len(labels), n_rows):
            labels.append(f"model_{i+1}")
        return labels


def plot_with_colormap(tdf, is_pair, numeric_data, quality, out_png, custom_labels):
    n = len(tdf)
    labels = assign_custom_labels(n, custom_labels)

    # Create figure with colorbar area on the right

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.8, 0.8, 0.8, 0.14], wspace=0.8, hspace=0.45)

    # lower the top of the subplot area so the first row sits lower on the page
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.14)

    # Create six axes (2 rows x 3 cols) so they match six metrics
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),

    ]
    # cax spans right column initially; then shrink it to ~50% height so the legend is half size
    cax = fig.add_subplot(gs[:, 3])
    pos = cax.get_position()
    new_y0 = pos.y0 + (pos.height * 0.25)
    new_height = pos.height * 0.5
    cax.set_position([pos.x0, new_y0, pos.width, new_height])
    cax.set_title("quality\n(red worst → green best)", fontsize=10, pad=60)

    # colormap red -> yellow -> green
    cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["#d62728", "#ffcc00", "#2ca02c"])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for ax_idx, c in enumerate(CRITERIA):
        ax = axes[ax_idx]
        # full descriptive label text placed above the axis
        full_label = FULL_NAMES.get(c, c)
        ax.text(0.5, 1.06, full_label, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=SUBTITLE_FONTSIZE, weight=LABEL_FONTWEIGHT)
        ax.set_title("", fontsize=AX_TITLE_FONTSIZE)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
        q = quality[c]
        if is_pair[c]:
            v1 = numeric_data[c + "_1"]
            v2 = numeric_data[c + "_2"]
            x_pos = np.arange(n)
            for i in range(n):
                qi = q[i] if not np.isnan(q[i]) else 0.5
                color = cmap(qi)
                a = v1[i]; b = v2[i]
                if (not np.isnan(a)) and (not np.isnan(b)):
                    ax.plot([x_pos[i], x_pos[i]], [a, b], color="#666666", linewidth=0.9, zorder=1)
                    ax.scatter(x_pos[i], a, color=color, s=72, zorder=3, edgecolor="k", linewidth=0.4)
                    ax.scatter(x_pos[i], b, color=color, s=72, zorder=3, edgecolor="k", linewidth=0.4)
                elif not np.isnan(a):
                    ax.scatter(x_pos[i], a, color=color, s=84, zorder=3, edgecolor="k", linewidth=0.4)
                elif not np.isnan(b):
                    ax.scatter(x_pos[i], b, color=color, s=84, zorder=3, edgecolor="k", linewidth=0.4)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=XTICK_FONTSIZE)
        else:
            vals = numeric_data[c]
            x_pos = np.arange(n)
            bar_colors = [cmap(qi if not np.isnan(qi) else 0.5) for qi in q]
            ax.bar(x_pos, vals, color=bar_colors, edgecolor="k", linewidth=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=XTICK_FONTSIZE)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # colorbar in cax
    cb = fig.colorbar(sm, cax=cax)
    cax.set_title("quality\n(red worst → green best)", fontsize=10)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["worst", "0.25", "0.5", "0.75", "best"])

    plt.suptitle("Interface_Analyzer_PISA)", fontsize=TITLE_FONTSIZE, y=0.99)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_png}")
    plt.show()

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

    # require description; allow composite_score OR derive it from available *_quality columns
    if "description" not in df.columns:
        raise ValueError("CSV must contain column: description")

    if "composite_score" not in df.columns:
        quality_cols = [c for c in df.columns if c.endswith("_quality")]
        if len(quality_cols) == 0:
            raise ValueError(
                "CSV must contain either column: composite_score or at least one '*_quality' column"
            )
        # compute composite_score as mean of available quality columns (ignoring NaNs)
        df["composite_score"] = df[quality_cols].mean(axis=1, skipna=True)

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
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "red_yellow_green", ["#d62728", "#ffcc00", "#2ca02c"]
    )

    # handle all-NaN or constant arrays robustly
    if np.all(~np.isfinite(scores)):
        # fallback uniform gray if no finite scores
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        bar_colors = [(0.7, 0.7, 0.7, 1.0) for _ in scores]
    else:
        finite_scores = scores[np.isfinite(scores)]
        vmin = np.nanmin(finite_scores)
        vmax = np.nanmax(finite_scores)
        if math.isclose(vmin, vmax):
            # avoid zero-range normalize
            norm = mpl.colors.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        bar_colors = [cmap(norm(s)) if np.isfinite(s) else (0.7, 0.7, 0.7, 1.0) for s in scores]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, scores, color=bar_colors, edgecolor="#222222", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)

    # annotate numeric score above bars (closer and lighter)
    max_abs = 1.0 if not np.isfinite(np.nanmax(np.abs(scores))) else np.nanmax(np.abs(scores))
    y_off_unit = 0.003 * max(1.0, max_abs)
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
        xticks = ax.get_xticklabels()
        if len(xticks) > 0:
            xticks[0].set_fontweight("bold")
            xticks[0].set_color("#2ca02c")
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


def main():
    try:
        df = parse_scorefile(SCORE_PATH)
        tdf, is_pair, numeric_data = prepare_data(df)
        quality = compute_quality_arrays(tdf, is_pair, numeric_data)

        OUT_CSV = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\parsed_metrics.csv"
        save_parsed_metrics_csv(df, tdf, is_pair, numeric_data, quality, OUT_CSV)

        # main plot with custom labels
        plot_with_colormap(tdf, is_pair, numeric_data, quality, OUT_PNG, CUSTOM_LABELS)

        # add short labels and create vertical composite-score plot from CSV
        tdf["short"] = tdf["description"].astype(str).map(short_label)
        csv_path = OUT_CSV
        out_png = r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Results\RosettaAnalyzer\composite_scores_from_csv.png"
        plot_from_scored_csv_vertical(csv_path, out_png, figsize=(14, 7), xlabel="Model",
                                      ylabel="Composite score (higher = better)", exclude_patterns=None)
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
