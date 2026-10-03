"""
evaluation/plot_validation.py
-----------------------------
Figure for the bandwidth-validation extension, in the style of the paper's
Fig. 6: for each scenario/window, the fusion strategy BEFORE and AFTER the
feedback round (top) and the required rate vs available bandwidth (bottom).

Reads validation_log.csv (written by --validate_bandwidth); one or more files
can be given (e.g. train + test).

Usage
-----
python evaluation/plot_validation.py \
    --validation_log results/validated/validation_log.csv \
    --output results/artifact_evaluation/validation.pdf
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_FM_LEVEL = {"late": 0, "intermediate": 1, "early": 2}
_FM_LABELS = ["Late", "Intermediate", "Early"]


def _scenario_dt(name):
    try:
        return datetime.strptime(name, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return datetime.min


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"frame_name": str}) for p in paths],
                   ignore_index=True)
    df["scenario"] = df["scenario_path"].map(lambda p: os.path.basename(os.path.normpath(p)))
    df = df.sort_values(by=["scenario", "frame_name"],
                        key=lambda c: c.map(_scenario_dt) if c.name == "scenario" else c)
    # Same labels as the paper: S01/W1, S01/W2, ...
    ids = {sc: f"S{i + 1:02d}" for i, sc in enumerate(dict.fromkeys(df["scenario"]))}
    df["label"] = [f"{ids[sc]}/W{k + 1}" for sc, k in
                   zip(df["scenario"], df.groupby("scenario").cumcount())]
    df["corrected"] = df["n_feedback_rounds"] > 0
    return df.reset_index(drop=True)


def plot(df, output):
    x = np.arange(len(df))
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7.16, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})

    # -- Top: fusion strategy before / after feedback, with bandwidth ----------
    init = df["initial_fusion"].map(_FM_LEVEL)
    final = df["final_fusion"].map(_FM_LEVEL)
    ax1.scatter(x, init, marker="x", s=30, c="#C0392B", zorder=3,
                label="LLM initial decision")
    ax1.scatter(x, final, marker="o", s=22, facecolors="none",
                edgecolors="#1F618D", linewidths=1.2, zorder=4,
                label="After validation")
    for xi, a, b in zip(x[df["corrected"]], init[df["corrected"]], final[df["corrected"]]):
        ax1.annotate("", xy=(xi, b), xytext=(xi, a),
                     arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=0.9))
    # CAV-count changes (e.g. "3→2 CAVs"), invisible in the fusion level alone
    n0 = df["initial_cavs"].astype(str).str.split("|").str.len()
    n1 = df["final_cavs"].astype(str).str.split("|").str.len()
    for xi, a, b, lvl in zip(x, n0, n1, final):
        if a != b:
            ax1.text(xi, lvl - 0.32, f"{a}→{b} CAVs", ha="center", fontsize=7,
                     color="#1F618D")
    ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(_FM_LABELS, fontsize=9)
    ax1.set_ylim(-0.5, 2.5); ax1.set_ylabel("Fusion strategy", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, df["bandwidth_mbps"], "--D", ms=2, lw=0.9, color="#6BBF59",
             alpha=0.8, label="Bandwidth (Mbps)")
    ax2.set_ylim(0, df["bandwidth_mbps"].max() * 1.15)
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)

    # -- Bottom: required rate before / after vs available bandwidth ----------
    lo = 1e-2
    ax3.step(x, df["bandwidth_mbps"], where="mid", color="#6BBF59", lw=1.2,
             label="Available bandwidth")
    ax3.scatter(x, df["initial_rate_mbps"].clip(lower=lo), marker="x", s=30,
                c="#C0392B", zorder=3, label="Required rate, initial")
    ax3.scatter(x, df["final_rate_mbps"].clip(lower=lo), marker="o", s=22,
                facecolors="none", edgecolors="#1F618D", linewidths=1.2, zorder=4,
                label="Required rate, after validation")
    ax3.set_yscale("log"); ax3.set_ylim(lo, 300)
    ax3.set_ylabel("Rate (Mbps, log)", fontsize=9)
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False)
    ax3.set_xticks(x); ax3.set_xticklabels(df["label"], rotation=90, fontsize=8)
    ax3.set_xlabel("Scenario / Window", fontsize=9)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved to {output}")


def main():
    p = argparse.ArgumentParser(description="Plot validator decisions before/after feedback")
    p.add_argument("--validation_log", nargs="+", required=True)
    p.add_argument("--output", default="results/artifact_evaluation/validation.pdf")
    args = p.parse_args()

    df = load(args.validation_log)
    n_corr = int(df["corrected"].sum())
    print(f"{len(df)} windows | {n_corr} corrected by feedback | "
          f"{int(df['fallback_used'].sum())} fallbacks | "
          f"{int((~df['feasible']).sum())} still infeasible")
    for _, r in df[df["corrected"]].iterrows():
        print(f"  {r['label']:8s} {r['scenario']}/{r['frame_name']}  "
              f"{r['initial_fusion']}[{r['initial_cavs']}] {r['initial_rate_mbps']:.1f} Mbps "
              f"-> {r['final_fusion']}[{r['final_cavs']}] {r['final_rate_mbps']:.2f} Mbps "
              f"(B = {r['bandwidth_mbps']:.1f})")
    plot(df, args.output)


if __name__ == "__main__":
    main()