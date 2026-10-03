"""
evaluation/make_static_fallback.py
----------------------------------
Build a selection CSV for a STATIC FALLBACK baseline, to compare with the
LLM-feedback validator on exactly the same windows.

For every window, start from the LLM's INITIAL decision (validation_log.csv):
  * if it is feasible               -> keep it unchanged
  * if it exceeds the bandwidth     -> keep the SAME CAVs, change only the
                                       fusion strategy (deterministic rule)

Rules (--mode):
  richest (default)  downgrade to the richest fusion that fits:
                     early -> intermediate if it fits, else late;
                     intermediate -> late
  late               always switch to late fusion

Costs use the same model as feasibility_validator.py (intermediate =
140,800 B per non-ego CAV per frame, 10 Hz); late fusion always fits.

The other columns (ego, frame_idx, in-range CAVs, ...) are copied from the
selection.csv of the run WITH validation, so Phase 2 can run on the output.

Usage
-----
python evaluation/make_static_fallback.py \
    --validation_log results/validated/gemma4_12b_w/validation_log.csv \
    --selection_csv  results/validated/gemma4_12b_w/selection.csv \
    --mode richest \
    --output results/validated/gemma4_12b_static/selection.csv
"""

import argparse
import os

import pandas as pd

FRAME_RATE_HZ = 10.0
INTERMEDIATE_BYTES_PER_CAV = 140_800          # same as feasibility_validator.py
LEVEL = {"late": 0, "intermediate": 1, "early": 2}


def intermediate_rate_mbps(n_tx: int) -> float:
    return n_tx * INTERMEDIATE_BYTES_PER_CAV * 8 * FRAME_RATE_HZ / 1e6


def static_decision(fusion: str, n_tx: int, rate: float, bw: float, mode: str):
    """Return (fusion_after_fallback, was_infeasible)."""
    if rate <= bw:
        return fusion, False
    if mode == "late":
        return "late", True
    # richest feasible fusion strictly below the initial one, same CAVs
    if fusion == "early" and intermediate_rate_mbps(n_tx) <= bw:
        return "intermediate", True
    return "late", True


def main():
    p = argparse.ArgumentParser(description="Static-fallback baseline selection CSV")
    p.add_argument("--validation_log", required=True)
    p.add_argument("--selection_csv", required=True,
                   help="selection.csv of the run WITH validation (template)")
    p.add_argument("--mode", choices=["richest", "late"], default="richest")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    log = pd.read_csv(args.validation_log, dtype={"frame_name": str, "initial_cavs": str,
                                                  "final_cavs": str})
    log["scenario"] = log["scenario_path"].map(lambda s: os.path.basename(os.path.normpath(s)))
    sel = pd.read_csv(args.selection_csv, dtype=str)

    key = ["scenario", "frame_name"]
    merged = sel.merge(log, on=key, how="left", validate="one_to_one")
    missing = merged["initial_fusion"].isna().sum()
    if missing or len(sel) != len(log):
        raise SystemExit(f"Windows do not match: selection={len(sel)}, log={len(log)}, "
                         f"unmatched={missing}. Use the selection.csv of the SAME run.")

    out_rows, changed = [], []
    for _, r in merged.iterrows():
        cavs = [c for c in str(r["initial_cavs"]).split("|") if c]
        n_tx = len([c for c in cavs if c != r["ego_cav"]])
        fm, infeasible = static_decision(r["initial_fusion"], n_tx,
                                         float(r["initial_rate_mbps"]),
                                         float(r["bandwidth_mbps_y"]), args.mode)
        row = {c: r[c] for c in sel.columns if c in r and not c.endswith("_y")}
        row.update({
            "bandwidth_mbps": r["bandwidth_mbps_x"],
            "selected_cavs": "|".join(cavs),
            "n_selected": len(cavs),
            "fusion_method": fm,
            "reason": (f"Static fallback ({args.mode}): {r['initial_fusion']} needed "
                       f"{float(r['initial_rate_mbps']):.2f} Mbps > "
                       f"{float(r['bandwidth_mbps_y']):.2f}; switched to {fm}, same CAVs."
                       if infeasible else "LLM initial decision (feasible, unchanged)."),
            "llm_response_time_s": "",       # not comparable: no feedback round here
        })
        out_rows.append(row)
        if infeasible:
            changed.append((r["scenario"], r["frame_name"], r["initial_fusion"], fm,
                            len(cavs), r["final_fusion"], r["final_cavs"]))

    out = pd.DataFrame(out_rows)[sel.columns.tolist()]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"{len(out)} windows | {len(changed)} infeasible initial decisions "
          f"corrected by the static fallback ({args.mode})")
    print(f"{'window':34s} {'initial':13s} {'static':13s} {'LLM feedback':s}")
    for sc, fr, f0, fs, n, ff, fc in changed:
        n_llm = len(str(fc).split("|"))
        mark = "" if (fs == ff and n == n_llm) else "   <- differs"
        print(f"{sc + '/' + fr:34s} {f0 + f'({n})':13s} {fs + f'({n})':13s} "
              f"{ff}({n_llm}){mark}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()