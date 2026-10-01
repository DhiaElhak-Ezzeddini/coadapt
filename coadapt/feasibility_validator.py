"""
coadapt/feasibility_validator.py
--------------------------------
Bandwidth-feasibility check with LLM feedback (extension of CoAdapt).

Where it sits
-------------
Between Plan and Execute in the MAPE-K loop: after the LLM has produced its
joint decision (P(t), alpha(t)) and BEFORE the decision is written to
selection.csv.  Phase 2 (inference) is not touched.

What it does
------------
1. Estimate the rate the decision needs (paper Eq. 2-3, converted to Mbps):
       R = 8 * f * sum_{non-ego r_i in P(t)} delta(alpha, r_i) / 1e6
   using the same per-CAV costs as run_pipeline.py's measurement
   (early: K_i points x 4 fields x 4 B; late: n_boxes x 7 x 4 B;
   intermediate: fixed feature-map size per CAV).
2. If R <= B(t): return the decision unchanged.
3. Otherwise re-prompt the LLM with: the original prompt + its previous
   answer + how much it exceeds the budget + what is still feasible.
   The LLM may drop CAVs, change the fusion strategy, or both.
4. After `max_retries` failed attempts: deterministic fallback
   (late fusion with the same participants).

Design choice: the original RobotAndStrategySelector.select() is reused
unchanged.  The feedback is appended to the prompt by a thin proxy around
the LLM backend, so the original JSON-parse retry and the hard constraints
C1-C3 still apply to the corrected decision.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional

FRAME_RATE_HZ = 10.0
POINT_FIELDS = 4             # x, y, z, intensity (as measured in run_pipeline.py;
                             # the paper's Eq. 3 writes 3)
B_FLOAT = 4                  # bytes per float32
BOX_FLOATS = 7               # x, y, z, l, w, h, yaw per detected box
# Compressed intermediate feature map per CAV, calibrated from the released
# eval.csv files (avg_comm_bytes_per_frame / number of transmitting CAVs).
INTERMEDIATE_BYTES_PER_CAV = 140_800

LOG_FIELDS = [
    "scenario_path", "frame_name", "bandwidth_mbps",
    "initial_cavs", "initial_fusion", "initial_rate_mbps",
    "n_feedback_rounds", "final_cavs", "final_fusion", "final_rate_mbps",
    "feasible", "fallback_used",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. Cost model (Eq. 3 per CAV, Eq. 2 summed, converted to Mbps)
# ════════════════════════════════════════════════════════════════════════════

def _pcd_point_count(pcd_path: str) -> int:
    """Number of points K_i from the PCD header (same as run_pipeline.py)."""
    try:
        with open(pcd_path, "rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("POINTS"):
                    return int(line.split()[1])
                if line.startswith("DATA"):
                    break
    except OSError:
        pass
    return 0


def _yaml_object_count(yaml_path: str) -> int:
    """Number of boxes a CAV would send in late fusion (same as run_pipeline.py)."""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return len((yaml.safe_load(f) or {}).get("vehicles", {}))
    except Exception:
        return 0


def cav_bytes(fusion: str, cav_id: str, scenario_path: str, frame: str) -> float:
    """delta(alpha, r_i): bytes one CAV transmits per frame (paper Eq. 3)."""
    base = os.path.join(scenario_path, cav_id, frame)
    if fusion == "early":
        return _pcd_point_count(base + ".pcd") * POINT_FIELDS * B_FLOAT
    if fusion == "intermediate":
        return INTERMEDIATE_BYTES_PER_CAV
    if fusion == "late":
        return _yaml_object_count(base + ".yaml") * BOX_FLOATS * B_FLOAT
    return math.inf


def required_rate_mbps(fusion: str, cavs: List[str], ego: str,
                       scenario_path: str, frame: str) -> float:
    """R(t): Eq. 2 summed over transmitting (non-ego) CAVs, in Mbps."""
    total = sum(cav_bytes(fusion, c, scenario_path, frame) for c in cavs if c != ego)
    return total * 8 * FRAME_RATE_HZ / 1e6


# ════════════════════════════════════════════════════════════════════════════
# 2. Feedback message
# ════════════════════════════════════════════════════════════════════════════

def feasible_options(cavs: List[str], ego: str, scenario_path: str,
                     frame: str, bandwidth: float) -> Dict[str, Dict]:
    """For each fusion strategy: rate of ONE CAV (largest among the
    candidates) and how many non-ego CAVs fit in the budget."""
    out = {}
    candidates = [c for c in cavs if c != ego]
    for fm in ("early", "intermediate", "late"):
        per_cav = [cav_bytes(fm, c, scenario_path, frame) * 8 * FRAME_RATE_HZ / 1e6
                   for c in candidates]
        worst = max(per_cav) if per_cav else 0.0
        max_n = len(candidates) if worst == 0 else min(len(candidates),
                                                       int(bandwidth // worst))
        out[fm] = {"per_cav_mbps": worst, "max_non_ego_cavs": max_n}
    return out


def build_feedback(raw_response: str, decision: dict, rate: float,
                   bandwidth: float, options: Dict[str, Dict]) -> str:
    fm = decision["fusion_method"]
    n_tx = options[fm]["max_non_ego_cavs"]
    if n_tx == 0:
        severity = (f"Even ONE non-ego CAV with {fm} fusion exceeds the budget, "
                    f"so removing CAVs alone cannot fix this: you MUST change "
                    f"the fusion strategy (you may also change the CAVs).")
    else:
        severity = (f"You can fix this by reducing the number of non-ego CAVs "
                    f"(at most {n_tx} with {fm} fusion), by changing the fusion "
                    f"strategy, or both. Keep the CAVs whose views matter most "
                    f"for this scene.")
    lines = [
        "",
        "=== BANDWIDTH FEEDBACK (your previous decision is NOT feasible) ===",
        f"Your previous answer was:\n{raw_response.strip()}",
        f"It requires {rate:.1f} Mbps, but only {bandwidth:.1f} Mbps is available "
        f"({rate / bandwidth:.1f}x the budget).",
        severity,
        "Measured cost per non-ego CAV at 10 Hz in this scene:",
    ]
    for f, o in options.items():
        lines.append(f"  - {f}: {o['per_cav_mbps']:.2f} Mbps per CAV "
                     f"-> at most {o['max_non_ego_cavs']} non-ego CAV(s) fit")
    lines.append("Reconsider your decision and output ONLY the corrected JSON:")
    lines.append('{"selected_cavs": [...], "fusion_method": "early"|"intermediate"|"late", '
                 '"reason": "..."}')
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. Wrapper around the original selector
# ════════════════════════════════════════════════════════════════════════════

class _FeedbackLLM:
    """Proxy that appends the feedback block to every prompt sent to the LLM."""

    def __init__(self, llm, feedback: str):
        self._llm, self._feedback = llm, feedback

    def generate(self, prompt: str) -> str:
        return self._llm.generate(prompt + "\n" + self._feedback)


class ValidatedSelector:
    """Drop-in replacement for RobotAndStrategySelector with the same .select()."""

    def __init__(self, base_selector, log_csv: Optional[str] = None,
                 max_retries: int = 1):
        self.base = base_selector
        self.max_retries = max_retries
        self.log_csv = log_csv
        if log_csv:
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    def select(self, scenario_data: dict, bandwidth_mbps: float = 50.0,
               prev_result: Optional[dict] = None, prev_ap: Optional[dict] = None):
        sp = scenario_data["scenario_path"]          # set in run_selection
        frame = scenario_data["frame_name"]
        ego = scenario_data["ego_cav"]
        in_range = [c for c, i in scenario_data["cavs"].items()
                    if i["dist_to_ego"] <= scenario_data["com_range"]]

        # Original CoAdapt decision (prompt, parse retry, C1-C3 unchanged)
        result, raw = self.base.select(scenario_data, bandwidth_mbps,
                                       prev_result, prev_ap)
        if result.get("fusion_method") == "unknown":
            return result, raw                       # nothing to validate

        def rate_of(r):
            return required_rate_mbps(r["fusion_method"], r["selected_cavs"],
                                      ego, sp, frame)

        log = {"scenario_path": sp, "frame_name": frame,
               "bandwidth_mbps": round(bandwidth_mbps, 2),
               "initial_cavs": "|".join(result["selected_cavs"]),
               "initial_fusion": result["fusion_method"],
               "initial_rate_mbps": round(rate_of(result), 2),
               "n_feedback_rounds": 0, "fallback_used": False}

        rate = rate_of(result)
        rounds = 0
        while rate > bandwidth_mbps and rounds < self.max_retries:
            rounds += 1
            options = feasible_options(in_range, ego, sp, frame, bandwidth_mbps)
            feedback = build_feedback(raw, result, rate, bandwidth_mbps, options)
            print(f"[validator] infeasible ({rate:.1f} > {bandwidth_mbps:.1f} Mbps)"
                  f" -> feedback round {rounds}")

            real_llm = self.base._llm
            self.base._llm = _FeedbackLLM(real_llm, feedback)
            try:
                new_result, new_raw = self.base.select(scenario_data, bandwidth_mbps,
                                                       prev_result, prev_ap)
            finally:
                self.base._llm = real_llm            # always restore the backend
            if new_result.get("fusion_method") == "unknown":
                break                                # keep previous, go to fallback
            result, raw = new_result, new_raw
            rate = rate_of(result)

        if rate > bandwidth_mbps:                    # deterministic fallback
            result = dict(result, fusion_method="late",
                          reason=result.get("reason", "") +
                          " [validator fallback: switched to late fusion]")
            rate = rate_of(result)
            log["fallback_used"] = True
            print(f"[validator] fallback -> late fusion ({rate:.2f} Mbps)")

        log.update({"n_feedback_rounds": rounds,
                    "final_cavs": "|".join(result["selected_cavs"]),
                    "final_fusion": result["fusion_method"],
                    "final_rate_mbps": round(rate, 2),
                    "feasible": rate <= bandwidth_mbps})
        if self.log_csv:
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(log)
        return result, raw