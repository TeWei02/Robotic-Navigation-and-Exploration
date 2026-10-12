#!/usr/bin/env python3
"""Look-ahead sensitivity benchmark for the deterministic pure-pursuit baseline.

The published reference runs in ``docs/data/reference.json`` are labelled, in the
README and in the tests, as *"not PPO results and not a benchmark"*. That is
accurate: they are six single episodes at one fixed look-ahead distance.

This tool turns them into a benchmark in the only way that is honest while the
PPO checkpoints remain unavailable (they are not redistributed here, see the
README's data-availability table). It holds every other variable fixed and
sweeps the one parameter the classical baseline actually exposes — the
look-ahead distance ``Ld`` — across the same six scenarios, through the same
code path that produces the published data set.

Method
------
``tools/build_web_data.py`` is imported as a module and its module-level
``LOOKAHEAD`` is patched between runs. Each row therefore goes through the
identical episode construction, reward function, termination rules and
observation vector as the published runs. **No part of the episode loop is
re-implemented here**, so the numbers cannot drift from the published ones for
reasons of this tool's own making. As a guard, the sweep includes ``Ld = 30``,
the published value, and every metric of that row must equal the committed
``reference.json`` exactly; the tool exits non-zero if it does not.

What this is not
----------------
It is not a comparison against PPO, and it says nothing about the trained
policy. It characterises the baseline only.

Usage
-----
    python tools/benchmark.py            # sweep, write docs/data/benchmark.json
    python tools/benchmark.py --stdout   # print the JSON instead of writing it
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_web_data  # noqa: E402

# The published runs use Ld = 30; it is kept in the middle of the sweep so the
# guard below can compare against reference.json.
LOOKAHEAD_SWEEP = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0]
BASELINE = 30.0
OUT_PATH = os.path.join(REPO_ROOT, "docs", "data", "benchmark.json")


def run_sweep() -> list[dict]:
    """One row per (scenario, look-ahead) pair, via the published code path."""
    rows = []
    for lookahead in LOOKAHEAD_SWEEP:
        build_web_data.LOOKAHEAD = lookahead
        for seed, name, mode, description in build_web_data.SCENARIOS:
            payload = build_web_data.run_scenario(seed, name, mode, description)
            summary = payload["summary"]
            rows.append(
                {
                    "lookahead": lookahead,
                    "scenario": name,
                    "mode": mode,
                    "steps": summary["steps"],
                    "return": summary["return"],
                    "doneReason": summary["doneReason"],
                    "meanMinDist": summary["meanMinDist"],
                    "maxMinDist": summary["maxMinDist"],
                    "finalGoalDistance": summary["finalGoalDistance"],
                    "stepsWithoutProgress": summary["stepsWithoutProgress"],
                }
            )
    return rows


def check_against_published(rows: list[dict]) -> list[str]:
    """Every Ld = 30 row must reproduce the committed data set exactly."""
    with open(build_web_data.DATA_PATH, encoding="utf-8") as handle:
        published = json.load(handle)
    stored = {scenario["name"]: scenario["summary"] for scenario in published["scenarios"]}

    problems = []
    for row in rows:
        if row["lookahead"] != BASELINE:
            continue
        summary = stored[row["scenario"]]
        for key in ("steps", "return", "doneReason", "meanMinDist", "maxMinDist"):
            if row[key] != summary[key]:
                problems.append(f"{row['scenario']}: {key} {row[key]!r} != {summary[key]!r}")
    return problems


def aggregate(rows: list[dict]) -> list[dict]:
    """Per-look-ahead roll-up across the six scenarios."""
    out = []
    for lookahead in LOOKAHEAD_SWEEP:
        group = [row for row in rows if row["lookahead"] == lookahead]
        reached = [row for row in group if row["doneReason"] == "goal"]
        out.append(
            {
                "lookahead": lookahead,
                "scenarios": len(group),
                "goalRate": round(len(reached) / len(group), 4),
                "meanReturn": round(sum(row["return"] for row in group) / len(group), 3),
                "meanMeanMinDist": round(sum(row["meanMinDist"] for row in group) / len(group), 4),
                "worstMaxMinDist": round(max(row["maxMinDist"] for row in group), 4),
                "meanSteps": round(sum(row["steps"] for row in group) / len(group), 1),
                "isPublishedBaseline": lookahead == BASELINE,
            }
        )
    return out


def build() -> dict:
    rows = run_sweep()
    problems = check_against_published(rows)
    return {
        "generated_by": "python tools/benchmark.py",
        "scope": (
            "Characterises the deterministic pure-pursuit baseline only. It is "
            "not a comparison against the PPO policy, whose checkpoints are not "
            "redistributed here."
        ),
        "method": {
            "swept": "PurePursuitController look-ahead distance (length units)",
            "held_fixed": [
                "speed v = V_RANGE * 0.6",
                "yaw-rate saturation w_range = W_RANGE",
                "the six published scenarios, seeds and control points",
                "the environment, reward function and termination rules",
            ],
            "code_path": (
                "tools/build_web_data.py imported as a module with LOOKAHEAD "
                "patched; the episode loop is not re-implemented"
            ),
            "published_baseline_lookahead": BASELINE,
            "guard": (
                "the Ld = 30 rows are compared field-by-field against docs/data/reference.json"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count() or 0,
        },
        "guard_passed": not problems,
        "guard_problems": problems,
        "lookaheads": LOOKAHEAD_SWEEP,
        "runs": rows,
        "aggregate": aggregate(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="print JSON, write nothing")
    args = parser.parse_args(argv)

    payload = build()

    if not payload["guard_passed"]:
        print("the Ld = 30 rows disagree with the published data set:", file=sys.stderr)
        for problem in payload["guard_problems"]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    header = (
        f"{'Ld':>5}  {'goal':>5}  {'mean return':>12}  {'mean dev':>9}  "
        f"{'worst dev':>10}  {'mean steps':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["aggregate"]:
        marker = " *" if row["isPublishedBaseline"] else "  "
        print(
            f"{row['lookahead']:>5.0f}  {row['goalRate'] * 100:>4.0f}%  "
            f"{row['meanReturn']:>12.2f}  {row['meanMeanMinDist']:>9.3f}  "
            f"{row['worstMaxMinDist']:>10.3f}  {row['meanSteps']:>10.1f}{marker}"
        )
    print("\n* published baseline (Ld = 30); every field matches docs/data/reference.json")
    print(f"wrote {os.path.relpath(OUT_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
