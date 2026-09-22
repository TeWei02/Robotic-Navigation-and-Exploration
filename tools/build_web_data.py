#!/usr/bin/env python3
"""Headless reference generator for the browser path-tracking demo.

The published demo (``docs/``) re-implements the deterministic part of the
``HW3-1`` training environment in JavaScript so that visitors can run episodes
in the browser without installing Python.  This tool produces the reference
data set that the browser implementation is checked against, and it uses the
repository's own classes for every produced number:

* ``wrapper.PathTrackingEnv`` — episode construction, reward function,
  termination rules and the 14-dimensional observation vector;
* ``cubic_spline.cubic_spline_2d`` — path generation (control points are
  captured through a recording wrapper so the browser can rebuild the spline);
* ``PathTracking.pure_pursuit.PurePursuitController`` — deterministic
  geometry-based baseline that supplies the control sequence.

Two modes::

    python tools/build_web_data.py            # regenerate docs/data/reference.json
    python tools/build_web_data.py --check    # verify the committed data set

``--check`` re-runs the generator and compares every stored field against the
committed file, printing the largest absolute deviation per field.  It exits
with a non-zero status when a field exceeds its tolerance, which makes the data
set verifiable both locally and in CI.

Rendering is never used; ``cv2`` is only needed because the simulator modules
import it at module level.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW31 = os.path.join(REPO_ROOT, "HW3-1")
DATA_PATH = os.path.join(REPO_ROOT, "docs", "data", "reference.json")
sys.path.insert(0, HW31)

import wrapper  # noqa: E402
from PathTracking.pure_pursuit import PurePursuitController  # noqa: E402

# Scenario table: seed -> (name, mode, description).
# ``pure``     : pure-pursuit baseline, look-ahead 30 length units.
# ``perturb``  : pure pursuit plus a deterministic sinusoidal steering bias, which
#                drives the vehicle off the path and back on repeatedly.
# ``recovery`` : a saturated turn for the first steps, which carries the vehicle
#                backwards along the path before the baseline recovers; this run
#                covers the negative branch of the progress reward.
SCENARIOS = [
    (1, "corridor-turn", "pure", "Left-bending spline, random start pose; nominal tracking run."),
    (2, "wide-sweep", "pure", "Mirrored control points produce a long right-hand sweep."),
    (3, "tight-arc", "pure", "Short control-point spacing; the largest curvature of the set."),
    (
        4,
        "offset-entry",
        "pure",
        "Start pose is offset from the path, so the controller must converge.",
    ),
    (
        5,
        "biased-driver",
        "perturb",
        "Steering bias of +-25 deg/s injected, so the run includes off-path segments.",
    ),
    (
        6,
        "recovery",
        "recovery",
        "Saturated turn for 30 steps, then baseline tracking; includes regressing steps.",
    ),
]

LOOKAHEAD = 30.0
V_RANGE = 20.0
W_RANGE = 60.0
DT = 0.1
BIAS_AMPLITUDE = 25.0
BIAS_FREQUENCY = 0.05
RECOVERY_STEPS = 30

# Tolerances used by --check.  Two error sources are present: the reference path
# is solved in float32 (``cubic_spline`` allocates its tridiagonal system as
# float32 and inverts it with ``np.linalg.pinv``) while the browser solves the
# same system in float64, and every stored number is rounded to reduce the size
# of the published data set.  The measured deviations are reported by --check;
# the limits below carry a safety margin over them.
TOL_PATH = 1e-2
TOL_TRAJ = 1e-6
TOL_REWARD = 1e-3
TOL_DIST_SQ = 1e-1
TOL_OBS = 1e-4


def capture_control_points():
    """Patch ``cubic_spline_2d`` so the raw control points survive the call."""
    recorded = {}
    original = wrapper.cubic_spline.cubic_spline_2d

    def recording(path, interval=2):
        recorded["points"] = [[float(p[0]), float(p[1])] for p in path]
        return original(path, interval=interval)

    wrapper.cubic_spline.cubic_spline_2d = recording
    return recorded, original


def run_scenario(seed, name, mode, description):
    """Run one deterministic episode and return the JSON payload for it."""
    recorded, original = capture_control_points()
    try:
        np.random.seed(seed)
        env = wrapper.PathTrackingEnv()
    finally:
        wrapper.cubic_spline.cubic_spline_2d = original

    control_points = recorded["points"]
    path = env.path
    controller = PurePursuitController(lookahead=LOOKAHEAD, v=V_RANGE * 0.6, w_range=W_RANGE)
    controller.set_path(path)

    state = env.simulator.state
    start = [float(state.x), float(state.y), float(state.yaw)]

    commands = []
    actions = []
    trajectory = []
    rewards = []
    min_idx = []
    min_dist_sq = []
    min_dist = []
    error_yaw = []
    progress = []
    obs = []

    step_index = 0
    while True:
        pose = env.simulator.state.pose()
        _, w = controller.command(pose)
        if mode == "perturb":
            w += BIAS_AMPLITUDE * np.sin(BIAS_FREQUENCY * step_index)
            w = float(np.clip(w, -W_RANGE, W_RANGE))
        elif mode == "recovery" and step_index < RECOVERY_STEPS:
            w = W_RANGE
        action = w / W_RANGE

        state_vec, reward, done, info = env.step(np.array([action]))
        state = env.simulator.state

        commands.append(round(float(w), 6))
        actions.append(round(float(action), 9))
        trajectory.append(
            [
                round(float(state.x), 6),
                round(float(state.y), 6),
                round(float(state.yaw), 6),
                round(float(state.v), 6),
            ]
        )
        rewards.append(round(float(reward), 9))
        min_idx.append(int(info["min_idx"]))
        position = (state.x, state.y)
        # ``search_nearest`` returns the *squared* distance, which is the quantity
        # the reward function consumes; the Euclidean distance is derived for display.
        _, dist_sq = wrapper.PathTracking.utils.search_nearest(path, position)
        min_dist_sq.append(round(float(dist_sq), 9))
        min_dist.append(round(float(np.sqrt(dist_sq)), 6))
        target = path[info["min_idx"]]
        yaw_err = (target[2] - state.yaw) % 360
        if yaw_err > 180:
            yaw_err = 360 - yaw_err
        error_yaw.append(round(float(yaw_err), 6))
        # Mirrors the progress term of PathTrackingEnv.step: +0.1 for advancing one
        # sample, 0 for staying, -1.0 for regressing.
        diff = 0 if len(min_idx) == 1 else min_idx[-1] - min_idx[-2]
        progress.append(0.1 if diff > 0 else (0.0 if diff == 0 else -1.0))
        obs.append([round(float(x), 6) for x in state_vec])
        step_index += 1
        if done:
            break

    goal = path[-1]
    goal_distance = float(np.hypot(state.x - goal[0], state.y - goal[1]))
    reached = int(min_idx[-1]) == len(path) - 1 or goal_distance < 10.0
    reason = "goal" if reached else "max_step"

    return {
        "id": f"scenario-{seed}",
        "name": name,
        "description": description,
        "seed": seed,
        "mode": mode,
        "controller": {
            "pure": "pure-pursuit (look-ahead 30)",
            "perturb": "pure-pursuit + sinusoidal steering bias (+-25 deg/s)",
            "recovery": "saturated turn (60 deg/s) for 30 steps, then pure-pursuit",
        }[mode],
        "controlPoints": control_points,
        "path": [[round(float(v), 6) for v in sample] for sample in path],
        "start": start,
        "commands": commands,
        "actions": actions,
        "trajectory": trajectory,
        "rewards": rewards,
        "minIdx": min_idx,
        "minDistSq": min_dist_sq,
        "minDist": min_dist,
        "errorYaw": error_yaw,
        "progress": progress,
        "observation": obs,
        "summary": {
            "steps": len(commands),
            "return": round(float(np.sum(rewards)), 6),
            "pathSamples": int(path.shape[0]),
            "finalGoalDistance": round(goal_distance, 6),
            "doneReason": reason,
            "maxMinDist": round(float(np.max(min_dist)), 6),
            "meanMinDist": round(float(np.mean(min_dist)), 6),
            "stepsWithoutProgress": int(sum(1 for p in progress if p < 0)),
        },
    }


def build():
    scenarios = [run_scenario(seed, name, mode, desc) for seed, name, mode, desc in SCENARIOS]
    return {
        "meta": {
            "generator": "tools/build_web_data.py",
            "source": (
                "HW3-1 (wrapper.PathTrackingEnv, cubic_spline.cubic_spline_2d, "
                "PathTracking.pure_pursuit)"
            ),
            "simulatorType": "basic",
            "dt": DT,
            "vRange": V_RANGE,
            "wRange": W_RANGE,
            "maxSteps": 400,
            "initRange": 20,
            "lookahead": LOOKAHEAD,
            "units": {
                "position": "simulator length unit (canvas spans 0-600)",
                "velocity": "length unit per second",
                "yaw": "degree",
                "yawRate": "degree per second",
            },
            "note": (
                "Control sequences were produced by the deterministic pure-pursuit baseline, "
                "not by a trained PPO policy: the trained checkpoint (save/model.pt) is not "
                "distributed with this repository."
            ),
        },
        "scenarios": scenarios,
    }


def compare(field, stored, regenerated, tol):
    """Return the largest absolute deviation between two nested lists."""
    flat_a = np.asarray(stored, dtype=float).ravel()
    flat_b = np.asarray(regenerated, dtype=float).ravel()
    if flat_a.shape != flat_b.shape:
        raise SystemExit(f"field {field}: shape mismatch {flat_a.shape} vs {flat_b.shape}")
    diff = float(np.max(np.abs(flat_a - flat_b))) if flat_a.size else 0.0
    status = "ok" if diff <= tol else "FAIL"
    print(f"  {field:<14} max|diff| = {diff:.3e}  tol = {tol:.1e}  {status}")
    return status == "ok"


def check():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"missing {DATA_PATH}; run without --check first")
    with open(DATA_PATH, encoding="utf-8") as handle:
        stored = json.load(handle)
    regenerated = build()

    ok = True
    if len(stored["scenarios"]) != len(regenerated["scenarios"]):
        raise SystemExit("scenario count mismatch")
    for old, new in zip(stored["scenarios"], regenerated["scenarios"], strict=False):
        print(f"{old['id']} ({old['name']})")
        if old["controlPoints"] != new["controlPoints"]:
            print("  controlPoints  FAIL (recorded start points differ)")
            ok = False
        else:
            print("  controlPoints  ok (exact match)")
        ok &= compare("path", old["path"], new["path"], TOL_PATH)
        ok &= compare("trajectory", old["trajectory"], new["trajectory"], TOL_TRAJ)
        ok &= compare("commands", old["commands"], new["commands"], TOL_TRAJ)
        ok &= compare("rewards", old["rewards"], new["rewards"], TOL_REWARD)
        ok &= compare("minDistSq", old["minDistSq"], new["minDistSq"], TOL_DIST_SQ)
        ok &= compare("errorYaw", old["errorYaw"], new["errorYaw"], TOL_PATH)
        ok &= compare("observation", old["observation"], new["observation"], TOL_OBS)
        if old["minIdx"] != new["minIdx"] or old["progress"] != new["progress"]:
            print("  minIdx/progress FAIL (discrete fields differ)")
            ok = False
        else:
            print("  minIdx/progress ok (exact match)")
        if old["summary"] != new["summary"]:
            print(f"  summary  FAIL: {old['summary']} vs {new['summary']}")
            ok = False
        else:
            summary = old["summary"]
            print(
                f"  summary  ok ({summary['doneReason']}, {summary['steps']} steps, "
                f"return {summary['return']})"
            )
    print("\nreference data set:", "verified" if ok else "MISMATCH")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the committed data set instead of writing it"
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check())

    payload = build()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=False)
        handle.write("\n")
    size = os.path.getsize(DATA_PATH)
    print(f"wrote {DATA_PATH} ({size / 1024.0:.1f} KiB, {len(payload['scenarios'])} scenarios)")
    for scenario in payload["scenarios"]:
        summary = scenario["summary"]
        name = scenario["name"]
        print(
            f"  {name:<9} steps={summary['steps']:>3d} return={summary['return']:>8.2f} "
            f"maxDist={summary['maxMinDist']:>6.2f} meanDist={summary['meanMinDist']:>5.2f} "
            f"noProgress={summary['stepsWithoutProgress']:>2d} {summary['doneReason']}"
        )


if __name__ == "__main__":
    main()
