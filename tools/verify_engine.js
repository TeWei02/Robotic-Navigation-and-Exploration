/*!
 * verify_engine.js — cross-language parity harness for the published demo.
 *
 * The published page runs the HW3-1 path-tracking environment in JavaScript
 * (docs/engine.js). This script checks that port against the Python-generated
 * reference data set (docs/data/reference.json) without a browser, so the claim
 * "the page reproduces the repository" can be verified from a terminal.
 *
 * Run with either runtime (JavaScriptCore resolves paths against the working
 * directory, so start it from the repository root; Node.js resolves them
 * against this file):
 *
 *     jsc tools/verify_engine.js      # JavaScriptCore shell (ships with macOS)
 *     node tools/verify_engine.js     # Node.js
 *
 * For every scenario it re-runs the recorded control sequence through the
 * browser dynamics (mode "replay") and also recomputes the controls in
 * JavaScript (mode "closed-loop"), then compares both results with the stored
 * episode. Disagreements are printed as FAIL lines; the last line is either
 * RESULT: PASS or RESULT: FAIL <n>.
 *
 * Tolerances: the stored numbers are rounded to six decimals and written by a
 * Python/NumPy run, so the comparison allows for that rounding plus
 * double-precision drift between the two languages. The residual is printed, not
 * hidden, and the tolerances below are the only thing that turns it into a
 * pass/fail verdict.
 *
 * License: MIT (see LICENSE in the repository root).
 */
"use strict";

var isNode = typeof process !== "undefined" && process.versions && process.versions.node;
var readText, report;

if (isNode) {
  /* Node resolves both the data set and the engine against this file, so the
   * harness can be started from any working directory. */
  var fs = require("fs");
  var path = require("path");
  var repoRoot = path.join(__dirname, "..");
  readText = function (rel) {
    return fs.readFileSync(path.join(repoRoot, rel), "utf8");
  };
  report = function (line) {
    console.log(line);
  };
  require(path.join(repoRoot, "docs", "engine.js"));
} else {
  load("docs/engine.js");
  readText = readFile;
  report = print;
}

var E = globalThis.RNE;
if (!E) {
  report("FAIL could not load docs/engine.js (run this script from the repository root)");
  report("RESULT: FAIL 1");
  throw new Error("engine unavailable");
}

var ref = JSON.parse(readText("docs/data/reference.json"));

var TOL_STATE = 1e-5; /* positions, yaw, observation, commands: 6-decimal storage */
var TOL_REWARD = 1e-4; /* per-step reward and episode return */
var TOL_DIST = 1e-3; /* squared/eucldiean distances, larger magnitudes */
var failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
  }
}

function compareSeries(actual, stored, tolerance, label) {
  if (actual.length !== stored.length) {
    check(false, label + ": length " + actual.length + " != " + stored.length);
    return;
  }
  var worst = 0;
  var worstAt = -1;
  for (var i = 0; i < stored.length; i++) {
    var a = actual[i];
    var b = stored[i];
    if (Array.isArray(b)) {
      for (var k = 0; k < b.length; k++) {
        var dv = Math.abs(a[k] - b[k]);
        if (dv > worst) {
          worst = dv;
          worstAt = i;
        }
      }
    } else {
      var d = Math.abs(a - b);
      if (d > worst) {
        worst = d;
        worstAt = i;
      }
    }
  }
  check(
    worst <= tolerance,
    label +
      ": max |delta| " +
      worst.toExponential(3) +
      " at index " +
      worstAt +
      " exceeds " +
      tolerance.toExponential(1)
  );
  return worst;
}

function compareExact(actual, stored, label) {
  if (actual.length !== stored.length) {
    check(false, label + ": length " + actual.length + " != " + stored.length);
    return;
  }
  for (var i = 0; i < stored.length; i++) {
    if (actual[i] !== stored[i]) {
      check(false, label + ": index " + i + " " + actual[i] + " != " + stored[i]);
      return;
    }
  }
}

/* ---- 1. environment constants ------------------------------------------ */
var meta = ref.meta;
check(Math.abs(E.config.dt - meta.dt) < 1e-12, "dt mismatch: " + E.config.dt + " != " + meta.dt);
check(E.config.vRange === meta.vRange, "vRange mismatch");
check(E.config.wRange === meta.wRange, "wRange mismatch");
check(E.config.maxSteps === meta.maxSteps, "maxSteps mismatch");
check(E.config.goalTolerance === 10.0, "goal tolerance mismatch");
check(E.config.lookahead === meta.lookahead, "look-ahead mismatch");

/* The published data set states where its numbers come from; the harness that
 * reads it should not silently accept a data set that dropped that note. */
check(
  typeof meta.note === "string" && meta.note.indexOf("not distributed") !== -1,
  "reference data set is missing its provenance note"
);

/* ---- 2. per-scenario replay and closed-loop parity --------------------- */
var worst = {
  position: 0,
  reward: 0,
  observation: 0,
  distance: 0,
  control: 0
};

ref.scenarios.forEach(function (scenario) {
  var name = scenario.name;
  var summary = scenario.summary;

  var replay = E.runEpisode(scenario, { mode: "replay" });

  check(
    replay.steps === summary.steps,
    name + ": replay steps " + replay.steps + " != " + summary.steps
  );
  compareExact(replay.commands, scenario.commands, name + " replay commands");
  var wTraj = compareSeries(replay.trajectory, scenario.trajectory, TOL_STATE, name + " replay trajectory");
  var wReward = compareSeries(replay.rewards, scenario.rewards, TOL_REWARD, name + " replay rewards");
  var wObs = compareSeries(replay.observation, scenario.observation, TOL_STATE, name + " replay observation");
  /* Per-step diagnostics that the page prints next to the trajectory. */
  compareExact(replay.minIdx, scenario.minIdx, name + " replay nearest-sample index");
  var wDistSq = compareSeries(replay.minDistSq, scenario.minDistSq, TOL_DIST, name + " replay squared distance");
  var wDist = compareSeries(replay.minDist, scenario.minDist, TOL_DIST, name + " replay distance to path");
  var wYaw = compareSeries(replay.errorYaw, scenario.errorYaw, TOL_STATE, name + " replay heading error");
  compareSeries(replay.progress, scenario.progress, 0, name + " replay progress term");
  check(
    replay.doneReason === summary.doneReason,
    name + ": replay termination " + replay.doneReason + " != " + summary.doneReason
  );
  check(
    Math.abs(replay.totalReward - summary["return"]) <= TOL_REWARD,
    name +
      ": return " +
      replay.totalReward.toFixed(6) +
      " != " +
      summary["return"].toFixed(6)
  );

  /* Closed loop: the browser recomputes the controls instead of replaying them.
   * The controller is deterministic, so it should land on the recorded episode
   * within the same drift budget. */
  var loop = E.runEpisode(scenario, { mode: "closed-loop" });
  check(
    loop.steps === summary.steps,
    name + ": closed-loop steps " + loop.steps + " != " + summary.steps
  );
  var wLoopCmd = compareSeries(loop.commands, scenario.commands, TOL_STATE, name + " closed-loop commands");
  var wLoopTraj = compareSeries(
    loop.trajectory,
    scenario.trajectory,
    TOL_STATE,
    name + " closed-loop trajectory"
  );
  check(
    loop.doneReason === summary.doneReason,
    name + ": closed-loop termination " + loop.doneReason + " != " + summary.doneReason
  );

  worst.position = Math.max(worst.position, wTraj || 0, wLoopTraj || 0);
  worst.reward = Math.max(worst.reward, wReward || 0);
  worst.observation = Math.max(worst.observation, wObs || 0);
  worst.distance = Math.max(worst.distance, wDistSq || 0, wDist || 0, wYaw || 0);
  worst.control = Math.max(worst.control, wLoopCmd || 0);
});

/* ---- 3. report --------------------------------------------------------- */
failures.forEach(function (message) {
  report("FAIL " + message);
});
report("scenarios checked: " + ref.scenarios.length);
report("worst position deviation (stored vs browser): " + worst.position.toExponential(3));
report("worst reward deviation: " + worst.reward.toExponential(3));
report("worst observation deviation: " + worst.observation.toExponential(3));
report("worst distance / heading deviation: " + worst.distance.toExponential(3));
report("worst control deviation (closed loop vs recorded): " + worst.control.toExponential(3));
report(failures.length === 0 ? "RESULT: PASS" : "RESULT: FAIL " + failures.length);

if (typeof quit === "function") {
  quit(failures.length === 0 ? 0 : 1);
}
