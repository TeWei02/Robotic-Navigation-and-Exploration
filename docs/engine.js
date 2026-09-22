/*!
 * engine.js — browser port of the HW3-1 path-tracking environment.
 *
 * This file re-implements, in plain JavaScript, the pieces of the repository
 * that are needed to run a path-tracking episode without Python:
 *
 *   HW3-1/cubic_spline.py        -> ensureUniqueCoordinates, cubicSpline1d,
 *                                   cubicSpline2d
 *   HW3-1/wrapper.py             -> PathTrackingEnv: gen_path, reset, step,
 *                                   reward, done, observation
 *   HW3-1/Simulation/            -> SimulatorBasic.step, KinematicModelBasic.step
 *   HW3-1/PathTracking/pure_pursuit.py -> PurePursuitController.command
 *
 * The port is deliberately literal: every formula below (including the sign
 * conventions, the modulo arithmetic on the heading, the nearest-point search
 * and the discrete progress reward) follows the Python source line by line, so
 * that the same episode can be reproduced in a browser and compared against the
 * recorded reference data set in docs/data/reference.json.
 *
 * Differences from the Python implementation, all deliberate:
 *   - NumPy's float32 tridiagonal system is solved here in float64 (see
 *     solveLinearSystem).  The resulting path samples differ by a small amount,
 *     which is measured and reported by the parity page.
 *   - NumPy uses the Mersenne Twister; the "random path" button uses the
 *     browser's Math.random.  Random paths are therefore not reproducible
 *     across implementations, which is why all published reference runs use the
 *     fixed control points stored in the data set.
 *
 * License: MIT (see LICENSE in the repository root).
 */
(function (global) {
  "use strict";

  var ENGINE_VERSION = "1.0.0";

  /* Configuration equivalent to PathTrackingEnv.__init__ + the constants used by
   * the headless reference driver (PathTracking/main.py):
   *   v_range = 20, w_range = 60          -> SimulatorBasic defaults used by env
   *   command speed = v_range * 0.6       -> wrapper.PathTrackingEnv.step
   *   pure-pursuit look-ahead = 30 units  -> reference runs in docs/data
   */
  var CFG = {
    dt: 0.1,
    vRange: 20.0,
    wRange: 60.0,
    controlSpeed: 12.0,
    maxSteps: 400,
    initRange: 20,
    goalTolerance: 10.0,
    obsInterval: 16,
    canvas: 600,
    pathInterval: 3,
    lookahead: 30.0
  };

  /* ------------------------------------------------------------------ *
   * Small numeric helpers
   * ------------------------------------------------------------------ */

  function mod360(a) {
    /* Python's % on floats returns a result with the sign of the divisor. */
    return ((a % 360) + 360) % 360;
  }

  function clamp(value, low, high) {
    if (value > high) return high;
    if (value < low) return low;
    return value;
  }

  function deg2rad(d) {
    return (d * Math.PI) / 180.0;
  }

  function rad2deg(r) {
    return (r * 180.0) / Math.PI;
  }

  /* ------------------------------------------------------------------ *
   * cubic_spline.py
   * ------------------------------------------------------------------ */

  /* Mirrors ensure_unique_coordinates: shift a duplicate x or y by +1 until it
   * is unique.  The numpy implementation works in place on the caller's list,
   * so the caller side effect (control points are modified) is preserved. */
  function ensureUniqueCoordinates(points) {
    var xs = {}, ys = {};
    for (var i = 0; i < points.length; i++) {
      while (xs[points[i][0]]) {
        points[i][0] += 1;
      }
      while (ys[points[i][1]]) {
        points[i][1] += 1;
      }
      xs[points[i][0]] = true;
      ys[points[i][1]] = true;
    }
    return points;
  }

  /* Dense linear solve with partial pivoting, used in place of
   * np.linalg.pinv(A).dot(B).  The coefficient matrix built by cubic_spline is
   * non-singular (the first and last rows are identity rows), so the solve
   * agrees with the pseudo-inverse solution. */
  function solveLinearSystem(A, B) {
    var n = A.length;
    var i, j, k;
    var M = [];
    for (i = 0; i < n; i++) {
      M.push(A[i].slice(0));
      M[i].push(B[i]);
    }
    for (k = 0; k < n; k++) {
      var pivot = k;
      var best = Math.abs(M[k][k]);
      for (i = k + 1; i < n; i++) {
        if (Math.abs(M[i][k]) > best) {
          best = Math.abs(M[i][k]);
          pivot = i;
        }
      }
      if (best === 0) continue;
      if (pivot !== k) {
        var tmp = M[k];
        M[k] = M[pivot];
        M[pivot] = tmp;
      }
      for (i = k + 1; i < n; i++) {
        var factor = M[i][k] / M[k][k];
        if (factor === 0) continue;
        for (j = k; j <= n; j++) {
          M[i][j] -= factor * M[k][j];
        }
      }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) {
      var sum = M[i][n];
      for (j = i + 1; j < n; j++) {
        sum -= M[i][j] * x[j];
      }
      x[i] = M[i][i] === 0 ? 0 : sum / M[i][i];
    }
    return x;
  }

  /* Natural cubic spline through (param, values), sampled at
   * param[0], param[0]+interval, ... param[n-1].  Faithful port of
   * cubic_spline.cubic_spline, including the index walk of the sampling loop. */
  function cubicSpline1d(param, values, interval) {
    var size = values.length;
    var h = [];
    var i;
    for (i = 0; i < param.length - 1; i++) {
      h.push(param[i + 1] - param[i]);
    }

    var A = [];
    for (i = 0; i < size; i++) {
      A.push(new Array(size).fill(0));
    }
    for (i = 0; i < size; i++) {
      if (i === 0) {
        A[i][0] = 1;
      } else if (i === size - 1) {
        A[i][size - 1] = 1;
      } else {
        A[i][i - 1] = h[i - 1];
        A[i][i] = 2 * (h[i - 1] + h[i]);
        A[i][i + 1] = h[i];
      }
    }

    var B = new Array(size).fill(0);
    for (i = 1; i < size - 1; i++) {
      B[i] = (values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1];
    }
    for (i = 0; i < size; i++) {
      B[i] *= 6;
    }

    var m = solveLinearSystem(A, B);

    var a = [];
    var b = [];
    var c = [];
    var d = [];
    for (i = 0; i < size - 1; i++) {
      a.push(values[i]);
      b.push((values[i + 1] - values[i]) / h[i] - (h[i] * m[i]) / 2 - (h[i] * (m[i + 1] - m[i])) / 6);
      c.push(m[i] / 2);
      d.push((m[i + 1] - m[i]) / (6 * h[i]));
    }

    var out = [];
    var idx = 0;
    var s = param[0];
    var last = param[param.length - 1];
    /* The loop below reproduces the Python while-loop, which advances the
     * segment index by at most one position per iteration.  With the control
     * point spacing used in this project (>= 70 length units) and interval 3,
     * the index never lags behind. */
    for (;;) {
      if (s >= last) {
        s = last;
      } else if (idx + 1 < param.length && s > param[idx + 1]) {
        idx += 1;
      }
      var seg = Math.min(idx, size - 2);
      var ds = s - param[seg];
      out.push(a[seg] + b[seg] * ds + c[seg] * ds * ds + d[seg] * ds * ds * ds);
      if (s === last) break;
      s += interval;
    }
    return out;
  }

  /* Port of cubic_spline.cubic_spline_2d.  Returns [[x, y, yaw, curvature], ...]
   * with yaw in degrees, sampled every `interval` units of the cumulative
   * polyline length. */
  function cubicSpline2d(points, interval) {
    interval = interval === undefined ? 3 : interval;
    var path = ensureUniqueCoordinates(points.map(function (p) {
      return [p[0], p[1]];
    }));

    var x = [];
    var y = [];
    var i;
    for (i = 0; i < path.length; i++) {
      x.push(path[i][0]);
      y.push(path[i][1]);
    }

    var dist = [];
    for (i = 0; i < path.length - 1; i++) {
      dist.push(Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]));
    }
    var distCum = [0];
    for (i = 0; i < dist.length; i++) {
      distCum.push(distCum[distCum.length - 1] + dist[i]);
    }

    var xs = cubicSplineWithDerivatives(distCum, x, interval);
    var ys = cubicSplineWithDerivatives(distCum, y, interval);

    var out = [];
    for (i = 0; i < xs[0].length; i++) {
      var dx = xs[1][i];
      var dy = ys[1][i];
      var ddx = xs[2][i];
      var ddy = ys[2][i];
      var yaw = rad2deg(Math.atan2(dy, dx));
      var denom = Math.pow(dx * dx + dy * dy, 1.5);
      out.push([xs[0][i], ys[0][i], yaw, denom === 0 ? 0 : (ddy * dx - ddx * dy) / denom]);
    }
    return out;
  }

  /* Same sampling loop as cubicSpline1d, but returning value / first derivative
   * / second derivative per sample, as the Python code does internally. */
  function cubicSplineWithDerivatives(param, values, interval) {
    var size = values.length;
    var h = [];
    var i;
    for (i = 0; i < param.length - 1; i++) {
      h.push(param[i + 1] - param[i]);
    }
    var A = [];
    for (i = 0; i < size; i++) {
      A.push(new Array(size).fill(0));
    }
    for (i = 0; i < size; i++) {
      if (i === 0) {
        A[i][0] = 1;
      } else if (i === size - 1) {
        A[i][size - 1] = 1;
      } else {
        A[i][i - 1] = h[i - 1];
        A[i][i] = 2 * (h[i - 1] + h[i]);
        A[i][i + 1] = h[i];
      }
    }
    var B = new Array(size).fill(0);
    for (i = 1; i < size - 1; i++) {
      B[i] = (values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1];
    }
    for (i = 0; i < size; i++) {
      B[i] *= 6;
    }
    var m = solveLinearSystem(A, B);
    var a = [];
    var b = [];
    var c = [];
    var d = [];
    for (i = 0; i < size - 1; i++) {
      a.push(values[i]);
      b.push((values[i + 1] - values[i]) / h[i] - (h[i] * m[i]) / 2 - (h[i] * (m[i + 1] - m[i])) / 6);
      c.push(m[i] / 2);
      d.push((m[i + 1] - m[i]) / (6 * h[i]));
    }
    var vList = [];
    var dList = [];
    var ddList = [];
    var idx = 0;
    var s = param[0];
    var last = param[param.length - 1];
    for (;;) {
      if (s >= last) {
        s = last;
      } else if (idx + 1 < param.length && s > param[idx + 1]) {
        idx += 1;
      }
      var seg = Math.min(idx, size - 2);
      var ds = s - param[seg];
      vList.push(a[seg] + b[seg] * ds + c[seg] * ds * ds + d[seg] * ds * ds * ds);
      dList.push(b[seg] + 2 * c[seg] * ds + 3 * d[seg] * ds * ds);
      ddList.push(2 * c[seg] + 6 * d[seg] * ds);
      if (s === last) break;
      s += interval;
    }
    return [vList, dList, ddList];
  }

  /* ------------------------------------------------------------------ *
   * wrapper.py : PathTrackingEnv
   * ------------------------------------------------------------------ */

  /* search_nearest / search_future_nearest: squared Euclidean distance, first
   * index wins on ties. */
  function searchNearest(path, position) {
    var idx = 0;
    var dist = Math.pow(path[0][0] - position[0], 2) + Math.pow(path[0][1] - position[1], 2);
    for (var i = 1; i < path.length; i++) {
      var d = Math.pow(path[i][0] - position[0], 2) + Math.pow(path[i][1] - position[1], 2);
      if (d < dist) {
        dist = d;
        idx = i;
      }
    }
    return [idx, dist];
  }

  function searchFutureNearest(path, position, future_idx) {
    var idx = future_idx;
    var dist = Math.pow(path[future_idx][0] - position[0], 2) + Math.pow(path[future_idx][1] - position[1], 2);
    for (var i = future_idx; i < path.length; i++) {
      var d = Math.pow(path[i][0] - position[0], 2) + Math.pow(path[i][1] - position[1], 2);
      if (d < dist) {
        dist = d;
        idx = i;
      }
    }
    return [idx, dist];
  }

  /* Cumulative arc length of the sampled path (PurePursuitController.set_path). */
  function cumulativeArcLength(path) {
    var s = [0];
    for (var i = 0; i < path.length - 1; i++) {
      s.push(s[s.length - 1] + Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]));
    }
    return s;
  }

  /* get_record_path: xy in canvas units, yaw in radians; normalised by 600. */
  function recordPathSample(record, idx) {
    var n = record.length;
    var i = idx < 0 ? n + idx : idx;
    if (i < 0) i = 0;
    var sample = record[i];
    return [sample[0] / CFG.canvas, sample[1] / CFG.canvas, deg2rad(sample[2])];
  }

  /* get_future_path: eight numbers, four waypoints spaced obsInterval samples
   * apart from the nearest path index, clamped at the end of the path. */
  function futurePathSample(path, idx) {
    var last = path.length - 1;
    var out = [];
    for (var i = 0; i < 4; i++) {
      var j = idx + i * CFG.obsInterval;
      if (j > last) j = last;
      out.push(path[j][0] / CFG.canvas, path[j][1] / CFG.canvas);
    }
    return out;
  }

  function buildObservation(path, record, minIdx) {
    return recordPathSample(record, -2)
      .concat(recordPathSample(record, -1))
      .concat(futurePathSample(path, minIdx));
  }

  /* wrapper.PathTrackingEnv.reward */
  function computeReward(path, lastIdx, state) {
    var nearest = searchNearest(path, [state.x, state.y]);
    var minIdx = nearest[0];
    var minDistSq = nearest[1];

    var target = path[minIdx];
    var errorYaw = mod360(target[2] - state.yaw);

    var idxDiff = minIdx - lastIdx;
    var progress;
    if (idxDiff > 0) {
      progress = 0.1;
    } else if (idxDiff === 0) {
      progress = 0.0;
    } else {
      progress = -1.0;
    }

    var reward =
      0.8 * Math.exp(-0.1 * minDistSq) + 0.2 * Math.exp(-0.1 * errorYaw * errorYaw) + progress;

    var goal = [path[path.length - 1][0], path[path.length - 1][1]];
    var goalDist = Math.hypot(state.x - goal[0], state.y - goal[1]);
    var done = minIdx === path.length - 1 || goalDist < CFG.goalTolerance;
    var reason = "running";
    if (minIdx === path.length - 1) reason = "goal";
    else if (goalDist < CFG.goalTolerance) reason = "goal-tolerance";

    return {
      minIdx: minIdx,
      minDistSq: minDistSq,
      errorYaw: errorYaw,
      progress: progress,
      reward: reward,
      goalDist: goalDist,
      done: done,
      doneReason: reason,
      heading: errorYaw > 180 ? 360 - errorYaw : errorYaw
    };
  }

  /* KinematicModelBasic.step, including the one-step steering lag: the heading
   * increment uses the control input stored in the previous state.  The forward
   * speed is the constant command speed of the environment (v_range * 0.6). */
  function stepKinematics(state, wheelAngle, cfg) {
    cfg = cfg || CFG;
    var v = clamp(cfg.controlSpeed, -cfg.vRange, cfg.vRange);
    var w = clamp(wheelAngle, -cfg.wRange, cfg.wRange);
    return {
      x: state.x + v * Math.cos(deg2rad(state.yaw)) * cfg.dt,
      y: state.y + v * Math.sin(deg2rad(state.yaw)) * cfg.dt,
      yaw: mod360(state.yaw + state.w * cfg.dt),
      v: v,
      w: w
    };
  }

  /* ControlState(...).command -> wheel_angle; the environment then divides the
   * wheel angle by w_range to obtain the normalised action. */
  function actionFromWheelAngle(wheelAngle, cfg) {
    cfg = cfg || CFG;
    var command = clamp(wheelAngle, -cfg.wRange, cfg.wRange);
    return command / cfg.wRange;
  }

  /* Port of PurePursuitController.command (PathTracking/pure_pursuit.py):
   * nearest sample, look-ahead point interpolated along the arc length, heading
   * error to that point, curvature kappa = 2 sin(alpha) / look-ahead, yaw rate
   * w = kappa * v, saturated at the simulator yaw-rate limit. */
  function purePursuitCommand(path, s, pose, cfg) {
    cfg = cfg || CFG;
    var n = path.length;
    var best = 0;
    var bestD2 = Infinity;
    for (var i = 0; i < n; i++) {
      var d2 = (path[i][0] - pose[0]) * (path[i][0] - pose[0]) + (path[i][1] - pose[1]) * (path[i][1] - pose[1]);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    var sTarget = s[best] + cfg.lookahead;
    var target;
    var targetIdx = n - 1;
    if (sTarget >= s[n - 1]) {
      target = [path[n - 1][0], path[n - 1][1]];
    } else {
      /* searchsorted(s, sTarget, side="right") - 1, clamped to [0, n-2]. */
      var lo = 0;
      var hi = n;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (s[mid] <= sTarget) lo = mid + 1;
        else hi = mid;
      }
      var j = lo - 1;
      if (j < 0) j = 0;
      if (j > n - 2) j = n - 2;
      targetIdx = j;
      var span = s[j + 1] - s[j];
      var ratio = span <= 0 ? 0 : (sTarget - s[j]) / span;
      target = [
        path[j][0] + ratio * (path[j + 1][0] - path[j][0]),
        path[j][1] + ratio * (path[j + 1][1] - path[j][1])
      ];
    }
    var alpha = Math.atan2(target[1] - pose[1], target[0] - pose[0]) - deg2rad(pose[2]);
    alpha = Math.atan2(Math.sin(alpha), Math.cos(alpha));
    var kappa = (2.0 * Math.sin(alpha)) / cfg.lookahead;
    var w = rad2deg(kappa * cfg.controlSpeed);
    return { wheelAngle: clamp(w, -cfg.wRange, cfg.wRange), targetIdx: targetIdx, lookahead: target };
  }

  /* The three controllers used in the published reference runs.  ``pure`` is the
   * repository baseline; ``perturb`` adds a deterministic sinusoidal steering
   * bias; ``recovery`` saturates the steering for the first 30 steps. */
  function controlCommand(mode, stepIndex, path, s, pose, cfg) {
    cfg = cfg || CFG;
    var base = purePursuitCommand(path, s, pose, cfg).wheelAngle;
    if (mode === "perturb") {
      base += 25.0 * Math.sin(0.05 * stepIndex);
    } else if (mode === "recovery" && stepIndex < 30) {
      base = cfg.wRange;
    }
    return clamp(base, -cfg.wRange, cfg.wRange);
  }

  /* ------------------------------------------------------------------ *
   * Episode runners
   * ------------------------------------------------------------------ */

  function randomControlPoints() {
    var p = [
      [200 + Math.floor(Math.random() * 60) - 30, 50 + Math.floor(Math.random() * 60) - 30],
      [200 + Math.floor(Math.random() * 60) - 30, 200 + Math.floor(Math.random() * 60) - 30],
      [150 + Math.floor(Math.random() * 60) - 30, 250 + Math.floor(Math.random() * 60) - 30],
      [50 + Math.floor(Math.random() * 60) - 30, 350 + Math.floor(Math.random() * 60) - 30]
    ];
    if (Math.random() > 0.5) {
      for (var i = 0; i < 4; i++) {
        p[i][0] = 400 - p[i][0];
      }
    }
    return p;
  }

  function makeState(scenario) {
    var start = scenario.start;
    return { x: start[0], y: start[1], yaw: start[2], v: 0.0, w: 0.0 };
  }

  /* Runs one episode.
   *
   *   mode "replay"      executes the commands recorded by the Python reference
   *                      run in docs/data/reference.json (dynamics, reward and
   *                      observation port only).
   *   mode "closed-loop" recomputes the control input in the browser with the
   *                      same controller definition, so the whole control loop
   *                      is exercised (vehicle model + controller + reward).
   */
  function runEpisode(scenario, options) {
    options = options || {};
    var mode = options.mode || "replay";
    var cfg = Object.assign({}, CFG, options.config || {});
    var path = scenario.path;
    var s = cumulativeArcLength(path);
    var ctrlMode = scenario.mode || "pure";
    var recorded = scenario.commands || [];
    var limit = mode === "replay" ? recorded.length : cfg.maxSteps;

    var state = makeState(scenario);
    var record = [[state.x, state.y, state.yaw]];
    var lastIdx = 0;
    var rewards = [];
    var trajectory = [];
    var commands = [];
    var observations = [];
    var doneReason = "running";

    for (var t = 0; t < limit; t++) {
      var w =
        mode === "replay"
          ? recorded[t]
          : controlCommand(ctrlMode, t, path, s, [state.x, state.y, state.yaw], cfg);

      state = stepKinematics(state, w, cfg);
      record.push([state.x, state.y, state.yaw]);

      var info = computeReward(path, lastIdx, state);
      lastIdx = info.minIdx;
      rewards.push(info.reward);
      trajectory.push([state.x, state.y, state.yaw, state.v]);
      commands.push(w);
      observations.push(buildObservation(path, record, info.minIdx));

      doneReason = info.done ? info.doneReason : "running";
      if (info.done) break;
      if (t + 1 >= cfg.maxSteps) {
        doneReason = "max_step";
        break;
      }
    }

    var total = rewards.reduce(function (a, b) {
      return a + b;
    }, 0);
    return {
      engine: ENGINE_VERSION,
      mode: mode,
      steps: trajectory.length,
      trajectory: trajectory,
      commands: commands,
      rewards: rewards,
      observation: observations,
      doneReason: doneReason,
      totalReward: total
    };
  }

  global.RNE = {
    version: ENGINE_VERSION,
    config: CFG,
    mod360: mod360,
    clamp: clamp,
    ensureUniqueCoordinates: ensureUniqueCoordinates,
    cubicSpline1d: cubicSpline1d,
    cubicSpline2d: cubicSpline2d,
    solveLinearSystem: solveLinearSystem,
    searchNearest: searchNearest,
    searchFutureNearest: searchFutureNearest,
    cumulativeArcLength: cumulativeArcLength,
    recordPathSample: recordPathSample,
    futurePathSample: futurePathSample,
    buildObservation: buildObservation,
    computeReward: computeReward,
    stepKinematics: stepKinematics,
    actionFromWheelAngle: actionFromWheelAngle,
    purePursuitCommand: purePursuitCommand,
    controlCommand: controlCommand,
    randomControlPoints: randomControlPoints,
    runEpisode: runEpisode
  };
})(typeof window !== "undefined" ? window : globalThis);
