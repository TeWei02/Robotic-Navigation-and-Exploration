/* Look-ahead benchmark panel.
 *
 * Reads data/benchmark.json — the output of tools/benchmark.py — and draws the
 * baseline's response to its own look-ahead distance. No third-party code and
 * no build step, matching the rest of this demo.
 *
 * The panel is deliberately data-driven: the optimum, the percentages and the
 * table are all derived from the JSON, so re-running the sweep and regenerating
 * the file updates this page without editing markup. Nothing here restates a
 * number that is not in the file.
 */
(function () {
  "use strict";

  var statusEl = document.getElementById("bench-status");
  var chartEl = document.getElementById("bench-chart");
  var tableEl = document.getElementById("bench-table");
  var noteEl = document.getElementById("bench-note");

  function fail(message) {
    statusEl.textContent = message;
    statusEl.className = "bad";
  }

  function esc(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* ------------------------------------------------------------------ chart */
  function niceTop(value, ticks) {
    var raw = value / ticks;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    return step * ticks;
  }

  function fmtTick(value, step) {
    return step < 1 ? value.toFixed(1) : value.toFixed(0);
  }

  function drawChart(rows) {
    var W = 900;
    var H = 400;
    var pad = { l: 66, r: 66, t: 30, b: 56 };
    var iw = W - pad.l - pad.r;
    var ih = H - pad.t - pad.b;
    var n = rows.length;

    function xAt(i) {
      return pad.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
    }

    var maxReturn = Math.max.apply(null, rows.map(function (r) { return r.meanReturn; }));
    var maxDev = Math.max.apply(null, rows.map(function (r) { return r.meanMeanMinDist; }));
    // Round the axis tops up to a readable step so the gridlines land on whole
    // numbers instead of on 0/80/161/241/321.
    var TICKS = 4;
    var returnTop = niceTop(maxReturn * 1.05, TICKS);
    var devTop = niceTop(maxDev * 1.05, TICKS);
    var returnStep = returnTop / TICKS;
    var devStep = devTop / TICKS;

    function yReturn(v) { return pad.t + ih - (v / returnTop) * ih; }
    function yDev(v) { return pad.t + ih - (v / devTop) * ih; }

    var parts = [];
    parts.push('<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" ' +
      'aria-label="Mean return and mean tracking deviation against look-ahead distance" ' +
      'style="max-width:100%;height:auto">');

    // horizontal gridlines, labelled on both axes
    for (var g = 0; g <= 4; g++) {
      var frac = g / 4;
      var y = pad.t + ih - frac * ih;
      parts.push('<line x1="' + pad.l + '" y1="' + y.toFixed(1) + '" x2="' + (pad.l + iw) +
        '" y2="' + y.toFixed(1) + '" stroke="#243149" stroke-width="1" ' +
        (g === 0 ? '' : 'stroke-dasharray="3 4"') + '/>');
      parts.push('<text x="' + (pad.l - 10) + '" y="' + (y + 4).toFixed(1) +
        '" fill="#a7b4c8" font-size="11" text-anchor="end" font-family="ui-monospace,monospace">' +
        fmtTick(frac * returnTop, returnStep) + '</text>');
      parts.push('<text x="' + (pad.l + iw + 10) + '" y="' + (y + 4).toFixed(1) +
        '" fill="#a7b4c8" font-size="11" text-anchor="start" font-family="ui-monospace,monospace">' +
        fmtTick(frac * devTop, devStep) + '</text>');
    }

    // axis titles
    parts.push('<text x="' + pad.l + '" y="18" fill="#2fd6c3" font-size="12" text-anchor="start">' +
      'mean return</text>');
    parts.push('<text x="' + (pad.l + iw) + '" y="18" fill="#ffb454" font-size="12" ' +
      'text-anchor="end">mean deviation</text>');

    // the published configuration, as a vertical marker behind the series
    var publishedIndex = -1;
    var bestIndex = 0;
    rows.forEach(function (r, i) {
      if (r.isPublishedBaseline) { publishedIndex = i; }
      if (r.meanReturn > rows[bestIndex].meanReturn) { bestIndex = i; }
    });
    if (publishedIndex >= 0) {
      parts.push('<line x1="' + xAt(publishedIndex).toFixed(1) + '" y1="' + pad.t +
        '" x2="' + xAt(publishedIndex).toFixed(1) + '" y2="' + (pad.t + ih) +
        '" stroke="#ff7b8a" stroke-width="1.5" stroke-dasharray="5 4"/>');
      parts.push('<text x="' + (xAt(publishedIndex) + 6).toFixed(1) + '" y="' + (pad.t + 14) +
        '" fill="#ff7b8a" font-size="11">published Ld</text>');
    }

    function series(key, yFn, colour, dash) {
      var points = rows.map(function (r, i) {
        return xAt(i).toFixed(1) + "," + yFn(r[key]).toFixed(1);
      }).join(" ");
      parts.push('<polyline points="' + points + '" fill="none" stroke="' + colour +
        '" stroke-width="2.5"' + (dash ? ' stroke-dasharray="6 4"' : '') + '/>');
      rows.forEach(function (r, i) {
        parts.push('<circle cx="' + xAt(i).toFixed(1) + '" cy="' + yFn(r[key]).toFixed(1) +
          '" r="4" fill="#0b111c" stroke="' + colour + '" stroke-width="2"/>');
      });
    }

    series("meanReturn", yReturn, "#2fd6c3", false);
    series("meanMeanMinDist", yDev, "#ffb454", true);

    // x tick labels
    rows.forEach(function (r, i) {
      parts.push('<text x="' + xAt(i).toFixed(1) + '" y="' + (pad.t + ih + 20) +
        '" fill="#a7b4c8" font-size="11" text-anchor="middle" ' +
        'font-family="ui-monospace,monospace">' + r.lookahead + '</text>');
    });
    parts.push('<text x="' + (pad.l + iw / 2) + '" y="' + (H - 8) +
      '" fill="#a7b4c8" font-size="12" text-anchor="middle">look-ahead distance Ld (length units)</text>');

    // mark the measured optimum
    parts.push('<circle cx="' + xAt(bestIndex).toFixed(1) + '" cy="' + yReturn(rows[bestIndex].meanReturn).toFixed(1) +
      '" r="8" fill="none" stroke="#2fd6c3" stroke-width="2"/>');
    parts.push('<text x="' + xAt(bestIndex).toFixed(1) + '" y="' + (yReturn(rows[bestIndex].meanReturn) - 16).toFixed(1) +
      '" fill="#2fd6c3" font-size="11" text-anchor="middle">best measured</text>');

    parts.push('</svg>');
    chartEl.innerHTML = parts.join("");
  }

  /* ------------------------------------------------------------------ table */
  function drawTable(rows) {
    var bestReturn = Math.max.apply(null, rows.map(function (r) { return r.meanReturn; }));
    var head = "<tr><th>Ld</th><th>Goal rate</th><th>Mean return</th><th>Mean deviation</th>" +
      "<th>Worst deviation</th><th>Mean steps</th></tr>";
    var body = rows.map(function (r) {
      var marks = [];
      if (r.isPublishedBaseline) { marks.push("published"); }
      if (r.meanReturn === bestReturn) { marks.push("best return"); }
      var label = r.lookahead + (marks.length ? " (" + marks.join(", ") + ")" : "");
      var weight = marks.length ? ' style="font-weight:600"' : "";
      return "<tr" + weight + "><td>" + esc(label) + "</td><td>" +
        (r.goalRate * 100).toFixed(0) + "%</td><td>" + r.meanReturn.toFixed(2) + "</td><td>" +
        r.meanMeanMinDist.toFixed(3) + "</td><td>" + r.worstMaxMinDist.toFixed(3) + "</td><td>" +
        r.meanSteps.toFixed(1) + "</td></tr>";
    }).join("");
    tableEl.innerHTML = "<table>" + head + body + "</table>";
  }

  /* ------------------------------------------------------------------- note */
  function drawNote(data, rows) {
    var best = rows.reduce(function (a, b) { return b.meanReturn > a.meanReturn ? b : a; });
    var published = rows.filter(function (r) { return r.isPublishedBaseline; })[0];
    var text = "Measured optimum in this sweep: Ld = " + best.lookahead +
      " at a mean return of " + best.meanReturn.toFixed(2) + " and a mean deviation of " +
      best.meanMeanMinDist.toFixed(3) + ".";
    if (published && published !== best) {
      // Report the shortfall of the published value against the best measured
      // one, not the other way round: "33% lower" would be the wrong direction.
      var shortfall = ((1 - published.meanReturn / best.meanReturn) * 100).toFixed(0);
      text += " The published configuration, Ld = " + published.lookahead + ", returns " +
        published.meanReturn.toFixed(2) + " at a mean deviation of " + published.meanMeanMinDist.toFixed(3) +
        " — " + shortfall + "% lower return than that and " +
        (published.meanMeanMinDist / best.meanMeanMinDist).toFixed(1) +
        "x the tracking deviation.";
    }
    var goals = rows.map(function (r) { return r.goalRate; });
    if (Math.min.apply(null, goals) === Math.max.apply(null, goals)) {
      text += " The goal rate is " + (rows[0].goalRate * 100).toFixed(0) +
        "% at every setting, so it does not discriminate between them.";
    }
    text += " " + data.scope;
    noteEl.textContent = text;
  }

  /* ------------------------------------------------------------------- load */
  fetch("data/benchmark.json", { cache: "no-cache" })
    .then(function (response) {
      if (!response.ok) { throw new Error("HTTP " + response.status); }
      return response.json();
    })
    .then(function (data) {
      var rows = data.aggregate || [];
      if (!rows.length) { throw new Error("the file contains no aggregate rows"); }
      drawChart(rows);
      drawTable(rows);
      drawNote(data, rows);
      var guard = data.guard_passed ? "guard passed" : "guard FAILED";
      statusEl.textContent = rows.length + " look-ahead settings x " +
        (data.runs ? (data.runs.length / rows.length) : "?") + " scenarios · " + guard +
        " · generated by " + data.generated_by;
      statusEl.className = data.guard_passed ? "ok" : "bad";
    })
    .catch(function (error) {
      fail("Could not load data/benchmark.json (" + error.message +
        "). Run tools/benchmark.py to regenerate it.");
    });
})();