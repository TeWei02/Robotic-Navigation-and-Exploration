"""Tests for the published browser demo and its reference data set.

Run with::

    python3 -m pytest -q

Two jobs:

* keep the committed data set honest — it must be exactly reproducible from the
  repository's own simulator, and its provenance note must keep saying that no
  trained checkpoint is distributed;
* keep the published demo from rotting — assets present, web-app manifest and
  service worker consistent, scope notice still on the page, no template text
  left behind.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data" / "reference.json"
TOOL = ROOT / "tools" / "build_web_data.py"

SCENARIO_NAMES = [
    "corridor-turn",
    "wide-sweep",
    "tight-arc",
    "offset-entry",
    "biased-driver",
    "recovery",
]
DEMO_FILES = [
    "index.html",
    "app.js",
    "engine.js",
    "manifest.json",
    "sw.js",
    "icons/icon-192.png",
    "icons/icon-512.png",
]


def _load_tool():
    """Import tools/build_web_data.py as a module."""
    pytest.importorskip("cv2", reason="the HW3-1 simulator imports cv2 at module level")
    spec = importlib.util.spec_from_file_location("build_web_data", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_published_dataset_structure(dataset):
    scenarios = dataset["scenarios"]
    assert [s["name"] for s in scenarios] == SCENARIO_NAMES
    assert [s["id"] for s in scenarios] == [f"scenario-{i}" for i in range(1, len(scenarios) + 1)]

    for scenario in scenarios:
        steps = scenario["summary"]["steps"]
        assert steps > 0
        assert steps <= dataset["meta"]["maxSteps"]
        assert len(scenario["commands"]) == steps
        assert len(scenario["actions"]) == steps
        assert len(scenario["rewards"]) == steps
        assert len(scenario["trajectory"]) == steps
        assert len(scenario["minIdx"]) == steps
        assert len(scenario["progress"]) == steps
        assert len(scenario["observation"]) == steps
        # 14-dimensional observation: two previous poses + four look-ahead waypoints
        assert all(len(obs) == 14 for obs in scenario["observation"])
        assert scenario["summary"]["doneReason"] in {"goal", "max_step"}
        assert len(scenario["path"]) == scenario["summary"]["pathSamples"]


def test_meta_states_provenance_honestly(dataset):
    meta = dataset["meta"]
    assert meta["generator"] == "tools/build_web_data.py"
    assert meta["dt"] == pytest.approx(0.1)
    assert meta["wRange"] == pytest.approx(60.0)
    note = meta["note"]
    assert "pure-pursuit baseline" in note
    assert "not distributed with this repository" in note


def test_scenarios_cover_the_three_control_modes(dataset):
    scenarios = {s["name"]: s for s in dataset["scenarios"]}
    assert {s["mode"] for s in scenarios.values()} == {"pure", "perturb", "recovery"}

    # the perturbed run leaves the path measurably; the recovery run includes
    # regressing steps, i.e. the negative branch of the progress reward
    assert scenarios["biased-driver"]["summary"]["maxMinDist"] > 1.0
    assert scenarios["recovery"]["summary"]["stepsWithoutProgress"] > 0


def test_reference_dataset_is_reproducible(dataset):
    tool = _load_tool()
    assert tool.check() == 0, "tools/build_web_data.py --check reported a mismatch"

    rebuilt = tool.build()["scenarios"][0]
    stored = dataset["scenarios"][0]
    assert rebuilt["commands"] == stored["commands"]
    assert rebuilt["trajectory"] == stored["trajectory"]
    assert rebuilt["observation"] == stored["observation"]
    assert rebuilt["summary"] == stored["summary"]


def test_baseline_controller_is_deterministic():
    tool = _load_tool()
    first = tool.run_scenario(1, "corridor-turn", "pure", "determinism check")
    second = tool.run_scenario(1, "corridor-turn", "pure", "determinism check")
    assert first["commands"] == second["commands"]
    assert first["trajectory"] == second["trajectory"]
    assert first["start"] == second["start"]


def test_demo_shell_is_complete():
    for name in DEMO_FILES:
        assert (DOCS / name).is_file(), f"docs/{name} is missing"
    assert DATA.is_file(), "docs/data/reference.json is missing"


def test_manifest_and_service_worker_are_consistent():
    manifest = json.loads((DOCS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["start_url"] in {"./index.html", "index.html"}
    assert manifest["icons"], "manifest declares no icons"
    for icon in manifest["icons"]:
        assert (DOCS / icon["src"]).is_file(), f"manifest icon missing: {icon['src']}"

    worker = (DOCS / "sw.js").read_text(encoding="utf-8")
    cached = re.findall(r'"\./([^"]*)"', worker)
    assert cached, "service worker caches nothing"
    for path in cached:
        target = DOCS / path if path else DOCS / "index.html"
        assert target.is_file(), f"service worker caches missing file: {path}"


def test_demo_keeps_its_scope_notice():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "not distributed with this repository" in html
    assert "pure-pursuit" in html
    assert "github.com/TeWei02/Robotic-Navigation-and-Exploration" in html
    for asset in ["engine.js", "app.js", "manifest.json"]:
        assert asset in html, f"index.html does not reference {asset}"


def test_script_and_markup_agree_on_element_ids():
    """Every element app.js looks up must exist in index.html."""
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    script = (DOCS / "app.js").read_text(encoding="utf-8")
    present = set(re.findall(r'id="([^"]+)"', html))
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', script))
    assert wanted, "app.js looks up no elements"
    assert wanted <= present, f"missing elements in index.html: {sorted(wanted - present)}"


def test_demo_has_no_placeholder_text():
    for name in ["index.html", "app.js", "engine.js"]:
        text = (DOCS / name).read_text(encoding="utf-8").lower()
        for token in ["fixme", "lorem ipsum", "placeholder", "your text here", "sample text"]:
            assert token not in text, f"docs/{name} still contains {token!r}"
