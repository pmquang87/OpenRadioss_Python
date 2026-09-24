"""
M41 GUI tests — pyradioss.gui (the run-and-monitor Tkinter front-end).

The GUI has no Fortran counterpart, so there is no reference trace to match;
these tests instead pin the observable contracts the GUI depends on:

* the Engine listing-line + TERMINATION-banner parsers, against REAL lines
  emitted by pyradioss.engine.engine;
* the Radioss file-naming derivation (_0000.rad -> _0001.rad -> T01.csv);
* the T01 CSV loader (including tolerance of a truncated live row);
* the read-only deck-summary builder on examples/tensile_bar (and its
  never-crash guarantee on a bad deck);
* the JSON config save/load round-trip;
* JobRunner driving a REAL tiny run (tensile_bar) to ENGINE NORMAL, plus a
  clean start/stop;
* head-less-safe widget construction (skipped with a reason if no display).
"""

import os
import shutil

import pytest

from pyradioss.gui import runner as R
from pyradioss.gui import plots as P


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tensile_deck(tmp_path):
    """Copy the tensile_bar starter+engine decks into a temp dir (generating
    them first if the example has not been materialised)."""
    ex = os.path.join(os.path.dirname(__file__), "..", "examples",
                      "tensile_bar")
    for fname in ("TENSILE_0000.rad", "TENSILE_0001.rad"):
        src = os.path.join(ex, fname)
        if not os.path.exists(src):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "gen", os.path.join(ex, "generate_deck.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        shutil.copy(src, tmp_path / fname)
    return str(tmp_path / "TENSILE_0000.rad")


# ---------------------------------------------------------------------------
# listing-line parser (real engine lines)
# ---------------------------------------------------------------------------

def test_parse_listing_line_real():
    line = ("    100  1.09640E-02  1.09676E-04  3.73293E-04  1.65685E-05  "
            "3.70680E-07  0.00000E+00  7.54321E-06  3.97775E-04     0.00")
    st = R.parse_listing_line(line)
    assert st is not None
    assert st["cycle"] == 100
    assert st["time"] == pytest.approx(1.09640e-2)
    assert st["dt"] == pytest.approx(1.09676e-4)
    assert st["ie"] == pytest.approx(3.73293e-4)
    assert st["ke"] == pytest.approx(1.65685e-5)
    assert st["ew"] == pytest.approx(3.97775e-4)
    assert st["err"] == pytest.approx(0.0)


def test_parse_listing_line_final_cycle():
    # the closing partial-cycle line of a real run (dt collapses to hit t_end)
    line = ("   1847  2.00000E-01  2.87490E-05  3.15308E-01  5.03357E-05  "
            "1.55790E-04  0.00000E+00  1.01404E-04  3.15615E-01     0.00")
    st = R.parse_listing_line(line)
    assert st["cycle"] == 1847
    assert st["time"] == pytest.approx(0.2)


def test_parse_listing_line_rejects_header_and_junk():
    header = ("  CYCLE       TIME        TIME-STEP    ENERGY-IE    "
              "ENERGY-KE    ENERGY-HE    ENERGY-CE    ENERGY-EN    "
              "EXT-WORK   ERROR%")
    assert R.parse_listing_line(header) is None
    assert R.parse_listing_line("") is None
    assert R.parse_listing_line(" -- ELEMENT DELETION: 2 ELEMENT(S)") is None
    assert R.parse_listing_line(
        "     FINAL TIME    . . . . . . :  2.0000000E-01") is None
    # ten tokens but the first is not an integer
    assert R.parse_listing_line("a b c d e f g h i j") is None


# ---------------------------------------------------------------------------
# termination parser
# ---------------------------------------------------------------------------

def test_parse_termination_engine_normal():
    assert R.parse_termination(
        "     ENGINE TERMINATION : NORMAL") == ("ENGINE", "NORMAL", "")


def test_parse_termination_engine_error_emdash():
    kind, status, reason = R.parse_termination(
        "     ENGINE TERMINATION : ERROR — dt below dt_min")
    assert (kind, status) == ("ENGINE", "ERROR")
    assert reason == "dt below dt_min"


def test_parse_termination_engine_error_hyphen():
    assert R.parse_termination(
        "     ENGINE TERMINATION : ERROR - boom") == (
        "ENGINE", "ERROR", "boom")


def test_parse_termination_starter():
    assert R.parse_termination(
        "     STARTER TERMINATION : NORMAL") == ("STARTER", "NORMAL", "")
    assert R.parse_termination(
        "     STARTER TERMINATION : ERROR") == ("STARTER", "ERROR", "")


def test_parse_termination_none():
    assert R.parse_termination("  100  1.0E-02  ...") is None
    assert R.parse_termination("random text") is None


# ---------------------------------------------------------------------------
# file-naming contract
# ---------------------------------------------------------------------------

def test_derive_engine_deck():
    got = R.derive_engine_deck(os.path.join("d", "TENSILE_0000.rad"))
    assert os.path.basename(got) == "TENSILE_0001.rad"
    # unmatched name returned unchanged
    assert R.derive_engine_deck("foo.rad") == "foo.rad"


def test_run_name_and_num_and_t01():
    assert R.run_name_and_num("x/RUN_0001.rad") == ("RUN", 1)
    t01 = R.t01_path_for(os.path.join("dir", "RUN_0001.rad"))
    assert os.path.basename(t01) == "RUNT01.csv"


# ---------------------------------------------------------------------------
# T01 loader
# ---------------------------------------------------------------------------

def test_load_t01_roundtrip(tmp_path):
    csv = tmp_path / "RUNT01.csv"
    csv.write_text(
        "# pyradioss time history (T01 equivalent)\n"
        "TIME,IE,KE,EW,ERR%\n"
        "0.000000000E+00,0.0E+00,0.0E+00,0.0E+00,0.0E+00\n"
        "1.000000000E-03,2.5E-04,1.0E-05,2.6E-04,0.10\n"
        "2.000000000E-03,5.0E-04,1.5E-05,5.2E-04,0.05\n")
    t01 = R.load_t01(str(csv))
    assert t01.columns == ["TIME", "IE", "KE", "EW", "ERR%"]
    assert t01.nrows == 3
    assert t01.time[-1] == pytest.approx(2.0e-3)
    assert t01.column("IE")[-1] == pytest.approx(5.0e-4)
    assert t01.has("ERR%") and not t01.has("NOPE")


def test_load_t01_skips_truncated_row(tmp_path):
    # a live reader can catch a half-written final row -> it must be skipped
    csv = tmp_path / "RUNT01.csv"
    csv.write_text(
        "# comment\n"
        "TIME,IE,KE\n"
        "0.0E+00,1.0E-04,2.0E-04\n"
        "1.0E-03,3.0E-04\n")            # short row
    t01 = R.load_t01(str(csv))
    assert t01.nrows == 1
    assert t01.column("IE")[-1] == pytest.approx(1.0e-4)


# ---------------------------------------------------------------------------
# deck summary (read-only, never-crash)
# ---------------------------------------------------------------------------

def test_build_deck_summary_tensile(tensile_deck):
    s = R.build_deck_summary(tensile_deck)
    assert s["error"] == ""
    assert s["title"] == "TENSILE"
    assert s["nodes"] == 99
    assert s["elements"].get("BRICK") == 40
    assert s["errors"] == 0
    mats = {m["id"]: m for m in s["materials"]}
    assert 1 in mats and mats[1]["law"] == 2
    props = {p["id"]: p for p in s["properties"]}
    assert props[1]["type"] == 14
    parts = {p["id"]: p for p in s["parts"]}
    assert parts[1]["mat_id"] == 1 and parts[1]["prop_id"] == 1
    kws = dict(s["keywords"])
    assert "NODE" in kws and "MAT/LAW2" in kws and "PART" in kws


def test_build_deck_summary_bad_deck_never_crashes():
    s = R.build_deck_summary("this_does_not_exist_0000.rad")
    assert s["error"]                       # a message, not an exception
    assert s["nodes"] == 0


def test_build_deck_summary_garbage_deck(tmp_path):
    bad = tmp_path / "BAD_0000.rad"
    bad.write_text("this is not a valid radioss deck at all\n%%%\n")
    s = R.build_deck_summary(str(bad))      # must return, not raise
    assert isinstance(s, dict)
    assert "nodes" in s


# ---------------------------------------------------------------------------
# config save / load
# ---------------------------------------------------------------------------

def test_config_roundtrip(tmp_path):
    path = str(tmp_path / "cfg" / "config.json")
    cfg = R.GuiConfig(path=path)
    cfg.set("last_dir", "C:/decks")
    cfg.set("backend", "numba")
    cfg.set("nthread", 4)
    cfg.set("channels", ["IE", "EW"])
    cfg.save()

    reloaded = R.GuiConfig(path=path).load()
    assert reloaded.get("last_dir") == "C:/decks"
    assert reloaded.get("backend") == "numba"
    assert reloaded.get("nthread") == 4
    assert reloaded.get("channels") == ["IE", "EW"]


def test_config_missing_file_uses_defaults(tmp_path):
    cfg = R.GuiConfig(path=str(tmp_path / "nope.json")).load()
    assert cfg.get("backend") == "auto"
    assert cfg.get("channels") == ["IE", "KE", "EW", "ERR%"]


def test_config_corrupt_file_uses_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ not valid json ")
    cfg = R.GuiConfig(path=str(path)).load()
    assert cfg.get("backend") == "auto"


# ---------------------------------------------------------------------------
# plotting helpers (matplotlib-independent text fallback)
# ---------------------------------------------------------------------------

def test_format_channel_table():
    t01 = R.T01Data(["TIME", "IE", "KE"],
                    [[0.0, 1.0, 2.0], [1e-3, 3.0, 4.0]])
    txt = P.format_channel_table(t01, ["IE", "KE"])
    assert "TIME" in txt and "IE" in txt and "KE" in txt
    assert "3.00000E+00" in txt
    # a channel not present is dropped, not an error
    assert P.format_channel_table(t01, ["NOPE"]).startswith("(no matching")


def test_available_channels():
    t01 = R.T01Data(["TIME", "IE", "KE", "EW"], [])
    assert P.available_channels(t01) == ["IE", "KE", "EW"]


# ---------------------------------------------------------------------------
# JobRunner — real tiny run + clean stop
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_runner_run_to_completion_normal(tensile_deck):
    run = R.JobRunner(tensile_deck, backend="auto")
    res = run.run_to_completion(timeout=180)
    assert res["returncode"] == 0
    assert res["termination"] == ("ENGINE", "NORMAL", "")
    assert res["t_end"] == pytest.approx(0.2)
    assert res["last_status"]["time"] == pytest.approx(0.2)
    # the T01 the run wrote loads and has the expected global channels
    assert os.path.exists(res["t01_path"])
    t01 = R.load_t01(res["t01_path"])
    assert t01.nrows > 0
    for chan in ("IE", "KE", "EW", "ERR%"):
        assert t01.has(chan)
    assert t01.column("IE")[-1] > 0.0


@pytest.mark.slow
def test_runner_start_then_stop_is_clean(tensile_deck):
    run = R.JobRunner(tensile_deck, backend="auto")
    run.start()
    assert run.is_running() or run.returncode is not None
    run.stop()
    run.join(timeout=30)
    assert not run.is_running()          # stopped or finished, never hung


# ---------------------------------------------------------------------------
# head-less-safe widget construction
# ---------------------------------------------------------------------------

def _make_root():
    try:
        import tkinter as tk
    except ImportError as exc:            # Python built without Tk
        pytest.skip(f"tkinter unavailable: {exc}")
    try:
        root = tk.Tk()
    except tk.TclError as exc:            # no display available
        pytest.skip(f"no Tk display available: {exc}")
    root.withdraw()
    return root


def test_widget_construction(tensile_deck):
    root = _make_root()
    try:
        from pyradioss.gui.app import PyradiossGUI
        gui = PyradiossGUI(root, deck=tensile_deck)
        # the deck was pre-selected and the engine deck auto-derived
        assert gui.deck_var.get() == tensile_deck
        assert gui.engine_var.get() == "TENSILE_0001.rad"
        assert gui.backend_var.get() in ("auto", "numpy", "numba")
        # Log / Results / Deck info / Post-processing
        assert len(gui.nb.tabs()) == 4
    finally:
        root.destroy()


def test_widget_deck_inspection(tensile_deck):
    root = _make_root()
    try:
        from pyradioss.gui.app import PyradiossGUI
        import tkinter as tk
        gui = PyradiossGUI(root, deck=tensile_deck)
        gui.inspect_deck()               # drives build_deck_summary + render
        text = gui.info_text.get("1.0", tk.END)
        assert "TENSILE" in text
        assert "BRICK" in text
        assert "MAT" in text
    finally:
        root.destroy()
