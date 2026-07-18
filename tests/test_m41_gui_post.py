"""
M41 GUI post-processing tests — pyradioss.gui.postproc + the Post-processing
tab / auto-convert hook (no Fortran counterpart, so no reference trace; these
pin the observable contracts the converters depend on).

Covered head-less (no display, no exe, no vortex needed):

* artifact detection (A-file family, binary Txx, port-native CSV/VTK, d3plot),
  including natural-order sorting and the exclusion of ``…A001.vtk`` /
  ``…A001.d3plot`` from the animation family;
* converter command construction (d3plot subprocess driver, anim->VTK per-file
  commands, TH->CSV command, exec-dir resolution);
* the streaming converters' error paths — missing exe, missing binary, missing
  animation family, and (monkeypatched) missing Vortex install returning the
  install hint rather than a traceback;
* config persistence of the exec dir + the three auto-convert toggles;
* PostProcRunner / run_post_actions results + emitted event shapes;
* head-less-safe Post-processing tab construction + artifact rendering.

Plus one ``slow`` empirical acceptance test: a REAL Johnson-Cook plastic run's
animation files -> d3plot, read back with lasso-python, asserting the shell
effective-plastic-strain array is present and non-zero at the final state.
"""

import os
import queue

import pytest

from pyradioss.gui import postproc as PP
from pyradioss.gui import runner as R


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def artifact_dir(tmp_path):
    """A run dir populated with fake artifacts of every kind."""
    d = tmp_path / "run"
    d.mkdir()
    # animation family (natural order test: A2 before A10)
    for n in (1, 2, 10):
        (d / f"MODELA{n:03d}").write_bytes(b"fake-anim")
    # a stray vtk of an anim file must NOT be taken for an anim file
    (d / "MODELA001.vtk").write_text("# vtk")
    # binary time-history + its would-be csv sibling (port-native csv)
    (d / "MODELT01").write_bytes(b"fake-th-binary")
    (d / "MODELT01.csv").write_text("time,IE\n0,0\n")
    # port-native vtk + an existing d3plot master
    (d / "MODEL.vtk").write_text("# vtk")
    (d / "MODEL.d3plot").write_bytes(b"fake-d3plot")
    return str(d)


# ---------------------------------------------------------------------------
# artifact detection
# ---------------------------------------------------------------------------

def test_find_anim_files_natural_order_and_exclusions(artifact_dir):
    anim = PP.find_anim_files(artifact_dir)
    names = [os.path.basename(a) for a in anim]
    assert names == ["MODELA001", "MODELA002", "MODELA010"]   # A10 last
    assert all(not n.endswith(".vtk") for n in names)


def test_anim_file_stem(artifact_dir):
    stem = PP.anim_file_stem(artifact_dir)
    assert os.path.basename(stem) == "MODEL"
    assert os.path.dirname(stem) == artifact_dir


def test_find_th_binary_and_port_outputs(artifact_dir):
    assert os.path.basename(PP.find_th_binary(artifact_dir)) == "MODELT01"
    assert [os.path.basename(p) for p in PP.find_port_csv(artifact_dir)] \
        == ["MODELT01.csv"]
    # both .vtk files are reported (the anim-named one is still a real vtk)
    assert [os.path.basename(p) for p in PP.find_port_vtk(artifact_dir)] \
        == ["MODEL.vtk", "MODELA001.vtk"]
    assert os.path.basename(PP.find_d3plot(artifact_dir)) == "MODEL.d3plot"


def test_detect_artifacts(artifact_dir):
    a = PP.detect_artifacts(artifact_dir)
    assert len(a["anim_files"]) == 3
    assert a["can_d3plot"] and a["can_vtk"] and a["can_th_csv"]
    assert os.path.basename(a["anim_stem"]) == "MODEL"


def test_detect_artifacts_empty_dir(tmp_path):
    a = PP.detect_artifacts(str(tmp_path))
    assert a["anim_files"] == [] and a["anim_stem"] is None
    assert not a["can_d3plot"] and not a["can_vtk"] and not a["can_th_csv"]
    assert a["th_binary"] is None and a["d3plot"] is None


def test_detect_artifacts_missing_dir_never_raises():
    a = PP.detect_artifacts("no_such_dir_xyz")
    assert a["anim_files"] == [] and not a["can_d3plot"]


# ---------------------------------------------------------------------------
# command construction (pure)
# ---------------------------------------------------------------------------

def test_exec_path_default_and_override():
    assert PP.exec_path(None, PP.ANIM_TO_VTK_EXE) == \
        os.path.join(PP.DEFAULT_EXEC_DIR, PP.ANIM_TO_VTK_EXE)
    assert PP.exec_path("D:/tools", PP.TH_TO_CSV_EXE) == \
        os.path.join("D:/tools", PP.TH_TO_CSV_EXE)


def test_d3plot_command(artifact_dir):
    cmd = PP.d3plot_command(artifact_dir, python_exe="py.exe")
    assert cmd[0] == "py.exe" and cmd[1] == "-u" and cmd[2] == "-c"
    assert "readAndConvert" in cmd[3]
    assert os.path.basename(cmd[4]) == "MODEL"      # the stem, not a file
    assert PP.d3plot_command("no_such_dir") is None


def test_anim_to_vtk_commands(artifact_dir):
    cmds = PP.anim_to_vtk_commands(artifact_dir, exec_dir="D:/x")
    assert len(cmds) == 3
    cmd0, out0 = cmds[0]
    assert cmd0[0] == os.path.join("D:/x", PP.ANIM_TO_VTK_EXE)
    assert os.path.basename(cmd0[1]) == "MODELA001"
    assert out0 == cmd0[1] + ".vtk"


def test_th_to_csv_command(artifact_dir):
    cmd, out = PP.th_to_csv_command(artifact_dir, exec_dir="D:/x")
    assert cmd[0] == os.path.join("D:/x", PP.TH_TO_CSV_EXE)
    assert cmd[1] == "MODELT01"                     # basename, run in cwd
    assert os.path.basename(out) == "MODELT01.csv"


def test_th_to_csv_command_none_without_binary(tmp_path):
    assert PP.th_to_csv_command(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# streaming converters — error paths (no exe / no vortex / no artifacts)
# ---------------------------------------------------------------------------

def _collect(events):
    def emit(ev):
        events.append(ev)
    return emit


def test_convert_to_d3plot_no_anim(tmp_path):
    events = []
    res = PP.convert_to_d3plot(str(tmp_path), emit=_collect(events))
    assert res["ok"] is False and "no animation files" in res["message"]
    assert ("post_done", "d3plot", False, [], res["message"]) in events


def test_convert_to_d3plot_vortex_missing(artifact_dir, monkeypatch):
    monkeypatch.setattr(PP, "vortex_available", lambda: False)
    events = []
    res = PP.convert_to_d3plot(artifact_dir, emit=_collect(events))
    assert res["ok"] is False
    assert PP.VORTEX_PIN in res["message"]           # carries the pin/hint
    assert any(PP.VORTEX_PIN in e[2] for e in events if e[0] == "line")


def test_convert_anim_to_vtk_missing_exe(artifact_dir):
    events = []
    res = PP.convert_anim_to_vtk(
        artifact_dir, exec_dir=str(artifact_dir), emit=_collect(events))
    assert res["ok"] is False
    assert "converter not found" in res["message"]


def test_convert_anim_to_vtk_no_anim(tmp_path):
    res = PP.convert_anim_to_vtk(str(tmp_path), exec_dir="D:/x")
    assert res["ok"] is False and "no animation files" in res["message"]


def test_convert_th_to_csv_no_binary(tmp_path):
    events = []
    res = PP.convert_th_to_csv(str(tmp_path), emit=_collect(events))
    assert res["ok"] is False
    assert "no binary time-history" in res["message"]
    assert events[-1][0] == "post_done"


def test_convert_th_to_csv_missing_exe(artifact_dir):
    res = PP.convert_th_to_csv(artifact_dir, exec_dir=str(artifact_dir))
    assert res["ok"] is False and "converter not found" in res["message"]


# ---------------------------------------------------------------------------
# PostProcRunner / run_post_actions
# ---------------------------------------------------------------------------

def test_run_post_actions_collects_results(tmp_path):
    # both actions fail gracefully (no artifacts) but return result dicts
    results = PP.run_post_actions(
        str(tmp_path), ["vtk", "th_csv"], exec_dir="D:/x")
    assert set(results) == {"vtk", "th_csv"}
    assert results["vtk"]["ok"] is False
    assert results["th_csv"]["ok"] is False


def test_postproc_runner_to_completion_and_events(tmp_path):
    run = PP.PostProcRunner(str(tmp_path), ["th_csv"], exec_dir="D:/x")
    results = run.run_to_completion()
    assert results["th_csv"]["ok"] is False
    # the queue received a post_done event for th_csv
    seen = []
    try:
        while True:
            seen.append(run.queue.get_nowait())
    except queue.Empty:
        pass
    kinds = [e for e in seen if e[0] == "post_done"]
    assert kinds and kinds[0][1] == "th_csv"


def test_postproc_runner_ignores_unknown_actions(tmp_path):
    run = PP.PostProcRunner(str(tmp_path), ["nope", "vtk"], exec_dir="D:/x")
    assert run.actions == ["vtk"]


# ---------------------------------------------------------------------------
# config persistence of exec dir + auto-convert toggles
# ---------------------------------------------------------------------------

def test_config_postproc_roundtrip(tmp_path):
    path = str(tmp_path / "cfg" / "config.json")
    cfg = R.GuiConfig(path=path)
    cfg.set("exec_dir", "D:/OR/exec")
    cfg.set("auto_d3plot", True)
    cfg.set("auto_vtk", False)
    cfg.set("auto_th_csv", True)
    cfg.save()

    r = R.GuiConfig(path=path).load()
    assert r.get("exec_dir") == "D:/OR/exec"
    assert r.get("auto_d3plot") is True
    assert r.get("auto_vtk") is False
    assert r.get("auto_th_csv") is True


def test_config_postproc_defaults(tmp_path):
    cfg = R.GuiConfig(path=str(tmp_path / "none.json")).load()
    assert cfg.get("exec_dir") == ""
    assert cfg.get("auto_d3plot") is False
    assert cfg.get("auto_vtk") is False
    assert cfg.get("auto_th_csv") is False


# ---------------------------------------------------------------------------
# JobRunner auto-convert hook (no real run — just the plumbing contract)
# ---------------------------------------------------------------------------

def test_jobrunner_accepts_post_actions(tmp_path):
    deck = tmp_path / "RUN_0000.rad"
    deck.write_text("/BEGIN\n")
    run = R.JobRunner(str(deck), post_actions=["d3plot", "vtk"],
                      exec_dir="D:/x")
    assert run.post_actions == ["d3plot", "vtk"]
    assert run.exec_dir == "D:/x"
    assert run.post_results == {}


def test_jobrunner_engine_normal_gate():
    # the hook only fires on an ENGINE NORMAL termination
    class _Stub(R.JobRunner):
        def __init__(self):
            pass
    s = _Stub()
    s.termination = ("ENGINE", "NORMAL", "")
    assert s._engine_terminated_normally() is True
    s.termination = ("ENGINE", "ERROR", "boom")
    assert s._engine_terminated_normally() is False
    s.termination = ("STARTER", "NORMAL", "")
    assert s._engine_terminated_normally() is False
    s.termination = None
    assert s._engine_terminated_normally() is False


# ---------------------------------------------------------------------------
# head-less-safe Post-processing tab construction
# ---------------------------------------------------------------------------

def _make_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display available: {exc}")
    root.withdraw()
    return root


def test_postproc_tab_constructed_and_renders(artifact_dir):
    root = _make_root()
    try:
        from pyradioss.gui.app import PyradiossGUI
        gui = PyradiossGUI(root)
        # four tabs now: Log / Results / Deck info / Post-processing
        assert len(gui.nb.tabs()) == 4
        titles = [gui.nb.tab(t, "text") for t in gui.nb.tabs()]
        assert "Post-processing" in titles
        # point it at the fixture dir and detect
        gui.post_dir_var.set(artifact_dir)
        gui.refresh_artifacts()
        summary = gui.artifacts_var.get()
        assert "A-files (anim): 3" in summary
        # the three converter buttons exist
        assert gui.d3_btn is not None and gui.vtk_btn is not None \
            and gui.thcsv_btn is not None
    finally:
        root.destroy()


def test_auto_convert_toggles_persist(tmp_path, monkeypatch):
    root = _make_root()
    try:
        from pyradioss.gui.app import PyradiossGUI
        # isolate the config file
        cfgpath = str(tmp_path / "config.json")
        monkeypatch.setattr(R.GuiConfig, "default_path",
                            staticmethod(lambda: cfgpath))
        gui = PyradiossGUI(root)
        gui.auto_d3plot_var.set(True)
        gui.auto_th_csv_var.set(True)
        gui.exec_dir_var.set("D:/OR/exec")
        gui._persist_post_opts()
        reloaded = R.GuiConfig(path=cfgpath).load()
        assert reloaded.get("auto_d3plot") is True
        assert reloaded.get("auto_th_csv") is True
        assert reloaded.get("exec_dir") == "D:/OR/exec"
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# EMPIRICAL acceptance: real plastic run -> d3plot -> effective plastic strain
# ---------------------------------------------------------------------------

_BALL_SRC = (r"E:/openradioss_run/Ryan_Lee_Examples/ton-mm-s/runs/"
             r"W3_SETUP_Ball_impact__ton-mm-s")
_BALL_STEM = "W3_SETUP_Ball_impact__ton-mm-s"


@pytest.mark.slow
def test_convert_to_d3plot_effective_plastic_strain(tmp_path):
    """Convert a real Johnson-Cook plastic run's A-files to d3plot and prove
    effective plastic strain survives (present + non-zero at the final
    state). Skips cleanly when the fixture / vortex / lasso are unavailable."""
    import glob
    import shutil

    src_anims = sorted(glob.glob(os.path.join(_BALL_SRC, _BALL_STEM + "A0*[0-9]")))
    if not src_anims:
        pytest.skip("ball-impact animation fixture not present")
    if not PP.vortex_available():
        pytest.skip("Vortex-Radioss not installed (" + PP.VORTEX_INSTALL_HINT
                    + ")")
    try:
        from lasso.dyna import D3plot
    except Exception as exc:                        # noqa: BLE001
        pytest.skip(f"lasso-python not installed: {exc}")

    # copy the A-files into a scratch dir (source is read-only)
    for a in src_anims:
        shutil.copy(a, tmp_path)

    events = []
    res = PP.convert_to_d3plot(str(tmp_path), emit=lambda e: events.append(e))
    assert res["ok"], f"conversion failed: {res['message']}"
    d3 = PP.find_d3plot(str(tmp_path))
    assert d3 and os.path.exists(d3)

    import numpy as np
    plot = D3plot(d3)
    key = "element_shell_effective_plastic_strain"
    assert key in plot.arrays, "shell effective plastic strain MISSING"
    eps = np.asarray(plot.arrays[key])
    final = eps[-1]
    assert float(np.max(final)) > 0.0, "effective plastic strain all-zero"
    # sanity: the streamed events ended with a successful post_done
    assert events[-1][0] == "post_done" and events[-1][2] is True
