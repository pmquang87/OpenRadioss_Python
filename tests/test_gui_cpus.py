"""
GUI CPU-count control — the ``-np N`` (SPMD domains) setting of the
run-and-monitor GUI (pyradioss.gui).

No Fortran counterpart (the GUI is port-side only); what is pinned here is
the contract between the GUI and the two console programs:

* ``JobRunner(nspmd=N)`` passes ``-np N`` to BOTH the Starter (which then
  writes one restart per domain) and the Engine (which runs the domains):
  the Engine's inipar.F coherence test requires the two to agree;
* ``nspmd=1`` (the default) is the plain serial command line — no flag;
* the JSON config remembers the CPU count and defaults to 1;
* ``available_cpus()`` is the spinbox bound and is at least 1;
* the widget shows the control, validates the entry and clamps a remembered
  value to this machine (skipped without a Tk display);
* a REAL tensile_bar run through the GUI runner with 2 CPUs terminates
  ENGINE NORMAL and gives the serial energies (slow).
"""

import os
import shutil

import pytest

from pyradioss.gui import runner as R


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


def _make_root():
    """A withdrawn Tk root, or a skip when this Python has no tkinter or no
    display (the GUI is optional; the runner logic above is what CI pins)."""
    try:
        import tkinter as tk
    except ImportError as exc:
        pytest.skip(f"tkinter unavailable: {exc}")
    try:
        root = tk.Tk()
    except tk.TclError as exc:            # no display available
        pytest.skip(f"no Tk display available: {exc}")
    root.withdraw()
    return root


def test_available_cpus_is_positive():
    n = R.available_cpus()
    assert isinstance(n, int) and n >= 1
    assert n <= max(1, os.cpu_count() or 1)


def test_config_defaults_to_one_cpu_and_roundtrips(tmp_path):
    path = str(tmp_path / "config.json")
    cfg = R.GuiConfig(path=path).load()
    assert cfg.get("nspmd") == 1
    cfg.set("nspmd", 3)
    cfg.save()
    assert R.GuiConfig(path=path).load().get("nspmd") == 3


def test_runner_default_is_serial_no_np_flag(tensile_deck):
    run = R.JobRunner(tensile_deck, backend="numpy")
    assert run.nspmd == 1
    assert "-np" not in run._starter_cmd()
    assert "-np" not in run._engine_cmd()


def test_runner_passes_np_to_starter_and_engine(tensile_deck):
    run = R.JobRunner(tensile_deck, backend="numpy", nthread=2, nspmd=4)
    st, en = run._starter_cmd(), run._engine_cmd()
    for cmd in (st, en):
        assert cmd[cmd.index("-np") + 1] == "4"
        assert cmd[cmd.index("-nt") + 1] == "2"
    assert st[st.index("-m") + 1] == "pyradioss.starter"
    assert en[en.index("-m") + 1] == "pyradioss.engine"
    assert en[en.index("-backend") + 1] == "numpy"
    # the same N to both programs — the inipar.F coherence test
    assert st[st.index("-np") + 1] == en[en.index("-np") + 1]


def test_runner_clamps_nonpositive_cpu_count_to_serial(tensile_deck):
    assert R.JobRunner(tensile_deck, nspmd=0).nspmd == 1
    assert R.JobRunner(tensile_deck, nspmd=-2).nspmd == 1


@pytest.mark.slow
def test_runner_two_cpus_matches_serial(tensile_deck, tmp_path):
    """The GUI's 2-CPU run of tensile_bar: Starter -np 2 writes the domain
    restarts, Engine -np 2 runs them (threads) and the global T01 carries
    the serial energies."""
    serial = R.JobRunner(tensile_deck, backend="numpy")
    res1 = serial.run_to_completion(timeout=300)
    assert res1["termination"] == ("ENGINE", "NORMAL", "")
    t01_serial = R.load_t01(res1["t01_path"])

    par = R.JobRunner(tensile_deck, backend="numpy", nspmd=2)
    res2 = par.run_to_completion(timeout=300)
    assert res2["returncode"] == 0
    assert res2["termination"] == ("ENGINE", "NORMAL", "")
    work = os.path.dirname(tensile_deck)
    assert os.path.exists(os.path.join(work, "TENSILE_0000_0001.rst"))
    assert os.path.exists(os.path.join(work, "TENSILE_0000_0002.rst"))
    t01_par = R.load_t01(res2["t01_path"])
    assert t01_par.nrows == t01_serial.nrows
    for chan in ("IE", "KE", "EW"):
        a, b = t01_serial.column(chan)[-1], t01_par.column(chan)[-1]
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12)


def test_widget_cpu_control(tensile_deck, tmp_path, monkeypatch):
    root = _make_root()
    try:
        from pyradioss.gui.app import PyradiossGUI
        # a remembered CPU count above this machine's is clamped
        cfg_path = str(tmp_path / "config.json")
        cfg = R.GuiConfig(path=cfg_path).load()
        cfg.set("nspmd", 10 ** 6)
        cfg.save()
        monkeypatch.setattr(R.GuiConfig, "default_path",
                            staticmethod(lambda: cfg_path))
        gui = PyradiossGUI(root, deck=tensile_deck)
        assert gui.max_cpus == R.available_cpus()
        assert gui.nspmd_var.get() == gui.max_cpus
        assert gui.cpu_count() == gui.max_cpus
        assert str(gui.cpu_spin.cget("to")) == str(gui.max_cpus)
        # validation: below 1 or above the machine's CPUs is rejected
        gui.nspmd_var.set(1)
        assert gui.cpu_count() == 1
        gui.nspmd_var.set(0)
        assert gui.cpu_count() is None
        gui.nspmd_var.set(gui.max_cpus + 1)
        assert gui.cpu_count() is None
    finally:
        root.destroy()
