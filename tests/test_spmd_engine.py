"""
End-to-end tests of the decomposed (SPMD) Engine on the REAL Starter
decomposition (``pyradioss-starter -np N`` -> pyradioss/spmd/domdec.py):

* serial vs 2 / 3 thread domains on the tensile bar (LAW2 bricks, /BCS,
  /IMPVEL, /TH/NODE + /TH/PART, /ANIM): the T-file and the final energies
  agree to round-off — the frontier sums of spmd_exch_a.F are formed in a
  different order than the serial assembly, nothing else differs;
* restart chaining (M6 contract) under SPMD: run 2 resumes every domain
  from its own ``RunName_0001_000p.rst`` and matches the serial chain; a
  SERIAL run 3 resumes from the gathered global ``RunName_0002.rst``;
* the per-domain listing option (PYRADIOSS_SPMD_LOG_ALL=1);
* the engine-side refusal of the options the decomposition does not
  support (/IMPL, /DT/AMS, /DYREL, /KEREL, /ADYREL, /EIG);
* ``mpirun -np 2 python -m pyradioss.engine`` (Mpi4pyComm) gives the same
  T-file, bit for bit, as the thread backend (skipped without mpirun).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.engine.engine import _energies, _refuse_spmd, run_engine
from pyradioss.spmd.driver import run_engine_spmd
from pyradioss.starter.starter import run_starter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "examples", "tensile_bar")

pytestmark = pytest.mark.skipif(
    "nspmd" not in inspect.signature(run_starter).parameters,
    reason="the Starter has no -np decomposition")


def _setup(d, t_end="0.04"):
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join(SRC, "TENSILE_0000.rad"), d)
    eng = open(os.path.join(SRC, "TENSILE_0001.rad")).read()
    eng = eng.replace("/RUN/TENSILE/1\n0.2", f"/RUN/TENSILE/1\n{t_end}")
    (d / "TENSILE_0001.rad").write_text(eng)
    return eng


def _next_run(d, eng, run, t_end):
    (d / f"TENSILE_{run:04d}.rad").write_text(
        eng.replace("/RUN/TENSILE/1\n", f"/RUN/TENSILE/{run}\n")
           .replace("\n0.04\n", f"\n{t_end}\n"))


def _t(d, run=1):
    return np.loadtxt(d / f"TENSILET{run:02d}.csv", delimiter=",",
                      skiprows=2, ndmin=2)


def _assert_th_close(ts, tp, tol=1e-8):
    assert ts.shape == tp.shape and ts.shape[0] >= 4
    scale = np.maximum(np.abs(ts).max(axis=0), 1e-6)
    rel = np.abs(ts - tp).max(axis=0) / scale
    rel[8] = 0.0                 # ERR% column: round-off noise (~1e-13 %)
    assert rel.max() < tol, rel
    assert np.abs(ts[:, 8] - tp[:, 8]).max() < 1e-8


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@pytest.mark.parametrize("nspmd", [2, 3])
def test_starter_np_then_engine_spmd_matches_serial(tmp_path, nspmd,
                                                    monkeypatch):
    monkeypatch.setenv("PYRADIOSS_SPMD_LOG_ALL", "1")
    ds, dp = tmp_path / "serial", tmp_path / "spmd"
    _setup(ds)
    _setup(dp)
    _quiet(run_starter, str(ds / "TENSILE_0000.rad"))
    ms = _quiet(run_engine, str(ds / "TENSILE_0001.rad"))
    _quiet(run_starter, str(dp / "TENSILE_0000.rad"), nspmd=nspmd)
    mp = _quiet(run_engine_spmd, str(dp / "TENSILE_0001.rad"), nspmd)
    _assert_th_close(_t(ds), _t(dp))
    es, ep = _energies(ms, ms.engine_state), _energies(mp, mp.engine_state)
    assert mp.engine_state.cycle == ms.engine_state.cycle
    assert mp.numnod == ms.numnod        # domain 0 returns the GLOBAL view
    for k in ("IE", "KE", "HE", "EW", "EN"):
        assert ep[k] == pytest.approx(es[k], rel=1e-8, abs=1e-12)
    np.testing.assert_allclose(mp.x, ms.x, rtol=0, atol=1e-12)
    for r in range(nspmd):
        assert (dp / f"TENSILE_0001_{r + 1:04d}.rst").is_file()
    assert (dp / "TENSILE_0001.rst").is_file()           # gathered global
    # the listing of P0 + (option) the other domains' listings
    assert (dp / "TENSILE_0001.out").is_file()
    for r in range(1, nspmd):
        lst = (dp / f"TENSILE_0001_{r + 1:04d}.out").read_text()
        assert "SPMD DOMAINS (NSPMD)" in lst


def test_spmd_restart_chain_matches_serial_chain(tmp_path):
    ds, dp = tmp_path / "serial", tmp_path / "spmd"
    eng = _setup(ds, "0.02")
    _setup(dp, "0.02")
    for d in (ds, dp):
        _next_run(d, eng.replace("\n0.02\n", "\n0.04\n"), 2, "0.04")
        _next_run(d, eng.replace("\n0.02\n", "\n0.04\n"), 3, "0.05")
    _quiet(run_starter, str(ds / "TENSILE_0000.rad"))
    _quiet(run_engine, str(ds / "TENSILE_0001.rad"))
    ms = _quiet(run_engine, str(ds / "TENSILE_0002.rad"))
    _quiet(run_starter, str(dp / "TENSILE_0000.rad"), nspmd=2)
    _quiet(run_engine_spmd, str(dp / "TENSILE_0001.rad"), 2)
    mp = _quiet(run_engine_spmd, str(dp / "TENSILE_0002.rad"), 2)
    assert (dp / "TENSILE_0002_0002.rst").is_file()
    _assert_th_close(_t(ds, 2), _t(dp, 2))
    es, ep = _energies(ms, ms.engine_state), _energies(mp, mp.engine_state)
    assert mp.engine_state.cycle == ms.engine_state.cycle
    for k in ("IE", "KE", "EW", "EN"):
        assert ep[k] == pytest.approx(es[k], rel=1e-8, abs=1e-12)
    # a SERIAL run 3 resumes from the gathered global restart of run 2
    ms3 = _quiet(run_engine, str(ds / "TENSILE_0003.rad"))
    mp3 = _quiet(run_engine, str(dp / "TENSILE_0003.rad"))
    _assert_th_close(_t(ds, 3), _t(dp, 3))
    assert mp3.engine_state.cycle == ms3.engine_state.cycle


def test_engine_spmd_refuses_unsupported_options():
    model = SimpleNamespace(eigen_modes={})
    _refuse_spmd(SimpleNamespace(), model)             # plain explicit: ok
    _refuse_spmd(SimpleNamespace(eig_off=[3]), model)  # bare /EIG/OFF: ok
    for flag, word in (("implicit", "/IMPL"), ("dt_ams", "/DT/AMS"),
                       ("dyrel_active", "/DYREL"), ("kerel_active", "/KEREL"),
                       ("adyrel_active", "/ADYREL"), ("impl_eigv", "/EIG")):
        with pytest.raises(RuntimeError, match=word):
            _refuse_spmd(SimpleNamespace(**{flag: True}), model)
    with pytest.raises(RuntimeError, match="/EIG"):
        _refuse_spmd(SimpleNamespace(), SimpleNamespace(eigen_modes={1: 0}))


def _have_mpi():
    if shutil.which("mpirun") is None:
        return False
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _have_mpi(), reason="mpirun or mpi4py not available")
def test_engine_under_mpirun_matches_thread_backend(tmp_path):
    dt, dm = tmp_path / "threads", tmp_path / "mpi"
    _setup(dt)
    _setup(dm)
    _quiet(run_starter, str(dt / "TENSILE_0000.rad"), nspmd=2)
    for f in os.listdir(dt):
        if f.endswith(".rst"):
            shutil.copy(dt / f, dm)
    _quiet(run_engine_spmd, str(dt / "TENSILE_0001.rad"), 2)
    env = dict(os.environ)
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(["mpirun", "-np", "2", sys.executable, "-m",
                        "pyradioss.engine", "-i", "TENSILE_0001.rad"],
                       cwd=dm, capture_output=True, text=True, timeout=600,
                       env=env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "(mpi)" in (dm / "TENSILE_0001.out").read_text()
    # same arithmetic, same exchange order: identical to the thread run
    np.testing.assert_array_equal(_t(dm), _t(dt))
    # the inipar.F check: 2 domain restarts, 3 MPI processes
    p = subprocess.run(["mpirun", "-np", "3", sys.executable, "-m",
                        "pyradioss.engine", "-i", "TENSILE_0001.rad"],
                       cwd=dm, capture_output=True, text=True, timeout=600,
                       env=env)
    assert p.returncode != 0
    assert "REQUIRED (number of .rst files) NSPMD = 2" in p.stdout + p.stderr
