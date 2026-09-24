"""
Tests of the engine-side SPMD layer (pyradioss/spmd/exchange.py and
pyradioss/spmd/driver.py) on SYNTHETIC decompositions built by hand — no
Starter decomposition is needed:

* ``exch_forces`` (spmd_exch_a.F): every holder of a frontier node ends
  with the global sum, bitwise identical on all holders even when three
  domains share a node (the deterministic rank-ordered accumulation);
* ``gather_nodal`` / ``gather_group_state`` / ``gather_model_view``
  (spmd_collect.F): domain 0 reconstructs the global arrays from the owned
  rows (WEIGHT == 1) and the element rows (elem_glob);
* ``glob_min`` (spmd_glob_min5.F): global dt with the critical element of
  the winning domain, stop flags, summed slots;
* ``reduce_state`` and the engine's ``_energies`` under SPMD: the partial
  ledgers and the weighted kinetic energy sum to the global values;
* the serial (no-op) context;
* the thread driver: the inipar.F restart-count check and the abort of the
  whole group when one domain raises.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.spmd.comm import ThreadComm
from pyradioss.spmd.exchange import (GlobMin, SpmdContext, crit_type_code,
                                     crit_type_name, serial_context)


# ------------------------------------------------------------------ helpers
def run_group(n, fn, timeout=60.0):
    comms = ThreadComm.group(n)
    out = [None] * n
    err = [None] * n

    def work(r):
        try:
            out[r] = fn(r, comms[r])
        except BaseException as exc:  # noqa: BLE001
            err[r] = exc
            try:
                comms[r].abort(2)
            except SystemExit:
                pass

    ths = [threading.Thread(target=work, args=(r,), daemon=True)
           for r in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout)
        assert not t.is_alive(), "domain group deadlocked"
    for e in err:
        if e is not None:
            raise e
    return out


def make_infos(holders, nglob, elem_glob=None, numel_glob=None):
    """Build the DomainInfo of every domain from ``holders`` = list of the
    GLOBAL node sets each domain holds (native nodes; main_proc = lowest
    holder).  Duck-typed stand-in of pyradioss.spmd.domdec.DomainInfo."""
    nspmd = len(holders)
    holders = [np.array(sorted(h), dtype=np.int64) for h in holders]
    main = np.full(nglob, -1, dtype=np.int64)
    for r in range(nspmd - 1, -1, -1):
        main[holders[r]] = r
    infos = []
    for r in range(nspmd):
        ng = holders[r]
        front = {}
        for q in range(nspmd):
            if q == r:
                continue
            shared = np.intersect1d(ng, holders[q])
            if len(shared):
                front[q] = np.searchsorted(ng, shared).astype(np.int64)
        infos.append(SimpleNamespace(
            nspmd=nspmd, ispmd=r, numnod_glob=nglob, nodglob=ng,
            main_proc=main[ng], weight=(main[ng] == r).astype(np.float64),
            frontier=front,
            elem_glob=(elem_glob[r] if elem_glob else {}),
            numel_glob=(numel_glob or {}), owned_interfaces=[],
            owned_monvols=[], global_rst=""))
    return infos


# two domains: global nodes 0..5, domain 0 holds 0..3, domain 1 holds 2..5
TWO = [{0, 1, 2, 3}, {2, 3, 4, 5}]
# three domains, node 3 on all of them, node 5 on domains 1 and 2
THREE = [{0, 1, 2, 3}, {2, 3, 4, 5}, {3, 5, 6}]


def test_synthetic_info_matches_contract():
    i0, i1 = make_infos(TWO, 6)
    np.testing.assert_array_equal(i0.weight, [1, 1, 1, 1])
    np.testing.assert_array_equal(i1.weight, [0, 0, 1, 1])
    np.testing.assert_array_equal(i0.frontier[1], [2, 3])
    np.testing.assert_array_equal(i1.frontier[0], [0, 1])


# ------------------------------------------------------------ exch_forces
@pytest.mark.parametrize("holders,nglob", [(TWO, 6), (THREE, 7)])
def test_exch_forces_gives_global_sum_on_every_holder(holders, nglob):
    infos = make_infos(holders, nglob)
    rng = np.random.default_rng(7)
    nsp = len(holders)
    # per-domain partial contributions (zero on nodes a domain lacks)
    parts = []
    for r in range(nsp):
        n = len(infos[r].nodglob)
        parts.append({
            "fint": rng.normal(size=(n, 3)), "fext": rng.normal(size=(n, 3)),
            "fcont": rng.normal(size=(n, 3)), "mint": rng.normal(size=(n, 3)),
            "stifn": rng.random(n), "stifr": rng.random(n)})
    truth = {k: np.zeros((nglob,) + parts[0][k].shape[1:]) for k in parts[0]}
    for r in range(nsp):
        for k, v in parts[r].items():
            truth[k][infos[r].nodglob] += v

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        assert sp.active and sp.neighbours == sorted(infos[r].frontier)
        a = {k: v.copy() for k, v in parts[r].items()}
        sp.exch_forces(a["fint"], a["fext"], a["fcont"], a["mint"],
                       a["stifn"], a["stifr"])
        return a
    res = run_group(nsp, fn)
    for r in range(nsp):
        for k in truth:
            np.testing.assert_allclose(res[r][k], truth[k][infos[r].nodglob],
                                       rtol=1e-13, atol=1e-13)
    # bitwise identical on every holder of every node
    for g in range(nglob):
        vals = [res[r]["fint"][np.searchsorted(infos[r].nodglob, g)]
                for r in range(nsp) if g in set(infos[r].nodglob)]
        assert all(v.tobytes() == vals[0].tobytes() for v in vals)


def test_exch_forces_rank_ordered_sum_is_deterministic():
    """Node 3 is shared by three domains with values that expose
    non-associativity: (1e16 + 1) - 1e16 != 1e16 + (1 - 1e16).  Every
    holder must obtain the SAME bits, namely the ascending-rank sum."""
    infos = make_infos(THREE, 7)
    contrib = [1e16, 1.0, -1e16]              # domain r's value at node 3
    expected = (contrib[0] + contrib[1]) + contrib[2]

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        f = np.zeros((len(infos[r].nodglob), 3))
        f[np.searchsorted(infos[r].nodglob, 3), 0] = contrib[r]
        sp.exch_forces(f, None, None, None)
        return f[np.searchsorted(infos[r].nodglob, 3), 0]
    res = run_group(3, fn)
    assert res[0] == res[1] == res[2] == expected


def test_exch_forces_skips_none_and_1d_only():
    infos = make_infos(TWO, 6)

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        st = np.full(4, float(r + 1))
        sp.exch_forces(None, stifn=st)
        return st
    res = run_group(2, fn)
    np.testing.assert_array_equal(res[0], [1, 1, 3, 3])
    np.testing.assert_array_equal(res[1], [3, 3, 2, 2])


# ------------------------------------------------------------- gathers
def test_gather_nodal_reconstructs_global_array_on_root():
    infos = make_infos(THREE, 7)
    truth = np.arange(21, dtype=np.float64).reshape(7, 3) * 1.5

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        return sp.gather_nodal(truth[infos[r].nodglob])
    res = run_group(3, fn)
    np.testing.assert_array_equal(res[0], truth)
    assert res[1] is None and res[2] is None


class _Group:
    def __init__(self, conn, state):
        self.conn = conn
        self.state = state


class _Model:
    """Minimal model: nodal arrays + element groups (by name)."""

    def __init__(self, groups, **arrays):
        self._groups = groups
        for k, v in arrays.items():
            setattr(self, k, v)

    def element_groups(self):
        return list(self._groups.items())


def test_gather_group_state_and_model_view():
    # global: 4 bricks; domain 0 owns rows 0 and 3, domain 1 rows 1 and 2
    elem_glob = [{"bricks": np.array([0, 3])}, {"bricks": np.array([1, 2])}]
    infos = make_infos(TWO, 6, elem_glob=elem_glob,
                       numel_glob={"bricks": 4})
    eint_g = np.array([10.0, 11.0, 12.0, 13.0])
    sig_g = np.arange(24.0).reshape(4, 6)
    x_g = np.arange(18.0).reshape(6, 3)
    mint_g = -x_g

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        rows = elem_glob[r]["bricks"]
        ng = infos[r].nodglob
        st = {"eint": eint_g[rows].copy(), "sig": sig_g[rows].copy(),
              "mat_extra": {"epsp36": eint_g[rows] * 2.0},
              "color_indices": np.array([1, 0]), "has_x": True}
        model = _Model({"bricks": _Group(np.zeros((2, 8), int), st)},
                       x=x_g[ng].copy(), v=x_g[ng] * 0.5,
                       mass=np.ones(len(ng)))
        model.rigid_bodies = {}
        gmodel = None
        if r == 0:
            gmodel = _Model({"bricks": _Group(
                np.zeros((4, 8), int), {"eint": np.zeros(4)})},
                x=np.zeros((6, 3)))
        got = sp.gather_group_state("bricks", st["eint"])
        ex = sp.gather_model_view(model, gmodel,
                                  extra_nodal={"mint": mint_g[ng]})
        return got, ex, gmodel
    res = run_group(2, fn)
    got, ex, gm = res[0]
    np.testing.assert_array_equal(got, eint_g)
    assert res[1][0] is None and res[1][1] is None
    np.testing.assert_array_equal(gm.x, x_g)
    np.testing.assert_array_equal(gm.v, x_g * 0.5)
    np.testing.assert_array_equal(ex["mint"], mint_g)
    gst = gm.element_groups()[0][1].state
    np.testing.assert_array_equal(gst["eint"], eint_g)
    np.testing.assert_array_equal(gst["sig"], sig_g)
    np.testing.assert_array_equal(gst["mat_extra"]["epsp36"], eint_g * 2.0)
    assert "color_indices" not in gst         # index tables never scattered
    assert gm.rigid_bodies == {}


# ------------------------------------------------------------- glob_min
def test_glob_min_global_dt_crit_element_stop_and_sums():
    infos = make_infos(THREE, 7)
    dts = [3.0e-6, 1.0e-6, 2.0e-6]
    crit = [("bricks", 11), ("shells", 22), ("NODE", 33)]

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        return sp.glob_min(dts[r], crit[r][0], crit[r][1], r == 2,
                           0.5 * (r + 1), 1.0, timet=(r == 0))
    res = run_group(3, fn)
    for g in res:
        assert isinstance(g, GlobMin)
        assert g.dt == 1.0e-6
        assert (g.crit_type, g.crit_id) == ("shells", 22)
        assert g.stop is True and g.timet is True
        assert g.sums == (3.0, 3.0)


def test_crit_type_table_roundtrip():
    for name in ("", "NODE", "bricks", "shells"):
        assert crit_type_name(crit_type_code(name)) == name
    assert crit_type_code("no-such-group") == 0
    assert crit_type_name(10 ** 6) == ""


# ------------------------------------------------ ledgers and energies
def test_reduce_state_and_spmd_energies():
    from pyradioss.engine.engine import EngineState, _energies
    infos = make_infos(TWO, 6)
    mass_g = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v_g = np.arange(18.0).reshape(6, 3) * 0.1
    eints = [np.array([1.0, 2.0]), np.array([3.0])]
    ledgers = [dict(wext=1.0, econt=0.25, e_num=-0.1, e_damp=0.5,
                    e_madd=0.0), dict(wext=2.0, econt=0.75, e_num=0.3,
                                      e_damp=0.25, e_madd=0.5)]

    def fn(r, comm):
        sp = SpmdContext(comm, infos[r])
        ng = infos[r].nodglob
        st = {"eint": eints[r], "ehour": np.zeros(len(eints[r]))}
        model = _Model({"bricks": _Group(None, st)}, mass=mass_g[ng].copy(),
                       v=v_g[ng].copy(), inertia=None, vr=None)
        state = EngineState()
        for k, val in ledgers[r].items():
            setattr(state, k, val)
        state.e0 = 7.0
        state.spmd = sp
        e = _energies(model, state, extra=[float(r + 1)])
        red = sp.reduce_state(state)
        return e, red, state
    res = run_group(2, fn)
    ke_true = 0.5 * float((mass_g[:, None] * v_g ** 2).sum())
    for e, red, st in res:
        assert e["IE"] == pytest.approx(6.0)
        assert e["KE"] == pytest.approx(ke_true, rel=1e-14)
        assert e["CE"] == pytest.approx(1.0)
        assert e["EN"] == pytest.approx(0.2)
        assert e["DE"] == pytest.approx(0.75)
        assert e["EW"] == pytest.approx(3.0)
        assert float(e["EXTRA"][0]) == 3.0
        total = e["IE"] + e["KE"] + e["HE"] + e["CE"] + e["EN"] + e["DE"]
        ref = max(abs(e["EW"]), e["KE"], e["IE"], st.epeak, 7.0, 1e-12)
        assert e["ERR"] == pytest.approx((total - 3.0 - 0.5 - 7.0) / ref * 100)
        assert red.wext == pytest.approx(3.0) and red.e_madd == 0.5
        assert red.spmd is None and st.spmd is not None
    assert res[0][0]["ERR"] == res[1][0]["ERR"]   # identical decisions


# --------------------------------------------------------- serial context
def test_serial_context_is_a_no_op():
    sp = serial_context()
    assert not sp.active and sp.weight is None and sp.is_root
    f = np.arange(6.0).reshape(2, 3)
    before = f.copy()
    sp.exch_forces(f, f, f, f, np.ones(2), np.ones(2))
    np.testing.assert_array_equal(f, before)
    g = sp.glob_min(1.5e-6, "bricks", 4, True, 2.0)
    assert g == GlobMin(1.5e-6, "bricks", 4, True, False, (2.0,))
    assert sp.sum(3.0) == 3.0 and sp.any_stop(False) is False
    assert sp.gather_nodal(f) is f
    assert sp.bcast("s") == "s" and sp.allgather(1) == [1]
    st = SimpleNamespace(wext=1.0, spmd=None)
    red = sp.reduce_state(st)
    assert red is not st and red.wext == 1.0
    # a size-1 communicator or a missing DomainInfo is serial as well
    assert not SpmdContext(ThreadComm.group(1)[0], None).active


def test_context_rejects_mismatched_rank():
    infos = make_infos(TWO, 6)

    def fn(r, comm):
        with pytest.raises(RuntimeError):
            SpmdContext(comm, infos[1 - r])
        return True
    assert run_group(2, fn) == [True, True]


# ---------------------------------------------------------------- driver
def _touch_domain_restarts(tmp_path, n, run="RUN"):
    for p in range(n):
        (tmp_path / f"{run}_0000_{p + 1:04d}.rst").write_bytes(b"")
    inp = tmp_path / f"{run}_0001.rad"
    inp.write_text("#RADIOSS ENGINE\n")
    return str(inp)


def test_driver_counts_restarts_and_checks_nspmd(tmp_path):
    from pyradioss.spmd.driver import count_domain_restarts, run_engine_spmd
    inp = _touch_domain_restarts(tmp_path, 3)
    assert count_domain_restarts(inp) == 3
    with pytest.raises(RuntimeError, match="REQUIRED \\(number of .rst files\\) "
                                           "NSPMD = 3"):
        run_engine_spmd(inp, 2)


def test_driver_aborts_group_on_domain_failure(tmp_path, monkeypatch):
    """Domain 1 raises before sending; domain 0 blocks in the frontier
    receive and domain 2 in a collective: the group is aborted (no
    deadlock) and the ORIGINAL exception reaches the caller."""
    import pyradioss.engine.engine as eng
    from pyradioss.spmd.driver import run_engine_spmd
    from pyradioss.spmd.exchange import MSGOFF_EXCH_A
    inp = _touch_domain_restarts(tmp_path, 3)

    def fake_run_engine(input_file, log=None, comm=None):
        if comm.rank == 1:
            raise ValueError("domain 1 exploded")
        if comm.rank == 0:
            comm.irecv(np.empty((2, 3)), 1, MSGOFF_EXCH_A).wait()
        else:
            comm.barrier()
        return "unreachable"
    monkeypatch.setattr(eng, "run_engine", fake_run_engine)
    with pytest.raises(ValueError, match="domain 1 exploded"):
        run_engine_spmd(inp, 3)


def test_driver_returns_rank0_result(tmp_path, monkeypatch):
    import pyradioss.engine.engine as eng
    from pyradioss.spmd.driver import run_engine_spmd
    inp = _touch_domain_restarts(tmp_path, 2)

    def fake_run_engine(input_file, log=None, comm=None):
        tot = comm.allreduce_scalar(comm.rank + 1.0)
        return ("model", comm.rank, tot, log is None)
    monkeypatch.setattr(eng, "run_engine", fake_run_engine)
    # domain 0 keeps the caller's log (None here), others get None too
    assert run_engine_spmd(inp, 2) == ("model", 0, 3.0, True)


def test_group_payload_copies_state():
    """The thread backend hands gathered objects over BY REFERENCE and the
    sender keeps integrating in place: every payload array must be a copy
    (regression — a live view made the rank-0 T-file read the sender's
    next-cycle element energies)."""
    st = {"eint": np.arange(3.0), "mat_extra": {"a": np.ones(3)},
          "color_indices": np.arange(3)}
    pay = SpmdContext._group_payload(_Group(None, st), 3)
    assert not np.shares_memory(pay["eint"], st["eint"])
    assert not np.shares_memory(pay["mat_extra"]["a"], st["mat_extra"]["a"])
    assert "color_indices" not in pay


# ------------------------------------------ engine end to end (engine side)
def _slice_domains(rst0, nspmd):
    """A MINIMAL element decomposition of a model made only of element
    groups + node groups (the tensile bar): slabs of element centroids
    along x, native nodes, main_proc = lowest holder, the frontier tables
    of the contract — enough to drive the engine side without the Starter
    decomposition (pyradioss/spmd/domdec.py covers the general case)."""
    import copy
    from pyradioss.engine.coloring import compute_element_colors
    from pyradioss.starter.initialization import _subset_element_group
    from pyradioss.starter.restart import read_restart, write_restart
    gm, _ = read_restart(rst0)
    ng = gm.numnod
    groups = list(gm.element_groups())
    cents = [gm.x0[g.conn].mean(axis=1)[:, 0] for _, g in groups]
    qs = np.quantile(np.concatenate(cents), np.linspace(0, 1, nspmd + 1)[1:-1])
    ranks = [np.searchsorted(qs, c, side="right") for c in cents]
    holders = []
    for r in range(nspmd):
        s = set()
        for (_, g), rk in zip(groups, ranks):
            s.update(np.unique(g.conn[rk == r]).tolist())
        holders.append(s)
    infos = make_infos(holders, ng)
    base = rst0[:-len(".rst")]
    for r, info in enumerate(infos):
        loc = info.nodglob
        g2l = np.full(ng, -1, dtype=np.int64)
        g2l[loc] = np.arange(len(loc))
        m = copy.copy(gm)
        m.__dict__ = dict(gm.__dict__)
        for k, v in gm.__dict__.items():
            if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == ng:
                setattr(m, k, v[loc].copy())
        m._id2idx = {int(nid): i for i, nid in enumerate(m.node_ids)}
        info.elem_glob, info.numel_glob = {}, {}
        for (name, g), rk in zip(groups, ranks):
            mask = rk == r
            sub = _subset_element_group(g, mask)
            for key, val in g.state.items():
                if key in ("slices", "part_ids"):
                    continue
                if isinstance(val, np.ndarray) and val.ndim and \
                        val.shape[0] == g.n:
                    sub.state[key] = val[mask].copy()
                elif isinstance(val, dict):
                    sub.state[key] = {
                        k2: (v2[mask].copy() if isinstance(v2, np.ndarray)
                             and v2.ndim and v2.shape[0] == g.n else v2)
                        for k2, v2 in val.items()}
                else:
                    sub.state[key] = val
            sub.conn = g2l[sub.conn]
            ci, co = compute_element_colors(sub.conn, len(loc))
            sub.state["color_indices"], sub.state["color_offsets"] = ci, co
            setattr(m, name, sub)
            info.elem_glob[name] = np.nonzero(mask)[0].astype(np.int64)
            info.numel_glob[name] = g.n
        ngr = {}
        for gid, grp in gm.node_groups.items():
            gg = copy.copy(grp)
            if grp.node_idx is not None:
                li = g2l[grp.node_idx]
                gg.node_idx = li[li >= 0]
            ngr[gid] = gg
        m.node_groups = ngr
        info.global_rst = rst0
        m.spmd = info
        write_restart(m, f"{base}_{r + 1:04d}.rst")


def test_engine_spmd_two_domains_matches_serial(tmp_path):
    """Tensile bar (LAW2 bricks, /BCS + /IMPVEL, /TH/NODE + /TH/PART, ANIM)
    run serially and as 2 thread domains of a hand-made decomposition: the
    T-file and the final energies agree to round-off (the frontier sums
    are formed in a different order than the serial assembly)."""
    import contextlib
    import io
    import os
    import shutil
    from pyradioss.engine.engine import _energies, run_engine
    from pyradioss.spmd.driver import run_engine_spmd
    from pyradioss.starter.starter import run_starter
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples", "tensile_bar")
    runs = {}
    for mode in ("serial", "spmd"):
        d = tmp_path / mode
        d.mkdir()
        shutil.copy(os.path.join(src, "TENSILE_0000.rad"), d)
        eng = open(os.path.join(src, "TENSILE_0001.rad")).read()
        eng = eng.replace("/RUN/TENSILE/1\n0.2", "/RUN/TENSILE/1\n0.04")
        (d / "TENSILE_0001.rad").write_text(eng)
        with contextlib.redirect_stdout(io.StringIO()):
            run_starter(str(d / "TENSILE_0000.rad"))
            if mode == "serial":
                model = run_engine(str(d / "TENSILE_0001.rad"))
            else:
                _slice_domains(str(d / "TENSILE_0000.rst"), 2)
                model = run_engine_spmd(str(d / "TENSILE_0001.rad"), 2)
        runs[mode] = (model, np.loadtxt(d / "TENSILET01.csv", delimiter=",",
                                        skiprows=2), d)
    (ms, ts, _), (mp, tp, dp) = runs["serial"], runs["spmd"]
    assert ts.shape == tp.shape and ts.shape[0] > 5
    scale = np.maximum(np.abs(ts).max(axis=0), 1e-6)
    rel = np.abs(ts - tp).max(axis=0) / scale
    rel[8] = 0.0                  # ERR% column: round-off noise (~1e-13 %)
    assert rel.max() < 1e-8
    assert np.abs(ts[:, 8] - tp[:, 8]).max() < 1e-8
    es = _energies(ms, ms.engine_state)
    ep = _energies(mp, mp.engine_state)      # the gathered global view
    assert mp.engine_state.cycle == ms.engine_state.cycle
    for k in ("IE", "KE", "HE", "EW"):
        assert ep[k] == pytest.approx(es[k], rel=1e-9, abs=1e-14)
    # per-domain + global restarts, one listing, the animation frames
    for f in ("TENSILE_0001.rst", "TENSILE_0001_0001.rst",
              "TENSILE_0001_0002.rst", "TENSILE_0001.out", "TENSILEA000.vtk"):
        assert (dp / f).is_file(), f
    assert not (dp / "TENSILE_0001_0002.out").exists()
