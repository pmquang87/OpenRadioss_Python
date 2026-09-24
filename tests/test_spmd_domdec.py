"""SPMD domain decomposition — the Starter side (pyradioss/spmd/domdec.py).

Checks the element-cut invariants (domdec1.F), the frontier / MAIN_PROC /
WEIGHT tables (w_front.F, w_master_proc_weight.F), the per-domain restart
files (ddsplit.F naming) and the entity ownership / replication rules
(domdec2.F) including the contact ghost ring.
"""

import contextlib
import glob
import io
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import (GJoint, MonitoredVolume,
                                      PressureLoad, RigidWall, Sensor)
from pyradioss.model.model import ElementGroup, Model
from pyradioss.spmd.domdec import (DomainInfo, check_spmd_support, decompose,
                                   element_weights, slice_model)
from pyradioss.starter.restart import read_restart
from pyradioss.starter.starter import run_starter

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples")


def _copy_deck(tmp_path, name):
    src = os.path.join(EXAMPLES, name)
    dst = tmp_path / name
    dst.mkdir()
    for f in glob.glob(os.path.join(src, "*.rad")):
        shutil.copy(f, dst)
    return str(next(p for p in dst.iterdir() if p.name.endswith("_0000.rad")))


def _starter(deck, nspmd=1):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(deck, MessageLog(), nspmd=nspmd)


def _load_domains(deck, nspmd):
    base = deck[: -len("_0000.rad")]
    out = []
    for p in range(nspmd):
        path = f"{base}_0000_{p + 1:04d}.rst"
        assert os.path.isfile(path), path
        lm, eng = read_restart(path)
        assert eng is None
        out.append(lm)
    return out


def _native_global(lm):
    """Global indices of the nodes of the local elements of a domain."""
    nodes = [np.asarray(g.conn).reshape(-1) for _, g in lm.element_groups()]
    nodes = np.concatenate(nodes) if nodes else np.zeros(0, np.int64)
    nodes = np.unique(nodes[nodes >= 0])
    return lm.spmd.nodglob[nodes]


def _check_invariants(gm, locs):
    """(a) every element on exactly one domain; per node the weights sum
    to one and MAIN_PROC is the lowest native holder; frontier lists are
    symmetric and identically ordered; NODGLOB sorted; local connectivity
    consistent with the global one."""
    nspmd = len(locs)
    numnod = gm.numnod
    for p, lm in enumerate(locs):
        sp = lm.spmd
        assert isinstance(sp, DomainInfo)
        assert sp.nspmd == nspmd and sp.ispmd == p
        assert sp.numnod_glob == numnod
        assert np.all(np.diff(sp.nodglob) > 0)
        assert lm.numnod == len(sp.nodglob)
        assert np.array_equal(lm.node_ids, gm.node_ids[sp.nodglob])
        assert np.array_equal(lm.x0, gm.x0[sp.nodglob])
        assert np.array_equal(lm.mass, gm.mass[sp.nodglob])
        assert all(lm.node_index(int(u)) == i for i, u in enumerate(lm.node_ids))
        assert np.array_equal(sp.weight, (sp.main_proc == p).astype(float))

    # each global element exactly once, with consistent connectivity
    for name, g in gm.element_groups():
        rows = np.concatenate([lm.spmd.elem_glob[name] for lm in locs])
        assert np.array_equal(np.sort(rows), np.arange(g.n))
        for lm in locs:
            assert lm.spmd.numel_glob[name] == g.n
            lg = getattr(lm, name)
            er = lm.spmd.elem_glob[name]
            assert lg.n == len(er)
            assert np.array_equal(lg.ids, g.ids[er])
            if lg.n:
                assert np.array_equal(lm.x0[lg.conn], gm.x0[g.conn[er]])
                assert np.array_equal(lm.spmd.nodglob[lg.conn], g.conn[er])
                assert sum(sl.stop - sl.start for sl, _, _ in lg.state["slices"]) == lg.n

    # node ownership
    wsum = np.zeros(numnod)
    holders = [[] for _ in range(numnod)]
    natives = [[] for _ in range(numnod)]
    for p, lm in enumerate(locs):
        np.add.at(wsum, lm.spmd.nodglob, lm.spmd.weight)
        for n in lm.spmd.nodglob:
            holders[n].append(p)
        for n in _native_global(lm):
            natives[n].append(p)
    assert np.allclose(wsum, 1.0)
    for n in range(numnod):
        expect = min(natives[n]) if natives[n] else min(holders[n])
        for p in holders[n]:
            lm = locs[p]
            i = int(np.searchsorted(lm.spmd.nodglob, n))
            assert lm.spmd.main_proc[i] == expect

    # frontier: symmetric, same global order on both sides, complete
    for p, lp in enumerate(locs):
        for q, lq in enumerate(locs):
            if p == q:
                continue
            shared = np.intersect1d(lp.spmd.nodglob, lq.spmd.nodglob)
            if len(shared) == 0:
                assert q not in lp.spmd.frontier
                continue
            fp = lp.spmd.nodglob[lp.spmd.frontier[q]]
            fq = lq.spmd.nodglob[lq.spmd.frontier[p]]
            assert np.array_equal(fp, fq)
            assert np.array_equal(fp, shared)


# ---------------------------------------------------------------------------
# (a) + (b) tensile bar, 2 and 4 domains, through run_starter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nspmd", [2, 4])
def test_tensile_bar_partition_invariants(tmp_path, nspmd):
    deck = _copy_deck(tmp_path, "tensile_bar")
    gm = _starter(deck, nspmd=nspmd)
    locs = _load_domains(deck, nspmd)
    _check_invariants(gm, locs)
    # every domain gets elements and the load is balanced
    counts = [lm.bricks.n for lm in locs]
    assert min(counts) > 0 and sum(counts) == gm.bricks.n
    # (b) the global restart is still there and is what DomainInfo names
    base = deck[: -len("_0000.rad")]
    assert os.path.isfile(base + "_0000.rst")
    for lm in locs:
        assert lm.spmd.global_rst == os.path.abspath(base + "_0000.rst")
        assert not lm.spmd.owned_interfaces and not lm.spmd.owned_monvols
        assert lm.spmd_ghost == {}
    listing = open(base + "_0000.out").read()
    assert "SPMD DOMAIN DECOMPOSITION" in listing
    assert "STARTER TERMINATION : NORMAL" in listing


def test_serial_starter_writes_no_domain_restart(tmp_path):
    deck = _copy_deck(tmp_path, "tensile_bar")
    gm = _starter(deck)
    assert getattr(gm, "spmd", None) is None
    base = deck[: -len("_0000.rad")]
    assert os.path.isfile(base + "_0000.rst")
    assert not glob.glob(base + "_0000_*.rst")
    assert "SPMD DOMAIN DECOMPOSITION" not in open(base + "_0000.out").read()


def test_cli_np_writes_domain_restarts(tmp_path):
    from pyradioss.starter.__main__ import main
    deck = _copy_deck(tmp_path, "tensile_bar")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["-i", deck, "-np", "3"]) == 0
    base = deck[: -len("_0000.rad")]
    assert sorted(os.path.basename(p) for p in glob.glob(base + "_0000_*.rst")) == [
        "TENSILE_0000_0001.rst", "TENSILE_0000_0002.rst", "TENSILE_0000_0003.rst"]


def test_element_weights_tables(tmp_path):
    deck = _copy_deck(tmp_path, "tensile_bar")
    gm = _starter(deck)
    w = element_weights(gm)
    # LAW-dependent (SOLTELT(1) + SOL1TNL(law,1)) / TPSREF, all > 0
    assert set(w) == {"bricks"}
    assert np.all(w["bricks"] > 1.0) and np.all(w["bricks"] < 2.0)


# ---------------------------------------------------------------------------
# (c) synthetic: two trusses sharing one node
# ---------------------------------------------------------------------------

def _two_truss_model():
    m = Model()
    m.node_ids = np.array([1, 2, 3], dtype=np.int64)
    m._id2idx = {1: 0, 2: 1, 3: 2}
    m.x0 = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    m.x = m.x0.copy()
    m.v = np.zeros((3, 3))
    m.vr = np.zeros((3, 3))
    m.mass = np.array([1.0, 2.0, 1.0])
    m.mass0 = m.mass.copy()
    m.inertia = np.zeros(3)
    mat = SimpleNamespace(id=1, law=1, E=1.0, params={})
    m.trusses = ElementGroup(
        ids=np.array([10, 11], dtype=np.int64),
        conn=np.array([[0, 1], [1, 2]], dtype=np.int64),
        part=np.zeros(2, dtype=np.int64),
        state={"slices": [(slice(0, 2), mat, None)],
               "part_ids": np.zeros(2, dtype=np.int64),
               "off": np.ones(2), "L0": np.ones(2)})
    return m


def test_two_elements_shared_node():
    m = _two_truss_model()
    dec = decompose(m, 2, MessageLog())
    assert list(dec.elem_domain["trusses"]) == [0, 1]
    locs = [slice_model(m, dec, p, MessageLog()) for p in range(2)]
    l0, l1 = locs
    # node 2 (global index 1) is the frontier node, on both domains
    assert list(l0.spmd.nodglob) == [0, 1]
    assert list(l1.spmd.nodglob) == [1, 2]
    assert list(l0.spmd.nodglob[l0.spmd.frontier[1]]) == [1]
    assert list(l1.spmd.nodglob[l1.spmd.frontier[0]]) == [1]
    # MAIN_PROC = lowest native holder: weight 1 on rank 0, 0 on rank 1
    assert list(l0.spmd.main_proc) == [0, 0] and list(l0.spmd.weight) == [1.0, 1.0]
    assert list(l1.spmd.main_proc) == [0, 1] and list(l1.spmd.weight) == [0.0, 1.0]
    # local connectivity, element buffers and rows
    assert l0.trusses.conn.tolist() == [[0, 1]]
    assert l1.trusses.conn.tolist() == [[0, 1]]
    assert l1.trusses.ids.tolist() == [11]
    assert l1.spmd.elem_glob["trusses"].tolist() == [1]
    assert l1.trusses.state["slices"][0][0] == slice(0, 1)
    assert l1.trusses.state["L0"].shape == (1,)
    assert l1.mass.tolist() == [2.0, 1.0]
    assert l1.node_index(2) == 0 and l1.node_index(3) == 1
    _check_invariants(m, locs)
    # the global model is untouched
    assert m.numnod == 3 and m.trusses.n == 2 and getattr(m, "spmd", None) is None


# ---------------------------------------------------------------------------
# (d) entity rules: contact + rigid body + rigid walls + pressure load
# ---------------------------------------------------------------------------

def _dense(lm, gidx):
    """Local indices of global nodes (asserting they are all local)."""
    loc = np.searchsorted(lm.spmd.nodglob, gidx)
    assert np.all(loc < len(lm.spmd.nodglob))
    assert np.array_equal(lm.spmd.nodglob[loc], gidx)
    return loc


def test_contact_rbody_rwall_pload_rules(tmp_path):
    from pyradioss.contact import ContactType7
    from pyradioss.spmd.domdec import _interface_nodes

    deck = _copy_deck(tmp_path, "rigid_impactor")
    gm = _starter(deck)
    nspmd = 3
    itf = gm.interfaces[0]
    # add a fixed wall (all nodes), a moving wall on a node group and a
    # pressure load on the contact surface
    grp_id = next(g for g in sorted(gm.node_groups) if len(gm.node_groups[g].node_idx) > 3)
    cand = np.asarray(gm.node_groups[grp_id].node_idx)
    carrier_uid = int(gm.node_ids[0])
    gm.rwalls.append(RigidWall(id=1, point=np.zeros(3), normal=np.array([0, 0, 1.0])))
    gm.rwalls.append(RigidWall(id=2, point=np.zeros(3), normal=np.array([0, 0, 1.0]),
                               grnod_id=grp_id, node_id=carrier_uid))
    gm.ploads.append(PressureLoad(id=7, surf_id=itf.surf_id, funct_id=1))
    gm.monitored_volumes[5] = MonitoredVolume(id=5, vol_type="AIRBAG1",
                                              surf_id=itf.surf_id)
    sensor_uid = int(gm.node_ids[-1])
    gm.sensors.append(Sensor(id=3, kind="DISP", node_id=sensor_uid, dmin=1.0))
    check_spmd_support(gm)

    dec = decompose(gm, nspmd, MessageLog())
    locs = [slice_model(gm, dec, p, MessageLog(), global_rst="x.rst") for p in range(nspmd)]
    _check_invariants(gm, locs)

    # penalty contact: kept on exactly one rank, which holds every node
    owners = [p for p, lm in enumerate(locs) if lm.interfaces]
    assert len(owners) == 1
    ow = owners[0]
    lo = locs[ow]
    assert lo.spmd.owned_interfaces == [itf.id]
    inodes = _interface_nodes(gm, itf)
    native_holders = [p for p, lm in enumerate(locs)
                      if np.intersect1d(_native_global(lm), inodes).size]
    assert ow == min(native_holders)
    _dense(lo, inodes)
    for p, lm in enumerate(locs):
        if p != ow:
            assert lm.interfaces == [] and lm.spmd.owned_interfaces == []
    # the ghost ring: every non-local element touching an interface node
    assert lo.spmd_ghost
    for name, g in gm.element_groups():
        touch = np.isin(g.conn, inodes).any(axis=1)
        nonlocal_ = np.ones(g.n, dtype=bool)
        nonlocal_[lo.spmd.elem_glob[name]] = False
        expect = np.flatnonzero(touch & nonlocal_)
        gh = lo.spmd_ghost.get(name)
        got = np.zeros(0, np.int64) if gh is None else \
            np.array([int(np.flatnonzero(g.ids == i)[0]) for i in gh.ids])
        assert np.array_equal(np.sort(got), expect)
        if gh is not None:
            assert np.array_equal(lo.x0[gh.conn], gm.x0[g.conn[got]])
    # contact surface: full copy on the owner, provenance local or ghost
    gs, ls = gm.surfaces[itf.surf_id], lo.surfaces[itf.surf_id]
    assert np.array_equal(lo.spmd.nodglob[ls.segments], gs.segments)
    for k in range(len(gs.segments)):
        gname, erow = str(gs.seg_gtype[k]), int(gs.seg_elem[k])
        lname, lrow = str(ls.seg_gtype[k]), int(ls.seg_elem[k])
        if gname == "":
            assert lname == "" and lrow == -1
            continue
        gid = getattr(gm, gname).ids[erow]
        if lname.startswith("ghost:"):
            assert lo.spmd_ghost[gname].ids[lrow] == gid
        else:
            assert lname == gname and getattr(lo, gname).ids[lrow] == gid
    # stiffness / gap on the owner identical to the serial ones
    cg = ContactType7(itf, gm, MessageLog())
    cl = ContactType7(lo.interfaces[0], lo, MessageLog())
    assert np.array_equal(lo.spmd.nodglob[cl.segs], cg.segs)
    assert np.allclose(cl.Km, cg.Km, rtol=0, atol=0)
    assert np.array_equal(lo.spmd.nodglob[cl.nodes], cg.nodes)
    assert np.allclose(cl.Ks, cg.Ks, rtol=0, atol=0)

    # rigid body: replicated on every rank holding any body node, with
    # the master and all slaves local there
    rb = gm.rbodies[0]
    body = np.unique(np.r_[rb.slaves, [rb.master]])
    for p, lm in enumerate(locs):
        holds = np.intersect1d(lm.spmd.nodglob, body).size > 0
        assert (len(lm.rbodies) == 1) == holds
        if holds:
            lrb = lm.rbodies[0]
            assert lm.spmd.nodglob[lrb.master] == rb.master
            assert np.array_equal(lm.spmd.nodglob[lrb.slaves], rb.slaves)

    # fixed wall on every rank; moving wall on every holder of a
    # candidate / the carrier, all candidates + carrier local there
    carrier = gm.node_index(carrier_uid)
    for p, lm in enumerate(locs):
        ids = [w.id for w in lm.rwalls]
        assert 1 in ids
        holds = np.intersect1d(_native_global(lm), np.r_[cand, carrier]).size > 0
        if holds:
            assert 2 in ids
        if 2 in ids:
            _dense(lm, np.unique(np.r_[cand, carrier]))
            assert len(lm.node_groups[grp_id].node_idx) == len(cand)
        # node groups keep their order, restricted to local nodes
        for gid, g in gm.node_groups.items():
            li = lm.node_groups[gid].node_idx
            gl = lm.spmd.nodglob[li]
            assert np.array_equal(gl, [n for n in g.node_idx if n in set(lm.spmd.nodglob)])

    # monitored volume: kept on the lowest native holder of its surface,
    # every surface node local there
    snodes = np.unique(gm.surfaces[itf.surf_id].segments)
    mv_owner = min(p for p, lm in enumerate(locs)
                   if np.intersect1d(_native_global(lm), snodes).size)
    for p, lm in enumerate(locs):
        assert (5 in lm.monitored_volumes) == (p == mv_owner)
        assert lm.spmd.owned_monvols == ([5] if p == mv_owner else [])
    _dense(locs[mv_owner], snodes)
    # /SENSOR/DISP node on every rank
    for lm in locs:
        lm.node_index(sensor_uid)

    # pressure load: every segment owned by exactly one rank, on a
    # private surface whose corners are local
    seen = []
    for p, lm in enumerate(locs):
        for pl in lm.ploads:
            assert pl.id == 7 and pl.surf_id != itf.surf_id
            assert pl.surf_id not in gm.surfaces
            ps = lm.surfaces[pl.surf_id]
            seen.append(lm.spmd.nodglob[ps.segments])
            for k in range(len(ps.segments)):
                if ps.seg_gtype[k] != "":
                    assert not str(ps.seg_gtype[k]).startswith("ghost:")
    allsegs = np.concatenate(seen)
    assert len(allsegs) == len(gs.segments)
    assert sorted(map(tuple, allsegs)) == sorted(map(tuple, gs.segments))


def test_tied_interface_replicated(tmp_path):
    from pyradioss.contact import ContactType2
    from pyradioss.spmd.domdec import _interface_nodes
    deck = _copy_deck(tmp_path, "spot_weld")
    gm = _starter(deck)
    dec = decompose(gm, 3, MessageLog())
    locs = [slice_model(gm, dec, p, MessageLog()) for p in range(3)]
    _check_invariants(gm, locs)
    itf = gm.interfaces[0]
    assert itf.type == 2
    with contextlib.redirect_stdout(io.StringIO()):
        cg = ContactType2(itf, gm, MessageLog())
    nodes = _interface_nodes(gm, itf)
    nrep = 0
    for p, lm in enumerate(locs):
        holds = np.intersect1d(_native_global(lm), nodes).size > 0
        if holds:
            assert [i.id for i in lm.interfaces] == [itf.id]
        if not lm.interfaces:
            continue
        nrep += 1
        _dense(lm, nodes)
        assert lm.spmd.owned_interfaces == []
        sec = gm.node_groups[itf.grnod_id].node_idx
        assert np.array_equal(lm.spmd.nodglob[lm.node_groups[itf.grnod_id].node_idx], sec)
        # the replica ties the same nodes to the same segments with the
        # same weights, and sees the same element references (ghost ring)
        with contextlib.redirect_stdout(io.StringIO()):
            cl = ContactType2(lm.interfaces[0], lm, MessageLog())
        ng = lm.spmd.nodglob
        assert np.array_equal(ng[cl.snode], cg.snode)
        assert np.array_equal(ng[cl.seg], cg.seg)
        assert np.array_equal(cl.w, cg.w)
        assert cl.deletable == cg.deletable
        if getattr(cg, "ref_total", None) is not None:
            assert np.array_equal(cl.ref_total[cl.snode], cg.ref_total[cg.snode])
    assert nrep >= 2


# ---------------------------------------------------------------------------
# (e) unsupported features are refused
# ---------------------------------------------------------------------------

def test_check_spmd_support_refuses_lagmul(tmp_path):
    deck = _copy_deck(tmp_path, "rigid_impactor")
    gm = _starter(deck)
    check_spmd_support(gm)                      # supported as is
    gm.interfaces[0].lagmul = True
    gm.rbodies[0].lagmul = True
    with pytest.raises(StarterError) as exc:
        check_spmd_support(gm)
    msg = str(exc.value)
    assert "/INTER/LAGMUL" in msg and "/RBODY/LAGMUL" in msg


def test_run_starter_np_refuses_before_restart(tmp_path, monkeypatch):
    deck = _copy_deck(tmp_path, "rigid_impactor")
    import pyradioss.spmd.domdec as dd

    def _refuse(model):
        raise StarterError("SPMD (-np > 1) does not support: /INTER/LAGMUL/TYPE7/1")
    monkeypatch.setattr(dd, "check_spmd_support", _refuse)
    with pytest.raises(StarterError):
        _starter(deck, nspmd=2)
    base = deck[: -len("_0000.rad")]
    assert not os.path.isfile(base + "_0000.rst")
    assert not glob.glob(base + "_0000_*.rst")

def test_gjoint_refused_under_spmd():
    model = Model()
    model.gjoints = [GJoint(id=1, subtype="GEAR", node_id0=1, node_id1=2, node_id2=3)]
    with pytest.raises(StarterError) as exc:
        check_spmd_support(model, np=2)
    assert "/GJOINT" in str(exc.value)
    # Serial (np=1) is supported
    check_spmd_support(model, np=1)


def test_type2_with_fail_refused_under_spmd(tmp_path):
    from pyradioss.model.entities import Interface
    deck = _copy_deck(tmp_path, "spot_weld")
    gm = _starter(deck)
    check_spmd_support(gm)                      # supported as is (no failure)

    # TYPE2 with failure enabled is refused
    gm.shells.state["chk_fail"] = True
    with pytest.raises(StarterError) as exc:
        check_spmd_support(gm)
    msg = str(exc.value)
    assert "/INTER/TYPE2" in msg
    assert "SPMD_EXCH_IDEL" in msg
    assert "chkstfn3.F" in msg

    # Reset failure flag
    gm.shells.state["chk_fail"] = False
    check_spmd_support(gm)

    # Penalty contacts with idel >= 1 (or idel10 >= 1) are refused under SPMD
    for itype in (7, 10, 11, 24):
        if itype == 10:
            itf = Interface(id=100 + itype, type=itype, idel10=1)
        else:
            itf = Interface(id=100 + itype, type=itype, idel=1)
        gm.interfaces.append(itf)
        with pytest.raises(StarterError) as exc:
            check_spmd_support(gm)
        msg = str(exc.value)
        assert f"/INTER/TYPE{itype}" in msg
        assert "SPMD_EXCH_IDEL" in msg
        assert "chkstfn3.F" in msg
        gm.interfaces.pop()


def test_kjoint_refused_under_spmd():
    model = Model()
    model.kjoints = [object()]
    with pytest.raises(StarterError) as exc:
        check_spmd_support(model, np=2)
    msg = str(exc.value)
    assert "/KJOINT" in msg
    assert "ruser33.F" in msg
    # Serial (np=1) is supported
    check_spmd_support(model, np=1)

