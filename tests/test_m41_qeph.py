"""M41 — QEPH shell (Ishell=24): the physically-stabilized 1-point shell.

Fortran reference: ``engine/source/elements/shell/coquez/`` — czforc3.F
driver, czcorc.F frame/kinematics (+ czcorp5.F warped projection), czdef.F
strain + hourglass rates, czfintce.F constant part, czfintn.F CZFINTN1 the
PHYSICAL (stiffness) hourglass with CVIS = GEO(17) = 1 and the material's
own plane-stress moduli, czproj.F CZPROJ1 reconstruction/projection.  See
the ``pyradioss/elements/shell_qeph.py`` module docstring for the map.

What these tests pin down:

* dispatch — Ishell 22/23/24 parts leave the BT group for shells_qeph
  (hm_read_prop01.F lines 185-192 folds 22/23 into 24);
* patch tests — constant membrane strain, constant curvature and constant
  transverse shear are reproduced EXACTLY on distorted flat geometry with
  every hourglass slot silent (the Flanagan-Belytschko property of the
  czdef.F mx13/my13 orthogonalization, in covariant form);
* objectivity — rigid translation and the strain OPERATOR on a rigid spin
  are exact zeros; the czcorc.F 2nd-order correction cancels the leapfrog
  chord artifact (V at t+dt/2 paired with X at t+dt) to O(theta^4) on its
  covered (membrane) components; frame invariance is exact;
* the physical stabilization — closed-form modal stiffnesses built from
  E, t and geometry each cycle (membrane k = A11 t/3 per node on the unit
  square, bending m = A11 t^3/36, transverse 2/3 G SHF t), the czfintn.F
  linear damper on top, the EINT/EVIS(8) energy split (elastic work into
  INTERNAL energy, only damper work into the hourglass ledger), and the
  COEFH = 0.999 plastic relaxation;
* the native dt claim — cndt3.F on the czcorc.F condensed length
  (FACDT = 5/4), byte-identical to the M40 BT-side claim formula;
* energy balance — a QEPH cantilever strip runs the engine to NORMAL with
  ERR ~ 0 and HE orders below IE.

MEASURED (M41, the milestone target): RD-E-1000 c08 (QEPH Sf_0.8) and c09
(Sf_0.9) FULL RUNS on both engines, fresh — port vs Fortran engine_win64:

    case  cycles(F)  coverage  max_rel_rms   HE(port)   HE(Fortran)
    c08   107837     1.0006    5.84e-06      ~1e-18     0.0
    c09    95856     1.0006    6.58e-06      ~1e-18     0.0

(c08 re-measured IDENTICAL to all printed digits after the czcorp5.F
plat-gate fix — Z1^2 < LM*TOL with LM=(L13+L24)/2, czcorc.F line 397 —
which moves only sub-CSV-precision digits; HE stays physical-zero.  Both
cases re-run FRESH on the final concurrent-builders M41 tree — QBAT +
BT-forensics + engine guard-conditioning edits merged in — and the
max_rel_rms reproduces to ALL printed digits: c08 5.841708325388154e-06,
c09 6.580144914995088e-06, HE ~9e-19, ENERGY ERROR -0.00 %.)

against the M40 BT-fallback baseline 0.233/0.231 (DEVIATION class, aborts
at t ~ 1060 of 1605 with HE ~ 2e5): the QEPH element technology closes the
RD-E-1000 family to MATCH at trajectory level (target was < 0.05).
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import SHELL_ISHELL_GROUPS, shell_bt4, shell_qeph
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)
from pyradioss.starter.starter import run_starter

STEEL = "/MAT/LAW1/1\nsteel elastic\n7.8e-6\n210. 0.3\n"
LAW2 = ("/MAT/LAW2/1\nsteel plastic\n7.8e-6\n210. 0.3\n"
        "0.2 0.1 0.1 1e30 0\n")

T = 0.1                       # plate thickness used by the kernel tests
DT = 1e-3

# single-element node blocks (flat unless noted)
SQ = "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"                  # unit square
RECT = "1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n"                # 2:1 rectangle
DIST = "1 0 0 0\n2 1.3 0.1 0\n3 1.1 0.9 0\n4 -0.2 1.2 0\n"   # tapered quad
PAR = "1 0 0 0\n2 1 0.2 0\n3 1.4 1.2 0\n4 0.4 1 0\n"         # parallelogram
WARP = "1 0 0 0\n2 1 0 0\n3 1 1 0.3\n4 0 1 0\n"              # node 3 lifted

# material constants of STEEL in the closed forms
E_, NU_ = 210.0, 0.3
A11 = E_ / (1.0 - NU_ * NU_)
G_ = E_ / (2.0 * (1.0 + NU_))
RHO = 7.8e-6
SHF = 5.0 / 6.0
DN = 0.015                    # QEPH default numerical damping (ZEP015)


def _build(tmp_path, nodes, ishell=24, t=T, nip=3, mat=STEEL, name="K"):
    deck = ("/BEGIN\nm41 qeph\n/NODE\n" + nodes +
            "/SHELL/1\n1 1 2 3 4\n"
            "/PART/1\nplate\n1 1\n" + mat +
            f"/PROP/SHELL/1\nplate prop\n{ishell} 0 0 0\n0 0 0 0 0\n"
            f"{nip} 0 {t}\n/END\n")
    f = tmp_path / f"{name}_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


def _forces(model, v=None, vr=None, dt=DT):
    g = model.shells_qeph
    if v is None:
        v = np.zeros_like(model.x)
    if vr is None:
        vr = np.zeros_like(model.x)
    f = np.zeros_like(model.x)
    mm = np.zeros_like(model.x)
    dte = shell_qeph.forces(g, model.x, v, vr, dt, f, mm)
    return g, f, mm, dte


def _rot(axis, th):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K, axis


def _rates(model, v, vr, dt_kin):
    """White-box: kinematics + czdef rates only (no material update)."""
    g = model.shells_qeph
    G = shell_qeph._geometry(model.x[g.conn])
    v13, v24, vhi, rl, plat, vqn, di, db = shell_qeph._kinematics(
        G, v[g.conn], vr[g.conn], dt_kin, g.state["npt1"])
    vdef, vhg = shell_qeph._rates(G, v13, v24, vhi, rl,
                                  np.ones(g.n, bool))
    return G, vdef, vhg


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_ishell_dispatch_table():
    """22/23/24 -> QEPH (the starter folds 22/23 into 24), 12 -> QBAT."""
    assert SHELL_ISHELL_GROUPS[24] == "shells_qeph"
    assert SHELL_ISHELL_GROUPS[22] == "shells_qeph"
    assert SHELL_ISHELL_GROUPS[23] == "shells_qeph"
    assert SHELL_ISHELL_GROUPS[12] == "shells_qbat"


@pytest.mark.parametrize("ishell", [22, 24])
def test_dispatch_routes_to_qeph(tmp_path, ishell):
    m = _build(tmp_path, SQ, ishell=ishell)
    assert m.shells is None                       # whole part rerouted
    assert m.shells_qeph is not None and m.shells_qeph.n == 1
    assert m.shells_qbat is None


def test_dispatch_leaves_bt_untouched(tmp_path):
    m = _build(tmp_path, SQ, ishell=1)
    assert m.shells is not None and m.shells.n == 1
    assert m.shells_qeph is None and m.shells_qbat is None


def test_dispatch_mixed_deck_splits_parts(tmp_path):
    """A deck mixing a BT part (Ishell=1) and a QEPH part (Ishell=24):
    each part routes wholly to its kernel, ids preserved."""
    deck = ("/BEGIN\nm41 mixed\n/NODE\n"
            "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
            "5 2 0 0\n6 3 0 0\n7 3 1 0\n8 2 1 0\n"
            "/SHELL/1\n11 1 2 3 4\n"
            "/SHELL/2\n22 5 6 7 8\n"
            "/PART/1\nbt part\n1 1\n"
            "/PART/2\nqeph part\n2 1\n" + STEEL +
            "/PROP/SHELL/1\nbt prop\n1 0 0 0\n0 0 0 0 0\n3 0 0.1\n"
            "/PROP/SHELL/2\nqeph prop\n24 0 0 0\n0 0 0 0 0\n3 0 0.1\n"
            "/END\n")
    f = tmp_path / "KM_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    assert model.shells is not None and model.shells.n == 1
    assert model.shells.ids.tolist() == [11]
    assert model.shells_qeph is not None and model.shells_qeph.n == 1
    assert model.shells_qeph.ids.tolist() == [22]


# ---------------------------------------------------------------------------
# patch tests (constant states exact, hourglass silent) — czdef.F
# ---------------------------------------------------------------------------

def test_membrane_patch_constant_strain(tmp_path):
    """Linear in-plane velocity on a tapered flat quad: exact constant
    stress, every hourglass slot silent, nodal forces equilibrated."""
    m = _build(tmp_path, DIST)
    exx, eyy, gxy = 1e-3, -4e-4, 6e-4
    v = np.zeros_like(m.x)
    v[:, 0] = exx * m.x[:, 0] + 0.5 * gxy * m.x[:, 1]
    v[:, 1] = eyy * m.x[:, 1] + 0.5 * gxy * m.x[:, 0]
    g, f, mm, _ = _forces(m, v)
    st = g.state
    # in-plane state is frame-invariant here (e1 = global x for this mesh
    # is NOT guaranteed — so compare invariants: trace and von Mises)
    s = st["sig"][0, 0]
    de = np.array([exx, eyy, gxy]) * DT
    s_exp = np.array([A11 * de[0] + NU_ * A11 * de[1],
                      A11 * de[1] + NU_ * A11 * de[0], G_ * de[2]])
    assert np.isclose(s[0] + s[1], s_exp[0] + s_exp[1], rtol=1e-9)
    vm = lambda a: a[0] ** 2 - a[0] * a[1] + a[1] ** 2 + 3 * a[2] ** 2
    assert np.isclose(vm(s), vm(s_exp), rtol=1e-9)
    assert np.all(st["sig"][0] == st["sig"][0, 0])   # same at every layer
    assert np.abs(st["hgstr"]).max() < 1e-15         # FB property
    assert st["ehour"][0] < 1e-25
    assert np.abs(f.sum(axis=0)).max() < 1e-16
    tq = np.cross(m.x, f).sum(axis=0) + mm.sum(axis=0)
    assert np.abs(tq).max() < 1e-16


def test_bending_patch_constant_curvature(tmp_path):
    """theta_y = eps*x on the axis-aligned rectangle: per-layer stress is
    exactly A11*z*kxx*dt / A12*z*kxx*dt, no hourglass, no membrane."""
    m = _build(tmp_path, RECT)
    eps = 1e-3
    vr = np.zeros_like(m.x)
    vr[:, 1] = eps * m.x[:, 0]
    g, f, mm, _ = _forces(m, None, vr)
    st = g.state
    zg, _ = st["zw"][0]
    for k in range(3):
        zk = zg[k] * T
        exp = np.array([A11 * zk * eps * DT, NU_ * A11 * zk * eps * DT, 0.0])
        assert np.allclose(st["sig"][0, k], exp, rtol=1e-12, atol=1e-22)
    assert np.abs(st["hgstr"]).max() < 1e-18
    # constant transverse shear rides along: gxz = w,x + theta_y with the
    # element-mean theta_y = eps * x_center = eps (center x = 1.0)
    assert np.isclose(st["qshear"][0, 0], st["gs"][0] * eps * 1.0 * DT,
                      rtol=1e-12)
    assert np.abs(f.sum(axis=0)).max() < 1e-16
    tq = np.cross(m.x, f).sum(axis=0) + mm.sum(axis=0)
    assert np.abs(tq).max() < 1e-15


def test_shear_patch_constant_rect_and_par(tmp_path):
    """Constant rotations + linear w = constant transverse shear.  On any
    parallelogram (mx13 = my13 = 0) every hourglass slot is EXACTLY zero
    (czdef.F lines 105-108) and |qshear| = GS*|g|*dt (frame-invariant);
    on the axis-aligned rectangle the components match directly."""
    b, c, thx, thy = 2e-3, -1e-3, 4e-4, 7e-4
    for sub, nodes in (("r", RECT), ("p", PAR)):
        m = _build(tmp_path, nodes, name=f"KS{sub}")
        v = np.zeros_like(m.x)
        v[:, 2] = b * m.x[:, 0] + c * m.x[:, 1]
        vr = np.zeros_like(m.x)
        vr[:, 0] = thx
        vr[:, 1] = thy
        g, f, mm, _ = _forces(m, v, vr)
        st = g.state
        assert np.abs(st["hgstr"]).max() < 1e-18, nodes
        q = st["qshear"][0]
        q_exp = st["gs"][0] * np.hypot(b + thy, c - thx) * DT
        assert np.isclose(np.hypot(q[0], q[1]), q_exp, rtol=1e-12), nodes
        if nodes is RECT:          # axis-aligned frame: componentwise
            assert np.isclose(q[0], st["gs"][0] * (b + thy) * DT,
                              rtol=1e-12)
            assert np.isclose(q[1], st["gs"][0] * (c - thx) * DT,
                              rtol=1e-12)


def test_hourglass_silent_on_linear_fields_distorted(tmp_path):
    """The M39 implicit lesson, QEPH form: the stabilization must see
    NOTHING on linear velocity / rotation fields.  Membrane (slots 1,2,
    7,8) and bending (3,4,9,10) are silent on ARBITRARY flat quads —
    czdef.F's mx13/my13 orthogonalization; the shear families are silent
    on parallelograms (taper-coupled otherwise, by upstream design —
    tested next)."""
    m = _build(tmp_path, DIST)
    rng = np.random.default_rng(7)
    L = rng.normal(size=(2, 3)) * 1e-3
    Lr = rng.normal(size=(2, 3)) * 1e-3
    v = np.zeros_like(m.x)
    v[:, :2] = m.x @ L.T
    vr = np.zeros_like(m.x)
    vr[:, :2] = m.x @ Lr.T
    g, f, mm, _ = _forces(m, v, vr)
    hg = g.state["hgstr"][0]
    assert np.abs(hg[[0, 1, 6, 7]]).max() < 1e-15      # membrane eta/ksi
    assert np.abs(hg[[2, 3, 8, 9]]).max() < 1e-15      # bending eta/ksi


def test_shear_hourglass_taper_closed_form(tmp_path):
    """On a TAPERED quad the constant-shear state feeds the shear
    hourglass through mx13/my13 exactly as czdef.F writes it:
    VHG5 = VHG6 = 16*(mx13*gxz + my13*gyz) (local frame)."""
    m = _build(tmp_path, DIST)
    b, c, thx, thy = 2e-3, -1e-3, 4e-4, 7e-4
    v = np.zeros_like(m.x)
    v[:, 2] = b * m.x[:, 0] + c * m.x[:, 1]
    vr = np.zeros_like(m.x)
    vr[:, 0] = thx
    vr[:, 1] = thy
    G, vdef, vhg = _rates(m, v, vr, DT)
    gxz = vdef[0, 3]                      # local-frame constant shear
    gyz = vdef[0, 4]
    exp = 16.0 * (G["mx13"][0] * gxz + G["my13"][0] * gyz)
    assert np.isclose(vhg[0, 4], exp, rtol=1e-10)
    assert np.isclose(vhg[0, 5], exp, rtol=1e-10)
    assert np.abs(vhg[0, :4]).max() < 1e-18


# ---------------------------------------------------------------------------
# objectivity — czcorc.F (+ czcorp5.F for warped)
# ---------------------------------------------------------------------------

def test_rigid_translation_inert(tmp_path):
    m = _build(tmp_path, WARP)
    v = np.tile([1e-3, -2e-3, 5e-4], (m.numnod, 1))
    g, f, mm, _ = _forces(m, v)
    st = g.state
    assert np.abs(f).max() == 0.0 and np.abs(mm).max() == 0.0
    assert np.abs(st["sig"]).max() == 0.0
    assert np.abs(st["hgstr"]).max() == 0.0
    assert st["eint"][0] == 0.0 and st["ehour"][0] == 0.0


def test_rigid_spin_operator_exact(tmp_path):
    """The czdef.F strain operator annihilates a rigid velocity field
    exactly (dt-correction off: it targets only the leapfrog chord)."""
    m = _build(tmp_path, DIST)
    Rm, axis = _rot([0.2, -0.4, 0.9], 0.0)
    w = np.array([0.3, -0.5, 0.8]) * 1e-2
    ce = m.x[:4].mean(axis=0)
    v = np.cross(w, m.x - ce)
    vr = np.tile(w, (m.numnod, 1))
    _, vdef, vhg = _rates(m, v, vr, 0.0)
    assert np.abs(vdef).max() < 1e-16
    assert np.abs(vhg).max() < 1e-16


def test_leapfrog_chord_correction_second_order(tmp_path):
    """czcorc.F lines 540-568: V(t+dt/2) with X(t+dt) — the chord that
    ARRIVES at the current geometry.  Uncorrected it reads theta^2/2 of
    spurious membrane strain per step; the 2nd-order correction cancels
    it to O(theta^4) for in-plane and out-of-plane axes."""
    for ax in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]):
        th = 1e-3
        m = _build(tmp_path, SQ, name=f"KC{int(ax[0])}")
        Rm, axis = _rot(ax, -th)               # back-rotation R(-theta)
        ce = m.x[:4].mean(axis=0)
        v = ((m.x - ce) - (m.x - ce) @ Rm.T) / DT
        vr = np.tile(axis * th / DT, (m.numnod, 1))
        _, vd_on, _ = _rates(m, v, vr, DT)
        _, vd_off, _ = _rates(m, v, vr, 0.0)
        mem_on = np.abs(vd_on[0, :3]).max() * DT
        mem_off = np.abs(vd_off[0, :3]).max() * DT
        assert np.isclose(mem_off, th * th / 2.0, rtol=1e-2)   # the artifact
        assert mem_on < th ** 3                                # killed
        assert mem_on < 1e-2 * mem_off


def test_skew_rotation_bounded(tmp_path):
    """Mixed-axis rotation on warped/tapered elements: the correction
    strictly reduces the chord artifact (formulation-inherent O(theta^2)
    crumbs remain in the shear coupling — upstream identical)."""
    for sub, nodes in (("w", WARP), ("d", DIST)):
        th = 1e-3
        m = _build(tmp_path, nodes, name=f"KX{sub}")
        Rm, axis = _rot([0.3, -0.5, 0.8], -th)
        ce = m.x[:4].mean(axis=0)
        v = ((m.x - ce) - (m.x - ce) @ Rm.T) / DT
        vr = np.tile(axis * th / DT, (m.numnod, 1))
        _, vd_on, _ = _rates(m, v, vr, DT)
        _, vd_off, _ = _rates(m, v, vr, 0.0)
        r_on = np.abs(vd_on).max() * DT
        r_off = np.abs(vd_off).max() * DT
        assert r_on < 0.55 * r_off
        assert r_on < 0.5 * th * th


def test_frame_invariance(tmp_path):
    """Rotate geometry + velocities by a fixed Q: forces/moments rotate
    by Q and the dt claim is unchanged (covariant frame, CLSKEW3)."""
    m1 = _build(tmp_path, DIST, name="KF1")
    rng = np.random.default_rng(11)
    v = rng.normal(size=m1.x.shape) * 1e-3
    vr = rng.normal(size=m1.x.shape) * 1e-3
    g1, f1, mm1, dte1 = _forces(m1, v, vr)
    Q, _ = _rot([1.0, 2.0, -1.5], 0.7)
    m2 = _build(tmp_path, DIST, name="KF2")
    m2.x = m1.x @ Q.T
    g2, f2, mm2, dte2 = _forces(m2, v @ Q.T, vr @ Q.T)
    scale = np.abs(f1).max()
    assert np.abs(f2 - f1 @ Q.T).max() < 1e-12 * scale
    assert np.abs(mm2 - mm1 @ Q.T).max() < 1e-12 * max(np.abs(mm1).max(),
                                                       1e-30)
    assert np.isclose(dte1[0], dte2[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# the physical stabilization — czfintn.F closed forms (unit square)
# ---------------------------------------------------------------------------

def test_membrane_hourglass_physical_stiffness(tmp_path):
    """h-pattern in-plane velocity: modal stress DG1 = A11*HXX*q with the
    FULL plane-stress modulus (CVIS = 1), per-node force k = A11*t/3 plus
    the czfintn.F linear damper; elastic work -> EINT, damper -> EHOUR."""
    m = _build(tmp_path, SQ)
    gv = 1e-3
    v = np.zeros_like(m.x)
    v[:4, 0] = np.array([1.0, -1.0, 1.0, -1.0]) * gv
    g, f, mm, _ = _forces(m, v)
    st = g.state
    # modal stress: HXX = 4*A_I*MY34 = 2 on the unit square
    assert np.isclose(st["hgstr"][0, 0], A11 * 2.0 * gv * DT, rtol=1e-12)
    k_el = A11 * T / 3.0
    hvl = DN * np.sqrt(RHO * 1.0)
    ss1v = (4.0 / 3.0) * np.sqrt(A11) * hvl * gv
    f_exp = k_el * gv * DT + 0.25 * T * ss1v
    assert np.isclose(f[0, 0], -f_exp, rtol=1e-12)
    assert np.isclose(f[1, 0], +f_exp, rtol=1e-12)     # h-pattern reaction
    # trapezoidal elastic energy from zero stress: 0.5*k*q^2 (4 nodes)
    assert np.isclose(st["eint"][0],
                      0.5 * k_el * (4.0 * gv * DT) * (gv * DT), rtol=1e-12)
    # only the damper work lands in the hourglass ledger (czfintn.F TESY)
    assert np.isclose(st["ehour"][0], ss1v * T * gv * DT, rtol=1e-12)


def test_bending_hourglass_physical_stiffness(tmp_path):
    """theta_y h-pattern: DG3 = A11*HXX*rhi*dt, DG4 = -A12*HXX*rhi*dt,
    per-node moment A11*t^3/36 + the FBEND_V = 3.464 viscous rider."""
    m = _build(tmp_path, SQ)
    rv = 1e-3
    vr = np.zeros_like(m.x)
    vr[:4, 1] = np.array([1.0, -1.0, 1.0, -1.0]) * rv
    g, f, mm, _ = _forces(m, None, vr)
    st = g.state
    assert np.isclose(st["hgstr"][0, 2], A11 * 2.0 * rv * DT, rtol=1e-12)
    assert np.isclose(st["hgstr"][0, 3], -NU_ * A11 * 2.0 * rv * DT,
                      rtol=1e-12)
    hvl = DN * np.sqrt(RHO * 1.0)
    sf1_v = np.sqrt(A11) * hvl * (16.0 / 3.0) * 0.25 * rv * 3.464
    c6 = T * T / 12.0
    m_exp = A11 * T ** 3 * rv * DT / 36.0 + c6 * (T / 4.0) * sf1_v
    assert np.isclose(mm[0, 1], -m_exp, rtol=1e-12)
    # damper share of the trapezoidal work (czfintn.F TESY bending term)
    assert np.isclose(st["ehour"][0], sf1_v * rv * DT * T * c6, rtol=1e-12)


def test_transverse_hourglass_physical_stiffness(tmp_path):
    """w h-pattern: the shear families resist with the PHYSICAL modulus
    G*SHF (DG5 = G*SHF/64*HXX*dhg5), total per-node 2/3 G SHF t q plus
    the sqrt-moduli damper — the mode whose BT treatment left the M40
    RD-E-1000 residual."""
    m = _build(tmp_path, SQ)
    gv = 1e-3
    v = np.zeros_like(m.x)
    v[:4, 2] = np.array([1.0, -1.0, 1.0, -1.0]) * gv
    g, f, mm, _ = _forces(m, v)
    st = g.state
    assert np.isclose(st["hgstr"][0, 4],
                      G_ * SHF / 64.0 * 2.0 * 16.0 * gv * DT, rtol=1e-12)
    ss3_el = 2.0 / 3.0 * G_ * SHF * T * gv * DT
    hvl = DN * np.sqrt(RHO * 1.0)
    c2v = hvl * np.sqrt(G_) * np.sqrt(SHF) * np.sqrt(1.0 / 12.0)
    ss3_v = 8.0 * c2v * gv * T
    assert np.isclose(f[0, 2], -(ss3_el + ss3_v), rtol=1e-12)
    assert np.isclose(f[1, 2], +(ss3_el + ss3_v), rtol=1e-12)


def test_npt1_membrane_kept_shear_by_damper_only(tmp_path):
    """N=1 (membrane): FBEND = SHF = 0 kill the elastic bending/shear
    stabilization; the membrane closed form is unchanged and the
    transverse mode falls to the CZFINTNM 25*dn*sqrt(G rho A/12) damper,
    whose work books to the hourglass ledger."""
    m = _build(tmp_path, SQ, nip=1, name="K1m")
    gv = 1e-3
    v = np.zeros_like(m.x)
    v[:4, 0] = np.array([1.0, -1.0, 1.0, -1.0]) * gv
    g, f, mm, _ = _forces(m, v)
    hvl = DN * np.sqrt(RHO * 1.0)
    ss1v = (4.0 / 3.0) * np.sqrt(A11) * hvl * gv
    f_exp = A11 * T / 3.0 * gv * DT + 0.25 * T * ss1v
    assert np.isclose(f[0, 0], -f_exp, rtol=1e-12)

    m2 = _build(tmp_path, SQ, nip=1, name="K1t")
    v2 = np.zeros_like(m2.x)
    v2[:4, 2] = np.array([1.0, -1.0, 1.0, -1.0]) * gv
    g2, f2, mm2, _ = _forces(m2, v2)
    st2 = g2.state
    assert st2["hgstr"][0, 4] == 0.0                  # SHF = 0: no elastic
    hvl_nm = 25.0 * DN * np.sqrt(G_ * RHO * 1.0 / 12.0)
    ss3_nm = 2.0 * 0.25 * hvl_nm * 16.0 * gv * T
    assert np.isclose(f2[0, 2], -ss3_nm, rtol=1e-12)
    assert np.isclose(st2["ehour"][0], ss3_nm * 16.0 * gv * DT, rtol=1e-12)


def test_plastic_relaxation_coefh(tmp_path):
    """czfintn.F lines 301-351: once the combined constant+hourglass von
    Mises passes the yield, the modal-stress increment keeps only
    (1 - COEFH) = 1e-3 of its elastic value; the elastic element keeps
    the full increment."""
    m = _build(tmp_path, SQ, mat=LAW2, name="KP")
    g = m.shells_qeph
    st = g.state
    v = np.zeros_like(m.x)
    v[:, 0] = 1e-2 * m.x[:, 0]                 # stretch far past yield
    f = np.zeros_like(m.x)
    mm = np.zeros_like(m.x)
    for _ in range(20):
        f[:] = 0.0
        mm[:] = 0.0
        shell_qeph.forces(g, m.x, v, np.zeros_like(v), 1e-1, f, mm)
    assert st["epsp"][0].min() > 5e-3          # plastic flow happened
    v2 = np.zeros_like(m.x)
    v2[:4, 0] = np.array([1.0, -1.0, 1.0, -1.0]) * 1e-3
    f[:] = 0.0
    mm[:] = 0.0
    shell_qeph.forces(g, m.x, v2, np.zeros_like(v2), DT, f, mm)
    elastic_dg = A11 * 2.0 * 1e-3 * DT
    assert np.isclose(st["hgstr"][0, 0] / elastic_dg, 1.0e-3, rtol=1e-6)

    m2 = _build(tmp_path, SQ, name="KE")       # LAW1: criterion inert
    g2, f2, mm2, _ = _forces(m2, v2)
    assert np.isclose(g2.state["hgstr"][0, 0], elastic_dg, rtol=1e-12)


# ---------------------------------------------------------------------------
# warped elements — czcorp5.F / czproj.F
# ---------------------------------------------------------------------------

def test_plat_gate_mean_squared_diagonal(tmp_path):
    """czcorp5.F line 84: flat/warped classifies on Z1^2 < LM*TOL with
    LM = (L13+L24)/2 the MEAN SQUARED half-diagonal (czcorc.F line 397,
    passed into CZCORP5's LL dummy at line 581) and TOL = EM8 — NOT the
    condensed dt length ll.  On a 10:1 rectangle the two candidate
    thresholds differ ~5x; a lift between them must classify FLAT."""
    def mesh(h, name):
        return _build(tmp_path, f"1 0 0 0\n2 10 0 {h}\n3 10 1 0\n"
                      "4 0 1 0\n", name=name), None

    def plat_of(m, nip1=False):
        g = m.shells_qeph
        G = shell_qeph._geometry(m.x[g.conn])
        z1_abs = abs(G["z1"][0])
        *_, plat, _, _, _ = shell_qeph._kinematics(
            G, np.zeros((g.n, 4, 3)), np.zeros((g.n, 4, 3)), DT,
            g.state["npt1"])
        return G, z1_abs, plat[0]

    m, _ = mesh(1e-4, "KPG0")                      # probe lift
    G, z1p, _ = plat_of(m)
    thr_lm = np.sqrt(G["lm"][0] * 1e-8)            # the czcorp5.F gate
    thr_ll = np.sqrt(G["ll"][0] * 1e-8)            # the WRONG scale
    assert thr_lm > 3.0 * thr_ll                   # scales separated here
    target = np.sqrt(thr_lm * thr_ll)              # midway (geometric)
    m, _ = mesh(1e-4 * target / z1p, "KPG1")
    G, z1m, plat = plat_of(m)
    assert thr_ll < z1m < thr_lm                   # landed in the gap
    assert plat                                    # LM gate: still FLAT
    m, _ = mesh(0.3, "KPG2")                       # genuinely warped
    _, _, plat = plat_of(m)
    assert not plat
    m3 = _build(tmp_path, WARP, nip=1, name="KPG3")  # NPT==1 forces PLAT
    _, _, plat = plat_of(m3)
    assert plat


def test_warped_equilibrium(tmp_path):
    """Random nodal velocities/rotations on a warped quad: the czproj.F
    re-projection returns an EXACTLY self-equilibrated force/moment set
    (machine-zero resultant force AND torque)."""
    m = _build(tmp_path, WARP)
    rng = np.random.default_rng(3)
    v = rng.normal(size=m.x.shape) * 1e-3
    vr = rng.normal(size=m.x.shape) * 1e-3
    g, f, mm, _ = _forces(m, v, vr)
    scale = np.abs(f).max()
    assert scale > 0.0
    assert np.abs(f.sum(axis=0)).max() < 1e-14 * scale
    tq = np.cross(m.x, f).sum(axis=0) + mm.sum(axis=0)
    assert np.abs(tq).max() < 1e-14 * scale


# ---------------------------------------------------------------------------
# dt claim — cndt3.F on the czcorc.F condensed length (FACDT = 5/4)
# ---------------------------------------------------------------------------

def test_dt_claim_condensed_length(tmp_path):
    """The native QEPH claim equals the M40 BT-side formula (czcorc.F
    lines 377-402 with FACDT = 5/4) times cndt3.F's damping factor
    sqrt(1+dn^2)-dn with the ZEP015 default."""
    m = _build(tmp_path, DIST)
    g, f, mm, dte = _forces(m)
    G = shell_qeph._geometry(m.x[g.conn])
    xl = np.stack([G["corx"], G["cory"]], axis=2)
    ll = shell_bt4._condensed_length(xl, G["area"], 1.25)
    assert np.isclose(G["ll"][0], ll[0], rtol=1e-14)
    mat = g.state["slices"][0][1]
    visc = np.sqrt(1.0 + DN * DN) - DN
    assert np.isclose(dte[0], visc * ll[0] / mat.sound_speed_shell(),
                      rtol=1e-12)
    # and the M40 BT-side claim table carries the same (FACDT, dn) pair
    assert shell_bt4._CONDENSED_FACDT[24] == (1.25, 1.5e-2)


# ---------------------------------------------------------------------------
# engine integration — energy balance ~ 0, HE physical-small
# ---------------------------------------------------------------------------

def test_engine_energy_balance(tmp_path):
    """A clamped QEPH strip pushed at the tip runs the engine to NORMAL
    with a closed energy balance and HE orders below IE (the elastic
    stabilization work lives in EINT, only the damper in the ledger —
    czfintn.F lines 297/443; c08 full-run analogue: HE/IE ~ 1e-24)."""
    nodes = []
    nid = 0
    for i in range(5):
        for j in range(2):
            nid += 1
            nodes.append(f"{nid} {i * 10.0} {j * 10.0} 0")
    shells = []
    for e in range(4):
        n1 = 2 * e + 1
        shells.append(f"{e + 1} {n1} {n1 + 2} {n1 + 3} {n1 + 1}")
    starter = ("/BEGIN\nqeph strip\n/NODE\n" + "\n".join(nodes) + "\n"
               "/SHELL/1\n" + "\n".join(shells) + "\n"
               "/PART/1\nstrip\n1 1\n" + STEEL +
               "/PROP/SHELL/1\nqeph\n24 0 0 0\n0 0 0 0 0\n3 0 1.0\n"
               "/BCS/1\nclamp\n111 111 0 1\n"
               "/GRNOD/NODE/1\nroot\n1 2\n"
               "/FUNCT/1\npush\n0.0 0.05\n100.0 0.05\n"
               "/IMPVEL/1\ntip\n1 Z 2\n"
               "/GRNOD/NODE/2\ntip\n9 10\n"
               "/END\n")
    engine = "/RUN/QS/1\n2.0\n/DT\n0.9 0\n/PRINT/-10000\n/END\n"
    s = tmp_path / "QS_0000.rad"
    e = tmp_path / "QS_0001.rad"
    s.write_text(starter)
    e.write_text(engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(s))
        model = run_engine(str(e))
    txt = (tmp_path / "QS_0001.out").read_text()

    def num(label):
        return float(re.search(label + r"\s*\.[ .]*:\s*([-\d.Ee+]+)",
                               txt).group(1))

    assert "ENGINE TERMINATION : NORMAL" in txt
    assert abs(num("ENERGY ERROR")) < 0.5              # percent
    ie = num("INTERNAL ENERGY")
    assert ie > 1e-6                                   # it actually bent
    assert num("HOURGLASS ENERGY") < 1e-4 * ie         # physical-small
    # the QEPH group really carried the run
    assert model.shells_qeph is not None and model.shells_qeph.n == 4
    assert model.shells is None
