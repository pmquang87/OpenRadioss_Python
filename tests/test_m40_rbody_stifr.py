"""M40 — the ROTATIONAL (STIFR) completion of the /RBODY master-dt transport.

M39 transported only the TRANSLATIONAL member stiffness to the /RBODY
master (``STIFN`` + the DD parallel-axis term), flooring the RD-E-1000 c04
roll at 2.067e-2 where the Fortran floors at 1.644e-2 — because rgbodfp.F
also transports the members' OWN rotational stiffness,

    STIFR(M) += Sum( STIFR(s) + DD * STIFN(s) )     (rgbodfp.F 116-118)

and dtnoda.F bounds the step with the master's rotational nodal dt
``sqrt(2 IN(M)/STIFR(M))`` (line 469).  M40 adds the element-side STIFR
claim (the rotational mirror of the M6 equivalent-spring nodal stiffness:
``kr = 2 I_lumped / dt_e^2``, reproducing cndt3.F's
``STIR = STI*(t^2+A)/12`` — the factor MATCHES the Starter's inertia
lumping, cinmas.F FAC=TWELVE for IHBE>=11), the rgbodfp gather above, the
free-node rotational dt of dtnoda.F (ALL nodes with IN>0 — non-binding on
element-lumped nodes BY CONSTRUCTION, see engine/mass_scaling.py), and the
BATOZ-family condensed characteristic length (cbacoor.F/czcorc.F — QBAT
Ishell=12 / QEPH Ishell=22/24 claim ``STI = 0.5 V A11/LC^2`` with LC
SHORTER than the BT side-length reading; on the c04 welded rectangles
LC = 13.411 vs 15.2, which is exactly the M39->M40 floor gap).

These tests pin each piece in closed form.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.common.messages import MessageLog
from pyradioss.engine.mass_scaling import NodalTimeStep
from pyradioss.starter.starter import run_starter

STEEL = "/MAT/LAW1/1\nsteel\n7.8e-6\n210. 0.3\n"

#: the c04 ROLLING material (E=1000, nu=0, rho=0.01) and geometry
#: (16 x 15.2 rectangles, t=20) — the deck whose floor M40 fixes
C04MAT = "/MAT/LAW1/1\nroll\n0.01\n1000. 0.\n"


def _one_shell(ishell: int, thick: float = 20.0) -> str:
    """One 16 x 15.2 rectangle (the c04 welded-element geometry)."""
    return ("/BEGIN\nm40\n/NODE\n"
            "1 0 0 0\n2 16 0 0\n3 16 15.2 0\n4 0 15.2 0\n"
            "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n" + C04MAT +
            f"/PROP/SHELL/1\nsh\n{ishell} 0 0 0\n0 0 0 0 0\n"
            f"3 0 {thick}\n/END\n")


class _Controls:
    dt_noda = "NODA"          # truthy, and not "CST"
    dt_scale = 0.9
    dt_min = 0.0


class _CstControls:
    dt_noda = "CST"
    dt_scale = 0.9
    dt_min = 0.0              # set per test


def _noda_after_claims(model):
    """Build a NodalTimeStep and run one force/assemble pass (the engine's
    priming pass, dt=0)."""
    from pyradioss.elements import KERNELS
    noda = NodalTimeStep(model, _Controls(), MessageLog())
    n = model.numnod
    fint, mint = np.zeros((n, 3)), np.zeros((n, 3))
    claims = []
    for name, group in model.element_groups():
        claims.append(KERNELS[name].forces(group, model.x, model.v,
                                           model.vr, 0.0, fint, mint))
    noda.assemble(claims)
    return noda, claims


def _starter(make_deck, name, deck):
    s, _ = make_deck(name, deck, f"/RUN/{name}/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


# ---------------------------------------------------------------------------
# 1. the rgbodfp.F gather, closed form
# ---------------------------------------------------------------------------

def test_stifr_transport_closed_form(make_deck):
    """``_rigid_body_dt`` must return the exact rgbodfp.F/dtnoda.F master
    minimum with the ROTATIONAL term: K_rot = Sum(STIFR + DD*STIFN), not
    the M39 DD-only sum."""
    model = _starter(make_deck, "M40CF", _one_shell(1))
    noda = NodalTimeStep(model, _Controls(), MessageLog())
    assert noda._rigid_body_dt() == EP30          # nothing registered

    nodes = np.arange(4)
    stifn = np.array([100.0, 100.0, 200.0, 200.0])
    stifr = np.array([5.0e4, 6.0e4, 7.0e4, 8.0e4])
    noda.stifn[nodes] = stifn
    noda.stifr[nodes] = stifr
    mass, inertia = 5.0, np.diag([2.0, 3.0, 4.0])
    noda.add_rigid_body(nodes, nodes[0], mass, inertia, model.x0)

    dd = ((model.x0[nodes] - model.x0[nodes[0]]) ** 2).sum(axis=1)
    k_tra = stifn.sum()
    k_rot = float((stifr + dd * stifn).sum())     # rgbodfp F2 = STIFR+DD*STIFN
    expect = min(np.sqrt(2.0 * mass / k_tra),
                 np.sqrt(2.0 * 2.0 / k_rot))      # 2.0 = min principal
    got = noda._rigid_body_dt()
    assert got == pytest.approx(expect, rel=1e-12)
    # and the rotational term genuinely participates (the M39 DD-only sum
    # would give a LARGER dt)
    k_rot_m39 = float((dd * stifn).sum())
    assert got < 0.999 * min(np.sqrt(2.0 * mass / k_tra),
                             np.sqrt(2.0 * 2.0 / k_rot_m39))


# ---------------------------------------------------------------------------
# 2. the element STIFR claim mirrors the inertia lumping (cndt3.F/cinmas.F)
# ---------------------------------------------------------------------------

def test_shell_stifr_claim_mirror(make_deck):
    """The assembled rotational stiffness must equal the translational one
    times the FAMILY inertia-lumping factor — upstream's STIR = STI*fac
    with the SAME factor the Starter lumps the nodal inertia with
    (cinmas.F lines 919-925: FAC=TWELVE for engine IHBE>=11, FAC=NINE
    otherwise).  Ishell=1 here is the BT family (engine IHBE=1), whose
    engine claim is chvis3.F's ``STIR = STI*(THK02*INV12 + AREA*INV9)``
    — i.e. t^2/12 + A/9, NOT the (t^2+A)/12 the M40 code applied to
    every family (that form is cndt3.F 209-218, the IHBE>=11
    BATOZ/QEPH branch; corrected in M41 with the cinmas.F FAC=9 BT
    lumping).  Consequence (dtnoda.F scope): the free-node rotational
    dt sqrt(2 IN/STIFR) EQUALS the translational sqrt(2 M/STIFN) on
    element-lumped nodes, so adding it never moves a non-/RBODY deck."""
    model = _starter(make_deck, "M40MIR", _one_shell(1))
    noda, _ = _noda_after_claims(model)
    name, group = next(iter(model.element_groups()))
    t = group.state["thick"][0]
    a = group.state["area0"][0]
    fac = t ** 2 / 12.0 + a / 9.0
    nodes = group.conn[0]
    assert noda.stifr[nodes] == pytest.approx(noda.stifn[nodes] * fac,
                                              rel=1e-12)
    # the invariant: rotational nodal dt == translational nodal dt
    dt_tra = np.sqrt(2.0 * model.mass[nodes] / noda.stifn[nodes])
    dt_rot = np.sqrt(2.0 * model.inertia[nodes] / noda.stifr[nodes])
    assert dt_rot == pytest.approx(dt_tra, rel=1e-12)
    # apply() with the inertia arrays returns the SAME minimum as the
    # translational-only reading (the non-/RBODY byte-identity guarantee,
    # up to the one-ulp product round-off of the shared factor)
    inv_inertia = np.where(model.inertia > 0, 1.0 /
                           np.maximum(model.inertia, 1e-30), 0.0)
    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    got = noda.apply(mass_eff, inv_mass, model.v, 0.0,
                     model.inertia, inv_inertia)
    assert got == pytest.approx(float(dt_tra.min()), rel=1e-12)


def test_beam_stifr_claim(make_deck):
    """The beam's rotational claim (pmcum3.F's per-node STIR accumulation)
    mirrors its own lumped inertia: kr = 2 I_c/dt0^2 with
    I_c = m/2 (L^2/12 + (Iyy+Izz)/A) — so sqrt(2 I/kr) reproduces the
    element dt exactly (the equivalent-spring contract)."""
    deck = ("/BEGIN\nm40beam\n/NODE\n1 0 0 0\n2 10 0 0\n3 5 5 0\n"
            "/BEAM/1\n1 1 2 3\n/PART/1\nb\n1 1\n" + STEEL +
            "/PROP/BEAM/1\nbm\n0 0 0\n2.0 1.0 1.5 2.0\n/END\n")
    model = _starter(make_deck, "M40BM", deck)
    noda, claims = _noda_after_claims(model)
    for name, group in model.element_groups():
        if "beam" not in name:
            continue
        iner_c = group.state["dt_iner"][0]
        dt0 = group.state["dt0"][0]
        ends = group.state["mass_conn"][0]
        kr = 2.0 * iner_c / dt0 ** 2
        assert noda.stifr[ends] == pytest.approx(kr, rel=1e-12)
        # orientation node claims nothing
        assert noda.stifr[group.conn[0, 2]] == 0.0


# ---------------------------------------------------------------------------
# 3. the BATOZ-family condensed characteristic length (cbacoor.F/czcorc.F)
# ---------------------------------------------------------------------------

def _condensed_lc(a, b, facdt):
    """Analytic cbacoor.F length of a flat a x b rectangle: LL = half-
    diagonal^2, C1 = 2a, C2 = 2b, LM = 0."""
    ll = (a ** 2 + b ** 2) / 4.0
    c1, c2 = 2.0 * a, 2.0 * b
    cmax, cmin = max(c1, c2), min(c1, c2)
    fac1 = min(0.5, 0.25 * (cmax / cmin - 1.0)) + 1.0
    area = a * b
    f2 = 3.413 * max(0.0, 4.0 * area / (c1 * c2) - 0.7071)
    fac2 = 0.78 + 0.22 * f2 ** 3
    faci = 2.0 * fac1 * fac2
    return area / np.sqrt(faci * facdt * ll)


@pytest.mark.parametrize("ishell,facdt,dn", [(12, 4.0 / 3.0, 1.0e-3),
                                             (24, 1.25, 1.5e-2)])
def test_condensed_length_claim(make_deck, ishell, facdt, dn):
    """An Ishell=12 (QBAT) / 24 (QEPH) shell claims its dt with the
    condensed length LC (cbacoor.F 1080-1098 / czcorc.F 377-402) times
    the numerical-damping factor sqrt(1+dn^2)-dn (cndt3.F 84-88, dn
    defaulting to 1e-3 / 0.015 per formulation — the values the c04/c08
    Fortran starter listings print), not the BT side length.  On the c04
    16 x 15.2 rectangle LC(QBAT) = 13.410 — the exact origin of the
    Fortran 1.644e-2 floor."""
    model = _starter(make_deck, f"M40LC{ishell}", _one_shell(ishell))
    noda, claims = _noda_after_claims(model)
    lc = _condensed_lc(16.0, 15.2, facdt)
    visc = np.sqrt(1.0 + dn * dn) - dn
    c = np.sqrt(1000.0 / 0.01)                    # sqrt(E/rho), nu = 0
    assert claims[0][0] == pytest.approx(visc * lc / c, rel=1e-9)
    if ishell == 12:
        assert lc == pytest.approx(13.4102, abs=2e-4)   # the c04 number


def test_bt_claim_unchanged(make_deck):
    """Ishell=1 (BT) keeps the side-length claim — dtfac stays the exact-
    eigenvalue factor (1.0 for this element), lc = A/max side = 15.2:
    the byte-identity guarantee for every pre-M40 BT deck."""
    model = _starter(make_deck, "M40BT", _one_shell(1))
    noda, claims = _noda_after_claims(model)
    c = np.sqrt(1000.0 / 0.01)
    assert claims[0][0] == pytest.approx(15.2 / c, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. the c04 master floor, end to end in closed form
# ---------------------------------------------------------------------------

def test_c04_master_floor_closed_form(make_deck):
    """A 3-element Ishell=12 strip whose z=0 edge nodes are welded to a
    /RBODY (the c04 ROLLING topology, scaled down): the nodal dt returned
    by apply() must equal the analytic master rotational floor
    0.9-unscaled sqrt(2 IN_min / Sum(STIFR + DD*STIFN)) — the complete
    rgbodfp.F/dtnoda.F chain through real starter + kernels."""
    # nodes: 2 rows (y = 0 and 15.2), 4 columns 16 apart; master at the
    # centroid of the y=0 edge, offset like c04's node 1020 (on the edge)
    lines = []
    nid = 0
    for y in (0.0, 15.2):
        for i in range(4):
            nid += 1
            lines.append(f"{nid} {16.0 * i} {y} 0")
    lines.append("100 24 0 0")                    # master on the y=0 edge
    shells = [f"{e + 1} {e + 1} {e + 2} {e + 6} {e + 5}" for e in range(3)]
    deck = ("/BEGIN\nm40floor\n/NODE\n" + "\n".join(lines) + "\n"
            "/SHELL/1\n" + "\n".join(shells) + "\n"
            "/PART/1\np\n1 1\n" + C04MAT +
            "/PROP/SHELL/1\nsh\n12 0 0 0\n0 0 0 0 0\n3 0 20\n"
            "/GRNOD/NODE/2\nsl\n1 2 3 4\n"
            "/RBODY/1\nrb\n100 2 0 0\n/END\n")
    model = _starter(make_deck, "M40FLR", deck)
    noda, claims = _noda_after_claims(model)

    rb = model.rbodies[0]
    from pyradioss.engine.kinematics import LoadsAndConstraints
    from pyradioss.engine.rigid_body import build_rigid_bodies
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        loads = LoadsAndConstraints(model, log)
        rbs = build_rigid_bodies(model, loads, log)
    body = rbs[0]
    noda.set_prescribed(body.nodes)
    noda.add_rigid_body(body.nodes, body.master, body.M, body.J0, model.x0)

    # analytic: per-element claim k = 2 (m/4)/dt_e^2, STIR = k (t^2+A)/12
    dd = ((model.x0[body.nodes] - model.x0[body.master]) ** 2).sum(axis=1)
    st = noda.stifn[body.nodes]
    sr = noda.stifr[body.nodes]
    k_rot = float((sr + dd * st).sum())
    k_tra = float(st.sum())
    in_min = float(np.linalg.eigvalsh(body.J0)[0])
    expect = min(np.sqrt(2.0 * body.M / k_tra),
                 np.sqrt(2.0 * in_min / k_rot))

    inv_inertia = np.where(model.inertia > 0, 1.0 /
                           np.maximum(model.inertia, 1e-30), 0.0)
    mass_eff = model.mass.copy()
    got = noda.apply(mass_eff, 1.0 / mass_eff, model.v, 0.0,
                     model.inertia, inv_inertia)
    # the master floor governs (free nodes are far above it here)
    assert got == pytest.approx(expect, rel=1e-12)
    # and the rotational branch is the binding one, well below the
    # translational master dt — the c04 mechanism
    assert expect < 0.7 * np.sqrt(2.0 * body.M / k_tra)


# ---------------------------------------------------------------------------
# 5. /DT/NODA/CST holds the rotational dt by ADDING INERTIA (dtnoda.F
#    482-516, the DINERT branch)
# ---------------------------------------------------------------------------

def test_cst_inertia_scaling(make_deck):
    """With CST and a dt_min ABOVE the natural nodal dt, apply() must hold
    BOTH branches at the target: mass added for the translational one
    (pre-M40 behavior) and inertia added for the rotational one
    (IN = MAX(INER, IN)); the returned dt is the held target and the
    added inertia matches the closed form Sum(stifr (dt_min/sca)^2/2 - I)."""
    model = _starter(make_deck, "M40CST", _one_shell(1))
    ctl = _CstControls()
    from pyradioss.elements import KERNELS
    n = model.numnod
    fint, mint = np.zeros((n, 3)), np.zeros((n, 3))
    claims = []
    for name, group in model.element_groups():
        claims.append(KERNELS[name].forces(group, model.x, model.v,
                                           model.vr, 0.0, fint, mint))
    # natural nodal dt of this element (tra == rot by the mirror)
    probe = NodalTimeStep(model, _Controls(), MessageLog())
    probe.assemble(claims)
    m_eff = model.mass.copy()
    inv_i = np.where(model.inertia > 0, 1.0 /
                     np.maximum(model.inertia, 1e-30), 0.0)
    dt_nat = probe.apply(m_eff, 1.0 / m_eff, model.v, 0.0,
                         model.inertia.copy(), inv_i.copy())

    ctl.dt_min = 1.3 * 0.9 * dt_nat               # target above natural
    noda = NodalTimeStep(model, ctl, MessageLog())
    noda.assemble(claims)
    inertia = model.inertia.copy()
    inv_inertia = np.where(inertia > 0, 1.0 /
                           np.maximum(inertia, 1e-30), 0.0)
    stifr_before = noda.stifr.copy()
    mass_eff = model.mass.copy()
    i_before = inertia.copy()
    got = noda.apply(mass_eff, 1.0 / mass_eff, model.v, 0.0,
                     inertia, inv_inertia)
    assert got == pytest.approx(ctl.dt_min / 0.9, rel=1e-12)
    assert noda.iner_added > 0.0
    loaded = stifr_before > 0.0
    expect_di = np.maximum(
        stifr_before[loaded] * (ctl.dt_min / 0.9) ** 2 / 2.0
        - i_before[loaded], 0.0).sum()
    assert noda.iner_added == pytest.approx(expect_di, rel=1e-12)
    assert inertia[loaded] == pytest.approx(
        stifr_before[loaded] * (ctl.dt_min / 0.9) ** 2 / 2.0, rel=1e-12)
    # the inverse the integrator uses was refreshed in place
    assert inv_inertia[loaded] == pytest.approx(1.0 / inertia[loaded],
                                                rel=1e-12)
