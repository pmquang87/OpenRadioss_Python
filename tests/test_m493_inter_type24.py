"""
Unit tests for /INTER/TYPE24 penalty contact interface hardening (M493).

Validates:
1. Deck parsing: fixed-format 6-card layout and compact layout (/INTER/TYPE24).
2. Interface modes: self-impact (grnod_id=0, surf_id1=0), surface-to-surface (surf_id1>0),
   and node-to-surface (grnod_id>0).
3. Momentum conservation: exact linear momentum sum(F) = 0 and angular momentum sum(r x F) = 0.
4. Penalty stiffness formulations (Istf 0..5) and stfac scaling.
5. Gap calculation: constant gap (Igap=0) vs variable gap (Igap=1) with gap_min / gap_max clipping.
6. Viscous damping: active during approach (vn < 0), inactive during separation (vn > 0).
7. Coulomb friction and MFROT 1..4 dynamic friction models.
8. Tangential force filtering (IFQ 1..3) exponential moving average.
9. /DT/NODA stiffness accumulation into stifn and interface time step bound dt_int.
10. Triangular segments (3-node quads / SH3N).
11. Dynamic element deletion tracking (idel >= 1).
12. Time window gating [tstart, tstop] and dt <= 0 safety.
13. Defensive fallback for empty / missing surface or node group.
14. Factory integration via build_contacts().
"""

from __future__ import annotations

import contextlib
import io
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.contact.inter_type24 import ContactType24
from pyradioss.contact import build_contacts
from pyradioss.model.entities import Interface
from pyradioss.starter.starter import run_starter


STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

SHELL_PROPS = """\
/PROP/SHELL/1
lower prop
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.1
/PROP/SHELL/2
upper prop
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.1
"""

BASE_DECK = (
    "/BEGIN\n"
    "base contact deck\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "11 0.25 0.25 0.05\n12 0.75 0.25 0.05\n"
    "13 0.75 0.75 0.05\n14 0.25 0.75 0.05\n"
    "/SHELL/1\n1 1 2 3 4\n"
    "/SHELL/2\n2 11 12 13 14\n"
    "/PART/1\nlower\n1 1\n"
    "/PART/2\nupper\n2 1\n"
    + STEEL_LAW1
    + SHELL_PROPS
    + "/GRNOD/PART/2\nupper nodes\n2\n"
    "/SURF/PART/1\nlower surf\n1\n"
    "/SURF/PART/2\nupper surf\n2\n"
)


def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


# ============================================================================
# 1. Deck Parsing (Fixed and Compact)
# ============================================================================

def test_type24_fixed_format_parsing(make_deck):
    """Verify parsing of 6-card fixed format /INTER/TYPE24 deck."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/10\n"
        "full fixed type24\n"
        + f"{0:>10}{1:>10}{2:>10}{0:>10}{0:>10}{0:>10}{1:>10}{0:>10}{0:>10}\n"
        + f"{2:>10}{'':>20}{0:>10}{'':>20}{0.0:>20.1f}{0.0:>20.1f}\n"
        + f"{0.0:>20.1f}{0.0:>20.1f}{1:>10}{0:>10}{0.0:>20.1f}{0.0:>20.1f}\n"
        + f"{1.5:>20.1f}{0.2:>20.1f}{0.0:>20.1f}{0.01:>20.2f}{0.08:>20.2f}\n"
        + f"{'':>7}{'0':>1}{'0':>1}{'0':>1}{'':>20}{0:>10}{0.08:>20.2f}{0.0:>20.1f}{0.0:>20.1f}\n"
        + f"{1:>10}{1:>10}{0.5:>20.1f}{0:>10}{0:>10}{0.0:>20.1f}{0.0:>20.1f}{0:>10}\n"
        + "0.1 0.2 0.3 0.4 0.5\n"
        + "/END\n"
    )
    model = _starter_only(make_deck, "FIX24", starter)
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 10
    assert itf.type == 24
    assert itf.grnod_id == 2
    assert itf.surf_id == 1
    assert itf.istf == 2
    assert itf.igap == 1
    assert itf.idel == 1
    assert itf.stfac == pytest.approx(1.5)
    assert itf.fric == pytest.approx(0.2)
    assert itf.tstart == pytest.approx(0.01)
    assert itf.tstop == pytest.approx(0.08)
    assert itf.stiff_dc == pytest.approx(0.08)
    assert itf.mfrot == 1
    assert itf.ifq == 1
    assert itf.xfiltr == pytest.approx(0.5)
    assert itf.fric_c[:5] == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5))


def test_type24_compact_format_parsing(make_deck):
    """Verify parsing of compact 2-card format /INTER/TYPE24."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\n"
        "compact\n"
        "2 1 1 0 0 0 0 0 1\n"
        "2.5 0.15 0.05 0.0 0.0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "CMP24", starter)
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 1
    assert itf.type == 24
    assert itf.grnod_id == 2
    assert itf.surf_id == 1
    assert itf.istf == 1
    assert itf.stfac == pytest.approx(2.5)
    assert itf.fric == pytest.approx(0.15)
    assert itf.gap == pytest.approx(0.05)
    assert itf.idel == 1


# ============================================================================
# 2. Defensive Fallbacks for Missing / Empty Surface or Node Group
# ============================================================================

def test_type24_empty_or_missing_surface_fallback(make_deck):
    """Missing or empty main surface must trigger _init_empty() gracefully."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ncompact\n2 1 1 0\n1.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "MSF", starter)
    itf = Interface(id=99, type=24, grnod_id=2, surf_id=999)
    log = MessageLog()
    ct = ContactType24(itf, model, log)
    assert len(ct.segs) == 0
    assert len(ct.nodes) == 0
    assert ct.dt_bound == np.inf

    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert wrk == 0.0
    assert dt_i == np.inf
    assert np.all(fcont == 0.0)


def test_type24_empty_or_missing_node_group_fallback(make_deck):
    """Missing or empty secondary node group must trigger _init_empty()."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ncompact\n2 1 1 0\n1.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "MNG", starter)
    itf = Interface(id=99, type=24, grnod_id=999, surf_id=1)
    log = MessageLog()
    ct = ContactType24(itf, model, log)
    assert len(ct.nodes) == 0
    assert ct.dt_bound == np.inf

    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert wrk == 0.0
    assert dt_i == np.inf
    assert np.all(fcont == 0.0)


# ============================================================================
# 3. Interface Modes: Self-Impact and Surface-to-Surface
# ============================================================================

def test_type24_surface_to_surface_mode(make_deck):
    """Surface-to-surface mode (grnod_id=0, surf_id1=2): secondary nodes extracted from surface 2."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\nsurf2surf\n0 1 1 0\n10.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "S2S", starter)
    # Set surf_id1 = 2 on itf
    model.interfaces[0].surf_id1 = 2
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    assert np.array_equal(ct.nodes, np.sort(sec_nodes))

    # Upper plate is at z = 0.05, gap = 0.1 -> penetrates 0.05
    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    # Secondary nodes pushed +Z, main corners pushed -Z
    assert np.all(fcont[sec_nodes, 2] > 0.0)
    main_nodes = model.node_indices([1, 2, 3, 4])
    assert np.all(fcont[main_nodes, 2] < 0.0)
    # Exact linear momentum balance
    assert np.abs(fcont.sum(axis=0)).max() < 1e-12


def test_type24_self_impact_mode(make_deck):
    """Self-impact mode (grnod_id=0, surf_id1=0): secondary nodes are main surface nodes."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\nself\n0 1 1 0\n10.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "SLF", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    main_nodes = model.node_indices([1, 2, 3, 4])
    assert np.array_equal(ct.nodes, np.sort(main_nodes))

    # Self-exclusion: a node cannot impact a segment of which it is a corner
    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert np.all(fcont == 0.0)


# ============================================================================
# 4. Exact Linear and Angular Momentum Conservation
# ============================================================================

def test_type24_momentum_conservation_inclined(make_deck):
    """Off-center impact on an inclined segment must conserve both linear and angular momentum."""
    starter = (
        "/BEGIN\ninclined impact\n"
        "/NODE\n"
        "1 0 0 0\n2 2 0 1\n3 2 2 1\n4 0 2 0\n"
        "10 0.6 0.8 0.2\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n"
        + STEEL_LAW1
        + "/PROP/SHELL/1\nsh\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/SURF/PART/1\nplate surf\n1\n"
        "/GRNOD/NODE/2\nprobe\n10\n"
        "/INTER/TYPE24/1\nimpact\n2 1 1 0\n50.0 0.0 0.2\n/END\n"
    )
    model = _starter_only(make_deck, "INC", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    p_idx = model.node_indices([10])[0]
    assert np.linalg.norm(fcont[p_idx]) > 0.0
    # Exact linear momentum: sum(F) = 0
    assert np.allclose(fcont.sum(axis=0), 0.0, atol=1e-12)
    # Exact angular momentum about origin: sum(r x F) = 0
    torque = np.cross(model.x, fcont).sum(axis=0)
    assert np.allclose(torque, 0.0, atol=1e-12)


# ============================================================================
# 5. Penalty Stiffness Laws (Istf 0..5)
# ============================================================================

@pytest.mark.parametrize("istf", [0, 1, 2, 3, 4, 5])
def test_type24_istf_combinations(make_deck, istf):
    """Verify that all Istf settings (0=main, 1=constant, 2=avg, 3=max, 4=min, 5=series) produce positive force."""
    stfac = 20.0
    starter = (
        BASE_DECK
        + f"/INTER/TYPE24/1\nimpact\n2 1 {istf} 0\n{stfac} 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, f"STF{istf}", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    sec_nodes = model.node_indices([11, 12, 13, 14])
    assert np.all(fcont[sec_nodes, 2] > 0.0)
    assert np.allclose(fcont.sum(axis=0), 0.0, atol=1e-12)

    if istf == 1:
        # F = K * pen = 20.0 * 0.05 = 1.0 each
        assert np.allclose(fcont[sec_nodes, 2], 1.0, atol=1e-10)


# ============================================================================
# 6. Gap Calculation: Constant (Igap=0) vs Variable (Igap=1) with Clipping
# ============================================================================

def test_type24_gap_calculation_and_clipping(make_deck):
    """Verify Igap=1 variable gap and gap_min / gap_max clipping."""
    # Shell thickness = 0.1 each -> physical half-thickness sum = 0.05 + 0.05 = 0.10
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ngap test\n2 1 1 1\n10.0 0.0 0.12 0.20\n/END\n"
    )
    model = _starter_only(make_deck, "GAPCLP", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)
    assert ct.gap_bound >= 0.12

    sec_nodes = model.node_indices([11, 12, 13, 14])
    # Place nodes at z = 0.11: penetrates gap of 0.12
    model.x[sec_nodes, 2] = 0.11
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert np.all(fcont[sec_nodes, 2] > 0.0)

    # Now test gap_max clipping with gap_max = 0.08
    itf_max = Interface(id=2, type=24, grnod_id=2, surf_id=1, istf=1, stfac=10.0,
                        igap=1, gap=0.01, gap_max=0.08)
    ct_max = ContactType24(itf_max, model, log)
    # With gap_max=0.08, z=0.11 is beyond gap_max -> no contact
    fcont.fill(0.0)
    ct_max.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert np.all(fcont == 0.0)


# ============================================================================
# 7. Viscous Damping: Approaching vs Separating Motion
# ============================================================================

def test_type24_viscous_damping_approach_vs_separation(make_deck):
    """Viscous damping must oppose approach (vn < 0) and be zero during separation (vn > 0)."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ndamp\n2 1 1 0\n100.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "DMP", starter)
    model.interfaces[0].stiff_dc = 0.1
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])

    # Static penetration
    fcont_stat = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont_stat, 0)
    f_stat = fcont_stat[sec_nodes[0], 2]

    # Approaching motion (vz = -10.0)
    model.v[sec_nodes, 2] = -10.0
    fcont_app = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont_app, 0)
    f_app = fcont_app[sec_nodes[0], 2]
    assert f_app > f_stat

    # Separating motion (vz = +10.0)
    model.v[sec_nodes, 2] = +10.0
    fcont_sep = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont_sep, 0)
    f_sep = fcont_sep[sec_nodes[0], 2]
    assert f_sep == pytest.approx(f_stat)


# ============================================================================
# 8. Friction and Dynamic MFROT Models
# ============================================================================

def test_type24_coulomb_friction(make_deck):
    """Coulomb friction must oppose relative tangential velocity and bound Ft <= mu * Fn."""
    mu = 0.3
    starter = (
        BASE_DECK
        + f"/INTER/TYPE24/1\nfric\n2 1 1 0\n100.0 {mu} 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "FRIC", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    model.v[sec_nodes, 0] = 2.0  # Sliding +X
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    fn = fcont[sec_nodes[0], 2]
    assert fn > 0.0
    fx = fcont[sec_nodes[0], 0]
    assert fx < 0.0  # Opposing motion
    assert abs(fx) <= mu * fn * (1.0 + 1e-6)
    assert np.allclose(fcont.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("mfrot", [1, 2, 3, 4])
def test_type24_mfrot_friction_models(make_deck, mfrot):
    """Dynamic friction MFROT 1..4 must compute pressure/velocity-dependent friction."""
    fric_c = (0.2, 0.1, 10.0, 1.0, 0.5, 0.2)
    starter = (
        BASE_DECK
        + f"/INTER/TYPE24/1\nmfrot\n2 1 1 0 0 {mfrot} 0\n100.0 0.1 0.1\n"
        f"{' '.join(str(c) for c in fric_c)}\n/END\n"
    )
    model = _starter_only(make_deck, f"MFR{mfrot}", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    model.v[sec_nodes, 0] = 5.0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    assert fcont[sec_nodes[0], 2] > 0.0
    assert fcont[sec_nodes[0], 0] < 0.0
    assert np.allclose(fcont.sum(axis=0), 0.0, atol=1e-10)


# ============================================================================
# 9. IFQ Tangential Force Filtering
# ============================================================================

def test_type24_ifq_filtering(make_deck):
    """IFQ exponential filtering must smooth tangential friction force over cycles."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\nifq\n2 1 1 0 0 0 1\n100.0 0.4 0.1 0.0 0.5\n/END\n"
    )
    model = _starter_only(make_deck, "IFQ", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    model.v[sec_nodes, 0] = 10.0
    dt = 1e-4
    fcont1 = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt, fcont1, 0)
    fx1 = fcont1[sec_nodes[0], 0]

    # Change velocity abruptly: vx = 0.0
    model.v[sec_nodes, 0] = 0.0
    fcont2 = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt, fcont2, 1)
    fx2 = fcont2[sec_nodes[0], 0]

    assert abs(fx1) > 0.0
    assert abs(fx2) < abs(fx1)


# ============================================================================
# 10. Triangular Segments (3-node shells and negative 4th index)
# ============================================================================

def test_type24_triangular_segment_normalization(make_deck):
    """Triangles with negative 4th index [-1] must be normalized to degenerate quads [n0, n1, n2, n2]."""
    starter = (
        "/BEGIN\ntriangle contact\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n"
        "10 0.2 0.2 0.05\n"
        "/SH3N/1\n1 1 2 3\n"
        "/PART/1\ntri\n1 1\n"
        + STEEL_LAW1
        + "/PROP/SHELL/1\nsh\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/SURF/PART/1\ntri surf\n1\n"
        "/GRNOD/NODE/2\nprobe\n10\n"
        "/INTER/TYPE24/1\nimpact\n2 1 1 0\n50.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "TRI", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    # 4th corner should be normalized to corner 2
    assert ct.segs[0, 3] == ct.segs[0, 2]

    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    p_idx = model.node_indices([10])[0]
    assert fcont[p_idx, 2] > 0.0
    tri_nodes = model.node_indices([1, 2, 3])
    assert np.all(fcont[tri_nodes, 2] < 0.0)
    assert np.allclose(fcont.sum(axis=0), 0.0, atol=1e-12)


# ============================================================================
# 11. /DT/NODA Stiffness Accumulation and Interface Stability Bound
# ============================================================================

def test_type24_dt_noda_stiffness_accumulation(make_deck):
    """Contact springs must accumulate stiffness into stifn array for /DT/NODA mass scaling."""
    K = 80.0
    starter = (
        BASE_DECK
        + f"/INTER/TYPE24/1\nimpact\n2 1 1 0\n{K} 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "DTNODA", starter)
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    stifn = np.zeros(len(model.x), dtype=float)
    fcont = np.zeros_like(model.x)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0, stifn=stifn)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    assert np.allclose(stifn[sec_nodes], K)
    main_nodes = model.node_indices([1, 2, 3, 4])
    assert np.all(stifn[main_nodes] > 0.0)
    assert dt_i <= np.sqrt(2.0 * model.mass.min() / K) * (1.0 + 1e-10)


# ============================================================================
# 12. Time Window Gating and dt <= 0 Safety
# ============================================================================

def test_type24_time_window_gating_and_zero_dt(make_deck):
    """Interface must remain inactive outside [tstart, tstop] or when dt <= 0."""
    tstart, tstop = 0.005, 0.010
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ntime gate\n2 1 1 0\n10.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "TGATE", starter)
    model.interfaces[0].tstart = tstart
    model.interfaces[0].tstop = tstop
    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)

    sec_nodes = model.node_indices([11, 12, 13, 14])
    fcont = np.zeros_like(model.x)

    # 1. Before tstart (t = 0.002) -> inactive
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0, t=0.002)
    assert wrk == 0.0
    assert np.all(fcont == 0.0)

    # 2. Within window (t = 0.007) -> active
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0, t=0.007)
    assert np.all(fcont[sec_nodes, 2] > 0.0)

    # 3. After tstop (t = 0.015) -> inactive
    fcont.fill(0.0)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0, t=0.015)
    assert wrk == 0.0
    assert np.all(fcont == 0.0)

    # 4. dt <= 0.0 -> no-op
    fcont.fill(0.0)
    wrk, dt_i = ct.forces(model.x, model.v, model.mass, 0.0, fcont, 0, t=0.007)
    assert wrk == 0.0
    assert np.all(fcont == 0.0)


# ============================================================================
# 13. Dynamic Element Deletion Tracking
# ============================================================================

def test_type24_element_deletion(make_deck):
    """When parent element is deleted (idel >= 1), segment forces must vanish."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ndel\n2 1 1 0 0 0 0 0 1\n10.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "ELDEL", starter)
    # Set up failure state on shells so tracking.any_deletable sees it
    model.shells.state["off"] = np.ones(model.shells.n, dtype=float)
    model.shells.state["chk_fail"] = True

    log = MessageLog()
    ct = ContactType24(model.interfaces[0], model, log)
    assert ct.deletable is True

    sec_nodes = model.node_indices([11, 12, 13, 14])
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert np.all(fcont[sec_nodes, 2] > 0.0)

    # Delete shell 1 (lower plate)
    model.shells.state["off"][0] = 0.0
    fcont.fill(0.0)
    # Trigger refresh cycle
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 20)
    assert np.all(fcont == 0.0)


# ============================================================================
# 14. Factory Integration via build_contacts()
# ============================================================================

def test_type24_build_contacts_factory(make_deck):
    """build_contacts(model, log) must instantiate ContactType24 in penalty list."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE24/1\ncompact\n2 1 1 0\n10.0 0.0 0.1\n/END\n"
    )
    model = _starter_only(make_deck, "FACT", starter)
    log = MessageLog()
    penalty, tied = build_contacts(model, log)
    assert len(penalty) == 1
    assert isinstance(penalty[0], ContactType24)
    assert len(tied) == 0
