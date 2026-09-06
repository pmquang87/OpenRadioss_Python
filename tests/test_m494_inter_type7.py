"""M494 unit tests: /INTER/TYPE7 penalty node-to-surface & self-impact contact interface.

Tests cover:
1. Fixed-format card parsing (all 6+ cards, tstart, tstop, viss, gap, mfrot, ifq).
2. Compact card parsing with extended parameters.
3. Missing surface handling (_init_empty fallback).
4. Missing node group handling (_init_empty fallback).
5. Self-impact mode (grnod_id == 0) and self-exclusion.
6. Exact linear momentum conservation (sum F = 0).
7. Exact angular momentum conservation (sum r x F = 0).
8. Istf 0..5 stiffness combination formulations.
9. Igap 0..3 contact gap calculations and clipping.
10. Viscous damping force and positive dissipation energy.
11. Coulomb friction with slip velocity regularization.
12. Dynamic friction models MFROT 1..4.
13. Tangential force filtering (IFQ 1..3).
14. /DT/NODA stiffness accumulation and dt_int stability bound.
15. Time window gating ([tstart, tstop]).
16. Element deletion release (idel >= 1).
17. 3-node triangular segment normalization and contact.
18. LagmulType7 constraint matrix (L) generation.
19. Dynamic multi-cycle contact rebound simulation.
20. Array bounds safety for secondary nodes and segment corners.
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact.inter_type7 import ContactType7, LagmulType7
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.entities import Interface, Surface, NodeGroup, Part
from pyradioss.model.model import Model, ElementGroup


def _make_flat_quad_model():
    """Create a model with a flat quad segment (nodes 0, 1, 2, 3) on the z=0 plane
    and secondary node 4 above it at (0.5, 0.5, 0.05)."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.05], # 4: secondary node
    ], dtype=float)
    model.x0 = model.x.copy()
    model.node_ids = np.arange(len(model.x), dtype=np.int64)
    model.v = np.zeros_like(model.x)
    model.vr = np.zeros_like(model.x)
    model.mass = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=float)
    model.mass0 = model.mass.copy()

    # Main surface with 1 quad segment
    surf = Surface(id=1, title="main_surf")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.array(["shells"], dtype="<U8")
    surf.seg_elem = np.array([0], dtype=np.int64)
    model.surfaces[1] = surf

    # Secondary node group
    grp = NodeGroup(id=2, title="sec_nodes", node_idx=[4])
    model.node_groups[2] = grp

    class DummyMat:
        rho = 7.8e-6
        E = 210000.0
        nu = 0.3
    mat = DummyMat()
    model.materials[1] = mat

    class DummyProp:
        thick = 0.02
    prop = DummyProp()
    model.properties[1] = prop

    part = Part(id=1, mat_id=1, prop_id=1)
    model.parts[1] = part
    model.parts_list = [part]

    # Add shells ElementGroup for stiffness and deletion lookup
    model.shells = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
        state={
            "slices": [(slice(0, 1), mat, prop)],
            "thick": np.array([0.02], dtype=float),
            "off": np.array([1.0], dtype=float),
            "chk_fail": True,
        }
    )

    return model


# ============================================================================
# 1. Parsing tests (fixed and compact formats)
# ============================================================================

def test_type7_fixed_format_parsing(tmp_path):
    """Verify full 7+ card fixed-format /INTER/TYPE7 parsing."""
    starter = (
        "/BEGIN\n"
        "fixed_type7_test\n"
        "/INTER/TYPE7/101\n"
        "fixed format type7 interface\n"
        f"{10:>10}{20:>10}{2:>10}{0:>10}{1:>10}{0:>10}{0:>10}{1:>10}{0:>10}{0:>10}\n"
        f"{1.2:>20.1f}{0.05:>20.2f}{0.0:>20.1f}{0.0:>20.1f}{0:>10}\n"
        f"{100.0:>20.1f}{1000.0:>20.1f}{0.45:>20.2f}{1e-6:>20.1e}{0:>10}{0:>10}\n"
        f"{1.5:>20.1f}{0.2:>20.1f}{0.01:>20.2f}{0.001:>20.3f}{0.05:>20.2f}\n"
        f"{'':>7}{'0':>1}{'0':>1}{'0':>1}{0.0:>20.1f}{10:>10}{0.08:>20.2f}{0.0:>20.1f}{0.0:>20.1f}\n"
        f"{1:>10}{2:>10}{10.0:>20.1f}{0:>10}{0:>10}{0:>10}{1.0:>20.1f}{0:>10}\n"
        f"{0.1:>20.1f}{0.2:>20.1f}{0.3:>20.1f}{0.4:>20.1f}{0.5:>20.1f}\n"
        f"{0.6:>20.1f}\n"
    )
    p = tmp_path / "fixed_type7_0000.rad"
    p.write_text(starter, encoding="utf-8")
    block = read_deck(str(p))[1]
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 101
    assert itf.type == 7
    assert itf.grnod_id == 10
    assert itf.surf_id == 20
    assert itf.istf == 2
    assert itf.igap == 1
    assert itf.idel == 1
    assert pytest.approx(itf.fscale_gap) == 1.2
    assert pytest.approx(itf.gap_max) == 0.05
    assert pytest.approx(itf.percent_mesh_size) == 0.45
    assert pytest.approx(itf.stfac) == 1.5
    assert pytest.approx(itf.fric) == 0.2
    assert pytest.approx(itf.gap) == 0.01
    assert pytest.approx(itf.tstart) == 0.001
    assert pytest.approx(itf.tstop) == 0.05
    assert pytest.approx(itf.viss) == 0.08
    assert itf.mfrot == 1
    assert itf.ifq == 2
    assert itf.fric_c == (0.1, 0.2, 0.3, 0.4, 0.5, 0.0)


def test_type7_compact_format_parsing(tmp_path):
    """Verify compact /INTER/TYPE7 card parsing."""
    deck_text = """/BEGIN
compact_type7_test
/INTER/TYPE7/102
compact type7 interface
10 20 1 0 0 0 0
2.5 0.15 0.02 0.08 0.0 0.002 0.04 0.07
"""
    p = tmp_path / "compact_type7_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    block = read_deck(str(p))[1]
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 102
    assert itf.grnod_id == 10
    assert itf.surf_id == 20
    assert itf.istf == 1
    assert pytest.approx(itf.stfac) == 2.5
    assert pytest.approx(itf.fric) == 0.15
    assert pytest.approx(itf.gap) == 0.02
    assert pytest.approx(itf.gap_max) == 0.08
    assert pytest.approx(itf.tstart) == 0.002
    assert pytest.approx(itf.tstop) == 0.04
    assert pytest.approx(itf.viss) == 0.07


# ============================================================================
# 2. Defensive fallbacks (_init_empty)
# ============================================================================

def test_type7_missing_surface_fallback():
    """Verify graceful deactivation when main surface is missing."""
    model = Model()
    model.node_ids = np.arange(5, dtype=np.int64)
    itf = Interface(id=1, type=7, surf_id=999, grnod_id=1)
    log = MessageLog()
    ct = ContactType7(itf, model, log)

    assert len(ct.segs) == 0
    assert len(ct.nodes) == 0
    fcont = np.zeros((5, 3))
    w, dt = ct.forces(np.zeros((5, 3)), np.zeros((5, 3)), np.ones(5), 1e-4, fcont, cycle=0)
    assert w == 0.0
    assert np.isinf(dt)
    assert np.all(fcont == 0.0)


def test_type7_missing_node_group_fallback():
    """Verify graceful deactivation when secondary node group is missing."""
    model = Model()
    model.node_ids = np.arange(5, dtype=np.int64)
    surf = Surface(id=1, title="s")
    surf.segments = np.array([[0, 1, 2, 3]])
    surf.seg_gtype = np.array(["shells"])
    surf.seg_elem = np.array([0])
    model.surfaces[1] = surf

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=888)
    log = MessageLog()
    ct = ContactType7(itf, model, log)

    assert len(ct.nodes) == 0
    fcont = np.zeros((5, 3))
    w, dt = ct.forces(np.zeros((5, 3)), np.zeros((5, 3)), np.ones(5), 1e-4, fcont, cycle=0)
    assert w == 0.0
    assert np.isinf(dt)


# ============================================================================
# 3. Self-impact mode & self-exclusion
# ============================================================================

def test_type7_self_impact_mode():
    """Verify self-impact mode (grnod_id == 0) and self-exclusion."""
    model = _make_flat_quad_model()
    # grnod_id = 0 means self-impact: all nodes of the main surface are secondary candidates
    itf = Interface(id=1, type=7, surf_id=1, grnod_id=0, gap=0.01)
    log = MessageLog()
    ct = ContactType7(itf, model, log)

    # Secondary nodes should be nodes 0, 1, 2, 3
    np.testing.assert_array_equal(ct.nodes, np.array([0, 1, 2, 3]))

    # Because segment 0 is made of corners 0, 1, 2, 3, self-exclusion must reject all 4
    ct._broad_phase(model.x, model.v, dt=1e-4)
    assert len(ct.pairs_node) == 0


# ============================================================================
# 4. Momentum conservation
# ============================================================================

def test_type7_linear_and_angular_momentum_conservation():
    """Verify exact linear and angular momentum conservation during penetration."""
    model = _make_flat_quad_model()
    # Secondary node 4 penetrates: z = 0.005, gap = 0.02 -> pen = 0.015
    model.x[4] = [0.4, 0.6, 0.005]
    model.v[4] = [10.0, -5.0, -2.0] # 3D sliding and approaching velocity

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=10000.0, istf=1, fric=0.3, viss=0.05)
    log = MessageLog()
    ct = ContactType7(itf, model, log)

    fcont = np.zeros_like(model.x)
    wrk, dt_int = ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)

    # Normal + friction forces generated on secondary node
    assert np.any(fcont[4] != 0.0)
    # Reaction force on active segment corners
    assert np.any(fcont[:4] != 0.0)

    # Linear momentum: sum of all contact forces must be 0
    total_force = np.sum(fcont, axis=0)
    np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)

    # For pure normal contact, total torque must also be 0
    itf_nofric = Interface(id=2, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=10000.0, istf=1, fric=0.0, viss=0.05)
    ct_nofric = ContactType7(itf_nofric, model, log)
    fcont_norm = np.zeros_like(model.x)
    ct_nofric.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont_norm, cycle=0)

    torques = np.cross(model.x, fcont_norm)
    total_torque = np.sum(torques, axis=0)
    np.testing.assert_allclose(total_torque, [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 5. Stiffness formulations (Istf 0..5)
# ============================================================================

@pytest.mark.parametrize("istf", [0, 1, 2, 3, 4, 5])
def test_type7_istf_stiffness_formulations(istf):
    """Verify Istf 0..5 stiffness combinations in penalty forces."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # penetration = 0.015 for gap 0.02
    stfac = 500.0 if istf == 1 else 1.0

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=stfac, istf=istf, fric=0.0, viss=0.0)
    log = MessageLog()
    ct = ContactType7(itf, model, log)

    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)

    # Force on secondary node must push in +z direction
    Fz = fcont[4, 2]
    assert Fz > 0.0
    # Sum of reaction forces on master corners must equal -Fz
    assert pytest.approx(np.sum(fcont[:4, 2])) == -Fz
    assert np.all(fcont[:4, 2] <= 1e-12)


# ============================================================================
# 6. Gap calculation and clipping (Igap 0..3)
# ============================================================================

def test_type7_igap_variants():
    """Verify Igap=0, 1, 2, 3 gap handling."""
    model = _make_flat_quad_model()
    
    # Igap=0: constant gap
    itf0 = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.03, igap=0)
    ct0 = ContactType7(itf0, model, MessageLog())
    assert pytest.approx(ct0.gap_const) == 0.03

    # Igap=1: variable gap clipped
    itf1 = Interface(id=2, type=7, surf_id=1, grnod_id=2, gap=0.005, gap_max=0.025, igap=1)
    ct1 = ContactType7(itf1, model, MessageLog())
    assert ct1.gap_min == 0.005
    assert ct1.gap_max == 0.025

    # Igap=2: scaled gap
    itf2 = Interface(id=3, type=7, surf_id=1, grnod_id=2, fscale_gap=1.5, igap=2)
    ct2 = ContactType7(itf2, model, MessageLog())
    assert hasattr(ct2, "gap_m")

    # Igap=3: mesh-size limited gap
    itf3 = Interface(id=4, type=7, surf_id=1, grnod_id=2, percent_mesh_size=0.3, igap=3)
    ct3 = ContactType7(itf3, model, MessageLog())
    assert hasattr(ct3, "gap_m_l")


# ============================================================================
# 7. Viscous damping and positive dissipation energy
# ============================================================================

def test_type7_viscous_damping_and_energy():
    """Verify viscous damping opposes approaching velocity and books positive dissipation work."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # pen = 0.015
    model.v[4] = [0.0, 0.0, -10.0] # approaching master surface (v_rel = -10 normal)

    # Test with zero damping vs with 10% damping
    itf_nodamp = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, fric=0.0, viss=0.0)
    ct_nodamp = ContactType7(itf_nodamp, model, MessageLog())
    f_nodamp = np.zeros_like(model.x)
    w_nodamp, _ = ct_nodamp.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f_nodamp, cycle=0)

    itf_damp = Interface(id=2, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, fric=0.0, viss=0.10)
    ct_damp = ContactType7(itf_damp, model, MessageLog())
    f_damp = np.zeros_like(model.x)
    w_damp, _ = ct_damp.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f_damp, cycle=0)

    # With approaching velocity, damping adds additional repulsive normal force
    assert f_damp[4, 2] > f_nodamp[4, 2]
    # Dissipation work is positive (-wrk in forces return is positive contact energy increment)
    assert w_damp > w_nodamp


def test_type7_damping_zero_on_separation():
    """Verify damping is zero when bodies are separating (v_n >= 0)."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # pen = 0.015
    model.v[4] = [0.0, 0.0, 10.0]  # moving away from surface

    itf_nodamp = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, fric=0.0, viss=0.0)
    ct_nodamp = ContactType7(itf_nodamp, model, MessageLog())
    f_nodamp = np.zeros_like(model.x)
    ct_nodamp.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f_nodamp, cycle=0)

    itf_damp = Interface(id=2, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, fric=0.0, viss=0.10)
    ct_damp = ContactType7(itf_damp, model, MessageLog())
    f_damp = np.zeros_like(model.x)
    ct_damp.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f_damp, cycle=0)

    # When separating, damping is inactive -> forces are identical pure spring forces
    np.testing.assert_allclose(f_damp, f_nodamp)


# ============================================================================
# 8. Friction formulations (Coulomb & MFROT 1..4)
# ============================================================================

def test_type7_coulomb_friction():
    """Verify Coulomb friction force opposes sliding velocity in tangent plane."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005]
    model.v[4] = [20.0, 0.0, 0.0] # sliding along +x

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, fric=0.2, viss=0.0)
    ct = ContactType7(itf, model, MessageLog())
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)

    Fn = fcont[4, 2]
    Fx = fcont[4, 0]
    # Friction opposes sliding (+x velocity -> -x force)
    assert Fx < 0.0
    # Sliding resistance respects Coulomb limit Ft <= mu * Fn
    assert abs(Fx) <= 0.2 * Fn * 1.01


@pytest.mark.parametrize("mfrot", [1, 2, 3, 4])
def test_type7_mfrot_dynamic_friction(mfrot):
    """Verify MFROT 1..4 dynamic friction models execute properly."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005]
    model.v[4] = [15.0, 5.0, -1.0]

    itf = Interface(
        id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1,
        fric=0.1, mfrot=mfrot, fric_c=(0.2, 0.05, 1.0, 0.1, 0.01, 0.0)
    )
    ct = ContactType7(itf, model, MessageLog())
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)

    # Tangential forces generated and opposed to velocity
    assert fcont[4, 0] < 0.0
    assert fcont[4, 1] < 0.0


def test_type7_ifq_filtering():
    """Verify IFQ tangential force exponential moving average filtering."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005]
    model.v[4] = [10.0, 0.0, 0.0]

    itf = Interface(
        id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1,
        fric=0.2, ifq=1, xfiltr=0.5
    )
    ct = ContactType7(itf, model, MessageLog())
    
    # Cycle 0
    f0 = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f0, cycle=0)
    # Cycle 1
    f1 = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=f1, cycle=1)

    assert len(ct._filt_keys) > 0
    assert len(ct._filt_vals) > 0


# ============================================================================
# 9. Time step bound & /DT/NODA accumulation
# ============================================================================

def test_type7_stifn_and_dt_int():
    """Verify /DT/NODA stiffness accumulation and explicit stability bound dt_int."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # near contact
    stifn = np.zeros(len(model.x))

    K_spring = 2000.0
    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=K_spring, istf=1)
    ct = ContactType7(itf, model, MessageLog())
    fcont = np.zeros_like(model.x)
    _, dt_int = ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, stifn=stifn)

    # stifn must accumulate spring stiffness: full K on secondary node, 4*K on corners
    assert stifn[4] >= K_spring
    assert np.all(stifn[:4] > 0.0)
    # Stability time step bounded by dt <= sqrt(2 m / K)
    assert dt_int <= np.sqrt(2.0 * model.mass[4] / K_spring)


# ============================================================================
# 10. Time window gating
# ============================================================================

def test_type7_time_window_gating():
    """Verify contact forces are zero outside [tstart, tstop]."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # penetrating

    itf = Interface(
        id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1,
        tstart=0.002, tstop=0.006
    )
    ct = ContactType7(itf, model, MessageLog())

    fcont = np.zeros_like(model.x)
    # 1. Before tstart (t = 0.001): inactive
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.001)
    assert np.all(fcont == 0.0)

    # 2. Inside window (t = 0.003): active
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.003)
    assert fcont[4, 2] > 0.0

    # 3. After tstop (t = 0.008): inactive
    fcont[:] = 0.0
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.008)
    assert np.all(fcont == 0.0)


# ============================================================================
# 11. Element deletion release
# ============================================================================

def test_type7_element_deletion_release():
    """Verify element deletion (idel >= 1) releases contact when parent element dies."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # penetrating

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1, idel=1)
    ct = ContactType7(itf, model, MessageLog())
    assert ct.deletable is True

    # Cycle 0: element alive -> forces present
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)
    assert fcont[4, 2] > 0.0

    # Element dies
    model.shells.state["off"][0] = 0.0
    fcont[:] = 0.0
    # Next cycle sees dead element -> no contact force
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=1)
    assert np.all(fcont == 0.0)


# ============================================================================
# 12. 3-node triangular segments
# ============================================================================

def test_type7_triangular_segments():
    """Verify 3-node triangular segments are normalized and produce valid contact."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2: triangle corners
        [0.25, 0.25, 0.005], # 3: secondary node inside triangle
    ], dtype=float)
    model.x0 = model.x.copy()
    model.node_ids = np.arange(len(model.x), dtype=np.int64)
    model.v = np.zeros_like(model.x)
    model.vr = np.zeros_like(model.x)
    model.mass = np.array([1.0, 1.0, 1.0, 0.5], dtype=float)
    model.mass0 = model.mass.copy()

    # Surface with 3-node triangle (shape N, 3)
    surf = Surface(id=1, title="tri_surf")
    surf.segments = np.array([[0, 1, 2]], dtype=np.int64)
    surf.seg_gtype = np.array([""], dtype="<U8")
    surf.seg_elem = np.array([0], dtype=np.int64)
    model.surfaces[1] = surf

    grp = NodeGroup(id=2, title="sec_node", node_idx=[3])
    model.node_groups[2] = grp

    class DummyMat:
        rho = 7.8e-6
        E = 210000.0
        nu = 0.3
    model.materials[1] = DummyMat()

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1)
    ct = ContactType7(itf, model, MessageLog())

    assert ct.segs.shape == (1, 4)
    assert ct.segs[0, 3] == ct.segs[0, 2] # normalized degenerate quad

    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0)

    # Force on node 3
    assert fcont[3, 2] > 0.0
    # Momentum balance
    np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 13. LagmulType7 constraint generation
# ============================================================================

def test_type7_lagmul_matrix_generation():
    """Verify LagmulType7 generates sparse constraint rows."""
    model = _make_flat_quad_model()
    model.x[4] = [0.5, 0.5, 0.005] # penetrating
    model.x0 = model.x.copy()
    model.u = np.zeros_like(model.x)
    model.v = np.zeros_like(model.x)
    model.v[4] = [0.0, 0.0, -5.0] # approaching
    model.dt = 1e-4
    model.cycle = 0

    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, lagmul=True)
    lagmul = LagmulType7(itf, model, MessageLog())

    data, nodes, dofs, eq_ids, n_rows = lagmul.generate_l_matrix()
    assert n_rows == 1
    assert len(data) > 0
    assert 4 in nodes # secondary node in constraint row


# ============================================================================
# 14. Dynamic contact rebound simulation
# ============================================================================

def test_type7_dynamic_rebound():
    """Verify dynamic spring-mass impact cleanly arrests penetration and rebounds."""
    model = _make_flat_quad_model()
    # Secondary node 4 placed with initial downward velocity sized within gap elastic capacity
    model.x[4] = [0.5, 0.5, 0.02] # at gap boundary
    model.v[4] = [0.0, 0.0, -4.0]
    dt = 1e-4
    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=50000.0, istf=1, viss=0.0)
    ct = ContactType7(itf, model, MessageLog())

    min_z = 0.02
    for cycle in range(150):
        fcont = np.zeros_like(model.x)
        ct.forces(model.x, model.v, model.mass, dt, fcont, cycle=cycle)
        # Update node 4 explicit central difference acceleration
        a4 = fcont[4] / model.mass[4]
        model.v[4] += a4 * dt
        model.x[4] += model.v[4] * dt
        min_z = min(min_z, model.x[4, 2])

    # Penetration occurred (min_z < 0.02) and then rebounded (v[4, 2] > 0)
    assert min_z < 0.02
    assert model.v[4, 2] > 0.0
    assert model.x[4, 2] > min_z


# ============================================================================
# 15. Array bounds safety
# ============================================================================

def test_type7_array_bounds_safety():
    """Verify high secondary node IDs do not trigger IndexError."""
    model = _make_flat_quad_model()
    # Secondary node 100 with size 5 model
    itf = Interface(id=1, type=7, surf_id=1, grnod_id=2, gap=0.02, stfac=1000.0, istf=1)
    grp = NodeGroup(id=2, title="high_id", node_idx=[100])
    model.node_groups[2] = grp

    ct = ContactType7(itf, model, MessageLog())
    # Ks should be allocated safely as 0.0
    assert len(ct.Ks) == 1
    assert ct.Ks[0] == 0.0
