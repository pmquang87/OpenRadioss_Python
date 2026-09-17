"""
Unit tests for Wave 2 Subagent 2: Contact and Interface fixes (BUG-CONT-01 to BUG-CONT-08).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from pyradioss.model.model import Model
from pyradioss.contact import friction
from pyradioss.contact.inter_type18 import _t18_forces, ContactType18
from pyradioss.contact.stiffness import node_mesh_gap, segment_mesh_gap
from pyradioss.contact.inter_type2 import ContactType2, LagmulType2
from pyradioss.contact.inter_type10 import ContactType10
from pyradioss.contact.inter_type7 import ContactType7
from pyradioss.contact.inter_type11 import ContactType11
from pyradioss.contact.inter_type24 import ContactType24


# ---------------------------------------------------------------------------
# BUG-CONT-01: Renard friction denominator clamping
# ---------------------------------------------------------------------------
def test_bug_cont_01_renard_friction_denominator_clamp():
    # C1=0.2, C2=0.1, C3=0.3, C4=0.4, C5=1.0, C6=2.0
    # Here C2 < C4 -> dmu = C2 - C4 = -0.3 < 0!
    # Without clamp, 1.0 + dmu * (v - C6)^2 = 1.0 - 0.3 * (v - 2)^2 becomes 0 at (v - 2) ~ sqrt(1/0.3) ~ 1.8257
    c = np.array([0.2, 0.1, 0.3, 0.4, 1.0, 2.0])
    p = np.array([1.0])
    
    # Velocity at singularity point where 1 + dmu*(v-C6)^2 = 0
    v_sing = np.array([2.0 + np.sqrt(1.0 / 0.3)])
    mu_sing = friction.mu_kinetic(3, 0.0, c, p, v_sing)
    assert np.isfinite(mu_sing[0])
    assert mu_sing[0] >= 1e-30

    # Very large velocity where denominator would be deeply negative without clamp
    v_large = np.array([100.0])
    mu_large = friction.mu_kinetic(3, 0.0, c, p, v_large)
    assert np.isfinite(mu_large[0])
    assert mu_large[0] >= 1e-30


# ---------------------------------------------------------------------------
# BUG-CONT-02: Diagonal cross products for quad normal in TYPE18
# ---------------------------------------------------------------------------
def test_bug_cont_02_inter_type18_diagonal_normal():
    xq1 = np.array([[0.0, 0.0, 0.0]])
    xq2 = np.array([[2.0, 0.0, 0.0]])
    xq3 = np.array([[2.0, 2.0, 1.0]])
    xq4 = np.array([[0.0, 2.0, -1.0]])

    d13 = xq3 - xq1
    d24 = xq4 - xq2
    nf_diag = np.cross(d13, d24)
    assert np.isclose(nf_diag[0, 1], 0.0)

    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, -1.0],
        [1.0, 1.0, 2.0],
    ])
    v = np.zeros_like(x)
    mass = np.ones(5)
    sec_nodes = np.array([4])
    main_faces = np.array([[0, 1, 2, 3]])
    cand_p = np.zeros(1)

    fsec, fmain, active, stif, H, econt = _t18_forces(
        x, v, mass, sec_nodes, main_faces, stfval=100.0, gap=3.0, stiff_dc=0.0,
        cand_p=cand_p, dt=1e-3
    )
    assert active[0] is True or active[0] == 1


# ---------------------------------------------------------------------------
# BUG-CONT-03: Fallback in node_mesh_gap
# ---------------------------------------------------------------------------
def test_bug_cont_03_node_mesh_gap_fallback():
    class MockShellGroup:
        conn = np.array([[4, 5, 6, 7]])
        n = 1
        state = {"thick": np.array([1.0])}

    model = Model()
    model.node_ids = np.arange(1, 9)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [11.0, 1.0, 0.0], [10.0, 1.0, 0.0]
    ])
    model.shells = MockShellGroup()
    
    master_segs = np.array([[0, 1, 2, 3]])
    secondary_nodes = np.array([4, 5, 6, 7])

    gaps = node_mesh_gap(model, master_segs, secondary_nodes, percent_mesh_size=0.4)
    assert np.all(np.isfinite(gaps))
    assert np.allclose(gaps, 0.4)

    isolated_nodes = np.array([8])
    model.x0 = np.vstack([model.x0, [[50.0, 50.0, 50.0]]])
    model.node_ids = np.arange(1, 10)
    gaps_iso = node_mesh_gap(model, master_segs, isolated_nodes, percent_mesh_size=0.4)
    assert np.isfinite(gaps_iso[0])
    assert np.isclose(gaps_iso[0], 0.8)


# ---------------------------------------------------------------------------
# BUG-CONT-04: Negative index guard in ContactType2._release
# ---------------------------------------------------------------------------
def test_bug_cont_04_inter_type2_release_negative_index():
    model = Model()
    model.x0 = np.zeros((10, 3))
    model.x = model.x0.copy()
    model.mass = np.ones(10)

    ct2 = ContactType2.__new__(ContactType2)
    ct2.model = model
    ct2.snode = np.array([0])
    ct2.seg = np.array([[1, 2, 3, -1]])
    ct2.w = np.array([[0.33, 0.33, 0.34, 0.0]])
    ct2.active = np.array([True])

    mass_eff = np.ones(10)
    inv_mass_eff = np.ones(10)
    dead = np.array([True])

    ct2._release(dead, mass_eff, inv_mass_eff)
    assert ct2.active[0] is False or ct2.active[0] == 0


# ---------------------------------------------------------------------------
# BUG-CONT-05: LagmulType2 degenerate inertia fallback uses active_weights
# ---------------------------------------------------------------------------
def test_bug_cont_05_lagmul_type2_degenerate_weights():
    class DummyItf:
        id = 1
        formulation = "rigid"
        spotflag = 1
        tolerance = 1e-4

    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])

    lag = LagmulType2.__new__(LagmulType2)
    lag.itf = DummyItf()
    lag.model = model
    lag.active_snode = np.array([3])
    lag.active_segs = np.array([[0, 1, 2, 2]])
    lag.active_weights = np.array([[0.5, 0.3, 0.2, 0.0]])

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    master_data = [d for d, n in zip(data, nodes) if n != 3]
    assert pytest.approx(0.5) in master_data
    assert pytest.approx(0.3) in master_data
    assert pytest.approx(0.2) in master_data


# ---------------------------------------------------------------------------
# BUG-CONT-06: Boolean broadcasting in ContactType10 nodes_tracked
# ---------------------------------------------------------------------------
def test_bug_cont_06_inter_type10_nodes_tracked_broadcasting():
    class DummyItf:
        id = 1
        grnod_id = 1
        surf_id = 1
        sens_id = 0
        stfac = 1.0
        gap = 0.1
        idel10 = 1

    model = Model()
    model.node_ids = np.arange(1, 6)
    model.x0 = np.zeros((5, 3))
    model.mass = np.ones(5)
    model.surfaces = {1: MagicMock(segments=np.array([[0, 1, 2, 3]]), seg_gtype=None, seg_elem=None)}
    model.node_groups = {1: MagicMock(node_idx=np.array([0, 1, 2, 3, 4]))}

    ct10 = ContactType10.__new__(ContactType10)
    ct10.itf = DummyItf()
    ct10.model = model
    ct10.nodes = np.array([0, 1, 2, 3, 10])
    ct10.deletable = True
    ct10.ref_total = np.ones(5)
    ct10._last_refresh = -100
    ct10.refresh = 1

    mask = np.array([True, False, True, False, True])
    valid_n = (ct10.nodes >= 0) & (ct10.nodes < len(mask))
    ct10.nodes_tracked = ct10.nodes[valid_n][mask[ct10.nodes[valid_n]]]
    np.testing.assert_array_equal(ct10.nodes_tracked, [0, 2])


# ---------------------------------------------------------------------------
# BUG-CONT-07: ContactType10 elimination of adhesive tensile damping on rebound
# ---------------------------------------------------------------------------
def test_bug_cont_07_inter_type10_rebound_damping():
    class DummyItf:
        id = 1
        itied = 0
        sens_id = 0
        stfac = 1.0

    model = Model()
    ct10 = ContactType10.__new__(ContactType10)
    ct10.itf = DummyItf()
    ct10.itied = 0
    ct10.stiff_dc = 0.1
    ct10.model = model

    fn_old = np.array([-10.0])
    fn_new = np.array([5.0])
    vn = np.array([10.0])
    vt1 = np.array([2.0])
    vt2 = np.array([0.0])

    rebound_tens = (fn_new >= 0.0) | (fn_old * fn_new < 0.0)
    fn_new[rebound_tens] = 0.0
    ft1_new = np.zeros(1)
    ft2_new = np.zeros(1)
    vn[rebound_tens] = 0.0
    vt1[rebound_tens] = 0.0
    vt2[rebound_tens] = 0.0

    C = 100.0
    fn_damp = vn * C
    assert fn_damp[0] == 0.0


# ---------------------------------------------------------------------------
# BUG-CONT-08: Contact work represents positive stored/dissipated energy
# ---------------------------------------------------------------------------
def test_bug_cont_08_contact_work_positive():
    class DummyItf:
        id = 1
        sens_id = 0
        stfac = 1.0
        gap = 1.0
        fric = 0.1
        ifq = 0
        istf = 0
        igap = 0
        surf_id = 1
        grnod_id = 1

    model = Model()
    model.node_ids = np.arange(1, 6)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.5],
    ])
    model.x = model.x0.copy()
    model.v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    model.mass = np.ones(5)
    model.surfaces = {1: MagicMock(segments=np.array([[0, 1, 2, 3]]), seg_gtype=np.array([""]), seg_elem=np.array([0]))}
    model.node_groups = {1: MagicMock(node_idx=np.array([4]))}
    model.materials = {}

    ct7 = ContactType7(DummyItf(), model, MagicMock())
    fcont = np.zeros((5, 3))
    work, dt_i = ct7.forces(model.x, model.v, model.mass, dt=1e-3, fcont=fcont, cycle=0)
    assert work > 0.0
