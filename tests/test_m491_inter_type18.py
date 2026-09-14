"""
Tests for M491: /INTER/TYPE18 Fluid-Structure Penalty Contact Interface.

Verifies:
  - Upstream Fortran alignment with i18dst3.F, i18for3.F, i18tri.F, multi_i18_force_pon.F.
  - Sub-triangle decomposition and partition of unity (sum(Hi) = 1.0) for quads.
  - Direct projection for 3-node triangular segments.
  - Linear momentum conservation (sum(F) = 0).
  - Angular momentum conservation (sum(r x F) = 0).
  - Dynamic penetration stiffness K = Stfac * (pene / gap).
  - Normal displacement accumulation in cand_p and reset on separation.
  - Viscous damping dissipation for vn > 0.
  - /DT/NODA stiffness accumulation and dt_interface bound.
  - Defensive error handling for empty/missing surfaces and node groups.
  - Support for node 0 in quad corners and negative node 4 in triangles.
  - Large cloud chunked search.
  - Starter keyword reading and build_contacts integration.
  - Multi-cycle dynamic engine contact simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import build_contacts
from pyradioss.contact.inter_type18 import ContactType18, _t18_forces
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.model import Interface, Model, NodeGroup, Surface


class TestSubtrianglePartitionOfUnity:
    """Test 4-subtriangle decomposition and partition of unity (i18for3.F)."""

    def test_quad_flat_partition_of_unity(self):
        """Verify sum(Hi) == 1.0 for arbitrary query points over a flat quad."""
        # Quad corners in z=0 plane: [0,0,0], [2,0,0], [2,2,0], [0,2,0]
        x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.2],   # Secondary node 4 (inside, slightly above)
            [0.5, 0.2, 0.1],   # Secondary node 5 (sub-tri 0)
            [1.8, 0.5, 0.1],   # Secondary node 6 (sub-tri 1)
            [1.2, 1.7, 0.1],   # Secondary node 7 (sub-tri 2)
            [0.3, 1.5, 0.1],   # Secondary node 8 (sub-tri 3)
            [3.0, 3.0, 0.1],   # Secondary node 9 (outside)
        ])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.array([4, 5, 6, 7, 8, 9])
        main_nodes = np.tile(np.array([0, 1, 2, 3]), (len(sec_nodes), 1))
        cand_p = np.zeros(len(sec_nodes))

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=100.0, gap=1.0, stiff_dc=0.0, cand_p=cand_p, dt=1e-3
        )

        # Every query point must have sum(H_i) == 1.0 within machine precision
        h_sums = np.sum(H, axis=1)
        np.testing.assert_allclose(h_sums, 1.0, rtol=1e-14, atol=1e-14)

        # Inside points must have 0 <= H_i <= 1
        for i in range(5):  # nodes 4..8 are inside the quad
            assert np.all(H[i] >= -1e-12)
            assert np.all(H[i] <= 1.0 + 1e-12)

    def test_quad_tilted_3d_partition_of_unity(self):
        """Verify partition of unity for arbitrary 3D rotated quad."""
        # Rotated quad corners
        c = np.cos(0.5)
        s = np.sin(0.5)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        base_quad = np.array([
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0]
        ])
        quad_3d = base_quad @ R.T + np.array([5.0, -2.0, 3.0])

        # Generate 20 test points
        np.random.seed(42)
        pts_local = np.random.uniform(-1.5, 1.5, (20, 3))
        pts_local[:, 2] = np.random.uniform(-0.5, 0.5, 20)
        sec_pts = pts_local @ R.T + np.array([5.0, -2.0, 3.0])

        x = np.vstack([quad_3d, sec_pts])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.arange(4, 24)
        main_nodes = np.tile(np.array([0, 1, 2, 3]), (len(sec_nodes), 1))
        cand_p = np.zeros(len(sec_nodes))

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=50.0, gap=2.0, stiff_dc=0.0, cand_p=cand_p, dt=1e-3
        )

        h_sums = np.sum(H, axis=1)
        np.testing.assert_allclose(h_sums, 1.0, rtol=1e-14, atol=1e-14)


class TestTriangleDirectProjection:
    """Test 3-node triangular segments direct projection (i18for3.F lines 228-238)."""

    def test_triangle_degenerate_fourth_node(self):
        """Triangles with n4 == n3 have H4 == 0.0 and sum(H1..H3) == 1.0."""
        x = np.array([
            [0.0, 0.0, 0.0],  # Node 0
            [2.0, 0.0, 0.0],  # Node 1
            [0.0, 2.0, 0.0],  # Node 2
            [0.5, 0.5, 0.1],  # Node 3 (secondary, inside triangle)
        ])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.array([3])
        main_nodes = np.array([[0, 1, 2, 2]])  # n4 == n3
        cand_p = np.zeros(1)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=100.0, gap=0.5, stiff_dc=0.0, cand_p=cand_p, dt=1e-3
        )

        assert active[0]
        assert H[0, 3] == 0.0
        np.testing.assert_allclose(np.sum(H[0, :3]), 1.0, rtol=1e-14)
        assert H[0, 0] > 0.0
        assert H[0, 1] > 0.0
        assert H[0, 2] > 0.0

    def test_triangle_negative_fourth_node(self):
        """Triangles with n4 == -1 have H4 == 0.0 and sum(H1..H3) == 1.0."""
        x = np.array([
            [0.0, 0.0, 0.0],  # Node 0
            [1.0, 0.0, 0.0],  # Node 1
            [0.0, 1.0, 0.0],  # Node 2
            [0.2, 0.2, 0.05], # Node 3 (secondary)
        ])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.array([3])
        main_nodes = np.array([[0, 1, 2, -1]])  # n4 == -1
        cand_p = np.zeros(1)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=100.0, gap=0.5, stiff_dc=0.0, cand_p=cand_p, dt=1e-3
        )

        assert active[0]
        assert H[0, 3] == 0.0
        np.testing.assert_allclose(np.sum(H[0, :3]), 1.0, rtol=1e-14)


class TestMomentumConservation:
    """Verify exact conservation of linear and angular momentum."""

    def test_linear_momentum_conservation(self):
        """sum(F_sec) + sum(F_main) == 0 to machine precision."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.1],  # Sec 4
            [0.7, 0.8, 0.2],  # Sec 5
        ])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, -10.0]
        v[5] = [0.0, 0.0, -20.0]
        mass = np.ones(len(x))
        sec_nodes = np.array([4, 5])
        main_nodes = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ])
        cand_p = np.zeros(2)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=200.0, gap=0.5, stiff_dc=10.0, cand_p=cand_p, dt=1e-3
        )

        assert np.all(active)
        for i in range(len(sec_nodes)):
            f_total = fsec[i] + np.sum(fmain[i], axis=0)
            np.testing.assert_allclose(f_total, 0.0, atol=1e-13)

    def test_angular_momentum_conservation(self):
        """sum(r x F) == 0 to machine precision about any reference origin."""
        x = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [1.0, 4.0, 0.0],
            [2.2, 3.1, 0.1],  # Sec 4
        ])
        v = np.zeros_like(x)
        v[4] = [1.0, 2.0, -5.0]
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=300.0, gap=0.5, stiff_dc=5.0, cand_p=cand_p, dt=1e-3
        )

        assert active[0]
        # Test torque about multiple reference origins
        for x_ref in [np.array([0.0, 0.0, 0.0]), np.array([10.0, -5.0, 3.0])]:
            r_sec = x[sec_nodes[0]] - x_ref
            torque_sec = np.cross(r_sec, fsec[0])

            torque_main = np.zeros(3)
            for k in range(4):
                r_main_k = x[main_nodes[0, k]] - x_ref
                torque_main += np.cross(r_main_k, fmain[0, k])

            total_torque = torque_sec + torque_main
            np.testing.assert_allclose(total_torque, 0.0, atol=1e-13)


class TestDynamicStiffnessAndDisplacement:
    """Test dynamic stiffness K = Stfac * (pene / gap) and cand_p accumulation."""

    def test_dynamic_penetration_stiffness(self):
        """Stiffness scales linearly with penetration depth."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.25],  # pene = 1.0 - 0.25 = 0.75
            [0.5, 0.5, 0.75],  # pene = 1.0 - 0.75 = 0.25
        ])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.array([4, 5])
        main_nodes = np.tile(np.array([0, 1, 2, 3]), (2, 1))
        cand_p = np.zeros(2)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=100.0, gap=1.0, stiff_dc=0.0, cand_p=cand_p, dt=1e-3
        )

        assert np.all(active)
        # pene1 = 0.75 -> stif = 100 * 0.75 / 1.0 = 75.0
        # pene2 = 0.25 -> stif = 100 * 0.25 / 1.0 = 25.0
        np.testing.assert_allclose(stif, [75.0, 25.0], rtol=1e-12)

    def test_cand_p_accumulation_and_reset(self):
        """cand_p accumulates relative normal displacement and resets on separation."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.4],  # Initial distance 0.4, gap 1.0
        ])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, -2.0]  # moving in -z direction (penetrating)
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)
        dt = 0.01

        # Cycle 1: vn = -2.0 -> cand_p += -2.0 * 0.01 = -0.02
        _t18_forces(x, v, mass, sec_nodes, main_nodes, 100.0, 1.0, 0.0, cand_p, dt)
        np.testing.assert_allclose(cand_p[0], -0.02, rtol=1e-12)

        # Cycle 2: another step
        _t18_forces(x, v, mass, sec_nodes, main_nodes, 100.0, 1.0, 0.0, cand_p, dt)
        np.testing.assert_allclose(cand_p[0], -0.04, rtol=1e-12)

        # Cycle 3: node separates beyond gap (dist = 1.5 > gap = 1.0)
        x[4, 2] = 1.5
        _t18_forces(x, v, mass, sec_nodes, main_nodes, 100.0, 1.0, 0.0, cand_p, dt)
        assert cand_p[0] == 0.0

    def test_non_penetrating_zero_force(self):
        """Nodes outside gap experience zero force."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 2.0],  # dist = 2.0 > gap = 1.0
        ])
        v = np.zeros_like(x)
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes, 100.0, 1.0, 0.0, cand_p, 1e-3
        )

        assert not active[0]
        np.testing.assert_allclose(fsec, 0.0)
        np.testing.assert_allclose(fmain, 0.0)
        assert econt == 0.0


class TestViscousDamping:
    """Test viscous damping force and dissipation (i18for3.F lines 327-336)."""

    def test_viscous_damping_active_for_negative_vn(self):
        """Viscous damping adds C * vn for approaching motion (vn < 0)."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],  # pene = 0.5, gap = 1.0
        ])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, -10.0]  # vn = -10.0 < 0 (approaching)
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)
        dt = 0.001

        fsec_damp, _, _, _, _, econt_damp = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=0.0, gap=1.0, stiff_dc=20.0, cand_p=cand_p, dt=dt
        )

        # Damp = stiff_dc * (pene/gap) * vn = 20 * (0.5/1.0) * (-10) = -100.0
        # Force on secondary is -fni * n = -(-100) * [0, 0, 1] = [0, 0, 100]
        np.testing.assert_allclose(fsec_damp[0], [0.0, 0.0, 100.0], rtol=1e-12)
        assert econt_damp > 0.0

    def test_viscous_damping_inactive_for_positive_vn(self):
        """Viscous damping is zero when separating (vn >= 0, i18for3.F line 329)."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, 10.0]  # vn = 10.0 > 0 (separating)
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)

        fsec, _, _, _, _, _ = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=0.0, gap=1.0, stiff_dc=20.0, cand_p=cand_p, dt=0.001
        )
        np.testing.assert_allclose(fsec, 0.0)


class TestContactType18Class:
    """Test ContactType18 class integration, force scattering, and time step bounds."""

    def test_init_and_forces_scatter(self):
        """Test forces scattering into global fcont and /DT/NODA stifn accumulation."""
        model = Model()
        log = MessageLog()

        # Nodes: 0,1,2,3 for quad, 4 for secondary fluid node
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.2],
        ])
        model.v = np.zeros_like(model.x)
        model.v[4] = [0.0, 0.0, -5.0]
        model.mass = np.full(5, 2.0)

        # Node group 100 for secondary node
        grp = NodeGroup(id=100)
        grp.node_idx = [4]
        model.node_groups[100] = grp

        # Surface 200 for main face
        surf = Surface(id=200)
        surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
        model.surfaces[200] = surf

        # Interface
        itf = Interface(id=1, type=18)
        itf.grnod_id = 100
        itf.surf_id = 200
        itf.stfac = 100.0
        itf.gap = 1.0
        itf.stiff_dc = 10.0

        ct = ContactType18(itf, model, log)
        assert len(ct.sec_nodes) == 1
        assert len(ct.main_faces) == 1

        fcont = np.zeros_like(model.x)
        stifn = np.zeros(len(model.x))

        econt, dt_int = ct.forces(model.x, model.v, model.mass, 0.001, fcont, cycle=1, stifn=stifn)

        # Force must be scattered: fcont[4] is non-zero, fcont[0..3] is non-zero
        assert np.linalg.norm(fcont[4]) > 0.0
        for i in range(4):
            assert np.linalg.norm(fcont[i]) > 0.0

        # Net global contact force must be zero
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)

        # Stiffness accumulation: stifn[4] must equal sum(stifn[0..3])
        assert stifn[4] > 0.0
        np.testing.assert_allclose(stifn[4], np.sum(stifn[:4]), rtol=1e-12)

        # dt_interface bound
        assert np.isfinite(dt_int)
        assert dt_int > 0.0

    def test_quad_with_node0_corner(self):
        """Ensure node index 0 in quad corner is correctly scattered without exclusion."""
        model = Model()
        log = MessageLog()
        model.x = np.array([
            [0.0, 0.0, 0.0],  # Node 0 is a quad corner
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.1],  # Node 4 secondary
        ])
        model.v = np.zeros_like(model.x)
        model.v[4] = [0.0, 0.0, -1.0]
        model.mass = np.ones(5)

        grp = NodeGroup(id=1)
        grp.node_idx = [4]
        model.node_groups[1] = grp

        surf = Surface(id=2)
        # Corner 4 is node 0!
        surf.segments = np.array([[1, 2, 3, 0]], dtype=np.int64)
        model.surfaces[2] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 1
        itf.surf_id = 2
        itf.stfac = 50.0
        itf.gap = 0.5

        ct = ContactType18(itf, model, log)
        fcont = np.zeros_like(model.x)
        stifn = np.zeros(len(model.x))

        ct.forces(model.x, model.v, model.mass, 0.001, fcont, 1, stifn=stifn)

        # Node 0 must have non-zero force and non-zero stiffness
        assert np.linalg.norm(fcont[0]) > 0.0
        assert stifn[0] > 0.0

    def test_triangular_face_negative_node4(self):
        """Segment with node 4 == -1 (triangle) executes without negative index error."""
        model = Model()
        log = MessageLog()
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.1],
        ])
        model.v = np.zeros_like(model.x)
        model.v[3] = [0.0, 0.0, 2.0]
        model.mass = np.ones(4)

        grp = NodeGroup(id=1)
        grp.node_idx = [3]
        model.node_groups[1] = grp

        surf = Surface(id=2)
        surf.segments = np.array([[0, 1, 2, -1]], dtype=np.int64)
        model.surfaces[2] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 1
        itf.surf_id = 2
        itf.stfac = 100.0
        itf.gap = 0.5

        ct = ContactType18(itf, model, log)
        fcont = np.zeros_like(model.x)
        stifn = np.zeros(len(model.x))

        ct.forces(model.x, model.v, model.mass, 0.001, fcont, 1, stifn=stifn)

        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)
        assert stifn[3] > 0.0

    def test_missing_node_group_graceful_handling(self):
        """Missing secondary node group logs error and deactivates interface."""
        model = Model()
        log = MessageLog()
        surf = Surface(id=200)
        surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
        model.surfaces[200] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 999  # Does not exist
        itf.surf_id = 200

        ct = ContactType18(itf, model, log)
        assert len(ct.sec_nodes) == 0
        assert len(ct.main_faces) == 0

        fcont = np.zeros((10, 3))
        econt, dt_bound = ct.forces(np.zeros((10, 3)), np.zeros((10, 3)), np.ones(10), 0.01, fcont, 1)
        assert econt == 0.0
        assert dt_bound == np.inf
        assert np.all(fcont == 0.0)

    def test_missing_surface_graceful_handling(self):
        """Missing main surface logs error and deactivates interface."""
        model = Model()
        log = MessageLog()
        grp = NodeGroup(id=100)
        grp.node_idx = [0, 1]
        model.node_groups[100] = grp

        itf = Interface(id=1, type=18)
        itf.grnod_id = 100
        itf.surf_id = 999  # Does not exist

        ct = ContactType18(itf, model, log)
        assert len(ct.sec_nodes) == 0
        assert len(ct.main_faces) == 0

        fcont = np.zeros((10, 3))
        econt, dt_bound = ct.forces(np.zeros((10, 3)), np.zeros((10, 3)), np.ones(10), 0.01, fcont, 1)
        assert econt == 0.0
        assert dt_bound == np.inf

    def test_zero_dt_no_op(self):
        """Zero dt returns without modifying fcont."""
        model = Model()
        log = MessageLog()
        grp = NodeGroup(id=1)
        grp.node_idx = [0]
        model.node_groups[1] = grp
        surf = Surface(id=2)
        surf.segments = np.array([[1, 2, 3, 4]])
        model.surfaces[2] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 1
        itf.surf_id = 2
        ct = ContactType18(itf, model, log)

        fcont = np.zeros((5, 3))
        econt, dt_bound = ct.forces(np.zeros((5, 3)), np.zeros((5, 3)), np.ones(5), 0.0, fcont, 1)
        assert econt == 0.0
        np.testing.assert_allclose(fcont, 0.0)

    def test_degenerate_zero_distance_normal_fallback(self):
        """Secondary node at zero distance from surface uses face normal."""
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],  # Exactly on the face: dist == 0.0
        ])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, 1.0]
        mass = np.ones(len(x))
        sec_nodes = np.array([4])
        main_nodes = np.array([[0, 1, 2, 3]])
        cand_p = np.zeros(1)

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, sec_nodes, main_nodes,
            stfval=100.0, gap=1.0, stiff_dc=0.0, cand_p=cand_p, dt=0.001
        )

        assert active[0]
        # Normal must not be NaN or inf
        assert np.all(np.isfinite(fsec))
        assert np.all(np.isfinite(fmain))
        np.testing.assert_allclose(np.sum(fsec) + np.sum(fmain), 0.0, atol=1e-13)

    def test_chunked_search_large_cloud(self):
        """Verify chunked search operates properly with 500 secondary nodes."""
        model = Model()
        log = MessageLog()

        # 10 main quad faces
        n_faces = 10
        face_nodes = np.arange(n_faces * 4).reshape((n_faces, 4))
        x_main = np.zeros((n_faces * 4, 3))
        for i in range(n_faces):
            x_main[i*4 + 0] = [i, 0, 0]
            x_main[i*4 + 1] = [i+1, 0, 0]
            x_main[i*4 + 2] = [i+1, 1, 0]
            x_main[i*4 + 3] = [i, 1, 0]

        # 500 secondary nodes
        n_sec = 500
        x_sec = np.zeros((n_sec, 3))
        x_sec[:, 0] = np.linspace(0.1, 9.9, n_sec)
        x_sec[:, 1] = 0.5
        x_sec[:, 2] = 0.1  # slightly above z=0

        x_all = np.vstack([x_main, x_sec])
        v_all = np.zeros_like(x_all)
        v_all[n_faces*4:] = [0.0, 0.0, -1.0]
        mass_all = np.ones(len(x_all))

        grp = NodeGroup(id=1)
        grp.node_idx = list(range(n_faces * 4, len(x_all)))
        model.node_groups[1] = grp

        surf = Surface(id=2)
        surf.segments = face_nodes
        model.surfaces[2] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 1
        itf.surf_id = 2
        itf.stfac = 50.0
        itf.gap = 0.5

        ct = ContactType18(itf, model, log)
        fcont = np.zeros_like(x_all)
        econt, dt_bound = ct.forces(x_all, v_all, mass_all, 0.001, fcont, 1)

        assert econt > 0.0
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


class TestStarterAndBuildContactsIntegration:
    """Test Starter keyword reading and factory build_contacts integration."""

    def test_build_contacts_factory(self):
        """build_contacts creates ContactType18 in penalty contact list."""
        model = Model()
        log = MessageLog()

        grp = NodeGroup(id=10)
        grp.node_idx = [0]
        model.node_groups[10] = grp

        surf = Surface(id=20)
        surf.segments = np.array([[1, 2, 3, 4]])
        model.surfaces[20] = surf

        itf = Interface(id=5, type=18)
        itf.grnod_id = 10
        itf.surf_id = 20
        model.interfaces.append(itf)

        penalty, tied = build_contacts(model, log)
        assert len(penalty) == 1
        assert len(tied) == 0
        assert isinstance(penalty[0], ContactType18)
        assert penalty[0].id == 5

    def test_starter_deck_fixed_format(self, tmp_path):
        """Test parsing /INTER/TYPE18 from fixed format deck."""
        d = StarterDeck("test_t18")
        d.raw_block("INTER/TYPE18/42", [
            f"{101:10d}{202:10d}{0:10d}{0:20d}{1:20d}{0:10d}",
            f"{75.0:20.6f}{0.0:20.6f}{2.5:20.6f}",
            f"{'':40s}{12.5:20.6f}{0.3:20.6f}",
        ])
        deck_str = d.render()
        p = tmp_path / "t18_fixed.rad"
        p.write_text(deck_str)

        blocks = list(read_deck(str(p)))
        model = Model()
        log = MessageLog()
        read_inter(blocks[1], model, log)

        assert len(model.interfaces) == 1
        itf = model.interfaces[0]
        assert itf.id == 42
        assert itf.type == 18
        assert itf.grnod_id == 101
        assert itf.surf_id == 202
        assert itf.stfac == 75.0
        assert itf.gap == 2.5
        assert itf.stiff_dc == 12.5
        assert itf.sort_fact == 0.3

    def test_multi_cycle_dynamic_engine_contact(self):
        """Verify dynamic multi-cycle contact force evolution and energy monotonicity."""
        model = Model()
        log = MessageLog()

        # Quad main surface in z=0
        x_main = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Secondary node starting above surface moving downward
        x_sec = np.array([[0.5, 0.5, 0.8]])
        x = np.vstack([x_main, x_sec])
        v = np.zeros_like(x)
        v[4] = [0.0, 0.0, -100.0]
        mass = np.full(5, 0.5)

        grp = NodeGroup(id=1)
        grp.node_idx = [4]
        model.node_groups[1] = grp

        surf = Surface(id=2)
        surf.segments = np.array([[0, 1, 2, 3]])
        model.surfaces[2] = surf

        itf = Interface(id=1, type=18)
        itf.grnod_id = 1
        itf.surf_id = 2
        itf.stfac = 5000.0
        itf.gap = 1.0
        itf.stiff_dc = 100.0
        model.interfaces.append(itf)

        penalty, _ = build_contacts(model, log)
        ct = penalty[0]

        dt = 0.0005
        cum_econt = 0.0
        for cycle in range(1, 6):
            fcont = np.zeros_like(x)
            stifn = np.zeros(len(x))
            econt_step, dt_bound = ct.forces(x, v, mass, dt, fcont, cycle, stifn=stifn)

            # Upward reaction force on secondary node resisting penetration
            assert fcont[4, 2] > 0.0

            # Momentum conservation
            np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)

            # Energy booking
            assert econt_step > 0.0
            cum_econt += econt_step

            # Simple kinematic step forward
            x += v * dt

