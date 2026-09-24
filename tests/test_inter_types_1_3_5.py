"""
Tests for Contact Interfaces:
  - /INTER/TYPE1 (ContactType1) ALE-Lagrangian fluid-structure coupling
  - /INTER/TYPE3 (ContactType3) Surface-to-surface contact with Cartesian DOF deactivation
  - /INTER/TYPE5 (ContactType5) Node-to-surface contact with nonlinear Coulomb & velocity-dependent friction
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import build_contacts
from pyradioss.contact.inter_type1 import ContactType1, _t1_forces, _t1_project
from pyradioss.contact.inter_type3 import ContactType3, _t3_forces
from pyradioss.contact.inter_type5 import ContactType5, _t5_forces
from pyradioss.model.entities import Interface, NodeGroup, Surface
from pyradioss.model.model import Model


def _create_mock_quad_model(itf_type: int, **itf_kwargs):
    """Create a minimal model with a flat 4-node quad in z=0 and secondary nodes above it."""
    model = Model()
    log = MessageLog()

    # Master quad corners in z=0 plane: [0,0,0], [2,0,0], [2,2,0], [0,2,0]
    # Secondary nodes at z = 0.05 (inside gap 0.1)
    coords = np.array([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [2.0, 2.0, 0.0],  # 2
        [0.0, 2.0, 0.0],  # 3
        [1.0, 1.0, 0.05], # 4 (center)
        [0.5, 0.5, 0.05], # 5 (sub-tri 0)
        [1.5, 0.5, 0.05], # 6 (sub-tri 1)
        [1.5, 1.5, 0.05], # 7 (sub-tri 2)
        [0.5, 1.5, 0.05], # 8 (sub-tri 3)
        [5.0, 5.0, 0.05], # 9 (outside quad)
    ], dtype=np.float64)

    model.x = coords.copy()
    model.x0 = coords.copy()
    model.v = np.zeros_like(coords)
    model.mass = np.ones(len(coords), dtype=np.float64)

    # Master surface ID 1
    surf = Surface(id=1, title="MASTER_SURF")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    model.surfaces[1] = surf

    # Secondary node group ID 10
    grp = NodeGroup(id=10, title="SEC_NODES")
    grp.node_idx = np.array([4, 5, 6, 7, 8, 9], dtype=np.int64)
    model.node_groups[10] = grp

    # Secondary surface ID 2 (for surface-to-surface tests)
    surf2 = Surface(id=2, title="SEC_SURF")
    surf2.segments = np.array([[4, 5, 6, 7]], dtype=np.int64)
    model.surfaces[2] = surf2

    itf = Interface(
        id=100,
        type=itf_type,
        surf_id=1,
        grnod_id=10,
        gap=0.1,
        stfac=500.0,
        **itf_kwargs,
    )
    model.interfaces.append(itf)
    return model, itf, log


class TestPartitionOfUnityAndProjection:
    """Verify projection and shape functions sum to 1.0."""

    def test_quad_partition_of_unity(self):
        xs = np.array([
            [1.0, 1.0, 0.2],
            [0.5, 0.2, 0.1],
            [1.8, 0.5, 0.1],
            [1.2, 1.7, 0.1],
            [0.3, 1.5, 0.1],
            [3.0, 3.0, 0.1],
        ])
        main_nodes = np.tile(np.array([0, 1, 2, 3]), (len(xs), 1))
        x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ])

        dist, pt, H, n_face = _t1_project(xs, main_nodes, x)
        h_sums = np.sum(H, axis=1)
        np.testing.assert_allclose(h_sums, 1.0, atol=1e-14)
        for i in range(5):
            assert np.all(H[i] >= -1e-12)
            assert np.all(H[i] <= 1.0 + 1e-12)

    def test_triangle_partition_of_unity(self):
        xs = np.array([[0.2, 0.2, 0.05]])
        main_nodes = np.array([[0, 1, 2, -1]])
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        dist, pt, H, n_face = _t1_project(xs, main_nodes, x)
        assert H[0, 3] == 0.0
        np.testing.assert_allclose(np.sum(H[0, :3]), 1.0, atol=1e-14)


class TestContactType1:
    """Test /INTER/TYPE1 ALE-Lagrangian fluid-structure coupling."""

    def test_init_and_empty_fallback(self):
        model = Model()
        log = MessageLog()
        itf_empty = Interface(id=1, type=1, surf_id=999, grnod_id=888)
        ct1 = ContactType1(itf_empty, model, log)
        assert len(ct1.nodes) == 0
        assert len(ct1.segs) == 0
        fcont = np.zeros((10, 3))
        f_ret, dt_bound = ct1.forces(np.zeros((10, 3)), np.zeros((10, 3)), np.ones(10), 1e-4, fcont)
        assert f_ret is fcont
        np.testing.assert_array_equal(fcont, 0.0)

    def test_linear_and_angular_momentum(self):
        model, itf, log = _create_mock_quad_model(itf_type=1, fric=0.0, stiff_dc=1.0)
        ct1 = ContactType1(itf, model, log)

        # Give secondary nodes downward velocity
        model.v[4:9, 2] = -10.0
        fcont = np.zeros_like(model.x)
        f_ret, dt_bound = ct1.forces(model.x, model.v, model.mass, 1e-3, fcont, t=0.0)

        # Contact occurred
        assert np.any(np.abs(fcont) > 0.0)

        # 1. Exact linear momentum conservation
        total_f = np.sum(fcont, axis=0)
        np.testing.assert_allclose(total_f, 0.0, atol=1e-13)

        # 2. Exact angular momentum consistency about multiple reference points
        for x_ref in [np.array([0.0, 0.0, 0.0]), np.array([5.0, -3.0, 2.5])]:
            r = model.x - x_ref
            torque = np.sum(np.cross(r, fcont), axis=0)
            np.testing.assert_allclose(torque, 0.0, atol=1e-12)

    def test_build_contacts_dispatch(self):
        model, itf, log = _create_mock_quad_model(itf_type=1)
        pen, tied = build_contacts(model, log)
        assert len(pen) == 1
        assert isinstance(pen[0], ContactType1)


class TestContactType3:
    """Test /INTER/TYPE3 with Cartesian DOF deactivation."""

    def test_init_and_empty_fallback(self):
        model = Model()
        log = MessageLog()
        itf_empty = Interface(id=3, type=3, surf_id=999)
        ct3 = ContactType3(itf_empty, model, log)
        assert len(ct3.nodes) == 0
        fcont = np.zeros((5, 3))
        f_ret, dt = ct3.forces(np.zeros((5, 3)), np.zeros((5, 3)), np.ones(5), 1e-4, fcont)
        assert f_ret is fcont

    def test_dof_deactivation_x(self):
        """ibc1 = True deactivates X-direction forces."""
        model, itf, log = _create_mock_quad_model(itf_type=3, params={"ibc1": True})
        # Rotate quad so contact normal has non-zero X, Y, and Z components
        # 45 deg tilt about Y axis
        theta = np.pi / 4.0
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        model.x = model.x @ R.T

        ct3 = ContactType3(itf, model, log)
        assert ct3.ibc1 is True
        assert ct3.ibc2 is False

        fcont = np.zeros_like(model.x)
        f_ret, dt = ct3.forces(model.x, model.v, model.mass, 1e-3, fcont)

        # X forces must be identically zero on all nodes
        np.testing.assert_allclose(fcont[:, 0], 0.0, atol=1e-15)
        # Y or Z forces should be non-zero
        assert np.any(np.abs(fcont[:, 2]) > 0.0)

        # Linear momentum in all 3 directions must still be exactly conserved
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)

    def test_dof_deactivation_z(self):
        """ibc3 = True deactivates Z-direction forces."""
        model, itf, log = _create_mock_quad_model(itf_type=3, params={"ibc3": True})
        ct3 = ContactType3(itf, model, log)
        assert ct3.ibc3 is True

        fcont = np.zeros_like(model.x)
        f_ret, dt = ct3.forces(model.x, model.v, model.mass, 1e-3, fcont)

        # In flat quad, normal is in Z direction; deactivating Z should zero all forces
        np.testing.assert_allclose(fcont, 0.0, atol=1e-15)

    def test_combined_ibc_flags(self):
        """ibc = 6 means ibc1=1 (4) + ibc2=1 (2) -> X and Y deactivated."""
        model, itf, log = _create_mock_quad_model(itf_type=3, params={"ibc": 6})
        ct3 = ContactType3(itf, model, log)
        assert ct3.ibc1 is True
        assert ct3.ibc2 is True
        assert ct3.ibc3 is False

    def test_linear_and_angular_momentum_undeactivated(self):
        """Without deactivation, linear and angular momentum are machine-precision conserved."""
        model, itf, log = _create_mock_quad_model(itf_type=3, fric=0.0)
        ct3 = ContactType3(itf, model, log)
        fcont = np.zeros_like(model.x)
        ct3.forces(model.x, model.v, model.mass, 1e-3, fcont)

        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)
        for x_ref in [np.zeros(3), np.array([1.2, 3.4, -2.1])]:
            r = model.x - x_ref
            torque = np.sum(np.cross(r, fcont), axis=0)
            np.testing.assert_allclose(torque, 0.0, atol=1e-13)

    def test_build_contacts_dispatch(self):
        model, itf, log = _create_mock_quad_model(itf_type=3)
        pen, tied = build_contacts(model, log)
        assert len(pen) == 1
        assert isinstance(pen[0], ContactType3)


class TestContactType5:
    """Test /INTER/TYPE5 with nonlinear Coulomb and velocity-dependent friction."""

    def test_init_and_empty_fallback(self):
        model = Model()
        log = MessageLog()
        itf_empty = Interface(id=5, type=5, surf_id=999, grnod_id=888)
        ct5 = ContactType5(itf_empty, model, log)
        assert len(ct5.nodes) == 0
        fcont = np.zeros((4, 3))
        f_ret, dt = ct5.forces(np.zeros((4, 3)), np.zeros((4, 3)), np.ones(4), 1e-4, fcont)
        assert f_ret is fcont

    def test_coulomb_friction(self):
        """Standard Coulomb friction (mfrot=0)."""
        model, itf, log = _create_mock_quad_model(itf_type=5, fric=0.2, mfrot=0)
        # Give secondary nodes tangential sliding velocity in X
        model.v[4:9, 0] = 5.0

        ct5 = ContactType5(itf, model, log)
        fcont = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont)

        # Secondary nodes must experience negative X force (opposing sliding)
        assert np.all(fcont[4:9, 0] < 0.0)
        # Normal force in positive Z
        assert np.all(fcont[4:9, 2] > 0.0)

        # Exact linear momentum conservation
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)

    def test_nonlinear_viscous_friction_mfrot1(self):
        """MFROT=1: generalized viscous polynomial mu = mu0 + C1*p + C2*v + ..."""
        model, itf, log = _create_mock_quad_model(
            itf_type=5,
            fric=0.1,
            mfrot=1,
            c1=0.01,
            c2=0.05,
            c3=0.0,
            c4=0.0,
            c5=0.0,
        )
        ct5 = ContactType5(itf, model, log)
        assert ct5.mfrot == 1
        assert ct5.fric_c[0] == 0.01
        assert ct5.fric_c[1] == 0.05

        # Compare low vs high slip speed
        model.v[4:9, 0] = 1.0
        fcont_slow = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont_slow)

        model.v[4:9, 0] = 10.0
        fcont_fast = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont_fast)

        # Higher slip velocity produces higher friction force under MFROT=1
        assert np.abs(fcont_fast[4, 0]) > np.abs(fcont_slow[4, 0])

        # Exact linear momentum conservation for both
        np.testing.assert_allclose(np.sum(fcont_slow, axis=0), 0.0, atol=1e-13)
        np.testing.assert_allclose(np.sum(fcont_fast, axis=0), 0.0, atol=1e-13)

    def test_exponential_decay_friction_mfrot4(self):
        """MFROT=4: exponential decay mu = C1 + (mu0 - C1)*exp(-C2*v)."""
        model, itf, log = _create_mock_quad_model(
            itf_type=5,
            fric=0.5, # static
            mfrot=4,
            c1=0.2,   # dynamic
            c2=1.0,   # decay rate
        )
        ct5 = ContactType5(itf, model, log)

        model.v[4:9, 0] = 0.5  # moderate slip: mu ~ 0.38
        fcont_slow = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont_slow)

        model.v[4:9, 0] = 50.0 # high dynamic: mu ~ 0.20
        fcont_fast = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont_fast)

        # Friction coefficient decays at high speed
        # Fx_slow / Fz_slow should be higher than Fx_fast / Fz_fast
        ratio_slow = np.abs(fcont_slow[4, 0]) / fcont_slow[4, 2]
        ratio_fast = np.abs(fcont_fast[4, 0]) / fcont_fast[4, 2]
        assert ratio_slow > ratio_fast
        np.testing.assert_allclose(ratio_fast, 0.2, rtol=0.05)

        # Linear momentum conservation
        np.testing.assert_allclose(np.sum(fcont_fast, axis=0), 0.0, atol=1e-13)

    def test_dof_deactivation_in_type5(self):
        """TYPE5 also respects ibc1, ibc2, ibc3."""
        model, itf, log = _create_mock_quad_model(
            itf_type=5, fric=0.2, params={"ibc1": True}
        )
        model.v[4:9, 0] = 10.0
        ct5 = ContactType5(itf, model, log)
        fcont = np.zeros_like(model.x)
        ct5.forces(model.x, model.v, model.mass, 1e-3, fcont)

        # X forces deactivated
        np.testing.assert_allclose(fcont[:, 0], 0.0, atol=1e-15)
        # Z forces active
        assert np.any(fcont[:, 2] > 0.0)
        # Linear momentum conserved
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-13)

    def test_build_contacts_dispatch(self):
        model, itf, log = _create_mock_quad_model(itf_type=5)
        pen, tied = build_contacts(model, log)
        assert len(pen) == 1
        assert isinstance(pen[0], ContactType5)


class TestRobustnessAndEdgeCases:
    """Edge cases: 3D rotated quads, triangles, time gating, and stifn accumulation."""

    @pytest.mark.parametrize("itf_cls,itf_type", [
        (ContactType1, 1),
        (ContactType3, 3),
        (ContactType5, 5),
    ])
    def test_3d_arbitrary_rotation_momentum(self, itf_cls, itf_type):
        """Under arbitrary 3D rotation, normal penalty forces conserve linear and angular momentum."""
        model, itf, log = _create_mock_quad_model(itf_type=itf_type, fric=0.0, stiff_dc=0.0)
        # Random 3D rotation matrix
        alpha, beta, gamma = 0.3, -0.7, 1.1
        Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        model.x = model.x @ R.T + np.array([12.5, -7.3, 4.2])

        ct = itf_cls(itf, model, log)
        fcont = np.zeros_like(model.x)
        stifn = np.zeros(len(model.x))
        f_ret, dt_bound = ct.forces(model.x, model.v, model.mass, 1e-3, fcont, stifn=stifn)

        # Contact forces exist
        assert np.any(np.abs(fcont) > 0.0)
        assert np.any(stifn > 0.0)

        # Linear momentum: sum(F) == 0 within 1e-12
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)

        # Angular momentum: sum(r x F) == 0
        for x_ref in [np.zeros(3), np.array([12.5, -7.3, 4.2]), model.x.mean(axis=0)]:
            r = model.x - x_ref
            torque = np.sum(np.cross(r, fcont), axis=0)
            np.testing.assert_allclose(torque, 0.0, atol=1e-12)

    @pytest.mark.parametrize("itf_cls,itf_type", [
        (ContactType1, 1),
        (ContactType3, 3),
        (ContactType5, 5),
    ])
    def test_triangular_segment_momentum(self, itf_cls, itf_type):
        """Triangular segment (n4 = -1) maintains exact momentum conservation."""
        model = Model()
        log = MessageLog()
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.5, 0.02], # inside triangle
        ])
        model.x = coords.copy()
        model.x0 = coords.copy()
        model.v = np.zeros_like(coords)
        model.mass = np.ones(len(coords))

        surf = Surface(id=1, title="TRI_SURF")
        surf.segments = np.array([[0, 1, 2, -1]], dtype=np.int64)
        model.surfaces[1] = surf

        grp = NodeGroup(id=10, title="SEC")
        grp.node_idx = np.array([3], dtype=np.int64)
        model.node_groups[10] = grp

        itf = Interface(id=1, type=itf_type, surf_id=1, grnod_id=10, gap=0.05, stfac=100.0, fric=0.0)
        ct = itf_cls(itf, model, log)

        fcont = np.zeros_like(coords)
        ct.forces(model.x, model.v, model.mass, 1e-3, fcont)

        assert np.any(np.abs(fcont) > 0.0)
        np.testing.assert_allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)
        r = model.x
        torque = np.sum(np.cross(r, fcont), axis=0)
        np.testing.assert_allclose(torque, 0.0, atol=1e-12)

    @pytest.mark.parametrize("itf_cls,itf_type", [
        (ContactType1, 1),
        (ContactType3, 3),
        (ContactType5, 5),
    ])
    def test_time_gating(self, itf_cls, itf_type):
        """Contact is inactive when t < tstart or t > tstop."""
        model, itf, log = _create_mock_quad_model(
            itf_type=itf_type, tstart=0.1, tstop=0.5
        )
        ct = itf_cls(itf, model, log)

        fcont = np.zeros_like(model.x)
        # t = 0.0 < tstart: inactive
        ct.forces(model.x, model.v, model.mass, 1e-3, fcont, t=0.0)
        np.testing.assert_array_equal(fcont, 0.0)

        # t = 0.6 > tstop: inactive
        ct.forces(model.x, model.v, model.mass, 1e-3, fcont, t=0.6)
        np.testing.assert_array_equal(fcont, 0.0)

        # t = 0.2: active
        ct.forces(model.x, model.v, model.mass, 1e-3, fcont, t=0.2)
        assert np.any(np.abs(fcont) > 0.0)

