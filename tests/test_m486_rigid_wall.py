"""
Unit tests for /RWALL rigid wall kinematic treatment & energy accounting (M486).

Tests cover:
- RigidWalls._geometry:
  - PLANE: positive, negative, coplanar distances and arbitrary plane normals
  - SPHER: positive radius (external obstacle) and negative radius (containment)
  - CYL: positive radius (external cylinder) and negative radius (internal pipe)
  - PARAL: interior points (0 <= a, b <= 1), exterior points (s = inf),
           unnormalized edge vectors, skewed (non-orthogonal) parallelogram
- RigidWalls.apply:
  - Zero / negative dt no-op
  - Empty walls list no-op
  - Non-penetrating / moving-away nodes no-op
  - slide=0: frictionless sliding (normal velocity landing, tangential preserved)
  - Resting node on wall: zero work / no energy leakage (kinematics.py identity)
  - slide=1: tied wall condition (node rides wall, tangential arrested)
  - slide=2: Coulomb friction stick regime (|vt| <= mu*|dvn|)
  - slide=2: Coulomb friction slip regime (|vt| > mu*|dvn|)
  - Search band gating (dist > 0)
  - Driven wall (/IMPVEL): external work booking into wext
  - Free wall (/ADMAS):
    - Single-node impact matches exact analytical inelastic velocity in 1 cycle
    - Multi-node asymmetric impact solves 3x3 system A v_w' = b
    - Linear momentum conservation across wall and candidate nodes
    - Contact energy / KE dissipation balance closure
    - Tangential friction impulse reaction on free wall carrier
  - Internal sphere containment impact
  - Multiple sequential walls in a single model
- RigidWalls.__init__ defensive handling:
  - Node group filtering and exclusion (grnod_id, grnod_id2)
  - Frozen (mass >= 1e29) and carrier node exclusion
  - Missing carrier node or node group handled gracefully
  - Moving PARAL wall unnormalized edge vectors and normal computation
- Starter parser verification:
  - /RWALL/PARAL parser preserves unnormalized edge lengths
  - /RWALL/SPHER and /RWALL/CYL negative radius parsing

Fortran references:
- engine/source/constraints/general/rwall/rgwal0.F
- engine/source/constraints/general/rwall/rgwall.F
- engine/source/constraints/general/rwall/rgwalc.F
- engine/source/constraints/general/rwall/rgwals.F
- engine/source/constraints/general/rwall/rgwalp.F
- engine/source/constraints/general/rwall/rgwalt.F
- starter/source/constraints/general/rwall/hm_read_rwall_plane.F
- starter/source/constraints/general/rwall/hm_read_rwall_cyl.F
- starter/source/constraints/general/rwall/hm_read_rwall_spher.F
- starter/source/constraints/general/rwall/hm_read_rwall_paral.F
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.rigid_wall import RigidWalls
from pyradioss.model.entities import ImposedVelocity, NodeGroup, RigidWall
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


# ==============================================================================
# Helper to construct lightweight synthetic model for testing RigidWalls
# ==============================================================================

def make_test_model(numnod=10, mass_val=1.0):
    model = Model()
    model.node_ids = np.arange(1, numnod + 1, dtype=np.int64)
    model.x0 = np.zeros((numnod, 3), dtype=np.float64)
    model.x = np.zeros((numnod, 3), dtype=np.float64)
    model.v = np.zeros((numnod, 3), dtype=np.float64)
    model.vr = np.zeros((numnod, 3), dtype=np.float64)
    model.mass = np.full(numnod, mass_val, dtype=np.float64)
    model.inertia = np.zeros((numnod, 3), dtype=np.float64)
    model._id2idx = {int(nid): i for i, nid in enumerate(model.node_ids)}
    model.node_groups = {}
    model.rwalls = []
    model.impvel = []
    return model


# ==============================================================================
# 1. Geometry Projections: RigidWalls._geometry
# ==============================================================================

class TestRigidWallGeometry:
    def test_plane_distances_and_normals(self):
        """Fixed plane z = 0 with normal [0, 0, 1]."""
        rw = RigidWall(
            id=1, point=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE"
        )
        xc = np.array([
            [1.0, 2.0, 5.0],    # in front (s = 5)
            [0.0, 0.0, 0.0],    # on plane (s = 0)
            [-3.0, 4.0, -2.5],  # behind plane (s = -2.5)
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [5.0, 0.0, -2.5], rtol=1e-12)
        expected_n = np.tile([0.0, 0.0, 1.0], (3, 1))
        np.testing.assert_allclose(n, expected_n, rtol=1e-12)

    def test_plane_arbitrary_orientation(self):
        """Tilted plane passing through (1, 1, 1) with normal along (1, 1, 1)."""
        normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        point = np.array([1.0, 1.0, 1.0])
        rw = RigidWall(id=2, point=point, normal=normal, geom="PLANE")

        xc = np.array([
            [1.0, 1.0, 1.0],                   # on plane -> s = 0
            [1.0 + 2.0 / np.sqrt(3.0), 1.0 + 2.0 / np.sqrt(3.0), 1.0 + 2.0 / np.sqrt(3.0)],  # offset by 2*n -> s = 2.0
            [0.0, 0.0, 0.0],                   # origin: (0 - 1, 0 - 1, 0 - 1) . n = -3/sqrt(3) = -sqrt(3)
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [0.0, 2.0, -np.sqrt(3.0)], rtol=1e-12)
        assert n.shape == (3, 3)
        np.testing.assert_allclose(n[0], normal, rtol=1e-12)

    def test_sphere_external(self):
        """Sphere with center (0, 0, 0) and radius R = 5. Nodes live outside."""
        rw = RigidWall(
            id=3, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            radius=5.0, geom="SPHER"
        )
        xc = np.array([
            [0.0, 0.0, 10.0],   # d = 10, s = 10 - 5 = 5
            [3.0, 4.0, 0.0],    # d = 5, s = 0
            [1.0, 0.0, 0.0],    # d = 1, s = -4 (penetrated)
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [5.0, 0.0, -4.0], rtol=1e-12)
        np.testing.assert_allclose(n[0], [0.0, 0.0, 1.0], rtol=1e-12)
        np.testing.assert_allclose(n[1], [0.6, 0.8, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[2], [1.0, 0.0, 0.0], rtol=1e-12)

    def test_sphere_internal_containment(self):
        """Sphere with negative radius R = -5.0 (containment inside sphere)."""
        rw = RigidWall(
            id=4, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            radius=-5.0, geom="SPHER"
        )
        xc = np.array([
            [1.0, 0.0, 0.0],    # inside: d = 1, s = 5 - 1 = 4 (allowed)
            [0.0, 5.0, 0.0],    # on boundary: d = 5, s = 0
            [0.0, 0.0, 8.0],    # outside: d = 8, s = 5 - 8 = -3 (penetrated)
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [4.0, 0.0, -3.0], rtol=1e-12)
        # Inward-pointing unit normal (toward origin)
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[1], [0.0, -1.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[2], [0.0, 0.0, -1.0], rtol=1e-12)

    def test_cylinder_external(self):
        """Cylinder with axis along Z, point at origin, radius R = 2.0."""
        rw = RigidWall(
            id=5, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            radius=2.0, geom="CYL"
        )
        xc = np.array([
            [4.0, 0.0, 100.0],  # radial d = 4, s = 2, z has no effect
            [0.0, 2.0, -50.0],  # radial d = 2, s = 0
            [0.5, 0.0, 0.0],    # radial d = 0.5, s = -1.5
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [2.0, 0.0, -1.5], rtol=1e-12)
        np.testing.assert_allclose(n[0], [1.0, 0.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[1], [0.0, 1.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[2], [1.0, 0.0, 0.0], rtol=1e-12)

    def test_cylinder_internal_containment(self):
        """Cylinder with negative radius R = -3.0 (pipe containment)."""
        rw = RigidWall(
            id=6, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            radius=-3.0, geom="CYL"
        )
        xc = np.array([
            [1.0, 0.0, 20.0],   # inside: d = 1, s = 3 - 1 = 2
            [0.0, 3.0, -10.0],  # boundary: d = 3, s = 0
            [4.0, 0.0, 0.0],    # outside pipe: d = 4, s = 3 - 4 = -1
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s, [2.0, 0.0, -1.0], rtol=1e-12)
        # Normal points inward toward cylinder axis
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[1], [0.0, -1.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(n[2], [-1.0, 0.0, 0.0], rtol=1e-12)

    def test_parallelogram_interior_and_exterior(self):
        """Parallelogram with origin M=(0,0,0), M1=(10,0,0), M2=(0,5,0)."""
        axis1 = np.array([10.0, 0.0, 0.0])  # unnormalized span vector
        axis2 = np.array([0.0, 5.0, 0.0])   # unnormalized span vector
        normal = np.array([0.0, 0.0, 1.0])
        rw = RigidWall(
            id=7, point=np.zeros(3), normal=normal,
            axis1=axis1, axis2=axis2, geom="PARAL"
        )
        xc = np.array([
            [2.0, 2.0, 3.0],    # inside: a = 0.2, b = 0.4 -> s = 3.0
            [10.0, 5.0, -1.0],  # corner: a = 1.0, b = 1.0 -> s = -1.0
            [0.0, 0.0, 0.0],    # origin: a = 0.0, b = 0.0 -> s = 0.0
            [12.0, 2.0, 2.0],   # outside (a = 1.2 > 1.0) -> s = inf
            [-1.0, 2.0, 2.0],   # outside (a = -0.1 < 0.0) -> s = inf
            [5.0, 6.0, 1.0],    # outside (b = 1.2 > 1.0) -> s = inf
            [5.0, -1.0, 1.0],   # outside (b = -0.2 < 0.0) -> s = inf
        ])
        s, n = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s[:3], [3.0, -1.0, 0.0], rtol=1e-12)
        assert np.isinf(s[3])
        assert np.isinf(s[4])
        assert np.isinf(s[5])
        assert np.isinf(s[6])
        np.testing.assert_allclose(n, np.tile([0.0, 0.0, 1.0], (7, 1)), rtol=1e-12)

    def test_parallelogram_skewed_non_orthogonal(self):
        """Parallelogram with non-orthogonal axes: axis1=(2, 0, 0), axis2=(1, 2, 0)."""
        axis1 = np.array([2.0, 0.0, 0.0])
        axis2 = np.array([1.0, 2.0, 0.0])
        n_unnorm = np.cross(axis1, axis2)
        normal = n_unnorm / np.linalg.norm(n_unnorm)
        rw = RigidWall(
            id=8, point=np.zeros(3), normal=normal,
            axis1=axis1, axis2=axis2, geom="PARAL"
        )
        # Point at a=0.5, b=0.5 -> x = 0.5*2 + 0.5*1 = 1.5, y = 0.5*2 = 1.0
        xc = np.array([
            [1.5, 1.0, 4.0],   # inside: a=0.5, b=0.5 -> s = 4.0
            [0.5, 1.0, -2.0],  # a=0.0, b=0.5 -> on boundary -> s = -2.0
            [2.5, 1.0, 1.0],   # a=1.0, b=0.5 -> on boundary -> s = 1.0
            [2.6, 1.0, 1.0],   # a=1.05 > 1 -> s = inf
        ])
        s, _ = RigidWalls._geometry(rw, rw.point, xc)
        np.testing.assert_allclose(s[:3], [4.0, -2.0, 1.0], rtol=1e-12)
        assert np.isinf(s[3])


# ==============================================================================
# 2. Kinematic Velocity Update & Friction: RigidWalls.apply
# ==============================================================================

class TestRigidWallApplyKinematics:
    def test_apply_zero_dt_or_empty_noop(self):
        model = make_test_model(numnod=2)
        log = MessageLog()
        walls = RigidWalls(model, log)
        de, dw = walls.apply(model.x, model.v, model.v, model.mass, dt=0.0)
        assert de == 0.0 and dw == 0.0
        de, dw = walls.apply(model.x, model.v, model.v, model.mass, dt=-1e-3)
        assert de == 0.0 and dw == 0.0

    def test_apply_fixed_plane_slide0_frictionless(self):
        """Node flying into fixed z=0 plane with slide=0 (frictionless)."""
        model = make_test_model(numnod=1, mass_val=2.5)
        model.x[0] = [0.0, 0.0, 0.02]      # starts at z = 0.02
        model.v[0] = [3.0, 4.0, -1.0]      # vx=3, vy=4, vz=-1
        v_old = model.v.copy()
        dt = 0.05                           # end of step z would be 0.02 - 0.05 = -0.03

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Normal velocity must land exactly on plane: v_n = -s/dt = -0.02 / 0.05 = -0.4
        # (so end-of-step pos would be s + v_n * dt = 0.02 - 0.4*0.05 = 0)
        assert model.v[0, 2] == pytest.approx(-0.4, rel=1e-12)
        # Tangential velocity components (vx, vy) must be EXACTLY preserved
        assert model.v[0, 0] == pytest.approx(3.0, rel=1e-12)
        assert model.v[0, 1] == pytest.approx(4.0, rel=1e-12)
        # Fixed wall books impact dissipation -U into de (contact energy, econt):
        # U = m [|v_new|^2/2 + |v_old|^2/2 - v_trial . v_old]
        # v_trial = v_old = [3, 4, -1] -> |v_old|^2 = 26
        # v_new = [3, 4, -0.4] -> |v_new|^2 = 25 + 0.16 = 25.16
        # U = 2.5 * [25.16/2 + 26/2 - 26] = 2.5 * [12.58 - 13.0] = 2.5 * (-0.42) = -1.05
        # de = -U = 1.05, dw = 0.0
        assert de == pytest.approx(1.05, rel=1e-12)
        assert dw == 0.0

    def test_resting_node_no_energy_leakage(self):
        """A resting node (v_old = 0) pushed against wall by external acceleration."""
        model = make_test_model(numnod=1, mass_val=5.0)
        model.x[0] = [0.0, 0.0, 0.0]       # sitting on wall (s = 0)
        model.v[0] = [0.0, 0.0, -0.5]      # acquired trial velocity from a*dt
        v_old = np.zeros((1, 3))           # entered cycle at rest
        dt = 0.01

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Node arrested: v_new = 0
        assert model.v[0, 2] == pytest.approx(0.0, abs=1e-14)
        # Dissipation must be exactly 0: resting node does not leak energy
        assert de == pytest.approx(0.0, abs=1e-14)
        assert dw == pytest.approx(0.0, abs=1e-14)

    def test_apply_fixed_plane_slide1_tied(self):
        """Node hitting fixed plane with slide=1 (tied: tangential killed)."""
        model = make_test_model(numnod=1, mass_val=2.0)
        model.x[0] = [0.0, 0.0, 0.01]
        model.v[0] = [5.0, -3.0, -1.0]     # large tangential velocity
        v_old = model.v.copy()
        dt = 0.02

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=1
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Tied: tangential velocity is fully arrested!
        assert model.v[0, 0] == pytest.approx(0.0, abs=1e-14)
        assert model.v[0, 1] == pytest.approx(0.0, abs=1e-14)
        # Normal velocity lands on plane: -0.01 / 0.02 = -0.5
        assert model.v[0, 2] == pytest.approx(-0.5, rel=1e-12)
        # Fixed wall books impact dissipation into de (contact energy)
        assert de > 0.0
        assert dw == 0.0

    def test_apply_coulomb_friction_stick_regime(self):
        """slide=2 with high friction: tangential velocity is completely stopped."""
        model = make_test_model(numnod=1, mass_val=1.0)
        model.x[0] = [0.0, 0.0, 0.0]
        # Normal impact: dvn = 1.0. With fric = 1.0, max friction dv = 1.0.
        # Tangential velocity vt = 0.3 < 1.0 -> stick!
        model.v[0] = [0.3, 0.0, -1.0]
        v_old = model.v.copy()
        dt = 0.01

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=2, fric=1.0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        walls.apply(model.x, model.v, v_old, model.mass, dt)

        assert model.v[0, 0] == pytest.approx(0.0, abs=1e-14)
        assert model.v[0, 2] == pytest.approx(0.0, abs=1e-14)

    def test_apply_coulomb_friction_slip_regime(self):
        """slide=2 with moderate friction: tangential velocity reduced by mu * |dvn|."""
        model = make_test_model(numnod=1, mass_val=1.0)
        model.x[0] = [0.0, 0.0, 0.0]
        # dvn = 2.0. With fric = 0.2, max dv_fric = 0.4.
        # Initial vt = 1.0 -> final vt should be 1.0 - 0.4 = 0.6.
        model.v[0] = [1.0, 0.0, -2.0]
        v_old = model.v.copy()
        dt = 0.01

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=2, fric=0.2
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        walls.apply(model.x, model.v, v_old, model.mass, dt)

        assert model.v[0, 0] == pytest.approx(0.6, rel=1e-12)
        assert model.v[0, 2] == pytest.approx(0.0, abs=1e-14)

    def test_search_distance_gating(self):
        """Search distance dist filters out far nodes even if they would cross."""
        model = make_test_model(numnod=2, mass_val=1.0)
        # Node 0 is outside search band (s = 0.10 > dist = 0.05)
        model.x[0] = [0.0, 0.0, 0.10]
        model.v[0] = [0.0, 0.0, -2.0]  # would end at -0.10 behind wall
        # Node 1 is inside search band (s = 0.04 <= dist = 0.05)
        model.x[1] = [0.0, 0.0, 0.04]
        model.v[1] = [0.0, 0.0, -2.0]  # would end at -0.16 behind wall

        v_old = model.v.copy()
        dt = 0.1

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=0, dist=0.05
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Node 0 was gated out -> velocity unchanged
        assert model.v[0, 2] == pytest.approx(-2.0, rel=1e-12)
        # Node 1 was hit -> normal velocity corrected
        assert model.v[1, 2] == pytest.approx(-0.04 / 0.1, rel=1e-12)


# ==============================================================================
# 3. Driven and Free Walls: Recoil, Momentum & Energy Balance
# ==============================================================================

class TestDrivenAndFreeWalls:
    def test_driven_wall_external_work(self):
        """Imposed-motion wall (/IMPVEL) kicks a stationary node."""
        model = make_test_model(numnod=2, mass_val=2.0)
        carrier_node_id = 99
        model.node_ids[1] = carrier_node_id
        model._id2idx[carrier_node_id] = 1
        model.mass[1] = 1e30  # carrier node is kinematically driven

        # Node 0 is resting on the wall (s = 0)
        model.x[0] = [0.0, 0.0, 0.0]
        model.v[0] = [0.0, 0.0, 0.0]
        # Carrier node moves at v_w = 4.0
        model.x[1] = [0.0, 0.0, 0.0]
        model.v[1] = [0.0, 0.0, 4.0]
        v_old = model.v.copy()
        dt = 0.01

        # Mark carrier node as driven via model.impvel
        model.node_groups[1] = NodeGroup(id=1, node_ids=[carrier_node_id], node_idx=np.array([1]))
        model.impvel = [ImposedVelocity(id=1, grnod_id=1, funct_id=0, dof=2, scale=1.0)]

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=carrier_node_id, slide=0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Node 0 must take the wall's velocity (4.0)
        assert model.v[0, 2] == pytest.approx(4.0, rel=1e-12)
        # Driven wall does positive external work:
        # U = m [|v_new|^2/2 + |v_old|^2/2 - v_trial.v_old] = 2.0 * [16/2 + 0 - 0] = 16.0
        assert dw == pytest.approx(16.0, rel=1e-12)
        assert de == 0.0

    def test_free_wall_single_node_exact_inelastic_collision(self):
        """Free wall with mass Mw colliding with node m1: exact momentum conservation."""
        model = make_test_model(numnod=2)
        carrier_node_id = 99
        model.node_ids[1] = carrier_node_id
        model._id2idx[carrier_node_id] = 1

        m_wall = 6.0
        m_node = 2.0
        model.mass[0] = m_node
        model.mass[1] = m_wall

        # Wall flies upward at v_w = 4.0
        model.x[1] = [0.0, 0.0, 0.0]
        model.v[1] = [0.0, 0.0, 4.0]
        # Node sits at s = 0 with v = 0
        model.x[0] = [0.0, 0.0, 0.0]
        model.v[0] = [0.0, 0.0, 0.0]
        v_old = model.v.copy()
        dt = 0.01

        # Total initial linear momentum
        p_initial = m_wall * 4.0 + m_node * 0.0  # 24.0

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=carrier_node_id, slide=0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Exact analytical post-collision common velocity:
        v_star = (m_wall * 4.0) / (m_wall + m_node)  # 24.0 / 8.0 = 3.0
        assert model.v[0, 2] == pytest.approx(v_star, rel=1e-12)
        assert model.v[1, 2] == pytest.approx(v_star, rel=1e-12)

        # Total final linear momentum
        p_final = m_wall * model.v[1, 2] + m_node * model.v[0, 2]
        assert p_final == pytest.approx(p_initial, rel=1e-14)

        # Energy balance:
        # KE_initial = 0.5 * 6 * 16 = 48.0
        # KE_final = 0.5 * 8 * 9 = 36.0
        # Expected dissipation = 48.0 - 36.0 = 12.0
        assert de == pytest.approx(12.0, rel=1e-12)
        assert dw == 0.0

    def test_free_wall_multinode_3d_momentum_conservation(self):
        """Free wall with mass Mw colliding with 3 nodes with different masses and normals."""
        model = make_test_model(numnod=4)
        carrier_node_id = 99
        model.node_ids[3] = carrier_node_id
        model._id2idx[carrier_node_id] = 3

        m_wall = 10.0
        model.mass[0] = 1.0
        model.mass[1] = 2.0
        model.mass[2] = 3.0
        model.mass[3] = m_wall

        # Wall flies in 3D direction
        model.v[3] = [2.0, 1.0, 5.0]
        model.x[3] = [0.0, 0.0, 0.0]

        # 3 candidate nodes flying in various directions towards wall
        model.x[0] = [1.0, 0.0, 0.01]
        model.v[0] = [0.5, 0.0, -1.0]

        model.x[1] = [-1.0, 2.0, 0.02]
        model.v[1] = [0.0, 1.0, -0.5]

        model.x[2] = [0.0, -1.0, 0.00]
        model.v[2] = [-0.5, -0.5, 0.0]

        v_old = model.v.copy()
        dt = 0.02

        p_init = (model.mass[:, None] * model.v).sum(axis=0)

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=carrier_node_id, slide=0
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        p_final = (model.mass[:, None] * model.v).sum(axis=0)
        # All 3 momentum components must be conserved to round-off
        np.testing.assert_allclose(p_final, p_init, atol=1e-13)

        # Closed energy accounting
        ke_init = 0.5 * (model.mass * np.sum(v_old**2, axis=1)).sum()
        ke_final = 0.5 * (model.mass * np.sum(model.v**2, axis=1)).sum()
        # Contact energy de must match kinetic energy loss
        assert de == pytest.approx(ke_init - ke_final, rel=1e-12)

    def test_free_wall_tangential_friction_recoil(self):
        """Free wall with Coulomb friction: tangential impulses react on carrier."""
        model = make_test_model(numnod=2)
        carrier_node_id = 99
        model.node_ids[1] = carrier_node_id
        model._id2idx[carrier_node_id] = 1
        model.mass[0] = 2.0
        model.mass[1] = 10.0

        model.x[1] = [0.0, 0.0, 0.0]
        model.v[1] = [0.0, 0.0, 2.0]  # wall moving in Z

        # Node moving in X and into wall in Z
        model.x[0] = [0.0, 0.0, 0.0]
        model.v[0] = [5.0, 0.0, -1.0]
        v_old = model.v.copy()
        dt = 0.01

        p_init = (model.mass[:, None] * model.v).sum(axis=0)

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=carrier_node_id, slide=2, fric=0.5
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, _ = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Friction should cause wall to accelerate in X direction!
        assert model.v[1, 0] > 0.0
        # Node decelerated in X
        assert model.v[0, 0] < 5.0

        p_final = (model.mass[:, None] * model.v).sum(axis=0)
        np.testing.assert_allclose(p_final, p_init, atol=1e-13)

    def test_multiple_sequential_walls(self):
        """Two walls: floor at z=0 and side wall at x=1."""
        model = make_test_model(numnod=1, mass_val=1.0)
        model.x[0] = [0.99, 0.0, 0.01]
        model.v[0] = [2.0, 0.0, -2.0]  # flying into both floor and side wall
        v_old = model.v.copy()
        dt = 0.02

        rw1 = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", slide=0
        )
        # Side wall at x=1 with normal [-1, 0, 0] (allowed region is x < 1)
        rw2 = RigidWall(
            id=2, point=np.array([1.0, 0.0, 0.0]), normal=np.array([-1.0, 0.0, 0.0]),
            geom="PLANE", slide=0
        )
        model.rwalls = [rw1, rw2]
        log = MessageLog()
        walls = RigidWalls(model, log)

        de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)

        # Both velocities should be corrected
        assert model.v[0, 2] == pytest.approx(-0.01 / 0.02, rel=1e-12)
        # For rw2: s = (1.0 - 0.99) = 0.01. vn = v . [-1, 0, 0] = -2.0.
        assert model.v[0, 0] == pytest.approx(0.5, rel=1e-12)
        assert de > 0.0
        assert dw == 0.0


# ==============================================================================
# 4. Defensive Initialization & Group Handling
# ==============================================================================

class TestRigidWallInitDefensive:
    def test_init_with_grnod_filter_and_exclusion(self):
        """grnod_id includes nodes, grnod_id2 excludes nodes."""
        model = make_test_model(numnod=6)
        # group 1 includes nodes 1, 2, 3, 4 (indices 0, 1, 2, 3)
        model.node_groups[1] = NodeGroup(id=1, node_ids=[1, 2, 3, 4], node_idx=np.array([0, 1, 2, 3]))
        # group 2 excludes node 2 (index 1)
        model.node_groups[2] = NodeGroup(id=2, node_ids=[2], node_idx=np.array([1]))

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", grnod_id=1, grnod_id2=2
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        candidates = walls.walls[0][1]
        np.testing.assert_array_equal(candidates, [0, 2, 3])

    def test_init_carrier_and_frozen_nodes_excluded(self):
        """Carrier node and nodes with mass >= 1e29 are never candidates."""
        model = make_test_model(numnod=4)
        carrier_node_id = 4
        model.mass[2] = 1e30  # frozen node (index 2)

        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=carrier_node_id
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)

        candidates = walls.walls[0][1]
        # index 2 (mass 1e30) and index 3 (carrier node 4) must not be in candidates
        assert 2 not in candidates
        assert 3 not in candidates
        np.testing.assert_array_equal(candidates, [0, 1])

    def test_init_missing_group_handled_gracefully(self):
        """Referencing a nonexistent grnod_id does not raise KeyError."""
        model = make_test_model(numnod=3)
        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", grnod_id=999
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)
        candidates = walls.walls[0][1]
        assert len(candidates) == 0

    def test_init_missing_carrier_node_logs_error(self):
        """Moving wall referencing missing carrier node logs error without crashing."""
        model = make_test_model(numnod=3)
        rw = RigidWall(
            id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
            geom="PLANE", node_id=999
        )
        model.rwalls = [rw]
        log = MessageLog()
        walls = RigidWalls(model, log)
        assert len(walls.walls) == 1
        assert any("carrier node 999 not found" in m for m in log.errors)


# ==============================================================================
# 5. Deck Reader Verification for /RWALL
# ==============================================================================

class TestDeckReaderRwall:
    def test_reader_rwall_paral_unnormalized_axes(self, tmp_path):
        """Starter deck reader preserves true unnormalized edge vectors for /RWALL/PARAL."""
        deck = """\
/BEGIN
Test RWALL PARAL
/NODE
1 0.0 0.0 0.0
/RWALL/PARAL/1
Parallelogram floor
0 0 0.0 0.0 0
0.0 0.0 0.0
10.0 0.0 0.0
0.0 5.0 0.0
"""
        dfile = tmp_path / "test_0000.rad"
        dfile.write_text(deck)
        model = run_starter(str(dfile))

        assert len(model.rwalls) == 1
        rw = model.rwalls[0]
        assert rw.geom == "PARAL"
        # axis1 must be unnormalized (length 10)
        np.testing.assert_allclose(rw.axis1, [10.0, 0.0, 0.0], rtol=1e-12)
        # axis2 must be unnormalized (length 5)
        np.testing.assert_allclose(rw.axis2, [0.0, 5.0, 0.0], rtol=1e-12)
        # normal must be unit length
        np.testing.assert_allclose(rw.normal, [0.0, 0.0, 1.0], rtol=1e-12)

    def test_reader_rwall_spher_negative_radius(self, tmp_path):
        """Starter deck reader handles negative radius for /RWALL/SPHER containment."""
        deck = """\
/BEGIN
Test RWALL SPHER
/NODE
1 0.0 0.0 0.0
/RWALL/SPHER/2
Spherical container
0 0 0.0 0.0 0
0.0 0.0 0.0
-7.5
"""
        dfile = tmp_path / "test_0000.rad"
        dfile.write_text(deck)
        model = run_starter(str(dfile))

        assert len(model.rwalls) == 1
        rw = model.rwalls[0]
        assert rw.geom == "SPHER"
        assert rw.radius == pytest.approx(-7.5, rel=1e-12)
