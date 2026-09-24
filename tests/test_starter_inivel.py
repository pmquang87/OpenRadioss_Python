"""
Unit tests for /INIVEL (Initial Velocity Generator).

Ported from OpenRadioss Fortran:
- ``starter/source/initial_conditions/general/inivel/inivel.F``
- ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F``
"""

import math
from typing import List, Optional

import numpy as np
import pytest

from pyradioss.starter.inivel import (
    InivelRecord,
    InivelType,
    apply_bcs_mask,
    apply_inivel,
    build_cylindrical_inivel,
    build_rotational_inivel,
    build_spherical_inivel,
    build_translational_inivel,
)


class MockNodeGroup:
    """Mock node group matching pyradioss NodeGroup."""

    def __init__(self, id: int, node_idx: Optional[List[int]] = None, node_ids: Optional[List[int]] = None):
        self.id = id
        self.node_idx = node_idx if node_idx is not None else []
        self.node_ids = node_ids if node_ids is not None else []


class MockBoundaryCondition:
    """Mock boundary condition matching pyradioss BoundaryCondition."""

    def __init__(
        self,
        id: int = 1,
        grnod_id: Optional[int] = None,
        node_idx: Optional[List[int]] = None,
        node_ids: Optional[List[int]] = None,
        fix_tra: Optional[List[bool]] = None,
        skew_id: int = 0,
    ):
        self.id = id
        self.grnod_id = grnod_id
        self.node_idx = node_idx
        self.node_ids = node_ids
        self.fix_tra = fix_tra if fix_tra is not None else [False, False, False]
        self.skew_id = skew_id


class MockSkews:
    """Mock skew registry with coordinate transformations."""

    def __init__(self):
        self.axes = {}

    def index(self, kind: str, skew_id: int) -> int:
        if skew_id in self.axes:
            return skew_id
        return -1


class MockModel:
    """Mock FE Model containing nodal coordinates, node groups, and BCS."""

    def __init__(self, x0: np.ndarray):
        self.x0 = np.asarray(x0, dtype=float)
        self.x = np.copy(self.x0)
        self.numnod = len(self.x0)
        self.v = np.zeros((self.numnod, 3), dtype=float)
        self.vr = np.zeros((self.numnod, 3), dtype=float)
        self.node_groups = {}
        self.bcs = []
        self._id2idx = {i + 1: i for i in range(self.numnod)}

    def node_index(self, user_id: int) -> int:
        return self._id2idx[user_id]


def test_inivel_record_initialization_and_units():
    """Verify InivelRecord normalization, defaults, and angular velocity conversions."""
    # 1. Type normalization aliases
    rec_tra = InivelRecord(type="TRA", vx=1.0)
    assert rec_tra.type == InivelType.TRANSLATIONAL

    rec_rot = InivelRecord(type="AXIS", omega=10.0)
    assert rec_rot.type == InivelType.ROTATIONAL

    rec_cyl = InivelRecord(type="CYL", vr=5.0)
    assert rec_cyl.type == InivelType.CYLINDRICAL

    rec_sph = InivelRecord(type="SPHERICAL", vr=100.0)
    assert rec_sph.type == InivelType.SPHERICAL

    # 2. Angular velocity unit conversion
    rec_rad = InivelRecord(type=InivelType.ROTATIONAL, omega=15.5, omega_unit="rad/s")
    assert rec_rad.omega_rad_s == pytest.approx(15.5)

    # 60 RPM = 2 * pi * 60 / 60 = 2 * pi rad/s
    rec_rpm = InivelRecord(type=InivelType.ROTATIONAL, omega=60.0, omega_unit="rev/min")
    assert rec_rpm.omega_rad_s == pytest.approx(2.0 * math.pi)

    rec_rpm2 = InivelRecord(type=InivelType.ROTATIONAL, omega=120.0, omega_unit="rpm")
    assert rec_rpm2.omega_rad_s == pytest.approx(4.0 * math.pi)

    # 3. Vector normalization
    rec_axis = InivelRecord(type=InivelType.ROTATIONAL, u_rot=[0.0, 3.0, 4.0])
    assert np.allclose(rec_axis.u_rot, [0.0, 0.6, 0.8])
    assert np.allclose(rec_axis.u_axis, [0.0, 0.6, 0.8])


def test_uniform_translational_velocity_across_nodes():
    """Verify uniform translational velocity (vx, vy, vz) applied across nodes."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-5.0, 10.0, 2.5],
        [100.0, -50.0, 0.0],
    ])
    model = MockModel(coords)

    rec = build_translational_inivel(vx=12.5, vy=-8.0, vz=4.2)
    v = apply_inivel(model, rec)

    assert v.shape == (4, 3)
    expected = np.array([12.5, -8.0, 4.2])
    for i in range(4):
        assert np.allclose(v[i], expected)
        assert np.allclose(model.v[i], expected)


def test_rigid_body_rotation_around_z_axis():
    """Verify rigid body rotation v = omega x r around Z-axis."""
    # Z-axis through origin: x0 = (0, 0, 0), u_rot = (0, 0, 1), omega = 10 rad/s
    coords = np.array([
        [0.0, 0.0, 0.0],   # On rotation axis -> v = 0
        [1.0, 0.0, 0.0],   # (1, 0, 0) -> v = (0, 10, 0)
        [0.0, 2.0, 0.0],   # (0, 2, 0) -> v = (-20, 0, 0)
        [-1.0, 0.0, 0.0],  # (-1, 0, 0) -> v = (0, -10, 0)
        [0.0, -3.0, 0.0],  # (0, -3, 0) -> v = (30, 0, 0)
        [0.0, 0.0, 50.0],  # On rotation axis at z=50 -> v = 0
    ])
    model = MockModel(coords)

    rec = build_rotational_inivel(
        x0=[0.0, 0.0, 0.0],
        u_rot=[0.0, 0.0, 1.0],
        omega=10.0,
    )
    v = apply_inivel(model, rec)

    assert np.allclose(v[0], [0.0, 0.0, 0.0])
    assert np.allclose(v[1], [0.0, 10.0, 0.0])
    assert np.allclose(v[2], [-20.0, 0.0, 0.0])
    assert np.allclose(v[3], [0.0, -10.0, 0.0])
    assert np.allclose(v[4], [30.0, 0.0, 0.0])
    assert np.allclose(v[5], [0.0, 0.0, 0.0])

    # Rotational DOFs in model.vr
    expected_omega = np.array([0.0, 0.0, 10.0])
    for i in range(6):
        assert np.allclose(model.vr[i], expected_omega)


def test_rigid_body_rotation_around_arbitrary_3d_axis():
    """Verify rigid body rotation around arbitrary 3D axis with translation."""
    # Rotation axis passing through x0 = (1, 2, 3), pointing along (1, 1, 1)/sqrt(3)
    x0 = [1.0, 2.0, 3.0]
    u_rot = [1.0, 1.0, 1.0]  # will be normalized to [1, 1, 1] / sqrt(3)
    omega = 5.0
    v_trans = [2.0, -1.0, 4.0]

    rec = build_rotational_inivel(
        x0=x0,
        u_rot=u_rot,
        omega=omega,
        vx=v_trans[0],
        vy=v_trans[1],
        vz=v_trans[2],
    )

    # Test nodes
    u_unit = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    coords = np.array([
        [1.0, 2.0, 3.0],                  # Center x0 -> v = v_trans
        [1.0 + 2.0 / math.sqrt(3),        # Point on the rotation axis -> v = v_trans
         2.0 + 2.0 / math.sqrt(3),
         3.0 + 2.0 / math.sqrt(3)],
        [1.0, 3.0, 3.0],                  # Off-axis point: r = (0, 1, 0)
        [5.0, -2.0, 7.0],                 # General 3D point
    ])
    model = MockModel(coords)
    v = apply_inivel(model, rec)

    # Check center & on-axis nodes
    assert np.allclose(v[0], v_trans)
    assert np.allclose(v[1], v_trans)

    # For each node, verify analytical expression: v = v_trans + omega * (u x (x - x0))
    for i, pt in enumerate(coords):
        r = pt - np.array(x0)
        v_rot_expected = omega * np.cross(u_unit, r)
        v_expected = np.array(v_trans) + v_rot_expected
        assert np.allclose(v[i], v_expected)

        # Rotational velocity component must be orthogonal to rotation axis
        assert abs(np.dot(v[i] - np.array(v_trans), u_unit)) < 1e-12


def test_cylindrical_swirl_and_radial_expansion():
    """Verify cylindrical decomposition: v = vz * u_axis + vr * e_r + vtheta * e_theta."""
    # Cylinder axis along Z through (0, 0, 0)
    # vr = 10, vtheta = 20, vz = 30
    rec = build_cylindrical_inivel(
        x0=[0.0, 0.0, 0.0],
        u_axis=[0.0, 0.0, 1.0],
        vr=10.0,
        vtheta=20.0,
        vz=30.0,
    )

    coords = np.array([
        [0.0, 0.0, 5.0],  # On axis: distance=0 -> vr=0, vtheta=0, vz=30
        [3.0, 4.0, 10.0], # r_perp = (3, 4, 0), d_perp = 5 -> e_r = (0.6, 0.8, 0), e_theta = (-0.8, 0.6, 0)
        [5.0, 0.0, 0.0],  # r_perp = (5, 0, 0) -> e_r = (1, 0, 0), e_theta = (0, 1, 0)
        [0.0, -2.0, -4.0],# r_perp = (0, -2, 0) -> e_r = (0, -1, 0), e_theta = (1, 0, 0)
    ])
    model = MockModel(coords)
    v = apply_inivel(model, rec)

    # 1. Node on axis: purely axial
    assert np.allclose(v[0], [0.0, 0.0, 30.0])

    # 2. Node at (3, 4, 10):
    # e_r = (0.6, 0.8, 0.0), e_theta = (0, 0, 1) x (0.6, 0.8, 0) = (-0.8, 0.6, 0.0)
    # v = 30*(0,0,1) + 10*(0.6, 0.8, 0) + 20*(-0.8, 0.6, 0)
    # vx = 6 - 16 = -10.0
    # vy = 8 + 12 = 20.0
    # vz = 30.0
    assert np.allclose(v[1], [-10.0, 20.0, 30.0])

    # 3. Node at (5, 0, 0):
    # e_r = (1, 0, 0), e_theta = (0, 1, 0)
    # v = (10, 20, 30)
    assert np.allclose(v[2], [10.0, 20.0, 30.0])

    # 4. Node at (0, -2, -4):
    # e_r = (0, -1, 0), e_theta = (1, 0, 0)
    # vx = 20.0, vy = -10.0, vz = 30.0
    assert np.allclose(v[3], [20.0, -10.0, 30.0])


def test_cylindrical_arbitrary_orientation():
    """Verify cylindrical velocity field with tilted cylinder axis."""
    # Axis through (1, 1, 0) directed along X-axis (1, 0, 0)
    rec = build_cylindrical_inivel(
        x0=[1.0, 1.0, 0.0],
        u_axis=[1.0, 0.0, 0.0],
        vr=15.0,
        vtheta=25.0,
        vz=5.0,
    )

    # Point at (10.0, 1.0, 2.0) -> relative to (1, 1, 0) is (9, 0, 2)
    # Axial projection: z_proj = 9.0 along (1, 0, 0)
    # r_perp = (0, 0, 2) -> d_perp = 2 -> e_r = (0, 0, 1)
    # e_theta = (1, 0, 0) x (0, 0, 1) = (0, -1, 0)
    coords = np.array([[10.0, 1.0, 2.0]])
    model = MockModel(coords)
    v = apply_inivel(model, rec)

    # v = 5*(1, 0, 0) + 15*(0, 0, 1) + 25*(0, -1, 0) = (5, -25, 15)
    assert np.allclose(v[0], [5.0, -25.0, 15.0])


def test_spherical_expansion():
    """Verify spherical radial expansion radiating from explosion center."""
    x0 = [2.0, 3.0, 4.0]
    vr = 50.0

    rec = build_spherical_inivel(x0=x0, vr=vr)

    coords = np.array([
        [2.0, 3.0, 4.0],  # Center -> distance=0 -> v = (0, 0, 0)
        [2.0, 6.0, 8.0],  # r = (0, 3, 4), dist = 5 -> e_rad = (0, 0.6, 0.8) -> v = (0, 30, 40)
        [5.0, 3.0, 4.0],  # r = (3, 0, 0), dist = 3 -> e_rad = (1, 0, 0) -> v = (50, 0, 0)
        [1.0, 2.0, 4.0],  # r = (-1, -1, 0), dist = sqrt(2) -> v = 50*(-1/sqrt(2), -1/sqrt(2), 0)
    ])
    model = MockModel(coords)
    v = apply_inivel(model, rec)

    # Center point has zero velocity
    assert np.allclose(v[0], [0.0, 0.0, 0.0])

    # (2, 6, 8): v = 50 * (0, 0.6, 0.8) = (0, 30, 40)
    assert np.allclose(v[1], [0.0, 30.0, 40.0])

    # (5, 3, 4): v = (50, 0, 0)
    assert np.allclose(v[2], [50.0, 0.0, 0.0])

    # (-1, -1, 0)/sqrt(2)
    s2 = math.sqrt(2.0)
    assert np.allclose(v[3], [-50.0 / s2, -50.0 / s2, 0.0])


def test_bcs_constraint_zeroing_fixed_dofs():
    """Verify that /BCS fixed degrees of freedom mask and zero velocity components."""
    coords = np.array([
        [0.0, 0.0, 0.0],  # Node 0 (user ID 1): unconstrained
        [1.0, 0.0, 0.0],  # Node 1 (user ID 2): fix Tx
        [0.0, 1.0, 0.0],  # Node 2 (user ID 3): fix Ty and Tz
        [0.0, 0.0, 1.0],  # Node 3 (user ID 4): fully clamped (Tx, Ty, Tz)
    ])
    model = MockModel(coords)

    # Base translational velocity (vx=10, vy=20, vz=30)
    rec = build_translational_inivel(vx=10.0, vy=20.0, vz=30.0)

    # Setup boundary conditions
    bcs = [
        # Node 1: Tx fixed (tra="100")
        MockBoundaryCondition(id=1, node_idx=[1], fix_tra=[True, False, False]),
        # Node 2: Ty, Tz fixed
        MockBoundaryCondition(id=2, node_idx=[2], fix_tra=[False, True, True]),
        # Node 3: All fixed
        MockBoundaryCondition(id=3, node_idx=[3], fix_tra=[True, True, True]),
    ]
    model.bcs = bcs

    v = apply_inivel(model, rec)

    # Node 0: untouched -> (10, 20, 30)
    assert np.allclose(v[0], [10.0, 20.0, 30.0])
    # Node 1: vx=0 -> (0, 20, 30)
    assert np.allclose(v[1], [0.0, 20.0, 30.0])
    # Node 2: vy=0, vz=0 -> (10, 0, 0)
    assert np.allclose(v[2], [10.0, 0.0, 0.0])
    # Node 3: all zeroed -> (0, 0, 0)
    assert np.allclose(v[3], [0.0, 0.0, 0.0])


def test_node_group_targeting_and_additive_modes():
    """Verify selective application to node groups and additive superposition."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    model = MockModel(coords)

    # Define groups
    model.node_groups[10] = MockNodeGroup(id=10, node_idx=[0, 1])
    model.node_groups[20] = MockNodeGroup(id=20, node_idx=[1, 2])

    # Card 1: group 10 gets vx = 100.0
    rec1 = build_translational_inivel(vx=100.0, grnod_id=10)
    # Card 2: group 20 gets vy = 50.0 (additive)
    rec2 = build_translational_inivel(vy=50.0, grnod_id=20, additive=True)

    v = apply_inivel(model, [rec1, rec2])

    # Node 0 (group 10 only): vx=100, vy=0
    assert np.allclose(v[0], [100.0, 0.0, 0.0])
    # Node 1 (groups 10 & 20): vx=100, vy=50
    assert np.allclose(v[1], [100.0, 50.0, 0.0])
    # Node 2 (group 20 only): vx=0, vy=50
    assert np.allclose(v[2], [0.0, 50.0, 0.0])
    # Node 3 (neither group): untouched (0, 0, 0)
    assert np.allclose(v[3], [0.0, 0.0, 0.0])
