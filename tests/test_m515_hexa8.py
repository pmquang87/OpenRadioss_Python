"""
Comprehensive unit test suite for Milestone M515:
8-Node Hexahedral Solid Element (/BRICK, /HEXA8, /PROP/TYPE6, /PROP/TYPE14)
Hardening, Centroid Geometry & Degenerate Fallback, Flanagan-Belytschko Hourglass Control (shour3.F),
Physical Hourglass for LAW70 (s4phys.F), Jaumann Stress Rate (srota3.F),
24-DOF Material Tangent Stiffness (K_mat), 24-DOF Geometric Stiffness (K_geo),
2x2x2 Gauss Consistent Mass Matrix, Bulk Viscosity (sbulk3.F), Static Stabilization.
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.elements import solid_hexa8
from pyradioss.common.constants import EM20, EP30
from pyradioss.model.model import ElementGroup, Model


# ----------------------------------------------------------------------------
# Test Helpers & Dummy Objects
# ----------------------------------------------------------------------------
class _DummyMat:
    def __init__(self, law=1, rho0=1000.0, E=2.1e11, nu=0.3):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0 else 0.0
        self.K = E / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0 else 0.0
        self.fail = None
        self.eos = None
        self.params = {
            "E": E,
            "nu": nu,
            "sig_y": 200.0e6,
            "b": 500.0e6,
            "n": 0.5,
            "eps_max": 1e30,
            "eps_p_max": 1e30,
        }

    def sound_speed_solid(self):
        return np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0) if (self.rho0 > 0.0 and self.E > 0.0) else 0.0



class _DummyProp:
    def __init__(self, qa=1.1, qb=0.05, h=0.1):
        self.qa = qa
        self.qb = qb
        self.h = h
        self.params = {"qa": qa, "qb": qb, "h": h}


class _DummyLog:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, tag=""):
        self.warnings.append((tag, msg))

    def error(self, msg, tag=""):
        self.errors.append((tag, msg))

    def info(self, msg, tag=""):
        pass


def _create_unit_hexa8():
    """Create a model with a single 8-node unit cube [0, 1]^3."""
    model = Model()
    # Map reference coordinates [-1, 1]^3 to unit cube [0, 1]^3: x = 0.5 * (xi + 1)
    x0 = 0.5 * (solid_hexa8._XI + 1.0)
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((8, 3), dtype=np.float64)
    model.vr = np.zeros((8, 3), dtype=np.float64)

    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp(qa=1.1, qb=0.05, h=0.1)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


# ----------------------------------------------------------------------------
# 1. Standard init_group
# ----------------------------------------------------------------------------
def test_hexa8_init_group_standard():
    model, group, mat, prop = _create_unit_hexa8()
    log = _DummyLog()
    nids, masses, _ = solid_hexa8.init_group(group, model, log)

    assert len(log.errors) == 0
    assert len(nids) == 8
    # Total volume = 1.0, rho = 1000.0 -> total mass = 1000.0 -> 125.0 per node
    assert np.isclose(masses.sum(), 1000.0)
    assert np.allclose(masses, 125.0)
    assert np.isclose(group.state["vol0"][0], 1.0)
    assert group.state["sig"].shape == (1, 6)
    assert group.state["off"][0] == 1.0
    assert group.state["dtfac"][0] > 0.0


# ----------------------------------------------------------------------------
# 2. Empty init_group
# ----------------------------------------------------------------------------
def test_hexa8_init_group_empty():
    model = Model()
    model.x0 = np.zeros((0, 3), dtype=np.float64)
    group = ElementGroup(ids=np.empty(0, dtype=np.int64), conn=np.empty((0, 8), dtype=np.int64), part=np.empty(0, dtype=np.int64))
    group.state["slices"] = []
    log = _DummyLog()
    nids, masses, _ = solid_hexa8.init_group(group, model, log)

    assert len(nids) == 0
    assert len(masses) == 0
    assert group.state["sig"].shape == (0, 6)
    assert len(group.state["vol0"]) == 0


# ----------------------------------------------------------------------------
# 3. Geometry regular unit cube
# ----------------------------------------------------------------------------
def test_hexa8_geometry_regular():
    model, group, _, _ = _create_unit_hexa8()
    xe = model.x0[group.conn]
    dndx, vol = solid_hexa8._geometry(xe)

    assert np.isclose(vol[0], 1.0)
    # Partition of unity: sum_i dN_i/dx = 0
    assert np.allclose(dndx[0].sum(axis=0), 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 4. Geometry degenerate / zero volume fallback
# ----------------------------------------------------------------------------
def test_hexa8_geometry_degenerate_zero_volume():
    # Collapsed element where all z coordinates are zero
    xe = np.zeros((1, 8, 3), dtype=np.float64)
    xe[0, :, :2] = 0.5 * (solid_hexa8._XI[:, :2] + 1.0)
    dndx, vol = solid_hexa8._geometry(xe)

    assert vol[0] >= EM20
    assert not np.any(np.isnan(dndx))
    assert not np.any(np.isnan(vol))
    assert np.all(dndx == 0.0)


# ----------------------------------------------------------------------------
# 5. Global DOF mapping _edofs
# ----------------------------------------------------------------------------
def test_hexa8_edofs():
    conn = np.array([[1, 3, 5, 7, 2, 4, 6, 8]], dtype=np.int64)
    edofs = solid_hexa8._edofs(conn)

    assert edofs.shape == (1, 24)
    # Check node 1 (node id 3): dofs 18, 19, 20
    assert np.array_equal(edofs[0, 3:6], [18, 19, 20])
    # Check empty conn
    empty_edofs = solid_hexa8._edofs(np.empty((0, 8), dtype=np.int64))
    assert empty_edofs.shape == (0, 24)


# ----------------------------------------------------------------------------
# 6. Forces empty group
# ----------------------------------------------------------------------------
def test_hexa8_forces_empty():
    group = ElementGroup(ids=np.empty(0, dtype=np.int64), conn=np.empty((0, 8), dtype=np.int64), part=np.empty(0, dtype=np.int64))
    group.state["slices"] = []
    x = np.zeros((0, 3))
    v = np.zeros((0, 3))
    dt_crit = solid_hexa8.forces(group, x, v, None, 1e-6, None, None)
    assert len(dt_crit) == 0


# ----------------------------------------------------------------------------
# 7. Forces cycle 0 Courant probe (dt <= 0 or v is None)
# ----------------------------------------------------------------------------
def test_hexa8_forces_cycle0_dt0():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    dt_crit = solid_hexa8.forces(group, model.x, None, None, 0.0, None, None)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
    assert dt_crit[0] < 1.0
    # Verify no stress or energy was accumulated
    assert np.all(group.state["sig"] == 0.0)
    assert np.all(group.state["eint"] == 0.0)


# ----------------------------------------------------------------------------
# 8. Rigid body translation (zero strain, zero forces)
# ----------------------------------------------------------------------------
def test_hexa8_forces_rigid_body_translation():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Apply uniform translation velocity [10.0, -5.0, 3.0]
    model.v[:] = [10.0, -5.0, 3.0]
    fint = np.zeros((8, 3))
    dt = 1e-6
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # Internal and hourglass forces must remain zero
    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(group.state["sig"], 0.0, atol=1e-10)
    assert np.allclose(group.state["eint"], 0.0, atol=1e-10)
    assert np.allclose(group.state["ehour"], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 9. Rigid body rotation (Jaumann stress rotation)
# ----------------------------------------------------------------------------
def test_hexa8_forces_rigid_body_rotation():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Initialize with pure normal stress in x: sig_xx = 100 MPa
    group.state["sig"][0] = [100.0e6, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Rigid rotation about z with angular velocity omega = 100 rad/s
    omega = 100.0
    for i in range(8):
        x, y, z = model.x[i]
        model.v[i] = [-omega * y, omega * x, 0.0]

    dt = 1e-4
    fint = np.zeros((8, 3))
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # Jaumann rate: sig_xy should grow by omega * dt * sig_xx
    expected_sxy = omega * dt * 100.0e6
    assert np.isclose(group.state["sig"][0, 3], expected_sxy, rtol=1e-3)
    # Energy error should remain zero (no plastic or strain dissipation)
    assert np.isclose(group.state["eint"][0], 0.0, atol=1e-3)


# ----------------------------------------------------------------------------
# 10. Uniform tension and nodal equilibrium
# ----------------------------------------------------------------------------
def test_hexa8_forces_uniform_tension():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Apply uniform strain rate eps_dot_xx = 1.0
    for i in range(8):
        model.v[i] = [1.0 * model.x[i, 0], 0.0, 0.0]

    dt = 1e-4
    fint = np.zeros((8, 3))
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # Normal stress sig_xx > 0, others from Poisson's effect
    assert group.state["sig"][0, 0] > 0.0
    # Global equilibrium sum(fint) == 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)
    # Face equilibrium: nodes with x=1 should have fx > 0 (resisting pull)
    x1_nodes = np.where(model.x[:, 0] > 0.5)[0]
    assert np.all(fint[x1_nodes, 0] < 0.0)  # fint is -integral(B^T sig)


# ----------------------------------------------------------------------------
# 11. Hourglass viscous damping
# ----------------------------------------------------------------------------
def test_hexa8_hourglass_viscous_damping():
    model, group, _, prop = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Excite hourglass mode 0: h = [1, 1, -1, -1, -1, -1, 1, 1]
    h0 = solid_hexa8._H[0]
    for i in range(8):
        model.v[i] = [h0[i] * 10.0, 0.0, 0.0]

    fint = np.zeros((8, 3))
    dt = 1e-5
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # Forces should oppose velocity: fint . v should dissipate energy
    assert group.state["ehour"][0] > 0.0
    # Equilibrium still holds: sum(fint) == 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 12. Hourglass orthogonality to linear fields
# ----------------------------------------------------------------------------
def test_hexa8_hourglass_orthogonal_to_linear_field():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Pure shear velocity: vx = 5.0 * y, vy = 5.0 * x
    for i in range(8):
        model.v[i] = [5.0 * model.x[i, 1], 5.0 * model.x[i, 0], 0.0]

    fint = np.zeros((8, 3))
    dt = 1e-5
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # Hourglass energy must be identically zero on linear velocity fields
    assert np.isclose(group.state["ehour"][0], 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 13. Bulk viscosity under compression
# ----------------------------------------------------------------------------
def test_hexa8_bulk_viscosity():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Volumetric compression: v = -10.0 * x
    model.v[:] = -10.0 * model.x
    fint = np.zeros((8, 3))
    dt = 1e-5
    solid_hexa8.forces(group, model.x, model.v, None, dt, fint, None)

    # qvw_pend should be strictly positive under compression
    assert group.state["qvw_pend"][0] > 0.0
    assert group.state["eint"][0] > 0.0


# ----------------------------------------------------------------------------
# 14. Tangent empty group
# ----------------------------------------------------------------------------
def test_hexa8_tangent_empty():
    group = ElementGroup(ids=np.empty(0, dtype=np.int64), conn=np.empty((0, 8), dtype=np.int64), part=np.empty(0, dtype=np.int64))
    group.state["slices"] = []
    ke, edofs = solid_hexa8.tangent(group, np.zeros((0, 3)))

    assert ke.shape == (0, 24, 24)
    assert edofs.shape == (0, 24)


# ----------------------------------------------------------------------------
# 15. Tangent symmetry and rigid body modes
# ----------------------------------------------------------------------------
def test_hexa8_tangent_symmetry_and_rigid_modes():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    ke, edofs = solid_hexa8.tangent(group, model.x)

    assert ke.shape == (1, 24, 24)
    K = ke[0]
    # Symmetry: K = K^T
    assert np.allclose(K, K.T, atol=1e-5)

    # Eigenvalues: 6 zero rigid-body modes (3 translations + 3 rotations)
    eigvals = np.sort(np.linalg.eigvalsh(K))
    assert np.all(eigvals[:6] < 1e-4 * eigvals[-1])
    assert eigvals[6] > 1e-3 * eigvals[-1]  # stabilized by K_hg


# ----------------------------------------------------------------------------
# 16. Tangent zero for void or dead elements
# ----------------------------------------------------------------------------
def test_hexa8_tangent_void_or_dead():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Mark element as deleted
    group.state["off"][0] = 0.0
    ke, edofs = solid_hexa8.tangent(group, model.x)
    assert np.all(ke[0] == 0.0)

    # Restore and set law=0
    group.state["off"][0] = 1.0
    group.state["slices"][0][1].law = 0
    ke, edofs = solid_hexa8.tangent(group, model.x)
    assert np.all(ke[0] == 0.0)


# ----------------------------------------------------------------------------
# 17. Consistent mass row sum conservation
# ----------------------------------------------------------------------------
def test_hexa8_consistent_mass_conservation():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    me, edofs = solid_hexa8.consistent_mass(group, model.x0)
    assert me.shape == (1, 24, 24)
    M = me[0]

    # For rectangular brick, row sums of M corresponding to translation DOFs
    # must equal lumped nodal mass = total mass / 8 = 1000 / 8 = 125.0
    row_sums = M.sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], 125.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 18. Consistent mass positive definiteness
# ----------------------------------------------------------------------------
def test_hexa8_consistent_mass_positive_definite():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    me, edofs = solid_hexa8.consistent_mass(group, model.x0)
    M = me[0]
    # M must be symmetric
    assert np.allclose(M, M.T, atol=1e-12)
    # All eigenvalues must be strictly positive
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)


# ----------------------------------------------------------------------------
# 19. Geometric stiffness K_geo under initial stress
# ----------------------------------------------------------------------------
def test_hexa8_kgeo_initial_stress():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    # Zero stress -> K_geo == 0
    ke_zero, _ = solid_hexa8.kgeo(group, model.x)
    assert np.all(ke_zero == 0.0)

    # Uniaxial tension sig_xx = 100 MPa
    group.state["sig"][0] = [100.0e6, 0.0, 0.0, 0.0, 0.0, 0.0]
    ke, _ = solid_hexa8.kgeo(group, model.x)
    assert not np.all(ke == 0.0)
    assert np.allclose(ke[0], ke[0].T)

    # Deleted element gets zero K_geo
    group.state["off"][0] = 0.0
    ke_dead, _ = solid_hexa8.kgeo(group, model.x)
    assert np.all(ke_dead == 0.0)


# ----------------------------------------------------------------------------
# 20. Static stabilization and static internal forces
# ----------------------------------------------------------------------------
def test_hexa8_static_stabilization_and_internal_forces():
    model, group, _, _ = _create_unit_hexa8()
    log = _DummyLog()
    solid_hexa8.init_group(group, model, log)

    u = np.zeros((8, 3))
    # Apply displacement along hourglass mode 0
    h0 = solid_hexa8._H[0]
    for i in range(8):
        u[i] = [h0[i] * 0.01, 0.0, 0.0]

    fint = np.zeros((8, 3))
    solid_hexa8.static_stabilization(group, model.x, u, None, fint, None)

    # Should produce restoring forces and accumulate modal displacement
    assert not np.allclose(fint, 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-10)
    assert not np.allclose(group.state["hgq"], 0.0)

    # Now test static_internal_forces at deformed configuration
    fint_stat = np.zeros((8, 3))
    solid_hexa8.static_internal_forces(group, model.x + u, u, None, fint_stat, None)
    assert np.allclose(fint_stat.sum(axis=0), 0.0, atol=1e-4)
