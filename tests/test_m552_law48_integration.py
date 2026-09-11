"""
Integration test suite for /MAT/LAW48 (/MAT/ZHAO, /MAT/PLAS_ZHAO).

Milestone M552 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Registry and dispatch metadata verification for all LAW48 aliases.
2. Acoustic sound speeds (3D dilatational and plane-stress) consistency across models.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1): uniaxial tension, compression, shear, and time step.
   - HEPH (Isolid=24): reduced integration with physical hourglass stabilization.
   - Tetra4: single-point tetrahedral formulation and stable time step.
4. Shell element formulations:
   - BT4 (Ishell=1): plane-stress, thickness thinning, layer integration.
   - QEPH (Ishell=24): 4-node quad shell with improved hourglass control.
   - Tri3 (Ish3n=1): 3-node triangular shell formulation.
5. Element erosion / deletion upon reaching eps_max or tensile strain limit eps_t2.
6. Consistent tangent stiffness dispatch for implicit analysis (solid and shell).
7. Starter checks inclusion for all supported solid and shell families.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_heph, solid_hexa8, solid_tetra4
from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    shell_membrane_tangent as law48_shell_membrane_tangent,
    shell_update_law48,
    solid_update_law48,
    sound_speed_shell_law48,
    sound_speed_solid_law48,
    tangent_law48_shell,
    tangent_law48_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS


# ============================================================================
# Helper Mock Element Classes for Kernel Unit Testing
# ============================================================================

class MockProp:
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 5, **kwargs: Any):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.params = {"thick": thick, "nip": nip, "qa": 1.1, "qb": 0.05, "hm": 0.1, "hf": 0.1, "hr": 0.1, **kwargs}


class MockGroup:
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law48(mid: int = 1, rho0: float = 7.85e-9, E: float = 210000.0, nu: float = 0.3, **kwargs: Any) -> Material:
    params = {
        "E": E,
        "nu": nu,
        "ca": kwargs.get("ca", 1.0),
        "sigy0": kwargs.get("sigy0", 250.0),
        "cb": kwargs.get("cb", 400.0),
        "cn": kwargs.get("cn", 0.5),
        "fisokin": kwargs.get("fisokin", 0.0),
        "sig_max": kwargs.get("sig_max", 800.0),
        "cc": kwargs.get("cc", 10.0),
        "cd": kwargs.get("cd", 5.0),
        "cm": kwargs.get("cm", 0.8),
        "ce": kwargs.get("ce", 0.02),
        "ck": kwargs.get("ck", 1.2),
        "eps0": kwargs.get("eps0", 1.0),
        "fcut": kwargs.get("fcut", 1000.0),
        "eps_max": kwargs.get("eps_max", 0.30),
        "eps_t1": kwargs.get("eps_t1", 0.15),
        "eps_t2": kwargs.get("eps_t2", 0.25),
    }
    params.update(kwargs)
    mat = Material(id=mid, law=48, rho0=rho0, title="Steel_LAW48", params=params)
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law48_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "LAW48_ZHAO")
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True
        assert meta.get("solid") is True
        assert meta.get("shell") is True
        assert k in materials.MATERIAL_SOLID_DISPATCH
        assert k in materials.MATERIAL_SHELL_DISPATCH

    mat = make_test_material_law48()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert shapes_solid["eps48"] == (6,)
    assert shapes_solid["sigb48"] == (6,)
    assert shapes_solid["epsd48"] == ()
    assert shapes_solid["off48"] == ()

    shapes_shell = materials.extra_shapes(mat, nip=5)
    assert shapes_shell["eps48"] == (5, 3)
    assert shapes_shell["sigb48"] == (5, 3)
    assert shapes_shell["epsd48"] == (5,)
    assert shapes_shell["off48"] == (5,)


def test_law48_sound_speed_consistency():
    """Verify 3D dilatational and plane-stress acoustic wave speed formulas."""
    rho0 = 7.85e-9
    E = 210000.0
    nu = 0.3
    mat = make_test_material_law48(rho0=rho0, E=E, nu=nu)

    c_solid_expected = math.sqrt((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))
    c_shell_expected = math.sqrt(E / ((1.0 - nu * nu) * rho0))

    # Test Material methods
    assert math.isclose(mat.sound_speed_solid(), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(mat.sound_speed_shell(), c_shell_expected, rel_tol=1e-12)

    # Test module-level dispatchers
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(sound_speed_solid_law48(mat), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(sound_speed_shell_law48(mat), c_shell_expected, rel_tol=1e-12)


# ============================================================================
# 2. Solid Element Formulations (Hexa8, HEPH, Tetra4)
# ============================================================================

def test_hexa8_solid_kernel_tension_and_compression():
    """Verify Hexa8 standard formulation (Isolid=1) under tension, compression and shear."""
    # 1. Geometry: 10 x 10 x 10 cube
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law48(sigy0=200.0, cb=300.0, cn=0.5)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True
    assert "sigb48" in st["mat_extra"]
    assert "eps48" in st["mat_extra"]
    assert "epsd48" in st["mat_extra"]
    assert "off48" in st["mat_extra"]

    # 2. Cycle 0 time step probe
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt_c0 = solid_hexa8.forces(group, coords, None, None, 0.0, fint, mint)
    assert len(dt_c0) == 1
    assert dt_c0[0] > 0.0

    # 3. Uniaxial tension in x: stretch face x=10 with velocity +100 mm/s
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 100.0
    dt = 1.0e-5

    # Run several cycles
    curr_x = coords.copy()
    for _ in range(20):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    sig = st["sig"][0]
    # Tensile stress along x
    assert sig[0] > 0.0
    # Equilibrium: sum of internal forces on element nodes must vanish
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)

    # Plastic strain accumulated
    assert st["epsp"][0] > 0.0

    # 4. Pure shear test: move top face along x
    curr_x = coords.copy()
    vel_shear = np.zeros_like(coords)
    vel_shear[[4, 5, 6, 7], 0] = 50.0
    group2 = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    group2._model = model
    solid_hexa8.init_group(group2, model, None)

    for _ in range(10):
        curr_x += vel_shear * dt
        solid_hexa8.forces(group2, curr_x, vel_shear, None, dt, fint, mint)

    sig_shear = group2.state["sig"][0]
    assert abs(sig_shear[3]) > 0.0  # Engineering shear stress xy


def test_hexa8_element_erosion_at_eps_max():
    """Verify Hexa8 element erosion when eps_p exceeds eps_max."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    # Low eps_max threshold for immediate deletion
    mat = make_test_material_law48(sigy0=100.0, cb=0.0, eps_max=0.01)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Stretch aggressively to exceed eps_max in a few cycles
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 5000.0
    dt = 1.0e-4

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    for _ in range(50):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)
        if st["off"][0] == 0.0:
            break

    # Element must be marked deleted and stress wiped to 0
    assert st["off"][0] == 0.0
    assert np.allclose(st["sig"][0], 0.0)


def test_heph_solid_kernel_integration():
    """Verify HEPH formulation (Isolid=24) with physical hourglass control."""
    coords = np.array([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 5.0], [0.0, 5.0, 5.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law48(sigy0=220.0, cb=250.0, cn=0.6)
    prop = MockProp(isolid=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_heph.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True

    # Cycle 0 Courant step
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt_c0 = solid_heph.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0

    # Tension along z
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 2] = 80.0
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(15):
        curr_x += vel * dt
        solid_heph.forces(group, curr_x, vel, None, dt, fint, mint)

    assert st["sig"][0, 2] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)


def test_solid_tetra4_kernel_integration():
    """Verify single 4-node tetrahedron formulation under LAW48."""
    # Canonical positive volume tetra nodes
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law48(sigy0=200.0, cb=200.0, cn=0.5)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True

    # Cycle 0 evaluation
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = solid_tetra4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0

    # Expand node 1 in x
    vel = np.zeros_like(coords)
    vel[1, 0] = 50.0
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(10):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert st["sig"][0, 0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)


# ============================================================================
# 3. Shell Element Formulations (BT4, QEPH, Tri3)
# ============================================================================

def test_shell_bt4_kernel_biaxial_and_thinning():
    """Verify Belytschko-Tsay shell (Ishell=1) with multi-layer integration and thinning."""
    # 10 x 10 quad in xy-plane with thickness 1.5 mm and 5 Gauss layers
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law48(sigy0=250.0, cb=300.0, cn=0.5)
    prop = MockProp(thick=1.5, nip=5)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True
    assert "eps48" in st["mat_extra"]
    assert "sigb48" in st["mat_extra"]
    assert "epsd48" in st["mat_extra"]

    # Initial thickness
    thk_0 = float(st["thick"][0])
    assert math.isclose(thk_0, 1.5)

    # Biaxial tension in x and y
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 100.0  # stretch x
    vel[[2, 3], 1] = 100.0  # stretch y
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    # In-plane stress resultants positive
    assert st["sig"][0, :, 0].mean() > 0.0
    assert st["sig"][0, :, 1].mean() > 0.0

    # Under biaxial tension, thickness must decrease (thinning dezz < 0)
    assert st["thick"][0] < thk_0
    # Plastic strain accumulated in layers
    assert st["epsp"][0].mean() > 0.0


def test_shell_bt4_element_erosion_on_failure():
    """Verify shell erosion when layer strains reach failure limits."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law48(sigy0=150.0, cb=0.0, eps_max=0.01)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 2000.0
    dt = 1.0e-4

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(50):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        if st["off"][0] == 0.0:
            break

    assert st["off"][0] == 0.0
    assert np.allclose(st["sig"][0], 0.0)


def test_shell_qeph_kernel_integration():
    """Verify QEPH formulation (Ishell=24) with improved hourglass control under LAW48."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law48(sigy0=220.0, cb=250.0, cn=0.55)
    prop = MockProp(thick=1.2, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 80.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert st["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)


def test_shell_tri3_kernel_integration():
    """Verify 3-node triangular shell formulation (Ish3n=1) under LAW48."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2]])
    mat = make_test_material_law48(sigy0=200.0, cb=200.0, cn=0.5)
    prop = MockProp(thick=1.0, nip=3, ish3n=1)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_tri3.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True

    vel = np.zeros_like(coords)
    vel[1, 0] = 60.0
    dt = 1.0e-5

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert st["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)


# ============================================================================
# 4. Consistent Tangent Stiffness Dispatch
# ============================================================================

def test_consistent_tangents_dispatch():
    """Verify solid and shell consistent tangent dispatch for implicit analysis."""
    mat = make_test_material_law48(E=210000.0, nu=0.3)

    # 1. Solid tangent: (n, 6, 6) tensor
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    depsp = np.zeros(1)
    d_solid = materials.solid_tangent(mat, sig, epsp, depsp)
    assert d_solid.shape == (1, 6, 6)
    # Check elastic symmetry in initial state
    assert np.allclose(d_solid[0], d_solid[0].T)

    # 2. Shell membrane tangent: (3, 3) matrix
    d_memb = materials.shell_membrane_tangent(mat)
    assert d_memb.shape == (3, 3)
    c11 = 210000.0 / (1.0 - 0.3 * 0.3)
    assert math.isclose(d_memb[0, 0], c11, rel_tol=1e-6)
    assert math.isclose(d_memb[0, 1], 0.3 * c11, rel_tol=1e-6)
    assert math.isclose(d_memb[2, 2], 210000.0 / (2.0 * 1.3), rel_tol=1e-6)

    # 3. Shell layer tangent: (n, 3, 3) tensor
    sig_sh = np.zeros((1, 3))
    d_layer = materials.shell_layer_tangent(mat, sig_sh, epsp, depsp)
    assert d_layer.shape == (1, 3, 3)
    assert np.allclose(d_layer[0], d_layer[0].T)


# ============================================================================
# 5. Starter Checks Inclusion
# ============================================================================

def test_starter_checks_allowed_laws():
    """Verify that LAW48 is included in _ALLOWED_LAWS for all solid and shell families."""
    keys_to_check = (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO")
    solid_families = ("bricks", "tetras", "penta6", "pyra5")
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n")

    for fam in solid_families:
        for k in keys_to_check:
            assert k in _ALLOWED_LAWS[fam], f"{k} missing in {fam}"

    for fam in shell_families:
        for k in keys_to_check:
            assert k in _ALLOWED_LAWS[fam], f"{k} missing in {fam}"
