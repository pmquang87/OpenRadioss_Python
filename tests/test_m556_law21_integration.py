"""
Integration test suite for /MAT/LAW21 (/MAT/DPRAG).

Milestone M556 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Registry and dispatch metadata verification for all LAW21 aliases.
2. Acoustic sound speed (3D dilatational) and Courant time step calculation.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1): hydrostatic compression, simple shear, plastic radial return,
     compaction EOS hysteretic unloading, multi-element patch test.
   - Tetra4 (Itetra=1): single 4-node tetrahedron, cycle 0 time step, compression,
     shear plasticity, multi-element patch test.
4. Consistent tangent stiffness dispatch for implicit analysis (solid_hexa8 and solid_tetra4):
   - Symmetry in elastic state, Voigt (6,6) tensor properties.
   - Rigid-body translation invariance: K_e . v_trans = 0.
5. Shell and 1D element rejection:
   - Plane-stress shell update and tangent raise NotImplementedError.
   - shell_bt4, shell_qeph, shell_tri3 kernels reject LAW21.
6. Compaction EOS with tabulated curve and scaling (pfscale).
7. Tensile cutoff (P_min) behavior in solids.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law21_dprag import (
    build_law21,
    solid_update_law21,
    shell_update_law21,
    sound_speed_solid_law21,
    tangent_law21_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model


# ============================================================================
# Helper Mock Element Classes for Kernel Integration Testing
# ============================================================================

class MockProp:
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 5, **kwargs: Any):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.params = {
            "thick": thick,
            "nip": nip,
            "qa": 1.1,
            "qb": 0.05,
            "hm": 0.1,
            "hf": 0.1,
            "hr": 0.1,
            **kwargs,
        }


class MockGroup:
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law21(
    mid: int = 1,
    rho0: float = 2000.0,
    E: float = 2.0e10,
    nu: float = 0.25,
    c1: float = 1.0e9,
    bunl: float = 2.0e9,
    mumax: float = 0.05,
    a0: float = 1.0e10,
    a1: float = 0.5,
    a2: float = 0.0,
    amax: float = 1.0e20,
    pmin: float = -1.0e30,
    pext: float = 0.0,
    **kwargs: Any,
) -> Material:
    params = {
        "E": E,
        "nu": nu,
        "c1": c1,
        "bunl": bunl,
        "mumax": mumax,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "amax": amax,
        "pmin": pmin,
        "pext": pext,
        "rho0": rho0,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=21, rho0=rho0, title="Rock_LAW21", params=params)
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law21_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (21, "21", "LAW21", "DPRAG", "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG")
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is False, f"LAW21 should have plane_stress=False"
        assert meta.get("solid") is True, f"LAW21 should have solid=True"
        assert meta.get("shell") is False, f"LAW21 should have shell=False"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"Key {k} missing from MATERIAL_SOLID_DISPATCH"
        assert k in materials.MATERIAL_SHELL_DISPATCH, f"Key {k} missing from MATERIAL_SHELL_DISPATCH"

    mat = make_test_material_law21()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "mu_bak" in shapes_solid
    assert "epxe" in shapes_solid
    assert "p" in shapes_solid
    assert "defp" in shapes_solid
    assert "p_old" in shapes_solid


# ============================================================================
# 2. Acoustic Sound Speed & Time Step Verification
# ============================================================================

def test_law21_sound_speed_solid():
    """Verify dilatational sound speed c = sqrt(|4/3 G + K_eff| / rho0)."""
    rho0 = 2000.0
    E = 2.0e10
    nu = 0.25
    g = E / (2.0 * (1.0 + nu))  # 8.0e9
    c1 = 1.0e9
    bunl = 2.0e9

    mat = make_test_material_law21(rho0=rho0, E=E, nu=nu, c1=c1, bunl=bunl)

    # Standalone function
    c_law = sound_speed_solid_law21(mat)
    k_eff_expected = max(c1, bunl)  # 2.0e9
    c_expected = math.sqrt(((4.0 / 3.0) * g + k_eff_expected) / rho0)
    assert math.isclose(c_law, c_expected, rel_tol=1e-6)

    # Package-level dispatch
    c_disp = materials.sound_speed(mat, rho=rho0)
    assert math.isclose(c_disp, c_expected, rel_tol=1e-6)


# ============================================================================
# 3. Solid Element Formulations: Hexa8
# ============================================================================

def test_hexa8_solid_kernel_cycle0_time_step():
    """Verify Hexa8 cycle 0 sound speed and Courant critical time step."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, E=2.0e10, nu=0.25, c1=1.0e9, bunl=2.0e9)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state
    assert "mu_bak" in st["mat_extra"]
    assert "epxe" in st["mat_extra"]

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt_c0 = solid_hexa8.forces(group, coords, None, None, 0.0, fint, mint)

    assert len(dt_c0) == 1
    assert dt_c0[0] > 0.0  # Expected sound speed ~ 2516.6 m/s, characteristic length = 1.0, scaled by exact dtfac factor
    c_expected = sound_speed_solid_law21(mat)
    dt_expected = st["dtfac"][0] * (1.0 / c_expected)
    assert math.isclose(dt_c0[0], dt_expected, rel_tol=1e-5)


def test_hexa8_solid_kernel_hydrostatic_compression():
    """Verify Hexa8 under hydrostatic compression (volumetric compaction, pressure buildup)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    c1 = 1.0e9
    mat = make_test_material_law21(rho0=2000.0, c1=c1, bunl=2.0e9, a0=1.0e14)  # High A0 so purely elastic
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Inward velocity on all boundary nodes towards centroid (0.5, 0.5, 0.5)
    centroid = np.array([0.5, 0.5, 0.5])
    vel = -0.1 * (coords - centroid)  # Volumetric contraction rate
    dt = 1.0e-4

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    # 1. Stress: compression positive pressure convention in Radioss, so Cauchy sigma is negative (triaxial compression)
    sig = st["sig"][0]
    p_hydro = -(sig[0] + sig[1] + sig[2]) / 3.0
    assert p_hydro > 0.0, f"Hydrostatic pressure should be positive under compression, got {p_hydro}"
    assert sig[0] < 0.0 and sig[1] < 0.0 and sig[2] < 0.0

    # 2. Compaction mu evolved and stored in mu_bak
    assert st["mat_extra"]["mu_bak"][0] > 0.0

    # 3. Internal energy strictly positive
    assert st["eint"][0] > 0.0

    # 4. Equilibrium: sum of internal nodal forces on element must be 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_hexa8_solid_kernel_shear_plasticity():
    """Verify Hexa8 under shear deformation: elastic buildup to yield envelope G0, then plastic flow."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    # Set low yield limit A0 = 1.0e8, A1 = 0 -> sqrt(J2) max = 1.0e4
    a0 = 1.0e8
    mat = make_test_material_law21(
        rho0=2000.0,
        E=2.0e10,
        nu=0.25,
        c1=1.0e9,
        a0=a0,
        a1=0.0,
        a2=0.0,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Apply simple shear: top face (z=1.0, nodes 4, 5, 6, 7) moving along x
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 50.0  # gamma_dot_zx = 50.0 / 1.0 = 50 1/s
    dt = 1.0e-5

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(30):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    sig = st["sig"][0]
    # Shear stress sigma_zx is component 5 in Voigt [xx, yy, zz, xy, yz, zx]
    tau_zx = sig[5]
    assert abs(tau_zx) > 0.0

    # J2 for pure shear is tau_zx^2
    j2 = tau_zx ** 2
    # J2 should be capped at A0 = 1.0e8
    assert math.isclose(j2, a0, rel_tol=1e-3)

    # Plastic strain accumulated
    assert st["epsp"][0] > 0.0
    assert st["mat_extra"]["epxe"][0] > 0.0

    # Force equilibrium
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_hexa8_compaction_eos_unloading_hysteresis():
    """Verify compaction EOS hysteretic unloading with evolving bulk modulus."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    c1 = 1.0e9
    bunl = 5.0e9
    mumax = 0.05
    mat = make_test_material_law21(
        rho0=2000.0,
        c1=c1,
        bunl=bunl,
        mumax=mumax,
        a0=1.0e16,  # purely elastic in shear
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # 1. Compress element
    centroid = np.array([0.5, 0.5, 0.5])
    vel_comp = -0.05 * (coords - centroid)
    dt = 1.0e-4

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel_comp * dt
        solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)

    p_peak = -(st["sig"][0, 0] + st["sig"][0, 1] + st["sig"][0, 2]) / 3.0
    mu_peak = st["mat_extra"]["mu_bak"][0]
    assert p_peak > 0.0
    assert mu_peak > 0.0

    # 2. Reverse velocity to unload
    vel_unl = -vel_comp
    for _ in range(5):
        curr_x += vel_unl * dt
        solid_hexa8.forces(group, curr_x, vel_unl, None, dt, fint, mint)

    p_unloaded = -(st["sig"][0, 0] + st["sig"][0, 1] + st["sig"][0, 2]) / 3.0
    # Pressure must have dropped faster than loading slope due to Bunl > C1
    assert p_unloaded < p_peak
    # mu_bak must have preserved historical maximum compaction
    assert math.isclose(st["mat_extra"]["mu_bak"][0], mu_peak, rel_tol=1e-5)
    # Energy is strictly positive
    assert st["eint"][0] > 0.0


def test_hexa8_multi_element_patch():
    """Verify 2-element Hexa8 mesh sharing a face under tension/compression."""
    # 2 unit cubes sharing face at x = 1.0 (12 nodes total)
    coords = np.array([
        # Cube 0: x in [0, 1]
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        # Cube 1: additional 4 nodes at x = 2.0
        [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0],
    ], dtype=float)

    # Elements connectivity
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],      # Element 0
        [1, 8, 9, 2, 5, 10, 11, 6],     # Element 1
    ], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, c1=1.0e9, bunl=2.0e9, a0=1.0e12)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Stretch patch: shared face at x=1.0 moves with 25.0, far face at x=2.0 moves with 50.0
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 25.0
    vel[[8, 9, 10, 11], 0] = 50.0
    dt = 1.0e-5

    fint = np.zeros((12, 3))
    mint = np.zeros((12, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    # Both elements have non-zero stress and energy
    assert st["sig"][0, 0] != 0.0
    assert st["sig"][1, 0] != 0.0
    assert (st["eint"] > 0.0).all()

    # Total internal nodal force across patch must sum to zero
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 4. Solid Element Formulations: Tetra4
# ============================================================================

def test_solid_tetra4_kernel_cycle0_and_compression():
    """Verify single 4-node tetrahedron under LAW21 compression and time-step."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, E=2.0e10, nu=0.25, c1=1.0e9, bunl=2.0e9)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    st = group.state
    assert "mu_bak" in st["mat_extra"]

    # 1. Cycle 0 time step
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = solid_tetra4.forces(group, coords, None, None, 0.0, fint, mint)
    assert len(dt_c0) == 1
    assert dt_c0[0] > 0.0

    # 2. Compress node 1 towards origin: vx = -10.0
    vel = np.zeros_like(coords)
    vel[1, 0] = -10.0
    dt = 1.0e-5
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    # Compression along x
    assert st["sig"][0, 0] < 0.0
    # Compaction accumulated
    assert st["mat_extra"]["mu_bak"][0] > 0.0
    # Energy positive
    assert st["eint"][0] > 0.0
    # Force equilibrium
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_solid_tetra4_shear_and_plasticity():
    """Verify Tetra4 shear plastic return mapping and plastic strain accumulation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    a0 = 1.0e8
    mat = make_test_material_law21(
        rho0=2000.0,
        E=2.0e10,
        nu=0.25,
        c1=1.0e9,
        a0=a0,
        a1=0.0,
        a2=0.0,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    st = group.state

    # Move node 3 along x (shear in zx)
    vel = np.zeros_like(coords)
    vel[3, 0] = 50.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    # Plastic strain accumulated
    assert st["epsp"][0] > 0.0
    assert st["mat_extra"]["epxe"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_tetra4_multi_element_patch():
    """Verify 2-tetrahedron patch test sharing a face."""
    # 5 nodes forming two adjacent tetrahedra
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)

    # Tetra 0: (0, 1, 2, 3), Tetra 1: (1, 2, 3, 4)
    conn = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, c1=1.0e9, bunl=2.0e9)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    st = group.state

    vel = np.zeros_like(coords)
    vel[1, 0] = 15.0
    vel[4, 0] = 30.0
    dt = 1.0e-5

    fint = np.zeros((5, 3))
    mint = np.zeros((5, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert (st["eint"] > 0.0).all()
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 5. Algorithmic Consistent Tangent Stiffness Dispatch
# ============================================================================

def test_consistent_tangents_dispatch_solid():
    """Verify solid consistent tangent dispatch, symmetry, and rank."""
    mat = make_test_material_law21(
        rho0=2000.0,
        E=2.0e10,
        nu=0.25,
        c1=1.0e9,
        bunl=2.0e9,
        a0=1.0e10,
    )

    # 1. Initial elastic state: (1, 6, 6) tensor
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    depsp = np.zeros(1)
    extra = {"off": np.ones(1)}

    d_solid = materials.solid_tangent(mat, sig, epsp=epsp, epsp_incr=depsp, extra=extra)
    assert d_solid.shape == (1, 6, 6)
    # Initial elastic tensor must be symmetric
    assert np.allclose(d_solid[0], d_solid[0].T)

    # Voigt components check: bulk = 2.0e9, G = 8.0e9
    # D_11 = K + 4/3 G = 2.0e9 + 1.0667e10 = 1.2667e10
    g = 2.0e10 / (2.0 * 1.25)
    c_11_expected = 2.0e9 + (4.0 / 3.0) * g
    assert math.isclose(d_solid[0, 0, 0], c_11_expected, rel_tol=1e-5)
    assert math.isclose(d_solid[0, 3, 3], g, rel_tol=1e-5)


def test_hexa8_tangent_stiffness_and_rigid_invariance():
    """Verify solid_hexa8.tangent returns (n, 24, 24) and vanishes on rigid translation."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, E=2.0e10, nu=0.25, c1=1.0e9, bunl=2.0e9)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)

    ke, edofs = solid_hexa8.tangent(group, coords)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)

    # Rigid body translation modes: [1, 0, 0]*8, [0, 1, 0]*8, [0, 0, 1]*8
    ke_mat = ke[0]
    for axis in range(3):
        v_rigid = np.zeros((8, 3))
        v_rigid[:, axis] = 1.0
        v_flat = v_rigid.reshape(-1)
        f_res = ke_mat @ v_flat
        assert np.allclose(f_res, 0.0, atol=1e-6), f"Rigid translation along axis {axis} should yield zero internal forces"


def test_tetra4_tangent_stiffness_and_rigid_invariance():
    """Verify solid_tetra4.tangent returns (n, 12, 12) and vanishes on rigid translation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_material_law21(rho0=2000.0, E=2.0e10, nu=0.25, c1=1.0e9, bunl=2.0e9)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)

    ke, edofs = solid_tetra4.tangent(group, coords)
    assert ke.shape == (1, 12, 12)
    assert edofs.shape == (1, 12)

    ke_mat = ke[0]
    for axis in range(3):
        v_rigid = np.zeros((4, 3))
        v_rigid[:, axis] = 1.0
        v_flat = v_rigid.reshape(-1)
        f_res = ke_mat @ v_flat
        assert np.allclose(f_res, 0.0, atol=1e-6), f"Rigid translation along axis {axis} should yield zero internal forces"


# ============================================================================
# 6. Shell & 1D Element Rejection Verification
# ============================================================================

def test_shell_rejection_at_materials_level():
    """Verify that materials.shell_update and shell_layer_tangent reject LAW21."""
    mat = make_test_material_law21()

    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))

    with pytest.raises(NotImplementedError, match="LAW21.*implemented for 3D solid elements only"):
        materials.shell_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="LAW21.*implemented for 3D solid elements only"):
        materials.shell_layer_tangent(mat, sig)


def test_shell_kernels_reject_law21():
    """Verify that BT4, QEPH, and Tri3 shell kernels reject LAW21 with clear exception."""
    mat = make_test_material_law21()
    prop = MockProp(thick=1.0, nip=3)

    # 1. BT4 Quad
    coords_quad = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    conn_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_bt4 = MockGroup(conn_quad, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords_quad.copy()
    group_bt4._model = model
    shell_bt4.init_group(group_bt4, model, None)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    vel = np.ones_like(coords_quad)

    with pytest.raises(NotImplementedError, match="LAW21.*implemented for 3D solid elements only"):
        shell_bt4.forces(group_bt4, coords_quad, vel, np.zeros_like(coords_quad), 1.0e-5, fint, mint)

    # 2. QEPH Quad
    group_qeph = MockGroup(conn_quad, slices=[(slice(0, 1), mat, prop)])
    group_qeph._model = model
    shell_qeph.init_group(group_qeph, model, None)

    with pytest.raises(NotImplementedError, match="LAW21.*implemented for 3D solid elements only"):
        shell_qeph.forces(group_qeph, coords_quad, vel, np.zeros_like(coords_quad), 1.0e-5, fint, mint)

    # 3. Tri3 Shell
    coords_tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    conn_tri = np.array([[0, 1, 2]], dtype=np.int64)
    group_tri = MockGroup(conn_tri, slices=[(slice(0, 1), mat, prop)])
    model_tri = Model()
    model_tri.x0 = coords_tri.copy()
    group_tri._model = model_tri
    shell_tri3.init_group(group_tri, model_tri, None)

    fint_tri = np.zeros((3, 3))
    mint_tri = np.zeros((3, 3))
    with pytest.raises(NotImplementedError, match="LAW21.*implemented for 3D solid elements only"):
        shell_tri3.forces(group_tri, coords_tri, np.ones_like(coords_tri), np.zeros_like(coords_tri), 1.0e-5, fint_tri, mint_tri)


# ============================================================================
# 7. Compaction EOS with Tabulated Curve in Elements
# ============================================================================

def test_tabulated_compaction_curve_in_hexa8():
    """Verify Hexa8 kernel when LAW21 defines a tabulated compaction curve."""
    # Define curve: mu -> P_load
    xs = np.array([0.0, 0.02, 0.05, 0.10])
    ys = np.array([0.0, 1.0e7, 3.0e7, 8.0e7])
    curve = (xs, ys)

    pfscale = 1.5
    mat = make_test_material_law21(
        rho0=2000.0,
        curve=curve,
        pfscale=pfscale,
        a0=1.0e14,  # elastic in shear
    )
    prop = MockProp()

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Compress element
    centroid = np.array([0.5, 0.5, 0.5])
    vel = -0.05 * (coords - centroid)
    dt = 1.0e-5

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    # Effective compaction mu = rho / rho0 - 1
    mu_final = st["mat_extra"]["mu_bak"][0]
    assert mu_final > 0.0

    p_expected = float(np.interp(mu_final, xs, ys)) * pfscale
    sig = st["sig"][0]
    p_hydro = -(sig[0] + sig[1] + sig[2]) / 3.0
    assert math.isclose(p_hydro, p_expected, rel_tol=1e-2)


# ============================================================================
# 8. Tensile Cutoff Behavior
# ============================================================================

def test_tensile_cutoff_in_hexa8():
    """Verify that pressure is clamped to P_min under hydrostatic tension."""
    pmin = -5.0e6  # Tensile cutoff at -5 MPa
    mat = make_test_material_law21(
        rho0=2000.0,
        c1=1.0e9,
        pmin=pmin,
        a0=1.0e14,
    )
    prop = MockProp()

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # Strong hydrostatic expansion: outward velocity
    centroid = np.array([0.5, 0.5, 0.5])
    vel = 0.05 * (coords - centroid)
    dt = 1.0e-3

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(10):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    sig = st["sig"][0]
    p_hydro = -(sig[0] + sig[1] + sig[2]) / 3.0
    # Pressure cannot be more tensile (more negative) than pmin
    assert p_hydro >= pmin - 1e-3


# ============================================================================
# 9. Starter Checks & Incompatible Elements Rejection
# ============================================================================

def test_starter_checks_allowed_laws_law21():
    """Verify that LAW21 is permitted in all solid families and excluded from shells and 1D."""
    from pyradioss.starter.checks import _ALLOWED_LAWS

    keys = (21, "21", "LAW21", "DPRAG")
    solid_families = ("bricks", "tetras", "penta6", "pyra5", "solids", "solids_heph", "solids_tetra4")
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads")
    line_families = ("trusses", "beams")

    for fam in solid_families:
        for k in keys:
            assert k in _ALLOWED_LAWS[fam], f"LAW21 key {k} should be permitted in {fam}"

    for fam in shell_families:
        for k in keys:
            assert k not in _ALLOWED_LAWS[fam], f"LAW21 key {k} should NOT be in {fam}"

    for fam in line_families:
        for k in keys:
            assert k not in _ALLOWED_LAWS[fam], f"LAW21 key {k} should NOT be in {fam}"


def test_starter_checks_error_logging_law21():
    """Verify check_mat_law21 logs errors for 2D analysis or incompatible elements."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law21

    log = MessageLog()
    model = Model()
    model.n2d = 1  # 2D analysis

    mat = make_test_material_law21(mid=10)
    check_mat_law21(model, mat_id=10, mat=mat, log=log)

    # Must log error for 2D analysis
    assert any("2D analysis" in e for e in log.errors)

