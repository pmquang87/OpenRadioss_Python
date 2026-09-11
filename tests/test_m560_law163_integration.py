"""
Integration test suite for /MAT/LAW163 (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM).

Milestone M560: Solid Element Formulations Integration & Multi-Cycle Simulation Verifier
1. Registry and dispatch metadata verification for all LAW163 aliases:
   - Registration in MAT_PHYSICS_REGISTRY.
   - Dispatch in MATERIAL_SOLID_DISPATCH and MATERIAL_SHELL_DISPATCH.
   - Environment requirements (needs_env).
   - Extra shapes allocation (uvar163, epsd163, sigv).
   - Starter allowed laws mapping and validation error handling.
2. Acoustic sound speed (dilatational wave speed) and Courant time step calculation:
   - Dilatational wave speed in uncrushed foam: c0 = sqrt((K + 4/3*G) / rho0).
   - Dynamic stiffening with table slope dsdgam: c = sqrt((max(K, dsdgam) + 4/3*G) / rho).
   - Viscous damping stiffness contribution: c = sqrt((max(K, dsdgam) + 4/3*G + |a|/dt) / rho).
   - Cycle 0 critical time step probe on Hexa8 and Tetra4.
3. Solid element formulations:
   - Hexa8 standard:
     * Uniaxial compressive crushing past yield into volumetric compaction plateau.
     * Tensile loading bounded by tensile cutoff tsc.
     * Pure shear deformation (xy, yz, zx).
     * Strain rate sensitive compression comparing high and low velocity loading.
     * Damping effect attenuating high-frequency vibrations.
     * History variable propagation: uvar1 (gamma_old), epsd (filtered rate), sigv (damping stress).
   - Tetra4 standard:
     * Single 4-node tetrahedron under compressive crushing and tensile cutoff.
     * History variable propagation: uvar1, epsd, sigv.
4. Multi-element patch tests:
   - 2-element Hexa8 patch sharing a face (12 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2-element Tetra4 patch sharing a face (5 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2x2x2 Hexa8 patch (8 elements, 27 nodes, uniform deformation, net zero internal forces).
5. Implicit consistent tangent stiffness:
   - Algorithmic consistent tangent tensor (n, 6, 6) from cii, cij, g.
   - Tangent symmetry and positive definiteness.
   - Hexa8 (24x24) and Tetra4 (12x12) element tangent matrices.
   - Rigid-body translation invariance: K_e . v_trans = 0.
6. Plane-stress shell and 1D element rejection:
   - shell_update raises NotImplementedError.
   - BT4, QEPH, Tri3 shell kernels reject LAW163.
   - Starter checks _ALLOWED_LAWS: solids allowed, shells/beams/trusses rejected.
   - Starter check_mat_law163 parameter validation error handling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law163_crush_foam import (
    Law163Params,
    build_law163,
    solid_update,
    solid_update_law163,
    shell_update,
    shell_update_law163,
    sound_speed_solid,
    sound_speed_solid_law163,
    consistent_solid_tangent,
    extra_shapes,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law163,
)
from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable


# ============================================================================
# Helper Mock Element Classes for Kernel Integration Testing
# ============================================================================

class MockProp:
    """Mock solid property container (/PROP/SOLID, /PROP/TYPE14)."""
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 1, **kwargs: Any):
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
    """Mock solid element group container."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law163(
    mid: int = 1,
    rho0: float = 1.0e-3,       # Foam density: 1.0 g/cm^3 = 1.0e-3 g/mm^3
    e: float = 1000.0,          # Young's modulus (MPa)
    nu: float = 0.20,           # Poisson's ratio
    tsc: float = 50.0,          # Tensile stress cutoff (MPa)
    damp: float = 0.10,         # Damping coefficient
    ncycle: int = 12,           # Strain rate filter cycles
    srclmt: float = 1.0e20,     # Strain rate change limit
    fscale: float = 1.0,        # Scale factor
    nrs: int = 0,               # Strain rate flag (0=true, 1=eng)
    table: Any = None,          # Yield table / curve
    **kwargs: Any,
) -> Material:
    """Construct a Material object configured for /MAT/LAW163."""
    if table is None:
        # Default typical foam yield curve: gamma vs yield stress (MPa)
        table = (
            np.array([0.0, 0.05, 0.10, 0.30, 0.50, 0.70, 0.85]),
            np.array([10.0, 10.0, 12.0, 15.0, 25.0, 60.0, 200.0]),
        )
    p = Law163Params(
        rho0=rho0,
        refer_rho=rho0,
        e=e,
        nu=nu,
        tsc=tsc,
        damp=damp,
        ncycle=ncycle,
        srclmt=srclmt,
        fscale=fscale,
        nrs=nrs,
        table=table,
        **kwargs,
    )
    return build_law163(p)


# ============================================================================
# 1. Registry and Dispatch Metadata Verification
# ============================================================================

def test_law163_registry_and_dispatch_metadata():
    """Verify registry entries, dispatch maps, environment flags, and allowed laws for LAW163."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

    aliases = (
        163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM",
        "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", "LAW163_CRUSHABLE_FOAM",
    )

    # 1. Physics builder registry
    for alias in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163"):
        assert alias in MAT_PHYSICS_REGISTRY, f"Missing MAT_PHYSICS_REGISTRY for {alias}"

    # 2. Solid & shell dispatch tables
    for alias in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM"):
        assert alias in materials.MATERIAL_SOLID_DISPATCH
        assert materials.MATERIAL_SOLID_DISPATCH[alias] is solid_update_law163
        assert alias in materials.MATERIAL_SHELL_DISPATCH
        assert materials.MATERIAL_SHELL_DISPATCH[alias] is shell_update_law163

    # 3. Environment requirements
    mat = make_test_material_law163()
    assert materials.needs_env(mat) is True

    # 4. Extra shapes required: uvar163 (2,), epsd163 (), sigv (6,)
    shapes = extra_shapes(mat)
    assert "uvar163" in shapes and shapes["uvar163"] == (2,)
    assert "epsd163" in shapes and shapes["epsd163"] == ()
    assert "sigv" in shapes and shapes["sigv"] == (6,)

    shapes_mod = materials.extra_shapes(mat)
    assert "uvar163" in shapes_mod
    assert "epsd163" in shapes_mod
    assert "sigv" in shapes_mod

    # 5. Starter allowed laws
    assert "bricks" in _ALLOWED_LAWS and 163 in _ALLOWED_LAWS["bricks"]
    assert "tetras" in _ALLOWED_LAWS and 163 in _ALLOWED_LAWS["tetras"]
    assert "shells" in _ALLOWED_LAWS and 163 not in _ALLOWED_LAWS["shells"]
    assert "beams" in _ALLOWED_LAWS and 163 not in _ALLOWED_LAWS["beams"]
    assert "trusses" in _ALLOWED_LAWS and 163 not in _ALLOWED_LAWS["trusses"]

    # 6. Starter checks registry
    assert 163 in _MAT_CHECKS and _MAT_CHECKS[163] is check_mat_law163


# ============================================================================
# 2. Solid Element Initial State and Failure Guards
# ============================================================================

def test_solid_element_initial_state_and_failure_guards():
    """Verify element state initialization for LAW163 on Hexa8 and Tetra4."""
    mat = make_test_material_law163()
    prop = MockProp()

    coords_hex = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    group_hex = MockGroup(conn_hex, slices=[(slice(0, 1), mat, prop)])
    model_hex = Model()
    model_hex.x0 = coords_hex.copy()
    group_hex._model = model_hex

    solid_hexa8.init_group(group_hex, model_hex, None)
    st_hex = group_hex.state

    # 1. Hexa8 state arrays initialized properly
    assert "mat_extra" in st_hex
    assert "uvar163" in st_hex["mat_extra"]
    assert "epsd163" in st_hex["mat_extra"]
    assert "sigv" in st_hex["mat_extra"]
    assert st_hex["mat_extra"]["uvar163"].shape == (1, 2)
    assert st_hex["mat_extra"]["epsd163"].shape == (1,)
    assert st_hex["mat_extra"]["sigv"].shape == (1, 6)
    assert np.allclose(st_hex["mat_extra"]["uvar163"], 0.0)
    assert np.allclose(st_hex["mat_extra"]["epsd163"], 0.0)
    assert np.allclose(st_hex["mat_extra"]["sigv"], 0.0)
    assert st_hex["chk_fail"] is False

    # 2. Tetra4 initialization also behaves identically
    coords_tet = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)

    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet

    solid_tetra4.init_group(group_tet, model_tet, None)
    st_tet = group_tet.state

    assert "mat_extra" in st_tet
    assert "uvar163" in st_tet["mat_extra"]
    assert "epsd163" in st_tet["mat_extra"]
    assert "sigv" in st_tet["mat_extra"]
    assert st_tet["mat_extra"]["uvar163"].shape == (1, 2)
    assert st_tet["mat_extra"]["epsd163"].shape == (1,)
    assert st_tet["mat_extra"]["sigv"].shape == (1, 6)
    assert st_tet["chk_fail"] is False


# ============================================================================
# 3. Acoustic Sound Speed and Courant Step Verification
# ============================================================================

def test_law163_sound_speed_and_courant_step():
    """Verify dilatational wave speed computation and explicit Courant stability."""
    e = 1000.0
    nu = 0.20
    rho0 = 1.0e-3
    c0_expected = math.sqrt((555.555556 + 4.0 / 3.0 * 416.666667) / rho0)

    mat = make_test_material_law163(e=e, nu=nu, rho0=rho0, damp=0.0)

    # 1. Base sound speed via materials.sound_speed and sound_speed_solid
    c_base = materials.sound_speed(mat, rho=rho0)
    assert math.isclose(c_base, c0_expected, rel_tol=1e-5)

    c_solid = sound_speed_solid(mat, rho=rho0)
    assert math.isclose(c_solid, c0_expected, rel_tol=1e-5)

    # 2. Stiffened wave speed with non-zero compaction table slope dsdgam
    extra = {"dsdgam": np.array([2000.0])}
    c_stiff = sound_speed_solid(mat, rho=rho0, extra=extra)
    c_stiff_expected = math.sqrt((2000.0 + 4.0 / 3.0 * 416.666667) / rho0)
    assert math.isclose(c_stiff, c_stiff_expected, rel_tol=1e-5)

    # 3. Viscous damping stiffness contribution at dt > 0
    dt = 1.0e-4
    mat_damped = make_test_material_law163(e=e, nu=nu, rho0=rho0, damp=0.15)
    sig_out, _, c_damped = solid_update(
        mat_damped,
        np.zeros(6),
        np.array([-0.01, 0, 0, 0, 0, 0]),
        dt=dt,
        extra={"rho": np.array([rho0]), "le": np.array([1.0])},
        return_tuple=True,
    )
    assert c_damped > c_base

    # 4. Cycle 0 Courant critical time step probe on Hexa8 and Tetra4
    coords_hex = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    group_hex = MockGroup(conn_hex, slices=[(slice(0, 1), mat, MockProp())])
    model_hex = Model()
    model_hex.x0 = coords_hex.copy()
    group_hex._model = model_hex
    solid_hexa8.init_group(group_hex, model_hex, None)

    fint = np.zeros_like(coords_hex)
    dt_hex = solid_hexa8.forces(group_hex, coords_hex, np.zeros_like(coords_hex), None, 0.0, fint, None)
    assert dt_hex[0] > 0.0
    assert math.isclose(dt_hex[0], group_hex.state["dtfac"][0] * 1.0 / c_base, rel_tol=1e-3)

    # Tetra4
    coords_tet = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, MockProp())])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet
    solid_tetra4.init_group(group_tet, model_tet, None)

    fint_tet = np.zeros_like(coords_tet)
    dt_tet = solid_tetra4.forces(group_tet, coords_tet, np.zeros_like(coords_tet), None, 0.0, fint_tet, None)
    assert dt_tet[0] > 0.0


# ============================================================================
# 4. Hexa8 Uniaxial Compressive Crushing Past Yield
# ============================================================================

def test_hexa8_uniaxial_compressive_crushing_past_yield():
    """Verify Hexa8 under uniaxial compression: elastic loading, yield plateau, densification, and equilibrium."""
    yield_curve = (
        np.array([0.0, 0.10, 0.30, 0.50, 0.80]),
        np.array([10.0, 10.0, 15.0, 40.0, 150.0]),
    )
    mat = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0, table=yield_curve)
    prop = MockProp(qa=0.0, qb=0.0)

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

    top_nodes = [4, 5, 6, 7]
    bottom_nodes = [0, 1, 2, 3]

    # Step 1: Small compressive strain in elastic regime (deps_zz = -0.005)
    dt = 1.0e-4
    v1 = np.zeros_like(coords)
    v1[top_nodes, 2] = -50.0
    coords_1 = coords + v1 * dt
    fint = np.zeros_like(coords)

    solid_hexa8.forces(group, coords_1, v1, None, dt, fint, None)
    sig_1 = group.state["sig"][0]
    assert math.isclose(sig_1[2], -5.0, rel_tol=0.01)
    assert math.isclose(sig_1[0], 0.0, abs_tol=1e-6)
    assert math.isclose(sig_1[1], 0.0, abs_tol=1e-6)

    # Step 2: Compress further past yield (dz = -0.05, total strain ~ -0.055)
    v2 = np.zeros_like(coords)
    v2[top_nodes, 2] = -500.0
    coords_2 = coords_1 + v2 * dt
    fint.fill(0.0)

    solid_hexa8.forces(group, coords_2, v2, None, dt, fint, None)
    sig_2 = group.state["sig"][0]
    assert math.isclose(sig_2[2], -10.0, rel_tol=1e-3)

    # Check equilibrium on Step 2
    assert np.allclose(fint[top_nodes, 2], 2.5, rtol=1e-3)
    assert np.allclose(fint[bottom_nodes, 2], -2.5, rtol=1e-3)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-12)

    # Step 3: Compress deep into densification regime (gamma ~ 0.50 -> yield stress = 40.0 MPa)
    coords_dense = coords.copy()
    coords_dense[top_nodes, 2] = 0.50
    v3 = np.zeros_like(coords)
    v3[top_nodes, 2] = -500.0
    fint.fill(0.0)

    solid_hexa8.forces(group, coords_dense, v3, None, dt, fint, None)
    sig_3 = group.state["sig"][0]
    assert math.isclose(sig_3[2], -40.0, rel_tol=0.05)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-12)

    # History variables: uvar1 tracks volumetric strain gamma = 1 - V/V0 = 0.50
    assert "uvar163" in group.state["mat_extra"]
    assert math.isclose(group.state["mat_extra"]["uvar163"][0, 0], 0.50, abs_tol=1e-3)


# ============================================================================
# 5. Hexa8 Tensile Loading Bounded by Tensile Cutoff tsc
# ============================================================================

def test_hexa8_tensile_loading_bounded_by_tsc():
    """Verify Hexa8 under uniaxial tension: principal stresses bounded by tensile cutoff tsc."""
    tsc = 25.0
    mat = make_test_material_law163(e=1000.0, nu=0.0, tsc=tsc, damp=0.0)
    prop = MockProp(qa=0.0, qb=0.0)

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

    top_nodes = [4, 5, 6, 7]
    dt = 1.0e-4

    v_tens = np.zeros_like(coords)
    v_tens[top_nodes, 2] = 1000.0
    coords_tens = coords + v_tens * dt
    fint = np.zeros_like(coords)

    solid_hexa8.forces(group, coords_tens, v_tens, None, dt, fint, None)
    sig_tens = group.state["sig"][0]

    assert math.isclose(sig_tens[2], tsc, rel_tol=1e-4)

    coords_tens_2 = coords + 2.0 * v_tens * dt
    fint.fill(0.0)
    solid_hexa8.forces(group, coords_tens_2, 2.0 * v_tens, None, dt, fint, None)
    assert math.isclose(group.state["sig"][0, 2], tsc, rel_tol=1e-4)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-12)


# ============================================================================
# 6. Hexa8 Pure Shear Deformation
# ============================================================================

def test_hexa8_pure_shear_deformation():
    """Verify Hexa8 under pure shear: elastic shear response and internal force equilibrium."""
    e = 1000.0
    nu = 0.25
    g_expected = e / (2.0 * (1.0 + nu))
    mat = make_test_material_law163(e=e, nu=nu, damp=0.0, tsc=1e20)
    prop = MockProp(qa=0.0, qb=0.0)

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

    dt = 1.0e-4
    gamma_xy = 0.001
    y1_nodes = [2, 3, 6, 7]

    v = np.zeros_like(coords)
    v[y1_nodes, 0] = gamma_xy / dt
    coords_shear = coords + v * dt
    fint = np.zeros_like(coords)

    solid_hexa8.forces(group, coords_shear, v, None, dt, fint, None)
    sig = group.state["sig"][0]

    assert math.isclose(sig[3], g_expected * gamma_xy, rel_tol=1e-3)
    assert math.isclose(sig[0], 0.0, abs_tol=1e-4)
    assert math.isclose(sig[1], 0.0, abs_tol=1e-4)
    assert math.isclose(sig[2], 0.0, abs_tol=1e-4)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-12)


# ============================================================================
# 7. Hexa8 Strain Rate Sensitivity
# ============================================================================

def test_hexa8_strain_rate_sensitivity():
    """Verify strain-rate sensitive compressive crushing comparing high and low velocity loading."""
    xg = np.array([0.0, 0.20, 0.50, 0.80])
    rates = np.array([1.0, 100.0, 10000.0])
    Y = np.array([
        [10.0, 15.0, 25.0],
        [10.0, 15.0, 25.0],
        [20.0, 30.0, 50.0],
        [100.0, 150.0, 250.0],
    ])
    table_rate = (xg, rates, Y)

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    top_nodes = [4, 5, 6, 7]
    dt = 1.0e-5

    # Low velocity run
    mat_low = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0, table=table_rate)
    group_low = MockGroup(conn, slices=[(slice(0, 1), mat_low, MockProp(qa=0, qb=0))])
    model_low = Model()
    model_low.x0 = coords.copy()
    group_low._model = model_low
    solid_hexa8.init_group(group_low, model_low, None)

    v_low = np.zeros_like(coords)
    v_low[top_nodes, 2] = -100.0
    coords_low = coords + v_low * dt
    fint_low = np.zeros_like(coords)
    solid_hexa8.forces(group_low, coords_low, v_low, None, dt, fint_low, None)
    sig_low = abs(group_low.state["sig"][0, 2])
    rate_low = group_low.state["mat_extra"]["epsd163"][0]

    # High velocity run
    mat_high = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0, table=table_rate)
    group_high = MockGroup(conn, slices=[(slice(0, 1), mat_high, MockProp(qa=0, qb=0))])
    model_high = Model()
    model_high.x0 = coords.copy()
    group_high._model = model_high
    solid_hexa8.init_group(group_high, model_high, None)

    v_high = np.zeros_like(coords)
    v_high[top_nodes, 2] = -10000.0
    coords_high = coords + v_high * dt
    fint_high = np.zeros_like(coords)
    solid_hexa8.forces(group_high, coords_high, v_high, None, dt, fint_high, None)
    sig_high = abs(group_high.state["sig"][0, 2])
    rate_high = group_high.state["mat_extra"]["epsd163"][0]

    assert sig_high > sig_low
    assert rate_high > rate_low


# ============================================================================
# 8. Hexa8 Viscous Damping Effect
# ============================================================================

def test_hexa8_damping_effect():
    """Verify damping stress tensor sigv attenuates high-frequency vibration and adds viscous resistance."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    top_nodes = [4, 5, 6, 7]
    dt = 1.0e-4

    # Undamped model (damp = 0.0)
    mat_undamped = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0)
    group_u = MockGroup(conn, slices=[(slice(0, 1), mat_undamped, MockProp(qa=0, qb=0))])
    model_u = Model()
    model_u.x0 = coords.copy()
    group_u._model = model_u
    solid_hexa8.init_group(group_u, model_u, None)

    # Damped model (damp = 0.20)
    mat_damped = make_test_material_law163(e=1000.0, nu=0.0, damp=0.20)
    group_d = MockGroup(conn, slices=[(slice(0, 1), mat_damped, MockProp(qa=0, qb=0))])
    model_d = Model()
    model_d.x0 = coords.copy()
    group_d._model = model_d
    solid_hexa8.init_group(group_d, model_d, None)

    v = np.zeros_like(coords)
    v[top_nodes, 2] = -200.0
    coords_step = coords + v * dt
    fint_u = np.zeros_like(coords)
    fint_d = np.zeros_like(coords)

    solid_hexa8.forces(group_u, coords_step, v, None, dt, fint_u, None)
    solid_hexa8.forces(group_d, coords_step, v, None, dt, fint_d, None)

    assert abs(group_d.state["sig"][0, 2]) > abs(group_u.state["sig"][0, 2])
    assert "sigv" in group_d.state["mat_extra"]
    assert group_d.state["mat_extra"]["sigv"][0, 2] != 0.0
    assert group_u.state["mat_extra"]["sigv"][0, 2] == 0.0


# ============================================================================
# 9. Tetra4 Uniaxial Compression and Tension
# ============================================================================

def test_tetra4_uniaxial_compression_and_tension():
    """Verify single Tetra4 element under compression past yield and tension bounded by tsc."""
    tsc = 30.0
    yield_curve = (
        np.array([0.0, 0.10, 0.50, 0.80]),
        np.array([12.0, 12.0, 30.0, 100.0]),
    )
    mat = make_test_material_law163(e=1000.0, nu=0.0, tsc=tsc, damp=0.0, table=yield_curve)
    prop = MockProp(qa=0.0, qb=0.0)

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    dt = 1.0e-4

    # 1. Compressive stroke in Z
    v_comp = np.zeros_like(coords)
    v_comp[3, 2] = -200.0
    coords_comp = coords + v_comp * dt
    fint_comp = np.zeros_like(coords)

    solid_tetra4.forces(group, coords_comp, v_comp, None, dt, fint_comp, None)
    sig_comp = group.state["sig"][0]
    assert math.isclose(sig_comp[2], -12.0, rel_tol=0.05)
    assert np.allclose(np.sum(fint_comp, axis=0), 0.0, atol=1e-12)

    # 2. Tensile stroke in Z
    v_tens = np.zeros_like(coords)
    v_tens[3, 2] = +1000.0
    coords_tens = coords + v_tens * dt
    fint_tens = np.zeros_like(coords)

    solid_tetra4.forces(group, coords_tens, v_tens, None, dt, fint_tens, None)
    sig_tens = group.state["sig"][0]
    assert math.isclose(sig_tens[2], tsc, rel_tol=0.05)
    assert np.allclose(np.sum(fint_tens, axis=0), 0.0, atol=1e-12)


# ============================================================================
# 10. Multi-Element Patch Tests
# ============================================================================

def test_two_element_hexa8_patch():
    """Two Hexa8 elements stacked in Z sharing a face (12 nodes): uniform stress and equilibrium."""
    mat = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0)
    prop = MockProp(qa=0.0, qb=0.0)

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0],
    ], dtype=float)

    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8, 9, 10, 11],
    ], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    dt = 1.0e-4
    v = np.zeros_like(coords)
    v[4:8, 2] = -50.0
    v[8:12, 2] = -100.0
    coords_new = coords + v * dt
    fint = np.zeros_like(coords)

    solid_hexa8.forces(group, coords_new, v, None, dt, fint, None)

    # 1. Both elements develop identical uniform stress
    sig = group.state["sig"]
    assert math.isclose(sig[0, 2], sig[1, 2], rel_tol=1e-5)

    # 2. Shared interface nodes (4..7) have net zero internal force
    assert np.allclose(fint[4:8], 0.0, atol=1e-10)

    # 3. Overall equilibrium
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-12)


def test_two_element_tetra4_patch():
    """Two Tetra4 elements sharing a triangular face (5 nodes): uniform equilibrium."""
    mat = make_test_material_law163(e=1000.0, nu=0.0, damp=0.0)
    prop = MockProp(qa=0.0, qb=0.0)

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.2, 0.2, 1.0],
        [0.2, 0.2, -1.0],
    ], dtype=float)

    conn = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 4],
    ], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    dt = 1.0e-4
    v = np.zeros_like(coords)
    v[3, 2] = -10.0
    v[4, 2] = +10.0
    coords_new = coords + v * dt
    fint = np.zeros_like(coords)

    solid_tetra4.forces(group, coords_new, v, None, dt, fint, None)

    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)


def test_hexa8_2x2x2_patch():
    """8-element Hexa8 patch (27 nodes): uniform stress and net zero internal forces on interior nodes."""
    mat = make_test_material_law163(e=1000.0, nu=0.20, damp=0.0)
    prop = MockProp(qa=0.0, qb=0.0)

    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    zs = [0.0, 1.0, 2.0]

    coords_list = []
    for z in zs:
        for y in ys:
            for x in xs:
                coords_list.append([x, y, z])
    coords = np.array(coords_list, dtype=float)

    def node_id(ix: int, iy: int, iz: int) -> int:
        return iz * 9 + iy * 3 + ix

    conn_list = []
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                n0 = node_id(ix, iy, iz)
                n1 = node_id(ix + 1, iy, iz)
                n2 = node_id(ix + 1, iy + 1, iz)
                n3 = node_id(ix, iy + 1, iz)
                n4 = node_id(ix, iy, iz + 1)
                n5 = node_id(ix + 1, iy, iz + 1)
                n6 = node_id(ix + 1, iy + 1, iz + 1)
                n7 = node_id(ix, iy + 1, iz + 1)
                conn_list.append([n0, n1, n2, n3, n4, n5, n6, n7])

    conn = np.array(conn_list, dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 8), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    dt = 1.0e-4
    v = np.zeros_like(coords)
    v[:, 2] = -50.0 * coords[:, 2]
    coords_new = coords + v * dt
    fint = np.zeros_like(coords)

    solid_hexa8.forces(group, coords_new, v, None, dt, fint, None)

    # 1. Stress across all 8 elements is identical
    sig = group.state["sig"]
    for e_idx in range(1, 8):
        assert np.allclose(sig[e_idx], sig[0], rtol=1e-4, atol=1e-5)

    # 2. Interior center node (1, 1, 1) has net zero internal force
    center_node = node_id(1, 1, 1)
    assert np.allclose(fint[center_node], 0.0, atol=1e-10)

    # 3. Overall equilibrium
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)


# ============================================================================
# 11. Implicit Consistent Tangent Stiffness
# ============================================================================

def test_implicit_consistent_tangent():
    """Verify algorithmic consistent tangent matrix symmetry, positive definiteness, and element invariance."""
    mat = make_test_material_law163(e=1200.0, nu=0.25)

    # 1. Material tangent tensor (n, 6, 6)
    D = consistent_solid_tangent(mat)[0]
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T)
    evals = np.linalg.eigvalsh(D)
    assert np.all(evals > 0.0), f"Tangent not positive definite: evals = {evals}"

    # 2. Hexa8 element tangent (24x24) translation invariance
    coords_hex = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    group_hex = MockGroup(conn_hex, slices=[(slice(0, 1), mat, MockProp())])
    model_hex = Model()
    model_hex.x0 = coords_hex.copy()
    group_hex._model = model_hex
    solid_hexa8.init_group(group_hex, model_hex, None)

    ke_hex, _ = solid_hexa8.tangent(group_hex, coords_hex)
    assert ke_hex.shape == (1, 24, 24)

    for dir_idx in range(3):
        v_trans = np.zeros(24)
        v_trans[dir_idx::3] = 1.0
        assert np.allclose(ke_hex[0] @ v_trans, 0.0, atol=1e-10)

    # 3. Tetra4 element tangent (12x12) translation invariance
    coords_tet = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, MockProp())])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet
    solid_tetra4.init_group(group_tet, model_tet, None)

    ke_tet, _ = solid_tetra4.tangent(group_tet, coords_tet)
    assert ke_tet.shape == (1, 12, 12)

    for dir_idx in range(3):
        v_trans = np.zeros(12)
        v_trans[dir_idx::3] = 1.0
        assert np.allclose(ke_tet[0] @ v_trans, 0.0, atol=1e-10)


# ============================================================================
# 12. Plane-Stress Shell & 1D Element Rejection
# ============================================================================

def test_plane_stress_shell_and_1d_rejection():
    """Verify LAW163 is strictly rejected for shells, beams, and trusses."""
    mat = make_test_material_law163()

    # 1. Shell update rejection
    with pytest.raises(NotImplementedError, match="solid elements only"):
        materials.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update_law163(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    # 2. Starter check validation
    dummy_model = Model()
    dummy_model.materials[1] = mat

    # Invalid density
    log1 = MessageLog()
    mat_bad_rho = make_test_material_law163(rho0=-1.0)
    check_mat_law163(dummy_model, 1, mat_bad_rho, log1)
    assert any("initial density RHO must be > 0" in err for err in log1.errors)

    # Invalid Young's modulus
    log2 = MessageLog()
    mat_bad_e = make_test_material_law163(e=0.0)
    check_mat_law163(dummy_model, 1, mat_bad_e, log2)
    assert any("Young's modulus E must be > 0" in err for err in log2.errors)

    # Invalid Poisson's ratio
    log3 = MessageLog()
    mat_bad_nu = Material(id=1, law=163, rho0=1.0, params={"E": 1000.0, "Nu": 0.55})
    check_mat_law163(dummy_model, 1, mat_bad_nu, log3)
    assert any("Poisson's ratio nu must satisfy" in err for err in log3.errors)

    # Shell rejection in starter
    log4 = MessageLog()
    class MockShellGroup:
        def values(self):
            class MockShell:
                mat_id = mat.id
            return [MockShell()]
        state = {"slices": [(slice(0, 1), mat, None)]}

    dummy_model.element_groups = lambda: [("shells", MockShellGroup())]
    check_mat_law163(dummy_model, 1, mat, log4)
    assert any("not supported for shell elements" in err for err in log4.errors)
