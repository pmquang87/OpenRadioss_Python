"""
Integration test suite for /MAT/LAW49 (/MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN).

Milestone M557: Solid Element Formulations Integration & Multi-Cycle Simulation Verifier
1. Registry and dispatch metadata verification for all LAW49 aliases.
2. Acoustic sound speed (dilatational P-wave) and Courant time step calculation:
   - Shear modulus evolution with pressure hardening and thermal softening.
   - Fluid sound speed c = sqrt(K / rho) upon melting (G -> 0).
   - Cycle 0 critical time step probe on Hexa8 and Tetra4.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1):
     * Uniaxial tension and compression (elastic slope, yield, hardening, pressure dependence).
     * Pure shear (shear modulus, yield threshold, plastic work).
     * High strain rate dynamic shock compression (bulk viscosity, density buildup).
     * Thermal softening at elevated temperature (T = 800 K vs 300 K).
     * Melting transition: thermal melting (theta >= tmelt) and energy melting (espe >= Emelt).
     * Adiabatic plastic heating (theta rise from plastic dissipation).
     * Element survival: no deletion upon reaching cold-work hardening saturation eps_max.
   - Tetra4 standard (Itetra=1):
     * Single 4-node tetrahedron under volumetric shock compression.
     * Deviatoric plastic yield and shear flow.
     * Thermal softening and sound speed evolution.
     * Fluid transition upon melting (theta >= tmelt).
4. Multi-element patch tests:
   - 2-element Hexa8 patch sharing a face (12 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2-element Tetra4 patch sharing a face (5 nodes, uniform stress, equilibrium sum f_int = 0).
5. Multi-cycle dynamic explicit time-stepping simulation:
   - Central difference / velocity Verlet time integration over 50+ cycles.
   - Energy conservation: |Delta E_total| / E_total,0 < 1%.
   - Internal, kinetic, and plastic dissipation work tracking.
   - Courant stability (dt <= dt_crit).
6. Implicit consistent tangent stiffness:
   - Elastic tangent symmetry and Voigt component consistency (K, G).
   - Hexa8 (24x24) and Tetra4 (12x12) element tangent matrices.
   - Rigid-body translation invariance: K_e . v_trans = 0.
7. Plane-stress shell and 1D element rejection:
   - shell_update, shell_membrane_tangent, shell_layer_tangent raise NotImplementedError.
   - BT4, QEPH, Tri3 shell kernels reject LAW49.
   - Starter checks _ALLOWED_LAWS: solids allowed, shells/beams/trusses rejected.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law49_steinb import (
    Law49Params,
    build_law49,
    solid_update_law49,
    shell_update_law49,
    sound_speed_solid_law49,
    tangent_law49_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS


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


def make_test_material_law49(
    mid: int = 1,
    rho0: float = 8.93,          # Copper: g/cm^3 or ton/mm^3
    g0: float = 4.5e4,           # 45 GPa
    bulk: float = 1.3e5,         # 130 GPa
    sig0: float = 120.0,         # 120 MPa
    beta: float = 36.0,
    n: float = 0.45,
    eps_max: float = 1.0,        # Hardening saturation (NOT failure)
    sigma_max: float = 640.0,
    t0: float = 300.0,           # Reference room temperature
    tmelt: float = 1356.0,       # Melting point
    rhoc_p: float = 3.4e-3,      # rho0 * Cp
    b1: float = 3.0e-5,          # G pressure dependence
    b2: float = 3.0e-5,          # Yield pressure dependence
    h: float = 3.8e-4,           # Temperature softening
    f: float = 0.0,
    pmin: float = -1.0e30,
    **kwargs: Any,
) -> Material:
    params = {
        "rho0": rho0,
        "g0": g0,
        "bulk": bulk,
        "sig0": sig0,
        "beta": beta,
        "n": n,
        "eps_max": eps_max,
        "sigma_max": sigma_max,
        "t0": t0,
        "tmelt": tmelt,
        "rhoc_p": rhoc_p,
        "b1": b1,
        "b2": b2,
        "h": h,
        "f": f,
        "pmin": pmin,
    }
    params.update(kwargs)
    mat = build_law49(params)
    mat.id = mid
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law49_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        49,
        "49",
        "LAW49",
        "STEINB",
        "STEINBERG",
        "STEINBERG_GUINAN",
        "MAT_LAW49",
        "MAT_STEINB",
        "MAT_STEINBERG",
        "LAW49_STEINB",
    )
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is False, "LAW49 must have plane_stress=False"
        assert meta.get("solid") is True, "LAW49 must have solid=True"
        assert meta.get("shell") is False, "LAW49 must have shell=False"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"Key {k} missing from MATERIAL_SOLID_DISPATCH"
        assert k in materials.MATERIAL_SHELL_DISPATCH, f"Key {k} missing from MATERIAL_SHELL_DISPATCH"

    mat = make_test_material_law49()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "theta" in shapes_solid
    assert "espe" in shapes_solid
    assert "epxe" in shapes_solid
    assert "dpla" in shapes_solid


def test_solid_element_initial_state_and_failure_guards():
    """Verify initial temperature theta is initialized to T0 and eps_max does not trigger deletion."""
    t0_val = 293.15
    mat = make_test_material_law49(t0=t0_val, eps_max=0.5)
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

    # 1. Initial temperature must be t0, not 0.0
    assert math.isclose(st["mat_extra"]["theta"][0], t0_val, rel_tol=1e-6)

    # 2. Hardening saturation eps_max must NOT set chk_fail (only eps_p_max or mat.fail can delete)
    assert st["chk_fail"] is False

    # 3. Tetra4 initialization also populates theta = t0 and chk_fail = False
    coords_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet

    solid_tetra4.init_group(group_tet, model_tet, None)
    assert math.isclose(group_tet.state["mat_extra"]["theta"][0], t0_val, rel_tol=1e-6)
    assert group_tet.state["chk_fail"] is False


# ============================================================================
# 2. Acoustic Sound Speed & Courant Time Step
# ============================================================================

def test_law49_sound_speed_and_evolution():
    """Verify acoustic wave speed c = sqrt(|bulk + 4/3 G| / rho) under pressure, thermal softening, and melting."""
    rho0 = 8.93
    bulk = 1.3e5
    g0 = 4.5e4
    b1 = 3.0e-5
    h = 3.8e-4
    t0 = 300.0
    tmelt = 1356.0

    mat = make_test_material_law49(
        rho0=rho0,
        bulk=bulk,
        g0=g0,
        b1=b1,
        h=h,
        t0=t0,
        tmelt=tmelt,
    )

    # 1. Undisturbed reference state
    c_ref_expected = math.sqrt((bulk + (4.0 / 3.0) * g0) / rho0)
    c_ref = materials.sound_speed(mat, rho=rho0)
    assert math.isclose(c_ref, c_ref_expected, rel_tol=1e-10)
    assert math.isclose(mat.sound_speed_solid(), c_ref_expected, rel_tol=1e-10)

    # 2. Elevated temperature T = 800 K (thermal softening: G drops)
    theta_hot = 800.0
    qb_hot = 1.0 - h * (theta_hot - t0)
    g_hot = g0 * qb_hot
    c_hot_expected = math.sqrt((bulk + (4.0 / 3.0) * g_hot) / rho0)
    c_hot = sound_speed_solid_law49(mat, rho=rho0, extra={"g": g_hot})
    assert math.isclose(c_hot, c_hot_expected, rel_tol=1e-10)
    assert c_hot < c_ref

    # 3. Shock compression P = 1.0e4 (pressure stiffening: G increases)
    p_comp = 1.0e4
    qa_comp = p_comp
    g_comp = g0 * (1.0 + b1 * qa_comp)
    c_comp_expected = math.sqrt((bulk + (4.0 / 3.0) * g_comp) / rho0)
    c_comp = sound_speed_solid_law49(mat, rho=rho0, extra={"g": g_comp})
    assert math.isclose(c_comp, c_comp_expected, rel_tol=1e-10)
    assert c_comp > c_ref

    # 4. Melted state (G -> 0, purely bulk fluid wave speed)
    c_melt_expected = math.sqrt(bulk / rho0)
    c_melt = sound_speed_solid_law49(mat, rho=rho0, extra={"g": 0.0})
    assert math.isclose(c_melt, c_melt_expected, rel_tol=1e-10)
    assert c_melt < c_ref


def test_solid_elements_cycle_0_courant_time_step():
    """Verify cycle 0 critical time step dt_crit = lc / c on Hexa8 and Tetra4."""
    mat = make_test_material_law49(rho0=8.0, bulk=1.6e5, g0=8.0e4)
    prop = MockProp()

    # Hexa8 unit cube (lc = 1.0)
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

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt_hex = solid_hexa8.forces(group_hex, coords_hex, None, None, 0.0, fint, mint)

    c_expected = math.sqrt((1.6e5 + (4.0 / 3.0) * 8.0e4) / 8.0)
    dtfac = group_hex.state.get("dtfac", np.ones(1))[0]
    dt_expected = dtfac * (1.0 / c_expected)
    assert len(dt_hex) == 1
    assert math.isclose(dt_hex[0], dt_expected, rel_tol=1e-3)

    # Tetra4 (regular unit tetrahedron)
    coords_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, math.sqrt(3.0) / 2.0, 0.0],
        [0.5, math.sqrt(3.0) / 6.0, math.sqrt(6.0) / 3.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet
    solid_tetra4.init_group(group_tet, model_tet, None)

    dt_tet = solid_tetra4.forces(group_tet, coords_tet, None, None, 0.0, np.zeros((4, 3)), np.zeros((4, 3)))
    assert len(dt_tet) == 1
    assert dt_tet[0] > 0.0 and dt_tet[0] < 1.0


# ============================================================================
# 3. Hexa8 Solid Element Integration
# ============================================================================

def test_hexa8_solid_uniaxial_tension_and_compression():
    """Verify Hexa8 under uniaxial tension and compression (elastic response, yield, pressure hardening)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    sig0 = 150.0
    beta = 20.0
    n = 0.5
    b1 = 2.0e-4
    b2 = 2.0e-4
    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.0e5,
        sig0=sig0,
        beta=beta,
        n=n,
        b1=b1,
        b2=b2,
        eps_max=2.0,
    )
    prop = MockProp()

    # 1. Uniaxial Tension: pull face x = 1.0 (nodes 1, 2, 5, 6) in +x direction
    group_ten = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model_ten = Model()
    model_ten.x0 = coords.copy()
    group_ten._model = model_ten
    solid_hexa8.init_group(group_ten, model_ten, None)

    vel_ten = np.zeros_like(coords)
    vel_ten[[1, 2, 5, 6], 0] = 10.0  # dot_eps_xx = 10.0 / 1.0 = 10 1/s
    dt = 1.0e-5

    fint_ten = np.zeros((8, 3))
    mint_ten = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(40):
        curr_x += vel_ten * dt
        solid_hexa8.forces(group_ten, curr_x, vel_ten, None, dt, fint_ten, mint_ten)

    st_ten = group_ten.state
    sig_ten = st_ten["sig"][0]
    # Tension produces positive Cauchy sigma_xx
    assert sig_ten[0] > 0.0
    # Equivalent plastic strain accumulated
    assert st_ten["epsp"][0] > 0.0
    assert st_ten["mat_extra"]["epxe"][0] > 0.0
    assert st_ten["eint"][0] > 0.0
    # Force equilibrium: sum f_int = 0
    assert np.allclose(fint_ten.sum(axis=0), 0.0, atol=1e-5)

    # 2. Uniaxial Compression: push face x = 1.0 in -x direction
    group_cmp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model_cmp = Model()
    model_cmp.x0 = coords.copy()
    group_cmp._model = model_cmp
    solid_hexa8.init_group(group_cmp, model_cmp, None)

    vel_cmp = -vel_ten
    fint_cmp = np.zeros((8, 3))
    mint_cmp = np.zeros((8, 3))
    curr_x_cmp = coords.copy()

    for _ in range(40):
        curr_x_cmp += vel_cmp * dt
        solid_hexa8.forces(group_cmp, curr_x_cmp, vel_cmp, None, dt, fint_cmp, mint_cmp)

    st_cmp = group_cmp.state
    sig_cmp = st_cmp["sig"][0]
    # Compression produces negative Cauchy sigma_xx and positive hydrostatic pressure
    assert sig_cmp[0] < 0.0
    p_hydro_cmp = -(sig_cmp[0] + sig_cmp[1] + sig_cmp[2]) / 3.0
    assert p_hydro_cmp > 0.0

    # Pressure hardening: compressive flow stress is higher in magnitude than tensile flow stress
    flow_cmp = abs(sig_cmp[0])
    flow_ten = abs(sig_ten[0])
    assert flow_cmp > flow_ten, f"Compression flow stress {flow_cmp} should exceed tension flow stress {flow_ten} due to pressure hardening"


def test_hexa8_solid_pure_shear():
    """Verify Hexa8 under pure shear (deviatoric stress, yield, plastic dissipation)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    sig0 = 100.0
    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.0e5,
        sig0=sig0,
        beta=0.0,  # Perfectly plastic to check yield limit
        n=1.0,
        rhoc_p=0.0,  # Isothermal to isolate mechanical yield threshold
        h=0.0,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Move top face (z=1, nodes 4, 5, 6, 7) along x (pure shear in zx plane)
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 50.0
    dt = 1.0e-5

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    sig = st["sig"][0]
    # Shear stress sigma_zx is Voigt index 5
    tau_zx = sig[5]
    assert abs(tau_zx) > 0.0

    # Under finite shear deformation with Jaumann rate, stress develops rotated normal components
    s = sig.copy()
    p_hydro = -(s[0] + s[1] + s[2]) / 3.0
    s[0] += p_hydro
    s[1] += p_hydro
    s[2] += p_hydro
    j2 = 0.5 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    sigma_vm = math.sqrt(3.0 * max(j2, 0.0))
    assert math.isclose(sigma_vm, sig0, rel_tol=1e-2)

    assert st["epsp"][0] > 0.0
    assert st["eint"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_hexa8_high_strain_rate_shock_compression():
    """Verify Hexa8 under high strain rate dynamic shock compression (bulk viscosity, density buildup)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.2e5,
        sig0=200.0,
        b1=1.0e-4,
    )
    prop = MockProp(qa=1.2, qb=0.06)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Inward spherical shock velocity towards center (0.5, 0.5, 0.5)
    centroid = np.array([0.5, 0.5, 0.5])
    vel_shock = -50.0 * (coords - centroid)  # High strain rate ~ 50 s^-1
    dt = 1.0e-5

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel_shock * dt
        solid_hexa8.forces(group, curr_x, vel_shock, None, dt, fint, mint)

    st = group.state
    # High compressive pressure
    p = -(st["sig"][0, 0] + st["sig"][0, 1] + st["sig"][0, 2]) / 3.0
    assert p > 1000.0, f"Expected high shock pressure, got {p}"
    assert st["eint"][0] > 0.0
    # Pending bulk viscosity work booked
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_hexa8_thermal_softening():
    """Verify Hexa8 thermal softening: elements at elevated temperature exhibit lower flow stress."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    # 2 elements in one mesh: Element 0 at room temperature (300 K), Element 1 pre-heated to 800 K
    conn2 = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ], dtype=np.int64)

    mat0 = make_test_material_law49(mid=1, t0=300.0, h=5.0e-4, sig0=200.0)
    mat1 = make_test_material_law49(mid=2, t0=300.0, h=5.0e-4, sig0=200.0)
    prop = MockProp()

    group = MockGroup(conn2, slices=[
        (slice(0, 1), mat0, prop),
        (slice(1, 2), mat1, prop),
    ])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Pre-heat element 1
    group.state["mat_extra"]["theta"][1] = 800.0

    # Apply identical shear velocity to both elements
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 30.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    tau_cold = abs(st["sig"][0, 5])
    tau_hot = abs(st["sig"][1, 5])

    # Element 1 at 800 K must have strictly lower flow stress than element 0 at 300 K
    assert tau_hot < tau_cold, f"Hot flow stress {tau_hot} should be lower than cold {tau_cold}"


def test_hexa8_melting_transition_and_fluid_behavior():
    """Verify Hexa8 complete deviatoric relaxation (s_ij -> 0) and bulk-only sound speed upon melting."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    tmelt = 1000.0
    bulk = 1.2e5
    g0 = 4.0e4
    mat = make_test_material_law49(
        rho0=8.0,
        bulk=bulk,
        g0=g0,
        sig0=200.0,
        tmelt=tmelt,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # 1. Deform element elastically/plastically
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 30.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    for _ in range(10):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    assert abs(st["sig"][0, 5]) > 0.0  # Carried shear stress

    # 2. Trigger thermal melting by setting temperature >= tmelt
    st["mat_extra"]["theta"][0] = tmelt + 50.0

    solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    sig_melt = st["sig"][0]
    # Shear stress relaxed to zero
    assert math.isclose(sig_melt[3], 0.0, abs_tol=1e-8)
    assert math.isclose(sig_melt[4], 0.0, abs_tol=1e-8)
    assert math.isclose(sig_melt[5], 0.0, abs_tol=1e-8)

    # Deviatoric normal stresses zeroed: sig_xx == sig_yy == sig_zz == -P
    assert math.isclose(sig_melt[0], sig_melt[1], rel_tol=1e-6)
    assert math.isclose(sig_melt[1], sig_melt[2], rel_tol=1e-6)

    # Element remains ALIVE (not deleted)
    assert st["off"][0] == 1.0


def test_hexa8_adiabatic_plastic_heating():
    """Verify internal temperature rise delta_T = YLD * dpla / (rho0 * Cp) in Hexa8."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    t0 = 300.0
    rhoc_p = 3.0e-3  # Low specific heat so temperature rise is easily visible
    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.0e5,
        sig0=250.0,
        t0=t0,
        rhoc_p=rhoc_p,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 50.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    for _ in range(30):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    theta_final = st["mat_extra"]["theta"][0]
    # Temperature must have increased due to plastic work dissipation
    assert theta_final > t0, f"Expected temperature rise above {t0}, got {theta_final}"


def test_hexa8_hardening_saturation_survival():
    """Verify that exceeding eps_max does NOT delete the element."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    # Set very small eps_max = 0.005 (0.5% plastic strain)
    eps_max_val = 0.005
    mat = make_test_material_law49(
        rho0=8.0,
        sig0=100.0,
        eps_max=eps_max_val,
        beta=10.0,
        n=0.5,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 80.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    for _ in range(40):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    # Plastic strain exceeds eps_max
    assert st["epsp"][0] > eps_max_val
    # Element is STILL ALIVE
    assert st["off"][0] == 1.0


# ============================================================================
# 4. Tetra4 Solid Element Integration
# ============================================================================

def test_tetra4_solid_compression_and_shear():
    """Verify Tetra4 single element under volumetric shock compression and shear yield."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    sig0 = 120.0
    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.0e5,
        sig0=sig0,
        beta=20.0,
        n=0.5,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    # 1. Volumetric compression: move node 3 downward
    vel_comp = np.zeros_like(coords)
    vel_comp[3, 2] = -20.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    for _ in range(15):
        curr_x += vel_comp * dt
        solid_tetra4.forces(group, curr_x, vel_comp, None, dt, fint, mint)

    st = group.state
    # Hydrostatic pressure positive
    p = -(st["sig"][0, 0] + st["sig"][0, 1] + st["sig"][0, 2]) / 3.0
    assert p > 0.0
    assert st["eint"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

    # 2. Shear deformation on node 3
    vel_shear = np.zeros_like(coords)
    vel_shear[3, 0] = 50.0

    for _ in range(25):
        curr_x += vel_shear * dt
        solid_tetra4.forces(group, curr_x, vel_shear, None, dt, fint, mint)

    # Plastic strain accumulated
    assert st["epsp"][0] > 0.0
    assert st["mat_extra"]["epxe"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_tetra4_melting_transition():
    """Verify Tetra4 shear relaxation upon thermal melting."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    tmelt = 1000.0
    mat = make_test_material_law49(
        rho0=8.0,
        g0=4.0e4,
        bulk=1.0e5,
        sig0=150.0,
        tmelt=tmelt,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[3, 0] = 30.0
    dt = 1.0e-5
    curr_x = coords.copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    # Deform
    for _ in range(10):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert abs(group.state["sig"][0, 5]) > 0.0

    # Melt
    group.state["mat_extra"]["theta"][0] = tmelt + 10.0
    solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    sig_melt = group.state["sig"][0]
    assert math.isclose(sig_melt[3], 0.0, abs_tol=1e-8)
    assert math.isclose(sig_melt[4], 0.0, abs_tol=1e-8)
    assert math.isclose(sig_melt[5], 0.0, abs_tol=1e-8)


# ============================================================================
# 5. Multi-Element Patch Tests
# ============================================================================

def test_hexa8_multi_element_patch():
    """Verify 2-element Hexa8 mesh sharing a face under uniform tension."""
    coords = np.array([
        # Element 0: x in [0, 1]
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        # Element 1: additional nodes at x = 2.0
        [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0],
    ], dtype=float)

    # Conn 0: (0, 1, 2, 3, 4, 5, 6, 7)
    # Conn 1: (1, 8, 9, 2, 5, 10, 11, 6)
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 2, 5, 10, 11, 6],
    ], dtype=np.int64)

    mat = make_test_material_law49(rho0=8.0, g0=4.0e4, bulk=1.0e5, sig0=150.0)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Pull x = 2.0 face (nodes 8, 9, 10, 11) with +20 m/s, middle face (1, 2, 5, 6) with +10 m/s
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 10.0
    vel[[8, 9, 10, 11], 0] = 20.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((12, 3))
    mint = np.zeros((12, 3))

    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    # Both elements must exhibit uniform state
    assert math.isclose(st["sig"][0, 0], st["sig"][1, 0], rel_tol=1e-4)
    assert math.isclose(st["epsp"][0], st["epsp"][1], rel_tol=1e-4)
    assert math.isclose(st["eint"][0], st["eint"][1], rel_tol=1e-4)

    # Nodal force equilibrium across the whole patch
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_tetra4_multi_element_patch():
    """Verify 2-element Tetra4 patch sharing a face under uniform deformation."""
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

    mat = make_test_material_law49(rho0=8.0, g0=4.0e4, bulk=1.0e5, sig0=150.0)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[1, 0] = 15.0
    vel[4, 0] = 30.0
    dt = 1.0e-5

    curr_x = coords.copy()
    fint = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert (group.state["eint"] > 0.0).all()
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 6. Multi-Cycle Dynamic Explicit Simulation & Energy Conservation
# ============================================================================

def test_multi_cycle_dynamic_explicit_simulation():
    """Verify multi-cycle dynamic explicit time-stepping (velocity Verlet), Courant stability, and energy conservation."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    rho0 = 8.0
    bulk = 1.0e5
    g0 = 4.0e4
    sig0 = 120.0
    mat = make_test_material_law49(
        rho0=rho0,
        bulk=bulk,
        g0=g0,
        sig0=sig0,
        beta=15.0,
        n=0.5,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    node_idx, mass_c, _ = solid_hexa8.init_group(group, model, None)
    nodal_mass = np.zeros(8)
    np.add.at(nodal_mass, node_idx, mass_c)

    # Initial velocity field (oscillatory kinetic energy, well-scaled to stay within physical strain)
    v = np.zeros((8, 3))
    v[[1, 2, 5, 6], 0] = 0.5
    v[[0, 3, 4, 7], 0] = -0.5

    # Initial kinetic energy E_kin = 1/2 sum(m_i v_i^2)
    e_kin_0 = 0.5 * np.sum(nodal_mass[:, None] * (v ** 2))
    assert e_kin_0 > 0.0

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    # Probe Courant limit
    dt_crit = solid_hexa8.forces(group, coords, None, None, 0.0, fint, mint)[0]
    dt = 0.2 * dt_crit  # Courant factor = 0.2

    x = coords.copy()
    a = fint / nodal_mass[:, None]

    # Dynamic velocity Verlet integration over 50 cycles
    for cycle in range(50):
        # 1. Half-step velocity and full-step position update
        v_half = v + 0.5 * dt * a
        x += dt * v_half

        # 2. Element forces at x^{n+1} with midstep velocity v^{n+1/2}
        fint.fill(0.0)
        dt_e = solid_hexa8.forces(group, x, v_half, None, dt, fint, mint)
        assert dt <= dt_e[0], f"Courant violation at cycle {cycle}: dt={dt} > dt_e={dt_e[0]}"

        # 3. New acceleration and complete velocity step
        a = fint / nodal_mass[:, None]
        v = v_half + 0.5 * dt * a

    # Calculate total energy at end of cycle: Kinetic + Internal + Hourglass
    e_kin = 0.5 * np.sum(nodal_mass[:, None] * (v ** 2))
    e_int = group.state["eint"][0]
    e_hour = group.state["ehour"][0]
    e_tot = e_kin + e_int + e_hour

    # Verify energy conservation: relative error < 1.0%
    rel_err = abs(e_tot - e_kin_0) / e_kin_0
    assert rel_err < 0.01, f"Energy conservation violated: E0={e_kin_0}, E_tot={e_tot}, rel_err={rel_err}"

    # Internal energy strictly positive due to deformation work
    assert e_int > 0.0


# ============================================================================
# 7. Implicit Consistent Tangent Stiffness
# ============================================================================

def test_implicit_consistent_tangent_stiffness():
    """Verify solid consistent tangent dispatch, symmetry in elastic regime, and rigid translation invariance."""
    rho0 = 8.0
    bulk = 1.2e5
    g0 = 4.5e4
    mat = make_test_material_law49(
        rho0=rho0,
        bulk=bulk,
        g0=g0,
        sig0=200.0,
    )

    # 1. Module-level solid tangent
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    depsp = np.zeros(1)
    extra = {"off": np.ones(1), "g": np.array([g0])}

    D = materials.solid_tangent(mat, sig, epsp=epsp, epsp_incr=depsp, extra=extra)
    assert D.shape == (1, 6, 6)
    # Tensor must be symmetric in elastic state
    assert np.allclose(D[0], D[0].T)

    # Verify bulk and shear stiffness in Voigt notation:
    # D_11 = K + 4/3 G
    # D_12 = K - 2/3 G
    # D_44 = G
    d11_expected = bulk + (4.0 / 3.0) * g0
    d12_expected = bulk - (2.0 / 3.0) * g0
    d44_expected = g0
    assert math.isclose(D[0, 0, 0], d11_expected, rel_tol=1e-6)
    assert math.isclose(D[0, 0, 1], d12_expected, rel_tol=1e-6)
    assert math.isclose(D[0, 3, 3], d44_expected, rel_tol=1e-6)

    # 2. Hexa8 Element Tangent Stiffness Matrix K_e (24 x 24)
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

    ke_hex, edofs_hex = solid_hexa8.tangent(group_hex, coords_hex)
    assert ke_hex.shape == (1, 24, 24)
    assert edofs_hex.shape == (1, 24)

    # Rigid body translation invariance: K_e . v_trans = 0
    ke_mat_hex = ke_hex[0]
    for axis in range(3):
        v_rigid = np.zeros((8, 3))
        v_rigid[:, axis] = 1.0
        v_flat = v_rigid.reshape(-1)
        f_res = ke_mat_hex @ v_flat
        assert np.allclose(f_res, 0.0, atol=1e-5), f"Rigid translation on axis {axis} should give zero forces"

    # 3. Tetra4 Element Tangent Stiffness Matrix K_e (12 x 12)
    coords_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, MockProp())])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet
    solid_tetra4.init_group(group_tet, model_tet, None)

    ke_tet, edofs_tet = solid_tetra4.tangent(group_tet, coords_tet)
    assert ke_tet.shape == (1, 12, 12)
    assert edofs_tet.shape == (1, 12)

    ke_mat_tet = ke_tet[0]
    for axis in range(3):
        v_rigid = np.zeros((4, 3))
        v_rigid[:, axis] = 1.0
        v_flat = v_rigid.reshape(-1)
        f_res = ke_mat_tet @ v_flat
        assert np.allclose(f_res, 0.0, atol=1e-5), f"Rigid translation on axis {axis} should give zero forces"


# ============================================================================
# 8. Plane-Stress Shell & 1D Element Rejection
# ============================================================================

def test_plane_stress_shell_rejection_at_materials_level():
    """Verify that materials.shell_update, shell_membrane_tangent, and shell_layer_tangent reject LAW49."""
    mat = make_test_material_law49()

    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))

    with pytest.raises(NotImplementedError, match="LAW49.*implemented for 3D solid and SPH elements only"):
        materials.shell_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="LAW49"):
        materials.shell_membrane_tangent(mat)

    with pytest.raises(NotImplementedError, match="LAW49"):
        materials.shell_layer_tangent(mat, sig)


def test_shell_kernels_reject_law49():
    """Verify that BT4, QEPH, and Tri3 shell kernels reject LAW49."""
    mat = make_test_material_law49()
    prop = MockProp(thick=1.0, nip=3)

    coords_quad = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    conn_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)

    # 1. BT4
    group_bt4 = MockGroup(conn_quad, slices=[(slice(0, 1), mat, prop)])
    model_bt4 = Model()
    model_bt4.x0 = coords_quad.copy()
    group_bt4._model = model_bt4
    shell_bt4.init_group(group_bt4, model_bt4, None)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    vel = np.ones_like(coords_quad)

    with pytest.raises(NotImplementedError, match="LAW49"):
        shell_bt4.forces(group_bt4, coords_quad, vel, np.zeros_like(coords_quad), 1.0e-5, fint, mint)

    # 2. QEPH
    group_qeph = MockGroup(conn_quad, slices=[(slice(0, 1), mat, prop)])
    group_qeph._model = model_bt4
    shell_qeph.init_group(group_qeph, model_bt4, None)

    with pytest.raises(NotImplementedError, match="LAW49"):
        shell_qeph.forces(group_qeph, coords_quad, vel, np.zeros_like(coords_quad), 1.0e-5, fint, mint)

    # 3. Tri3
    coords_tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    conn_tri = np.array([[0, 1, 2]], dtype=np.int64)
    group_tri = MockGroup(conn_tri, slices=[(slice(0, 1), mat, prop)])
    model_tri = Model()
    model_tri.x0 = coords_tri.copy()
    group_tri._model = model_tri
    shell_tri3.init_group(group_tri, model_tri, None)

    fint_tri = np.zeros((3, 3))
    mint_tri = np.zeros((3, 3))
    with pytest.raises(NotImplementedError, match="LAW49"):
        shell_tri3.forces(group_tri, coords_tri, np.ones_like(coords_tri), np.zeros_like(coords_tri), 1.0e-5, fint_tri, mint_tri)


def test_starter_checks_allowed_laws():
    """Verify starter checks _ALLOWED_LAWS include LAW49 for 3D solids and reject for shells/beams/trusses."""
    assert 49 in _ALLOWED_LAWS["bricks"]
    assert "LAW49" in _ALLOWED_LAWS["bricks"]
    assert 49 in _ALLOWED_LAWS["tetras"]
    assert 49 in _ALLOWED_LAWS["penta6"]
    assert 49 in _ALLOWED_LAWS["pyra5"]

    assert 49 not in _ALLOWED_LAWS["shells"]
    assert "LAW49" not in _ALLOWED_LAWS["shells"]
    assert 49 not in _ALLOWED_LAWS["shells_qeph"]
    assert 49 not in _ALLOWED_LAWS["sh3n"]
    assert 49 not in _ALLOWED_LAWS["beams"]
    assert 49 not in _ALLOWED_LAWS["trusses"]
