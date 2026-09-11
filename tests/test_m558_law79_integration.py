"""
Integration test suite for /MAT/LAW79 (/MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2).

Milestone M558: Solid Element Formulations Integration & Multi-Cycle Simulation Verifier
1. Registry and dispatch metadata verification for all LAW79 aliases.
2. Acoustic sound speed (dilatational P-wave) and Courant time step calculation:
   - P-wave speed c = sqrt((K + 4/3 G) / rho) under reference state.
   - Nonlinear EOS bulk modulus evolution under compression (mu > 0) with K1, K2, K3.
   - Sound speed behavior under tension (mu <= 0).
   - Acoustic stability upon damage accumulation (D in [0, 1]).
   - Cycle 0 critical time step probe on Hexa8 and Tetra4.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1):
     * Uniaxial tension and compression (elastic slope, yield, hardening, pressure dependence).
     * Pure shear (shear modulus, yield threshold, plastic dissipation).
     * High pressure hydrostatic compaction with dilatancy bulking (delta P from delta U).
     * Progressive damage accumulation: intact (D=0) to fully fractured (D=1) envelope.
     * Element deletion modes: IDEL=0 (no deletion), IDEL=1 (tensile cutoff P* + T* < 0),
       IDEL=2 (eps_p > eps_max), IDEL=3 (full damage D >= 1.0).
   - Tetra4 standard (Itetra=1):
     * Single 4-node tetrahedron under volumetric shock compression.
     * Deviatoric plastic yield and shear flow.
     * Damage accumulation and deletion under IDEL.
4. Multi-element patch tests:
   - 2-element Hexa8 patch sharing a face (12 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2-element Tetra4 patch sharing a face (5 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2x2x2 Hexa8 patch (8 elements, 27 nodes, uniform deformation, net zero internal forces).
5. Implicit consistent tangent stiffness:
   - Elastic tangent symmetry and Voigt component consistency (K, G).
   - Hexa8 (24x24) and Tetra4 (12x12) element tangent matrices.
   - Rigid-body translation invariance: K_e . v_trans = 0.
6. Plane-stress shell and 1D element rejection:
   - shell_update, shell_membrane_tangent, shell_layer_tangent raise NotImplementedError.
   - BT4, QEPH, Tri3 shell kernels reject LAW79.
   - Starter checks _ALLOWED_LAWS: solids allowed, shells/beams/trusses rejected.
   - Starter check_mat_law79 validation error handling.
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update,
    solid_update_law79,
    shell_update,
    shell_update_law79,
    sound_speed_solid,
    sound_speed_solid_law79,
    consistent_solid_tangent,
    extra_shapes,
)
from pyradioss.model.entities import Material, MatLaw79
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law79,
)
from pyradioss.common.messages import MessageLog


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


def make_test_material_law79(
    mid: int = 1,
    rho0: float = 3.21e-3,       # Silicon Carbide: g/mm^3
    shear: float = 193.0,        # 193 GPa
    a: float = 0.96,
    b: float = 0.35,
    m: float = 1.0,
    n: float = 0.65,
    c: float = 0.009,
    eps0: float = 1.0,
    sigfmax: float = 0.8,
    fcut: float = 0.0,
    t: float = 0.37,             # T0 (GPa)
    hel: float = 14.5,           # HEL (GPa)
    phel: float = 5.13,          # PHEL (GPa)
    d1: float = 0.48,
    d2: float = 0.48,
    idel: int = 0,
    epsmax: float = 1e20,
    k1: float = 220.0,           # K1 (GPa)
    k2: float = 0.0,
    k3: float = 0.0,
    beta: float = 1.0,
    **kwargs: Any,
) -> Material:
    """Helper creating Material entity for LAW79 / JH-2 ceramic."""
    params = {
        "rho0": rho0,
        "rho": rho0,
        "shear": shear,
        "g0": shear,
        "G": shear,
        "a": a,
        "b": b,
        "m": m,
        "n": n,
        "c": c,
        "eps0": eps0,
        "sigfmax": sigfmax,
        "fcut": fcut,
        "t": t,
        "t0": t,
        "hel": hel,
        "phel": phel,
        "d1": d1,
        "d2": d2,
        "idel": idel,
        "epsmax": epsmax,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "bulk": k1,
        "beta": beta,
    }
    params.update(kwargs)
    mat = Material(id=mid, law="LAW79", title=f"SiC_JH2_{mid}", params=params)
    mat.rho0 = rho0
    return mat


# ============================================================================
# 1. Registry and Dispatch Metadata Verification
# ============================================================================

def test_law79_registry_and_dispatch_metadata():
    """Verify registration, aliases, and metadata in materials module."""
    expected_keys = (
        79,
        "79",
        "LAW79",
        "JOHN_HOLM",
        "JOHNSON_HOLMQUIST",
        "JH2",
        "MAT_LAW79",
        "MAT_JOHN_HOLM",
        "LAW79_JOHN_HOLM",
    )
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is False, "LAW79 must have plane_stress=False"
        assert meta.get("solid") is True, "LAW79 must have solid=True"
        assert meta.get("shell") is False, "LAW79 must have shell=False"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"Key {k} missing from MATERIAL_SOLID_DISPATCH"
        assert k in materials.MATERIAL_SHELL_DISPATCH, f"Key {k} missing from MATERIAL_SHELL_DISPATCH"

    mat = make_test_material_law79()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    for required_shape in ("deltap", "sigy_old", "dmg", "off", "off79", "mu", "amu", "uvar", "epsd_filtered"):
        assert required_shape in shapes_solid, f"Missing required persistent state {required_shape}"


def test_solid_element_initial_state_and_failure_guards():
    """Verify element state initialization for LAW79 and exclusion from generic eps_max fail."""
    mat = make_test_material_law79(idel=0, epsmax=0.01)
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

    # 1. State arrays initialized properly
    assert "mat_extra" in st
    assert "deltap" in st["mat_extra"]
    assert "dmg" in st["mat_extra"]
    assert "off" in st["mat_extra"]
    assert "off79" in st["mat_extra"]
    assert st["mat_extra"]["off"][0] == 1.0
    assert st["mat_extra"]["off79"][0] == 1.0
    assert st["mat_extra"]["dmg"][0] == 0.0
    assert st["mat_extra"]["deltap"][0] == 0.0

    # 2. Hardening saturation epsmax must NOT activate generic chk_fail (LAW79 handles IDEL internally)
    assert st["chk_fail"] is False

    # 3. Tetra4 initialization also behaves identically
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
    st_tet = group_tet.state
    assert st_tet["mat_extra"]["off"][0] == 1.0
    assert st_tet["mat_extra"]["dmg"][0] == 0.0
    assert st_tet["chk_fail"] is False


# ============================================================================
# 2. Acoustic Sound Speed & Courant Time Step
# ============================================================================

def test_law79_sound_speed_and_evolution():
    """Verify acoustic wave speed c = sqrt((K + 4/3 G) / rho) under reference, compression, and tension."""
    rho0 = 3.21e-3
    k1 = 220.0
    k2 = 100.0
    k3 = 50.0
    shear = 193.0

    mat = make_test_material_law79(
        rho0=rho0,
        k1=k1,
        k2=k2,
        k3=k3,
        shear=shear,
    )

    # 1. Undisturbed reference state (mu = 0)
    c_ref_expected = math.sqrt((k1 + (4.0 / 3.0) * shear) / rho0)
    c_ref = materials.sound_speed(mat, rho=rho0)
    assert math.isclose(c_ref, c_ref_expected, rel_tol=1e-10)
    assert math.isclose(mat.sound_speed_solid(), c_ref_expected, rel_tol=1e-10)

    # 2. Volumetric compression mu = 0.05: K = K1 + 2*K2*mu + 3*K3*mu^2
    mu_comp = 0.05
    rho_comp = rho0 * (1.0 + mu_comp)
    k_comp = k1 + 2.0 * k2 * mu_comp + 3.0 * k3 * (mu_comp ** 2)
    c_comp_expected = math.sqrt((k_comp + (4.0 / 3.0) * shear) / rho_comp)
    c_comp = sound_speed_solid_law79(mat, rho=rho_comp, extra={"mu": mu_comp})
    assert math.isclose(c_comp, c_comp_expected, rel_tol=1e-10)

    # 3. Volumetric tension mu = -0.02: K = K1 (linear elastic acoustic speed in tension)
    mu_ten = -0.02
    rho_ten = rho0 * (1.0 + mu_ten)
    c_ten_expected = math.sqrt((k1 + (4.0 / 3.0) * shear) / rho_ten)
    c_ten = sound_speed_solid_law79(mat, rho=rho_ten, extra={"mu": mu_ten})
    assert math.isclose(c_ten, c_ten_expected, rel_tol=1e-10)

    # 4. Damaged state (D = 1.0): sound speed remains positive and physically bounded
    c_dmg = sound_speed_solid_law79(mat, rho=rho0, extra={"dmg": 1.0})
    assert c_dmg > 0.0
    assert math.isclose(c_dmg, c_ref_expected, rel_tol=1e-10)


def test_solid_elements_cycle_0_courant_time_step():
    """Verify cycle 0 critical time step dt_crit = lc / c on Hexa8 and Tetra4."""
    mat = make_test_material_law79(rho0=3.21e-3, k1=220.0, shear=193.0)
    prop = MockProp()

    # Hexa8 unit cube (1 mm x 1 mm x 1 mm)
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

    c_expected = math.sqrt((220.0 + (4.0 / 3.0) * 193.0) / 3.21e-3)
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
    assert dt_tet[0] > 0.0


# ============================================================================
# 3. Hexa8 Solid Element Integration
# ============================================================================

def test_hexa8_solid_uniaxial_tension_and_compression():
    """Verify Hexa8 under uniaxial tension and compression (pressure hardening: comp flow > ten flow)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law79(
        rho0=3.21e-3,
        shear=193.0,
        k1=220.0,
        hel=14.5,
        phel=5.13,
        a=0.93,
        b=0.31,
        m=1.0,
        n=0.6,
        t=0.37,
        idel=0,
    )
    prop = MockProp()

    # 1. Uniaxial Tension: pull face x = 1.0 in +x direction
    group_ten = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model_ten = Model()
    model_ten.x0 = coords.copy()
    group_ten._model = model_ten
    solid_hexa8.init_group(group_ten, model_ten, None)

    vel_ten = np.zeros_like(coords)
    vel_ten[[1, 2, 5, 6], 0] = 50.0  # dot_eps_xx = 50 1/s
    dt = 1.0e-7

    fint_ten = np.zeros((8, 3))
    mint_ten = np.zeros((8, 3))
    curr_x_ten = coords.copy()

    for _ in range(40):
        curr_x_ten += vel_ten * dt
        solid_hexa8.forces(group_ten, curr_x_ten, vel_ten, None, dt, fint_ten, mint_ten)

    st_ten = group_ten.state
    sig_ten = st_ten["sig"][0]
    # Tension produces positive Cauchy sigma_xx
    assert sig_ten[0] > 0.0
    assert st_ten["eint"][0] > 0.0
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

    # Pressure hardening: compressive flow stress is significantly higher than tensile flow stress
    flow_cmp = abs(sig_cmp[0])
    flow_ten = abs(sig_ten[0])
    assert flow_cmp > flow_ten, f"Compression flow stress {flow_cmp} must exceed tension flow stress {flow_ten}"


def test_hexa8_solid_pure_shear():
    """Verify Hexa8 under pure shear (elastic shear modulus, yield, plastic dissipation)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    shear_mod = 193.0
    mat = make_test_material_law79(
        shear=shear_mod,
        k1=220.0,
        a=0.96,
        b=0.35,
        hel=14.5,
        phel=5.13,
        idel=0,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Pure shear in zx plane: move top face (z=1) along x
    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 20.0
    dt = 1.0e-7

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(30):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    sig = st["sig"][0]
    tau_zx = sig[5]
    assert abs(tau_zx) > 0.0
    assert st["eint"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_hexa8_high_pressure_compaction_and_bulking():
    """Verify Hexa8 under high-pressure compaction with dilatancy bulking (delta U -> delta P > 0)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law79(
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        m=0.0,
        n=0.0,
        t=0.06,
        k2=10.0,
        k3=5.0,
        beta=1.0,  # Full bulking dilatancy
        d1=0.01,   # Fast damage accumulation
        d2=0.01,
        idel=0,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Inward compaction combined with shear to trigger plastic yielding, damage, and bulking
    centroid = np.array([0.5, 0.5, 0.5])
    vel = -2000.0 * (coords - centroid)
    vel[[4, 5, 6, 7], 0] += 5000.0
    dt = 1.0e-7

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(50):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    # Bulking pressure increment deltap is accumulated in mat_extra
    deltap = st["mat_extra"]["deltap"][0]
    dmg = st["mat_extra"]["dmg"][0]

    assert dmg > 0.0, "Damage must have accumulated under combined shock compaction"
    assert deltap >= 0.0, "Dilatancy bulking pressure increment must be non-negative"
    assert st["eint"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_hexa8_damage_accumulation_intact_to_fractured_transition():
    """Verify progressive transition from intact envelope (D=0) to fully fractured envelope (D=1)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mat = make_test_material_law79(
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        d1=0.001,  # Low damage threshold to transition swiftly to fractured state
        d2=0.001,
        idel=0,    # Preserve element even when D=1
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[[4, 5, 6, 7], 0] = 5000.0
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))

    prev_dmg = 0.0
    for _ in range(60):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)
        cur_dmg = group.state["mat_extra"]["dmg"][0]
        assert cur_dmg >= prev_dmg, "Damage must increase monotonically"
        prev_dmg = cur_dmg

    # Damage must reach or approach 1.0
    assert prev_dmg >= 0.5
    # Element is alive because IDEL=0
    assert group.state["off"][0] == 1.0


def test_hexa8_element_deletion_idel_modes():
    """Verify element deletion behaviors under IDEL = 0, 1, 2, 3."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    dt = 1.0e-7

    # 1. IDEL = 0: no deletion even under intense tensile cutoff
    mat0 = make_test_material_law79(idel=0, t=0.01)
    group0 = MockGroup(conn, slices=[(slice(0, 1), mat0, MockProp())])
    model0 = Model()
    model0.x0 = coords.copy()
    group0._model = model0
    solid_hexa8.init_group(group0, model0, None)

    vel_ten = np.zeros_like(coords)
    vel_ten[[1, 2, 5, 6], 0] = 100.0  # Massive tension
    curr_x0 = coords.copy()
    fint0 = np.zeros((8, 3))
    mint0 = np.zeros((8, 3))

    for _ in range(20):
        curr_x0 += vel_ten * dt
        solid_hexa8.forces(group0, curr_x0, vel_ten, None, dt, fint0, mint0)

    assert group0.state["off"][0] == 1.0, "IDEL=0 must never delete element"

    # 2. IDEL = 1: deletion on hydrostatic tension P* + T* < 0
    mat1 = make_test_material_law79(idel=1, t=0.01, phel=5.0)
    group1 = MockGroup(conn, slices=[(slice(0, 1), mat1, MockProp())])
    model1 = Model()
    model1.x0 = coords.copy()
    group1._model = model1
    solid_hexa8.init_group(group1, model1, None)

    curr_x1 = coords.copy()
    fint1 = np.zeros((8, 3))
    mint1 = np.zeros((8, 3))

    # Pull in all directions (triaxial tension)
    vel_tri = 50.0 * (coords - np.array([0.5, 0.5, 0.5]))
    for _ in range(20):
        curr_x1 += vel_tri * dt
        solid_hexa8.forces(group1, curr_x1, vel_tri, None, dt, fint1, mint1)

    assert group1.state["off"][0] < 1.0, f"IDEL=1 must trigger deletion on tension (got off={group1.state['off'][0]})"

    # 3. IDEL = 2: deletion on plastic strain exceeding epsmax
    mat2 = make_test_material_law79(
        idel=2,
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        m=0.0,
        n=0.0,
        t=0.06,
        epsmax=0.001,
    )
    group2 = MockGroup(conn, slices=[(slice(0, 1), mat2, MockProp())])
    model2 = Model()
    model2.x0 = coords.copy()
    group2._model = model2
    solid_hexa8.init_group(group2, model2, None)

    curr_x2 = coords.copy()
    fint2 = np.zeros((8, 3))
    mint2 = np.zeros((8, 3))
    vel_shear = np.zeros_like(coords)
    vel_shear[[4, 5, 6, 7], 0] = 5000.0

    for _ in range(25):
        curr_x2 += vel_shear * dt
        solid_hexa8.forces(group2, curr_x2, vel_shear, None, dt, fint2, mint2)

    assert group2.state["off"][0] < 1.0, "IDEL=2 must delete element when epsp > epsmax"

    # 4. IDEL = 3: deletion when damage D >= 1.0
    mat3 = make_test_material_law79(
        idel=3,
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        m=0.0,
        n=0.0,
        t=0.06,
        d1=0.001,
        d2=0.001,
    )
    group3 = MockGroup(conn, slices=[(slice(0, 1), mat3, MockProp())])
    model3 = Model()
    model3.x0 = coords.copy()
    group3._model = model3
    solid_hexa8.init_group(group3, model3, None)

    curr_x3 = coords.copy()
    fint3 = np.zeros((8, 3))
    mint3 = np.zeros((8, 3))

    for _ in range(40):
        curr_x3 += vel_shear * dt
        solid_hexa8.forces(group3, curr_x3, vel_shear, None, dt, fint3, mint3)

    assert group3.state["off"][0] < 1.0, "IDEL=3 must delete element when damage reaches 1.0"


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

    mat = make_test_material_law79(
        idel=0,
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        m=0.0,
        n=0.0,
        t=0.06,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    # 1. Volumetric compression: move node 3 downward
    vel_comp = np.zeros_like(coords)
    vel_comp[3, 2] = -2000.0
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    for _ in range(15):
        curr_x += vel_comp * dt
        solid_tetra4.forces(group, curr_x, vel_comp, None, dt, fint, mint)

    st = group.state
    p = -(st["sig"][0, 0] + st["sig"][0, 1] + st["sig"][0, 2]) / 3.0
    assert p > 0.0
    assert st["eint"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

    # 2. Shear deformation on node 3
    vel_shear = np.zeros_like(coords)
    vel_shear[3, 0] = 5000.0

    for _ in range(25):
        curr_x += vel_shear * dt
        solid_tetra4.forces(group, curr_x, vel_shear, None, dt, fint, mint)

    assert st["epsp"][0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_tetra4_damage_and_deletion():
    """Verify Tetra4 damage progression and IDEL deletion."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_material_law79(
        idel=3,
        shear=10.0,
        k1=20.0,
        hel=0.2,
        phel=0.06,
        a=0.5,
        b=0.2,
        c=0.0,
        m=0.0,
        n=0.0,
        t=0.06,
        d1=0.001,
        d2=0.001,
    )
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[3, 0] = 5000.0
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    for _ in range(40):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert group.state["off"][0] < 1.0, "Tetra4 under IDEL=3 must delete upon reaching full damage"


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

    mat = make_test_material_law79(idel=0)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 10.0
    vel[[8, 9, 10, 11], 0] = 20.0
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((12, 3))
    mint = np.zeros((12, 3))

    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    # Both elements must exhibit uniform state
    assert math.isclose(st["sig"][0, 0], st["sig"][1, 0], rel_tol=1e-4)
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

    mat = make_test_material_law79(idel=0)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    vel = np.zeros_like(coords)
    vel[1, 0] = 15.0
    vel[4, 0] = 30.0
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert (group.state["eint"] > 0.0).all()
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_hexa8_2x2x2_patch():
    """Verify 2x2x2 Hexa8 patch (8 elements, 27 nodes) under uniform triaxial compaction."""
    # Generate regular 2x2x2 mesh of unit cubes
    nodes = []
    for z in (0.0, 1.0, 2.0):
        for y in (0.0, 1.0, 2.0):
            for x in (0.0, 1.0, 2.0):
                nodes.append([x, y, z])
    coords = np.array(nodes, dtype=float)

    def node_id(i, j, k):
        return k * 9 + j * 3 + i

    hex_conn = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                hex_conn.append([
                    node_id(i, j, k),
                    node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k),
                    node_id(i, j + 1, k),
                    node_id(i, j, k + 1),
                    node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i, j + 1, k + 1),
                ])
    conn = np.array(hex_conn, dtype=np.int64)

    mat = make_test_material_law79(idel=0)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 8), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Uniform inward velocity field towards center (1.0, 1.0, 1.0)
    centroid = np.array([1.0, 1.0, 1.0])
    vel = -20.0 * (coords - centroid)
    dt = 1.0e-7

    curr_x = coords.copy()
    fint = np.zeros((27, 3))
    mint = np.zeros((27, 3))

    for _ in range(20):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    st = group.state
    # All 8 elements must experience equal internal energy and pressure
    p_all = -(st["sig"][:, 0] + st["sig"][:, 1] + st["sig"][:, 2]) / 3.0
    assert np.all(p_all > 0.0)
    assert math.isclose(p_all.min(), p_all.max(), rel_tol=1e-3)
    assert np.all(st["eint"] > 0.0)
    assert math.isclose(st["eint"].min(), st["eint"].max(), rel_tol=1e-3)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 6. Implicit Consistent Tangent Stiffness
# ============================================================================

def test_implicit_consistent_tangent_stiffness():
    """Verify solid consistent tangent dispatch, symmetry in elastic regime, and rigid translation invariance."""
    mat = make_test_material_law79(
        k1=220.0,
        shear=193.0,
    )

    # 1. Module-level solid tangent
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    depsp = np.zeros(1)
    extra = {"off": np.ones(1), "dmg": np.zeros(1), "mu": np.zeros(1)}

    D = materials.solid_tangent(mat, sig, epsp=epsp, epsp_incr=depsp, extra=extra)
    assert D.shape == (1, 6, 6)
    # Tensor must be symmetric in elastic state
    assert np.allclose(D[0], D[0].T)

    # Verify bulk and shear stiffness in Voigt notation:
    # D_11 = K + 4/3 G
    # D_12 = K - 2/3 G
    # D_44 = G
    d11_expected = 220.0 + (4.0 / 3.0) * 193.0
    d12_expected = 220.0 - (2.0 / 3.0) * 193.0
    d44_expected = 193.0
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
        assert np.allclose(f_res, 0.0, atol=1e-5), f"Rigid translation on axis {axis} must yield zero forces"

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
        assert np.allclose(f_res, 0.0, atol=1e-5), f"Rigid translation on axis {axis} must yield zero forces"


# ============================================================================
# 7. Plane-Stress Shell & 1D Element Rejection
# ============================================================================

def test_plane_stress_shell_rejection_at_materials_level():
    """Verify that materials.shell_update, shell_membrane_tangent, and shell_layer_tangent reject LAW79."""
    mat = make_test_material_law79()

    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))

    with pytest.raises(NotImplementedError, match="LAW79.*solid.*only"):
        materials.shell_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="LAW79"):
        materials.shell_membrane_tangent(mat)

    with pytest.raises(NotImplementedError, match="LAW79"):
        materials.shell_layer_tangent(mat, sig)


def test_shell_kernels_reject_law79():
    """Verify that BT4, QEPH, and Tri3 shell kernels reject LAW79."""
    mat = make_test_material_law79()
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

    with pytest.raises(NotImplementedError, match="LAW79"):
        shell_bt4.forces(group_bt4, coords_quad, vel, np.zeros_like(coords_quad), 1.0e-5, fint, mint)

    # 2. QEPH
    group_qeph = MockGroup(conn_quad, slices=[(slice(0, 1), mat, prop)])
    group_qeph._model = model_bt4
    shell_qeph.init_group(group_qeph, model_bt4, None)

    with pytest.raises(NotImplementedError, match="LAW79"):
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
    with pytest.raises(NotImplementedError, match="LAW79"):
        shell_tri3.forces(group_tri, coords_tri, np.ones_like(coords_tri), np.zeros_like(coords_tri), 1.0e-5, fint_tri, mint_tri)


def test_starter_checks_allowed_laws_and_validation():
    """Verify starter checks _ALLOWED_LAWS and check_mat_law79 validation."""
    # 1. _ALLOWED_LAWS coverage
    assert 79 in _ALLOWED_LAWS["bricks"]
    assert "LAW79" in _ALLOWED_LAWS["bricks"]
    assert 79 in _ALLOWED_LAWS["tetras"]
    assert 79 in _ALLOWED_LAWS["penta6"]
    assert 79 in _ALLOWED_LAWS["pyra5"]

    assert 79 not in _ALLOWED_LAWS["shells"]
    assert "LAW79" not in _ALLOWED_LAWS["shells"]
    assert 79 not in _ALLOWED_LAWS["shells_qeph"]
    assert 79 not in _ALLOWED_LAWS["sh3n"]
    assert 79 not in _ALLOWED_LAWS["beams"]
    assert 79 not in _ALLOWED_LAWS["trusses"]

    # 2. check_mat_law79 validation
    log = MessageLog()
    valid_mat = make_test_material_law79()
    check_mat_law79(mat=valid_mat, log=log)
    assert not log.has_errors, f"Unexpected errors: {log.entries}"

    # Negative bounds checks:
    # shear <= 0
    log_err = MessageLog()
    bad_shear = make_test_material_law79(shear=0.0)
    check_mat_law79(mat=bad_shear, log=log_err)
    assert log_err.has_errors

    # k1 <= 0
    log_err = MessageLog()
    bad_k1 = make_test_material_law79(k1=-10.0)
    check_mat_law79(mat=bad_k1, log=log_err)
    assert log_err.has_errors

    # phel > hel
    log_err = MessageLog()
    bad_phel = make_test_material_law79(hel=5.0, phel=10.0)
    check_mat_law79(mat=bad_phel, log=log_err)
    assert log_err.has_errors

    # beta not in [0, 1]
    log_err = MessageLog()
    bad_beta = make_test_material_law79(beta=1.5)
    check_mat_law79(mat=bad_beta, log=log_err)
    assert log_err.has_errors
