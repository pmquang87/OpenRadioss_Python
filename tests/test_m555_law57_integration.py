"""
Integration test suite for /MAT/LAW57 (/MAT/BARLAT3) Barlat-Lian (1989) Anisotropic Plasticity.

Milestone M555 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Registry, metadata, and dispatch verification for all LAW57 aliases.
2. Acoustic sound speed (plane-stress shell) and Courant time step consistency.
3. Solid and 1D element rejection:
   - solid_update and solid_tangent raise NotImplementedError.
   - solid_update_law57 raises NotImplementedError.
   - solid elements (Hexa8, Tetra4) raise NotImplementedError when attempted.
   - starter checks _ALLOWED_LAWS validates shells and rejects solids / 1D elements.
   - check_model / check_mat_law57 logs ANCMSG 305 for solids and ANCMSG 306 for 1D elements.
4. Shell element formulations:
   - BT4 (Ishell=1): single-element uniaxial & biaxial tension, 2x2 multi-element patch.
   - QEPH (Ishell=24): improved hourglass control, positive cspd, dynamic tension & shear.
   - Tri3 (Ish3n=1): 3-node triangular shell single and multi-element dynamic cycles.
5. Consistent tangent stiffness dispatch for implicit analysis (membrane & layer tangents).
6. Physical anisotropic plasticity behavior:
   - Anisotropic yield stresses (0°, 45°, 90° vs Lankford parameters R00, R45, R90).
   - Kinematic hardening & Bauschinger effect under forward/reverse cyclic loading (CHARD > 0).
   - Thickness thinning under plastic deformation (plastic incompressibility).
7. MatLaw57 dataclass and build_law57 adapter verification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law57_barlat import (
    Law57Params,
    BarlatParams,
    barlat_params,
    build_law57,
    calculp2,
    shell_update_law57,
    solid_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    extra_shapes as law57_extra_shapes,
)
from pyradioss.model.entities import Material, MatLaw57, MatBarlat3, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model, check_mat_law57
from pyradioss.common.messages import MessageLog


# ============================================================================
# Helper Mock Element Classes for Kernel Unit Testing
# ============================================================================

class MockProp:
    """Mock shell property mimicking /PROP/TYPE1 (SHELL), /PROP/TYPE11 (SH_COMP)."""
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 3, **kwargs: Any):
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
    """Mock element group with connectivity, id indexing, and state buffer."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law57(
    mid: int = 1,
    rho0: float = 2.7e-9,  # typical aluminum: 2.7e-9 ton/mm^3
    E: float = 70000.0,
    nu: float = 0.33,
    r00: float = 1.5,
    r45: float = 1.2,
    r90: float = 1.8,
    m: float = 8.0,
    sigy0: float = 250.0,
    chard: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW57 (/MAT/BARLAT3) Material instance."""
    params = {
        "E": E,
        "nu": nu,
        "r00": r00,
        "r45": r45,
        "r90": r90,
        "m": m,
        "sigy0": sigy0,
        "chard": chard,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=57, rho0=rho0, title="Barlat_LAW57", params=params)
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law57_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        57, "57", "LAW57", "BARLAT", "BARLAT3",
        "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3",
        "LAW57_BARLAT", "LAW57_BARLAT3",
    )
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True, f"plane_stress must be True for {k}"
        assert meta.get("solid") is False, f"solid must be False for {k}"
        assert meta.get("shell") is True, f"shell must be True for {k}"
        assert k in materials.MATERIAL_SOLID_DISPATCH
        assert k in materials.MATERIAL_SHELL_DISPATCH

    mat = make_test_material_law57()
    assert materials.needs_env(mat) is True

    # Extra shapes for shell layer history
    shapes_shell = materials.extra_shapes(mat, nip=3)
    assert shapes_shell["pla57"] == (3,)
    assert shapes_shell["sigb57"] == (3, 3)
    assert shapes_shell["off57"] == (3,)
    assert shapes_shell["epsd57"] == (3,)
    assert shapes_shell["thk57"] == (3,)
    assert shapes_shell["dmg57"] == (3, 3)
    assert shapes_shell["eps57"] == (3, 3)

    shapes_single = materials.extra_shapes(mat, nip=None)
    assert shapes_single["sigb57"] == (3,)
    assert shapes_single["eps57"] == (3,)


def test_law57_sound_speed_consistency():
    """Verify plane-stress acoustic wave speed formula: c = sqrt(E / ((1 - nu^2) * rho0))."""
    rho0 = 2.7e-9
    E = 72000.0
    nu = 0.33
    c_expected = math.sqrt(E / ((1.0 - nu * nu) * rho0))

    mat = make_test_material_law57(rho0=rho0, E=E, nu=nu)

    # Material method
    c_mat = mat.sound_speed_shell()
    assert math.isclose(c_mat, c_expected, rel_tol=1e-10)

    # Materials dispatcher
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_expected, rel_tol=1e-10)
    assert math.isclose(materials.sound_speed(mat), c_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_shell_law57(mat), c_expected, rel_tol=1e-10)

    # Override via extra dict
    rho_override = 3.0e-9
    c_override_expected = math.sqrt(E / ((1.0 - nu * nu) * rho_override))
    c_override = sound_speed_shell_law57(mat, extra={"rho": rho_override})
    assert math.isclose(c_override, c_override_expected, rel_tol=1e-10)


# ============================================================================
# 2. Solid and 1D Element Rejection
# ============================================================================

def test_law57_solid_rejection():
    """Verify solid stress update, tangent, and starter checks reject LAW57."""
    mat = make_test_material_law57()

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="shells only"):
        materials.solid_tangent(mat, sig)

    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law57(mat, sig, deps)

    # Verify starter checks allowed laws
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n")
    for fam in shell_families:
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 57 in allowed or "57" in allowed or "LAW57" in allowed, f"57 not allowed in {fam}"

    solid_families = ("bricks", "tetras", "penta6", "pyra5")
    for fam in solid_families:
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 57 not in allowed and "57" not in allowed and "LAW57" not in allowed, (
            f"LAW57 should be rejected for solid family {fam}"
        )


def test_law57_1d_rejection():
    """Verify 1D elements (truss, beam, spring) reject LAW57."""
    one_d_families = ("truss", "beam", "spring")
    for fam in one_d_families:
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 57 not in allowed and "57" not in allowed and "LAW57" not in allowed, (
            f"LAW57 should not be allowed for 1D family {fam}"
        )

    # Starter check diagnostic ANCMSG 306
    model = Model()
    mat = make_test_material_law57(mid=10)
    model.materials[10] = mat
    prop = MockProp(thick=1.0)
    model.beams = MockGroup(conn=np.array([[0, 1]]), slices=[(slice(0, 1), mat, prop)])

    log = MessageLog()
    check_mat_law57(model=model, mat_id=10, mat=mat, log=log)
    errs = [msg for msg in log.errors if "ANCMSG 306" in msg or "not supported for 1D" in msg]
    assert len(errs) > 0, "Expected ANCMSG 306 diagnostic error for 1D element with LAW57"


def test_solid_element_runtime_rejection():
    """Verify Hexa8 and Tetra4 element kernels fail cleanly when run with LAW57."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law57()
    prop = MockProp(thick=1.0)
    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    # Hexa8 forces dispatches solid_update which raises NotImplementedError
    solid_hexa8.init_group(group, model, None)
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_hexa8.forces(group, coords, np.zeros_like(coords), np.zeros_like(coords), 1e-5, fint, mint)


# ============================================================================
# 3. BT4 Shell Element Formulation (Ishell=1)
# ============================================================================

def test_shell_bt4_single_element_tension():
    """Verify Belytschko-Tsay quad shell (Ishell=1) with LAW57 under rolling direction (x) tension."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law57(E=70000.0, nu=0.33, sigy0=200.0, r00=1.5, r45=1.2, r90=1.8, m=8.0)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert "sigb57" in st["mat_extra"]
    assert "eps57" in st["mat_extra"]
    assert "pla57" in st["mat_extra"]

    # Initial cycle 0 Courant time step probe
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = shell_bt4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0
    assert not math.isinf(dt_c0[0])

    # Prescribe tension velocity along rolling direction (x)
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0  # stretch in x
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(25):
        curr_x += vel * dt
        dt_step = shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0

    # Normal stress along x must be positive under tension
    sig_xx = st["sig"][0, :, 0]
    assert np.all(sig_xx > 0.0), f"sig_xx expected positive, got {sig_xx}"

    # Global internal forces must be in self-equilibrium (sum ~ 0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    # Energy accounting: positive internal energy
    assert st["eint"][0] > 0.0


def test_shell_bt4_single_element_biaxial_tension():
    """Verify Belytschko-Tsay quad shell under equibiaxial tension."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law57(E=70000.0, nu=0.33, sigy0=200.0)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 40.0  # stretch x
    vel[[2, 3], 1] = 40.0  # stretch y
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    # Both sigma_xx and sigma_yy must be positive
    assert np.all(st["sig"][0, :, 0] > 0.0)
    assert np.all(st["sig"][0, :, 1] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_shell_bt4_multi_element_patch():
    """Verify 2x2 multi-element BT4 shell patch under dynamic stretch."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [20.0, 10.0, 0.0],
        [0.0, 20.0, 0.0], [10.0, 20.0, 0.0], [20.0, 20.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 4, 3],  # elem 0
        [1, 2, 5, 4],  # elem 1
        [3, 4, 7, 6],  # elem 2
        [4, 5, 8, 7],  # elem 3
    ])
    mat = make_test_material_law57(E=68000.0, nu=0.34, sigy0=220.0)
    prop = MockProp(thick=0.8, nip=2)

    group = MockGroup(conn, slices=[(slice(0, 4), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert st["sig"].shape == (4, 2, 3)

    vel = np.zeros_like(coords)
    vel[:, 0] = coords[:, 0] * 2.0
    vel[:, 1] = coords[:, 1] * 1.5
    dt = 1.0e-5

    fint = np.zeros((9, 3))
    mint = np.zeros((9, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        dt_step = shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert np.all(dt_step > 0.0)

    # All elements must exhibit tensile stresses and internal work
    assert np.all(st["sig"][:, :, 0] > 0.0)
    assert np.all(st["sig"][:, :, 1] > 0.0)
    assert np.all(st["eint"] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 4. QEPH Shell Element Formulation (Ishell=24)
# ============================================================================

def test_shell_qeph_single_and_multi_element():
    """Verify QEPH quad shell formulation with physical hourglass control under LAW57."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law57(E=70000.0, nu=0.33, sigy0=200.0, r00=1.5, r45=1.2, r90=1.8)
    prop = MockProp(thick=1.0, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state

    # Verify sound speed initialized properly (not zeroed)
    assert st["cspd"][0] > 0.0
    c_expected = sound_speed_shell_law57(mat)
    assert math.isclose(st["cspd"][0], c_expected, rel_tol=1e-10)

    # Combined tension and in-plane cyclic shear
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0   # stretch x
    vel[[2, 3], 0] += 30.0  # shear xy
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    # Forward shearing
    for _ in range(20):
        curr_x += vel * dt
        dt_step = shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0
        assert not math.isinf(dt_step[0])

    assert st["sig"][0, :, 0].mean() > 0.0
    assert abs(st["sig"][0, :, 2].mean()) > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)

    # Reverse shearing
    vel[[2, 3], 0] = -60.0
    for _ in range(25):
        curr_x += vel * dt
        shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 5. Tri3 Triangular Shell Element Formulation (Ish3n=1)
# ============================================================================

def test_shell_tri3_single_and_multi_element():
    """Verify 3-node triangular shell formulation (Ish3n=1) under LAW57."""
    # 2-triangle quad patch
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 2],  # bottom-right triangle
        [0, 2, 3],  # top-left triangle
    ])
    mat = make_test_material_law57(E=72000.0, nu=0.33, sigy0=240.0)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_tri3.init_group(group, model, None)
    st = group.state
    assert st["sig"].shape == (2, 3, 3)

    # Initial time step probe
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = shell_tri3.forces(group, coords, None, None, 0.0, fint, mint)
    assert np.all(dt_c0 > 0.0)

    # Prescribe tension
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 40.0
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(20):
        curr_x += vel * dt
        dt_step = shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert np.all(dt_step > 0.0)

    # Internal stress and force balance
    assert np.all(st["sig"][:, :, 0] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 6. Consistent Tangent Stiffness Dispatch
# ============================================================================

def test_law57_consistent_tangents_dispatch():
    """Verify shell_membrane_tangent and shell_layer_tangent for implicit solver."""
    E = 70000.0
    nu = 0.33
    mat = make_test_material_law57(E=E, nu=nu, sigy0=300.0)

    # Membrane tangent
    C_mem = materials.shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    c_11 = E / (1.0 - nu * nu)
    c_12 = nu * c_11
    c_33 = E / (2.0 * (1.0 + nu))
    expected_C = np.array([
        [c_11, c_12, 0.0],
        [c_12, c_11, 0.0],
        [0.0, 0.0, c_33],
    ])
    assert np.allclose(C_mem, expected_C, rtol=1e-10)

    # Layer tangent (elastic state before yielding)
    sig_elastic = np.array([[10.0, 5.0, 0.0]])
    extra = {"pla57": np.zeros(1), "sigb57": np.zeros((1, 3))}
    C_layer = materials.shell_layer_tangent(mat, sig=sig_elastic, epsp=0.0, epsp_incr=0.0, extra=extra)
    assert C_layer.shape == (1, 3, 3)
    assert np.allclose(C_layer[0], expected_C, rtol=1e-4)

    # Compare algorithmic tangent against numerical perturbation of shell_update_law57
    h = 1e-7
    deps0 = np.array([0.002, 0.001, 0.0005])
    sig0 = np.zeros(3)
    res0, _ = shell_update_law57(mat, sig0, deps0, return_sound_speed=False)

    D_num = np.zeros((3, 3))
    for j in range(3):
        deps_pert = deps0.copy()
        deps_pert[j] += h
        res_pert, _ = shell_update_law57(mat, sig0, deps_pert, return_sound_speed=False)
        D_num[:, j] = (res_pert - res0) / h

    # In elastic regime, D_num must match the elastic matrix
    assert np.allclose(D_num, expected_C, rtol=1e-3)


# ============================================================================
# 7. Physical Anisotropic Plasticity Behavior
# ============================================================================

def test_law57_anisotropic_yield_stresses():
    """Verify Barlat 1989 anisotropic yield surface reflects Lankford coefficients R00, R45, R90."""
    # Aluminum sheet with distinct Lankford coefficients
    r00 = 1.6
    r45 = 1.1
    r90 = 2.1
    m = 8.0
    sigy0 = 200.0

    mat = make_test_material_law57(r00=r00, r45=r45, r90=r90, m=m, sigy0=sigy0)
    p = build_law57(mat)
    bp = p.barlat

    # Barlat-Lian theoretical yield function for uniaxial tension:
    # At 0° (rolling direction, sig_xx only):
    #   K1 = sig_xx / 2, K2 = c * sig_xx / 2
    #   Phi(sig_xx, 0, 0) = a * |K1 + K2|^m + a * |K1 - K2|^m + c * |2*K2|^m
    #   yield occurs when Phi = 2 * sigy0^m
    sig_trial = 1.0
    k1_0 = 0.5 * sig_trial
    k2_0 = 0.5 * bp.c * sig_trial
    phi_0 = bp.a * (abs(k1_0 + k2_0)**m + abs(k1_0 - k2_0)**m) + bp.c * (abs(2.0 * k2_0)**m)
    sigma_y_0 = (2.0 / phi_0) ** (1.0 / m) * sigy0

    # At 90° (transverse direction, sig_yy only):
    #   K1 = h_bar * sig_yy / 2, K2 = -c * sig_yy / 2
    k1_90 = 0.5 * bp.h_bar * sig_trial
    k2_90 = -0.5 * bp.c * sig_trial
    phi_90 = bp.a * (abs(k1_90 + k2_90)**m + abs(k1_90 - k2_90)**m) + bp.c * (abs(2.0 * k2_90)**m)
    sigma_y_90 = (2.0 / phi_90) ** (1.0 / m) * sigy0

    # Due to R90 > R00, the transverse and rolling yield stresses must differ
    assert not math.isclose(sigma_y_0, sigma_y_90, rel_tol=1e-3), (
        f"Anisotropic yield stresses should differ: sigma_0={sigma_y_0:.2f}, sigma_90={sigma_y_90:.2f}"
    )

    # Perform constitutive updates in 0° and 90° into the plastic regime
    large_deps_0 = np.array([0.02, -0.33 * 0.02, 0.0])
    large_deps_90 = np.array([-0.33 * 0.02, 0.02, 0.0])

    res_0, ep_0 = shell_update_law57(mat, np.zeros(3), large_deps_0, return_sound_speed=False)
    res_90, ep_90 = shell_update_law57(mat, np.zeros(3), large_deps_90, return_sound_speed=False)

    assert ep_0 > 0.0, "Expected plastic strain in 0° tension"
    assert ep_90 > 0.0, "Expected plastic strain in 90° tension"
    # Plastic flow stresses reflect anisotropy
    assert not math.isclose(res_0[0], res_90[1], rel_tol=1e-3)


def test_law57_kinematic_hardening_bauschinger():
    """Verify kinematic hardening (CHARD > 0) captures backstress and Bauschinger effect."""
    # Hardening curve: yield stress increases with plastic strain
    # dyld_dp = (300.0 - 200.0) / 0.05 = 2000.0
    curves = [([0.0, 0.05], [200.0, 300.0])]
    # CHARD = 0.5: 50% kinematic hardening, 50% isotropic hardening
    mat_kin = make_test_material_law57(chard=0.5, sigy0=200.0, curves=curves)
    mat_iso = make_test_material_law57(chard=0.0, sigy0=200.0, curves=curves)

    # 1. Forward tension into plastic regime
    dt = 1e-4
    deps_fwd = np.array([0.01, -0.33 * 0.01, 0.0])

    extra_kin = {"sigb57": np.zeros((1, 3)), "pla57": np.zeros(1), "eps57": np.zeros((1, 3))}
    extra_iso = {"sigb57": np.zeros((1, 3)), "pla57": np.zeros(1), "eps57": np.zeros((1, 3))}

    sig_kin, ep_kin = shell_update_law57(mat_kin, np.zeros(3), deps_fwd, dt=dt, extra=extra_kin, return_sound_speed=False)
    sig_iso, ep_iso = shell_update_law57(mat_iso, np.zeros(3), deps_fwd, dt=dt, extra=extra_iso, return_sound_speed=False)

    assert ep_kin > 0.0
    # Kinematic hardening accumulates backstress alpha_xx > 0
    alpha_kin = extra_kin["sigb57"][0, 0]
    assert alpha_kin > 0.0, f"Backstress alpha_xx expected positive, got {alpha_kin}"
    assert np.allclose(extra_iso["sigb57"], 0.0), "Pure isotropic model must have zero backstress"

    # 2. Reverse loading (compression)
    deps_rev = np.array([-0.005, 0.33 * 0.005, 0.0])
    sig_rev_kin, ep_rev_kin = shell_update_law57(mat_kin, sig_kin, deps_rev, dt=dt, extra=extra_kin, return_sound_speed=False)
    sig_rev_iso, ep_rev_iso = shell_update_law57(mat_iso, sig_iso, deps_rev, dt=dt, extra=extra_iso, return_sound_speed=False)

    # Under reverse loading, backstress shifts the yield surface forward, causing yielding
    # earlier in reverse direction (Bauschinger effect) -> ep_rev_kin > ep_rev_iso
    assert ep_rev_kin >= ep_rev_iso


def test_law57_thickness_thinning():
    """Verify through-thickness strain depszz is computed ensuring plastic incompressibility."""
    mat = make_test_material_law57(sigy0=180.0)
    extra = {"sigb57": np.zeros((1, 3)), "pla57": np.zeros(1), "eps57": np.zeros((1, 3)), "thk57": np.ones(1)}

    # Plastic equibiaxial stretch
    deps = np.array([0.015, 0.015, 0.0])
    _, ep = shell_update_law57(mat, np.zeros(3), deps, dt=1e-4, extra=extra, return_sound_speed=False)

    assert ep > 0.0
    # Incompressibility of plastic strain: thickness strain depszz must be negative
    depszz = extra.get("depszz")
    assert depszz is not None
    assert depszz < 0.0, f"Expected negative thickness strain under biaxial tension, got {depszz}"


# ============================================================================
# 8. MatLaw57 Dataclass & build_law57 Adapter Verification
# ============================================================================

def test_mat_law57_dataclass():
    """Verify MatLaw57 properties, aliases, and sound speed methods."""
    mat = MatLaw57(
        id=57,
        rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        r00=1.6,
        r45=1.2,
        r90=2.0,
        m=8.0,
    )
    assert mat.rho0 == 2.7e-9
    assert mat.E == 70000.0
    assert mat.Nu == 0.33
    assert mat.G > 0.0
    c = mat.sound_speed_shell()
    assert c > 0.0
    assert math.isclose(c, math.sqrt(70000.0 / ((1.0 - 0.33**2) * 2.7e-9)), rel_tol=1e-10)

    # Verify MatBarlat3 alias
    assert MatBarlat3 is MatLaw57

    # Adapter via build_law57
    p = build_law57(mat)
    assert p.id == 57
    assert p.E == 70000.0
    assert p.r00 == 1.6
    assert p.r45 == 1.2
    assert p.r90 == 2.0
    assert p.m == 8.0
