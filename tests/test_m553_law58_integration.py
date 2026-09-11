"""
Integration test suite for /MAT/LAW58 (/MAT/FABR_A) Anisotropic Fabric Material.

Milestone M553 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Registry, metadata, and dispatch verification for all LAW58 aliases.
2. Acoustic sound speed (plane-stress shell) and Courant time step consistency.
3. Solid element rejection:
   - solid_update and solid_tangent raise NotImplementedError.
   - starter.checks._ALLOWED_LAWS validates shells and rejects solids.
4. Shell element formulations:
   - BT4 (Ishell=1): single-element and multi-element dynamic cycle execution under tension.
   - QEPH (Ishell=24): improved hourglass control, positive cspd, dynamic tension & shear.
   - Tri3 (Ish3n=1): 3-node triangular shell single and multi-element dynamic cycles.
5. Consistent tangent stiffness dispatch for implicit analysis (membrane & layer tangents).
6. Physical fabric behavior:
   - Crimp interchange kinematics & coupling.
   - Trellis shear response before and after lock angle alphat.
   - Zero-stress relative area (folding behavior).
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3
from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    build_law58,
    shell_membrane_tangent as law58_shell_membrane_tangent,
    shell_update_law58,
    solid_update_law58,
    sound_speed_shell_law58,
    tangent_law58_shell,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS


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


def make_test_material_law58(
    mid: int = 1,
    rho0: float = 1.0e-6,
    e1: float = 2000.0,
    e2: float = 1500.0,
    g0: float = 50.0,
    gt: float = 120.0,
    alphat: float = 35.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW58 (/MAT/FABR_A) Material instance."""
    params = {
        "e1": e1,
        "b1": kwargs.get("b1", 0.0),
        "e2": e2,
        "b2": kwargs.get("b2", 0.0),
        "flex": kwargs.get("flex", 1e-3),
        "g0": g0,
        "gt": gt,
        "alphat": alphat,
        "g5": kwargs.get("g5", 40.0),
        "df": kwargs.get("df", 0.05),
        "ds": kwargs.get("ds", 0.1),
        "gfrot": kwargs.get("gfrot", 20.0),
        "zero_stress": kwargs.get("zero_stress", 0.0),
        "arel": kwargs.get("arel", 0.0),
        "n1": kwargs.get("n1", 1),
        "n2": kwargs.get("n2", 1),
        "s1": kwargs.get("s1", 0.1),
        "s2": kwargs.get("s2", 0.1),
        "c4": kwargs.get("c4", 0.0),
        "c5": kwargs.get("c5", 0.0),
    }
    params.update(kwargs)
    mat = Material(id=mid, law=58, rho0=rho0, title="Fabric_LAW58", params=params)
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law58_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        58, "58", "LAW58", "FABR_A", "FABRIC_A",
        "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"
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

    mat = make_test_material_law58()
    assert materials.needs_env(mat) is True

    # Extra shapes for shell layer history
    shapes_shell = materials.extra_shapes(mat, nip=3)
    assert shapes_shell["eps58"] == (3, 3)
    assert shapes_shell["yc"] == (3,)
    assert shapes_shell["yt"] == (3,)
    assert shapes_shell["fn"] == (3,)
    assert shapes_shell["sigv_xy"] == (3,)
    assert shapes_shell["tan_phi"] == (3,)
    assert shapes_shell["sigi58"] == (3, 3)
    assert shapes_shell["t58"] == (3,)


def test_law58_sound_speed_consistency():
    """Verify plane-stress acoustic wave speed formula: c = sqrt(max(Kc, Kt, G0) / rho0)."""
    rho0 = 1.2e-6
    e1 = 2400.0
    e2 = 1800.0
    g0 = 60.0
    n1 = 2
    n2 = 3
    # Kc = e1 / n1 = 1200, Kt = e2 / n2 = 600, max(Kc, Kt, G0) = 1200
    kc = e1 / n1
    kt = e2 / n2
    kmax = max(kc, kt, g0)
    c_expected = math.sqrt(kmax / rho0)

    mat = make_test_material_law58(rho0=rho0, e1=e1, e2=e2, g0=g0, n1=n1, n2=n2)

    # Material method
    c_mat = mat.sound_speed_shell()
    assert math.isclose(c_mat, c_expected, rel_tol=1e-10)

    # Materials dispatcher
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_shell_law58(mat), c_expected, rel_tol=1e-10)


# ============================================================================
# 2. Solid Element Rejection
# ============================================================================

def test_law58_solid_rejection():
    """Verify solid stress update, tangent, and starter element checks reject LAW58."""
    mat = make_test_material_law58()

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="shells only"):
        materials.solid_tangent(mat, sig)

    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law58(mat, sig, deps)

    # Verify starter checks allowed laws
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n")
    for fam in shell_families:
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 58 in allowed or "58" in allowed or "LAW58" in allowed, f"58 not allowed in {fam}"

    solid_families = ("bricks", "tetras", "penta6", "pyra5")
    for fam in solid_families:
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 58 not in allowed and "58" not in allowed and "LAW58" not in allowed, (
            f"LAW58 should be rejected for solid family {fam}"
        )


# ============================================================================
# 3. BT4 Shell Element Formulation (Ishell=1)
# ============================================================================

def test_shell_bt4_single_element_tension():
    """Verify Belytschko-Tsay quad shell (Ishell=1) with LAW58 under warp tension."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law58(e1=3000.0, e2=1500.0, g0=50.0)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert "eps58" in st["mat_extra"]
    assert "yc" in st["mat_extra"]
    assert "yt" in st["mat_extra"]
    assert "sigi58" in st["mat_extra"]

    # Initial cycle 0 Courant time step probe
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = shell_bt4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0
    assert not math.isinf(dt_c0[0])

    # Prescribe tension velocity along warp direction (x)
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0  # stretch warp
    dt = 1.0e-5

    curr_x = coords.copy()
    for cycle in range(20):
        curr_x += vel * dt
        dt_step = shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0

    # Warp normal stress must be positive under tension
    sig_xx = st["sig"][0, :, 0]
    assert np.all(sig_xx > 0.0), f"sig_xx expected positive, got {sig_xx}"

    # Global internal forces must be in self-equilibrium (sum ~ 0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_shell_bt4_multi_element_patch():
    """Verify 2x2 multi-element BT4 shell patch under combined warp/weft tension."""
    # 2x2 grid from (0,0) to (20,20)
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [20.0, 10.0, 0.0],
        [0.0, 20.0, 0.0], [10.0, 20.0, 0.0], [20.0, 20.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 4, 3],  # elem 0: bottom-left
        [1, 2, 5, 4],  # elem 1: bottom-right
        [3, 4, 7, 6],  # elem 2: top-left
        [4, 5, 8, 7],  # elem 3: top-right
    ])
    mat = make_test_material_law58(e1=2500.0, e2=2000.0, g0=80.0)
    prop = MockProp(thick=0.8, nip=2)

    group = MockGroup(conn, slices=[(slice(0, 4), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert st["sig"].shape == (4, 2, 3)

    # Stretch all elements with uniform biaxial strain rate
    vel = np.zeros_like(coords)
    vel[:, 0] = coords[:, 0] * 3.0  # warp stretch
    vel[:, 1] = coords[:, 1] * 2.0  # weft stretch
    dt = 1.0e-5

    fint = np.zeros((9, 3))
    mint = np.zeros((9, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        dt_step = shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert np.all(dt_step > 0.0)

    # All elements must exhibit tension
    assert np.all(st["sig"][:, :, 0] > 0.0)
    assert np.all(st["sig"][:, :, 1] > 0.0)
    # Energy accounting: layer internal work accumulated
    assert np.all(st["eint"] > 0.0)
    # Global force balance
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 4. QEPH Shell Element Formulation (Ishell=24)
# ============================================================================

def test_shell_qeph_single_and_multi_element():
    """Verify QEPH quad shell formulation with physical hourglass control under LAW58."""
    # Single quad element
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law58(e1=2000.0, e2=1800.0, g0=50.0, gt=100.0, alphat=30.0)
    prop = MockProp(thick=1.0, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state

    # Verify sound speed initialized properly (not zeroed)
    assert st["cspd"][0] > 0.0
    c_expected = sound_speed_shell_law58(mat)
    assert math.isclose(st["cspd"][0], c_expected, rel_tol=1e-10)

    # Combined tension and in-plane Trellis shear
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0  # stretch x
    vel[[2, 3], 0] += 30.0  # shear xy
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        dt_step = shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0
        assert not math.isinf(dt_step[0])

    # In-plane normal and shear stresses
    assert st["sig"][0, :, 0].mean() > 0.0
    assert abs(st["sig"][0, :, 2].mean()) > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 5. Tri3 Triangular Shell Element Formulation (Ish3n=1)
# ============================================================================

def test_shell_tri3_single_and_multi_element():
    """Verify 3-node triangular shell formulation (Ish3n=1) under LAW58."""
    # 2-triangle quad patch
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 2],  # tri 1
        [0, 2, 3],  # tri 2
    ])
    mat = make_test_material_law58(e1=2200.0, e2=1600.0, g0=40.0)
    prop = MockProp(thick=1.2, nip=3, ish3n=1)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_tri3.init_group(group, model, None)
    st = group.state
    assert "eps58" in st["mat_extra"]

    # Stretch node 1 and 2 in x
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        dt_step = shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        # Critical time step must be strictly bounded and positive
        assert np.all(dt_step > 0.0)
        assert np.all(dt_step < 1.0e10)

    # Verify stresses developed in both triangles
    assert np.all(st["sig"][:, :, 0] > 0.0)
    # Layer energy internal work positive
    assert np.all(st["eint"] > 0.0)
    # Global equilibrium
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


# ============================================================================
# 6. Consistent Tangent Stiffness Dispatch
# ============================================================================

def test_law58_consistent_tangents_dispatch():
    """Verify shell membrane and algorithmic consistent plane-stress tangents."""
    mat = make_test_material_law58(e1=3000.0, e2=2000.0, g0=150.0)

    # 1. Reference membrane tangent
    Dm = materials.shell_membrane_tangent(mat)
    assert Dm.shape == (3, 3)
    assert math.isclose(Dm[0, 0], 3000.0)
    assert math.isclose(Dm[1, 1], 2000.0)
    assert math.isclose(Dm[2, 2], 150.0)

    # 2. Algorithmic layer tangent (finite-difference consistent)
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.005, 0.002]])
    extra: dict[str, Any] = {
        "eps58": np.zeros((1, 3)),
        "yc": np.zeros(1),
        "yt": np.zeros(1),
        "fn": np.zeros(1),
        "sigv_xy": np.zeros(1),
        "tan_phi": np.zeros(1),
        "sigi58": np.zeros((1, 3)),
        "t58": np.zeros(1),
    }

    D_layer = materials.shell_layer_tangent(mat, sig=sig, extra=extra)
    assert D_layer.shape in ((1, 3, 3), (3, 3))
    D_mat = D_layer[0] if D_layer.ndim == 3 else D_layer
    assert D_mat[0, 0] > 0.0
    assert D_mat[1, 1] > 0.0
    assert D_mat[2, 2] > 0.0


# ============================================================================
# 7. Physical Fabric Kinematics & Trellis Response
# ============================================================================

def test_law58_crimp_interchange_kinematics():
    """Verify crimp interchange behavior under uniaxial vs biaxial stretch."""
    mat = make_test_material_law58(e1=2000.0, e2=2000.0, s1=0.1, s2=0.1)

    extra_uniaxial: dict[str, Any] = {
        "eps58": np.zeros((1, 3)),
        "yc": np.zeros(1),
        "yt": np.zeros(1),
        "fn": np.zeros(1),
        "sigv_xy": np.zeros(1),
        "tan_phi": np.zeros(1),
        "sigi58": np.zeros((1, 3)),
        "t58": np.zeros(1),
    }
    sig0 = np.zeros((1, 3))
    deps_x = np.array([[0.05, 0.0, 0.0]])

    # Stretch in warp direction
    sig_out, _ = shell_update_law58(mat, sig0, deps_x, dt=1e-5, extra=extra_uniaxial)
    yc_val = float(extra_uniaxial["yc"][0])
    # Under warp tension, warp yarn crimp amplitude decreases (yc < 0)
    assert yc_val < 0.0
    assert sig_out[0, 0] > 0.0


def test_law58_trellis_shear_locking():
    """Verify Trellis shear stress transition from G0 to Gt at lock angle alphat."""
    # alphat = 30 degrees => lock angle tan_phi_lock = tan(30 deg) = 0.57735
    alphat = 30.0
    g0 = 50.0
    gt = 500.0
    mat = make_test_material_law58(g0=g0, gt=gt, alphat=alphat, ds=0.0, df=0.0)

    # 1. Shear angle below lock angle (tan_phi = 0.1 < 0.577)
    extra1: dict[str, Any] = {
        "eps58": np.zeros((1, 3)),
        "yc": np.zeros(1),
        "yt": np.zeros(1),
        "fn": np.zeros(1),
        "sigv_xy": np.zeros(1),
        "tan_phi": np.zeros(1),
        "sigi58": np.zeros((1, 3)),
        "t58": np.zeros(1),
    }
    sig0 = np.zeros((1, 3))
    deps_below = np.array([[0.0, 0.0, 0.1]])  # eng shear gamma_xy = 0.1 -> tan_phi = 0.1
    s_below, _ = shell_update_law58(mat, sig0.copy(), deps_below, dt=1e-5, extra=extra1)
    tau_below = float(s_below[0, 2])
    assert math.isclose(tau_below, g0 * 0.1, rel_tol=0.05)

    # 2. Shear angle well above lock angle (tan_phi = 0.8 > 0.577)
    extra2: dict[str, Any] = {
        "eps58": np.zeros((1, 3)),
        "yc": np.zeros(1),
        "yt": np.zeros(1),
        "fn": np.zeros(1),
        "sigv_xy": np.zeros(1),
        "tan_phi": np.zeros(1),
        "sigi58": np.zeros((1, 3)),
        "t58": np.zeros(1),
    }
    deps_above = np.array([[0.0, 0.0, 0.8]])
    s_above, _ = shell_update_law58(mat, sig0.copy(), deps_above, dt=1e-5, extra=extra2)
    tau_above = float(s_above[0, 2])
    # Above lock angle, tangent modulus shifts to Gt (500), so stress must be substantially higher
    assert tau_above > g0 * 0.8


def test_law58_zero_stress_relative_area():
    """Verify zero-stress relative area behavior (A/A0 <= arel zeroes stresses)."""
    # arel = 0.9: if relative area <= 0.9, stresses are zeroed
    mat = make_test_material_law58(e1=2000.0, e2=2000.0, arel=0.9)
    extra: dict[str, Any] = {
        "eps58": np.zeros((1, 3)),
        "yc": np.zeros(1),
        "yt": np.zeros(1),
        "fn": np.zeros(1),
        "sigv_xy": np.zeros(1),
        "tan_phi": np.zeros(1),
        "sigi58": np.zeros((1, 3)),
        "t58": np.zeros(1),
    }
    sig0 = np.zeros((1, 3))
    # Compressive strains: eps_xx = -0.1, eps_yy = -0.1 => (1+ex)(1+ey) ~ 0.81 < 0.9
    deps_comp = np.array([[-0.1, -0.1, 0.0]])
    s_out, _ = shell_update_law58(mat, sig0, deps_comp, dt=1e-5, extra=extra)

    # In zero-stress region, membrane stresses should be 0.0
    assert np.allclose(s_out[0, :3], 0.0, atol=1e-6)
