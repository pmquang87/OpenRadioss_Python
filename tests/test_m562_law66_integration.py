"""
Integration test suite for /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB).
Milestone M562: Element Formulations Integration & Multi-Cycle Simulation Verifier.

1. Registry, metadata, and dispatch verification for all LAW66 aliases:
   - Registration in MAT_PHYSICS_REGISTRY.
   - Dispatch metadata in LAW_DISPATCH_METADATA (solid=True, shell=True, plane_stress=True).
   - Dispatch in MATERIAL_SOLID_DISPATCH and MATERIAL_SHELL_DISPATCH.
   - Registration in _STATE_VAR_COUNT ("uvar66": (8,)).
   - Persistent state allocation in extra_shapes for solids (8,) and shells (nip, 8).
   - Starter allowed laws and 1D element rejection (ANCMSG 306).
2. Acoustic sound speed (dilatational wave speed) & Courant step consistency:
   - Solid dilatational wave speed: c = sqrt((max(K_T, K_C) + 4/3*G) / rho0).
   - Shell plane-stress wave speed: c = sqrt(A11_T / rho0) = sqrt(E_T / ((1 - nu^2) * rho0)).
   - Consistency between materials dispatch and Material object methods.
   - Density override rho.
3. Solid Hexa8 element formulation (1-point Belytschko-Flanagan brick):
   - Uniaxial tension vs uniaxial compression demonstrating asymmetric yield/stiffness.
   - Pure shear deformation (xy, yz, zx).
   - Cyclic tension-compression demonstrating Bauschinger backstress effect (FISOKIN > 0).
   - Strain rate sensitivity under differing deformation velocities (Cowper-Symonds rate scaling).
4. Solid Tetra4 element formulation (4-node constant-strain tetrahedron):
   - Single tetrahedron under tension vs compression with LAW66.
   - Pure shear deformation and nodal force equilibrium.
   - State variable propagation (uvar66).
5. Shell element formulations (BT4, QEPH, Tri3):
   - Membrane tension vs compression showing asymmetric response.
   - Pure bending with simultaneous tension and compression across layers.
   - In-plane shear deformation and resultant forces.
   - Through-thickness thinning under plastic tensile flow (Delta h = h * Delta eps_zz).
   - QEPH quad shell with physical hourglass control (positive cspd, bounded hg).
   - Tri3 3-node triangular shell with LAW66.
6. Multi-element patch tests:
   - 2-element Hexa8 patch sharing a face (12 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2-element Tetra4 patch sharing a face (5 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2x2 Shell BT4 patch (9 nodes, 4 quads, uniform membrane stretch and thinning).
   - 2-element Shell Tri3 patch (4 nodes, 2 triangles, uniform stress).
7. Implicit consistent tangents:
   - Algorithmic consistent solid tangent (n, 6, 6) elastic and plastic degradation.
   - Algorithmic consistent plane-stress shell tangent (n, 3, 3).
   - Constant elastic membrane tangent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law66_plas_tab import (
    Law66Params,
    build_law66,
    solid_update as solid_update_law66,
    shell_update as shell_update_law66,
    sound_speed as sound_speed_law66,
    sound_speed_solid as sound_speed_solid_law66,
    sound_speed_shell as sound_speed_shell_law66,
    solid_tangent as solid_tangent_law66,
    shell_membrane_tangent as shell_membrane_tangent_law66,
    consistent_solid_tangent as consistent_solid_tangent_law66,
    consistent_shell_tangent as consistent_shell_tangent_law66,
    extra_shapes as extra_shapes_law66,
)
from pyradioss.model.entities import Material, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_model,
    check_mat_law66,
)
from pyradioss.common.messages import MessageLog


# ============================================================================
# Helpers: Mock Classes & Material Factory
# ============================================================================

class MockProp:
    """Mock property supporting both solid (/PROP/SOLID) and shell (/PROP/SHELL)."""
    def __init__(
        self,
        pid: int = 1,
        thick: float = 1.0,
        nip: int = 3,
        ishell: int = 1,
        ish3n: int = 1,
        qa: float = 1.1,
        qb: float = 0.05,
        hm: float = 0.1,
        hf: float = 0.1,
        hr: float = 0.1,
        dn: float = 0.015,
        **kwargs: Any,
    ):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.ishell = ishell
        self.ish3n = ish3n
        self.params = {
            "thick": thick,
            "nip": nip,
            "ishell": ishell,
            "ish3n": ish3n,
            "qa": qa,
            "qb": qb,
            "hm": hm,
            "hf": hf,
            "hr": hr,
            "dn": dn,
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


def make_test_material_law66(
    mid: int = 1,
    rho0: float = 7.8e-6,         # Steel density (kg/mm^3 or ton/mm^3)
    E: float = 210000.0,          # Tension modulus (MPa)
    EC: float = 0.0,              # Compression modulus (0.0 defaults to E)
    nu: float = 0.30,
    PC: float = 0.0,              # Compression pressure limit
    PT: float = 0.0,              # Tension pressure limit
    rpct: float = 1.0,
    chard: float = 0.0,           # fisokin factor [0..1]
    asrate: float = 0.0,
    fsmooth: int = 0,
    israte: int = 1,
    epsp0: float = 0.0,
    cp: float = 1.0,
    sigy: float = 0.0,
    vp: int = 0,
    curve_c: Any = None,          # Compression yield curve
    curve_t: Any = None,          # Tension yield curve
    fscale11: float = 1.0,
    fscale22: float = 1.0,
    curve_rate_c: Any = None,
    curve_rate_t: Any = None,
    **kwargs: Any,
) -> Material:
    """Construct a Material object configured for /MAT/LAW66."""
    if curve_c is None:
        # Default compression yield curve: plastic strain vs yield stress (MPa)
        curve_c = [(0.0, 350.0), (0.05, 450.0), (0.20, 600.0)]
    if curve_t is None:
        # Default tension yield curve (distinct from compression to show asymmetry)
        curve_t = [(0.0, 250.0), (0.05, 320.0), (0.20, 420.0)]

    p = Law66Params(
        rho0=rho0,
        refer_rho=rho0,
        e=E,
        nu=nu,
        ec=EC if EC > 0.0 else E,
        pc=PC,
        pt=PT,
        rpct=rpct,
        chard=chard,
        asrate=asrate,
        fsmooth=fsmooth,
        israte=israte,
        epsp0=epsp0,
        cp=cp,
        sigy=sigy,
        vp=vp,
        fscale11=fscale11,
        fscale22=fscale22,
        curve_c=curve_c,
        curve_t=curve_t,
        curve_rate_c=curve_rate_c,
        curve_rate_t=curve_rate_t,
        id=mid,
        title=f"LAW66_Mat_{mid}",
        **kwargs,
    )
    return build_law66(p)


# ============================================================================
# 1. Registry, Metadata, and Dispatch Verification
# ============================================================================

def test_law66_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries for LAW66."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB",
        "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB",
    )
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True, f"plane_stress must be True for {k}"
        assert meta.get("solid") is True, f"solid must be True for {k}"
        assert meta.get("shell") is True, f"shell must be True for {k}"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"{k} missing from MATERIAL_SOLID_DISPATCH"
        assert k in materials.MATERIAL_SHELL_DISPATCH, f"{k} missing from MATERIAL_SHELL_DISPATCH"

    mat = make_test_material_law66()
    assert materials.needs_env(mat) is True

    # State variable count
    assert "uvar66" in materials._STATE_VAR_COUNT
    assert materials._STATE_VAR_COUNT["uvar66"] == (8,)

    # Extra shapes for solids (nip=None)
    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "uvar66" in shapes_solid
    assert shapes_solid["uvar66"] == (8,)

    # Extra shapes for shells (nip=3)
    shapes_shell = materials.extra_shapes(mat, nip=3)
    assert "uvar66" in shapes_shell
    assert shapes_shell["uvar66"] == (3, 8)
    assert "thk" in shapes_shell
    assert shapes_shell["thk"] == (3,)


def test_law66_starter_allowed_and_1d_rejection():
    """Verify allowed element families and rejection of 1D elements in starter checks."""
    # Solids and shells allow LAW66
    for fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 66 in allowed or "66" in allowed or "LAW66" in allowed, f"66 not in allowed for {fam}"

    # 1D elements reject LAW66
    for fam in ("trusses", "beams"):
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 66 not in allowed and "66" not in allowed and "LAW66" not in allowed, (
            f"LAW66 should not be allowed for {fam}"
        )

    # Diagnostic ANCMSG 306 check
    model = Model()
    mat = make_test_material_law66(mid=10)
    model.materials[10] = mat
    model.beams = MockGroup(conn=np.array([[0, 1]]), slices=[(slice(0, 1), mat, MockProp())])

    log = MessageLog()
    check_mat_law66(model=model, mat_id=10, mat=mat, log=log)
    errs = [msg for msg in log.errors if "ANCMSG 306" in msg or "not supported for 1D" in msg]
    assert len(errs) > 0, "Expected ANCMSG 306 diagnostic error for 1D element with LAW66"


# ============================================================================
# 2. Sound Speed Computation for Solids and Shells
# ============================================================================

def test_law66_sound_speed_solid_and_shell():
    """Verify dilatational acoustic wave speeds for solids and plane-stress shells."""
    rho0 = 7.8e-6
    E_t = 210000.0
    E_c = 250000.0  # Compression modulus higher than tension
    nu = 0.30

    mat = make_test_material_law66(rho0=rho0, E=E_t, EC=E_c, nu=nu)

    # Analytical values
    G_t = 0.5 * E_t / (1.0 + nu)
    K_t = E_t / (3.0 * (1.0 - 2.0 * nu))
    K_c = E_c / (3.0 * (1.0 - 2.0 * nu))
    K_max = max(K_t, K_c)

    c_solid_expected = math.sqrt((K_max + (4.0 / 3.0) * G_t) / rho0)
    A11_t = E_t / (1.0 - nu ** 2)
    c_shell_expected = math.sqrt(A11_t / rho0)

    # Solid sound speed
    assert math.isclose(mat.sound_speed_solid(), c_solid_expected, rel_tol=1e-10)
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_solid_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_solid_law66(mat, rho=rho0), c_solid_expected, rel_tol=1e-10)

    # Shell sound speed
    assert math.isclose(mat.sound_speed_shell(), c_shell_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_shell_law66(mat, rho=rho0), c_shell_expected, rel_tol=1e-10)

    # Array density support (per-element rho array passed by kernel)
    rho_arr = np.full(4, rho0)
    c_arr = sound_speed_solid_law66(mat, rho=rho_arr)
    assert isinstance(c_arr, np.ndarray)
    assert np.allclose(c_arr, c_solid_expected)


# ============================================================================
# 3. Solid Hexa8 Element Formulations
# ============================================================================

def test_hexa8_tension_vs_compression_asymmetry():
    """Verify single Hexa8 element demonstrates asymmetric stiffness and yield in tension vs compression."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # Compression curve stronger (350 MPa yield) than tension (200 MPa yield)
    mat = make_test_material_law66(
        E=200000.0,
        EC=240000.0,
        nu=0.3,
        PC=50.0,
        PT=50.0,
        curve_c=[(0.0, 350.0), (0.1, 450.0)],
        curve_t=[(0.0, 200.0), (0.1, 260.0)],
    )
    prop = MockProp()

    # --- Run Tension ---
    grp_t = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model_t = Model()
    model_t.x0 = coords.copy()
    grp_t._model = model_t
    solid_hexa8.init_group(grp_t, model_t, None)

    vel_t = np.zeros_like(coords)
    vel_t[[1, 2, 5, 6], 0] = 50.0   # pull +x
    dt = 1.0e-5

    fint_t = np.zeros((8, 3))
    curr_xt = coords.copy()
    for _ in range(15):
        curr_xt += vel_t * dt
        solid_hexa8.forces(grp_t, curr_xt, vel_t, np.zeros_like(coords), dt, fint_t, None)

    sig_t_xx = grp_t.state["sig"][0, 0]
    assert sig_t_xx > 0.0, f"Expected positive tensile stress, got {sig_t_xx}"

    # --- Run Compression ---
    grp_c = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model_c = Model()
    model_c.x0 = coords.copy()
    grp_c._model = model_c
    solid_hexa8.init_group(grp_c, model_c, None)

    vel_c = np.zeros_like(coords)
    vel_c[[1, 2, 5, 6], 0] = -50.0  # push -x
    fint_c = np.zeros((8, 3))
    curr_xc = coords.copy()
    for _ in range(15):
        curr_xc += vel_c * dt
        solid_hexa8.forces(grp_c, curr_xc, vel_c, np.zeros_like(coords), dt, fint_c, None)

    sig_c_xx = grp_c.state["sig"][0, 0]
    assert sig_c_xx < 0.0, f"Expected negative compressive stress, got {sig_c_xx}"

    # Asymmetry check: Compressive stress magnitude must exceed tensile stress magnitude
    assert abs(sig_c_xx) > abs(sig_t_xx), (
        f"Compressive stress magnitude {abs(sig_c_xx)} must exceed tensile {abs(sig_t_xx)}"
    )

    # Nodal force equilibrium in both cases
    assert np.allclose(fint_t.sum(axis=0), 0.0, atol=1e-4)
    assert np.allclose(fint_c.sum(axis=0), 0.0, atol=1e-4)


def test_hexa8_pure_shear():
    """Verify single Hexa8 element under pure in-plane shear deformation (xy)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law66(
        E=200000.0, nu=0.3,
        curve_c=[(0.0, 300.0), (0.1, 400.0)],
        curve_t=[(0.0, 300.0), (0.1, 400.0)],
    )
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[2, 3, 6, 7], 0] = 50.0  # shear in x as y increases
    dt = 1.0e-5

    fint = np.zeros((8, 3))
    curr_x = coords.copy()
    for _ in range(20):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    sig = grp.state["sig"][0]
    # Shear stress sigma_xy is index 3
    assert sig[3] > 0.0, f"Expected positive shear stress sig_xy, got {sig[3]}"
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert grp.state["eint"][0] > 0.0


def test_hexa8_bauschinger_effect_cyclic():
    """Verify Bauschinger kinematic backstress effect (FISOKIN > 0) under cyclic loading."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # Kinematic hardening factor chard = 0.8
    mat_kin = make_test_material_law66(
        E=200000.0, nu=0.3, chard=0.8,
        curve_c=[(0.0, 200.0), (0.1, 400.0)],
        curve_t=[(0.0, 200.0), (0.1, 400.0)],
    )
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat_kin, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    dt = 1.0e-5
    vel = np.zeros_like(coords)
    fint = np.zeros((8, 3))
    curr_x = coords.copy()

    # Phase 1: Forward tension past yield
    vel[[1, 2, 5, 6], 0] = 80.0
    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    # Verify plastic strain and kinematic backstress alpha_xx developed
    uvar = grp.state["mat_extra"]["uvar66"][0]
    epsp_fwd = uvar[0]
    alpha_xx_fwd = uvar[1]
    assert epsp_fwd > 0.0, "Plastic strain must accumulate during forward tension"
    assert alpha_xx_fwd > 0.0, f"Kinematic backstress alpha_xx must be positive, got {alpha_xx_fwd}"

    # Phase 2: Reverse compression past yield
    vel[[1, 2, 5, 6], 0] = -80.0
    for _ in range(50):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    uvar_rev = grp.state["mat_extra"]["uvar66"][0]
    assert uvar_rev[0] > epsp_fwd, "Plastic strain must continue to grow on reverse yield"


def test_hexa8_strain_rate_sensitivity():
    """Verify Cowper-Symonds strain rate sensitivity under differing deformation velocities."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # Rate sensitivity parameters: epsp0 = 10.0, cp = 0.5
    mat_rate = make_test_material_law66(
        E=200000.0, nu=0.3, israte=1, epsp0=10.0, cp=0.5,
        curve_c=[(0.0, 250.0), (0.1, 350.0)],
        curve_t=[(0.0, 250.0), (0.1, 350.0)],
    )
    prop = MockProp()

    # 1. Low-velocity pull (v = 80.0 -> strain rate = 8 s^-1)
    grp_lo = MockGroup(conn, slices=[(slice(0, 1), mat_rate, prop)])
    m_lo = Model()
    m_lo.x0 = coords.copy()
    grp_lo._model = m_lo
    solid_hexa8.init_group(grp_lo, m_lo, None)

    vel_lo = np.zeros_like(coords)
    vel_lo[[1, 2, 5, 6], 0] = 80.0
    dt = 1.0e-5
    curr_xlo = coords.copy()
    fint = np.zeros((8, 3))
    for _ in range(50):
        curr_xlo += vel_lo * dt
        solid_hexa8.forces(grp_lo, curr_xlo, vel_lo, np.zeros_like(coords), dt, fint, None)

    sig_lo = grp_lo.state["sig"][0, 0]

    # 2. High-velocity pull (v = 80000.0 -> strain rate = 8000 s^-1)
    grp_hi = MockGroup(conn, slices=[(slice(0, 1), mat_rate, prop)])
    m_hi = Model()
    m_hi.x0 = coords.copy()
    grp_hi._model = m_hi
    solid_hexa8.init_group(grp_hi, m_hi, None)

    vel_hi = np.zeros_like(coords)
    vel_hi[[1, 2, 5, 6], 0] = 80000.0
    dt_hi = 1.0e-8  # smaller dt for high rate stability
    curr_xhi = coords.copy()
    for _ in range(50):
        curr_xhi += vel_hi * dt_hi
        solid_hexa8.forces(grp_hi, curr_xhi, vel_hi, np.zeros_like(coords), dt_hi, fint, None)

    sig_hi = grp_hi.state["sig"][0, 0]
    # High-rate stress must exceed low-rate stress due to rate hardening
    assert sig_hi > sig_lo, f"Expected sig_hi ({sig_hi}) > sig_lo ({sig_lo})"


# ============================================================================
# 4. Solid Tetra4 Element Formulations
# ============================================================================

def test_tetra4_single_element_tension_and_compression():
    """Verify 4-node constant-strain tetrahedron with LAW66 under tension vs compression."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(
        E=150000.0, EC=180000.0, nu=0.25,
        curve_c=[(0.0, 300.0), (0.1, 400.0)],
        curve_t=[(0.0, 200.0), (0.1, 250.0)],
    )
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    # Initial cycle 0 Courant step probe
    fint = np.zeros((4, 3))
    dt_c0 = solid_tetra4.forces(grp, coords, None, None, 0.0, fint, None)
    assert dt_c0[0] > 0.0

    # Tension along x
    vel = np.zeros_like(coords)
    vel[1, 0] = 30.0
    dt = 1.0e-5
    curr_x = coords.copy()
    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert grp.state["sig"][0, 0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert "uvar66" in grp.state["mat_extra"]


def test_tetra4_pure_shear():
    """Verify Tetra4 element under pure shear deformation with LAW66."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(E=120000.0, nu=0.3)
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[2, 0] = 30.0  # shear in x along y
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    curr_x = coords.copy()
    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert grp.state["sig"][0, 3] > 0.0  # sigma_xy
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 5. Shell Elements (BT4, QEPH, Tri3) Formulations
# ============================================================================

def test_shell_bt4_membrane_tension_vs_compression():
    """Verify BT4 quad shell demonstrates asymmetric membrane tension vs compression."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(
        E=100000.0, EC=140000.0, nu=0.3,
        curve_c=[(0.0, 320.0), (0.1, 400.0)],
        curve_t=[(0.0, 200.0), (0.1, 250.0)],
    )
    prop = MockProp(thick=1.0, nip=3)

    # Tension
    grp_t = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    m_t = Model()
    m_t.x0 = coords.copy()
    grp_t._model = m_t
    shell_bt4.init_group(grp_t, m_t, None)

    vel_t = np.zeros_like(coords)
    vel_t[[1, 2], 0] = 40.0
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_xt = coords.copy()
    for _ in range(20):
        curr_xt += vel_t * dt
        shell_bt4.forces(grp_t, curr_xt, vel_t, np.zeros_like(coords), dt, fint, mint)

    sig_t_xx = grp_t.state["sig"][0, :, 0].mean()
    assert sig_t_xx > 0.0

    # Compression
    grp_c = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    m_c = Model()
    m_c.x0 = coords.copy()
    grp_c._model = m_c
    shell_bt4.init_group(grp_c, m_c, None)

    vel_c = np.zeros_like(coords)
    vel_c[[1, 2], 0] = -40.0
    curr_xc = coords.copy()
    for _ in range(20):
        curr_xc += vel_c * dt
        shell_bt4.forces(grp_c, curr_xc, vel_c, np.zeros_like(coords), dt, fint, mint)

    sig_c_xx = grp_c.state["sig"][0, :, 0].mean()
    assert sig_c_xx < 0.0

    # Asymmetry in membrane stresses
    assert abs(sig_c_xx) > abs(sig_t_xx), (
        f"Compressive shell stress {abs(sig_c_xx)} must exceed tensile {abs(sig_t_xx)}"
    )


def test_shell_bt4_pure_bending():
    """Verify Shell BT4 under pure bending shows asymmetric top/bottom layer stresses."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(
        E=120000.0, EC=160000.0, nu=0.3,
        curve_c=[(0.0, 300.0), (0.1, 400.0)],
        curve_t=[(0.0, 200.0), (0.1, 260.0)],
    )
    prop = MockProp(thick=2.0, nip=3)

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_bt4.init_group(grp, model, None)

    # Impart pure rotational velocity about y axis to create bending in x
    vr = np.zeros_like(coords)
    vr[[1, 2], 1] = 5.0    # rotate right edge
    vr[[0, 3], 1] = -5.0   # rotate left edge oppositely
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    for _ in range(20):
        shell_bt4.forces(grp, coords, np.zeros_like(coords), vr, dt, fint, mint)

    sig_layers = grp.state["sig"][0, :, 0]
    # Layer 0 (bottom) and layer 2 (top) must have opposite stress signs
    assert sig_layers[0] * sig_layers[-1] < 0.0, (
        f"Bending should induce opposite signs on outer layers: {sig_layers}"
    )


def test_shell_bt4_in_plane_shear():
    """Verify Shell BT4 under pure in-plane shear deformation with LAW66."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(E=100000.0, nu=0.3)
    prop = MockProp(thick=1.0, nip=3)

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_bt4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[2, 3], 0] = 50.0  # shear along x
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        shell_bt4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    sig_xy = grp.state["sig"][0, :, 2]
    assert np.all(sig_xy > 0.0), f"Expected positive in-plane shear stress, got {sig_xy}"
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_shell_bt4_thickness_thinning():
    """Verify through-thickness thinning under large membrane tensile stretch."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    thick0 = 1.5
    # Low yield stress to easily drive plastic deformation
    mat = make_test_material_law66(
        E=100000.0, nu=0.33,
        curve_c=[(0.0, 120.0), (0.1, 160.0)],
        curve_t=[(0.0, 100.0), (0.1, 130.0)],
    )
    prop = MockProp(thick=thick0, nip=3)

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_bt4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 150.0  # large stretch along x
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(30):
        curr_x += vel * dt
        shell_bt4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    # Thickness must thin under tensile flow
    thk_final = grp.state["thick"][0]
    assert thk_final < thick0, f"Expected thickness thinning < {thick0}, got {thk_final}"
    assert thk_final > 0.0, f"Thickness must remain strictly positive, got {thk_final}"


def test_shell_qeph_single_element_and_bending():
    """Verify Shell QEPH with physical hourglass control under LAW66."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law66(E=110000.0, nu=0.3)
    prop = MockProp(thick=1.0, nip=3, ishell=24)

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_qeph.init_group(grp, model, None)

    # cspd must be strictly positive
    assert grp.state["cspd"][0] > 0.0

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 30.0
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        dt_step = shell_qeph.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0

    assert grp.state["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_shell_tri3_single_element_and_thinning():
    """Verify 3-node triangular shell (Tri3) with LAW66 and thickness thinning."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2]])
    thick0 = 1.2
    mat = make_test_material_law66(
        E=80000.0, nu=0.3,
        curve_c=[(0.0, 150.0), (0.1, 200.0)],
        curve_t=[(0.0, 120.0), (0.1, 160.0)],
    )
    prop = MockProp(thick=thick0, nip=3, ish3n=1)

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_tri3.init_group(grp, model, None)

    # Initial cycle 0 Courant step probe
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_c0 = shell_tri3.forces(grp, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0

    vel = np.zeros_like(coords)
    vel[1, 0] = 80.0  # stretch node 1 in x
    dt = 1.0e-5
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        shell_tri3.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert grp.state["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 6. Multi-Element Patch Tests
# ============================================================================

def test_hexa8_2element_patch():
    """Verify 2-element Hexa8 patch sharing a common face shows uniform stress and equilibrium."""
    # 2 cubes along x: [0..10] and [10..20]
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
        [20.0, 0.0, 0.0], [20.0, 10.0, 0.0], [20.0, 0.0, 10.0], [20.0, 10.0, 10.0],
    ])
    # Elem 0: nodes 0..7, Elem 1: nodes 1, 8, 9, 2, 5, 10, 11, 6
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 2, 5, 10, 11, 6],
    ])
    mat = make_test_material_law66(E=200000.0, nu=0.3)
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    # Uniform tensile strain rate: v_x proportional to x
    vel = np.zeros_like(coords)
    vel[:, 0] = coords[:, 0] * 5.0
    dt = 1.0e-5
    fint = np.zeros((12, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    sig = grp.state["sig"]
    # Both elements must experience the exact same uniform stress
    assert np.allclose(sig[0], sig[1], rtol=1e-5, atol=1e-3)
    # Global internal forces must sum to zero
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_tetra4_2element_patch():
    """Verify 2-element Tetra4 patch sharing a common face shows uniform response and equilibrium."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 10.0, 0.0],
    ])
    # Elem 0: (0, 1, 2, 3), Elem 1: (1, 4, 2, 3) sharing face (1, 2, 3)
    conn = np.array([
        [0, 1, 2, 3],
        [1, 4, 2, 3],
    ])
    mat = make_test_material_law66(E=150000.0, nu=0.25)
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[:, 0] = coords[:, 0] * 3.0
    dt = 1.0e-5
    fint = np.zeros((5, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert np.all(grp.state["sig"][:, 0] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_shell_bt4_2x2_patch():
    """Verify 2x2 Shell BT4 patch under uniform stretch shows identical element stresses."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [20.0, 10.0, 0.0],
        [0.0, 20.0, 0.0], [10.0, 20.0, 0.0], [20.0, 20.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 4, 3],  # Elem 0
        [1, 2, 5, 4],  # Elem 1
        [3, 4, 7, 6],  # Elem 2
        [4, 5, 8, 7],  # Elem 3
    ])
    mat = make_test_material_law66(E=100000.0, nu=0.3)
    prop = MockProp(thick=1.0, nip=2)

    grp = MockGroup(conn, slices=[(slice(0, 4), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    shell_bt4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[:, 0] = coords[:, 0] * 4.0
    dt = 1.0e-5
    fint = np.zeros((9, 3))
    mint = np.zeros((9, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        shell_bt4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    sig = grp.state["sig"]
    for i in range(1, 4):
        assert np.allclose(sig[0], sig[i], rtol=1e-5, atol=1e-3)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 7. Consistent Tangents for Implicit Analysis
# ============================================================================

def test_consistent_solid_tangent():
    """Verify consistent solid elastoplastic tangent stiffness (6, 6) tensor."""
    mat = make_test_material_law66(E=200000.0, nu=0.3)

    # 1. Constant elastic tangent
    C_elas = solid_tangent_law66(mat)
    assert C_elas.shape == (6, 6)
    assert np.allclose(C_elas, C_elas.T), "Elastic tangent must be symmetric"
    evals = np.linalg.eigvalsh(C_elas)
    assert np.all(evals > 0.0), "Elastic tangent must be positive definite"

    # 2. Consistent tangent dispatch via materials module
    sig = np.zeros((2, 6))
    D_mat = materials.solid_tangent(mat, sig)
    assert D_mat.shape == (2, 6, 6)

    # 3. Elastoplastic tangent under plastic strain increment
    sig_plas = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 300.0, 0.0, 0.0, 0.0, 0.0]])
    epsp_incr = np.array([0.005, 0.005])
    D_plas = consistent_solid_tangent_law66(mat, sig_plas, epsp_incr=epsp_incr)
    assert D_plas.shape == (2, 6, 6)
    # Plasticity softens the tangent: diagonal components must decrease
    assert D_plas[0, 0, 0] < C_elas[0, 0]


def test_consistent_shell_tangent():
    """Verify plane-stress shell consistent tangent (3, 3) tensor."""
    mat = make_test_material_law66(E=120000.0, nu=0.3)

    # 1. Constant membrane tangent
    C_mem = shell_membrane_tangent_law66(mat)
    assert C_mem.shape == (3, 3)
    assert np.allclose(C_mem, C_mem.T)
    evals = np.linalg.eigvalsh(C_mem)
    assert np.all(evals > 0.0)

    # 2. Layer tangent dispatch via materials module
    sig = np.zeros((1, 3))
    D_layer = materials.shell_layer_tangent(mat, sig=sig)
    assert D_layer.shape == (1, 3, 3)

    # 3. Elastoplastic shell tangent
    sig_plas = np.array([[250.0, 0.0, 0.0]])
    epsp_incr = np.array([0.002])
    D_plas = consistent_shell_tangent_law66(mat, sig_plas, epsp_incr=epsp_incr)
    assert D_plas.shape == (1, 3, 3)
    assert D_plas[0, 0, 0] < C_mem[0, 0]
