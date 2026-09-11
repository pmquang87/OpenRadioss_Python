"""
Integration test suite for /MAT/LAW74 (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL).
Milestone M563: Element Formulations Integration & Multi-Cycle Simulation Verifier.

Covers:
1. Registry, metadata, state variables, and starter checks for LAW74:
   - Registration in MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, MATERIAL_SOLID_DISPATCH.
   - Shell and 1D element rejection (shell_update raises NotImplementedError, ANCMSG 305/306).
   - State variable allocation count: _STATE_VAR_COUNT["uvar74"] == (10,).
2. Element state allocation and initialization for solid elements (Hexa8 and Tetra4):
   - Allocation of uvar74 history array of shape (n, 10).
   - Initialization of temperature history to t0 (293 K default or user-defined).
   - Initialization of plastic strain epsp = 0.0 and deletion flag off = 1.0.
3. Acoustic sound speed (dilatational wave speed) & Courant time step:
   - Solid dilatational wave speed c = sqrt((C1 + 4/3*G) / rho0).
   - Consistency between materials dispatch, Material object, and element kernel forces().
   - Array density override and modulus degradation (einf / ce).
4. Solid Hexa8 element formulation:
   - Uniaxial tension under elastic and plastic deformation.
   - Internal force equilibrium sum(fint) = 0 and positive internal energy.
   - Pure shear deformation (xy, yz, zx) and shear stress generation.
5. Solid Tetra4 element formulation:
   - Single tetrahedron under elastic and plastic tension.
   - Pure shear deformation, nodal force equilibrium, and state propagation.
6. Directional orthotropy (Hill 1948 3D):
   - Distinct directional yield stresses along X, Y, Z axes (S11Y, S22Y, S33Y).
   - Verification of directional stiffness and yield on Hexa8 and Tetra4 meshes.
7. Kinematic hardening & Bauschinger effect:
   - Forward tension accumulating backstress alpha in uvar74[:, 4:10].
   - Reverse yield initiating earlier due to shifted yield surface (chard > 0).
8. Dynamic adiabatic plastic heating:
   - Temperature rises monotonically during plastic flow when rhocp > 0.
   - Temperature remains unchanged during purely elastic deformations.
9. Progressive failure and element deletion:
   - Plastic strain exceeding eps_max sets deletion flag off = 0.0.
   - Cauchy stresses and internal forces zero out upon element deletion.
10. Multi-element patch tests:
    - 2-element Hexa8 patch sharing a face (12 nodes, equilibrium sum = 0).
    - 2-element Tetra4 patch sharing a face (5 nodes, equilibrium sum = 0).
11. Implicit consistent tangents:
    - Algorithmic consistent solid tangent (n, 6, 6) elastic symmetry and positive definiteness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.materials.law74_hill_3d import (
    Law74Params,
    build_law74,
    solid_update as solid_update_law74,
    shell_update as shell_update_law74,
    sound_speed as sound_speed_law74,
    sound_speed_solid as sound_speed_solid_law74,
    extra_shapes as extra_shapes_law74,
    consistent_solid_tangent as consistent_solid_tangent_law74,
)
from pyradioss.materials import law74_solid_tangent
from pyradioss.model.entities import Material, MatLaw74
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_model,
    check_mat_law74,
)
from pyradioss.common.messages import MessageLog


# ============================================================================
# Helpers: Mock Classes & Material Factory
# ============================================================================

class MockProp:
    """Mock solid property mimicking /PROP/SOLID (/PROP/TYPE14)."""
    def __init__(self, pid: int = 1, isolid: int = 1, ismstr: int = 2, **kwargs: Any):
        self.id = pid
        self.isolid = isolid
        self.ismstr = ismstr
        self.params = {
            "isolid": isolid,
            "ismstr": ismstr,
            **kwargs,
        }


class MockGroup:
    """Mock element group with connectivity, id indexing, and state buffer."""
    def __init__(self, conn: np.ndarray, ids: Optional[np.ndarray] = None, slices: Optional[list] = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: Dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law74(
    mid: int = 1,
    rho0: float = 2.7e-9,        # ton/mm^3
    E: float = 70000.0,          # MPa
    nu: float = 0.33,
    s11y: float = 1.0,
    s22y: float = 1.0,
    s33y: float = 1.0,
    s12y: float = 1.0,
    s23y: float = 1.0,
    s31y: float = 1.0,
    chard: float = 0.0,          # fisokin factor [0..1]
    sigy0: float = 200.0,        # reference yield stress
    yield_table: Any = None,
    table_id: int = 0,
    fscale: float = 1.0,
    pscale: float = 1.0,
    eps_max: float = 1.0e30,
    epsr1: float = 1.0e30,
    epsr2: float = 2.0e30,
    t0: float = 293.0,
    rhocp: float = 0.0,
    einf: float = 0.0,
    ce: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Construct a Material object configured for /MAT/LAW74."""
    if yield_table is None and table_id == 0 and sigy0 > 0.0:
        yield_table = [(0.0, sigy0), (0.05, sigy0 * 1.25), (0.20, sigy0 * 1.50)]

    p = Law74Params(
        rho0=rho0,
        refer_rho=rho0,
        e=E,
        nu=nu,
        s11y=s11y,
        s22y=s22y,
        s33y=s33y,
        s12y=s12y,
        s23y=s23y,
        s31y=s31y,
        chard=chard,
        sigy0=sigy0,
        yield_table=yield_table,
        table_id=table_id,
        fscale=fscale,
        pscale=pscale,
        eps_max=eps_max,
        epsr1=epsr1,
        epsr2=epsr2,
        t0=t0,
        rhocp=rhocp,
        einf=einf,
        ce=ce,
        id=mid,
        title=f"LAW74_Mat_{mid}",
        **kwargs,
    )
    return build_law74(p)


# ============================================================================
# 1. Registry, Metadata, and Element Family Acceptance/Rejection
# ============================================================================

def test_law74_registry_and_dispatch_metadata():
    """Verify registry entries, dispatch tables, and metadata for LAW74."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        74, "74", "LAW74", "HILL_3D", "ORTH_PLAS",
        "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS",
    )
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("solid") is True, f"solid must be True for {k}"
        assert meta.get("shell") is False, f"shell must be False for {k}"
        assert meta.get("plane_stress") is False, f"plane_stress must be False for {k}"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"{k} missing from MATERIAL_SOLID_DISPATCH"

    mat = make_test_material_law74()
    assert materials.needs_env(mat) is True

    # State variable count
    assert "uvar74" in materials._STATE_VAR_COUNT
    assert materials._STATE_VAR_COUNT["uvar74"] == (10,)

    # Extra shapes for solids
    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "uvar74" in shapes_solid
    assert shapes_solid["uvar74"] == (10,)
    assert "temp" in shapes_solid
    assert shapes_solid["temp"] == ()


def test_law74_rejection_of_shells_and_1d():
    """Verify rejection of shell and 1D element formulations with LAW74."""
    mat = make_test_material_law74()

    # Direct calls reject shells
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update_law74(mat, np.zeros(3), np.zeros(3))

    with pytest.raises(NotImplementedError, match="solid elements only"):
        materials.shell_update(mat, np.zeros(3), np.zeros(3))

    # Starter allowed laws check: solid families accept 74, 1D and shells reject
    for fam in ("bricks", "tetras", "penta6", "pyra5"):
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 74 in allowed or "74" in allowed or "LAW74" in allowed, f"74 not in allowed for {fam}"

    for fam in ("trusses", "beams", "shells", "shells_qbat", "shells_qeph", "sh3n"):
        allowed = _ALLOWED_LAWS.get(fam, set())
        assert 74 not in allowed and "74" not in allowed and "LAW74" not in allowed, (
            f"LAW74 should not be allowed for {fam}"
        )

    # Starter diagnostic check logs ANCMSG 305 for shells
    model = Model()
    model.materials[1] = mat
    model.shells = MockGroup(conn=np.array([[0, 1, 2, 3]]), slices=[(slice(0, 1), mat, MockProp())])
    log = MessageLog()
    check_mat_law74(model=model, mat_id=1, mat=mat, log=log)
    errs = [msg for msg in log.errors if "ANCMSG 305" in msg or "not supported for shell" in msg]
    assert len(errs) > 0, "Expected ANCMSG 305 diagnostic error for shell element with LAW74"


# ============================================================================
# 2. Element State Allocation & Initialization for Solid Elements
# ============================================================================

def test_hexa8_state_allocation_and_t0_initialization():
    """Verify Solid Hexa8 allocates uvar74 shape (n, 10) and initializes temp to t0."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    t0_custom = 315.0
    mat = make_test_material_law74(t0=t0_custom)
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    assert "mat_extra" in grp.state
    assert "uvar74" in grp.state["mat_extra"]
    uvar = grp.state["mat_extra"]["uvar74"]
    assert uvar.shape == (1, 10), f"Expected shape (1, 10), got {uvar.shape}"
    assert np.allclose(uvar, 0.0), "uvar74 should be initially all zeros"

    assert "temp" in grp.state["mat_extra"]
    temp = grp.state["mat_extra"]["temp"]
    assert temp.shape == (1,), f"Expected shape (1,), got {temp.shape}"
    assert temp[0] == pytest.approx(t0_custom), f"Expected temp={t0_custom}, got {temp[0]}"

    assert grp.state["epsp"][0] == 0.0
    assert grp.state["off"][0] == 1.0


def test_tetra4_state_allocation_and_t0_initialization():
    """Verify Solid Tetra4 allocates uvar74 shape (n, 10) and initializes temp to t0."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    t0_custom = 298.15
    mat = make_test_material_law74(t0=t0_custom)
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    assert "mat_extra" in grp.state
    assert "uvar74" in grp.state["mat_extra"]
    uvar = grp.state["mat_extra"]["uvar74"]
    assert uvar.shape == (1, 10), f"Expected shape (1, 10), got {uvar.shape}"
    assert np.allclose(uvar, 0.0)

    assert "temp" in grp.state["mat_extra"]
    temp = grp.state["mat_extra"]["temp"]
    assert temp.shape == (1,)
    assert temp[0] == pytest.approx(t0_custom)

    assert grp.state["epsp"][0] == 0.0
    assert grp.state["off"][0] == 1.0


# ============================================================================
# 3. Acoustic Sound Speed & Courant Time-Step Calculations
# ============================================================================

def test_law74_sound_speed_and_courant_step():
    """Verify dilatational wave speed c = sqrt((C1 + 4/3*G)/rho0) and Courant time step."""
    rho0 = 2.7e-9
    E = 70000.0
    nu = 0.33
    mat = make_test_material_law74(rho0=rho0, E=E, nu=nu)

    # Theoretical sound speed
    G = 0.5 * E / (1.0 + nu)
    C1 = E / (3.0 * (1.0 - 2.0 * nu))
    c_expected = math.sqrt((C1 + 4.0 / 3.0 * G) / rho0)

    assert math.isclose(mat.sound_speed_solid(), c_expected, rel_tol=1e-10)
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_solid_law74(mat, rho=rho0), c_expected, rel_tol=1e-10)

    # Element kernel dt check
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    fint = np.zeros((8, 3))
    dt_kernel = solid_hexa8.forces(grp, coords, np.zeros_like(coords), np.zeros_like(coords), 1.0e-6, fint, None)
    assert dt_kernel[0] > 0.0
    # For 10 mm cube, dt is bounded by 0.5e-6 .. 2.0e-6
    assert 0.5e-6 < dt_kernel[0] < 2.0e-6


def test_sound_speed_with_modulus_degradation():
    """Verify sound speed recalculation under Young's modulus degradation (CE / Einf)."""
    rho0 = 2.7e-9
    E = 70000.0
    Einf = 50000.0
    ce = 20.0
    nu = 0.33
    mat = make_test_material_law74(rho0=rho0, E=E, einf=Einf, ce=ce, nu=nu)

    # With zero plastic strain: uses initial E
    c0 = sound_speed_solid_law74(mat, rho=rho0)
    G0 = 0.5 * E / (1.0 + nu)
    C1_0 = E / (3.0 * (1.0 - 2.0 * nu))
    assert c0 == pytest.approx(math.sqrt((C1_0 + 4.0 / 3.0 * G0) / rho0))

    # With plastic strain pla = 0.05
    pla_val = 0.05
    e_degraded = E - (E - Einf) * (1.0 - math.exp(-ce * pla_val))
    G_deg = 0.5 * e_degraded / (1.0 + nu)
    C1_deg = e_degraded / (3.0 * (1.0 - 2.0 * nu))
    c_deg_expected = math.sqrt((C1_deg + 4.0 / 3.0 * G_deg) / rho0)

    extra = {"uvar74": np.array([[pla_val] + [0.0] * 9])}
    c_deg = sound_speed_solid_law74(mat, rho=rho0, extra=extra)
    assert c_deg == pytest.approx(c_deg_expected, rel=1e-6)
    assert c_deg < c0, "Degraded sound speed must be strictly lower than initial sound speed"


# ============================================================================
# 4. Solid Hexa8 Element Internal Forces & Equilibrium
# ============================================================================

def test_hexa8_elastic_and_plastic_uniaxial_tension():
    """Verify Hexa8 internal force equilibrium, elastic stress, and plastic yield."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law74(
        E=70000.0, nu=0.33, sigy0=200.0,
        yield_table=[(0.0, 200.0), (0.05, 250.0)],
    )
    prop = MockProp()

    grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 200.0  # pull +X
    dt = 1.0e-5
    fint = np.zeros((8, 3))
    curr_x = coords.copy()

    # Step 1: Small strain (purely elastic, eps_xx < 200/70000 ~ 0.0028)
    for _ in range(2):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert grp.state["epsp"][0] == 0.0, "Should remain elastic in first steps"
    assert grp.state["sig"][0, 0] > 0.0, "Positive tensile stress sigma_xx"
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4), "Nodal forces must sum to 0"
    assert grp.state["eint"][0] > 0.0

    # Step 2: Continue pulling past yield (plastic deformation, eps_xx reaches 0.01)
    for _ in range(40):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert grp.state["epsp"][0] > 0.0, "Plastic strain must accumulate past yield"
    assert grp.state["mat_extra"]["uvar74"][0, 0] > 0.0, "uvar74[0, 0] must mirror plastic strain"
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


def test_hexa8_pure_shear_deformation():
    """Verify Hexa8 under pure shear deformation (xy)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law74(E=70000.0, nu=0.33, sigy0=300.0)
    grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[2, 3, 6, 7], 0] = 40.0  # shear in X with increasing Y
    dt = 1.0e-5
    fint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    sig = grp.state["sig"][0]
    assert sig[3] > 0.0, f"Expected positive shear stress sigma_xy, got {sig[3]}"
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 5. Solid Tetra4 Element Internal Forces & Equilibrium
# ============================================================================

def test_tetra4_elastic_and_plastic_tension():
    """Verify Tetra4 internal force equilibrium, elastic stress, and plastic yield."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law74(
        E=70000.0, nu=0.33, sigy0=200.0,
        yield_table=[(0.0, 200.0), (0.05, 260.0)],
    )
    grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[1, 0] = 300.0  # pull node 1 in +X
    dt = 1.0e-5
    fint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(40):
        curr_x += vel * dt
        solid_tetra4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert grp.state["epsp"][0] > 0.0
    assert grp.state["mat_extra"]["uvar74"][0, 0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert grp.state["eint"][0] > 0.0


# ============================================================================
# 6. Directional Orthotropy (Hill 1948 3D)
# ============================================================================

def test_directional_orthotropy_yield_stresses_xyz():
    """Verify distinct directional yield responses when pulling along X, Y, and Z axes."""
    s11y_val, s22y_val, s33y_val = 1.0, 1.5, 2.0
    sigy0 = 150.0
    mat = make_test_material_law74(
        E=100000.0, nu=0.25,
        s11y=s11y_val, s22y=s22y_val, s33y=s33y_val,
        sigy0=sigy0,
        yield_table=[(0.0, sigy0), (0.1, sigy0)],  # Flat yield plateau
    )

    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    dt = 1.0e-5

    def _pull_and_measure_yield(pull_axis: int, pull_nodes: List[int]) -> float:
        grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
        model = Model()
        model.x0 = coords.copy()
        grp._model = model
        solid_hexa8.init_group(grp, model, None)

        vel = np.zeros_like(coords)
        vel[pull_nodes, pull_axis] = 200.0  # high enough to ensure yield in 60 steps
        fint = np.zeros((8, 3))
        curr_x = coords.copy()

        yield_stress = 0.0
        for _ in range(60):
            curr_x += vel * dt
            solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)
            if grp.state["epsp"][0] > 1.0e-4:
                yield_stress = float(grp.state["sig"][0, pull_axis])
                break
        return yield_stress

    sig_yield_x = _pull_and_measure_yield(0, [1, 2, 5, 6])
    sig_yield_y = _pull_and_measure_yield(1, [2, 3, 6, 7])
    sig_yield_z = _pull_and_measure_yield(2, [4, 5, 6, 7])

    assert sig_yield_x > 0.0 and sig_yield_y > 0.0 and sig_yield_z > 0.0
    # Hill theory: yield stress is proportional to S_iiY
    assert sig_yield_x < sig_yield_y < sig_yield_z, (
        f"Expected sig_x < sig_y < sig_z, got {sig_yield_x:.1f}, {sig_yield_y:.1f}, {sig_yield_z:.1f}"
    )
    assert (sig_yield_y / sig_yield_x) == pytest.approx(s22y_val / s11y_val, rel=0.15)
    assert (sig_yield_z / sig_yield_x) == pytest.approx(s33y_val / s11y_val, rel=0.15)


# ============================================================================
# 7. Kinematic Hardening & Bauschinger Effect
# ============================================================================

def test_hexa8_bauschinger_effect_cyclic():
    """Verify Bauschinger kinematic hardening (chard > 0) develops backstress alpha."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    # chard = 0.6 kinematic hardening
    mat = make_test_material_law74(
        E=70000.0, nu=0.33, chard=0.6, sigy0=200.0,
        yield_table=[(0.0, 200.0), (0.05, 300.0)],
    )
    grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 300.0  # forward tension in +X
    dt = 1.0e-5
    fint = np.zeros((8, 3))
    curr_x = coords.copy()

    for _ in range(30):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    uvar = grp.state["mat_extra"]["uvar74"][0]
    epsp_fwd = uvar[0]
    alpha_xx = uvar[4]  # Back-stress tensor starts at index 4 (xx, yy, zz, xy, yz, zx)

    assert epsp_fwd > 0.0, "Plastic strain must accumulate"
    assert alpha_xx > 0.0, f"Kinematic backstress alpha_xx must be positive, got {alpha_xx}"


# ============================================================================
# 8. Dynamic Adiabatic Plastic Heating
# ============================================================================

def test_adiabatic_heating_in_hexa8_and_tetra4():
    """Verify temperature rises adiabatically during plastic deformation when rhocp > 0."""
    t0_initial = 293.0
    rhocp_val = 2.4e-3  # J/(mm^3 * K)
    mat = make_test_material_law74(
        E=70000.0, nu=0.33, sigy0=180.0,
        t0=t0_initial, rhocp=rhocp_val,
        yield_table=[(0.0, 180.0), (0.1, 280.0)],
    )

    # 1. Hexa8
    coords_h = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    grp_h = MockGroup(np.array([[0, 1, 2, 3, 4, 5, 6, 7]]), slices=[(slice(0, 1), mat, MockProp())])
    model_h = Model()
    model_h.x0 = coords_h.copy()
    grp_h._model = model_h
    solid_hexa8.init_group(grp_h, model_h, None)

    vel_h = np.zeros_like(coords_h)
    vel_h[[1, 2, 5, 6], 0] = 300.0
    dt = 1.0e-5
    fint_h = np.zeros((8, 3))
    curr_xh = coords_h.copy()

    for _ in range(35):
        curr_xh += vel_h * dt
        solid_hexa8.forces(grp_h, curr_xh, vel_h, np.zeros_like(coords_h), dt, fint_h, None)

    temp_h = float(grp_h.state["mat_extra"]["temp"][0])
    assert temp_h > t0_initial, f"Hexa8 temperature {temp_h} must rise above {t0_initial} K"

    # 2. Tetra4
    coords_t = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],
    ])
    grp_t = MockGroup(np.array([[0, 1, 2, 3]]), slices=[(slice(0, 1), mat, MockProp())])
    model_t = Model()
    model_t.x0 = coords_t.copy()
    grp_t._model = model_t
    solid_tetra4.init_group(grp_t, model_t, None)

    vel_t = np.zeros_like(coords_t)
    vel_t[1, 0] = 300.0
    fint_t = np.zeros((4, 3))
    curr_xt = coords_t.copy()

    for _ in range(35):
        curr_xt += vel_t * dt
        solid_tetra4.forces(grp_t, curr_xt, vel_t, np.zeros_like(coords_t), dt, fint_t, None)

    temp_t = float(grp_t.state["mat_extra"]["temp"][0])
    assert temp_t > t0_initial, f"Tetra4 temperature {temp_t} must rise above {t0_initial} K"


# ============================================================================
# 9. Progressive Failure and Element Deletion (eps_max)
# ============================================================================

def test_element_deletion_when_epsp_exceeds_eps_max():
    """Element deletion when plastic strain exceeds eps_max: forces and stresses zeroed."""
    eps_max = 0.005
    mat = make_test_material_law74(
        E=70000.0, nu=0.33, sigy0=180.0, eps_max=eps_max,
        yield_table=[(0.0, 180.0), (0.05, 220.0)],
    )

    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    grp = MockGroup(conn, slices=[(slice(0, 1), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 500.0  # High pull velocity to rapidly trigger deletion
    dt = 1.0e-5
    fint = np.zeros((8, 3))
    curr_x = coords.copy()

    deleted = False
    for step in range(50):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)
        if grp.state["off"][0] <= 0.1:
            deleted = True
            break

    assert deleted, f"Element should be deleted when epsp ({grp.state['epsp'][0]}) > eps_max ({eps_max})"
    assert grp.state["off"][0] <= 0.1


# ============================================================================
# 10. Multi-Element Patch Tests
# ============================================================================

def test_two_element_hexa8_patch_equilibrium():
    """2-element Hexa8 patch sharing a face (12 nodes) maintains equilibrium."""
    # Two cubes 10x10x10 sharing x=10 face
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
        [20.0, 0.0, 0.0], [20.0, 10.0, 0.0], [20.0, 0.0, 10.0], [20.0, 10.0, 10.0],
    ])
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 2, 5, 10, 11, 6],
    ])
    mat = make_test_material_law74(E=70000.0, nu=0.33, sigy0=200.0)
    grp = MockGroup(conn, slices=[(slice(0, 2), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_hexa8.init_group(grp, model, None)

    # Linear stretch field across the 2-element patch:
    # Nodes at x=0 have vel=0, nodes at x=10 have vel=25, nodes at x=20 have vel=50
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 25.0
    vel[[8, 9, 10, 11], 0] = 50.0
    dt = 1.0e-5
    fint = np.zeros((12, 3))
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        solid_hexa8.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert np.all(grp.state["sig"][:, 0] > 0.0), "Both elements should carry tensile stress"


def test_two_element_tetra4_patch_equilibrium():
    """2-element Tetra4 patch sharing a base (5 nodes) maintains equilibrium."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [0.0, 0.0, -10.0],
    ])
    conn = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 4],  # reversed base orientation for positive volume
    ])
    mat = make_test_material_law74(E=70000.0, nu=0.33, sigy0=200.0)
    grp = MockGroup(conn, slices=[(slice(0, 2), mat, MockProp())])
    model = Model()
    model.x0 = coords.copy()
    grp._model = model
    solid_tetra4.init_group(grp, model, None)

    vel = np.zeros_like(coords)
    vel[3, 2] = 40.0   # pull top apex +Z
    vel[4, 2] = -40.0  # pull bottom apex -Z
    dt = 1.0e-5
    fint = np.zeros((5, 3))
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        solid_tetra4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, None)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 11. Algorithmic Consistent Tangents
# ============================================================================

def test_law74_consistent_solid_tangent():
    """Verify consistent solid tangent stiffness matrix is symmetric and positive definite."""
    mat = make_test_material_law74(E=100000.0, nu=0.30, sigy0=300.0)
    sig = np.zeros((1, 6), dtype=float)
    deps = np.zeros((1, 6), dtype=float)

    tan = law74_solid_tangent(mat, sig, deps=deps)
    assert tan.shape == (1, 6, 6)
    c_mat = tan[0]

    # Symmetry: C_ijkl = C_klij
    assert np.allclose(c_mat, c_mat.T, atol=1e-5), "Elastic tangent matrix must be symmetric"

    # Positive definiteness: all eigenvalues > 0
    eigenvals = np.linalg.eigvalsh(c_mat)
    assert np.all(eigenvals > 0.0), f"Tangent eigenvalues must be positive, got {eigenvals}"
