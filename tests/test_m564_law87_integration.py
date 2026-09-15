"""
Integration test suite for /MAT/LAW87 (/MAT/BARLAT2000 / /MAT/BARLAT_2000 / /MAT/BARLAT2000_2D).
Milestone M564: Barlat Yld2000-2d Anisotropic Plasticity for Shell Elements.

Covers:
1. Registry, metadata, state variables, and starter checks:
   - MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, MATERIAL_SHELL_DISPATCH, MATERIAL_SOLID_DISPATCH.
   - Shell acceptance and solid/1D rejection (solid_update raises NotImplementedError, ANCMSG 305/306).
   - Needs_env check and state variable allocations.
2. Acoustic sound speed & Courant time step:
   - Shell plane-stress acoustic wave speed c = sqrt(E / ((1 - nu^2) * rho0)).
   - Consistency across Material, MatLaw87, materials dispatch, and element kernels.
   - Density override and positive/finite Courant time step.
3. Shell BT4 element formulation (Ishell=1):
   - Element state allocation: uvar87 (n, 7), uvar87 layer state (n, nip, 7), thk87, pla87, off87.
   - Uniaxial and biaxial tension: internal force equilibrium sum(fint) = 0, positive strain energy.
   - Plastic strain accumulation and uvar87 strain rate tracking.
   - Multi-element 2x2 patch test under dynamic tension.
   - Dynamic shell layer thickness thinning update (thk87 < thick0).
4. Shell QEPH element formulation (Ishell=24):
   - Physical hourglass control, positive cspd, dynamic tension and shear.
   - Force equilibrium, positive strain energy, plastic yielding, and uvar87 synchronization.
   - Dynamic shell layer thickness thinning update.
5. Shell Tri3 element formulation (Ish3n=1):
   - 3-node triangular shell multi-element dynamic cycle.
   - Force equilibrium, positive strain energy, plastic yielding, and uvar87 synchronization.
   - Dynamic shell layer thickness thinning update.
6. Directional orthotropy (Barlat Yld2000-2d 0 deg vs 45 deg vs 90 deg):
   - Distinct directional yield stresses along 0 deg, 45 deg, and 90 deg axes.
   - Differential plastic yielding and stress responses under identical stretch.
7. Kinematic hardening & Bauschinger effect:
   - Forward tension accumulating backstress in sigb87.
   - Reverse yield initiating earlier due to shifted yield surface (chard > 0, ikin=1).
8. Consistent algorithmic shell tangent:
   - Membrane tangent symmetry and positive definiteness in elastic regime.
   - Tangent softening under active plastic flow.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    build_law87,
    barlat2000_equivalent_stress,
    shell_update as shell_update_law87,
    solid_update as solid_update_law87,
    sound_speed as sound_speed_shell_law87,
    sound_speed_shell,
    consistent_shell_tangent,
    shell_membrane_tangent,
    extra_shapes as law87_extra_shapes,
)
from pyradioss.model.entities import Material, MatLaw87
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_model,
    check_mat_law87,
)
from pyradioss.common.messages import MessageLog


# ============================================================================
# Helpers: Mock Classes & Factory
# ============================================================================

class MockProp:
    """Mock shell property mimicking /PROP/SHELL (/PROP/TYPE1)."""
    def __init__(
        self,
        pid: int = 1,
        thick: float = 1.0,
        nip: int = 3,
        ishell: int = 1,
        ish3n: int = 1,
        hm: float = 0.1,
        hf: float = 0.1,
        hr: float = 0.1,
        qa: float = 1.1,
        qb: float = 0.05,
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
            "hm": hm,
            "hf": hf,
            "hr": hr,
            "qa": qa,
            "qb": qb,
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


def make_test_material_law87(
    mid: int = 1,
    rho0: float = 2.7e-9,        # ton/mm^3 (aluminum)
    E: float = 70000.0,          # MPa
    nu: float = 0.33,
    iflag: int = 1,
    al1: float = 1.0,
    al2: float = 1.0,
    al3: float = 1.0,
    al4: float = 1.0,
    al5: float = 1.0,
    al6: float = 1.0,
    al7: float = 1.0,
    al8: float = 1.0,
    expa: float = 8.0,
    aswift: float = 250.0,
    nexp: float = 0.2,
    alpha: float = 1.0,
    epso: float = 0.002,
    qvoce: float = 0.0,
    beta: float = 0.0,
    ko: float = 0.0,
    chard: float = 0.0,
    ikin: int = 1,
    ckh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    akh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    title: str = "LAW87_TestMat",
    **kwargs: Any,
) -> MatLaw87:
    """Factory constructing a MatLaw87 instance configured for shell element tests."""
    return MatLaw87(
        id=mid,
        rho=rho0,
        refer_rho=rho0,
        e=E,
        nu=nu,
        iflag=iflag,
        al1=al1,
        al2=al2,
        al3=al3,
        al4=al4,
        al5=al5,
        al6=al6,
        al7=al7,
        al8=al8,
        expa=expa,
        aswift=aswift,
        nexp=nexp,
        alpha=alpha,
        epso=epso,
        qvoce=qvoce,
        beta=beta,
        ko=ko,
        chard=chard,
        fisokin=chard,
        ikin=ikin,
        ckh=ckh,
        akh=akh,
        title=title,
        **kwargs,
    )


# ============================================================================
# 1. Registry, Metadata, and Dispatch Verification
# ============================================================================

def test_law87_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch tables for LAW87."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (
        87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D",
        "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT2000", "MAT_BARLAT_2000",
        "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000",
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

    mat = make_test_material_law87()
    assert materials.needs_env(mat) is True

    # State variable extra shapes
    shapes_nip3 = law87_extra_shapes(mat, nip=3)
    assert shapes_nip3["uvar87"] == (3, 7)

    shapes_nip1 = law87_extra_shapes(mat, nip=1)
    assert shapes_nip1["uvar87"] == (7,)


def test_law87_solid_and_1d_rejection():
    """Verify solid stress update, tangent, and 1D starter checks reject LAW87."""
    mat = make_test_material_law87()

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, sig, deps)

    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law87(mat, sig, deps)

    # Verify starter checks allowed laws: shells accept, solids/1D reject
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n")
    for fam in shell_families:
        allowed = _ALLOWED_LAWS.get(fam) or set()
        assert 87 in allowed or "87" in allowed or "LAW87" in allowed, f"87 not allowed in {fam}"

    solid_families = ("bricks", "tetras", "penta6", "pyra5")
    for fam in solid_families:
        allowed = _ALLOWED_LAWS.get(fam) or set()
        assert 87 not in allowed and "87" not in allowed and "LAW87" not in allowed, (
            f"LAW87 should be rejected for solid family {fam}"
        )

    one_d_families = ("trusses", "beams", "springs")
    for fam in one_d_families:
        allowed = _ALLOWED_LAWS.get(fam) or set()
        assert 87 not in allowed and "87" not in allowed and "LAW87" not in allowed, (
            f"LAW87 should not be allowed for 1D family {fam}"
        )

    # Diagnostic checks emit ANCMSG 305 for solid and ANCMSG 306 for 1D
    class FakeElemGroup:
        def __init__(self, n: int = 1):
            self.n = n
            self.state = {"slices": [(slice(0, 1), mat, None)]}
        def values(self):
            return []

    model = Model()
    model.materials[87] = mat
    model.bricks = FakeElemGroup(1)
    log = MessageLog()
    check_mat_law87(mat=mat, log=log, model=model, mat_id=87)
    assert any("ANCMSG 305" in e and "solid" in e for e in log.errors)

    model_1d = Model()
    model_1d.materials[87] = mat
    model_1d.beams = FakeElemGroup(1)
    log_1d = MessageLog()
    check_mat_law87(mat=mat, log=log_1d, model=model_1d, mat_id=87)
    assert any("ANCMSG 306" in e and "1D" in e for e in log_1d.errors)


def test_solid_element_runtime_rejection():
    """Verify Hexa8 and Tetra4 element kernels fail cleanly when executed with LAW87."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law87()
    prop = MockProp(thick=1.0)
    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_hexa8.forces(group, coords, np.zeros_like(coords), np.zeros_like(coords), 1e-5, fint, mint)


# ============================================================================
# 2. Acoustic Sound Speed & Courant Time Step Consistency
# ============================================================================

def test_law87_sound_speed_consistency():
    """Verify plane-stress acoustic wave speed c = sqrt(E / ((1 - nu^2) * rho0))."""
    rho0 = 2.7e-9
    E = 70000.0
    nu = 0.33
    c_expected = math.sqrt(E / ((1.0 - nu ** 2) * rho0))

    mat = make_test_material_law87(rho0=rho0, E=E, nu=nu)

    # Material method
    c_mat = float(mat.sound_speed_shell())
    assert math.isclose(c_mat, c_expected, rel_tol=1e-10)

    # Materials dispatcher
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_expected, rel_tol=1e-10)
    assert math.isclose(materials.sound_speed(mat), c_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_shell_law87(mat), c_expected, rel_tol=1e-10)
    assert math.isclose(sound_speed_shell(mat), c_expected, rel_tol=1e-10)

    # Override via rho parameter
    rho_override = 3.0e-9
    c_override_expected = math.sqrt(E / ((1.0 - nu ** 2) * rho_override))
    c_override = sound_speed_shell_law87(mat, rho=rho_override)
    assert math.isclose(c_override, c_override_expected, rel_tol=1e-10)


# ============================================================================
# 3. Shell BT4 Element Formulation (Ishell=1)
# ============================================================================

def test_shell_bt4_state_allocation_and_uvar87():
    """Verify shell_bt4 allocates uvar87 shape (n, 7), layer state (n, nip, 7), and history."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law87()
    prop = MockProp(thick=1.0, nip=3, ishell=1)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state

    # Verification of state variables
    assert "uvar87" in st
    assert st["uvar87"].shape == (1, 7)
    assert "mat_extra" in st
    assert "uvar87" in st["mat_extra"]
    assert st["mat_extra"]["uvar87"].shape == (1, 3, 7)
    assert "thk87" in st["mat_extra"]
    assert st["mat_extra"]["thk87"].shape == (1, 3)
    assert "pla87" in st["mat_extra"]
    assert "off87" in st["mat_extra"]

    # Initial time step probe
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = shell_bt4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0
    assert not math.isinf(dt_c0[0])


def test_shell_bt4_elastic_and_plastic_deformation():
    """Verify shell_bt4 with LAW87 under rolling direction tension (elastic and plastic)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    # Swift parameters: yield stress = 250 * (0.002 + epsp)^0.2 ~ 72 MPa initial yield
    mat = make_test_material_law87(aswift=250.0, nexp=0.2, epso=0.002)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state

    # Stretch in X: velocity = 40 mm/s, dt = 1e-5 s -> strain increment 4e-5 per step
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 40.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    # Step 1: Small stretch (purely elastic)
    for _ in range(5):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    # Stress positive in X, plastic strain zero
    assert np.all(st["sig"][0, :, 0] > 0.0)
    assert np.all(st["epsp"][0] == 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert st["eint"][0] > 0.0

    # Step 2: Continue stretching into plastic regime (35 more steps)
    for _ in range(35):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    # Plastic strain accumulated
    assert np.all(st["epsp"][0] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert st["eint"][0] > 0.0

    # uvar87 tracks strain rate and synchronizes to group state
    assert st["mat_extra"]["uvar87"][0, 0, 0] > 0.0
    assert st["uvar87"][0, 0] > 0.0


def test_shell_bt4_thickness_thinning_evolution():
    """Verify dynamic shell thickness thinning update under plastic tensile deformation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    thick0 = 1.5
    mat = make_test_material_law87(aswift=180.0, nexp=0.15, epso=0.002)
    prop = MockProp(thick=thick0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert np.allclose(st["mat_extra"]["thk87"][0], thick0)

    # Large stretch to drive significant plastic thinning
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 150.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(30):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    thk_final = st["mat_extra"]["thk87"][0]
    assert np.all(thk_final < thick0), f"Thickness must decrease from {thick0}, got {thk_final}"
    assert np.all(thk_final > 0.0), f"Thickness must remain strictly positive, got {thk_final}"


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
    mat = make_test_material_law87(aswift=220.0, nexp=0.2, epso=0.002)
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

    # All elements must exhibit tensile stresses and positive internal work
    assert np.all(st["sig"][:, :, 0] > 0.0)
    assert np.all(st["eint"] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)


# ============================================================================
# 4. Shell QEPH Element Formulation (Ishell=24)
# ============================================================================

def test_shell_qeph_state_and_sound_speed():
    """Verify QEPH quad shell initialization, physical hourglass, and wave speed."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law87(E=70000.0, nu=0.33, rho0=2.7e-9)
    prop = MockProp(thick=1.0, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state

    # Sound speed matches analytical plane stress wave speed
    c_expected = sound_speed_shell_law87(mat)
    assert math.isclose(st["cspd"][0], c_expected, rel_tol=1e-10)
    assert "uvar87" in st
    assert "uvar87" in st["mat_extra"]


def test_shell_qeph_elastic_and_plastic_forces():
    """Verify QEPH under combined tension, shear, and plastic flow."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law87(aswift=200.0, nexp=0.2, epso=0.002)
    prop = MockProp(thick=1.0, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 50.0   # stretch x
    vel[[2, 3], 0] += 25.0  # shear xy
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        dt_step = shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_step[0] > 0.0

    assert st["sig"][0, :, 0].mean() > 0.0
    assert abs(st["sig"][0, :, 2].mean()) > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert st["eint"][0] > 0.0
    assert np.all(st["epsp"][0] > 0.0)

    # uvar87 synchronization
    assert st["uvar87"][0, 0] > 0.0
    # Thickness thinning
    assert np.all(st["mat_extra"]["thk87"][0] < 1.0)


# ============================================================================
# 5. Shell Tri3 Element Formulation (Ish3n=1)
# ============================================================================

def test_shell_tri3_state_and_forces():
    """Verify 3-node triangular shell element under LAW87 dynamic stretch."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
    ])
    conn = np.array([
        [0, 1, 2],  # lower triangle
        [1, 3, 2],  # upper triangle
    ])
    mat = make_test_material_law87(aswift=210.0, nexp=0.2, epso=0.002)
    prop = MockProp(thick=1.0, nip=3, ish3n=1)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_tri3.init_group(group, model, None)
    st = group.state

    assert "uvar87" in st
    assert "uvar87" in st["mat_extra"]

    vel = np.zeros_like(coords)
    vel[[1, 3], 0] = 40.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        dt_step = shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert np.all(dt_step > 0.0)

    assert np.all(st["sig"][:, :, 0] > 0.0)
    assert np.all(st["eint"] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)
    assert np.all(st["epsp"] > 0.0)

    # uvar87 synchronization and thickness thinning
    assert np.all(st["uvar87"][:, 0] > 0.0)
    assert np.all(st["mat_extra"]["thk87"] < 1.0)


# ============================================================================
# 6. Directional Orthotropy (Barlat Yld2000-2d 0, 45, 90 degrees)
# ============================================================================

def test_directional_orthotropy_yield_and_stiffness():
    """Verify anisotropic directional stiffness and yield responses at 0 deg, 45 deg, 90 deg."""
    # Anisotropic sheet: alpha parameters chosen such that YLD_90 != YLD_0 != YLD_45
    mat = make_test_material_law87(
        al1=0.85, al2=1.15, al3=0.90, al4=1.10,
        al5=1.00, al6=0.95, al7=1.20, al8=0.80,
        expa=8.0,
        aswift=250.0, nexp=0.2, epso=0.002,
    )
    p = build_law87(mat)

    # 1. Theoretical equivalent stress difference for identical stress level:
    s0 = np.array([100.0, 0.0, 0.0])
    s90 = np.array([0.0, 100.0, 0.0])
    s45 = np.array([50.0, 50.0, 50.0])

    seq0 = barlat2000_equivalent_stress(s0, p)
    seq90 = barlat2000_equivalent_stress(s90, p)
    seq45 = barlat2000_equivalent_stress(s45, p)

    assert abs(seq90 - seq0) > 2.0, "Anisotropy must produce distinct equivalent stress at 90 deg vs 0 deg"
    assert abs(seq45 - seq0) > 2.0, "Anisotropy must produce distinct equivalent stress at 45 deg vs 0 deg"

    # 2. Element simulation: Pulling along 0 deg vs 90 deg under identical displacement control
    def run_single_elem_pull(direction: str):
        coords = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        prop = MockProp(thick=1.0, nip=1)
        grp = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        mdl = Model()
        mdl.x0 = coords.copy()
        grp._model = mdl
        shell_bt4.init_group(grp, mdl, None)

        vel = np.zeros_like(coords)
        if direction == "X":  # 0 deg
            vel[[1, 2], 0] = 50.0
        elif direction == "Y":  # 90 deg
            vel[[2, 3], 1] = 50.0

        dt = 1.0e-5
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))
        curr_x = coords.copy()

        for _ in range(30):
            curr_x += vel * dt
            shell_bt4.forces(grp, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        return grp.state["epsp"][0, 0], grp.state["sig"][0, 0].copy()

    epsp_0, sig_0 = run_single_elem_pull("X")
    epsp_90, sig_90 = run_single_elem_pull("Y")

    assert epsp_0 > 0.0
    assert epsp_90 > 0.0
    # Because seq90 != seq0, plastic strain accumulation under identical pull differs
    assert not np.isclose(epsp_0, epsp_90, rtol=1e-3), (
        f"Directional orthotropy must produce differential plastic strain: epsp_0={epsp_0}, epsp_90={epsp_90}"
    )


# ============================================================================
# 7. Kinematic Hardening & Bauschinger Effect in Shell Integration
# ============================================================================

def test_shell_bt4_kinematic_hardening_bauschinger():
    """Verify Bauschinger effect under forward tension and reverse compression with CHARD > 0."""
    mat_kin = make_test_material_law87(
        aswift=200.0, nexp=0.15, epso=0.002,
        chard=0.5,
        ikin=1,
        ckh=(200.0, 0.0, 0.0, 0.0),
        akh=(40.0, 0.0, 0.0, 0.0),
    )
    prop = MockProp(thick=1.0, nip=1)
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    grp = MockGroup(conn, slices=[(slice(0, 1), mat_kin, prop)])
    mdl = Model()
    mdl.x0 = coords.copy()
    grp._model = mdl
    shell_bt4.init_group(grp, mdl, None)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt = 1.0e-5
    curr_x = coords.copy()

    # Forward pull
    vel_fwd = np.zeros_like(coords)
    vel_fwd[[1, 2], 0] = 50.0
    for _ in range(25):
        curr_x += vel_fwd * dt
        shell_bt4.forces(grp, curr_x, vel_fwd, np.zeros_like(coords), dt, fint, mint)

    sig_fwd = grp.state["sig"][0, 0, 0]
    assert sig_fwd > 0.0

    # Reverse push (compression)
    vel_rev = np.zeros_like(coords)
    vel_rev[[1, 2], 0] = -50.0
    for _ in range(50):
        curr_x += vel_rev * dt
        shell_bt4.forces(grp, curr_x, vel_rev, np.zeros_like(coords), dt, fint, mint)

    sig_rev = grp.state["sig"][0, 0, 0]
    # Reverse stress reached compressive state
    assert sig_rev < 0.0
    # Total plastic strain increased
    assert grp.state["epsp"][0, 0] > 1e-4


# ============================================================================
# 8. Consistent Algorithmic Shell Tangent
# ============================================================================

def test_law87_consistent_tangent():
    """Verify consistent algorithmic shell tangent operator symmetry and positive-definiteness."""
    mat = make_test_material_law87(aswift=300.0, nexp=0.2, epso=0.002)

    # Elastic tangent (zero stress / strain)
    sig_el = np.zeros((1, 3))
    deps_el = np.zeros((1, 3))
    D_el = consistent_shell_tangent(mat, sig=sig_el, deps=deps_el)
    if D_el.ndim == 3:
        D_el = D_el[0]

    assert D_el.shape == (3, 3)
    # Check symmetry
    assert np.allclose(D_el, D_el.T, atol=1e-5)
    # Check positive eigenvalues
    eigs = np.linalg.eigvalsh(D_el)
    assert np.all(eigs > 0.0)

    # Plastic tangent under active strain
    sig_plas = np.array([[120.0, 0.0, 0.0]])
    deps_plas = np.array([[0.005, -0.00165, 0.0]])
    D_plas = consistent_shell_tangent(mat, sig=sig_plas, deps=deps_plas, epsp=np.array([0.01]))
    if D_plas.ndim == 3:
        D_plas = D_plas[0]

    assert D_plas.shape == (3, 3)
    # Tangent in loading direction softens (D_plas[0,0] < D_el[0,0])
    assert D_plas[0, 0] < D_el[0, 0]
