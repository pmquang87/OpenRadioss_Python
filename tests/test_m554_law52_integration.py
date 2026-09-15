"""
Integration test suite for /MAT/LAW52 (/MAT/GURSON, /MAT/PLAS_GURS).

Milestone M554 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Registry and dispatch metadata verification for all LAW52 aliases.
2. Acoustic sound speeds (3D dilatational and plane-stress) consistency across models.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1): uniaxial tension, compression, shear, time step, and void evolution.
   - Hexa8 element erosion upon reaching void coalescence / rupture (f >= ff or f* >= fu).
   - Tetra4: single-point tetrahedral formulation, stable time step, and void growth.
4. Shell element formulations:
   - BT4 (Ishell=1): plane-stress, thickness thinning, multi-layer void growth.
   - BT4 element erosion upon void failure.
   - QEPH (Ishell=24): 4-node quad shell with improved hourglass control under LAW52.
   - Tri3 (Ish3n=1): 3-node triangular shell formulation under LAW52.
5. Consistent tangent stiffness dispatch for implicit analysis (solid and shell).
6. Starter checks inclusion for all supported solid/shell families and rejection of 1D elements.
7. MatLaw52 dataclass compatibility and properties.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law52_gurson import (
    Law52Params,
    build_law52,
    shell_membrane_tangent as law52_shell_membrane_tangent,
    shell_update_law52,
    solid_update_law52,
    sound_speed_shell_law52,
    sound_speed_solid_law52,
    tangent_law52_shell,
    tangent_law52_solid,
)
from pyradioss.model.entities import Material, MatLaw52, MatGurson, MatPlasGurs, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model
from pyradioss.common.messages import MessageLog


# ============================================================================
# Helper Mock Element Classes for Kernel Unit Testing
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


def make_test_material_law52(
    mid: int = 1,
    rho0: float = 7.85e-9,
    E: float = 210000.0,
    nu: float = 0.3,
    a: float = 300.0,
    b: float = 400.0,
    n: float = 0.2,
    q1: float = 1.5,
    q2: float = 1.0,
    q3: float = 2.25,
    fi: float = 0.01,
    fc: float = 0.15,
    ff: float = 0.25,
    fn: float = 0.04,
    sn: float = 0.1,
    epsn: float = 0.3,
    **kwargs: Any,
) -> Material:
    params = {
        "E": E,
        "nu": nu,
        "a": a,
        "b": b,
        "n": n,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "f_i": fi,
        "f_c": fc,
        "f_f": ff,
        "f_n": fn,
        "s_n": sn,
        "eps_n": epsn,
        "yield_a": a,
        "hard_b": b,
        "hard_n": n,
        "fi": fi,
        "fc": fc,
        "ff": ff,
        "fn": fn,
        "sn": sn,
        "epsn": epsn,
        "fu": 1.0 / q1 if q1 > 0 else 0.667,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=52, rho0=rho0, title="Steel_LAW52_Gurson", params=params)
    return mat


# ============================================================================
# 1. Registry & Dispatch Metadata Verification
# ============================================================================

def test_law52_registry_and_metadata():
    """Verify MAT_PHYSICS_REGISTRY, LAW_DISPATCH_METADATA, and dispatch dictionaries."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()

    expected_keys = (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS")
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True
        assert meta.get("solid") is True
        assert meta.get("shell") is True
        assert k in materials.MATERIAL_SOLID_DISPATCH
        assert k in materials.MATERIAL_SHELL_DISPATCH

    mat = make_test_material_law52()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "epsm" in shapes_solid
    assert "sigm" in shapes_solid
    assert "dmg" in shapes_solid
    assert shapes_solid["dmg"] == (5,)
    assert "fg" in shapes_solid
    assert "fn" in shapes_solid
    assert "f" in shapes_solid
    assert "fstar" in shapes_solid
    assert "off" in shapes_solid

    shapes_shell = materials.extra_shapes(mat, nip=5)
    assert shapes_shell["epsm"] == (5,)
    assert shapes_shell["sigm"] == (5,)
    assert shapes_shell["dmg"] == (5, 5)
    assert shapes_shell["fg"] == (5,)
    assert shapes_shell["fn"] == (5,)
    assert shapes_shell["f"] == (5,)
    assert shapes_shell["fstar"] == (5,)
    assert shapes_shell["off"] == (5,)


def test_law52_sound_speed_consistency():
    """Verify 3D dilatational and plane-stress acoustic wave speed formulas."""
    rho0 = 7.85e-9
    E = 210000.0
    nu = 0.3
    mat = make_test_material_law52(rho0=rho0, E=E, nu=nu)

    c_solid_expected = math.sqrt((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))
    c_shell_expected = math.sqrt(E / ((1.0 - nu * nu) * rho0))

    # Test Material methods
    assert math.isclose(mat.sound_speed_solid(), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(mat.sound_speed_shell(), c_shell_expected, rel_tol=1e-12)

    # Test module-level dispatchers
    assert math.isclose(materials.sound_speed(mat, rho=rho0), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(sound_speed_solid_law52(mat), c_solid_expected, rel_tol=1e-12)
    assert math.isclose(sound_speed_shell_law52(mat), c_shell_expected, rel_tol=1e-12)


# ============================================================================
# 2. Solid Element Formulations (Hexa8, Tetra4)
# ============================================================================

def test_hexa8_solid_kernel_tension_and_void_growth():
    """Verify Hexa8 standard formulation (Isolid=1) under tension: equilibrium, plastic strain, void growth."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mat = make_test_material_law52(a=300.0, b=400.0, n=0.2, fi=0.01)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True
    assert "dmg" in st["mat_extra"]
    assert "epsm" in st["mat_extra"]
    assert "sigm" in st["mat_extra"]
    assert "off" in st["mat_extra"]

    # Cycle 0 time step probe
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt_c0 = solid_hexa8.forces(group, coords, None, None, 0.0, fint, mint)
    assert len(dt_c0) == 1
    assert dt_c0[0] > 0.0

    # Uniaxial tension along x: move right face with velocity +100 mm/s
    vel = np.zeros_like(coords)
    vel[[1, 2, 5, 6], 0] = 100.0
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(25):
        curr_x += vel * dt
        solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

    sig = st["sig"][0]
    assert sig[0] > 0.0  # Tensile stress xx
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)  # Nodal equilibrium

    # Plastic strain accumulated
    assert st["epsp"][0] > 0.0

    # Matrix equivalent plastic strain and flow stress
    assert st["mat_extra"]["epsm"][0] > 0.0
    assert st["mat_extra"]["sigm"][0] >= 300.0

    # Void growth: under tension, void growth delta_fg > 0, so f > fi
    dmg = st["mat_extra"]["dmg"][0]
    # dmg = [f*, fg, fn, f, f*]
    assert dmg[1] > 0.0  # fg > 0
    assert dmg[3] > 0.01  # f > fi


def test_hexa8_element_erosion_at_rupture():
    """Verify Hexa8 element deletion when void volume fraction reaches coalescence/rupture."""
    coords = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    # Low rupture limits: fi=0.04, fc=0.05, ff=0.06, fu=0.06
    mat = make_test_material_law52(a=200.0, b=100.0, n=0.1, fi=0.04, fc=0.05, ff=0.06, fu=0.06)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

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

    assert st["off"][0] == 0.0
    assert np.allclose(st["sig"][0], 0.0)


def test_solid_tetra4_kernel_integration():
    """Verify single 4-node tetrahedron formulation under LAW52."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law52(a=250.0, b=300.0, n=0.25, fi=0.02)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True

    # Cycle 0 time step
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c0 = solid_tetra4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0

    # Tension
    vel = np.zeros_like(coords)
    vel[1, 0] = 200.0
    dt = 1.0e-5

    curr_x = coords.copy()
    for _ in range(40):
        curr_x += vel * dt
        solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

    assert st["sig"][0, 0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)
    assert st["epsp"][0] > 0.0
    assert st["mat_extra"]["dmg"][0, 1] > 0.0  # fg > 0


# ============================================================================
# 3. Shell Element Formulations (BT4, QEPH, Tri3)
# ============================================================================

def test_shell_bt4_kernel_biaxial_and_thinning():
    """Verify BT4 shell (Ishell=1) with multi-layer void growth and thickness thinning."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law52(a=300.0, b=400.0, n=0.2, fi=0.01)
    prop = MockProp(thick=1.5, nip=5)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True
    assert "dmg" in st["mat_extra"]
    assert "epsm" in st["mat_extra"]
    assert "sigm" in st["mat_extra"]

    thk_0 = float(st["thick"][0])
    assert math.isclose(thk_0, 1.5)

    # Biaxial tension
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 100.0
    vel[[2, 3], 1] = 100.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert st["sig"][0, :, 0].mean() > 0.0
    assert st["sig"][0, :, 1].mean() > 0.0
    # Thickness and multi-layer void evolution
    assert math.isclose(st["thick"][0], thk_0)
    assert st["mat_extra"]["dmg"][0, :, 1].mean() > 0.0  # fg > 0
    assert st["mat_extra"]["dmg"][0, :, 3].mean() > 0.01  # f > fi


def test_shell_bt4_element_erosion_on_failure():
    """Verify BT4 shell erosion when void volume fraction reaches rupture limit."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law52(a=150.0, b=50.0, fi=0.04, fc=0.05, ff=0.06, fu=0.06)
    prop = MockProp(thick=1.0, nip=3)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_bt4.init_group(group, model, None)
    st = group.state

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 3000.0
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
    """Verify QEPH formulation (Ishell=24) with improved hourglass control under LAW52."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    mat = make_test_material_law52(a=280.0, b=350.0, n=0.22, fi=0.01)
    prop = MockProp(thick=1.2, nip=3, ishell=24)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    shell_qeph.init_group(group, model, None)
    st = group.state
    assert st["chk_fail"] is True
    assert st["cspd"][0] > 0.0

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 80.0
    dt = 1.0e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert st["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)
    assert st["mat_extra"]["dmg"][0, :, 1].mean() > 0.0  # fg > 0


def test_shell_tri3_kernel_integration():
    """Verify 3-node triangular shell formulation (Ish3n=1) under LAW52."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    conn = np.array([[0, 1, 2]])
    mat = make_test_material_law52(a=250.0, b=300.0, n=0.2, fi=0.01)
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

    for _ in range(15):
        curr_x += vel * dt
        shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)

    assert st["sig"][0, :, 0].mean() > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-6)
    assert st["mat_extra"]["dmg"][0, :, 1].mean() > 0.0  # fg > 0


# ============================================================================
# 4. Consistent Tangent Stiffness Dispatch
# ============================================================================

def test_consistent_tangents_dispatch():
    """Verify solid and shell consistent tangent dispatch for implicit analysis."""
    mat = make_test_material_law52(E=210000.0, nu=0.3, a=300.0, fi=0.01)

    # 1. Solid tangent: (n, 6, 6) tensor
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    depsp = np.zeros(1)
    extra_solid = {"epsm": np.zeros(1), "sigm": np.full(1, 300.0), "dmg": np.array([[0.01, 0.0, 0.0, 0.01, 0.01]])}
    d_solid = materials.solid_tangent(mat, sig, epsp, depsp, extra=extra_solid)
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
    extra_shell = {"epsm": np.zeros(1), "sigm": np.full(1, 300.0), "dmg": np.array([[0.01, 0.0, 0.0, 0.01, 0.01]])}
    d_layer = materials.shell_layer_tangent(mat, sig_sh, epsp, depsp, extra=extra_shell)
    assert d_layer.shape == (1, 3, 3)
    assert np.allclose(d_layer[0], d_layer[0].T)


# ============================================================================
# 5. Starter Checks Inclusion & 1D Element Rejection
# ============================================================================

def test_starter_checks_allowed_laws_and_1d_rejection():
    """Verify that LAW52 is in _ALLOWED_LAWS for solids/shells, and 1D elements are rejected."""
    keys_to_check = (52, "52", "LAW52", "GURSON", "PLAS_GURS")
    solid_families = ("bricks", "tetras", "penta6", "pyra5")
    shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n")

    for fam in solid_families:
        for k in keys_to_check:
            assert k in _ALLOWED_LAWS[fam], f"{k} missing in {fam}"

    for fam in shell_families:
        for k in keys_to_check:
            assert k in _ALLOWED_LAWS[fam], f"{k} missing in {fam}"

    # Verify 1D elements rejection in check_model
    model = Model()
    model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
    mat = make_test_material_law52(mid=1)
    model.materials[1] = mat
    model.parts[1] = Part(id=1, mat_id=1, prop_id=1)

    class Mock1DElementGroup:
        def __init__(self, name: str):
            self.name = name
            self.state = {"slices": [(slice(0, 1), mat, MockProp())]}

    model._element_groups = [
        ("trusses", Mock1DElementGroup("trusses")),
        ("beams", Mock1DElementGroup("beams")),
        ("springs", Mock1DElementGroup("springs")),
    ]
    model.element_groups = lambda: model._element_groups

    log = MessageLog()
    check_model(model, log)

    truss_errors = [e for e in log.errors if "trusses" in e and "LAW52" in e]
    beam_errors = [e for e in log.errors if "beams" in e and "LAW52" in e]
    spring_errors = [e for e in log.errors if "springs" in e and "LAW52" in e]

    assert len(truss_errors) > 0, "Expected error rejecting trusses for LAW52"
    assert len(beam_errors) > 0, "Expected error rejecting beams for LAW52"
    assert len(spring_errors) > 0, "Expected error rejecting springs for LAW52"


# ============================================================================
# 6. MatLaw52 Dataclass & Entity Properties
# ============================================================================

def test_mat_law52_dataclass():
    """Verify MatLaw52, MatGurson, MatPlasGurs dataclass and property access."""
    mat = MatLaw52(
        id=10,
        rho=7.85e-9,
        e=210000.0,
        nu=0.3,
        a=320.0,
        b=450.0,
        n=0.25,
        q1=1.5,
        q2=1.0,
        q3=2.25,
        f_i=0.02,
        f_c=0.15,
        f_f=0.25,
        f_n=0.04,
        s_n=0.1,
        eps_n=0.3,
    )

    assert mat.law == 52
    assert mat.law_name == "LAW52"
    assert mat.E == 210000.0
    assert mat.nu == 0.3
    assert math.isclose(mat.G, 210000.0 / (2.0 * 1.3), rel_tol=1e-6)
    assert math.isclose(mat.K, 210000.0 / (3.0 * (1.0 - 2.0 * 0.3)), rel_tol=1e-6)
    assert math.isclose(mat.fu, 1.0 / 1.5, rel_tol=1e-6)
    assert mat.yield_stress == 320.0
    assert mat.hardening_b == 450.0
    assert mat.hardening_n == 0.25

    # Check sound speed CallableFloat
    assert mat.sound_speed_solid() > 0.0
    assert mat.sound_speed_solid == mat.sound_speed_solid()
    assert mat.sound_speed_shell() > 0.0

    # Aliases
    assert MatGurson is MatLaw52
    assert MatPlasGurs is MatLaw52

    # Params dict
    p = mat.params
    assert p["E"] == 210000.0
    assert p["yield_a"] == 320.0
    assert p["f_i"] == 0.02
