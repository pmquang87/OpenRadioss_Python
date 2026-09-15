"""
M565: Integration Test Suite for /MAT/LAW88 (Tabulated Hyperelastic).

Covers full element kernel integration and starter validation:
  - solid_hexa8: 8-node brick element with uvar88 state tracking & sound speed
  - solid_tetra4: 4-node tetrahedral element with uvar88 state tracking
  - shell_bt4: Belytschko-Tsay 4-node shell with plane-stress & thickness thinning
  - shell_qeph: QEPH 4-node shell with physical hourglass control
  - shell_tri3: 3-node triangular shell element
  - Starter checks: check_mat_law88 valid and error cases
  - Multi-element patch test: 2x2 brick patch under tension/compression/shear
"""

import math
import numpy as np
import pytest

from pyradioss.model.model import ElementGroup, Model
from pyradioss.model.entities import Material
from pyradioss.materials import law88_tab_hyp
from pyradioss.materials.law88_tab_hyp import TableData, MatparamLaw88
import pyradioss.materials as materials
from pyradioss.elements import solid_hexa8, solid_tetra4, shell_bt4, shell_qeph, shell_tri3
from pyradioss.starter.checks import check_mat_law88, _ALLOWED_LAWS


class DummyLog:
    """Mock MessageLog for starter checks."""
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def warning(self, msg, tag=""):
        self.warnings.append((tag, msg))

    def error(self, msg, tag=""):
        self.errors.append((tag, msg))

    def info(self, msg, tag=""):
        self.infos.append((tag, msg))


class DummyProp:
    """Mock Property for solid and shell elements."""
    def __init__(self, thick=1.0, nip=3, ishell=1, ish3n=1, qa=1.1, qb=0.05, h=0.1, hm=0.1, hf=0.1, hr=0.1):
        self.thick = thick
        self.nip = nip
        self.ishell = ishell
        self.ish3n = ish3n
        self.qa = qa
        self.qb = qb
        self.h = h
        self.hm = hm
        self.hf = hf
        self.hr = hr
        self.params = {
            "thick": thick,
            "nip": nip,
            "ishell": ishell,
            "ish3n": ish3n,
            "qa": qa,
            "qb": qb,
            "h": h,
            "hm": hm,
            "hf": hf,
            "hr": hr,
        }


def make_test_law88_material(
    mid: int = 1,
    rho0: float = 1000.0,
    bulk: float = 5000.0,
    shear: float = 200.0,
    nu: float = 0.495,
    young: float = 600.0,
    **kwargs,
) -> Material:
    """Create a verified LAW88 material with a monotonic loading table."""
    lam_table = np.linspace(0.1, 3.0, 30)
    g_table = 200.0 * lam_table
    table = TableData(x1=lam_table, y1d=g_table)

    params = {
        "rho0": rho0,
        "bulk": bulk,
        "shear": shear,
        "nu": nu,
        "young": young,
        "E": young,
        "G": shear,
        "K": bulk,
        "table": [table],
        "nl": 1,
        "func_load_list": [1],
        "rate_load_list": [0.0],
    }
    params.update(kwargs)
    mat = Material(id=mid, law=88, rho0=rho0, title="LAW88_Rubber", law_name="LAW88", params=params)
    return mat


# ============================================================================
# 1. Material Registry and Dispatch Metadata
# ============================================================================

def test_01_law88_registry_and_dispatch_metadata():
    """Verify LAW88 registration in MAT_PHYSICS_REGISTRY, dispatch dicts, and allowed laws."""
    materials.register_materials()
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

    expected_keys = (88, "88", "LAW88", "MAT_LAW88", "LAW88_TAB_HYP")
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"

    assert 88 in materials.MATERIAL_SOLID_DISPATCH
    assert 88 in materials.MATERIAL_SHELL_DISPATCH

    meta = materials.LAW_DISPATCH_METADATA.get(88)
    assert meta is not None
    assert meta.get("solid") is True
    assert meta.get("shell") is True
    assert meta.get("plane_stress") is True

    # Check allowed laws in starter
    assert 88 in _ALLOWED_LAWS["bricks"]
    assert 88 in _ALLOWED_LAWS["tetras"]
    assert 88 in _ALLOWED_LAWS["shells"]
    assert 88 in _ALLOWED_LAWS["sh3n"]
    for fam1d in ("trusses", "beams", "springs"):
        laws = _ALLOWED_LAWS.get(fam1d)
        if laws is not None:
            assert 88 not in laws


# ============================================================================
# 2. 8-node Brick Element (solid_hexa8) Integration
# ============================================================================

def test_02_solid_hexa8_kernel_integration():
    """Verify solid_hexa8 element initializes uvar88, evaluates cycle 0 time step,

    and updates stresses, energy, and uvar88 under tensile stretching.
    """
    model = Model()
    x0 = 0.5 * (solid_hexa8._XI + 1.0)  # Unit cube [0, 1]^3
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((8, 3), dtype=np.float64)
    model.vr = np.zeros((8, 3), dtype=np.float64)

    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = make_test_law88_material()
    prop = DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group._model = model

    log = DummyLog()
    nids, masses, _ = solid_hexa8.init_group(group, model, log)
    assert len(log.errors) == 0
    st = group.state
    assert "uvar88" in st
    assert st["uvar88"].shape == (1, 30)
    assert np.all(st["uvar88"] == 0.0)

    # Cycle 0 Courant probe
    fint = np.zeros((8, 3), dtype=np.float64)
    mint = np.zeros((8, 3), dtype=np.float64)
    dt_c0 = solid_hexa8.forces(group, model.x, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0
    assert not np.isnan(dt_c0[0])

    # Uniaxial tensile stretching: pull nodes with X=1 in +X direction
    vx = 5.0
    vel = np.zeros((8, 3), dtype=np.float64)
    x_nodes = np.where(x0[:, 0] > 0.5)[0]
    vel[x_nodes, 0] = vx
    dt = 1e-4

    curr_x = x0.copy()
    for _ in range(25):
        curr_x += vel * dt
        dt_crit = solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)
        assert dt_crit[0] > 0.0
        assert not np.isnan(dt_crit[0])

    # Verifications
    # Tensile stress sig_xx > 0
    assert st["sig"][0, 0] > 0.0
    assert not np.any(np.isnan(st["sig"]))

    # Internal energy accumulated and strictly positive
    assert st["eint"][0] > 0.0
    assert not np.isnan(st["eint"][0])

    # State history uvar88 updated (e.g. emax > 0, loadflg == 1)
    assert st["uvar88"][0, 0] > 0.0  # emax
    assert st["uvar88"][0, 4] == 1.0  # loadflg == 1.0 (loading)

    # Equilibrium of internal forces: sum(F_int) == 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 3. 4-node Tetrahedral Element (solid_tetra4) Integration
# ============================================================================

def test_03_solid_tetra4_kernel_integration():
    """Verify solid_tetra4 element initializes uvar88 and executes cleanly."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_law88_material()
    prop = DummyProp()
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros_like(coords)
    group._model = model

    log = DummyLog()
    solid_tetra4.init_group(group, model, log)
    assert len(log.errors) == 0
    st = group.state
    assert "uvar88" in st
    assert st["uvar88"].shape == (1, 30)

    # Cycle 0 Courant probe
    fint = np.zeros((4, 3), dtype=np.float64)
    mint = np.zeros((4, 3), dtype=np.float64)
    dt_c0 = solid_tetra4.forces(group, coords, None, None, 0.0, fint, mint)
    assert dt_c0[0] > 0.0

    # Tension deformation
    vel = np.zeros_like(coords)
    vel[1, 0] = 10.0
    dt = 1e-4
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        dt_crit = solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)
        assert dt_crit[0] > 0.0
        assert not np.isnan(dt_crit[0])

    assert st["sig"][0, 0] > 0.0
    assert st["eint"][0] > 0.0
    assert st["uvar88"][0, 0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 4. Belytschko-Tsay Shell Element (shell_bt4) Integration
# ============================================================================

def test_04_shell_bt4_kernel_integration():
    """Verify shell_bt4 element plane-stress update, thickness thinning, and uvar88 tracking."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_law88_material()
    prop = DummyProp(thick=2.0, nip=3, ishell=1)
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    group._model = model

    log = DummyLog()
    shell_bt4.init_group(group, model, log)
    assert len(log.errors) == 0
    st = group.state
    assert "uvar88" in st
    assert st["uvar88"].shape == (1, 30)

    # Biaxial tension stretching
    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 5.0
    vel[[2, 3], 1] = 5.0
    dt = 1e-4

    fint = np.zeros((4, 3), dtype=np.float64)
    mint = np.zeros((4, 3), dtype=np.float64)
    curr_x = coords.copy()

    for _ in range(25):
        curr_x += vel * dt
        dt_crit = shell_bt4.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_crit[0] > 0.0
        assert not np.isnan(dt_crit[0])

    # Stress checks
    assert st["sig"][0, :, 0].mean() > 0.0
    assert st["sig"][0, :, 1].mean() > 0.0
    assert not np.any(np.isnan(st["sig"]))

    # Thickness thinning: out-of-plane stretch lambda_3 < 1.0
    lam3 = st["uvar88"][0, 7]
    assert 0.5 < lam3 < 1.0  # Thinning occurred under biaxial tension
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 5. QEPH Shell Element (shell_qeph) Integration
# ============================================================================

def test_05_shell_qeph_kernel_integration():
    """Verify shell_qeph element with physical hourglass control under LAW88."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = make_test_law88_material()
    prop = DummyProp(thick=1.5, nip=3, ishell=24)
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    group._model = model

    log = DummyLog()
    shell_qeph.init_group(group, model, log)
    assert len(log.errors) == 0
    st = group.state
    assert "uvar88" in st

    vel = np.zeros_like(coords)
    vel[[1, 2], 0] = 8.0
    dt = 1e-4

    fint = np.zeros((4, 3), dtype=np.float64)
    mint = np.zeros((4, 3), dtype=np.float64)
    curr_x = coords.copy()

    for _ in range(20):
        curr_x += vel * dt
        dt_crit = shell_qeph.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_crit[0] > 0.0
        assert not np.isnan(dt_crit[0])

    assert st["sig"][0, :, 0].mean() > 0.0
    assert not np.any(np.isnan(st["sig"]))
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 6. Triangular Shell Element (shell_tri3) Integration
# ============================================================================

def test_06_shell_tri3_kernel_integration():
    """Verify shell_tri3 element integration under LAW88."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2]], dtype=np.int64)

    mat = make_test_law88_material()
    prop = DummyProp(thick=1.0, nip=3, ish3n=1)
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    group._model = model

    log = DummyLog()
    shell_tri3.init_group(group, model, log)
    assert len(log.errors) == 0
    st = group.state
    assert "uvar88" in st

    vel = np.zeros_like(coords)
    vel[1, 0] = 5.0
    dt = 1e-4

    fint = np.zeros((3, 3), dtype=np.float64)
    mint = np.zeros((3, 3), dtype=np.float64)
    curr_x = coords.copy()

    for _ in range(15):
        curr_x += vel * dt
        dt_crit = shell_tri3.forces(group, curr_x, vel, np.zeros_like(coords), dt, fint, mint)
        assert dt_crit[0] > 0.0
        assert not np.isnan(dt_crit[0])

    assert st["sig"][0, :, 0].mean() > 0.0
    assert not np.any(np.isnan(st["sig"]))
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ============================================================================
# 7. Starter Checks (check_mat_law88)
# ============================================================================

def test_07_starter_check_mat_law88_bounds():
    """Validate check_mat_law88 for valid inputs, density <= 0, no curves,

    descending strain rates, negative shear, and 1D element rejection.
    """
    # 1. Valid material
    mat_valid = make_test_law88_material(mid=10)
    log_valid = DummyLog()
    check_mat_law88(mat=mat_valid, log=log_valid)
    assert len(log_valid.errors) == 0

    # 2. Density rho0 <= 0 (ANCMSG 1514)
    mat_rho_bad = make_test_law88_material(mid=11, rho0=0.0)
    log_rho = DummyLog()
    check_mat_law88(mat=mat_rho_bad, log=log_rho)
    assert any("ANCMSG 1514" in msg for _, msg in log_rho.errors)

    # 3. No loading curve defined (NL = 0) (ANCMSG 866)
    mat_no_curve = make_test_law88_material(mid=12, nl=0, func_load_list=[])
    log_curve = DummyLog()
    check_mat_law88(mat=mat_no_curve, log=log_curve)
    assert any("ANCMSG 866" in msg for _, msg in log_curve.errors)

    # 4. Descending strain rates (ANCMSG 478)
    mat_desc_rate = make_test_law88_material(mid=13, rate_load_list=[100.0, 50.0])
    log_rate = DummyLog()
    check_mat_law88(mat=mat_desc_rate, log=log_rate)
    assert any("ANCMSG 478" in msg for _, msg in log_rate.errors)

    # 5. Negative shear modulus warning (ANCMSG 3109)
    mat_neg_shear = make_test_law88_material(mid=14, shear=-25.0)
    log_shear = DummyLog()
    check_mat_law88(mat=mat_neg_shear, log=log_shear)
    assert any("ANCMSG 3109" in msg for _, msg in log_shear.warnings)

    # 6. Incompatible 1D elements (ANCMSG 306)
    model = Model()
    model.materials[15] = make_test_law88_material(mid=15)
    # Mock a beam element referencing LAW88
    beam_group = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]), part=np.array([0]))
    beam_group.state["slices"] = [(slice(0, 1), model.materials[15], DummyProp())]
    model.beams = beam_group
    log_elem = DummyLog()
    check_mat_law88(model=model, mat_id=15, log=log_elem)
    assert any("ANCMSG 306" in msg for _, msg in log_elem.errors)


# ============================================================================
# 8. Multi-Element Patch Test (2x2 Brick Patch: Tension, Compression, Shear)
# ============================================================================

def test_08_multielement_brick_patch_cyclic():
    """Verify 4-element 2x2 brick patch undergoing multi-step cyclic deformation.

    Checks positive internal energy, zero net force (equilibrium), and finite states.
    """
    # Create 2x2 grid of unit cubes -> 4 elements, 18 nodes
    # Grid: X in [0, 2], Y in [0, 2], Z in [0, 1]
    xs = np.linspace(0.0, 2.0, 3)
    ys = np.linspace(0.0, 2.0, 3)
    zs = np.array([0.0, 1.0])

    grid_nodes = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_nodes.append([x, y, z])
    coords = np.array(grid_nodes, dtype=np.float64)  # 18 nodes

    def node_id(ix, iy, iz):
        return iz * 9 + iy * 3 + ix

    conn_list = []
    for ey in range(2):
        for ex in range(2):
            n0 = node_id(ex, ey, 0)
            n1 = node_id(ex + 1, ey, 0)
            n2 = node_id(ex + 1, ey + 1, 0)
            n3 = node_id(ex, ey + 1, 0)
            n4 = node_id(ex, ey, 1)
            n5 = node_id(ex + 1, ey, 1)
            n6 = node_id(ex + 1, ey + 1, 1)
            n7 = node_id(ex, ey + 1, 1)
            conn_list.append([n0, n1, n2, n3, n4, n5, n6, n7])

    conn = np.array(conn_list, dtype=np.int64)
    mat = make_test_law88_material(bulk=5000.0, shear=200.0)
    prop = DummyProp()

    group = ElementGroup(ids=np.arange(1, 5, dtype=np.int64), conn=conn, part=np.zeros(4, dtype=np.int64))
    group.state["slices"] = [(slice(0, 4), mat, prop)]

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros_like(coords)
    group._model = model

    log = DummyLog()
    solid_hexa8.init_group(group, model, log)
    assert len(log.errors) == 0

    fint = np.zeros((18, 3), dtype=np.float64)
    mint = np.zeros((18, 3), dtype=np.float64)
    dt = 1e-4

    curr_x = coords.copy()

    # Phase 1: Tension in X (20 cycles) - affine stretch vx(x) = (x / 2.0) * 5.0
    vel_tension = np.zeros_like(coords)
    vel_tension[:, 0] = (coords[:, 0] / 2.0) * 5.0

    for _ in range(20):
        curr_x += vel_tension * dt
        dt_crit = solid_hexa8.forces(group, curr_x, vel_tension, None, dt, fint, mint)
        assert np.all(dt_crit > 0.0)

    st = group.state
    assert np.all(st["eint"] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-7)

    # Phase 2: Compression Reversal (20 cycles)
    vel_comp = np.zeros_like(coords)
    vel_comp[:, 0] = -(coords[:, 0] / 2.0) * 5.0

    for _ in range(20):
        curr_x += vel_comp * dt
        dt_crit = solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)
        assert np.all(dt_crit > 0.0)

    assert np.all(st["eint"] > 0.0)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-7)

    # Phase 3: Simple Shear in XY (20 cycles)
    vel_shear = np.zeros_like(coords)
    vel_shear[:, 0] = (coords[:, 1] / 2.0) * 4.0

    for _ in range(20):
        curr_x += vel_shear * dt
        dt_crit = solid_hexa8.forces(group, curr_x, vel_shear, None, dt, fint, mint)
        assert np.all(dt_crit > 0.0)

    assert np.all(st["eint"] > 0.0)
    assert not np.any(np.isnan(st["sig"]))
    assert not np.any(np.isnan(st["uvar88"]))
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-7)
