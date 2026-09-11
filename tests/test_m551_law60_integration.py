"""
Integration test suite for /MAT/LAW60 (/MAT/PLAS_T3, /MAT/FABRIC).

Milestone M551 Subagent 1C: Element Integration & Multi-Formulation Harness
1. Starter parsing of /MAT/LAW60, /MAT/PLAS_T3, /MAT/FABRIC decks (fixed and free format).
2. DeckWriter generation, validation error diagnostics, and round-trip re-reading.
3. Registry and dispatch metadata verification for all LAW60 aliases.
4. Solid element formulations: Hexa8 (Isolid=1), HEPH (Isolid=24), Tetra4.
   - Dynamic engine runs.
   - Uniaxial tension, uniaxial compression, and pure shear stress updates.
   - Direct kernel evaluation with internal force equilibrium and plastic strain tracking.
5. Shell element formulations: BT4 (Ishell=1), QEPH (Ishell=24), Tri3 (Ish3n=1).
   - Dynamic engine runs.
   - Plane stress condition and thickness thinning under biaxial tension.
   - Direct shell kernel evaluation and through-thickness plastic strain integration.
6. Element deletion / erosion when equivalent plastic strain reaches eps_max.
7. Courant time step bounding and wave speed degradation under damage.
8. Algorithmic consistent tangent stiffness dispatch for solids and shells.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.common.tables import FunctTable
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_heph, solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    consistent_shell_tangent,
    consistent_solid_tangent,
    extra_shapes,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_materials
from pyradioss.starter.starter import run_starter


# ============================================================================
# 1. Starter Parsing & DeckWriter Round-Trip
# ============================================================================

def test_starter_parsing_law60_fixed_format(tmp_path: Path):
    """Verify parsing of 10-card fixed-format /MAT/LAW60 deck."""
    deck_text = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "test_law60_fixed\n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/FUNCT/101\n"
        "YieldCurve1\n"
        "                 0.0               250.0\n"
        "                0.05               350.0\n"
        "                0.20               450.0\n"
        "/FUNCT/102\n"
        "YieldCurve2\n"
        "                 0.0               280.0\n"
        "                0.05               380.0\n"
        "                0.20               480.0\n"
        "/MAT/LAW60/1\n"
        "Fabric LAW60\n"
        "#              RHO_I           REFER_RHO\n"
        f"{fmt_float(7.85e-9, 20)}{fmt_float(7.85e-9, 20)}\n"
        "#                  E                  NU           EPS_P_MAX              EPS_T1              EPS_T2\n"
        f"{fmt_float(210000.0, 20)}{fmt_float(0.3, 20)}{fmt_float(0.4, 20)}{fmt_float(0.15, 20)}{fmt_float(0.25, 20)}\n"
        "#      NFUNC   Fsmooth            MAT_HARD                Fcut\n"
        f"{fmt_int(2, 10)}{fmt_int(1, 10)}{fmt_float(0.5, 20)}{fmt_float(1000.0, 20)}\n"
        "#     Xr_fun          MAT_FScale    fct_ID_k                 E_R              MAT_C1\n"
        f"{fmt_int(0, 10)}{fmt_float(1.0, 20)}{fmt_int(0, 10)}{fmt_float(50000.0, 20)}{fmt_float(10.0, 20)}\n"
        "#    fct_ID1   fct_ID2\n"
        f"{fmt_int(101, 10)}{fmt_int(102, 10)}\n"
        "#    Fscale1   Fscale2\n"
        f"{fmt_float(1.0, 20)}{fmt_float(1.0, 20)}\n"
        "#      Rate1     Rate2\n"
        f"{fmt_float(0.0, 20)}{fmt_float(10.0, 20)}\n"
        "/END\n"
    )
    rad_file = tmp_path / "test_law60_0000.rad"
    rad_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.id == 1
    assert getattr(mat, "law", None) == 60
    assert np.isclose(mat.rho0, 7.85e-9)
    assert np.isclose(mat.params["E"], 210000.0)
    assert np.isclose(mat.params["nu"], 0.3)
    assert np.isclose(mat.params["eps_p_max"], 0.4)
    assert np.isclose(mat.params["eps_t1"], 0.15)
    assert np.isclose(mat.params["eps_t2"], 0.25)
    assert mat.params["nfunc"] == 2
    assert mat.params["fsmooth"] == 1
    assert np.isclose(mat.params["mat_hard"], 0.5)
    assert np.isclose(mat.params["fcut"], 1000.0)
    assert mat.params["funcs"] == [101, 102]
    assert np.allclose(mat.params["fscales"], [1.0, 1.0])
    assert np.allclose(mat.params["rates"], [0.0, 10.0])

    assert 1 in model.mat_law60s
    m60 = model.mat_law60s[1]
    assert m60.id == 1
    assert m60.nfunc == 2
    assert m60.funcs == [101, 102]


def test_starter_parsing_synonyms(tmp_path: Path):
    """Verify parsing of /MAT/PLAS_T3 and /MAT/FABRIC synonyms."""
    deck_text = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "test_synonyms\n"
        "        10         0\n"
        "/MAT/PLAS_T3/2\n"
        "Plastic T3\n"
        "7.8e-9\n"
        "200000.0 0.28 0.3 1e30 2e30\n"
        "1 0 0.0 1e30\n"
        "0 1.0 0 0.0 0.0\n"
        "201\n"
        "1.0\n"
        "0.0\n"
        "/MAT/FABRIC/3\n"
        "Woven Fabric\n"
        "1.5e-9\n"
        "50000.0 0.35 0.5 1e30 2e30\n"
        "1 0 0.0 1e30\n"
        "0 1.0 0 0.0 0.0\n"
        "301\n"
        "1.0\n"
        "0.0\n"
        "/END\n"
    )
    rad_file = tmp_path / "test_syn_0000.rad"
    rad_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 2 in model.materials
    assert model.materials[2].law == 60
    assert np.isclose(model.materials[2].params["E"], 200000.0)

    assert 3 in model.materials
    assert model.materials[3].law == 60
    assert np.isclose(model.materials[3].params["E"], 50000.0)


def test_deck_writer_roundtrip_law60(tmp_path: Path):
    """Verify StarterDeck.mat_law60 generation and round-trip re-reading."""
    d = StarterDeck("test_roundtrip")
    d.funct(10, "Curve1", [(0.0, 200.0), (0.1, 350.0)])
    d.funct(20, "Curve2", [(0.0, 240.0), (0.1, 400.0)])
    d.mat_law60(
        mid=5,
        title="RoundtripMat",
        rho=7.8e-9,
        e=205000.0,
        nu=0.29,
        eps_p_max=0.35,
        eps_t1=0.12,
        eps_t2=0.22,
        nfunc=2,
        fsmooth=1,
        mat_hard=0.3,
        fcut=800.0,
        xr_fun=0,
        mat_fscale=1.0,
        einf=60000.0,
        ce=8.0,
        funcs=[10, 20],
        fscales=[1.0, 1.2],
        rates=[0.0, 50.0],
    )
    rad_file = tmp_path / "test_roundtrip_0000.rad"
    d.write(str(rad_file))

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 5 in model.materials
    mat = model.materials[5]
    assert mat.law == 60
    assert np.isclose(mat.rho0, 7.8e-9)
    assert np.isclose(mat.params["E"], 205000.0)
    assert np.isclose(mat.params["nu"], 0.29)
    assert np.isclose(mat.params["eps_p_max"], 0.35)
    assert np.isclose(mat.params["eps_t1"], 0.12)
    assert np.isclose(mat.params["eps_t2"], 0.22)
    assert mat.params["nfunc"] == 2
    assert mat.params["fsmooth"] == 1
    assert np.isclose(mat.params["mat_hard"], 0.3)
    assert np.isclose(mat.params["fcut"], 800.0)
    assert np.isclose(mat.params["einf"], 60000.0)
    assert np.isclose(mat.params["ce"], 8.0)
    assert mat.params["funcs"] == [10, 20]
    assert np.allclose(mat.params["fscales"], [1.0, 1.2])
    assert np.allclose(mat.params["rates"], [0.0, 50.0])


def test_starter_validation_errors(tmp_path: Path):
    """Verify starter validation checks catch invalid LAW60 parameters."""
    # 1. Negative E
    m_bad_e = Material(id=1, law=60, rho0=7.8e-9, params={"E": -100.0, "nu": 0.3})
    log = MessageLog()
    from pyradioss.starter.checks import check_mat_law60
    check_mat_law60(m_bad_e, log)
    assert any("Young's modulus" in err for err in log.errors)

    # 2. nu >= 0.5
    m_bad_nu = Material(id=2, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.52})
    log = MessageLog()
    check_mat_law60(m_bad_nu, log)
    assert any("Poisson's ratio" in err for err in log.errors)

    # 3. eps_t1 >= eps_t2
    m_bad_fail = Material(id=3, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.3, "eps_t1": 0.3, "eps_t2": 0.2})
    log = MessageLog()
    check_mat_law60(m_bad_fail, log)
    assert any("eps_t1 < eps_t2" in err for err in log.errors)

    # 4. Non-monotonic strain rates
    m_bad_rate = Material(
        id=4,
        law=60,
        rho0=7.8e-9,
        params={"E": 200000.0, "nu": 0.3, "nfunc": 2, "rates": [10.0, 5.0]},
    )
    log = MessageLog()
    check_mat_law60(m_bad_rate, log)
    assert any("strictly increasing" in err for err in log.errors)


# ============================================================================
# 2. Registry & Dispatch Metadata Verification
# ============================================================================

def test_registry_contains_all_law60_aliases():
    """Verify that all LAW60 aliases are registered in MAT_PHYSICS_REGISTRY and metadata."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    from pyradioss.materials import LAW_DISPATCH_METADATA

    expected_keys = (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC")
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True
        assert meta.get("solid") is True
        assert meta.get("shell") is True


def test_extra_shapes_allocation():
    """Verify that extra_shapes allocates uvar state array with (5 + nfunc,) per IP."""
    mat_solid = Material(id=1, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.3, "nfunc": 3})
    shapes_solid = materials.extra_shapes(mat_solid, nip=None)
    assert "uvar" in shapes_solid
    assert shapes_solid["uvar"] == (8,)  # 5 + 3 = 8
    assert "off60" in shapes_solid

    mat_shell = Material(id=2, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.3, "nfunc": 2})
    shapes_shell = materials.extra_shapes(mat_shell, nip=5)
    assert "uvar" in shapes_shell
    assert shapes_shell["uvar"] == (5, 7)  # (nip, 5 + 2)
    assert "off60" in shapes_shell
    assert shapes_shell["off60"] == (5,)


# ============================================================================
# 3. Solid Formulations: Hexa8, HEPH, Tetra4 Integration
# ============================================================================

def test_hexa8_solid_law60_engine_run(tmp_path: Path):
    """Test Hexa8 standard solid formulation (Isolid=1) with LAW60 through engine."""
    d = StarterDeck("test_hexa8_law60")
    d.funct(1, "YieldCurve", [(0.0, 250.0), (0.1, 350.0), (0.3, 450.0)])
    d.mat_law60(mid=1, title="SteelLAW60", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
    d.prop_solid(pid=1, title="PropSolidHexa", isolid=1)
    d.part(pid=1, title="PartHexa", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_hexa8_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_hexa8_law60_0000.rad"
    eng_file = tmp_path / "test_hexa8_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert model is not None
    assert 1 in model.materials
    assert getattr(model.materials[1], "law", None) == 60

    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_heph_solid_law60_engine_run(tmp_path: Path):
    """Test HEPH solid formulation (Isolid=24) with LAW60 through engine."""
    d = StarterDeck("test_heph_law60")
    d.funct(1, "YieldCurve", [(0.0, 250.0), (0.1, 350.0), (0.3, 450.0)])
    d.mat_law60(mid=1, title="SteelHEPH", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
    d.prop_solid(pid=1, title="PropHEPH", isolid=24)
    d.part(pid=1, title="PartHEPH", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
        (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_heph_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_heph_law60_0000.rad"
    eng_file = tmp_path / "test_heph_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert hasattr(model, "bricks_heph")
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_tetra4_solid_law60_engine_run(tmp_path: Path):
    """Test Tetra4 standard solid formulation with LAW60 through engine."""
    d = StarterDeck("test_tetra4_law60")
    d.funct(1, "YieldCurve", [(0.0, 250.0), (0.1, 350.0), (0.3, 450.0)])
    d.mat_law60(mid=1, title="SteelTetra", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
    d.prop_solid(pid=1, title="PropTetra")
    d.part(pid=1, title="PartTetra", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
    ]
    d.node(nodes)
    d.tetra4(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_tetra4_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_tetra4_law60_0000.rad"
    eng_file = tmp_path / "test_tetra4_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert hasattr(model, "tetras")
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_hexa8_solid_tension_compression_shear():
    """Verify Hexa8 3D stress responses under uniaxial tension, compression, and shear."""
    f1 = FunctTable(1, [0.0, 0.05, 0.20], [300.0, 400.0, 500.0])
    p = Law60Params(e0=200000.0, nu=0.3, nfunc=1, funcs=[f1])

    # 1. Uniaxial Tension: eps_xx = 0.05
    # Elastic strain to reach yield: eps_el ~ 300 / 200000 = 0.0015
    # Prescribe longitudinal tension increment: deps = [0.05, -0.015, -0.015, 0, 0, 0]
    sig0 = np.zeros(6, dtype=float)
    deps_t = np.array([0.05, -0.015, -0.015, 0.0, 0.0, 0.0])
    sig_t, epsp_t, _ = solid_update(p, sig0, deps_t, epsp_old=np.array([0.0]))

    assert sig_t[0] > 0.0
    epsp_t_val = float(np.squeeze(epsp_t))
    assert epsp_t_val > 0.0
    # In explicit radial return (sigeps60.F:534), trial stress is projected to YLD at epsp_old
    vm_t = np.sqrt(0.5 * ((sig_t[0] - sig_t[1])**2 + (sig_t[1] - sig_t[2])**2 + (sig_t[2] - sig_t[0])**2))
    assert np.isclose(vm_t, f1(0.0), rtol=1e-2)

    # In a subsequent increment, yield stress has hardened to f1(epsp_t_val)
    sig_t2, epsp_t2, _ = solid_update(p, sig_t, deps_t * 0.1, epsp_old=epsp_t)
    vm_t2 = np.sqrt(0.5 * ((sig_t2[0] - sig_t2[1])**2 + (sig_t2[1] - sig_t2[2])**2 + (sig_t2[2] - sig_t2[0])**2))
    assert np.isclose(vm_t2, f1(epsp_t_val), rtol=1e-2)
    assert float(np.squeeze(epsp_t2)) > epsp_t_val

    # 2. Uniaxial Compression: deps = [-0.05, +0.015, +0.015, 0, 0, 0]
    deps_c = np.array([-0.05, 0.015, 0.015, 0.0, 0.0, 0.0])
    sig_c, epsp_c, _ = solid_update(p, sig0, deps_c, epsp_old=np.array([0.0]))

    assert sig_c[0] < 0.0
    epsp_c_val = float(np.squeeze(epsp_c))
    assert epsp_c_val > 0.0
    assert np.isclose(epsp_c_val, epsp_t_val, rtol=1e-2)
    assert np.isclose(-sig_c[0], sig_t[0], rtol=1e-2)

    # 3. Pure Shear: deps = [0, 0, 0, 0.08, 0, 0] (engineering shear gamma_xy = 0.08)
    deps_s = np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.0])
    sig_s, epsp_s, _ = solid_update(p, sig0, deps_s, epsp_old=np.array([0.0]))

    assert sig_s[3] > 0.0  # Positive shear stress
    epsp_s_val = float(np.squeeze(epsp_s))
    assert epsp_s_val > 0.0
    # For pure shear, von Mises vm = sqrt(3) * tau_xy = yld
    tau_xy = sig_s[3]
    vm_s = np.sqrt(3.0) * tau_xy
    assert np.isclose(vm_s, f1(0.0), rtol=1e-2)


def test_hexa8_single_element_explicit_forces(tmp_path: Path):
    """Verify Hexa8 internal forces, equilibrium, and plastic strain accumulation."""
    d = StarterDeck("test_hexa8_forces")
    d.funct(1, "YieldCurve", [(0.0, 200.0), (0.05, 300.0), (0.2, 400.0)])
    d.mat_law60(mid=1, title="Steel", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
    d.prop_solid(pid=1, title="PropSolid", isolid=1)
    d.part(pid=1, title="Part1", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
    rad_file = tmp_path / "test_hexa8_forces_0000.rad"
    d.write(str(rad_file))

    model = run_starter(str(rad_file))
    group = model.bricks
    assert group.n == 1

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Pull right face (nodes 2, 3, 6, 7 at x=10) with velocity in +x into plasticity
    v[[1, 2, 5, 6], 0] = 5000.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    # Step through 25 cycles
    for _ in range(25):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        dtc = solid_hexa8.forces(group, model.x, v, model.vr, dt, fint, mint)

    # 1. Critical time step is strictly positive
    assert dtc[0] > 0.0

    # 2. Stability of nodal forces and stresses
    assert np.isfinite(fint).all()
    assert np.isfinite(group.state["sig"]).all()

    # 3. Global force equilibrium (sum of all 8 nodal internal forces is 0)
    np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

    # 4. Reaction signs: pulling face opposes motion, back face pulls forward
    assert fint[[1, 2, 5, 6], 0].sum() < 0.0
    assert fint[[0, 3, 4, 7], 0].sum() > 0.0

    # 5. Tensile stress and plastic strain developed
    assert group.state["sig"][0, 0] > 0.0
    assert group.state["epsp"][0] > 0.0


# ============================================================================
# 4. Shell Formulations: BT4, QEPH, Tri3 Integration
# ============================================================================

def test_quad_shell_bt4_law60_engine_run(tmp_path: Path):
    """Test Belytschko-Tsay Quad shell (BT4, Ishell=1) with LAW60 through engine."""
    d = StarterDeck("test_shell_bt4_law60")
    d.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 300.0)])
    d.mat_law60(mid=1, title="FabricBT4", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
    d.prop_shell(pid=1, title="PropBT4", ishell=1, thick=1.0, nip=3)
    d.part(pid=1, title="PartBT4", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_bt4_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_bt4_law60_0000.rad"
    eng_file = tmp_path / "test_shell_bt4_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert hasattr(model, "shells")
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_quad_shell_qeph_law60_engine_run(tmp_path: Path):
    """Test QEPH Quad shell (Ishell=24) with LAW60 through engine."""
    d = StarterDeck("test_shell_qeph_law60")
    d.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 300.0)])
    d.mat_law60(mid=1, title="FabricQEPH", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
    d.prop_shell(pid=1, title="PropQEPH", ishell=24, thick=1.0, nip=3)
    d.part(pid=1, title="PartQEPH", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_qeph_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_qeph_law60_0000.rad"
    eng_file = tmp_path / "test_shell_qeph_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert hasattr(model, "shells_qeph")
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_tri3_shell_law60_engine_run(tmp_path: Path):
    """Test Tri3 3-node triangular shell (Ish3n=1) with LAW60 through engine."""
    d = StarterDeck("test_tri3_law60")
    d.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 300.0)])
    d.mat_law60(mid=1, title="FabricTri3", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
    d.prop_shell(pid=1, title="PropTri3", ishell=1, thick=1.0, nip=3)
    d.part(pid=1, title="PartTri3", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 5.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.sh3n(1, [[1, 1, 2, 3]])

    eng_content = (
        "/RUN/test_tri3_law60/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_tri3_law60_0000.rad"
    eng_file = tmp_path / "test_tri3_law60_0001.rad"
    d.write(str(rad_file))
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert hasattr(model, "sh3n")
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_shell_plane_stress_and_thinning():
    """Verify plane-stress enforcement (sigma_zz=0) and thickness thinning under biaxial tension."""
    f1 = FunctTable(1, [0.0, 0.05, 0.20], [250.0, 320.0, 420.0])
    mat = build_law60(e0=210000.0, nu=0.3, nfunc=1, funcs=[f1])

    sig0 = np.zeros(3, dtype=float)
    deps = np.array([0.04, 0.04, 0.0])
    thk = np.array([1.5])
    extra = {"thk": thk, "off": np.array([1.0])}

    sig_new, epsp_new, c = shell_update(mat, sig0, deps, epsp_old=np.array([0.0]), extra=extra)

    # 1. Stresses are in plane [xx, yy, xy]
    assert sig_new[0] > 0.0
    assert sig_new[1] > 0.0
    assert np.isclose(sig_new[0], sig_new[1], rtol=1e-3)
    assert np.isclose(sig_new[2], 0.0, atol=1e-6)

    # 2. Thickness thinning: h < h0
    assert thk[0] < 1.5
    assert thk[0] > 0.0

    # 3. Plastic strain accumulated
    assert float(np.squeeze(epsp_new)) > 0.0


def test_shell_bt4_explicit_forces(tmp_path: Path):
    """Single BT4 shell explicit step under tension with plastic strain and thinning."""
    d = StarterDeck("test_shell_bt4_forces")
    d.funct(1, "YieldCurve", [(0.0, 200.0), (0.05, 300.0), (0.2, 400.0)])
    d.mat_law60(mid=1, title="Fabric", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
    d.prop_shell(pid=1, title="PropShell", ishell=1, thick=2.0, nip=3)
    d.part(pid=1, title="Part1", prop_id=1, mat_id=1)
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])
    rad_file = tmp_path / "test_shell_bt4_forces_0000.rad"
    d.write(str(rad_file))

    model = run_starter(str(rad_file))
    group = model.shells
    assert group.n == 1

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Pull right edge nodes 2 and 3 in +x into plasticity
    v[[1, 2], 0] = 5000.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    initial_thick = float(group.state["thick"][0])
    assert np.isclose(initial_thick, 2.0)

    for _ in range(25):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        dtc = shell_bt4.forces(group, model.x, v, model.vr, dt, fint, mint)

    # 1. Stable time step
    assert dtc[0] > 0.0

    # 2. Force equilibrium
    np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

    # 3. Plastic strain accumulation across layers
    assert np.any(group.state["epsp"][0] > 0.0)

    # 4. Thickness thinning
    assert group.state["thick"][0] < initial_thick


# ============================================================================
# 5. Element Deletion / Erosion
# ============================================================================

def test_solid_element_deletion_eps_max():
    """Verify solid element deletion when epsp reaches eps_max (zero stress and off=0.0)."""
    f1 = FunctTable(1, [0.0, 0.1], [300.0, 350.0])
    eps_limit = 0.08
    mat = build_law60(e0=200000.0, nu=0.3, eps_max=eps_limit, funcs=[f1])

    sig0 = np.zeros(6, dtype=float)
    off = np.array([1.0])
    off60 = np.array([1.0])
    extra = {"off": off, "off60": off60}

    # Step 1: Moderate strain below eps_max
    deps1 = np.array([0.02, -0.006, -0.006, 0.0, 0.0, 0.0])
    sig1, epsp1, _ = solid_update(mat, sig0, deps1, epsp_old=np.array([0.0]), extra=extra)
    assert float(np.squeeze(epsp1)) < eps_limit
    assert sig1[0] > 0.0
    assert off[0] == 1.0
    assert off60[0] == 1.0

    # Step 2: Large strain exceeding eps_max
    deps2 = np.array([0.15, -0.045, -0.045, 0.0, 0.0, 0.0])
    sig2, epsp2, _ = solid_update(mat, sig1, deps2, epsp_old=epsp1, extra=extra)
    assert float(np.squeeze(epsp2)) >= eps_limit
    assert np.allclose(sig2, 0.0)
    assert off[0] == 0.0
    assert off60[0] == 0.0


def test_shell_element_deletion_eps_max():
    """Verify shell layer failure and deletion when epsp reaches eps_max."""
    f1 = FunctTable(1, [0.0, 0.1], [250.0, 300.0])
    eps_limit = 0.05
    mat = build_law60(e0=200000.0, nu=0.3, eps_max=eps_limit, funcs=[f1])

    sig0 = np.zeros(3, dtype=float)
    off = np.array([1.0])
    layfail = np.array([1.0])
    extra = {"off": off, "layfail": layfail, "thk": np.array([1.0])}

    # Strain exceeding eps_max
    deps = np.array([0.10, 0.0, 0.0])
    sig_out, epsp_out, _ = shell_update(mat, sig0, deps, epsp_old=np.array([0.0]), extra=extra)

    assert float(np.squeeze(epsp_out)) >= eps_limit
    assert np.allclose(sig_out, 0.0)
    assert off[0] == 0.0
    assert layfail[0] == 0.0


# ============================================================================
# 6. Courant Time Step & Modulus Degradation
# ============================================================================

def test_courant_dt_degraded_modulus():
    """Verify sound speed reduction and Courant time step adjustment under modulus degradation."""
    f1 = FunctTable(1, [0.0, 0.1], [300.0, 350.0])
    # Exponential modulus degradation: ce=0.5, einf=0.05
    # E(epsp) = E0 * (1 - 0.5 * (1 - exp(-epsp / 0.05)))
    e0 = 200000.0
    nu = 0.3
    rho = 7.85e-9
    mat = build_law60(e0=e0, nu=nu, rho0=rho, ce=0.5, einf=0.05, funcs=[f1])

    # Undeformed rest sound speed
    c_solid_0 = sound_speed(mat, rho=rho)
    k0 = e0 / (3.0 * (1.0 - 2.0 * nu))
    g0 = e0 / (2.0 * (1.0 + nu))
    expected_c0 = np.sqrt((k0 + (4.0 / 3.0) * g0) / rho)
    assert np.isclose(c_solid_0, expected_c0, rtol=1e-4)

    # Shell undeformed sound speed
    c_shell_0 = mat.sound_speed_shell(rho=rho)
    expected_c_shell_0 = np.sqrt((e0 / (1.0 - nu * nu)) / rho)
    assert np.isclose(c_shell_0, expected_c_shell_0, rtol=1e-4)

    # At epsp = 0.10, modulus is degraded
    # D = 0.5 * (1 - exp(-0.10 / 0.05)) = 0.5 * (1 - exp(-2)) ~ 0.5 * (1 - 0.1353) ~ 0.4323
    # E_deg ~ 0.5677 * E0
    c_solid_deg = mat.sound_speed_solid(rho=rho, epsp=0.10)
    c_shell_deg = mat.sound_speed_shell(rho=rho, epsp=0.10)

    assert c_solid_deg < c_solid_0
    assert c_shell_deg < c_shell_0

    # Courant time step with characteristic length lc = 10.0 mm
    lc = 10.0
    dt_0 = lc / c_solid_0
    dt_deg = lc / c_solid_deg
    # Because wave speed decreases under damage, element dt limit increases safely
    assert dt_deg > dt_0


def test_courant_dt_curve_modulus_degradation():
    """Verify sound speed reduction when modulus degradation is controlled by /FUNCT curve."""
    f_yield = FunctTable(1, [0.0, 0.1], [300.0, 350.0])
    # Modulus curve f_e: factor multiplying E0 vs epsp
    f_e = FunctTable(2, [0.0, 0.05, 0.20], [1.0, 0.8, 0.5])
    functs = {1: f_yield, 2: f_e}

    mat = build_law60(
        e0=200000.0,
        nu=0.3,
        rho0=7.85e-9,
        ifunce=2,
        functs=functs,
    )

    c0 = mat.sound_speed_solid(rho=7.85e-9, epsp=0.0)
    c_half = mat.sound_speed_solid(rho=7.85e-9, epsp=0.20)

    # At epsp = 0.20, E factor is 0.5, so c should be approximately sqrt(0.5) * c0
    assert c_half < c0
    assert np.isclose(c_half / c0, np.sqrt(0.5), rtol=1e-2)


# ============================================================================
# 7. Algorithmic Consistent Tangent Wiring
# ============================================================================

def test_consistent_tangents_dispatch():
    """Verify materials.solid_tangent and shell_membrane_tangent dispatch for LAW60."""
    f1 = FunctTable(1, [0.0, 0.1], [300.0, 400.0])
    mat = build_law60(e0=210000.0, nu=0.3, nfunc=1, funcs=[f1])

    # 1. Shell membrane tangent (elastic 3x3)
    c_mem = materials.shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    assert c_mem[0, 0] > 0.0
    assert np.isclose(c_mem[0, 0], c_mem[1, 1])
    assert np.isclose(c_mem[0, 1], c_mem[1, 0])

    # 2. Consistent shell layer tangent (elastoplastic 3x3)
    sig = np.array([[300.0, 0.0, 0.0]])
    c_layer = materials.shell_layer_tangent(mat, sig=sig, epsp=np.array([0.05]), epsp_incr=0.001)
    assert c_layer.shape == (1, 3, 3)
    assert np.isfinite(c_layer).all()

    # 3. Consistent solid tangent (elastoplastic 6x6)
    sig_solid = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    c_solid = materials.solid_tangent(mat, sig=sig_solid, epsp=np.array([0.05]), epsp_incr=0.001)
    assert c_solid.shape == (1, 6, 6)
    assert np.isfinite(c_solid).all()
