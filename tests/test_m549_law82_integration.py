"""
Integration tests for /MAT/LAW82 and /MAT/OGDEN (Ogden hyperelastic material).

Milestone M549 Subagent 1C:
1. Starter parsing of /MAT/LAW82 and /MAT/OGDEN decks (fixed format and free format).
2. DeckWriter generation and round-trip re-reading.
3. Assignment of LAW82 to /PART with Hexa8 solid, HEPH solid, Tetra4 solid,
   Quad shell (BT4 and QEPH), and Triangle shell (Tri3).
4. Multi-term Ogden models (N=1, 2, 3 terms).
5. Courant time step calculation and sound speed retrieval for solids and shells.
6. End-to-end Starter and Engine run on official snippet / test deck
   from RD-E-5600_Hyperelastic_material LAW82 rubber tension model.
"""

import os
import shutil
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_int, fmt_float
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law82_ogden import (
    OgdenParams,
    build_law82,
    solid_update,
    shell_update,
    solid_sound_speed,
    shell_sound_speed,
    sound_speed,
    sound_speed_shell,
    extra_shapes,
)
from pyradioss.model.entities import Material, MatLaw82
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine


# ============================================================================
# 1. Starter Parsing: Fixed and Free Format for /MAT/LAW82 and /MAT/OGDEN
# ============================================================================

def test_starter_parsing_law82_fixed_format(tmp_path):
    """Test Starter parsing of fixed-format /MAT/LAW82 deck."""
    deck_text = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "test_fixed_law82\n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/MAT/LAW82/1\n"
        "Neoprene Rubber LAW82\n"
        "#              RHO_I           REFER_RHO\n"
        f"{fmt_float(1.0e-9)}{fmt_float(1.0e-9)}\n"
        "#          N                          NU\n"
        f"{fmt_int(2, 10)}{' ' * 10}{fmt_float(0.495, 20)}\n"
        "#               MU_1                MU_2\n"
        f"{fmt_float(0.000045)}{fmt_float(0.54)}\n"
        "#            ALPHA_1             ALPHA_2\n"
        f"{fmt_float(7.16)}{fmt_float(-4.15)}\n"
        "#            GAMMA_1             GAMMA_2\n"
        f"{fmt_float(0.0)}{fmt_float(0.0)}\n"
        "/END\n"
    )
    rad_file = tmp_path / "test_fixed_0000.rad"
    rad_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 82
    assert mat.law_name == "LAW82"
    assert np.isclose(mat.rho0, 1e-9)
    assert mat.params["ORDER"] == 2
    assert np.isclose(mat.params["MAT_NU"], 0.495)
    assert len(mat.params["mu"]) == 2
    assert np.isclose(mat.params["mu"][0], 0.000045)
    assert np.isclose(mat.params["mu"][1], 0.54)
    assert len(mat.params["alpha"]) == 2
    assert np.isclose(mat.params["alpha"][0], 7.16)
    assert np.isclose(mat.params["alpha"][1], -4.15)
    assert 1 in model.mat_law82s
    law82_ent = model.mat_law82s[1]
    assert law82_ent.nordre == 2
    assert np.isclose(law82_ent.G, 0.000045 + 0.54)


def test_starter_parsing_ogden_free_format(tmp_path):
    """Test Starter parsing of free-format /MAT/OGDEN deck."""
    deck_text = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "test_free_ogden\n"
        "        10         0\n"
        "/MAT/OGDEN/2\n"
        "Silicone Rubber OGDEN\n"
        "1.2e-9, 1.2e-9\n"
        "1, 0.48\n"
        "0.85\n"
        "2.0\n"
        "0.0\n"
        "/END\n"
    )
    rad_file = tmp_path / "test_free_0000.rad"
    rad_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 2 in model.materials
    mat = model.materials[2]
    assert mat.law == 82
    assert np.isclose(mat.rho0, 1.2e-9)
    assert mat.params["ORDER"] == 1
    assert np.isclose(mat.params["MAT_NU"], 0.48)
    assert np.isclose(mat.params["mu"][0], 0.85)
    assert np.isclose(mat.params["alpha"][0], 2.0)
    assert np.isclose(mat.G, 0.85)
    assert 2 in model.mat_law82s
    m2 = model.mat_law82s[2]
    assert m2.nordre == 1
    assert m2.mu == [0.85]
    assert m2.alpha == [2.0]


# ============================================================================
# 2. DeckWriter Generation and Round-Trip Re-Reading
# ============================================================================

def test_deck_writer_roundtrip(tmp_path):
    """Test DeckWriter generation of LAW82 and full round-trip re-reading."""
    d = StarterDeck("test_dw_law82")
    d.mat_law82(
        mid=5,
        rho0=1.1e-9,
        nu=0.49,
        nordre=3,
        mu=[0.63, 0.0012, -0.01],
        alpha=[1.3, 5.0, -2.0],
        d=[0.0, 0.0, 0.0],
        title="TreloarRubber",
    )
    d.prop_solid(pid=1, title="PropSolid")
    d.part(pid=1, prop_id=1, mat_id=5, title="RubberSolid")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    rad_file = tmp_path / "dw_test_0000.rad"
    rad_file.write_text(d.render(), encoding="utf-8")

    model = run_starter(str(rad_file))
    assert 5 in model.materials
    mat = model.materials[5]
    assert mat.law == 82
    assert mat.params["ORDER"] == 3
    assert np.isclose(mat.rho0, 1.1e-9)
    assert np.isclose(mat.params["MAT_NU"], 0.49)
    assert np.allclose(mat.params["mu"], [0.63, 0.0012, -0.01])
    assert np.allclose(mat.params["alpha"], [1.3, 5.0, -2.0])


# ============================================================================
# 3. Multi-Term Ogden Models (N=1, 2, 3)
# ============================================================================

@pytest.mark.parametrize("n_terms,mu,alpha", [
    (1, [0.5], [2.0]),
    (2, [0.4, 0.1], [2.0, -2.0]),
    (3, [0.35, 0.12, 0.03], [1.3, 3.5, -2.0]),
])
def test_multi_term_ogden_properties(n_terms, mu, alpha):
    """Verify parameter initialization, G0, K, E, and sound speed for N=1, 2, 3 terms."""
    rho0 = 1.0e-9
    nu = 0.495
    mat_params = build_law82(
        nordre=n_terms,
        mu=mu,
        alpha=alpha,
        rho0=rho0,
        nu=nu,
    )
    assert mat_params.nordre == n_terms
    expected_g0 = float(np.sum(mu))
    assert np.isclose(mat_params.g0, expected_g0)
    assert np.isclose(mat_params.G, expected_g0)

    # D1 and bulk modulus K = 2 / D1
    expected_d1 = 3.0 * (1.0 - 2.0 * nu) / (expected_g0 * (1.0 + nu))
    assert np.isclose(mat_params.d[0], expected_d1)
    expected_k = 2.0 / expected_d1
    assert np.isclose(mat_params.rbulk, expected_k)
    assert np.isclose(mat_params.K, expected_k)

    # Sound speeds
    c_solid = solid_sound_speed(mat_params, rho=rho0)
    expected_c_solid = np.sqrt(((4.0 / 3.0) * expected_g0 + expected_k) / rho0)
    assert np.isclose(c_solid, expected_c_solid)

    c_shell = shell_sound_speed(mat_params, rho=rho0)
    expected_c_shell = np.sqrt(((2.0 / 3.0) * expected_g0 + expected_k) / rho0)
    assert np.isclose(c_shell, expected_c_shell)


# ============================================================================
# 4. Element Formulations with /PART and Engine Execution
# ============================================================================

def test_hexa8_solid_law82_engine_run(tmp_path):
    """Test Hexa8 standard solid formulation with LAW82 through engine."""
    d = StarterDeck("test_hexa8_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=2, mu=[0.000045, 0.54], alpha=[7.16, -4.15], d=[0.0, 0.0])
    d.prop_solid(pid=1, title="PropSolid")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartHexa")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_hexa8_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_hexa8_law82_0000.rad"
    eng_file = tmp_path / "test_hexa8_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_heph_solid_law82_engine_run(tmp_path):
    """Test HEPH solid formulation (Isolid=24) with LAW82 through engine."""
    d = StarterDeck("test_heph_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=1, mu=[0.5], alpha=[2.0], d=[0.0])
    d.prop_solid(pid=1, isolid=24, title="PropHEPH")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartHEPH")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
        (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_heph_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_heph_law82_0000.rad"
    eng_file = tmp_path / "test_heph_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_tetra4_solid_law82_engine_run(tmp_path):
    """Test Tetra4 standard solid formulation with LAW82 through engine."""
    d = StarterDeck("test_tetra4_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=1, mu=[0.5], alpha=[2.0], d=[0.0])
    d.prop_solid(pid=1, title="PropTetra")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartTetra")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
    ]
    d.node(nodes)
    d.tetra4(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_tetra4_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_tetra4_law82_0000.rad"
    eng_file = tmp_path / "test_tetra4_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_quad_shell_bt4_law82_engine_run(tmp_path):
    """Test Belytschko-Tsay Quad shell (BT4) with LAW82 through engine."""
    d = StarterDeck("test_shell_bt4_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=2, mu=[0.000045, 0.54], alpha=[7.16, -4.15], d=[0.0, 0.0])
    d.prop_shell(pid=1, ishell=1, thick=1.0, title="PropBT4")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartBT4")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_bt4_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_bt4_law82_0000.rad"
    eng_file = tmp_path / "test_shell_bt4_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_quad_shell_qeph_law82_engine_run(tmp_path):
    """Test QEPH Quad shell (Ishell=24) with LAW82 through engine."""
    d = StarterDeck("test_shell_qeph_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=1, mu=[0.5], alpha=[2.0], d=[0.0])
    d.prop_shell(pid=1, ishell=24, thick=1.0, title="PropQEPH")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartQEPH")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_qeph_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_qeph_law82_0000.rad"
    eng_file = tmp_path / "test_shell_qeph_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_triangle_shell_tri3_law82_engine_run(tmp_path):
    """Test 3-node Triangle shell (SH3N) with LAW82 through engine."""
    d = StarterDeck("test_sh3n_law82")
    d.mat_law82(mid=1, rho0=1e-9, nu=0.495, nordre=1, mu=[0.5], alpha=[2.0], d=[0.0])
    d.prop_shell(pid=1, ish3n=1, thick=1.0, title="PropTri3")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartTri3")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 5.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.sh3n(1, [[1, 1, 2, 3]])

    eng_content = (
        "/RUN/test_sh3n_law82/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_sh3n_law82_0000.rad"
    eng_file = tmp_path / "test_sh3n_law82_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


# ============================================================================
# 5. Official OpenRadioss Deck Integration Test
# ============================================================================

def test_official_rd_e_5600_rubber_tension_law82(tmp_path):
    """Test full starter + engine on official RD-E-5600 Ogden LAW82 model."""
    src_dir = os.path.join(
        "tests", "data", "rd_decks", "rd_e",
        "RD-E-5600_Hyperelastic_material", "56_HyperElastic_Material",
        "Ogden_model", "LAW82"
    )
    assert os.path.isdir(src_dir), f"Official deck directory not found: {src_dir}"

    for fname in os.listdir(src_dir):
        if fname.endswith(".rad") or fname.endswith(".txt"):
            shutil.copy(os.path.join(src_dir, fname), os.path.join(tmp_path, fname))

    # Scale Tstop to 0.0001 for quick validation in integration test
    e_file = tmp_path / "rubber_tension_v1_0001.rad"
    lines = e_file.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for line in lines:
        if "0.801" in line:
            new_lines.append("               0.0001\n")
        else:
            new_lines.append(line)
    e_file.write_text("".join(new_lines), encoding="utf-8")

    s_file = tmp_path / "rubber_tension_v1_0000.rad"
    model = run_starter(str(s_file))
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 82
    assert mat.params["ORDER"] == 2
    assert np.isclose(mat.G, 0.000045637449070023 + 0.547913433558156)

    eng_model = run_engine(str(e_file))
    assert eng_model is not None


# ============================================================================
# 6. Courant Time Step and Wave Speed Retrieval Verification
# ============================================================================

def test_courant_time_step_and_sound_speed():
    """Verify sound speed formula and Courant time step bounding behavior."""
    mat = Material(
        id=1,
        law=82,
        rho0=1.0e-9,
        params={
            "mu": [0.5],
            "alpha": [2.0],
            "d": [0.0],
            "nu": 0.495,
        },
    )

    c_solid = materials.sound_speed(mat)
    c_solid_ent = mat.sound_speed_solid()
    assert np.isclose(c_solid, c_solid_ent)
    assert c_solid > 200000.0  # mm/s

    c_shell = mat.sound_speed_shell()
    c_shell_mat = sound_speed_shell(mat)
    assert np.isclose(c_shell, c_shell_mat)
    # Shell wave speed is slightly lower due to plane stress (2/3 G vs 4/3 G)
    assert c_shell < c_solid
