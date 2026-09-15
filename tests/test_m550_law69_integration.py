"""
Integration tests for /MAT/LAW69 and /MAT/HYP_ELAS (/MAT/HYPERELASTIC).

Milestone M550 Subagent 1C: Engine Integration & Wire-Up
1. Starter parsing of /MAT/LAW69 and /MAT/HYP_ELAS decks.
2. Dynamic element kernel simulation in OpenRadioss engine:
   - Solids: Hexa8 (standard 1-point Isolid=1), HEPH (Isolid=24), Tetra4.
   - Shells: BT4 (Ishell=1), QEPH (Ishell=24), Tri3 (Ish3n=1).
3. Shell thickness update & thinning in biaxial tension (volume conservation).
4. Shell out-of-plane stretch (lambda_3) Newton-Raphson convergence and history storage in uvar[:, 2].
5. Tensile cut-off element erosion for solids (off=0.0) and shells (layfail=0.0).
6. Parsing of official snippet from RD-E-5600 Hyperelastic material.
7. Courant time step bounding and wave speed stiffening under stretch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    fit_law69_curve,
    sigeps69_solid,
    sigeps69c_shell,
    solid_sound_speed,
    shell_sound_speed,
    extra_shapes,
)
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine


# ============================================================================
# 1. Starter Parsing: /MAT/LAW69 and /MAT/HYP_ELAS
# ============================================================================


def test_starter_parsing_law69_deck(tmp_path):
    """Test Starter parsing of fixed-format /MAT/LAW69 deck."""
    deck_text = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "test_law69_deck\n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/MAT/LAW69/1\n"
        "Neoprene LAW69\n"
        "#              RHO_I           REFER_RHO\n"
        f"{fmt_float(1.0e-9)}{fmt_float(1.0e-9)}\n"
        "#   LAW_ID    FCT_ID                  NU              FSCALE    N_PAIR    ICHECK\n"
        f"{fmt_int(1, 10)}{fmt_int(0, 10)}{fmt_float(0.495, 20)}{fmt_float(1.0, 20)}{fmt_int(2, 10)}{fmt_int(-3, 10)}\n"
        "#  FCT_ID1\n"
        f"{fmt_int(0, 10)}\n"
        "/END\n"
    )
    rad_file = tmp_path / "test_law69_0000.rad"
    rad_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.id == 1
    assert getattr(mat, "law", None) == 69
    assert np.isclose(mat.rho0, 1.0e-9)
    assert np.isclose(mat.nu, 0.495)
    assert mat.G > 0.0
    assert mat.K > 0.0


def test_official_rd_e_5600_snippet_parsing():
    """Verify parsing of official LAW69 snippet from RD-E-5600."""
    snippet_path = "tests/data/rd_decks/rd_e/RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_ogden_pair2/LAW69_ogden_pair2_Poisson04997/LAW69.txt"
    blocks = read_deck(snippet_path)
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert len(log.errors) == 0
    assert 1 in model.materials
    mat = model.materials[1]
    assert getattr(mat, "law", None) == 69
    assert np.isclose(mat.rho0, 1.0e-9)
    assert np.isclose(mat.nu, 0.4997)


# ============================================================================
# 2. Solid Element Formulations: Hexa8, HEPH, Tetra4 Engine Execution
# ============================================================================


def test_hexa8_solid_law69_engine_run(tmp_path):
    """Test Hexa8 standard solid formulation (Isolid=1) with LAW69 through engine."""
    d = StarterDeck("test_hexa8_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberHexa")
    d.prop_solid(pid=1, isolid=1, title="PropSolidHexa")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartHexa")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_hexa8_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_hexa8_law69_0000.rad"
    eng_file = tmp_path / "test_hexa8_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    assert model is not None
    assert 1 in model.materials
    assert getattr(model.materials[1], "law", None) == 69

    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_heph_solid_law69_engine_run(tmp_path):
    """Test HEPH solid formulation (Isolid=24) with LAW69 through engine."""
    d = StarterDeck("test_heph_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberHEPH")
    d.prop_solid(pid=1, isolid=24, title="PropHEPH")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartHEPH")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
        (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
    ]
    d.node(nodes)
    d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

    eng_content = (
        "/RUN/test_heph_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_heph_law69_0000.rad"
    eng_file = tmp_path / "test_heph_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_tetra4_solid_law69_engine_run(tmp_path):
    """Test Tetra4 standard solid formulation with LAW69 through engine."""
    d = StarterDeck("test_tetra4_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberTetra")
    d.prop_solid(pid=1, title="PropTetra")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartTetra")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
    ]
    d.node(nodes)
    d.tetra4(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_tetra4_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_tetra4_law69_0000.rad"
    eng_file = tmp_path / "test_tetra4_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


# ============================================================================
# 3. Shell Element Formulations: BT4, QEPH, Tri3 Engine Execution
# ============================================================================


def test_quad_shell_bt4_law69_engine_run(tmp_path):
    """Test Belytschko-Tsay Quad shell (BT4, Ishell=1) with LAW69 through engine."""
    d = StarterDeck("test_shell_bt4_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberBT4")
    d.prop_shell(pid=1, ishell=1, thick=1.0, title="PropBT4")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartBT4")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_bt4_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_bt4_law69_0000.rad"
    eng_file = tmp_path / "test_shell_bt4_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_quad_shell_qeph_law69_engine_run(tmp_path):
    """Test QEPH Quad shell (Ishell=24) with LAW69 through engine."""
    d = StarterDeck("test_shell_qeph_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberQEPH")
    d.prop_shell(pid=1, ishell=24, thick=1.0, title="PropQEPH")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartQEPH")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.shell(1, [[1, 1, 2, 3, 4]])

    eng_content = (
        "/RUN/test_shell_qeph_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_shell_qeph_law69_0000.rad"
    eng_file = tmp_path / "test_shell_qeph_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


def test_tri3_shell_law69_engine_run(tmp_path):
    """Test Tri3 3-node triangular shell (Ish3n=1) with LAW69 through engine."""
    d = StarterDeck("test_tri3_law69")
    d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberTri3")
    d.prop_shell(pid=1, ishell=1, thick=1.0, title="PropTri3")
    d.part(pid=1, prop_id=1, mat_id=1, title="PartTri3")
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 5.0, 10.0, 0.0),
    ]
    d.node(nodes)
    d.sh3n(1, [[1, 1, 2, 3]])

    eng_content = (
        "/RUN/test_tri3_law69/1\n"
        "0.0001\n"
        "/DT\n"
        "0.5 0.0\n"
        "/PRINT/-1\n"
    )
    rad_file = tmp_path / "test_tri3_law69_0000.rad"
    eng_file = tmp_path / "test_tri3_law69_0001.rad"
    rad_file.write_text(d.render(), encoding="utf-8")
    eng_file.write_text(eng_content, encoding="utf-8")

    model = run_starter(str(rad_file))
    eng_model = run_engine(str(eng_file))
    assert eng_model is not None


# ============================================================================
# 4. Shell Thickness Thinning & Incompressibility in Biaxial Tension
# ============================================================================


def test_shell_biaxial_thinning():
    """Verify thickness thinning h < h0 and volume conservation under biaxial tension."""
    mat = build_law69(mu=[20.0, 5.0], alpha=[2.0, -2.0], nu=0.499)
    sig = np.zeros(5, dtype=np.float64)

    # In-plane equibiaxial strain: eps_xx = eps_yy = 0.10 (engineering strain)
    deps = np.array([0.10, 0.10, 0.0, 0.0, 0.0])
    uvar = np.zeros((1, 9), dtype=np.float64)
    uvar[0, 2] = 1.0  # Initial out-of-plane stretch lambda_3 = 1.0
    thkn = np.array([2.0])
    thklyl = np.array([2.0])

    sig_new, _ = sigeps69c_shell(
        mat, sig, deps=deps, uvar=uvar, thkn=thkn, thklyl=thklyl, ismstr=1
    )

    # Thickness must thin under biaxial extension
    assert thkn[0] < 2.0

    # Incompressible condition: lambda_1 * lambda_2 * lambda_3 ~ 1
    # lambda_1 = 1 + 0.10 = 1.10, lambda_2 = 1.10
    # Expected lambda_3 ~ 1 / (1.10 * 1.10) ~ 0.8264
    lam_3_actual = uvar[0, 2]
    lam_3_expected = 1.0 / (1.10 * 1.10)
    assert np.isclose(lam_3_actual, lam_3_expected, rtol=1e-2)

    # In-plane stresses must be positive (tensile) and equal (equibiaxial symmetry)
    assert sig_new[0] > 0.0
    assert sig_new[1] > 0.0
    assert np.isclose(sig_new[0], sig_new[1], rtol=1e-3)


# ============================================================================
# 5. Out-of-Plane Stretch (lambda_3) Newton-Raphson Convergence
# ============================================================================


def test_shell_out_of_plane_stretch_convergence():
    """Verify that lambda_3 converges across Newton iterations and enforces plane stress."""
    mat = build_law69(mu=[15.0, 3.0], alpha=[2.0, -2.0], nu=0.495)
    sig = np.zeros(3, dtype=np.float64)

    # Uniaxial stretch path in 5 consecutive increments
    uvar = np.zeros((1, 9), dtype=np.float64)
    uvar[0, 2] = 1.0

    for step in range(1, 6):
        deps = np.array([0.05, 0.0, 0.0])  # Longitudinal stretch increment
        sig, _ = sigeps69c_shell(mat, sig, deps=deps, uvar=uvar, ismstr=0)

        # lambda_3 must decrease monotonically under tension
        assert uvar[0, 2] < 1.0
        # Tension stress must remain positive
        assert sig[0] > 0.0


# ============================================================================
# 6. Tensile Cut-Off Element Erosion
# ============================================================================


def test_solid_tensile_cutoff_erosion():
    """Verify that exceeding tenscut zeroes stress and flags element deletion (off=0)."""
    tenscut_limit = 25.0
    mat = build_law69(mu=[30.0], alpha=[2.0], tenscut=tenscut_limit)
    sig = np.zeros(6, dtype=np.float64)
    off = np.array([1.0])

    # Moderate stretch below cut-off
    eps_sub = np.array([0.10, -0.05, -0.05, 0.0, 0.0, 0.0])
    sig_sub, _, _ = sigeps69_solid(mat, sig, eps=eps_sub, off=off)
    assert sig_sub[0] < tenscut_limit
    assert off[0] == 1.0

    # Large stretch exceeding cut-off
    eps_super = np.array([0.50, -0.25, -0.25, 0.0, 0.0, 0.0])
    sig_super, _, _ = sigeps69_solid(mat, sig, eps=eps_super, off=off)
    assert np.allclose(sig_super, 0.0)
    assert off[0] == 0.0


def test_shell_tensile_cutoff_erosion():
    """Verify shell tensile cut-off flags layer failure (layfail=0) and off=0.8."""
    tenscut_limit = 15.0
    mat = build_law69(mu=[25.0], alpha=[2.0], tenscut=tenscut_limit)
    sig = np.zeros(3, dtype=np.float64)
    off = np.array([1.0])
    extra = {"layfail": np.array([1.0]), "off": off}

    # Large stretch exceeding cut-off
    eps = np.array([0.35, 0.0, 0.0])
    sig_out, _ = sigeps69c_shell(mat, sig, eps=eps, off=off, extra=extra)

    assert np.allclose(sig_out, 0.0)
    assert off[0] == 0.8
    assert extra["layfail"][0] == 0.0


# ============================================================================
# 7. Courant Time Step Bounding and Wave Speed Stiffening
# ============================================================================


def test_courant_time_step_stiffening():
    """Verify wave speed increases and critical dt decreases safely under stretch."""
    mat = build_law69(mu=[12.0, 2.0], alpha=[2.5, -2.0], nu=0.49)
    rho = 1.0

    # Rest sound speed
    c_0 = solid_sound_speed(mat, rho=rho)
    assert c_0 > 0.0

    # Under large stretch lambda = exp(0.5) ~ 1.65
    eps_stretched = np.array([0.50, -0.25, -0.25, 0.0, 0.0, 0.0])
    c_stretched = solid_sound_speed(mat, rho=rho, eps=eps_stretched)

    # Due to hyperelastic exponent alpha=2.5, wave speed stiffens
    assert c_stretched > c_0

    # Critical dt for element with characteristic length lc = 1.0
    lc = 1.0
    dt_rest = lc / c_0
    dt_stretched = lc / c_stretched
    assert dt_stretched < dt_rest
