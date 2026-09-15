"""Tests for Milestone M147: Failure Models (EMC, FABRIC, SPALLING, TBUTCHER,
WIERZBICKI, WILKINS) and Material Damping Sub-model (/MAT/VISC_PLAS).
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import resolve_materials


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_fail_emc_fixed_and_free(tmp_path):
    # Free format test
    free_deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/1/1
Steel
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/EMC/1
1.5 0.25 1.2 0.05
0.1 1.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, free_deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[1]
    assert mat.fail is not None
    assert mat.fail.type == "EMC"
    assert mat.fail.params["a_emc"] == 1.5
    assert mat.fail.params["n_emc"] == 0.25
    assert mat.fail.params["b0"] == 1.2
    assert mat.fail.params["c"] == 0.05
    assert mat.fail.params["gamma"] == 0.1
    assert mat.fail.params["eps_dot_0"] == 1.0e-5

    # Fixed format test
    fixed_deck = """# OpenRadioss Starter Deck
/BEGIN
Title
      2022         0
/MAT/PLAS_JOHNS/2/1
Steel2
            7.85e-09
            2.10e+05                 0.3
               200.0               400.0                 0.5
/FAIL/EMC/2
                 2.0                 0.3                 1.5                 0.1
                0.05              1.0e-4
/END
"""
    model2, log2 = _parse_starter(tmp_path, fixed_deck)
    assert len(log2.errors) == 0
    resolve_materials(model2, log2)
    mat2 = model2.materials[2]
    assert mat2.fail is not None
    assert mat2.fail.type == "EMC"
    assert mat2.fail.params["a_emc"] == 2.0
    assert mat2.fail.params["n_emc"] == 0.3
    assert mat2.fail.params["b0"] == 1.5
    assert mat2.fail.params["c"] == 0.1
    assert mat2.fail.params["gamma"] == 0.05
    assert mat2.fail.params["eps_dot_0"] == 1.0e-4


def test_fail_fabric(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/10/1
FabricMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/FABRIC/10
0.15 0.05 0.20 0.08 2 101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[10]
    assert mat.fail is not None
    assert mat.fail.type == "FABRIC"
    assert mat.fail.params["eps_f1"] == 0.15
    assert mat.fail.params["eps_r1"] == 0.05
    assert mat.fail.params["eps_f2"] == 0.20
    assert mat.fail.params["eps_r2"] == 0.08
    assert mat.fail.params["ndir"] == 2
    assert mat.fail.params["fct_id"] == 101


def test_fail_spalling(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/20/1
SpallMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/SPALLING/20
0.1 0.2 0.3 0.4 0.5
1.0e-3 -500.0 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[20]
    assert mat.fail is not None
    assert mat.fail.type == "SPALLING"
    assert mat.fail.params["D1"] == 0.1
    assert mat.fail.params["D2"] == 0.2
    assert mat.fail.params["D3"] == 0.3
    assert mat.fail.params["D4"] == 0.4
    assert mat.fail.params["D5"] == 0.5
    assert mat.fail.params["eps_dot_0"] == 1.0e-3
    assert mat.fail.params["p_min"] == -500.0
    assert mat.fail.params["ifail_so"] == 2


def test_fail_tbutcher(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/30/1
TButcherMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/TBUTCHER/30
0.5 150.0 350.0 1 2 1 0
0.2 0.8 0.90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[30]
    assert mat.fail is not None
    assert mat.fail.type == "TBUTCHER"
    assert mat.fail.ifail_sh == 1
    assert mat.fail.params["lambda"] == 0.5
    assert mat.fail.params["k"] == 150.0
    assert mat.fail.params["sigma_r"] == 350.0
    assert mat.fail.params["ifail_so"] == 2
    assert mat.fail.params["iduct"] == 1
    assert mat.fail.params["ixfem"] == 0
    assert mat.fail.params["a"] == 0.2
    assert mat.fail.params["b"] == 0.8
    assert mat.fail.params["dadv"] == 0.90


def test_fail_wierzbicki(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/40/1
WierzbickiMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/WIERZBICKI/40
1.1 2.2 3.3 4.4 0.75
0.5 1 2 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[40]
    assert mat.fail is not None
    assert mat.fail.type == "WIERZBICKI"
    assert mat.fail.ifail_sh == 1
    assert mat.fail.params["c1"] == 1.1
    assert mat.fail.params["c2"] == 2.2
    assert mat.fail.params["c3"] == 3.3
    assert mat.fail.params["c4"] == 4.4
    assert mat.fail.params["m"] == 0.75
    assert mat.fail.params["n"] == 0.5
    assert mat.fail.params["ifail_so"] == 2
    assert mat.fail.params["imoy"] == 1


def test_fail_wilkins(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/50/1
WilkinsMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/FAIL/WILKINS/50
0.4 1.2 500.0 0.85
1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    resolve_materials(model, log)
    mat = model.materials[50]
    assert mat.fail is not None
    assert mat.fail.type == "WILKINS"
    assert mat.fail.ifail_sh == 1
    assert mat.fail.params["alpha"] == 0.4
    assert mat.fail.params["beta"] == 1.2
    assert mat.fail.params["plim"] == 500.0
    assert mat.fail.params["df"] == 0.85
    assert mat.fail.params["ifail_so"] == 2


def test_visc_plas(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/60/1
ViscMat
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/MAT/VISC_PLAS/60
0.05 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 60 in model.visc_plas_models
    vp = model.visc_plas_models[60]
    assert vp.mat_id == 60
    assert vp.lsd_g == 0.05
    assert vp.lsdyna_sigf == 120.0

    resolve_materials(model, log)
    mat = model.materials[60]
    assert mat.visc_plas is not None
    assert mat.visc_plas.lsd_g == 0.05
    assert mat.visc_plas.lsdyna_sigf == 120.0


def test_visc_plas_alias(tmp_path):
    deck = """# OpenRadioss Starter Deck
/BEGIN
Title
/MAT/PLAS_JOHNS/70/1
ViscMat2
7.85e-9
210000.0 0.3
200.0 400.0 0.5
/VISC/PLAS/70
0.10 250.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 70 in model.visc_plas_models
    vp = model.visc_plas_models[70]
    assert vp.mat_id == 70
    assert vp.lsd_g == 0.10
    assert vp.lsdyna_sigf == 250.0

    resolve_materials(model, log)
    mat = model.materials[70]
    assert mat.visc_plas is not None
    assert mat.visc_plas.lsd_g == 0.10
    assert mat.visc_plas.lsdyna_sigf == 250.0

