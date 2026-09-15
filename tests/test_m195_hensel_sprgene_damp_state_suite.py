"""Tests for Milestone M195: Hensel-Spittel Hot-Forming, 6-DOF Generalized Springs,
Predefined Plasticity, Drucker-Prager 2nd Formulation, Spring Material Properties,
Smoothed Curves, Oriented Friction, Frequency/Function Damping, Initial State Tables,
Checksums and Engine Output Directives.
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_mat_law103_hensel_spittel(tmp_path: Path):
    """Verify /MAT/LAW103 (/MAT/HENSEL_SPITTEL) parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Law 103 Hensel-Spittel
                  10
/MAT/LAW103/101
Hot Forming Steel Law 103
#  RHO                 RHO_REF
   7.8e-6              7.8e-6
#  E                   NU
   210000.0            0.3
#  A0                  M1                  M2                  M3                  M4
   500.0               -0.001              0.15                0.05                0.0
#  M5                  M7
   -0.002              0.01
#  FSMOOTH             FCUT                EPS_0               PMIN
   1                   500.0               0.001               -1.0e20
#  RHOCP               T0                  ETA
   3.5e-3              293.15              0.9
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 101 in model.mat_law103s
    mat103 = model.mat_law103s[101]
    assert mat103.id == 101
    assert "Hot Forming" in mat103.title
    assert pytest.approx(mat103.rho) == 7.8e-6
    assert pytest.approx(mat103.e) == 210000.0
    assert pytest.approx(mat103.nu) == 0.3
    assert pytest.approx(mat103.a0) == 500.0
    assert pytest.approx(mat103.m1) == -0.001
    assert pytest.approx(mat103.m2) == 0.15
    assert pytest.approx(mat103.m3) == 0.05
    assert pytest.approx(mat103.m5) == -0.002
    assert pytest.approx(mat103.m7) == 0.01
    assert mat103.fsmooth == 1
    assert pytest.approx(mat103.fcut) == 500.0
    assert pytest.approx(mat103.eps_0) == 0.001
    assert pytest.approx(mat103.rhocp) == 3.5e-3
    assert pytest.approx(mat103.t0) == 293.15
    assert pytest.approx(mat103.eta) == 0.9

    assert 101 in model.materials
    assert model.materials[101].law == 103


def test_mat_law108_spr_gene(tmp_path: Path):
    """Verify /MAT/LAW108 (/MAT/SPR_GENE) parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Law 108 Generalized Spring
                  10
/MAT/LAW108/201
Generalized Spring Material
#  RHO
   1.5e-6
#  IFAIL               IEQUIL              IFAIL2
   1                   0                   2
# DOF 1: Tx
#  K1                  C1                  A1                  B1                  D1
   100.0               0.05                0.0                 0.0                 0.0
#  FCT_ID11            H1                  FCT_ID21            FCT_ID31            FCT_ID41            DMIN1               DMAX1
   10                  0.0                 0                   0                   0                   -10.0               10.0
#  F1                  E1                  ASCALE1             HSCALE1
   500.0               0.0                 1.0                 1.0
# DOF 2: Ty
   200.0               0.1                 0.0                 0.0                 0.0
   0                   0.0                 0                   0                   0                   -5.0                5.0
   300.0               0.0                 1.0                 1.0
# DOF 3: Tz
   300.0               0.15                0.0                 0.0                 0.0
   0                   0.0                 0                   0                   0                   -5.0                5.0
   300.0               0.0                 1.0                 1.0
# DOF 4: Rx
   50.0                0.01                0.0                 0.0                 0.0
   0                   0.0                 0                   0                   0                   -1.0                1.0
   50.0                0.0                 1.0                 1.0
# DOF 5: Ry
   60.0                0.02                0.0                 0.0                 0.0
   0                   0.0                 0                   0                   0                   -1.0                1.0
   60.0                0.0                 1.0                 1.0
# DOF 6: Rz
   70.0                0.03                0.0                 0.0                 0.0
   0                   0.0                 0                   0                   0                   -1.0                1.0
   70.0                0.0                 1.0                 1.0
#  FSMOOTH             FCUT
   1                   1000.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 201 in model.mat_law108s
    mat108 = model.mat_law108s[201]
    assert mat108.id == 201
    assert pytest.approx(mat108.rho) == 1.5e-6
    assert mat108.ifail == 1
    assert mat108.iequil == 0
    assert mat108.ifail2 == 2
    assert pytest.approx(mat108.k[0]) == 100.0
    assert pytest.approx(mat108.c[0]) == 0.05
    assert mat108.fct_id1[0] == 10
    assert pytest.approx(mat108.delta_min[0]) == -10.0
    assert pytest.approx(mat108.delta_max[0]) == 10.0
    assert pytest.approx(mat108.f_val[0]) == 500.0
    assert pytest.approx(mat108.k[1]) == 200.0
    assert pytest.approx(mat108.k[5]) == 70.0
    assert mat108.fsmooth == 1
    assert pytest.approx(mat108.fcut) == 1000.0

    assert 201 in model.materials
    assert model.materials[201].law == 108


def test_mat_plas_predef(tmp_path: Path):
    """Verify /MAT/PLAS_PREDEF predefined plasticity model."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Predefined Plasticity
                  10
/MAT/PLAS_PREDEF/301
Predefined Steel
STEEL
/MAT/PLAS_PREDEF/302
Custom Predefined Plasticity
#  RHO                 E                   NU
   2.7e-6              70000.0             0.33
#  SIGY                UTS                 E_UTS               EPSP_F              VP                  C                   P                   N
   200.0               350.0               0.20                0.30                1.0                 50.0                2.0                 5
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 301 in model.mat_plas_predefs
    m1 = model.mat_plas_predefs[301]
    assert m1.mat_name == "STEEL"
    assert pytest.approx(m1.rho) == 7.8e-6
    assert pytest.approx(m1.e) == 210.0

    assert 302 in model.mat_plas_predefs
    m2 = model.mat_plas_predefs[302]
    assert pytest.approx(m2.rho) == 2.7e-6
    assert pytest.approx(m2.e) == 70000.0
    assert pytest.approx(m2.sigy) == 200.0
    assert pytest.approx(m2.uts) == 350.0
    assert pytest.approx(m2.e_uts) == 0.20
    assert pytest.approx(m2.epsp_f) == 0.30
    assert pytest.approx(m2.c) == 50.0
    assert pytest.approx(m2.p) == 2.0
    assert m2.n == 5


def test_mat_dprag2(tmp_path: Path):
    """Verify /MAT/DPRAG2 (Drucker-Prager 2nd formulation)."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test DPRAG2
                  10
/MAT/DPRAG2/401
Drucker Prager 2nd
#  RHO                 E                   NU
   2.5e-6              35000.0             0.25
#  A0                  A1                  B0                  B1                  ICRIT
   10.0                0.5                 5.0                 0.2                 2
#  E                   NU
   35000.0             0.25
#  C                   PHI                 AMAX
   20.0                30.0                100.0
#  PMIN
   -50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 401 in model.mat_dprag2s
    m = model.mat_dprag2s[401]
    assert m.id == 401
    assert pytest.approx(m.rho) == 2.5e-6
    assert pytest.approx(m.e) == 35000.0
    assert pytest.approx(m.nu) == 0.25
    assert pytest.approx(m.c) == 20.0
    assert pytest.approx(m.phi) == 30.0
    assert pytest.approx(m.amax) == 100.0
    assert pytest.approx(m.pmin) == -50.0


def test_prop_type23_spr_mat(tmp_path: Path):
    """Verify /PROP/TYPE23 (/PROP/SPR_MAT) parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Spring Material Property
                  10
/PROP/TYPE23/501
Spring Material Prop
#  MASS                SKEW_ID             ISENS               IFLAG
   0.025               12                  1                   2
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 501 in model.prop_type23s
    p = model.prop_type23s[501]
    assert p.id == 501
    assert pytest.approx(p.mass) == 0.025
    assert p.skew_id == 12
    assert p.isens == 1
    assert p.iflag == 2
    assert 501 in model.properties
    assert model.properties[501].type == 23


def test_funct_smooth(tmp_path: Path):
    """Verify /FUNCT_SMOOTH parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Smoothed Function
                  10
/FUNCT_SMOOTH/601
Smooth Stress Strain Curve
#  XSC                 YSC                 XSH                 YSH
   1.0                 1.0                 0.0                 0.0
#  X                   Y
   0.0                 0.0
   0.05                150.0
   0.10                250.0
   0.25                380.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 601 in model.funct_smooths
    f = model.funct_smooths[601]
    assert f.id == 601
    assert len(f.x) == 4
    assert len(f.y) == 4
    assert pytest.approx(f.x[1]) == 0.05
    assert pytest.approx(f.y[1]) == 150.0
    assert 601 in model.functions


def test_fric_orient(tmp_path: Path):
    """Verify /FRIC_ORIENT parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Friction Orientation
                  10
/FRIC_ORIENT/701
Oriented Friction Direction
#  GRPART_ID           SKEW_ID             PHI                 VX                  VY                  VZ                  IFRIC
   5                   2                   45.0                1.0                 0.0                 0.0                 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 701 in model.fric_orients
    fo = model.fric_orients[701]
    assert fo.id == 701
    assert fo.grpart_id == 5
    assert fo.skew_id == 2
    assert pytest.approx(fo.phi) == 45.0
    assert pytest.approx(fo.vx) == 1.0
    assert fo.ifric == 1


def test_damp_freq_range_and_damp_funct(tmp_path: Path):
    """Verify /DAMP/FREQ_RANGE and /DAMP/FUNCT parsing."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Damping Keywords
                  10
/DAMP/FREQ_RANGE/801
Frequency Range Damping
#  FMIN                FMAX                DAMP                ITYPE
   10.0                1000.0              0.05                1
/DAMP/FUNCT/802
Function Damping
#  FCT_ID              DAMP_SCALE          ITYPE
   601                 1.5                 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert 801 in model.damp_freq_ranges
    dfr = model.damp_freq_ranges[801]
    assert dfr.id == 801
    assert pytest.approx(dfr.fmin) == 10.0
    assert pytest.approx(dfr.fmax) == 1000.0
    assert pytest.approx(dfr.damp) == 0.05
    assert dfr.itype == 1

    assert 802 in model.damp_functs
    df = model.damp_functs[802]
    assert df.id == 802
    assert df.fct_id == 601
    assert pytest.approx(df.damp_scale) == 1.5
    assert df.itype == 2


def test_initial_state_tables(tmp_path: Path):
    """Verify initial state tables (/INIBRI/STRS_FGLO, /INISPHCEL, /INISH3/FAIL, /INISHE/FAIL, /INISHE/STRA_F_GLOB)."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Initial State Tables
                  10
/INIBRI/STRS_FGLO/1
Initial Brick Global Stresses
#  ELEM_ID             SIG_XX              SIG_YY              SIG_ZZ              SIG_XY              SIG_YZ              SIG_ZX
   1001                10.0                20.0                30.0                5.0                 2.0                 1.0
/INISPHCEL/2
Initial SPH Cell State
#  P                   RHO                 E                   VX                  VY                  VZ
   1.5                 2.7e-6              100.0               10.0                0.0                 0.0
/INISH3/FAIL/3
Initial Shell 3N Failure
#  ELEM_ID             IFAIL               VAL
   3001                1                   0.5
/INISHE/FAIL/4
Initial Shell Failure
#  ELEM_ID             IFAIL               VAL
   4001                2                   0.8
/INISHE/STRA_F_GLOB/5
Initial Shell Global Strain
#  ELEM_ID             EPS_XX              EPS_YY              EPS_ZZ              EPS_XY              EPS_YZ              EPS_ZX
   5001                0.01                0.02                -0.01               0.005               0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)

    assert "INIBRI_STRS_FGLO_1" in model.ini_state_tables
    assert len(model.ini_state_tables["INIBRI_STRS_FGLO_1"].rows) == 1
    assert model.ini_state_tables["INIBRI_STRS_FGLO_1"].rows[0]["ELEM_ID"] == 1001

    assert "INISPHCEL_2" in model.ini_state_tables
    assert model.ini_state_tables["INISPHCEL_2"].rows[0]["P"] == 1.5

    assert "INISH3_FAIL_3" in model.ini_state_tables
    assert model.ini_state_tables["INISH3_FAIL_3"].rows[0]["ELEM_ID"] == 3001

    assert "INISHE_FAIL_4" in model.ini_state_tables
    assert model.ini_state_tables["INISHE_FAIL_4"].rows[0]["ELEM_ID"] == 4001

    assert "INISHE_STRA_F_GLOB_5" in model.ini_state_tables
    assert model.ini_state_tables["INISHE_STRA_F_GLOB_5"].rows[0]["ELEM_ID"] == 5001


def test_engine_checksum_anim_and_eng_outputs(tmp_path: Path):
    """Verify checksum directives, /ANIM/SPRING/FORC, /ANIM/BRICK/TENS, /ENG/STATE/DT, /ENG/DYNAIN/DT."""
    engine_deck = """# OpenRadioss Engine Deck
/RUN/M195_TEST/1
 10.0
/CHECKSUM/START
/ANIM/DT
0.0 1.0
/ANIM/SPRING/FORC
/ANIM/BRICK/TENS
/ENG/STATE/DT
0.0 2.0
/ENG/DYNAIN/DT
0.0 5.0
/CHECKSUM/END
/END
"""
    ec, log = _parse_engine(tmp_path, engine_deck)

    assert ec.checksum_mode == "END"
    assert "SPRING/FORC" in ec.anim_elem
    assert "BRICK/TENS" in ec.anim_tens
    assert pytest.approx(ec.state_dt) == 2.0
    assert pytest.approx(ec.dynain_dt) == 5.0
