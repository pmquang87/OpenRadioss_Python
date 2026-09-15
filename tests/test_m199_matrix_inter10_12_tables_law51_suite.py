"""Tests for Milestone M199:
Matrix Transformations, Standalone Transforms, Interfaces Type 10 & 12,
Initial State Tables & Drucker-Prager Brittle Material (LAW51) Suite.
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import TransformMatrix, InterType10, InterType12, MatLaw51


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_transform_matrix_fixed(tmp_path: Path):
    """Verify /TRANSFORM/MATRIX in fixed format."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/TRANSFORM/MATRIX/100
Transform 3D Matrix
# grnod_id     m11       m12       m13        tx    sub_id
        10     1.0       0.0       0.0      10.5         5
#              m21       m22       m23        ty
               0.0       1.0       0.0      20.5
#              m31       m33       m33        tz
               0.0       0.0       1.0      30.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 100 in model.transform_matrices
    tm = model.transform_matrices[100]
    assert isinstance(tm, TransformMatrix)
    assert tm.id == 100
    assert tm.title == "Transform 3D Matrix"
    assert tm.grnod_id == 10
    assert tm.sub_id == 5
    assert tm.m11 == 1.0
    assert tm.tx == 10.5
    assert tm.ty == 20.5
    assert tm.tz == 30.5


def test_standalone_transforms_aliases(tmp_path: Path):
    """Verify standalone transform keywords /ROT, /TRA, /SCA, /SYM, /MATRIX."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/ROT/1
Rotation 1
         1         2       45.0
/TRA/2
Translation 2
         1       5.0       0.0       0.0
/SCA/3
Scale 3
         1       2.0       2.0       2.0
/SYM/4
Symmetry 4
         1         2         3         4
/MATRIX/5
Matrix 5
         1       1.0       0.0       0.0       1.0         0
                 0.0       1.0       0.0       2.0
                 0.0       0.0       1.0       3.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert len(model.transforms) >= 4
    assert 5 in model.transform_matrices
    assert model.transform_matrices[5].tx == 1.0
    assert model.transform_matrices[5].ty == 2.0
    assert model.transform_matrices[5].tz == 3.0


def test_inter_type10(tmp_path: Path):
    """Verify /INTER/TYPE10 interface."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/INTER/TYPE10/50
Node to Surface Tied Contact
# grnod_id   surf_id               multimp                idel
        11        22                     1                   2
#    stfac                 gap    tstart     tstop
       1.5                 0.1       0.0      10.0
#                itied   inactiv  stiff_dc           sort_fact
                     1         2     100.0                 0.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 50 in model.inter_type10s
    inter = model.inter_type10s[50]
    assert isinstance(inter, InterType10)
    assert inter.id == 50
    assert inter.grnod_id == 11
    assert inter.surf_id == 22
    assert inter.multimp == 1
    assert inter.idel == 2
    assert inter.stfac == 1.5
    assert inter.gap == 0.1
    assert inter.itied == 1
    assert inter.inactiv == 2
    assert inter.stiff_dc == 100.0
    assert inter.sort_fact == 0.5


def test_inter_type12(tmp_path: Path):
    """Verify /INTER/TYPE12 interface."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/INTER/TYPE12/60
Segment Search Contact
#  surf_ids  surf_idm  interpol       tol    tstart     tstop
         15        25         1      0.05       0.0      50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 60 in model.inter_type12s
    inter = model.inter_type12s[60]
    assert isinstance(inter, InterType12)
    assert inter.id == 60
    assert inter.surf_ids == 15
    assert inter.surf_idm == 25
    assert inter.interpol == 1
    assert inter.tol == 0.05
    assert inter.tstop == 50.0


def test_initial_state_tables(tmp_path: Path):
    """Verify /TABLE/INISH3/ORTH_LOC, /TABLE/INISHE/STRS_F, /TABLE/INISHE/THICK."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/TABLE/INISH3/ORTH_LOC/1/1
Table 3-node shell local ortho
        10         1         1         1         0
      30.0      45.0
/TABLE/INISHE/STRS_F/2/1
Table shell stress full
        20         1       1.5
       0.0       0.0       0.0       0.0       0.0
     100.0      50.0      25.0       0.0       0.0
      0.05      10.0       5.0       2.5
/TABLE/INISHE/THICK/3/1
Table shell thickness
        30       2.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 10 in model.ini_shells
    assert model.ini_shells[10].orth_phi == [30.0]
    assert model.ini_shells[10].orth_alpha == [45.0]
    assert 20 in model.ini_shells
    assert model.ini_shells[20].thick == 1.5
    assert model.ini_shells[20].sigma[0] == 100.0
    assert 30 in model.ini_shells
    assert model.ini_shells[30].thick == 2.5


def test_mat_law51_drucker_prager(tmp_path: Path):
    """Verify /MAT/LAW51 (Drucker-Prager / Brittle) material model."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/MAT/LAW51/DRUCKER_PRAGER/201
Drucker Prager Concrete
#              rho                   E                  nu
            2.5e-9             30000.0                 0.2
#         cohesion          frict_angl            dilatang           tens_cut0
              25.0                35.0                15.0                 3.5
#            hardn
               0.1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 201 in model.materials
    mat = model.materials[201]
    assert mat.law == 51 or mat.law == "LAW51"
    assert mat.rho0 == 2.5e-9
    assert mat.params["E"] == 30000.0
    assert mat.params["nu"] == 0.2

    # Also check model.mat_law51s registry
    assert 201 in model.mat_law51s
    m51 = model.mat_law51s[201]
    assert isinstance(m51, MatLaw51)
    assert m51.rho0 == 2.5e-9
    assert m51.e == 30000.0
    assert m51.nu == 0.2


def test_mat_law51_alias_keywords(tmp_path: Path):
    """Verify /MAT/BRITTLE and /MAT/DRUCKER_PRAGER keywords."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/MAT/BRITTLE/202
Brittle Rock
            2.7e-9             45000.0                0.25
              30.0                40.0                20.0                 4.0
               0.2
/MAT/DRUCKER_PRAGER/203
Drucker Prager Rock
            2.8e-9             50000.0                0.22
              35.0                42.0                22.0                 4.5
               0.3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 202 in model.mat_law51s
    assert 203 in model.mat_law51s
    assert model.mat_law51s[202].title == "Brittle Rock"
    assert model.mat_law51s[203].title == "Drucker Prager Rock"


def test_inter_type12_extended(tmp_path: Path):
    """Verify /INTER/TYPE12 interface with all vector and kinematic options."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/INTER/TYPE12/65
Detailed Type 12 Contact
#  surf_ids  surf_idm  interpol       tol    tstart     tstop     itied     bcopt   skew_id    node_c
         12        24         2      0.01       1.0      20.0         1         0         3        40
#        xc        yc        zc     theta
        1.0       2.0       3.0      45.0
#        xn        yn        zn
        0.0       0.0       1.0
#        xt        yt        zt
        1.0       0.0       0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 65 in model.inter_type12s
    inter = model.inter_type12s[65]
    assert inter.surf_ids == 12
    assert inter.surf_idm == 24
    assert inter.interpol == 2
    assert inter.tol == 0.01
    assert inter.tstart == 1.0
    assert inter.tstop == 20.0
    assert inter.itied == 1
    assert inter.skew_id == 3
    assert inter.node_c == 40
    assert inter.xc == 1.0
    assert inter.yc == 2.0
    assert inter.zc == 3.0
    assert inter.theta == 45.0
    assert inter.xn == 0.0
    assert inter.yn == 0.0
    assert inter.zn == 1.0
    assert inter.xt == 1.0
    assert inter.yt == 0.0
    assert inter.zt == 0.0


def test_initial_state_inish3_tables(tmp_path: Path):
    """Verify /TABLE/INISH3/STRS_F and /TABLE/INISH3/THICK."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
Test Deck
                 1                 1
/TABLE/INISH3/STRS_F/10/1
Table 3-node shell stress full
        101        1       2.0
        0.0      0.0       0.0       0.0       0.0
      250.0    125.0      50.0       0.0       0.0
       0.02     20.0      10.0       5.0
/TABLE/INISH3/THICK/11/1
Table 3-node shell thickness
        102      3.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 101 in model.ini_shells
    assert model.ini_shells[101].thick == 2.0
    assert model.ini_shells[101].sigma[0] == 250.0
    assert 102 in model.ini_shells
    assert model.ini_shells[102].thick == 3.2

