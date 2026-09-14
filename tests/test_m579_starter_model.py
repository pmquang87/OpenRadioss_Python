"""Tests for /MAT/LAW110 (/MAT/VEGTER, /MAT/PLAS_VEGTER) Starter model entities, parsing, and checks (M579).

Verifies:
  - MaterialLaw110 dataclass fields, types, and defaults.
  - Model aliases and exports in pyradioss.model.
  - Fixed-format reading with cards 1-6 and conditional angle cards.
  - Free-format reading with comma-separated and slash syntax.
  - Starter checks:
      * Density RHO <= 0 (ANCMSG 1514)
      * Young's modulus E <= 0 (ANCMSG 1514)
      * Poisson's ratio Nu <= 0 or Nu >= 0.5 (ANCMSG 49 / 3068)
      * Formulation Icrit not in (1, 2, 3, 4) (ANCMSG 1776)
      * SIGMA_r <= 0 when tab_yld == 0 (ANCMSG 1777)
      * Tini == 0 warning (ANCMSG 1778)
      * Nangle > 10 (ANCMSG 1779)
      * VP > 3 (ANCMSG 1802)
      * Element compatibility: reject solids (ANCMSG 305) and 1D (ANCMSG 306), allow shells.
"""

from pathlib import Path
import pytest
import numpy as np

from pyradioss.model.entities import (
    MaterialLaw110,
    MatVegter,
    MaterialVegter,
    MatPlasVegter,
    MaterialPlasVegter,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_starter_deck
from pyradioss.starter.checks import check_model, check_mat_law110


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "deck_0000.rad"
    deck_path.write_text(text, encoding="utf-8")
    raw_deck = read_deck(deck_path)
    model = Model()
    log = MessageLog()
    read_starter_deck(raw_deck, model, log)
    return model, log


class TestMaterialLaw110Model:
    """Test dataclass attributes and aliases."""

    def test_default_values(self):
        mat = MaterialLaw110()
        assert mat.law == 110
        assert mat.law_name == "VEGTER"
        assert mat.icrit == 1
        assert mat.ires == 2
        assert mat.vp == 2
        assert mat.tini == 293.0
        assert mat.fbi == 1.0
        assert mat.rhobi == 1.0
        assert mat.angles_data == []

    def test_aliases(self):
        assert MatVegter is MaterialLaw110
        assert MaterialVegter is MaterialLaw110
        assert MatPlasVegter is MaterialLaw110
        assert MaterialPlasVegter is MaterialLaw110


class TestMaterialLaw110Parser:
    """Test fixed and free format parsing."""

    def test_fixed_format_icrit1(self, tmp_path: Path):
        text = """\
# Starter deck
/BEGIN
Test Vegter Icrit 1
                  Mg                  mm                   s
/MAT/LAW110/1101
Vegter Sheet Metal Grade
#              RHO_I           Refer_Rho
             7.8e-06             7.8e-06
#                  E                  Nu                Ires
            210000.0                 0.3                   2
#              Icrit             TAB_YLD            Xscale_h            Yscale_h                 FBI               RHOBI
                   1                   0                 1.0                 1.0                 1.0                 1.0
#            SIGMA_r               DSIGM                BETA               OMEGA                   n
               250.0               150.0                10.0                 0.5                0.22
#               EPS0                SIGS                 DG0               DEPS0                   m
               0.001               500.0               1.0e5                 1.0                 0.1
#               Tini               Chard                Fcut                  VP             ismooth            TAB_TEMP
               293.0                 0.0              1.0e20                   2                   1                   0
#                fun                   r                fps1                fps2                 fsh
                 1.0                 1.8                 1.0                 0.0                 0.5
                 1.0                 1.5                 1.0                 0.0                 0.5
                 1.0                 2.1                 1.0                 0.0                 0.5
/END
"""
        model, log = _parse_deck_string(tmp_path, text)
        assert not log.errors, f"Errors: {log.errors}"
        assert 1101 in model.mat_law110s
        m = model.mat_law110s[1101]
        assert m.rho0 == pytest.approx(7.8e-06)
        assert m.refer_rho == pytest.approx(7.8e-06)
        assert m.young == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.3)
        assert m.icrit == 1
        assert m.ires == 2
        assert m.sigma_r == pytest.approx(250.0)
        assert m.dsigm == pytest.approx(150.0)
        assert m.beta == pytest.approx(10.0)
        assert m.omega == pytest.approx(0.5)
        assert m.hard_n == pytest.approx(0.22)
        assert m.tini == pytest.approx(293.0)
        assert m.nangle == 3
        assert len(m.angles_data) == 3
        assert m.angles_data[0][1] == pytest.approx(1.8)
        assert m.angles_data[1][1] == pytest.approx(1.5)
        assert m.angles_data[2][1] == pytest.approx(2.1)

    def test_free_format_icrit4_vegter_lite(self, tmp_path: Path):
        text = """\
/BEGIN
Test Vegter Lite Free Format
                  Mg                  mm                   s
/MAT/VEGTER/2001
Vegter Lite
7.85e-06, 7.85e-06
205000.0, 0.33, 1
4, 0, 1.0, 1.0, 1.05, 0.98
300.0, 200.0, 15.0, 0.0, 0.18
0.0, 0.0, 0.0, 0.0, 0.0
293.15, 0.0, 1.0e15, 1, 1, 0
1.0, 1.2, 0.55, 0.52
1.0, 1.6, 0.58, 0.54
/END
"""
        model, log = _parse_deck_string(tmp_path, text)
        assert not log.errors, f"Errors: {log.errors}"
        assert 2001 in model.mat_law110s
        m = model.mat_law110s[2001]
        assert m.young == pytest.approx(205000.0)
        assert m.nu == pytest.approx(0.33)
        assert m.icrit == 4
        assert m.ires == 1
        assert m.nangle == 2
        assert m.fbi == pytest.approx(1.05)
        assert m.rhobi == pytest.approx(0.98)
        assert len(m.angles_data) == 2


class TestMaterialLaw110Checks:
    """Test starter diagnostics for /MAT/LAW110."""

    def test_density_check_ancmsg_1514(self):
        mat = MaterialLaw110(mid=1, rho0=0.0, young=210000.0, nu=0.3, sigma_r=200.0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1514" in e for e in log.errors)

    def test_young_check_ancmsg_1514(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=-100.0, nu=0.3, sigma_r=200.0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1514" in e for e in log.errors)

    def test_nu_check_ancmsg_49_3068(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.55, sigma_r=200.0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("49" in e or "3068" in e for e in log.errors)

    def test_icrit_check_ancmsg_1776(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=5, sigma_r=200.0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1776" in e for e in log.errors)

    def test_sigma_r_check_ancmsg_1777(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, sigma_r=0.0, tab_yld=0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1777" in e for e in log.errors)

    def test_tini_check_ancmsg_1778(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, sigma_r=200.0, tini=0.0)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1778" in w for w in log.warnings)

    def test_nangle_check_ancmsg_1779(self):
        angles_11 = [[1.0, 1.0, 1.0, 0.0, 0.5] for _ in range(11)]
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, sigma_r=200.0, angles_data=angles_11)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1779" in e for e in log.errors)

    def test_vp_check_ancmsg_1802(self):
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, sigma_r=200.0, vp=4)
        log = MessageLog()
        check_mat_law110(mat, log)
        assert any("1802" in e for e in log.errors)

    def test_element_compatibility(self):
        """Reject 3D solids (ANCMSG 305) and 1D (ANCMSG 306); allow shells."""
        class DummyElem:
            def __init__(self, pid: int):
                self.pid = pid

        model = Model()
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, sigma_r=200.0)
        model.materials[1] = mat
        model.mat_law110s[1] = mat

        # Add solid part using LAW110
        model.parts[1] = Part(id=1, prop_id=1, mat_id=1)
        model.properties[1] = Property(id=1, type=14)
        model.bricks = {1: DummyElem(pid=1)}

        log = MessageLog()
        check_mat_law110(model=model, mat=mat, log=log)
        assert any("305" in e for e in log.errors), f"Expected ANCMSG 305 for solid, got {log.errors}"

        # Now test shell part: should be valid!
        model_shell = Model()
        model_shell.materials[1] = mat
        model_shell.mat_law110s[1] = mat
        model_shell.parts[1] = Part(id=1, prop_id=1, mat_id=1)
        model_shell.properties[1] = Property(id=1, type=1)
        model_shell.shells = {1: DummyElem(pid=1)}

        log_shell = MessageLog()
        check_mat_law110(model=model_shell, mat=mat, log=log_shell)
        assert not any("305" in e or "306" in e for e in log_shell.errors)
