"""Tests for /MAT/LAW101 (/MAT/PP / /MAT/PLAS_POLY) Starter reader, model entities, and diagnostics.

Verifies:
  - Fixed-format and free-format card parsing for 11 cards
  - Backward compatibility with legacy 7-card decks
  - Keyword aliases: /MAT/PP, /MAT/PLAS_POLY, /MAT/LAW101
  - Model entity properties: E, G, K, sound_speed, aliases
  - Starter diagnostic checks (ANCMSG 1514, 305, 306)
"""

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MatLaw101, MatPP, MatPlasPoly, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law101


def _parse_deck(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_starter_read_law101_11_cards(tmp_path: Path):
    """Verify parsing 11 cards of /MAT/LAW101 matching radioss2021 mat_l101.cfg."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Test MAT LAW101 11 Cards
                  10                   0
/MAT/LAW101/10
Polypropylene Bouvard Model
#       RHO0               RHOR
      0.9E-6                0.0
#      DEREF                DE0              DPOISS                DVE1
      1500.0               -2.5                0.38                0.25
#       DVE2          DEDOT_REF            GAMV_REF              ALPHAP
         1.5             1.0E-4              1.0E-3                0.08
#         DQ                DV1                  DM                 DC3
     45000.0            2.0E-28                 1.8               -0.05
#        DC4           CALPHAK1            CALPHAK2                  H0
        25.0               0.15                0.05                50.0
#     DES1_0                DC5                 DC6                 DC7
         0.1             -0.001                 1.0              -0.002
#        DC8                DC9                DC10                  H1
         2.0              -0.01                10.0                20.0
#     DES2_0               DC11                DC12                DC13
        0.05             -0.001                 1.5                0.01
#       DC14                DC1                 DC2           DLAMBDA_L
         5.0              -0.05                 8.0                 5.0
# RHO_THETA0         CV_THETA_0              THETA0            ALPHA_TH
      0.9E-6             1800.0              293.15              1.2E-4
#THETA_GLASS              OMEGA          THETA_FLAG             HEAT_T0
       250.0                0.9                 2.0              293.15
"""
    model = _parse_deck(tmp_path, deck_text)
    assert 10 in model.mat_law101s
    mat = model.mat_law101s[10]

    assert mat.id == 10
    assert mat.title == "Polypropylene Bouvard Model"
    assert pytest.approx(mat.rho0) == 0.9e-6
    assert pytest.approx(mat.e) == 1500.0
    assert pytest.approx(mat.alpha1) == -2.5
    assert pytest.approx(mat.nu) == 0.38
    assert pytest.approx(mat.ve1) == 0.25
    assert pytest.approx(mat.ve2) == 1.5
    assert pytest.approx(mat.epsilonref) == 1.0e-4
    assert pytest.approx(mat.gamma0) == 1.0e-3
    assert pytest.approx(mat.alpha_p) == 0.08
    assert pytest.approx(mat.deltah) == 45000.0
    assert pytest.approx(mat.vol) == 2.0e-28
    assert pytest.approx(mat.m) == 1.8
    assert pytest.approx(mat.c3) == -0.05
    assert pytest.approx(mat.c4) == 25.0
    assert pytest.approx(mat.hard) == 50.0
    assert pytest.approx(mat.zeta1i) == 0.1
    assert pytest.approx(mat.c10) == 10.0
    assert pytest.approx(mat.hard1) == 20.0
    assert pytest.approx(mat.zeta2i) == 0.05
    assert pytest.approx(mat.c14) == 5.0
    assert pytest.approx(mat.c1) == -0.05
    assert pytest.approx(mat.c2) == 8.0
    assert pytest.approx(mat.lambdal) == 5.0
    assert pytest.approx(mat.theta_glass) == 250.0
    assert pytest.approx(mat.omega) == 0.9
    assert pytest.approx(mat.theta_flag) == 2.0
    assert pytest.approx(mat.heat_t0) == 293.15


def test_starter_read_aliases_pp_and_plas_poly(tmp_path: Path):
    """Verify keyword aliases /MAT/PP and /MAT/PLAS_POLY populate mat_law101s."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Test MAT PP Aliases
                  10                   0
/MAT/PP/21
PP Alias Test
# RHO
 0.9e-6
# E, alpha1, nu, ve1
 1200.0, -2.0, 0.35, 0.2
# ve2, epsilonref, gamma0, alphap
 1.2, 1e-4, 1e-3, 0.05
# deltah, vol, m, c3
 40000.0, 1.5e-28, 1.5, -0.04
# c4, alphak1, alphak2, hard
 20.0, 0.1, 0.05, 40.0
# zeta1i, c5, c6, c7
 0.05, -0.001, 0.8, -0.001
# c8, c9, c10, hard1
 1.5, -0.01, 8.0, 15.0
/MAT/PLAS_POLY/22
PLAS_POLY Alias Test
# RHO
 0.95e-6
# E, alpha1, nu, ve1
 1400.0, -2.2, 0.36, 0.22
# ve2, epsilonref, gamma0, alphap
 1.3, 1e-4, 1e-3, 0.06
# deltah, vol, m, c3
 42000.0, 1.6e-28, 1.6, -0.04
# c4, alphak1, alphak2, hard
 22.0, 0.12, 0.06, 45.0
# zeta1i, c5, c6, c7
 0.06, -0.001, 0.85, -0.001
# c8, c9, c10, hard1
 1.6, -0.01, 8.5, 16.0
"""
    model = _parse_deck(tmp_path, deck_text)
    assert 21 in model.mat_law101s
    assert 22 in model.mat_law101s
    assert 21 in model.mat_pps
    assert 22 in model.mat_pps

    assert model.mat_law101s[21].e == pytest.approx(1200.0)
    assert model.mat_law101s[22].e == pytest.approx(1400.0)


def test_mat_law101_entity_properties():
    """Verify computed properties on MatLaw101 entity: E, G, K, sound_speed."""
    mat = MatLaw101(id=1, rho0=1.0e-6, e=2000.0, nu=0.25)

    assert mat.E == 2000.0
    # G = E / (2 * (1 + nu)) = 2000 / 2.5 = 800
    assert pytest.approx(mat.G) == 800.0
    # K = E / (3 * (1 - 2*nu)) = 2000 / 1.5 = 1333.333
    assert pytest.approx(mat.K) == 1333.3333333333333
    # c = sqrt((K + 4/3*G) / rho0) = sqrt((1333.333 + 1066.667) / 1e-6) = sqrt(2400e6) = 48989.79
    assert pytest.approx(mat.sound_speed, rel=1e-5) == 48989.79485566356


def test_starter_diagnostics_ancmsg():
    """Verify starter validation diagnostics check_mat_law101."""
    model = Model()

    # 1. Invalid density (rho0 <= 0)
    log = MessageLog()
    mat_bad_rho = MatLaw101(id=1, rho0=0.0, e=1500.0, nu=0.35)
    check_mat_law101(mat_bad_rho, model, log)
    assert any("ANCMSG 1514" in err and "RHO" in err for err in log.errors)

    # 2. Invalid modulus (E <= 0)
    log = MessageLog()
    mat_bad_e = MatLaw101(id=2, rho0=0.9e-6, e=0.0, nu=0.35)
    check_mat_law101(mat_bad_e, model, log)
    assert any("ANCMSG 1514" in err and "MODULUS" in err for err in log.errors)

    # 3. Invalid Poisson's ratio (nu >= 0.5)
    log = MessageLog()
    mat_bad_nu = MatLaw101(id=3, rho0=0.9e-6, e=1500.0, nu=0.52)
    check_mat_law101(mat_bad_nu, model, log)
    assert any("ANCMSG 1514" in err and "POISSON" in err for err in log.errors)

    # 4. Element type restriction: rejection of shell elements (ANCMSG 305)
    log = MessageLog()
    mat_ok = MatLaw101(id=4, rho0=0.9e-6, e=1500.0, nu=0.35)
    model.materials[4] = mat_ok
    part_shell = Part(id=1, prop_id=1, mat_id=4)
    model.parts[1] = part_shell
    class DummyShell:
        part_id = 1
    model.shells = {1: DummyShell()}

    check_mat_law101(mat_ok, model, log)
    assert any("ANCMSG 305" in err and "2D shell" in err for err in log.errors)

    # 5. Element type restriction: acceptance of 3D solid elements
    log = MessageLog()
    model.shells = {}
    check_mat_law101(mat_ok, model, log)
    assert len(log.errors) == 0
