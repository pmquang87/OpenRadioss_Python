"""Tests for /MAT/LAW102 (/MAT/DPRAG2) Starter reader, model entities, and diagnostics.

Verifies:
  - Fixed-format and free-format card parsing for 5 cards
  - Backward compatibility with legacy 2-card test layout (M194)
  - Keyword aliases: /MAT/LAW102 and /MAT/DPRAG2
  - Model entity properties: E, G, K, sound_speed, aliases
  - Starter diagnostic checks (ANCMSG 1514, 305, 306)
"""

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MatLaw102, MatDPrag2, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law102, check_mat_dprag2, check_materials


def _parse_deck(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_starter_read_law102_5_cards(tmp_path: Path):
    """Verify parsing 5 cards of /MAT/LAW102 matching radioss2020 mat102_DPRAG2.cfg."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Test MAT LAW102 5 Cards
                  10                   0
/MAT/LAW102/10
Geological Rock DPRAG2 Model
#              RHO_I
              2.5E-6
#              IFORM
                   2
#                  E                  NU
             30000.0                0.25
#                  C                 PHI
                15.0                32.0
#              A_MAX               P_MIN
               250.0                -5.0
"""
    model = _parse_deck(tmp_path, deck_text)
    assert 10 in model.mat_law102s
    mat = model.mat_law102s[10]

    assert mat.id == 10
    assert mat.title == "Geological Rock DPRAG2 Model"
    assert pytest.approx(mat.rho) == 2.5e-6
    assert mat.iform == 2
    assert pytest.approx(mat.e) == 30000.0
    assert pytest.approx(mat.nu) == 0.25
    assert pytest.approx(mat.c) == 15.0
    assert pytest.approx(mat.phi) == 32.0
    assert pytest.approx(mat.amax) == 250.0
    assert pytest.approx(mat.pmin) == -5.0

    # Test entity computed properties
    assert pytest.approx(mat.E) == 30000.0
    g_expected = 30000.0 / (2.0 * (1.0 + 0.25))
    k_expected = 30000.0 / (3.0 * (1.0 - 2.0 * 0.25))
    assert pytest.approx(mat.G) == g_expected
    assert pytest.approx(mat.K) == k_expected
    assert mat.sound_speed > 0.0


def test_starter_read_alias_dprag2(tmp_path: Path):
    """Verify keyword alias /MAT/DPRAG2 populates mat_law102s and mat_dprag2s."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Test MAT DPRAG2
                  10                   0
/MAT/DPRAG2/15
Concrete Foundation
#              RHO_I
              2.4E-6
#              IFORM
                   1
#                  E                  NU
             25000.0                0.20
#                  C                 PHI
                10.0                35.0
#              A_MAX               P_MIN
               180.0                -3.0
"""
    model = _parse_deck(tmp_path, deck_text)
    assert 15 in model.mat_law102s
    assert 15 in model.mat_dprag2s
    mat = model.mat_dprag2s[15]
    assert mat.id == 15
    assert mat.iform == 1
    assert pytest.approx(mat.c) == 10.0
    assert pytest.approx(mat.phi) == 35.0


def test_starter_read_legacy_m194_2_cards(tmp_path: Path):
    """Verify backward compatibility with legacy 2-card test layout from M194."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Legacy M194 LAW102
                  10                   0
/MAT/LAW102/1
Legacy LAW102 Test Deck
 2.7e-6 70000. 0.33
 0.5 0.5 0.5 1.5 1.5 1.5
"""
    model = _parse_deck(tmp_path, deck_text)
    assert 1 in model.mat_law102s
    mat = model.mat_law102s[1]
    assert pytest.approx(mat.rho) == 2.7e-6
    assert pytest.approx(mat.e) == 70000.0
    assert pytest.approx(mat.nu) == 0.33


def test_diagnostics_validation_success():
    """Verify check_mat_law102 passes on valid parameters."""
    mat = MatLaw102(id=1, rho=2.5e-6, e=20000.0, nu=0.25, c=10.0, phi=30.0)
    model = Model()
    model.mat_law102s[1] = mat
    model.materials[1] = mat
    part = Part(id=1, mat_id=1, prop_id=1)
    model.parts[1] = part
    log = MessageLog()
    check_mat_law102(mat, model, log)
    assert len(log.errors) == 0


def test_diagnostics_validation_errors():
    """Verify diagnostic errors for invalid parameters and element restrictions."""
    # Negative density -> ANCMSG 1514
    mat1 = MatLaw102(id=1, rho=-1.0, e=20000.0, nu=0.25)
    model1 = Model()
    model1.mat_law102s[1] = mat1
    log1 = MessageLog()
    check_mat_law102(mat1, model1, log1)
    assert len(log1.errors) > 0
    assert any("ANCMSG 1514" in str(err) and "RHO" in str(err) for err in log1.errors)

    # Zero Young's modulus -> ANCMSG 1514
    mat2 = MatLaw102(id=2, rho=2.5e-6, e=0.0, nu=0.25)
    model2 = Model()
    model2.mat_law102s[2] = mat2
    log2 = MessageLog()
    check_mat_law102(mat2, model2, log2)
    assert len(log2.errors) > 0
    assert any("ANCMSG 1514" in str(err) and "MODULUS" in str(err) for err in log2.errors)

    # Invalid Poisson's ratio (nu >= 0.5) -> ANCMSG 1514
    mat3 = MatLaw102(id=3, rho=2.5e-6, e=20000.0, nu=0.52)
    model3 = Model()
    model3.mat_law102s[3] = mat3
    log3 = MessageLog()
    check_mat_law102(mat3, model3, log3)
    assert len(log3.errors) > 0
    assert any("ANCMSG 1514" in str(err) and "POISSON" in str(err) for err in log3.errors)

    # Rejection of Shell elements -> ANCMSG 305
    mat4 = MatLaw102(id=4, rho=2.5e-6, e=20000.0, nu=0.25)
    model4 = Model()
    model4.mat_law102s[4] = mat4
    model4.materials[4] = mat4
    part4 = Part(id=4, mat_id=4, prop_id=4)
    model4.parts[4] = part4
    class DummyShell:
        part_id = 4
    model4.shells = {1: DummyShell()}
    log4 = MessageLog()
    check_mat_law102(mat4, model4, log4)
    assert len(log4.errors) > 0
    assert any("LAW102" in str(err) and "305" in str(err) for err in log4.errors)

    # Rejection of 1D spring/beam elements -> ANCMSG 306
    mat5 = MatLaw102(id=5, rho=2.5e-6, e=20000.0, nu=0.25)
    model5 = Model()
    model5.mat_law102s[5] = mat5
    model5.materials[5] = mat5
    part5 = Part(id=5, mat_id=5, prop_id=5)
    model5.parts[5] = part5
    class DummyBeam:
        part_id = 5
    model5.beams = {1: DummyBeam()}
    log5 = MessageLog()
    check_mat_law102(mat5, model5, log5)
    assert len(log5.errors) > 0
    assert any("LAW102" in str(err) and "306" in str(err) for err in log5.errors)
