"""
Tests for M550 (/MAT/LAW69 /MAT/HYP_ELAS /MAT/HYPERELASTIC) input, starter, model, and checks integration.
"""

from io import StringIO
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.input.card_layouts import (
    MAT_LAW69_1,
    MAT_LAW69_2,
    MAT_LAW69_3,
    MAT_HYP_ELAS_1,
    MAT_HYP_ELAS_2,
    MAT_HYP_ELAS_3,
    MAT_HYPERELASTIC_1,
    MAT_HYPERELASTIC_2,
    MAT_HYPERELASTIC_3,
    MAT_LAW69_HYP_ELAS_1,
    MAT_LAW69_HYP_ELAS_2,
    MAT_LAW69_HYP_ELAS_3,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS
from pyradioss.model.entities import MatLaw69, MaterialLaw69, MatHypElas, MatHyperelastic
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law69, check_materials, _ALLOWED_LAWS


def test_m550_card_layouts():
    assert MAT_LAW69_1 == [20, 20]
    assert MAT_LAW69_2 == [10, 10, 20, 20, 10]
    assert MAT_LAW69_3 == [10]

    assert MAT_HYP_ELAS_1 == MAT_LAW69_1
    assert MAT_HYP_ELAS_2 == MAT_LAW69_2
    assert MAT_HYP_ELAS_3 == MAT_LAW69_3

    assert MAT_HYPERELASTIC_1 == MAT_LAW69_1
    assert MAT_HYPERELASTIC_2 == MAT_LAW69_2
    assert MAT_HYPERELASTIC_3 == MAT_LAW69_3

    assert MAT_LAW69_HYP_ELAS_1 == MAT_LAW69_1
    assert MAT_LAW69_HYP_ELAS_2 == MAT_LAW69_2
    assert MAT_LAW69_HYP_ELAS_3 == MAT_LAW69_3

    for k in (
        "MAT_LAW69_1", "MAT_LAW69_2", "MAT_LAW69_3",
        "MAT_HYP_ELAS_1", "MAT_HYP_ELAS_2", "MAT_HYP_ELAS_3",
        "MAT_HYPERELASTIC_1", "MAT_HYPERELASTIC_2", "MAT_HYPERELASTIC_3",
        "MAT_LAW69_HYP_ELAS_1", "MAT_LAW69_HYP_ELAS_2", "MAT_LAW69_HYP_ELAS_3",
    ):
        assert k in LAYOUTS


def test_m550_cfg_catalogue():
    assert LAW_MAP["LAW69"] == 69
    assert LAW_MAP["HYP_ELAS"] == 69
    assert LAW_MAP["HYPERELASTIC"] == 69
    assert LAW_MAP["LAW69_HYP_ELAS"] == 69

    assert LAW_SYNONYMS["LAW69"] == "MAT_LAW69"
    assert LAW_SYNONYMS["HYP_ELAS"] == "MAT_LAW69"
    assert LAW_SYNONYMS["HYPERELASTIC"] == "MAT_LAW69"
    assert LAW_SYNONYMS["LAW69_HYP_ELAS"] == "MAT_LAW69"


def test_m550_model_entities_matlaw69():
    # Canonical initialization
    m = MatLaw69(
        id=69,
        rho0=1.1e-9,
        ref_rho=1.1e-9,
        nu=0.49,
        iflag=1,
        fct_id_bulk=2,
        fscale=1.0,
        nip=2,
        fct_id_data=10,
        title="Hyperelastic Test",
    )
    assert m.id == 69
    assert m.rho0 == 1.1e-9
    assert m.rho == 1.1e-9
    assert m.ref_rho == 1.1e-9
    assert m.rhor == 1.1e-9
    assert m.nu == 0.49
    assert m.iflag == 1
    assert m.law_id == 1
    assert m.fct_id_bulk == 2
    assert m.fct_id == 2
    assert m.fscale == 1.0
    assert m.nip == 2
    assert m.n_pair == 2
    assert m.fct_id_data == 10
    assert m.fct_id1 == 10
    assert m.title == "Hyperelastic Test"

    # Property setters
    m.rho = 2.0e-9
    assert m.rho0 == 2.0e-9
    m.rhor = 1.9e-9
    assert m.ref_rho == 1.9e-9
    m.law_id = 2
    assert m.iflag == 2
    m.fct_id = 3
    assert m.fct_id_bulk == 3
    m.n_pair = 3
    assert m.nip == 3
    m.fct_id1 = 20
    assert m.fct_id_data == 20

    # Kwargs initialization
    m_kw = MatLaw69(
        id=70,
        rho=1.5e-9,
        rhor=1.5e-9,
        law_id=2,
        fct_id=4,
        nu=0.48,
        fscale=2.0,
        n_pair=2,
        fct_id1=12,
        title="Kwargs Mat",
    )
    assert m_kw.rho0 == 1.5e-9
    assert m_kw.ref_rho == 1.5e-9
    assert m_kw.iflag == 2
    assert m_kw.fct_id_bulk == 4
    assert m_kw.nu == 0.48
    assert m_kw.fscale == 2.0
    assert m_kw.nip == 2
    assert m_kw.fct_id_data == 12

    # Aliases
    assert MatHypElas is MaterialLaw69
    assert MatHyperelastic is MaterialLaw69


def test_m550_parse_fixed_format(tmp_path):
    rad = """/BEGIN
TEST_FIXED_69
/MAT/LAW69/1
Rubber LAW69 Fixed
             1.10000E-09         1.10000E-09
         1         0               0.490               1.000         2
        10
"""
    f = tmp_path / "deck_fixed_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law69s
    m = model.mat_law69s[1]
    assert m.id == 1
    assert m.rho0 == pytest.approx(1.1e-9)
    assert m.ref_rho == pytest.approx(1.1e-9)
    assert m.iflag == 1
    assert m.fct_id_bulk == 0
    assert m.nu == pytest.approx(0.49)
    assert m.fscale == pytest.approx(1.0)
    assert m.nip == 2
    assert m.fct_id_data == 10

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 69
    assert mat.rho0 == pytest.approx(1.1e-9)


def test_m550_parse_free_format(tmp_path):
    rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_69
/MAT/LAW69/2
Rubber Free Format
1.2e-9 1.2e-9
2 1 0.48 1.5 3
20
"""
    f = tmp_path / "deck_free_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 2 in model.mat_law69s
    m = model.mat_law69s[2]
    assert m.id == 2
    assert m.rho0 == pytest.approx(1.2e-9)
    assert m.ref_rho == pytest.approx(1.2e-9)
    assert m.iflag == 2
    assert m.fct_id_bulk == 1
    assert m.nu == pytest.approx(0.48)
    assert m.fscale == pytest.approx(1.5)
    assert m.nip == 3
    assert m.fct_id_data == 20


def test_m550_parse_aliases(tmp_path):
    rad = """# RADIOSS STARTER
/BEGIN
TEST_ALIASES
/MAT/HYP_ELAS/3
Hyp Elas Alias
1.0e-9
1 0 0.495 1.0 2
30
/MAT/HYPERELASTIC/4
Hyperelastic Alias
1.05e-9
2 0 0.49 1.0 2
40
"""
    f = tmp_path / "deck_aliases_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 3 in model.mat_law69s
    assert 4 in model.mat_law69s
    assert model.mat_law69s[3].fct_id_data == 30
    assert model.mat_law69s[4].fct_id_data == 40
    assert model.materials[3].law == 69
    assert model.materials[4].law == 69


def test_m550_starter_checks():
    class DummyMat:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 1.0e-9)
            self.ref_rho = kwargs.get("ref_rho", 1.0e-9)
            self.nu = kwargs.get("nu", 0.495)
            self.iflag = kwargs.get("iflag", 1)
            self.fct_id_bulk = kwargs.get("fct_id_bulk", 0)
            self.fscale = kwargs.get("fscale", 1.0)
            self.nip = kwargs.get("nip", 2)
            self.fct_id1 = kwargs.get("fct_id1", 0)
            self.fct_id_data = kwargs.get("fct_id_data", self.fct_id1)

    # 1. Valid without curve
    m_valid = DummyMat()
    log = MessageLog()
    check_mat_law69(m_valid, log)
    assert not log.has_errors

    # 2. rho0 <= 0
    m_bad_rho = DummyMat(rho0=0.0)
    log = MessageLog()
    check_mat_law69(m_bad_rho, log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    # 3. nu < 0 or nu >= 0.5
    m_bad_nu1 = DummyMat(nu=-0.05)
    log = MessageLog()
    check_mat_law69(m_bad_nu1, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    m_bad_nu2 = DummyMat(nu=0.5)
    log = MessageLog()
    check_mat_law69(m_bad_nu2, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    # 4. Curve FCT_ID1 missing in functions dictionary
    m_missing_curve = DummyMat(fct_id1=99)
    funcs = {1: FunctTable(1, [0.0, 0.1, 0.2], [0.0, 10.0, 25.0])}
    log = MessageLog()
    check_mat_law69(m_missing_curve, log, functions=funcs)
    assert log.has_errors
    assert any("99" in str(e) for e in log.errors)

    # 5. Curve FCT_ID1 monotonic check: valid curve
    m_good_curve = DummyMat(fct_id1=1)
    log = MessageLog()
    check_mat_law69(m_good_curve, log, functions=funcs)
    assert not log.has_errors

    # 6. Curve FCT_ID1 non-monotonic (stress drop)
    bad_fn_stress = {"x": [0.0, 0.1, 0.2, 0.3], "y": [0.0, 10.0, 8.0, 15.0]}
    funcs_bad = {2: bad_fn_stress}
    m_drop_curve = DummyMat(fct_id1=2)
    log = MessageLog()
    check_mat_law69(m_drop_curve, log, functions=funcs_bad)
    assert log.has_errors
    assert any("monotonic" in str(e).lower() for e in log.errors)

    # 7. Curve FCT_ID1 non-monotonic (abscissa decreasing)
    bad_fn_strain = {"x": [0.0, 0.2, 0.1, 0.3], "y": [0.0, 5.0, 10.0, 15.0]}
    funcs_bad_x = {3: bad_fn_strain}
    m_bad_x_curve = DummyMat(fct_id1=3)
    log = MessageLog()
    check_mat_law69(m_bad_x_curve, log, functions=funcs_bad_x)
    assert log.has_errors
    assert any("monotonic" in str(e).lower() or "increasing" in str(e).lower() for e in log.errors)

    # 8. Allowed element families check
    for part in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        assert 69 in _ALLOWED_LAWS[part]
        assert "69" in _ALLOWED_LAWS[part]
        assert "LAW69" in _ALLOWED_LAWS[part]
        assert "HYP_ELAS" in _ALLOWED_LAWS[part]
        assert "HYPERELASTIC" in _ALLOWED_LAWS[part]
        assert "LAW69_HYP_ELAS" in _ALLOWED_LAWS[part]

    # Trusses, beams, springs must reject LAW69
    for rejected in ("trusses", "beams"):
        assert 69 not in _ALLOWED_LAWS[rejected]
        assert "LAW69" not in _ALLOWED_LAWS[rejected]


def test_m550_deck_writer_roundtrip(tmp_path):
    deck = StarterDeck("LAW69_ROUNDTRIP")
    deck.mat_law69(
        mat_id=1,
        rho0=1.1e-9,
        rhor=1.1e-9,
        nu=0.495,
        iflag=1,
        fct_id_bulk=0,
        fscale=1.0,
        nip=2,
        fct_id1=10,
        title="Rubber Standard",
    )
    deck.mat_hyp_elas(
        mat_id=2,
        rho0=1.2e-9,
        rhor=1.2e-9,
        nu=0.48,
        iflag=2,
        fct_id_bulk=1,
        fscale=1.2,
        nip=3,
        fct_id1=20,
        title="Rubber HypElas",
    )
    deck.mat_hyperelastic(
        mat_id=3,
        rho0=1.3e-9,
        nu=0.49,
        iflag=1,
        fct_id_bulk=0,
        fscale=1.0,
        nip=2,
        fct_id1=30,
        title="Rubber Hyperelastic",
    )

    deck_path = tmp_path / "test_law69_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law69s
    m1 = model.mat_law69s[1]
    assert m1.id == 1
    assert m1.rho0 == pytest.approx(1.1e-9)
    assert m1.ref_rho == pytest.approx(1.1e-9)
    assert m1.nu == pytest.approx(0.495)
    assert m1.iflag == 1
    assert m1.fct_id_bulk == 0
    assert m1.fscale == pytest.approx(1.0)
    assert m1.nip == 2
    assert m1.fct_id_data == 10

    assert 2 in model.mat_law69s
    m2 = model.mat_law69s[2]
    assert m2.id == 2
    assert m2.rho0 == pytest.approx(1.2e-9)
    assert m2.ref_rho == pytest.approx(1.2e-9)
    assert m2.nu == pytest.approx(0.48)
    assert m2.iflag == 2
    assert m2.fct_id_bulk == 1
    assert m2.fscale == pytest.approx(1.2)
    assert m2.nip == 3
    assert m2.fct_id_data == 20

    assert 3 in model.mat_law69s
    m3 = model.mat_law69s[3]
    assert m3.id == 3
    assert m3.rho0 == pytest.approx(1.3e-9)
    assert m3.nu == pytest.approx(0.49)
    assert m3.iflag == 1
    assert m3.fct_id_data == 30


def test_m550_deck_writer_object_invocation(tmp_path):
    m = MatLaw69(
        id=5,
        rho0=1.0e-9,
        ref_rho=1.0e-9,
        nu=0.485,
        iflag=2,
        fct_id_bulk=3,
        fscale=1.1,
        nip=2,
        fct_id_data=50,
        title="Direct Obj",
    )
    deck = StarterDeck("OBJ_TEST")
    deck.mat_law69(m)
    deck_path = tmp_path / "obj_test_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert 5 in model.mat_law69s
    assert model.mat_law69s[5].nu == pytest.approx(0.485)
    assert model.mat_law69s[5].iflag == 2
    assert model.mat_law69s[5].fct_id_bulk == 3
    assert model.mat_law69s[5].fct_id_data == 50


def test_m550_check_materials_model():
    model = Model()
    m_valid = MatLaw69(id=1, rho0=1.0e-9, nu=0.495)
    m_valid.law = 69
    model.materials[1] = m_valid
    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors

    # Invalid density
    m_bad = MatLaw69(id=2, rho0=-1.0, nu=0.495)
    m_bad.law = 69
    model.materials[2] = m_bad
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)
