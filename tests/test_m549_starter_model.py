"""
Tests for M549 (/MAT/LAW82 /MAT/OGDEN) input, starter, model, and checks integration.
"""

from io import StringIO
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import (
    MAT_LAW82_1,
    MAT_LAW82_2,
    MAT_LAW82_LIST,
    MAT_OGDEN_1,
    MAT_OGDEN_2,
    MAT_OGDEN_LIST,
    MAT_LAW82_OGDEN_1,
    MAT_LAW82_OGDEN_2,
    MAT_LAW82_OGDEN_LIST,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS
from pyradioss.model.entities import MatLaw82
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law82, check_materials, _ALLOWED_LAWS
import pyradioss.materials as materials


def test_m549_card_layouts():
    assert MAT_LAW82_1 == [20, 20]
    assert MAT_LAW82_2 == [10, 10, 20]
    assert MAT_LAW82_LIST == [20, 20, 20, 20, 20]

    assert MAT_OGDEN_1 == MAT_LAW82_1
    assert MAT_OGDEN_2 == MAT_LAW82_2
    assert MAT_OGDEN_LIST == MAT_LAW82_LIST

    assert MAT_LAW82_OGDEN_1 == MAT_LAW82_1
    assert MAT_LAW82_OGDEN_2 == MAT_LAW82_2
    assert MAT_LAW82_OGDEN_LIST == MAT_LAW82_LIST

    for k in (
        "MAT_LAW82_1", "MAT_LAW82_2", "MAT_LAW82_LIST",
        "MAT_OGDEN_1", "MAT_OGDEN_2", "MAT_OGDEN_LIST",
        "MAT_LAW82_OGDEN_1", "MAT_LAW82_OGDEN_2", "MAT_LAW82_OGDEN_LIST",
    ):
        assert k in LAYOUTS


def test_m549_cfg_catalogue():
    assert LAW_MAP["LAW82"] == 82
    assert LAW_MAP["OGDEN"] == 82
    assert LAW_MAP["LAW82_OGDEN"] == 82

    assert LAW_SYNONYMS["LAW82"] == "LAW82"
    assert LAW_SYNONYMS["OGDEN"] == "LAW82"
    assert LAW_SYNONYMS["LAW82_OGDEN"] == "LAW82"


def test_m549_model_entities_matlaw82():
    # Canonical initialization
    m = MatLaw82(
        id=42,
        rho0=1.1e-9,
        rhor=1.1e-9,
        nu=0.49,
        nordre=2,
        mu=[10.0, 5.0],
        alpha=[2.0, -2.0],
        d=[0.01, 0.02],
        title="My Ogden",
    )
    assert m.id == 42
    assert m.rho0 == 1.1e-9
    assert m.rho == 1.1e-9
    assert m.rhor == 1.1e-9
    assert m.ref_rho == 1.1e-9
    assert m.nu == 0.49
    assert m.nordre == 2
    assert m.order == 2
    assert m.mu == [10.0, 5.0]
    assert m.mu_arr == [10.0, 5.0]
    assert m.alpha == [2.0, -2.0]
    assert m.alpha_arr == [2.0, -2.0]
    assert m.d == [0.01, 0.02]
    assert m.gamma_arr == [0.01, 0.02]

    # Legacy property setters
    m.rho = 2.0e-9
    assert m.rho0 == 2.0e-9
    m.ref_rho = 1.9e-9
    assert m.rhor == 1.9e-9
    m.order = 3
    assert m.nordre == 3
    m.mu_arr = [1.0, 2.0, 3.0]
    assert m.mu == [1.0, 2.0, 3.0]
    m.alpha_arr = [4.0, 5.0, 6.0]
    assert m.alpha == [4.0, 5.0, 6.0]
    m.gamma_arr = [7.0, 8.0, 9.0]
    assert m.d == [7.0, 8.0, 9.0]

    # Legacy initialization via kwargs
    m_leg = MatLaw82(
        id=7,
        rho=1.5e-9,
        ref_rho=1.5e-9,
        order=1,
        nu=0.475,
        mu_arr=[50.0],
        alpha_arr=[2.5],
        gamma_arr=[0.0],
    )
    assert m_leg.rho0 == 1.5e-9
    assert m_leg.nordre == 1
    assert m_leg.mu == [50.0]
    assert m_leg.alpha == [2.5]
    assert m_leg.d == [0.0]


def test_m549_deck_writer_roundtrip(tmp_path):
    deck = StarterDeck("LAW82_TEST")
    deck.mat_law82(
        mat_id=1,
        rho0=1.0e-9,
        rhor=1.0e-9,
        nu=0.495,
        nordre=2,
        mu=[0.5, 0.2],
        alpha=[2.0, -2.0],
        d=[0.0, 0.0],
        title="Rubber",
    )

    deck_path = tmp_path / "test_law82_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law82s
    m = model.mat_law82s[1]
    assert m.id == 1
    assert m.rho0 == pytest.approx(1.0e-9)
    assert m.nu == pytest.approx(0.495)
    assert m.nordre == 2
    assert m.mu == pytest.approx([0.5, 0.2])
    assert m.alpha == pytest.approx([2.0, -2.0])
    assert m.d == pytest.approx([0.0, 0.0])


def test_m549_starter_checks():
    class DummyMat:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 1.0)
            self.rhor = kwargs.get("rhor", 0.0)
            self.nu = kwargs.get("nu", 0.475)
            self.nordre = kwargs.get("nordre", 1)
            self.mu = kwargs.get("mu", [10.0])
            self.alpha = kwargs.get("alpha", [2.0])
            self.d = kwargs.get("d", [0.0])

    # Valid
    m_valid = DummyMat()
    log = MessageLog()
    check_mat_law82(m_valid, log)
    assert not log.has_errors

    # rho0 <= 0
    m_bad_rho = DummyMat(rho0=0.0)
    log = MessageLog()
    check_mat_law82(m_bad_rho, log)
    assert log.has_errors
    assert any("rho0" in str(e).lower() or "density" in str(e).lower() for e in log.errors)

    # nordre < 1
    m_bad_n = DummyMat(nordre=0)
    log = MessageLog()
    check_mat_law82(m_bad_n, log)
    assert log.has_errors
    assert any("order" in str(e).lower() or "nordre" in str(e).lower() for e in log.errors)

    # sum(mu) <= 0
    m_bad_mu = DummyMat(mu=[-1.0])
    log = MessageLog()
    check_mat_law82(m_bad_mu, log)
    assert log.has_errors
    assert any("shear" in str(e).lower() or "mu" in str(e).lower() for e in log.errors)

    # nu < 0 or nu >= 0.5
    m_bad_nu = DummyMat(nu=0.5)
    log = MessageLog()
    check_mat_law82(m_bad_nu, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    # alpha[i] == 0
    m_bad_alpha = DummyMat(alpha=[0.0])
    log = MessageLog()
    check_mat_law82(m_bad_alpha, log)
    assert log.has_errors
    assert any("alpha" in str(e).lower() for e in log.errors)

    # Allowed laws checks
    for part in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
        assert 82 in _ALLOWED_LAWS[part]
        assert "82" in _ALLOWED_LAWS[part]
        assert "LAW82" in _ALLOWED_LAWS[part]
        assert "OGDEN" in _ALLOWED_LAWS[part]


def test_m549_materials_exports():
    assert hasattr(materials, "OgdenParams")
    assert hasattr(materials, "build_law82")
    assert hasattr(materials, "law82_solid_update")
    assert hasattr(materials, "law82_shell_update")
    assert hasattr(materials, "law82_solid_sound_speed")
    assert hasattr(materials, "law82_shell_sound_speed")
    assert hasattr(materials, "law82_consistent_solid_tangent")
    assert hasattr(materials, "law82_consistent_shell_tangent")
