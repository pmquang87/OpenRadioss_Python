"""
Input, Starter, and Integration audit tests for LAW15 (M544).
(/MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.input.deck_writer import StarterDeck, _conv_mat
from pyradioss.input.cfg_catalogue import CfgCatalogue
from pyradioss.input.starter_keywords import (
    KEYWORD_PARSERS,
    KeywordBlock,
    Card,
    read_mat_law15,
    read_mat_chang,
    read_mat_plas_aniso,
    read_mat_comp_chang,
)
from pyradioss.model.model import Model
from pyradioss.model.entities import MatLaw15, Material
from pyradioss.starter.checks import check_mat_law15, check_materials, check_model, _ALLOWED_LAWS
from pyradioss.common.messages import MessageLog
from pyradioss import materials
from pyradioss.materials import _STATE_VAR_COUNT
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. DeckWriter API & Fluent Chaining
# ============================================================================

def test_deck_writer_mat_law15_fluent_and_aliases():
    """Verify StarterDeck.mat_law15 and its aliases write valid 7-card and 9-card blocks and return self."""
    deck = StarterDeck("TEST_LAW15")
    res1 = deck.mat_law15(
        mid=1,
        title="CHANG_7CARDS",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
    )
    assert res1 is deck

    res2 = deck.mat_chang(
        mid=2,
        title="CHANG_ALIAS",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
    )
    assert res2 is deck

    res3 = deck.mat_plas_aniso(
        mid=3,
        title="PLAS_ANISO_9CARDS",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
        beta=0.8,
        tmax=0.01,
        s1=2000.0,
        s2=50.0,
        s12=70.0,
        fsmooth=1,
        fcut=1000.0,
        c1=1400.0,
        c2=180.0,
    )
    assert res3 is deck

    res4 = deck.mat_comp_chang(
        mid=4,
        title="COMP_CHANG_ALIAS",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
    )
    assert res4 is deck

    content = deck.render()
    assert "/MAT/LAW15/1" in content
    assert "/MAT/CHANG/2" in content
    assert "/MAT/PLAS_ANISO/3" in content
    assert "/MAT/COMP_CHANG/4" in content


def test_deck_writer_conv_mat():
    """Verify _conv_mat parses and delegates KeywordBlock definitions for LAW15 and synonyms."""
    for law_name, mid in [("LAW15", 10), ("CHANG", 11), ("PLAS_ANISO", 12), ("COMP_CHANG", 13)]:
        deck = StarterDeck(f"CONV_{law_name}")
        block = KeywordBlock(
            keyword=f"MAT/{law_name}",
            parts=["MAT", law_name, str(mid)],
            user_id=mid,
            cards=[
                Card(raw=f"{law_name}_CONV_TEST"),
                Card(raw="1.5e-9"),
                Card(raw="140000.0 8500.0 0.28"),
                Card(raw="4200.0 2800.0 4200.0"),
                Card(raw="80.0 0.6 2200.0"),
                Card(raw="50.0 1.0 1"),
                Card(raw="1600.0 35.0 1100.0 150.0 1.0"),
                Card(raw="55.0 55.0 0.05 1.0 1"),
            ],
            source="test:1",
            fixed=False,
        )
        _conv_mat(deck, block)
        content = deck.render()
        assert f"/MAT/{law_name}/{mid}" in content


# ============================================================================
# 2. CFG Catalogue Mapping
# ============================================================================

def test_cfg_catalogue_law15_resolution():
    """Verify CfgCatalogue maps LAW15, CHANG, PLAS_ANISO, COMP_CHANG to matl15_chang.cfg."""
    cat = CfgCatalogue()
    for name in ("LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        schema = cat.schema(name)
        assert schema is not None, f"Failed to find schema for {name}"
        assert "matl15_chang.cfg" in schema.path.lower()


# ============================================================================
# 3. Starter Keyword Parsing (Fixed & Free Format)
# ============================================================================

def test_starter_keyword_parsing_fixed():
    """Verify read_mat_law15 fixed-format card reading and active material construction."""
    raw_cards = [
        "CHANG_FIXED_TEST",
        "        1.500000e-09",
        "        1.500000e+05        9.000000e+03        3.000000e-01",
        "        4.500000e+03        3.000000e+03        4.500000e+03",
        "        1.000000e+02        5.000000e-01        2.000000e+03",
        "        5.000000e+01        1.000000e+00         1",
        "        1.800000e+03        4.000000e+01        1.200000e+03        1.600000e+02        1.000000e+00",
        "        6.000000e+01        6.000000e+01        5.000000e-02        1.000000e+00         1",
    ]
    cards = [Card(raw=rc, source=f"test:{i+1}") for i, rc in enumerate(raw_cards)]

    block = KeywordBlock(
        keyword="/MAT/LAW15/150",
        user_id=150,
        parts=["MAT", "LAW15", "150"],
        cards=cards,
        fixed=True,
    )
    model = Model()
    log = MessageLog()
    read_mat_law15(block, model, log)

    assert 150 in model.mat_law15s
    m15 = model.mat_law15s[150]
    assert m15.rho0 == pytest.approx(1.5e-9)
    assert m15.e11 == pytest.approx(150000.0)
    assert m15.e22 == pytest.approx(9000.0)
    assert m15.nu12 == pytest.approx(0.3)
    assert m15.g12 == pytest.approx(4500.0)
    assert m15.sig_1yt == pytest.approx(1800.0)

    # Active material object populated in model.materials
    assert 150 in model.materials
    mat = model.materials[150]
    assert getattr(mat, "law", None) == 15
    assert not getattr(mat, "inactive", False)
    assert mat.params["E1"] == pytest.approx(150000.0)


def test_starter_keyword_parsing_free_format_and_aliases():
    """Verify free format card reading and alias functions."""
    raw_cards = [
        "CHANG_FREE_TEST",
        "1.6e-9",
        "140000.0 8500.0 0.32",
        "4200.0 2800.0 4200.0",
        "80.0 0.6 2200.0",
        "60.0 1.0 1",
        "1700.0 45.0 1150.0 150.0 1.0",
        "65.0 65.0 0.04 1.0 1",
        "0.75 0.01 1900.0 48.0 68.0",
        "1 1000.0 1350.0 175.0",
    ]
    cards = [Card(raw=rc, source=f"test:{i+1}") for i, rc in enumerate(raw_cards)]

    block = KeywordBlock(
        keyword="/MAT/CHANG/250",
        user_id=250,
        parts=["MAT", "CHANG", "250"],
        cards=cards,
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_mat_chang(block, model, log)

    assert 250 in model.mat_law15s
    m15 = model.mat_law15s[250]
    assert m15.rho0 == pytest.approx(1.6e-9)
    assert m15.beta == pytest.approx(0.75)
    assert m15.tmax == pytest.approx(0.01)
    assert m15.s1 == pytest.approx(1900.0)
    assert m15.c1 == pytest.approx(1350.0)
    assert m15.c2 == pytest.approx(175.0)


def test_keyword_parsers_registry():
    """Verify KEYWORD_PARSERS dispatches LAW15 and synonyms to read_mat."""
    from pyradioss.input.starter_keywords import read_mat
    for kw in ("LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG", "MAT_LAW15", "MAT_CHANG"):
        assert kw in KEYWORD_PARSERS, f"{kw} missing from KEYWORD_PARSERS"
        assert KEYWORD_PARSERS[kw] is read_mat, f"{kw} does not route to read_mat"


# ============================================================================
# 4. Starter Parameter Checks (check_mat_law15 & check_materials)
# ============================================================================

def test_check_mat_law15_valid():
    """Verify check_mat_law15 succeeds on fully valid material without errors."""
    m = MatLaw15(
        id=1,
        rho0=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=0.8,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
        beta=0.8,
        tmax=0.01,
        s1=2000.0,
        s2=50.0,
        s12=70.0,
        c1=1400.0,
        c2=180.0,
    )
    log = MessageLog()
    check_mat_law15(m, log)
    assert not log.has_errors


def test_check_mat_law15_invalid_bounds():
    """Verify check_mat_law15 catches invalid densities, moduli, yield stresses, hardening, strengths."""
    # 1. rho0 <= 0
    m_rho = MatLaw15(id=1, rho0=0.0, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                     sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_rho, log)
    assert log.has_errors
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    # 2. E1 <= 0 or E2 <= 0
    m_e = MatLaw15(id=2, rho0=1.5e-9, e11=-1.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                   sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_e, log)
    assert log.has_errors
    assert any("Young's modulus E1 must be > 0" in e for e in log.errors)

    # 3. detc <= 0 (nu12 * nu21 >= 1)
    m_detc = MatLaw15(id=3, rho0=1.5e-9, e11=1000.0, e22=1000.0, nu12=5.0, g12=4500.0, g23=3000.0, g31=4500.0,
                     sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_detc, log)
    assert log.has_errors
    assert any("detc = 1 - nu12*nu21 must be > 0" in e for e in log.errors)

    # 4. G12 <= 0
    m_g = MatLaw15(id=4, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=0.0, g23=3000.0, g31=4500.0,
                   sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_g, log)
    assert log.has_errors
    assert any("shear modulus G12 must be > 0" in e for e in log.errors)

    # 5. sig_1yt <= 0
    m_sig = MatLaw15(id=5, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                     sig_1yt=0.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_sig, log)
    assert log.has_errors
    assert any("tensile yield stress in dir 1 (sig_1yt) must be > 0" in e for e in log.errors)

    # 6. Hardening b > 1.0 or n > 1.0
    m_b = MatLaw15(id=6, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                   sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                   b=2.5, n=0.5)
    log = MessageLog()
    check_mat_law15(m_b, log)
    assert log.has_errors
    assert any("hardening parameter b must be <= 1.0" in e for e in log.errors)

    # 7. Strengths S1 <= 0 when failure active
    m_s1 = MatLaw15(id=7, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                    sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                    tmax=0.01, s1=-10.0, s2=50.0, s12=70.0, c1=1400.0, c2=180.0)
    log = MessageLog()
    check_mat_law15(m_s1, log)
    assert log.has_errors
    assert any("longitudinal tensile strength S1 must be > 0" in e for e in log.errors)

    # 8. tmax <= 0 when failure active
    m_tmax = MatLaw15(id=8, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                      sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                      tmax=0.0, s1=2000.0, s2=50.0, s12=70.0, c1=1400.0, c2=180.0)
    log = MessageLog()
    check_mat_law15(m_tmax, log)
    assert log.has_errors
    assert any("stress relaxation time tmax must be > 0" in e for e in log.errors)


# ============================================================================
# 5. Element Family Compatibility (check_model)
# ============================================================================

def test_element_family_compatibility():
    """Verify LAW15 is allowed for shells, shells_qbat, shells_qeph, sh3n, quads and rejected for solids, beams, trusses, springs."""
    # Check _ALLOWED_LAWS table directly
    for family in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        allowed = _ALLOWED_LAWS[family]
        assert 15 in allowed
        assert "15" in allowed
        assert "LAW15" in allowed
        assert "CHANG" in allowed
        assert "PLAS_ANISO" in allowed
        assert "COMP_CHANG" in allowed

    for family in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
        allowed = _ALLOWED_LAWS[family]
        assert 15 not in allowed
        assert "LAW15" not in allowed


@pytest.mark.parametrize("incompatible_elem", ["bricks", "tetras", "penta6", "pyra5", "trusses", "beams", "springs"])
def test_diagnostic_incompatible_element_types(incompatible_elem: str):
    """Verify check_model rejects LAW15 on solid, truss, beam, and spring element groups."""
    model = Model()
    model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
    mat15 = MatLaw15(
        id=15, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
        sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
        b=0.8, n=0.5, fmax=2000.0, wpmax=50.0, wpref=1.0, ioff=1,
    )
    model.materials[15] = mat15

    class FakeGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat15, None)]}

    model.element_groups = lambda: [(incompatible_elem, FakeGroup())]
    log = MessageLog()
    check_model(model, log)
    assert any(f"is not supported for {incompatible_elem} elements" in e for e in log.errors)


@pytest.mark.parametrize("compatible_elem", ["shells", "shells_qbat", "shells_qeph", "sh3n", "quads"])
def test_compatible_element_types_accepted(compatible_elem: str):
    """Verify check_model accepts LAW15 on shell element groups."""
    model = Model()
    model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
    mat15 = MatLaw15(
        id=15, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
        sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
        b=0.8, n=0.5, fmax=2000.0, wpmax=50.0, wpref=1.0, ioff=1,
    )
    model.materials[15] = mat15

    class FakeGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat15, None)]}

    model.element_groups = lambda: [(compatible_elem, FakeGroup())]
    log = MessageLog()
    check_model(model, log)
    assert not any("is not supported for" in e for e in log.errors)


# ============================================================================
# 6. Materials Package Integration & Dispatch
# ============================================================================

def test_materials_registry_and_state_vars():
    """Verify MAT_PHYSICS_REGISTRY and _STATE_VAR_COUNT registration for LAW15."""
    for key in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        assert key in MAT_PHYSICS_REGISTRY, f"Missing key {key} in MAT_PHYSICS_REGISTRY"
        builder = MAT_PHYSICS_REGISTRY[key]
        assert callable(builder)

    assert "uv15" in _STATE_VAR_COUNT
    assert _STATE_VAR_COUNT["uv15"] == (8,)


def test_materials_dispatch():
    """Verify package-level materials dispatch functions for LAW15."""
    mat = materials.law15_chang.build_law15(
        id=15, rho0=1.5e-9, E1=150000.0, E2=9000.0, nu12=0.3, G12=4500.0, G23=3000.0, G31=4500.0,
        sigyt1=1800.0, sigyc1=1200.0, sigyt2=40.0, sigyc2=160.0, sigt12=60.0, sigc12=60.0,
        b=100.0, n=0.5, fmax=2000.0, wpmax=50.0, wpref=1.0, itype=1,
    )

    # 1. sound_speed
    c = materials.sound_speed(mat)
    assert c > 0.0

    # 2. shell_membrane_tangent
    C_mem = materials.shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    assert C_mem[0, 0] > C_mem[1, 1]

    # 3. shell_update
    sig = np.zeros((2, 3), dtype=np.float64)
    deps = np.array([[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0]], dtype=np.float64)
    epsp = np.zeros(2, dtype=np.float64)
    signew, epspnew = materials.shell_update(mat, sig, deps, epsp, dt=1e-6)
    assert signew[0, 0] > 0.0

    # 4. consistent_shell_tangent
    Ct = materials.shell_layer_tangent(mat, signew, epspnew)
    assert Ct.shape == (2, 3, 3)

    # 5. solid_update must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), None, dt=1e-6)
