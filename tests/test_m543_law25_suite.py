"""Tests for /MAT/LAW25 composite plasticity (Tsai-Wu and CRASURV formulations)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.cfg_catalogue import CfgCatalogue, law_number, LAW_MAP, LAW_SYNONYMS
from pyradioss.input.deck_reader import Card
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.materials import (
    _STATE_VAR_COUNT,
    build_law25,
    consistent_shell_tangent,
    consistent_solid_tangent,
    extra_shapes,
    register_materials,
    shell_membrane_tangent,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.model.entities import Material, MatLaw25
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law25, check_materials


def test_cfg_catalogue_law25_resolution():
    cat = CfgCatalogue()
    for name in ("LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        law_num = law_number(name)
        assert law_num == 25, f"Expected 25 for {name}, got {law_num}"
        assert LAW_MAP[name] == 25
        assert LAW_SYNONYMS[name] == "LAW25"
        canonical = cat.canonical_law_name(name)
        assert canonical == "LAW25"
        schema = cat.schema(name)
        assert schema is not None, f"Failed to get schema for {name}"
        assert schema.law_number == 25


def test_starter_deck_law25_methods():
    deck = StarterDeck("TEST_0000.rad")
    assert hasattr(deck, "mat_law25")
    assert hasattr(deck, "mat_comp_plas")
    assert hasattr(deck, "mat_compsh")
    assert hasattr(deck, "mat_tsai_wu")
    assert hasattr(deck, "mat_crasurv")

    # Add Tsai-Wu material
    deck.mat_law25(
        mat_id=1,
        title="CARBON_EPOXY",
        rho0=1.55e-6,
        e11=140000.0,
        e22=9500.0,
        nu12=0.32,
        iform=0,
        e33=9500.0,
        g12=5000.0,
        g23=3200.0,
        g31=5000.0,
        eps_f1=0.02,
        eps_f2=0.04,
        eps_t1=0.01,
        eps_m1=0.015,
        eps_t2=0.02,
        eps_m2=0.03,
        dmax=0.8,
        wpmax=40.0,
        wpref=1.0,
        ioff=0,
        b=150.0,
        n=0.5,
        fmax=450.0,
        sig_1yt=1600.0,
        sig_2yt=45.0,
        sig_1yc=1300.0,
        sig_2yc=160.0,
        alpha=0.0,
        sig_12yc=70.0,
        sig_12yt=70.0,
        c=0.0,
        eps_rate_0=0.0,
        icc=0,
    )

    # Add CRASURV material
    deck.mat_crasurv(
        mat_id=2,
        title="CRASURV_MAT",
        rho0=1.6e-6,
        e11=120000.0,
        e22=8000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        eps_f1=0.025,
        eps_f2=0.035,
        eps_t1=0.012,
        eps_m1=0.018,
        eps_t2=0.022,
        eps_m2=0.032,
        dmax=0.85,
        b_1t=100.0,
        n_1t=0.6,
        sig_1maxt=1500.0,
        c_1t=10.0,
        eps_1t1=0.01,
        eps_2t1=0.02,
        sig_rst1=200.0,
        wpmax_t1=50.0,
    )

    text = deck.render()
    assert "/MAT/LAW25/1" in text or "/MAT/COMP_PLAS/1" in text
    assert "CARBON_EPOXY" in text
    assert "/MAT/LAW25/2" in text or "/MAT/CRASURV/2" in text or "/MAT/COMP_PLAS/2" in text
    assert "CRASURV_MAT" in text


def test_starter_read_tsai_wu_roundtrip():
    from pyradioss.input.starter_keywords import KeywordBlock, read_mat_law25

    raw_cards = [
        "Composite Plasticity",
        "1.55e-6 0.0",
        "140000.0 9500.0 0.32 0 9500.0",
        "5000.0 3200.0 5000.0 0.02 0.04",
        "0.01 0.015 0.02 0.03 0.8",
        "40.0 1.0 0",
        "150.0 0.5 450.0",
        "1600.0 45.0 1300.0 160.0 0.0",
        "70.0 70.0 0.0 0.0 0",
    ]
    cards = [Card(raw=rc, source=f"test:{i+1}") for i, rc in enumerate(raw_cards)]
    block = KeywordBlock(
        keyword="MAT/LAW25",
        parts=["MAT", "LAW25", "601"],
        user_id=601,
        cards=cards,
        source="test:1",
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_mat_law25(block, model, log)

    assert not log.has_errors
    assert 601 in model.mat_law25s
    mat = model.mat_law25s[601]
    assert isinstance(mat, MatLaw25)
    assert mat.rho0 == pytest.approx(1.55e-6)
    assert mat.e11 == pytest.approx(140000.0)
    assert mat.e22 == pytest.approx(9500.0)
    assert mat.nu12 == pytest.approx(0.32)
    assert mat.iform == 0
    assert mat.g12 == pytest.approx(5000.0)
    assert mat.g23 == pytest.approx(3200.0)
    assert mat.g31 == pytest.approx(5000.0)
    assert mat.dmax == pytest.approx(0.8)
    assert mat.sig_1yt == pytest.approx(1600.0)
    assert mat.sig_2yt == pytest.approx(45.0)
    assert mat.sig_1yc == pytest.approx(1300.0)
    assert mat.sig_2yc == pytest.approx(160.0)

    # Check model.materials
    assert 601 in model.materials
    phys_mat = model.materials[601]
    assert isinstance(phys_mat, Material)
    assert phys_mat.law == 25


def test_starter_read_crasurv():
    from pyradioss.input.starter_keywords import KeywordBlock, read_mat_law25

    raw_cards = [
        "CRASURV Formulation",
        "1.55e-6 0.0",
        "140000.0 9500.0 0.32 1 9500.0",
        "5000.0 3200.0 5000.0 0.02 0.04",
        "0.01 0.015 0.02 0.03 0.8",
        # CRASURV cards
        "40.0 1.0 0 1",                             # card 5: wpmax wpref ioff iflawp
        "0.0 0.0 0.0 0",                            # card 6: c eps_rate_0 alpha icc
        "1600.0 100.0 0.5 1500.0 10.0",             # card 7: sig_1yt b_1t n_1t sig_1maxt c_1t
        "0.01 0.02 200.0 50.0",                     # card 8: eps_1t1 eps_2t1 sig_rst1 wpmax_t1
        "45.0 80.0 0.5 1200.0 8.0",                 # card 9: sig_2yt b_2t n_2t sig_2maxt c_2t
        "0.01 0.02 150.0 40.0",                     # card 10: eps_1t2 eps_2t2 sig_rst2 wpmax_t2
        "1300.0 90.0 0.5 1300.0 9.0",               # card 11: sig_1yc b_1c n_1c sig_1maxc c_1c
        "0.01 0.02 180.0 45.0",                     # card 12: eps_1c1 eps_2c1 sig_rsc1 wpmax_c1
        "160.0 70.0 0.5 1100.0 7.0",                # card 13: sig_2yc b_2c n_2c sig_2maxc c_2c
        "0.01 0.02 140.0 35.0",                     # card 14: eps_1c2 eps_2c2 sig_rsc2 wpmax_c2
        "70.0 50.0 0.5 800.0 5.0",                  # card 15: sig_12yt b_12t n_12t sig_12maxt c_12t
        "0.01 0.02 100.0 25.0",                     # card 16: eps_1t12 eps_2t12 sig_rst12 wpmax_t12
        "0.005 0.05 0.9",                           # card 17: gamma_ini, gamma_max, d3max
        "1 1000.0",                                 # card 18: fsmooth, fcut
    ]
    cards = [Card(raw=rc, source=f"test:{i+1}") for i, rc in enumerate(raw_cards)]
    block = KeywordBlock(
        keyword="MAT/CRASURV",
        parts=["MAT", "CRASURV", "701"],
        user_id=701,
        cards=cards,
        source="test:1",
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_mat_law25(block, model, log)

    assert not log.has_errors
    assert 701 in model.mat_law25s
    mat = model.mat_law25s[701]
    assert isinstance(mat, MatLaw25)
    assert mat.b_1t == pytest.approx(100.0)
    assert mat.sig_1maxt == pytest.approx(1500.0)
    assert mat.gamma_ini == pytest.approx(0.005)
    assert mat.gamma_max == pytest.approx(0.05)
    assert mat.d3max == pytest.approx(0.9)
    assert mat.fsmooth == 1
    assert mat.fcut == pytest.approx(1000.0)

    # Check model.materials
    assert 701 in model.materials
    phys_mat = model.materials[701]
    assert isinstance(phys_mat, Material)
    assert phys_mat.law == 25


def test_checks_mat_law25():
    # Valid material
    valid_mat = MatLaw25(
        id=1,
        title="VALID_COMP",
        rho0=1.5e-6,
        e11=140000.0,
        e22=9500.0,
        nu12=0.3,
        g12=5000.0,
        g23=3200.0,
        g31=5000.0,
        sig_1yt=1500.0,
        sig_2yt=50.0,
        sig_1yc=1200.0,
        sig_2yc=150.0,
        sig_12yc=70.0,
        sig_12yt=70.0,
        n=0.5,
        dmax=0.8,
    )
    log = MessageLog()
    check_mat_law25(valid_mat, log)
    assert not log.has_errors

    # Invalid density
    bad_rho = MatLaw25(id=2, rho0=0.0, e11=100.0, e22=100.0)
    log = MessageLog()
    check_mat_law25(bad_rho, log)
    assert log.has_errors

    # Invalid Young's moduli
    bad_e = MatLaw25(id=3, rho0=1.0, e11=0.0, e22=-5.0)
    log = MessageLog()
    check_mat_law25(bad_e, log)
    assert log.has_errors

    # Invalid detc (nu12 * nu21 >= 1)
    bad_detc = MatLaw25(id=4, rho0=1.0, e11=100.0, e22=100.0, nu12=1.2)
    log = MessageLog()
    check_mat_law25(bad_detc, log)
    assert log.has_errors

    # Invalid hardening exponent n > 1.0
    bad_n = MatLaw25(id=5, rho0=1.0, e11=100.0, e22=100.0, nu12=0.3, g12=10.0, g23=10.0, g31=10.0,
                     sig_1yt=10.0, sig_2yt=10.0, sig_1yc=10.0, sig_2yc=10.0, sig_12yc=10.0, sig_12yt=10.0,
                     n=1.5)
    log = MessageLog()
    check_mat_law25(bad_n, log)
    assert log.has_errors

    # Invalid dmax > 1.0
    bad_dmax = MatLaw25(id=6, rho0=1.0, e11=100.0, e22=100.0, nu12=0.3, g12=10.0, g23=10.0, g31=10.0,
                       sig_1yt=10.0, sig_2yt=10.0, sig_1yc=10.0, sig_2yc=10.0, sig_12yc=10.0, sig_12yt=10.0,
                       n=0.5, dmax=1.5)
    log = MessageLog()
    check_mat_law25(bad_dmax, log)
    assert log.has_errors

    # Verify check_materials checks law25
    model = Model()
    model.materials[valid_mat.id] = valid_mat
    model.materials[bad_rho.id] = bad_rho
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors


def test_allowed_laws_includes_law25():
    for elem_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "bricks", "tetras", "penta6", "pyra5"):
        allowed = _ALLOWED_LAWS[elem_type]
        for key in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
            assert key in allowed, f"{key} missing in _ALLOWED_LAWS[{elem_type}]"


def test_mat_physics_registry_and_materials_init():
    register_materials()
    assert _STATE_VAR_COUNT["uv25"] == (12,)

    for key in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        assert key in MAT_PHYSICS_REGISTRY, f"{key} missing in MAT_PHYSICS_REGISTRY"

    builder = MAT_PHYSICS_REGISTRY["LAW25"]
    mat = builder(
        id=10,
        rho0=1.5e-6,
        e11=140000.0,
        e22=9500.0,
        nu12=0.32,
        g12=5000.0,
        g23=3200.0,
        g31=5000.0,
        sig_1yt=1600.0,
        sig_2yt=45.0,
        sig_1yc=1300.0,
        sig_2yc=160.0,
        sig_12yc=70.0,
        sig_12yt=70.0,
        n=0.5,
        dmax=0.8,
    )
    assert isinstance(mat, Material)
    assert mat.law == 25

    # Sound speed
    c = sound_speed(mat)
    assert c > 0.0

    # Extra shapes
    shapes_shell = extra_shapes(mat, nip=3)
    assert "dmg25" in shapes_shell
    assert "stra25" in shapes_shell
    assert shapes_shell["dmg25"] == (3, 4) or shapes_shell["dmg25"][0] == 3

    # Shell update
    sig = np.zeros(3)
    deps = np.array([0.001, 0.0002, 0.0001])
    extra = {"dmg25": np.zeros((1, 8)), "off25": np.ones(1), "stra25": np.zeros((1, 3)), "wpla25": np.zeros(1)}
    s_new, ep_new = shell_update(mat, sig, deps, epsp=0.0, dt=1e-6, extra=extra)
    assert s_new.shape == (3,)
    assert s_new[0] > 0.0

    # Solid update
    sig6 = np.zeros(6)
    deps6 = np.array([0.001, 0.0002, 0.0001, 0.0001, 0.0, 0.0])
    extra6 = {"dmg25": np.zeros((1, 8)), "off25": np.ones(1), "stra25": np.zeros((1, 6)), "wpla25": np.zeros(1)}
    s6_new, ep6_new, c_val = solid_update(mat, sig6, deps6, epsp=0.0, dt=1e-6, extra=extra6)
    assert s6_new.shape == (6,)
    assert s6_new[0] > 0.0
    assert c_val > 0.0

    # Tangents
    c_mem = shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)

    t_solid = consistent_solid_tangent(mat, np.zeros((1, 6)), epsp=np.zeros(1), epsp_incr=np.zeros(1))
    assert t_solid.shape == (1, 6, 6)

    t_shell = consistent_shell_tangent(mat, np.zeros((1, 3)), epsp=np.zeros(1), epsp_incr=np.zeros(1))
    assert t_shell.shape == (1, 3, 3)
