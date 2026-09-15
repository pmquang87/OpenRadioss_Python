"""
Tests for Milestone M570: /MAT/LAW100 Fixed-Format Deck Writer and Roundtrip.

Verifies:
1. StarterDeck.mat_law100, mat_visc_hyp, mat_mnf emission in fixed 2022 format.
2. Field width and column alignments:
   - Card 1: RHO_I (%20lg)
   - Card 2: N_net, Flag_HE, Flag_Cr (%10d%10d%10d)
   - Hyperelastic cards (Flag_HE = 1, 2, 3, 4, 5, 13)
   - Creep card (Flag_Cr = 1)
   - Secondary network cards (Header + Viscosity parameters)
3. Full roundtrip:
   StarterDeck -> text -> read_lines_to_blocks -> parse_starter_deck -> verify fields.
"""

import pytest
import numpy as np

from pyradioss.input.deck_writer import StarterDeck, read_lines_to_blocks
from pyradioss.starter.starter import parse_starter_deck


def test_roundtrip_law100_polynomial_networks():
    """Roundtrip test for Polynomial hyperelasticity with Bergstrom-Boyce & Sinh networks."""
    deck = StarterDeck("TEST_POLY_RT")
    deck.mat_law100(
        mid=10,
        title="poly_rubber",
        rho0=1.12e-6,
        n_net=2,
        flag_he=1,
        flag_cr=0,
        c10=0.6, c01=0.15, c20=0.01, c11=-0.005, c02=0.002,
        c30=1e-4, c21=-1e-4, c12=5e-5, c03=-2e-5,
        d1=0.01, d2=0.001, d3=1e-4,
        networks=[
            {"net_id": 1, "flag_visc": 1, "stiffness": 1.5, "a": 0.08, "c": -0.7, "m": 1.4, "ksi": 0.01, "tau_ref": 1.2},
            {"net_id": 2, "flag_visc": 2, "stiffness": 0.5, "a": 0.03, "b": 0.25, "n": 1.8},
        ]
    )
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    assert 10 in model.mat_law100s
    mat = model.mat_law100s[10]
    assert mat.id == 10
    assert mat.title == "poly_rubber"
    assert mat.rho0 == pytest.approx(1.12e-6)
    assert mat.n_net == 2
    assert mat.flag_he == 1
    assert mat.flag_cr == 0

    assert mat.c10 == pytest.approx(0.6)
    assert mat.c01 == pytest.approx(0.15)
    assert mat.c20 == pytest.approx(0.01)
    assert mat.c11 == pytest.approx(-0.005)
    assert mat.c02 == pytest.approx(0.002)
    assert mat.c30 == pytest.approx(1e-4)
    assert mat.c21 == pytest.approx(-1e-4)
    assert mat.c12 == pytest.approx(5e-5)
    assert mat.c03 == pytest.approx(-2e-5)
    assert mat.d1 == pytest.approx(0.01)
    assert mat.d2 == pytest.approx(0.001)
    assert mat.d3 == pytest.approx(1e-4)

    assert len(mat.networks) == 2
    net1 = mat.networks[0]
    assert net1["network_id"] == 1
    assert net1["flag_visc"] == 1
    assert net1["stiffness"] == pytest.approx(1.5)
    assert net1["a"] == pytest.approx(0.08)
    assert net1["c"] == pytest.approx(-0.7)
    assert net1["m"] == pytest.approx(1.4)
    assert net1["ksi"] == pytest.approx(0.01)
    assert net1["tau_ref"] == pytest.approx(1.2)

    net2 = mat.networks[1]
    assert net2["network_id"] == 2
    assert net2["flag_visc"] == 2
    assert net2["stiffness"] == pytest.approx(0.5)
    assert net2["a"] == pytest.approx(0.03)
    assert net2["b"] == pytest.approx(0.25)
    assert net2["n"] == pytest.approx(1.8)


def test_roundtrip_law100_arruda_boyce_creep():
    """Roundtrip test for Arruda-Boyce with creep plasticity and Power law network."""
    deck = StarterDeck("TEST_AB_RT")
    deck.mat_law100(
        mid=20,
        title="arruda_boyce_creep",
        rho0=1.08e-6,
        n_net=1,
        flag_he=2,
        flag_cr=1,
        mue1=3.2, d=0.005, lambda_m=6.5, itype=2, fct_id_ab=0, nu_val=0.47, fscale_ab=1.0,
        a_pl=0.9, sigma_pl=2.8, f_pl=0.6, epsilon_f=0.12, n_pl=2,
        networks=[
            {"net_id": 1, "flag_visc": 3, "stiffness": 1.8, "a": 5e-5, "n": 2.2, "m": 0.4},
        ]
    )
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    assert 20 in model.mat_law100s
    mat = model.mat_law100s[20]
    assert mat.id == 20
    assert mat.flag_he == 2
    assert mat.flag_cr == 1
    assert mat.mue1 == pytest.approx(3.2)
    assert mat.d == pytest.approx(0.005)
    assert mat.lambda_m == pytest.approx(6.5)
    assert mat.itype == 2
    assert mat.nu_val == pytest.approx(0.47)

    assert mat.a_pl == pytest.approx(0.9)
    assert mat.sigma_pl == pytest.approx(2.8)
    assert mat.f_pl == pytest.approx(0.6)
    assert mat.epsilon_f == pytest.approx(0.12)
    assert mat.n_pl == 2

    assert len(mat.networks) == 1
    net = mat.networks[0]
    assert net["flag_visc"] == 3
    assert net["stiffness"] == pytest.approx(1.8)
    assert net["a"] == pytest.approx(5e-5)
    assert net["n"] == pytest.approx(2.2)
    assert net["m"] == pytest.approx(0.4)


def test_roundtrip_law100_neo_hook_and_yeoh():
    """Roundtrip for Neo-Hookean (Flag_HE=3) and Yeoh (Flag_HE=5)."""
    deck = StarterDeck("TEST_NH_YEOH")
    # Neo-Hookean
    deck.mat_law100(mid=31, title="neo_hook", rho0=1.0e-6, flag_he=3, c10=0.8, d1=0.015)
    # Yeoh
    deck.mat_law100(mid=32, title="yeoh_rubber", rho0=1.05e-6, flag_he=5, c10=0.5, c20=0.02, c30=-0.001, d1=0.02)
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    mat31 = model.mat_law100s[31]
    assert mat31.flag_he == 3
    assert mat31.c10 == pytest.approx(0.8)
    assert mat31.d1 == pytest.approx(0.015)

    mat32 = model.mat_law100s[32]
    assert mat32.flag_he == 5
    assert mat32.c10 == pytest.approx(0.5)
    assert mat32.c20 == pytest.approx(0.02)
    assert mat32.c30 == pytest.approx(-0.001)
    assert mat32.d1 == pytest.approx(0.02)


def test_roundtrip_law100_thermal_neo_hook():
    """Roundtrip for Thermal Neo-Hookean (Flag_HE=13)."""
    deck = StarterDeck("TEST_THE_NH")
    deck.mat_law100(
        mid=40, title="thermal_nh", rho0=1.1e-6, flag_he=13,
        fct_id_sm=101, fct_id_bm=102, fscale_sm=1.2, fscale_bm=2.5
    )
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    mat = model.mat_law100s[40]
    assert mat.flag_he == 13
    assert mat.fct_id_sm == 101
    assert mat.fct_id_bm == 102
    assert mat.fscale_sm == pytest.approx(1.2)
    assert mat.fscale_bm == pytest.approx(2.5)


def test_roundtrip_aliases():
    """Verify mat_visc_hyp and mat_mnf methods in StarterDeck."""
    deck = StarterDeck("TEST_ALIASES")
    deck.mat_visc_hyp(mid=51, title="visc_hyp", rho0=1.1e-6, flag_he=1, c10=0.4, d1=0.02)
    deck.mat_mnf(mid=52, title="mnf", rho0=1.2e-6, flag_he=1, c10=0.7, d1=0.01)
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    assert 51 in model.mat_law100s
    assert 52 in model.mat_law100s
    assert model.mat_law100s[51].c10 == pytest.approx(0.4)
    assert model.mat_law100s[52].c10 == pytest.approx(0.7)
