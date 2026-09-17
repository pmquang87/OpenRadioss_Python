"""M580: Equations of State — /EOS/GRUNEISEN (IEOS=2) and /EOS/TILLOTSON (IEOS=5).

Validates:
1. Gruneisen EOS (common_source/eos/gruneisen.F):
   - Closed-form Hugoniot curve in compression (mu > 0)
   - Linear acoustic sound speed in expansion (mu < 0)
   - Initial state and acoustic sound speed c0 = C
   - Coupled implicit E-p update algebraic consistency
2. Tillotson EOS (common_source/eos/tillotson.F):
   - Region 1 (compression): bulk modulus C1 + quadratic C2 + nonlinear thermal
   - Region 2 (cold expansion): cavitation/spall branch with C2 dropped
   - Region 4 (hot expansion / complete vaporization): exponential cutoff
   - Initial state and bulk sound speed c0 = sqrt(C1 / rho0)
   - Predictor-corrector implicit E-p update
3. Starter deck parsing (/EOS/GRUNEISEN and /EOS/TILLOTSON).
4. Integration with solid element stress / sound speed calculation.
"""

import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState, Material
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_eos
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


# ============================================================================
# /EOS/GRUNEISEN tests
# ============================================================================

def test_gruneisen_initial_state_and_sound_speed():
    """Verify initial state and reference acoustic sound speed c0 = C."""
    rho0 = 8930.0  # Copper (kg/m^3)
    c0 = 3940.0    # Sound speed (m/s)
    gamma0 = 2.0
    a = 0.47
    eos = EquationOfState(
        kind="GRUNEISEN", rho0=rho0,
        params={"c": c0, "s1": 1.489, "s2": 0.0, "s3": 0.0,
                "gamma0": gamma0, "a": a, "e0": 0.0, "rho0_card": rho0})

    e0, p0 = eos_mod.initial_state(eos)
    assert e0 == 0.0
    assert p0 == 0.0

    # At mu = 0, E = 0: sound speed must equal C0 exactly
    c_bulk = eos_mod.sound_speed(eos, 0.0, 0.0)
    assert c_bulk == pytest.approx(c0, rel=1e-7)

    # Initial state with non-zero e0
    eos_e0 = EquationOfState(
        kind="GRUNEISEN", rho0=rho0,
        params={"c": c0, "s1": 1.489, "s2": 0.0, "s3": 0.0,
                "gamma0": gamma0, "a": a, "e0": 1.0e6, "rho0_card": rho0})
    e0_val, p0_val = eos_mod.initial_state(eos_e0)
    assert e0_val == 1.0e6
    assert p0_val == pytest.approx(gamma0 * 1.0e6, rel=1e-12)


def test_gruneisen_compression_hugoniot():
    """Verify compression branch against exact Hugoniot formula at E=0."""
    rho0 = 2700.0  # Aluminum
    c0 = 5328.0
    s1 = 1.338
    gamma0 = 2.0
    a = 0.48
    eos = EquationOfState(
        kind="GRUNEISEN", rho0=rho0,
        params={"c": c0, "s1": s1, "s2": 0.0, "s3": 0.0,
                "gamma0": gamma0, "a": a, "e0": 0.0, "rho0_card": rho0})

    mus = np.array([0.05, 0.15, 0.30])
    for mu in mus:
        # Exact Hugoniot pressure at E=0:
        # p_H = rho0 * C^2 * mu * [1 + (1 - gamma0/2)*mu - a/2 * mu^2] / [1 - (s1 - 1)*mu]^2
        ff = 1.0 + (1.0 - 0.5 * gamma0) * mu - 0.5 * a * (mu ** 2)
        fg = 1.0 - (s1 - 1.0) * mu
        p_expected = (ff / (fg ** 2)) * rho0 * (c0 ** 2) * mu

        p_calc = eos_mod.pressure(eos, mu, 0.0)
        assert p_calc == pytest.approx(p_expected, rel=1e-12)

        # Non-zero energy adds (gamma0 + a * mu) * E
        E_val = 5.0e6
        p_with_E = eos_mod.pressure(eos, mu, E_val)
        assert p_with_E == pytest.approx(p_expected + (gamma0 + a * mu) * E_val, rel=1e-12)


def test_gruneisen_expansion_branch():
    """In expansion (mu < 0), FAC=1: p = rho0 * C^2 * mu + (gamma0 + a*mu)*E."""
    rho0 = 7850.0  # Steel
    c0 = 4570.0
    gamma0 = 1.6
    a = 0.5
    eos = EquationOfState(
        kind="GRUNEISEN", rho0=rho0,
        params={"c": c0, "s1": 1.49, "s2": 0.0, "s3": 0.0,
                "gamma0": gamma0, "a": a, "e0": 0.0, "rho0_card": rho0})

    mu_exp = -0.05
    E_val = 1.0e6
    p_exp = eos_mod.pressure(eos, mu_exp, E_val)
    expected = rho0 * (c0 ** 2) * mu_exp + (gamma0 + a * mu_exp) * E_val
    assert p_exp == pytest.approx(expected, rel=1e-12)


def test_gruneisen_implicit_update_exact():
    """Implicit E-p update must satisfy E_new = E_old + dE - 0.5*dv*(p_old + p_new)."""
    rho0 = 8930.0
    c0 = 3940.0
    eos = EquationOfState(
        kind="GRUNEISEN", rho0=rho0,
        params={"c": c0, "s1": 1.489, "s2": 0.0, "s3": 0.0,
                "gamma0": 2.0, "a": 0.47, "e0": 0.0, "rho0_card": rho0})

    mu = np.array([0.1, -0.04, 0.25])
    dv = np.array([-0.015, 0.008, -0.035])
    e_old = np.array([1.0e6, 5.0e5, 2.0e6])
    p_old = np.array([2.0e9, -5.0e8, 8.0e9])
    de = np.array([5.0e4, 1.0e4, 1.0e5])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de)

    # Check implicit energy balance
    e_balance = e_old + de - 0.5 * dv * (p_old + p_new)
    assert np.allclose(e_new, e_balance, rtol=1e-12)

    # Check p_new = pressure(eos, mu, e_new)
    A, B = eos_mod.coefficients(eos, mu)
    assert np.allclose(p_new, A + B * e_new, rtol=1e-12)
    assert np.all(c2 >= 0.0)


# ============================================================================
# /EOS/TILLOTSON tests
# ============================================================================

def test_tillotson_initial_state_and_sound_speed():
    """Verify Tillotson initial state and acoustic bulk sound speed c0 = sqrt(C1/rho0)."""
    rho0 = 7800.0
    c1 = 1.3e11   # Bulk modulus A
    c2 = 1.0e11   # B
    a = 0.5
    b = 1.5
    er = 1.0e10   # E0
    es = 2.0e10   # Es
    vs = 1.5      # Vs
    alpha = 5.0
    beta = 5.0

    eos = EquationOfState(
        kind="TILLOTSON", rho0=rho0,
        params={"c1": c1, "c2": c2, "a": a, "b": b, "er": er, "es": es,
                "vs": vs, "alpha": alpha, "beta": beta, "e0": 0.0, "rho0_card": rho0})

    e0, p0 = eos_mod.initial_state(eos)
    assert e0 == 0.0
    assert p0 == 0.0

    c_bulk = eos_mod.sound_speed(eos, 0.0, 0.0)
    expected_c0 = np.sqrt(c1 / rho0)
    assert c_bulk == pytest.approx(expected_c0, rel=1e-7)


def test_tillotson_region1_compression():
    """Region 1 (mu >= 0): p = (a + b / (E / (E0*eta^2) + 1)) * eta * E + C1*mu + C2*mu^2."""
    rho0 = 7800.0
    c1, c2 = 1.3e11, 1.0e11
    a, b = 0.5, 1.5
    er = 1.0e10
    eos = EquationOfState(
        kind="TILLOTSON", rho0=rho0,
        params={"c1": c1, "c2": c2, "a": a, "b": b, "er": er, "es": 2.0e10,
                "vs": 1.5, "alpha": 5.0, "beta": 5.0, "e0": 0.0, "rho0_card": rho0})

    mu = 0.1
    eta = 1.0 + mu
    e = 1.0e9
    omega = 1.0 + e / (er * (eta ** 2))
    expected_p = (a + b / omega) * eta * e + c1 * mu + c2 * (mu ** 2)

    p_calc = eos_mod.pressure(eos, mu, e)
    assert p_calc == pytest.approx(expected_p, rel=1e-12)


def test_tillotson_region2_cold_expansion():
    """Region 2 (mu < 0, V/V0 <= Vs, E < Es): C2*mu^2 is dropped."""
    rho0 = 7800.0
    c1, c2 = 1.3e11, 1.0e11
    a, b = 0.5, 1.5
    er, es, vs = 1.0e10, 2.0e10, 1.5
    eos = EquationOfState(
        kind="TILLOTSON", rho0=rho0,
        params={"c1": c1, "c2": c2, "a": a, "b": b, "er": er, "es": es,
                "vs": vs, "alpha": 5.0, "beta": 5.0, "e0": 0.0, "rho0_card": rho0})

    mu = -0.1  # eta = 0.9, df = 1/0.9 = 1.111 <= vs (1.5)
    eta = 1.0 + mu
    e = 1.0e9  # e < es (2.0e10)
    omega = 1.0 + e / (er * (eta ** 2))
    expected_p = (a + b / omega) * eta * e + c1 * mu  # NO c2 * mu^2

    p_calc = eos_mod.pressure(eos, mu, e)
    assert p_calc == pytest.approx(expected_p, rel=1e-12)


def test_tillotson_region4_hot_expansion():
    """Region 4 (mu < 0 and (V/V0 > Vs or E >= Es)): exponential gas vapor branch."""
    rho0 = 7800.0
    c1, c2 = 1.3e11, 1.0e11
    a, b = 0.5, 1.5
    er, es, vs = 1.0e10, 2.0e10, 1.5
    alpha, beta = 5.0, 5.0
    eos = EquationOfState(
        kind="TILLOTSON", rho0=rho0,
        params={"c1": c1, "c2": c2, "a": a, "b": b, "er": er, "es": es,
                "vs": vs, "alpha": alpha, "beta": beta, "e0": 0.0, "rho0_card": rho0})

    mu = -0.4  # eta = 0.6, df = 1/0.6 = 1.667 > vs (1.5) -> Region 4
    eta = 1.0 + mu
    e = 3.0e10
    xx = mu / eta
    expa = np.exp(-alpha * (xx ** 2))
    expb = np.exp(beta * xx)
    omega = 1.0 + e / (er * (eta ** 2))

    expected_p = (a + expa * b / omega) * eta * e + (expa * expb) * c1 * mu
    p_calc = eos_mod.pressure(eos, mu, e)
    assert p_calc == pytest.approx(expected_p, rel=1e-12)


def test_tillotson_implicit_update_predictor_corrector():
    """Verify Tillotson implicit update handles multi-element arrays with positive c2."""
    rho0 = 7800.0
    eos = EquationOfState(
        kind="TILLOTSON", rho0=rho0,
        params={"c1": 1.3e11, "c2": 1.0e11, "a": 0.5, "b": 1.5,
                "er": 1.0e10, "es": 2.0e10, "vs": 1.5, "alpha": 5.0,
                "beta": 5.0, "e0": 0.0, "rho0_card": rho0})

    mu = np.array([0.08, -0.05, -0.35])
    dv = np.array([-0.01, 0.006, 0.05])
    e_old = np.array([1.0e9, 5.0e8, 2.5e10])
    p_old = np.array([1.5e10, -5.0e9, 4.0e9])
    de = np.array([1.0e7, 0.0, 5.0e7])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de)

    # Check implicit energy update: e_new ~ e_old + de - 0.5 * dv * (p_old + p_new)
    assert np.allclose(e_new, e_old + de - 0.5 * dv * (p_old + p_new), rtol=1e-5)
    assert np.all(c2 >= 0.0)


# ============================================================================
# Starter Keyword Parsing Tests
# ============================================================================

def test_gruneisen_starter_parsing(tmp_path):
    """Verify /EOS/GRUNEISEN deck block is parsed correctly by read_eos."""
    d = StarterDeck("test_grun")
    d.raw_block("EOS/GRUNEISEN/10", [
        "Gruneisen Copper",
        "              3940.0               1.489                 0.0                 0.0",
        "                 2.0                0.47                 0.0              8930.0",
    ])
    path = tmp_path / "grun.rad"
    path.write_text(d.render())

    blocks = list(read_deck(str(path)))
    model = Model()
    log = MessageLog()
    read_eos(blocks[1], model, log)

    assert not log.errors
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 10
    assert eos.kind == "GRUNEISEN"
    assert eos.params["c"] == pytest.approx(3940.0)
    assert eos.params["s1"] == pytest.approx(1.489)
    assert eos.params["gamma0"] == pytest.approx(2.0)
    assert eos.params["a"] == pytest.approx(0.47)
    assert eos.params["rho0_card"] == pytest.approx(8930.0)


def test_tillotson_starter_parsing(tmp_path):
    """Verify /EOS/TILLOTSON deck block is parsed correctly by read_eos."""
    d = StarterDeck("test_till")
    d.raw_block("EOS/TILLOTSON/20", [
        "Tillotson Iron",
        "             1.30e11             1.00e11                 0.5                 1.5",
        "             1.00e10             2.00e10                 1.5                 0.0              7800.0",
        "                 5.0                 5.0",
    ])
    path = tmp_path / "till.rad"
    path.write_text(d.render())

    blocks = list(read_deck(str(path)))
    model = Model()
    log = MessageLog()
    read_eos(blocks[1], model, log)

    assert not log.errors
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 20
    assert eos.kind == "TILLOTSON"
    assert eos.params["c1"] == pytest.approx(1.30e11)
    assert eos.params["c2"] == pytest.approx(1.00e11)
    assert eos.params["a"] == pytest.approx(0.5)
    assert eos.params["b"] == pytest.approx(1.5)
    assert eos.params["er"] == pytest.approx(1.00e10)
    assert eos.params["es"] == pytest.approx(2.00e10)
    assert eos.params["vs"] == pytest.approx(1.5)
    assert eos.params["alpha"] == pytest.approx(5.0)
    assert eos.params["beta"] == pytest.approx(5.0)
    assert eos.params["rho0_card"] == pytest.approx(7800.0)


# ============================================================================
# Solid Element Coupling Tests
# ============================================================================

def test_gruneisen_solid_element_coupling():
    """Verify solid element stress update and combined bulk+shear sound speed."""
    eos = EquationOfState(
        kind="GRUNEISEN", rho0=8930.0,
        params={"c": 3940.0, "s1": 1.489, "s2": 0.0, "s3": 0.0,
                "gamma0": 2.0, "a": 0.47, "e0": 0.0, "rho0_card": 8930.0})
    mat = Material(id=1, law=1, rho0=8930.0, params={"E": 110e9, "nu": 0.35}, eos=eos)

    mu = np.array([0.05])
    dv = np.array([-0.02])
    e_eos = np.array([0.0])
    p_eos = np.array([0.0])
    de_dev = np.array([1.0e5])

    p_new, e_new, c2 = eos_mod.update(mat.eos, mu, dv, e_eos, p_eos, de_dev)
    assert p_new[0] > 0.0
    assert e_new[0] > 0.0
    # Combined sound speed: c_solid = sqrt(c2_bulk + (4G/3)/rho)
    G = mat.params["E"] / (2.0 * (1.0 + mat.params["nu"]))
    c_tot = np.sqrt(c2[0] + (4.0 * G / 3.0) / (mat.rho0 * (1.0 + mu[0])))
    assert c_tot > mat.eos.params["c"]


def test_tillotson_solid_element_coupling():
    """Verify solid element stress update and combined bulk+shear sound speed."""
    eos = EquationOfState(
        kind="TILLOTSON", rho0=7800.0,
        params={"c1": 1.3e11, "c2": 1.0e11, "a": 0.5, "b": 1.5,
                "er": 1.0e10, "es": 2.0e10, "vs": 1.5, "alpha": 5.0,
                "beta": 5.0, "e0": 0.0, "rho0_card": 7800.0})
    mat = Material(id=2, law=1, rho0=7800.0, params={"E": 210e9, "nu": 0.3}, eos=eos)

    mu = np.array([0.05])
    dv = np.array([-0.02])
    e_eos = np.array([0.0])
    p_eos = np.array([0.0])
    de_dev = np.array([1.0e5])

    p_new, e_new, c2 = eos_mod.update(mat.eos, mu, dv, e_eos, p_eos, de_dev)
    assert p_new[0] > 0.0
    assert e_new[0] > 0.0
    G = mat.params["E"] / (2.0 * (1.0 + mat.params["nu"]))
    c_tot = np.sqrt(c2[0] + (4.0 * G / 3.0) / (mat.rho0 * (1.0 + mu[0])))
    assert c_tot > np.sqrt(eos.params["c1"] / eos.rho0)
