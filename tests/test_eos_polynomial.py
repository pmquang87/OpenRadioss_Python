"""Unit tests for /EOS/POLYNOMIAL (and alias /EOS/POLY) equation of state.

Cites Fortran reference:
- common_source/eos/eospolyno.F (EOSPOLYNO subroutine)
- starter/source/materials/eos/hm_read_eos_polynomial.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState


def test_polynomial_coefficients_c0_to_c6():
    """Verify exact polynomial form:
    mu2 = mu * max(0, mu)
    A = (C0 - Psh) + (C1 + C3*mu^2)*mu + C2*mu2
    B = C4 + C5*mu + C6*mu2
    """
    c0 = 1.0e5
    c1 = 2.0e9
    c2 = 1.5e9
    c3 = 5.0e8
    c4 = 0.4
    c5 = 0.2
    c6 = 0.1
    psh = 2000.0

    eos = EquationOfState(
        kind="POLYNOMIAL",
        rho0=1000.0,
        params={
            "c0": c0, "c1": c1, "c2": c2, "c3": c3,
            "c4": c4, "c5": c5, "c6": c6, "psh": psh
        }
    )

    # 1. Tension: mu = -0.1 -> mu2 = 0
    mu_neg = np.array([-0.1])
    A_neg, B_neg = eos_mod.coefficients(eos, mu_neg)
    expected_A_neg = (c0 - psh) + (c1 + c3 * 0.01) * (-0.1)
    expected_B_neg = c4 + c5 * (-0.1)
    assert pytest.approx(A_neg[0], rel=1e-6) == expected_A_neg
    assert pytest.approx(B_neg[0], rel=1e-6) == expected_B_neg

    # 2. Compression: mu = 0.1 -> mu2 = 0.01
    mu_pos = np.array([0.1])
    A_pos, B_pos = eos_mod.coefficients(eos, mu_pos)
    expected_A_pos = (c0 - psh) + (c1 + c3 * 0.01) * 0.1 + c2 * 0.01
    expected_B_pos = c4 + c5 * 0.1 + c6 * 0.01
    assert pytest.approx(A_pos[0], rel=1e-6) == expected_A_pos
    assert pytest.approx(B_pos[0], rel=1e-6) == expected_B_pos


def test_polynomial_pressure_evaluation():
    """Verify pressure P = A + B*E in tension and compression."""
    eos = EquationOfState(
        kind="POLYNOMIAL",
        rho0=1000.0,
        params={
            "c0": 0.0, "c1": 2.2e9, "c2": 1.0e9, "c3": 0.0,
            "c4": 0.4, "c5": 0.1, "c6": 0.05, "psh": 0.0
        }
    )
    e_val = 1.0e5

    # Scalar evaluation at mu = 0.05
    mu = 0.05
    p_val = eos_mod.pressure(eos, mu, e_val)
    A_exp = 2.2e9 * mu + 1.0e9 * (mu ** 2)
    B_exp = 0.4 + 0.1 * mu + 0.05 * (mu ** 2)
    assert pytest.approx(p_val, rel=1e-6) == A_exp + B_exp * e_val


def test_polynomial_sound_speed_and_derivatives():
    """Verify DPDM derivative and bulk sound speed:
    DPDM = C1 + 2*C2*mu_pos + 3*C3*mu^2 + (C5 + C6*mu_pos)*E + B*DF^2*(P + Psh)
    """
    rho0 = 1000.0
    c1 = 2.0e9
    c2 = 1.0e9
    c3 = 5.0e8
    c4 = 0.4
    c5 = 0.1
    c6 = 0.05
    e0 = 1.0e5

    eos = EquationOfState(
        kind="POLYNOMIAL",
        rho0=rho0,
        params={
            "c0": 0.0, "c1": c1, "c2": c2, "c3": c3,
            "c4": c4, "c5": c5, "c6": c6, "e0": e0
        }
    )

    # Reference state: mu = 0, P = C4 * E0 = 0.4 * 1.0e5 = 4.0e4
    p_ref = float(eos_mod.pressure(eos, 0.0, e0))
    dpdm_ref = c1 + c5 * e0 + c4 * p_ref
    c0_expected = math.sqrt(dpdm_ref / rho0)

    c0 = eos_mod.sound_speed(eos, 0.0, e0)
    assert pytest.approx(c0, rel=1e-5) == c0_expected


def test_polynomial_initial_state():
    """Verify initial_state() calculates P0 = (C0 - Psh) + C4*E0."""
    c0 = 1.0e5
    c4 = 0.4
    e0 = 2.0e5
    psh = 5000.0

    eos = EquationOfState(
        kind="POLYNOMIAL",
        rho0=1000.0,
        params={"c0": c0, "c4": c4, "e0": e0, "psh": psh}
    )

    e_init, p_init = eos_mod.initial_state(eos)
    assert pytest.approx(e_init, rel=1e-6) == e0
    assert pytest.approx(p_init, rel=1e-6) == (c0 - psh) + c4 * e0


def test_poly_alias_compatibility():
    """Verify /EOS/POLY alias works identically to /EOS/POLYNOMIAL."""
    params = {
        "c0": 0.0, "c1": 2.2e9, "c2": 0.0, "c3": 0.0,
        "c4": 0.4, "c5": 0.0, "c6": 0.0, "e0": 1.0e5
    }
    eos1 = EquationOfState(kind="POLYNOMIAL", rho0=1000.0, params=params)
    eos2 = EquationOfState(kind="POLY", rho0=1000.0, params=params)

    mu = np.array([-0.05, 0.0, 0.05])
    e = np.array([1.0e5, 1.0e5, 1.0e5])
    np.testing.assert_allclose(eos_mod.pressure(eos1, mu, e), eos_mod.pressure(eos2, mu, e))
    A1, B1 = eos_mod.coefficients(eos1, mu, e)
    A2, B2 = eos_mod.coefficients(eos2, mu, e)
    np.testing.assert_allclose(A1, A2)
    np.testing.assert_allclose(B1, B2)
