"""Unit tests for /EOS/IDEAL-GAS equation of state.

Cites Fortran reference:
- common_source/eos/idealgas.F (IDEALGAS subroutine)
- starter/source/materials/eos/hm_read_eos_ideal_gas.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck


def test_idealgas_coefficients():
    """Verify A(mu) = -Psh and B(mu) = (gamma - 1)*(1 + mu)."""
    eos = EquationOfState(
        kind="IDEAL-GAS",
        rho0=1.2,
        params={"gamma": 1.4, "p0": 1.013e5, "psh": 1000.0}
    )
    mu = np.array([-0.1, 0.0, 0.2])
    A, B = eos_mod.coefficients(eos, mu)

    # A should be constant -psh
    np.testing.assert_allclose(A, -1000.0)
    # B = (1.4 - 1.0) * (1 + mu) = 0.4 * [0.9, 1.0, 1.2]
    np.testing.assert_allclose(B, [0.36, 0.40, 0.48])


def test_idealgas_pressure_scalar_and_array():
    """Verify pressure evaluation P = (gamma - 1)*(1 + mu)*E - Psh."""
    gamma = 1.4
    rho0 = 1.2
    p0 = 101300.0
    e0 = p0 / (gamma - 1.0)  # 253250.0

    eos = EquationOfState(
        kind="IDEAL-GAS",
        rho0=rho0,
        params={"gamma": gamma, "p0": p0, "psh": 0.0}
    )

    # Reference state: mu = 0, E = E0 -> P = P0
    p_ref = eos_mod.pressure(eos, 0.0, e0)
    assert isinstance(p_ref, float)
    assert pytest.approx(p_ref, rel=1e-6) == p0

    # Compression mu = 0.5, same energy E0: P = 0.4 * 1.5 * E0 = 1.5 * P0
    p_comp = eos_mod.pressure(eos, 0.5, e0)
    assert pytest.approx(p_comp, rel=1e-6) == 1.5 * p0

    # Array evaluation
    mu_arr = np.array([-0.2, 0.0, 0.1, 0.5])
    e_arr = np.full_like(mu_arr, e0)
    p_arr = eos_mod.pressure(eos, mu_arr, e_arr)
    expected = (gamma - 1.0) * (1.0 + mu_arr) * e0
    np.testing.assert_allclose(p_arr, expected, rtol=1e-6)


def test_idealgas_sound_speed():
    """Verify sound speed c = sqrt(gamma * (P + Psh) / (rho0 * (1 + mu)))."""
    gamma = 1.4
    rho0 = 1.2
    p0 = 101300.0
    e0 = p0 / (gamma - 1.0)

    eos = EquationOfState(
        kind="IDEAL-GAS",
        rho0=rho0,
        params={"gamma": gamma, "p0": p0, "psh": 0.0}
    )

    # At reference state mu = 0: c0 = sqrt(gamma * p0 / rho0)
    c0_expected = math.sqrt(gamma * p0 / rho0)
    c0 = eos_mod.sound_speed(eos, 0.0, e0)
    assert pytest.approx(c0, rel=1e-6) == c0_expected

    # Compression mu = 0.2: rho = rho0 * (1 + mu) = 1.44
    # P = 0.4 * 1.2 * E0 = 1.2 * P0
    # c = sqrt(gamma * 1.2 * P0 / (rho0 * 1.2)) = sqrt(gamma * P0 / rho0) (isothermal with respect to mu*e)
    c_comp = eos_mod.sound_speed(eos, 0.2, e0)
    assert pytest.approx(c_comp, rel=1e-6) == c0_expected


def test_idealgas_initial_state():
    """Verify initial_state() calculates E0 = P0 / (gamma - 1) and Pinit = P0 - Psh."""
    gamma = 1.4
    p0 = 101300.0
    psh = 5000.0

    eos = EquationOfState(
        kind="IDEAL-GAS",
        rho0=1.2,
        params={"gamma": gamma, "p0": p0, "psh": psh}
    )
    e0, p_init = eos_mod.initial_state(eos)
    assert pytest.approx(e0, rel=1e-6) == p0 / (gamma - 1.0)
    assert pytest.approx(p_init, rel=1e-6) == p0 - psh


def test_idealgas_update_implicit():
    """Verify implicit Crank-Nicolson energy-pressure update."""
    gamma = 1.4
    rho0 = 1.2
    p0 = 101300.0
    e0 = p0 / (gamma - 1.0)

    eos = EquationOfState(
        kind="IDEAL-GAS",
        rho0=rho0,
        params={"gamma": gamma, "p0": p0, "psh": 0.0}
    )

    mu = np.array([0.05])
    dv = np.array([-0.02])  # compression
    e_old = np.array([e0])
    p_old = np.array([p0])
    de_other = np.array([0.0])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de_other)

    # Work done on gas: dE = -0.5 * dv * (P_old + P_new) > 0 for dv < 0
    assert e_new[0] > e0
    assert p_new[0] > p0
    # Check consistency: P_new = (gamma - 1) * (1 + mu) * E_new
    expected_p = (gamma - 1.0) * (1.0 + mu[0]) * e_new[0]
    assert pytest.approx(p_new[0], rel=1e-6) == expected_p
    # Sound speed squared c2 > 0
    assert c2[0] > 0.0
