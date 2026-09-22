"""Unit tests for /EOS/STIFFGAS (and aliases /EOS/STIFF-GAS, /EOS/STIFF_GAS, /EOS/STIFFENED_GAS, /EOS/SG) equation of state.

Cites Fortran reference:
- common_source/eos/stiffgas.F (STIFFGAS subroutine)
- starter/source/materials/eos/hm_read_eos_stiffened_gas.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState


def test_stiffgas_exact_formula():
    """Verify exact formula:
    P = -gamma*P_star - Psh + (gamma - 1)*(1 + mu)*E.
    """
    rho0 = 1000.0
    gamma = 4.4
    p_star = 6.0e8
    p0 = 1.0e5
    psh = 2000.0

    eos = EquationOfState(
        kind="STIFFGAS",
        rho0=rho0,
        params={"gamma": gamma, "p_star": p_star, "p0": p0, "psh": psh}
    )

    e0, p_init = eos_mod.initial_state(eos)
    assert pytest.approx(p_init, rel=1e-6) == p0 - psh
    expected_e0 = (p0 + gamma * p_star) / (gamma - 1.0)
    assert pytest.approx(e0, rel=1e-6) == expected_e0

    # At mu = 0 and e0: P = -gamma*p_star - psh + (gamma - 1)*e0 = p0 - psh
    p_eval = eos_mod.pressure(eos, 0.0, e0)
    assert pytest.approx(p_eval, rel=1e-6) == p0 - psh

    # At mu = 0.1
    p_comp = eos_mod.pressure(eos, 0.1, e0)
    expected_p_comp = -gamma * p_star - psh + (gamma - 1.0) * 1.1 * e0
    assert pytest.approx(p_comp, rel=1e-6) == expected_p_comp


def test_stiffgas_aliases_compatibility():
    """Verify aliases STIFFGAS, STIFF-GAS, STIFF_GAS, STIFFENED_GAS, SG."""
    rho0 = 1000.0
    params = {"gamma": 3.0, "p_star": 5.0e8, "p0": 1.0e5, "psh": 0.0}
    mu = np.array([-0.02, 0.0, 0.05])
    e = np.array([2.0e8, 2.0e8, 2.0e8])

    aliases = ["STIFFGAS", "STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS", "SG"]
    ref_press = None
    for alias in aliases:
        eos = EquationOfState(kind=alias, rho0=rho0, params=params)
        p = eos_mod.pressure(eos, mu, e)
        if ref_press is None:
            ref_press = p
        else:
            np.testing.assert_allclose(p, ref_press)


def test_stiffgas_sound_speed():
    """Verify total derivative DPDM and sound speed:
    DPDM = (gamma - 1)*E + (gamma - 1)/(1 + mu)*(P + Psh)
    """
    rho0 = 1000.0
    gamma = 4.4
    p_star = 6.0e8
    p0 = 1.0e5

    eos = EquationOfState(
        kind="STIFF-GAS",
        rho0=rho0,
        params={"gamma": gamma, "p_star": p_star, "p0": p0, "psh": 0.0}
    )

    e0, _ = eos_mod.initial_state(eos)
    # At mu = 0: P = P0
    # DPDM = (gamma - 1)*e0 + (gamma - 1)*p0 = (p0 + gamma*p_star) + (gamma - 1)*p0 = gamma*(p0 + p_star)
    dpdm_expected = gamma * (p0 + p_star)
    c0_expected = math.sqrt(dpdm_expected / rho0)

    c0 = eos_mod.sound_speed(eos, 0.0, e0)
    assert pytest.approx(c0, rel=1e-5) == c0_expected


def test_stiffgas_implicit_update():
    """Verify Crank-Nicolson implicit update."""
    rho0 = 1000.0
    eos = EquationOfState(
        kind="STIFF-GAS",
        rho0=rho0,
        params={"gamma": 4.4, "p_star": 6.0e8, "p0": 1.0e5, "psh": 0.0}
    )

    e0, p0 = eos_mod.initial_state(eos)
    mu = np.array([0.01])
    dv = np.array([-0.005])  # compression
    e_old = np.array([e0])
    p_old = np.array([p0])
    de_other = np.array([0.0])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de_other)
    assert e_new[0] > e0
    assert p_new[0] > p0
    assert c2[0] > 0.0
