"""Unit tests for /EOS/NASG (and aliases /EOS/NOBLE-ABEL-STIFFENED-GAS, /EOS/NOBLE_ABEL_STIFFENED_GAS) equation of state.

Cites Fortran reference:
- common_source/eos/nasg.F (NASG subroutine)
- starter/source/materials/eos/hm_read_eos_nasg.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState


def test_nasg_exact_formula_and_initial_state():
    """Verify exact formula:
    P = (gamma - 1)*(1 + mu)*(E - rho0*q) / (1 - b*rho0*(1 + mu)) - gamma*P_star - Psh.
    And initial_state:
    E0 = (P0 + gamma*P_star)*(1 - rho0*b) / (gamma - 1) + rho0*q.
    """
    rho0 = 1000.0
    b = 0.0005  # covolume
    gamma = 1.5
    p_star = 1.0e8
    q = -1.0e5
    p0 = 1.0e5
    psh = 2500.0

    eos = EquationOfState(
        kind="NASG",
        rho0=rho0,
        params={
            "b": b, "gamma": gamma, "p_star": p_star,
            "q": q, "p0": p0, "psh": psh
        }
    )

    e0, p_init = eos_mod.initial_state(eos)
    assert pytest.approx(p_init, rel=1e-6) == p0 - psh
    expected_e0 = (p0 + gamma * p_star) * (1.0 - rho0 * b) / (gamma - 1.0) + rho0 * q
    assert pytest.approx(e0, rel=1e-6) == expected_e0

    # At mu = 0 and e0, P should be exactly p0 - psh
    p_eval = eos_mod.pressure(eos, 0.0, e0)
    assert pytest.approx(p_eval, rel=1e-6) == p0 - psh

    # At mu = 0.05
    mu = 0.05
    eta = 1.0 + mu
    denom = 1.0 - b * rho0 * eta
    num = e0 - rho0 * q
    expected_p = (gamma - 1.0) * eta * num / denom - gamma * p_star - psh
    p_comp = eos_mod.pressure(eos, mu, e0)
    assert pytest.approx(p_comp, rel=1e-6) == expected_p


def test_nasg_aliases_compatibility():
    """Verify NASG, NOBLE-ABEL-STIFFENED-GAS, and NOBLE_ABEL_STIFFENED_GAS aliases."""
    rho0 = 1000.0
    params = {
        "b": 0.0004, "gamma": 1.6, "p_star": 5.0e7,
        "q": -5.0e4, "p0": 1.0e5, "psh": 0.0
    }
    mu = np.array([-0.02, 0.0, 0.04])
    e = np.array([2.0e7, 2.0e7, 2.0e7])

    aliases = ["NASG", "NOBLE-ABEL-STIFFENED-GAS", "NOBLE_ABEL_STIFFENED_GAS"]
    ref_p = None
    for alias in aliases:
        eos = EquationOfState(kind=alias, rho0=rho0, params=params)
        p = eos_mod.pressure(eos, mu, e)
        if ref_p is None:
            ref_p = p
        else:
            np.testing.assert_allclose(p, ref_p)


def test_nasg_sound_speed():
    """Verify total derivative DPDM and sound speed."""
    rho0 = 1000.0
    b = 0.0005
    gamma = 1.5
    p_star = 1.0e8
    q = -1.0e5
    p0 = 1.0e5

    eos = EquationOfState(
        kind="NASG",
        rho0=rho0,
        params={"b": b, "gamma": gamma, "p_star": p_star, "q": q, "p0": p0}
    )

    e0, _ = eos_mod.initial_state(eos)
    # At mu = 0: eta = 1, denom = 1 - b*rho0
    denom = 1.0 - b * rho0
    num = e0 - rho0 * q
    dpde = (gamma - 1.0) / denom
    dpdm = (gamma - 1.0) * num / (denom ** 2) + dpde * p0
    c0_expected = math.sqrt(dpdm / rho0)

    c0 = eos_mod.sound_speed(eos, 0.0, e0)
    assert pytest.approx(c0, rel=1e-5) == c0_expected


def test_nasg_implicit_update():
    """Verify Crank-Nicolson implicit update."""
    rho0 = 1000.0
    eos = EquationOfState(
        kind="NASG",
        rho0=rho0,
        params={"b": 0.0005, "gamma": 1.5, "p_star": 1.0e8, "q": -1.0e5, "p0": 1.0e5}
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
