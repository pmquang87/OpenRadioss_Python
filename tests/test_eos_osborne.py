"""Unit tests for /EOS/OSBORNE (and alias /EOS/OSBORN) equation of state.

Cites Fortran reference:
- common_source/eos/osborne.F (OSBORNE subroutine)
- starter/source/materials/eos/hm_read_eos_osborne.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState


def test_osborne_rational_formula_compression_and_tension():
    """Verify exact rational form:
    P = [a1*mu + a2*sgn(mu)*mu^2 + (b0 + b1*mu + b2*mu^2)*E + (c0 + c1*mu)*E^2] / (E + d0) - Psh.
    """
    rho0 = 2000.0
    a1 = 2.0e9
    a2 = 1.0e9
    b0 = 1.5
    b1 = 0.5
    b2 = 0.2
    c0 = 1.0e-4
    c1 = 5.0e-5
    d0 = 1.0e6
    psh = 2000.0
    e_val = 5.0e5

    eos = EquationOfState(
        kind="OSBORNE",
        rho0=rho0,
        params={
            "a1": a1, "a2": a2, "b0": b0, "b1": b1, "b2": b2,
            "c0": c0, "c1": c1, "d0": d0, "psh": psh, "e0": e_val
        }
    )

    # 1. Compression mu = 0.1 (a2* = +a2)
    mu_pos = 0.1
    num_pos = (a1 * mu_pos + a2 * (mu_pos ** 2)
               + (b0 + b1 * mu_pos + b2 * (mu_pos ** 2)) * e_val
               + (c0 + c1 * mu_pos) * (e_val ** 2))
    denom_pos = e_val + d0
    expected_p_pos = num_pos / denom_pos - psh
    p_pos = eos_mod.pressure(eos, mu_pos, e_val)
    assert pytest.approx(p_pos, rel=1e-6) == expected_p_pos

    # 2. Tension mu = -0.05 (a2* = -a2)
    mu_neg = -0.05
    num_neg = (a1 * mu_neg - a2 * (mu_neg ** 2)
               + (b0 + b1 * mu_neg + b2 * (mu_neg ** 2)) * e_val
               + (c0 + c1 * mu_neg) * (e_val ** 2))
    denom_neg = e_val + d0
    expected_p_neg = num_neg / denom_neg - psh
    p_neg = eos_mod.pressure(eos, mu_neg, e_val)
    assert pytest.approx(p_neg, rel=1e-6) == expected_p_neg


def test_osborn_alias_compatibility():
    """Verify /EOS/OSBORN alias produces identical results to /EOS/OSBORNE."""
    rho0 = 2000.0
    params = {
        "a1": 1.5e9, "a2": 8.0e8, "b0": 1.2, "b1": 0.3, "b2": 0.0,
        "c0": 2.0e-4, "c1": 0.0, "d0": 5.0e5, "p0": 1.0e5, "psh": 0.0
    }
    eos_long = EquationOfState(kind="OSBORNE", rho0=rho0, params=params)
    eos_short = EquationOfState(kind="OSBORN", rho0=rho0, params=params)

    mu = np.array([-0.05, 0.0, 0.05])
    e = np.array([2.0e5, 2.0e5, 2.0e5])

    p_long = eos_mod.pressure(eos_long, mu, e)
    p_short = eos_mod.pressure(eos_short, mu, e)
    np.testing.assert_allclose(p_long, p_short)

    A_long, B_long = eos_mod.coefficients(eos_long, mu, e)
    A_short, B_short = eos_mod.coefficients(eos_short, mu, e)
    np.testing.assert_allclose(A_long, A_short)
    np.testing.assert_allclose(B_long, B_short)


def test_osborne_initial_state_quadratic_root():
    """Verify initial_state solves C0*E0^2 + (B0 - P0)*E0 - P0*D0 = 0."""
    rho0 = 2000.0
    b0 = 1.5
    c0 = 1.0e-4
    d0 = 1.0e6
    p0 = 1.0e5

    eos = EquationOfState(
        kind="OSBORNE",
        rho0=rho0,
        params={"b0": b0, "c0": c0, "d0": d0, "p0": p0, "psh": 0.0}
    )

    e0, p_init = eos_mod.initial_state(eos)
    assert pytest.approx(p_init, rel=1e-6) == p0

    # Verify that at (mu=0, E0), P evaluates to P0
    p_eval = eos_mod.pressure(eos, 0.0, e0)
    assert pytest.approx(p_eval, rel=1e-6) == p0


def test_osborne_update_predictor_corrector():
    """Verify 2-step predictor-corrector update and total derivative sound speed."""
    rho0 = 2000.0
    eos = EquationOfState(
        kind="OSBORNE",
        rho0=rho0,
        params={
            "a1": 2.0e9, "a2": 1.0e9, "b0": 1.5, "b1": 0.5, "b2": 0.0,
            "c0": 1.0e-4, "c1": 0.0, "d0": 1.0e6, "p0": 1.0e5, "psh": 0.0
        }
    )

    e0, p0 = eos_mod.initial_state(eos)
    mu = np.array([0.02])
    dv = np.array([-0.01])
    e_old = np.array([e0])
    p_old = np.array([p0])
    de_other = np.array([0.0])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de_other)

    assert e_new[0] > e0
    assert p_new[0] > p0
    assert c2[0] > 0.0
