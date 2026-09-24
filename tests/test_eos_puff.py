"""Unit tests for /EOS/PUFF equation of state.

Cites Fortran reference:
- common_source/eos/puff.F (PUFF subroutine)
- starter/source/materials/eos/hm_read_eos_puff.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import eos as eos_mod
from pyradioss.model.entities import EquationOfState


def test_puff_compression_regime():
    """Verify PUFF in compression regime (mu >= 0)."""
    rho0 = 2700.0
    c1 = 7.5e10
    c2 = 6.5e10
    c3 = 1.0e10
    gamma0 = 2.0
    e0 = 1.0e5

    eos = EquationOfState(
        kind="PUFF",
        rho0=rho0,
        params={
            "c1": c1, "c2": c2, "c3": c3,
            "gamma0": gamma0, "e0": e0,
            "es": 1.0e6, "h": 0.5
        }
    )

    mu = np.array([0.0, 0.1, 0.2])
    A, B = eos_mod.coefficients(eos, mu, e=e0)

    # Reference state: mu = 0 -> A = 0, B = gamma0
    assert pytest.approx(A[0], abs=1e-5) == 0.0
    assert pytest.approx(B[0], rel=1e-6) == gamma0

    # P at reference
    p0 = eos_mod.pressure(eos, 0.0, e0)
    assert pytest.approx(p0, rel=1e-6) == gamma0 * e0

    # mu = 0.1
    eta = 1.1
    xx = 0.1 / 1.1
    gx = 1.0 - 0.5 * gamma0 * xx
    aa_comp = ((c1 + c3 * 0.01) * 0.1 + c2 * 0.01) * gx
    expected_p = aa_comp + gamma0 * e0
    p_comp = eos_mod.pressure(eos, 0.1, e0)
    assert pytest.approx(p_comp, rel=1e-6) == expected_p


def test_puff_cold_expansion_regime():
    """Verify PUFF in cold expansion regime (mu < 0 and E < Es)."""
    rho0 = 2700.0
    c1 = 7.5e10
    t2 = 1.0e9
    gamma0 = 2.0
    es = 1.0e6
    e_cold = 2.0e5  # < es

    # Note: t1 omitted, should default to c1
    eos = EquationOfState(
        kind="PUFF",
        rho0=rho0,
        params={
            "c1": c1, "c2": 0.0, "c3": 0.0,
            "t2": t2, "gamma0": gamma0,
            "es": es, "h": 0.5, "e0": e_cold
        }
    )

    mu = -0.05
    A, B = eos_mod.coefficients(eos, np.array([mu]), e=e_cold)

    # T1 defaults to C1
    t1 = c1
    eta = 1.0 + mu
    xx = mu / eta
    gx = 1.0 - 0.5 * gamma0 * xx
    aa_cold = (t1 + t2 * mu) * mu * gx
    assert pytest.approx(A[0], rel=1e-6) == aa_cold
    assert pytest.approx(B[0], rel=1e-6) == gamma0

    p_cold = eos_mod.pressure(eos, mu, e_cold)
    assert pytest.approx(p_cold, rel=1e-6) == aa_cold + gamma0 * e_cold


def test_puff_hot_sublimation_regime():
    """Verify PUFF in hot expansion / sublimation regime (mu < 0 and E >= Es)."""
    rho0 = 2700.0
    c1 = 7.5e10
    gamma0 = 2.0
    h = 0.5
    es = 1.0e6
    e_hot = 2.0e6  # >= es

    eos = EquationOfState(
        kind="PUFF",
        rho0=rho0,
        params={
            "c1": c1, "gamma0": gamma0,
            "h": h, "es": es, "e0": e_hot
        }
    )

    mu = -0.1
    eta = 1.0 + mu
    ee = math.sqrt(eta)
    bb_hot = (h + (gamma0 - h) * ee) * eta
    xx = mu / eta
    cc = c1 / (gamma0 * es)
    expa = math.exp(cc * xx)
    aa_hot = bb_hot * es * (expa - 1.0)

    A, B = eos_mod.coefficients(eos, np.array([mu]), e=e_hot)
    assert pytest.approx(B[0], rel=1e-6) == bb_hot
    assert pytest.approx(A[0], rel=1e-6) == aa_hot

    p_hot = eos_mod.pressure(eos, mu, e_hot)
    assert pytest.approx(p_hot, rel=1e-6) == aa_hot + bb_hot * e_hot


def test_puff_sound_speed():
    """Verify sound speed in compression and cold tension."""
    rho0 = 2700.0
    c1 = 7.5e10
    gamma0 = 2.0
    e0 = 1.0e5

    eos = EquationOfState(
        kind="PUFF",
        rho0=rho0,
        params={"c1": c1, "gamma0": gamma0, "e0": e0}
    )

    # At mu = 0: DPDM = C1 + gamma0 * E0 = K0
    # c0 = sqrt((C1 + gamma0 * E0) / rho0)
    k0 = c1 + gamma0 * e0
    c0_expected = math.sqrt(k0 / rho0)
    c0 = eos_mod.sound_speed(eos, 0.0, e0)
    assert pytest.approx(c0, rel=1e-5) == c0_expected


def test_puff_update():
    """Verify 2-step predictor-corrector update."""
    rho0 = 2700.0
    eos = EquationOfState(
        kind="PUFF",
        rho0=rho0,
        params={
            "c1": 5.0e10, "c2": 1.0e10, "gamma0": 1.8,
            "es": 1.0e6, "h": 0.6, "e0": 1.0e5
        }
    )

    mu = np.array([0.02])
    dv = np.array([-0.01])
    e_old = np.array([1.0e5])
    p_old = np.array([float(eos_mod.pressure(eos, 0.0, 1.0e5))])
    de_other = np.array([0.0])

    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de_other)
    assert e_new[0] > e_old[0]
    assert p_new[0] > p_old[0]
    assert c2[0] > 0.0
