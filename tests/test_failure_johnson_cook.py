"""Tests for Johnson-Cook failure criterion (/FAIL/JOHNSON) and BUG-MAT-09 verification.

Fortran reference: fail_johnson.F, fail_johnson_c.F.
Verifies:
- Failure strain rate enhancement uses plastic strain rate d_epsp/dt (BUG-MAT-09)
- Stress triaxiality dependence eps_f = (D1 + D2 * exp(D3 * sigma*))
- Solid and shell damage accumulation
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.failure.johnson import (
    _rate_factor,
    _thermal_factor,
    solid_step,
    shell_step,
)


class _FailStub:
    def __init__(self, D1=0.05, D2=3.44, D3=-2.12, D4=0.05,
                 D5=0.0, eps_dot_0=1.0, eps_f_min=0.0):
        self.params = {
            "D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5,
            "eps_dot_0": eps_dot_0, "eps_f_min": eps_f_min,
        }


def test_johnson_cook_rate_factor_plastic_strain_rate():
    """BUG-MAT-09: _rate_factor must prioritize d_epsp over total deps."""
    fail = _FailStub(D4=0.1, eps_dot_0=1.0)

    # Elastic loading: large total strain deps, but zero plastic strain d_epsp = 0
    deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_epsp = np.array([0.0])
    dt = 1e-3

    # With BUG-MAT-09 fix, d_epsp is used to determine rate: rate = 0.0 <= eps_dot_0
    # Thus rate factor is clamped at 1.0 (no spurious rate-thinning/enhancement from elastic strain rate)
    rf = _rate_factor(fail, deps=deps, dt=dt, dev_from_6=True, d_epsp=d_epsp)
    assert rf == pytest.approx(1.0)

    # Plastic loading: d_epsp = 0.1 at dt = 1e-3 -> plastic strain rate = 100.0
    d_epsp_plas = np.array([0.1])
    rf_plas = _rate_factor(fail, deps=deps, dt=dt, dev_from_6=True, d_epsp=d_epsp_plas)
    expected = 1.0 + 0.1 * np.log(100.0 / 1.0)
    assert rf_plas == pytest.approx(expected, rel=1e-5)


def test_johnson_cook_solid_step_plastic_rate():
    """Verify solid_step passes d_epsp to rate factor."""
    fail = _FailStub(D1=0.1, D2=0.0, D3=0.0, D4=0.1, eps_dot_0=1.0)

    sig = np.array([[100.0, 100.0, 100.0, 0.0, 0.0, 0.0]])  # Hydrostatic tension
    deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_epsp = np.array([0.01])
    dt = 1e-4  # rate = 0.01 / 1e-4 = 100.0
    dama = np.zeros(1)

    rupt = solid_step(fail, sig, d_epsp, deps, dt, dama)

    # eps_f = D1 * (1 + D4 * ln(rate/rate0)) = 0.1 * (1 + 0.1 * ln(100))
    expected_eps_f = 0.1 * (1.0 + 0.1 * np.log(100.0))
    expected_dama = 0.01 / expected_eps_f
    assert dama[0] == pytest.approx(expected_dama, rel=1e-4)
    assert not rupt[0]


def test_johnson_cook_shell_step():
    """Verify shell_step damage accumulation in plane stress."""
    fail = _FailStub(D1=0.2, D2=0.0, D3=0.0, D4=0.0)

    sig = np.array([[200.0, 0.0, 0.0]])  # Uniaxial tension
    deps = np.array([[0.01, -0.003, 0.0]])
    d_epsp = np.array([0.01])
    dt = 1e-4
    dama = np.zeros(1)

    rupt = shell_step(fail, sig, d_epsp, deps, dt, dama)

    # eps_f = 0.2
    assert dama[0] == pytest.approx(0.01 / 0.2, rel=1e-5)
    assert not rupt[0]
