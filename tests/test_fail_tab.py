"""Unit tests for classic /FAIL/TAB ported from fail_tab_old_s.F and fail_tab_old_c.F."""

from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.failure import tab, shell_step, solid_step
from pyradioss.model.entities import FailureModel


def test_tab_default_constant_fallback():
    """Verify fallback when no table is provided (constant threshold)."""
    fm = FailureModel(
        type="TAB",
        params={"eps_f": 0.2, "dcrit": 1.0, "n": 1.0, "d": 0.999},
        ifail_sh=1,
    )
    dama = np.zeros(1)
    sig = np.array([[200e6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_epsp = np.array([0.05])

    broken = solid_step(fm, sig, d_epsp, np.zeros((1, 6)), 1e-6, dama)
    assert not broken[0]
    # D += 0.05 / 0.2 = 0.25
    assert np.isclose(dama[0], 0.25, atol=1e-5)

    # Accumulate until rupture
    solid_step(fm, sig, np.array([0.15]), np.zeros((1, 6)), 1e-6, dama)
    assert dama[0] == pytest.approx(1.0)


def test_tab_triaxiality_table_interpolation():
    """Verify failure strain interpolation from curve vs triaxiality."""
    class DummyCurve:
        # Triaxiality points and corresponding failure strain
        x = np.array([-0.5, 0.0, 0.33333333, 0.66666667, 1.0])
        y = np.array([0.5, 0.3, 0.2, 0.1, 0.05])

    fm = FailureModel(
        type="TAB",
        params={
            "table": DummyCurve(),
            "dcrit": 1.0,
            "fscale": 1.0,
            "n": 1.0,
        },
        ifail_sh=1,
    )

    # 1. Uniaxial tension in 3D: sig = [300, 0, 0, 0, 0, 0] MPa -> P = 100 MPa, SVM = 300 MPa -> eta = 1/3
    # At eta = 1/3, eps_f = 0.2
    sig_tens = np.array([[300e6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dama = np.zeros(1)
    solid_step(fm, sig_tens, np.array([0.04]), np.zeros((1, 6)), 1e-6, dama)
    # dD = 0.04 / 0.2 = 0.2
    assert np.isclose(dama[0], 0.2, atol=1e-3)

    # 2. Biaxial tension in 3D: sig = [300, 300, 0, 0, 0, 0] MPa -> P = 200 MPa, SVM = 300 MPa -> eta = 2/3
    # At eta = 2/3, eps_f = 0.1
    sig_biax = np.array([[300e6, 300e6, 0.0, 0.0, 0.0, 0.0]])
    dama_biax = np.zeros(1)
    solid_step(fm, sig_biax, np.array([0.04]), np.zeros((1, 6)), 1e-6, dama_biax)
    # dD = 0.04 / 0.1 = 0.4
    assert np.isclose(dama_biax[0], 0.4, atol=1e-3)


def test_tab_shell_plane_stress():
    """Verify shell 2D plane-stress triaxiality and damage advance."""
    class DummyCurve:
        x = np.array([-0.333, 0.0, 0.333, 0.667])
        y = np.array([0.4, 0.25, 0.15, 0.08])

    fm = FailureModel(
        type="TAB",
        params={"table": DummyCurve(), "dcrit": 1.0, "n": 1.0},
        ifail_sh=1,
    )

    # Pure shear: sxx = 0, syy = 0, sxy = 150 MPa -> P = 0, eta = 0.0 -> eps_f = 0.25
    sig_shear = np.array([[0.0, 0.0, 150e6]])
    dama = np.zeros(1)
    shell_step(fm, sig_shear, np.array([0.05]), np.zeros((1, 3)), 1e-6, dama)
    # dD = 0.05 / 0.25 = 0.2
    assert np.isclose(dama[0], 0.2, atol=1e-3)


def test_tab_dp_nonlinear_evolution():
    """Verify DP = DN * DD ** (1 - 1/DN) accumulation factor."""
    # Test with n = 2.0, d = 0.25 -> 1 - 1/n = 0.5 -> DP = 2.0 * sqrt(0.25) = 1.0
    dp = tab._compute_dp(0.25, 2.0)
    assert np.isclose(dp, 1.0, atol=1e-6)

    # Test with n = 0.5, d = 0.0 -> guard against division by zero
    dp_zero = tab._compute_dp(0.0, 0.5)
    assert dp_zero == 1.0

    # Test with dn = 1.0 -> DP = 1.0
    assert tab._compute_dp(0.5, 1.0) == 1.0


def test_tab_vectorized_multi_element():
    """Verify multi-element vectorized execution."""
    fm = FailureModel(
        type="TAB",
        params={"eps_f": 0.1, "dcrit": 1.0, "n": 1.0},
        ifail_sh=1,
    )
    nel = 4
    sig = np.zeros((nel, 6))
    sig[:, 0] = [100e6, 200e6, 300e6, 400e6]
    d_epsp = np.array([0.02, 0.05, 0.08, 0.12])
    dama = np.zeros(nel)

    broken = solid_step(fm, sig, d_epsp, np.zeros((nel, 6)), 1e-6, dama)
    assert not broken[0]
    assert not broken[1]
    assert not broken[2]
    assert broken[3]  # 0.12 >= 0.10 -> broken
    assert dama[3] == 1.0
