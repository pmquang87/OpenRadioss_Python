"""Unit tests for LAW35: Kelvin-Voigt / Kelvin-Maxwell Viscoelastic Model with Softening.

Tests:
1. Rate-dependent viscoelastic stress: sigma = E * eps + eta * deps/dt.
2. Zero rate limit (Hooke's elastic response).
3. Optional maximum strain softening when eps > eps_max.
4. Analytical stress relaxation under constant strain.
5. 3D solid continuum update (solid_update) rate sensitivity matching sigeps35.F.
6. 3D solid continuum update with max strain softening.
7. Consistent solid tangent matrix.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law35_kelvinmax
from pyradioss.model.entities import Material


def test_kelvin_voigt_rate_dependence():
    """Verify rate-dependent viscoelastic stress: sigma = E * eps + eta * deps/dt."""
    E = 100.0   # MPa
    eta = 10.0  # MPa*s
    eps = 0.05  # 5% strain

    # Case 1: Quasi-static (deps_dt = 0) -> Pure elastic Hookean response
    sig_static = law35_kelvinmax.kelvin_voigt_stress(eps, 0.0, E, eta)
    assert pytest.approx(sig_static) == E * eps

    # Case 2: Dynamic strain rate
    deps_dt = 2.0  # 2.0 s^-1
    sig_dynamic = law35_kelvinmax.kelvin_voigt_stress(eps, deps_dt, E, eta)
    expected = E * eps + eta * deps_dt  # 5.0 + 20.0 = 25.0
    assert pytest.approx(sig_dynamic) == expected

    # Case 3: Rate sweep verifies linear dependence on strain rate
    rates = [0.1, 1.0, 5.0, 10.0]
    for r in rates:
        s = law35_kelvinmax.kelvin_voigt_stress(eps, r, E, eta)
        assert pytest.approx(s) == E * eps + eta * r


def test_kelvin_voigt_max_strain_softening():
    """Verify optional maximum strain softening when strain exceeds eps_max."""
    E = 100.0
    eta = 5.0
    rate = 1.0
    eps_max = 0.10      # Softening threshold at 10% strain
    softening = 5.0     # Softening rate

    # Below threshold: no softening
    eps_below = 0.08
    sig_below = law35_kelvinmax.kelvin_voigt_stress(
        eps_below, rate, E, eta, eps_max=eps_max, softening=softening
    )
    assert pytest.approx(sig_below) == E * eps_below + eta * rate

    # Above threshold: softened
    eps_above = 0.15
    sig_above = law35_kelvinmax.kelvin_voigt_stress(
        eps_above, rate, E, eta, eps_max=eps_max, softening=softening
    )
    unsoftened = E * eps_above + eta * rate
    soft_factor = 1.0 - softening * (eps_above - eps_max)  # 1.0 - 5.0*(0.05) = 0.75
    assert pytest.approx(sig_above) == unsoftened * soft_factor
    assert sig_above < unsoftened


def test_kelvin_maxwell_relaxation():
    """Verify analytical stress relaxation under constant strain eps0."""
    E = 50.0
    Et = 10.0
    mu = 25.0
    eps0 = 0.02
    tau = mu / E  # 0.5 s

    # At t = 0: sigma(0) = E * eps0
    s0 = law35_kelvinmax.kelvin_maxwell_relaxation(0.0, E, Et, mu, eps0)
    assert pytest.approx(s0) == E * eps0

    # At t = tau: sigma(tau) = sigma_inf + (sigma_0 - sigma_inf) * exp(-1)
    s_tau = law35_kelvinmax.kelvin_maxwell_relaxation(tau, E, Et, mu, eps0)
    sig_inf = Et * eps0
    expected_tau = sig_inf + (E * eps0 - sig_inf) * math.exp(-1.0)
    assert pytest.approx(s_tau) == expected_tau

    # At t -> infinity: sigma -> sigma_inf = Et * eps0
    s_inf = law35_kelvinmax.kelvin_maxwell_relaxation(100.0 * tau, E, Et, mu, eps0)
    assert pytest.approx(s_inf) == Et * eps0


def test_solid_update_viscoelastic_rate_dependence():
    """Verify that 3D solid_update exhibits higher stress at higher strain rates."""
    mat = law35_kelvinmax.build_law35({
        "MAT_E": 20.0,
        "MAT_NU": 0.25,
        "MAT_ETAN": 4.0,
        "MAT_NUt": 0.25,
        "MAT_ETA2": 10.0,
        "rho": 1.0e-3,
    })

    deps = np.array([[0.01, -0.0025, -0.0025, 0.0, 0.0, 0.0]])

    # Slow loading: dt = 1.0 s -> strain rate = 0.01 s^-1
    sig_slow = np.zeros((1, 6))
    extra_slow = {
        "eps35": np.zeros((1, 6)),
        "sigair35": np.zeros(1),
        "edot35": np.zeros(1),
        "rho": np.full(1, mat.rho0),
    }
    sig_slow, c_slow = law35_kelvinmax.solid_update(mat, sig_slow, deps, dt=1.0, extra=extra_slow)

    # Fast loading: dt = 1.0e-3 s -> strain rate = 10.0 s^-1
    sig_fast = np.zeros((1, 6))
    extra_fast = {
        "eps35": np.zeros((1, 6)),
        "sigair35": np.zeros(1),
        "edot35": np.zeros(1),
        "rho": np.full(1, mat.rho0),
    }
    sig_fast, c_fast = law35_kelvinmax.solid_update(mat, sig_fast, deps, dt=1.0e-3, extra=extra_fast)

    # Stress under fast rate must be significantly higher than slow rate due to viscous dashpot
    assert sig_fast[0, 0] > sig_slow[0, 0]
    assert c_fast[0] > 0.0
    assert c_slow[0] > 0.0


def test_solid_update_max_strain_softening():
    """Verify that 3D solid_update reduces stress when strain exceeds eps_max."""
    # Mat without softening
    mat_unsoftened = law35_kelvinmax.build_law35({
        "MAT_E": 30.0,
        "MAT_NU": 0.2,
        "MAT_ETAN": 5.0,
        "MAT_NUt": 0.2,
        "MAT_ETA2": 8.0,
        "rho": 1.0e-3,
        "eps_max": 0.0,  # Disabled
    })

    # Mat with softening
    mat_softened = law35_kelvinmax.build_law35({
        "MAT_E": 30.0,
        "MAT_NU": 0.2,
        "MAT_ETAN": 5.0,
        "MAT_NUt": 0.2,
        "MAT_ETA2": 8.0,
        "rho": 1.0e-3,
        "eps_max": 0.02,     # Softening threshold at 2% strain
        "softening": 15.0,   # Softening rate
    })

    deps_large = np.array([[0.10, -0.02, -0.02, 0.0, 0.0, 0.0]])

    sig_u = np.zeros((1, 6))
    sig_u, _ = law35_kelvinmax.solid_update(mat_unsoftened, sig_u, deps_large, dt=0.01)

    sig_s = np.zeros((1, 6))
    sig_s, _ = law35_kelvinmax.solid_update(mat_softened, sig_s, deps_large, dt=0.01)

    # Softened material must have lower stress at large strain
    assert sig_s[0, 0] < sig_u[0, 0]


def test_consistent_solid_tangent_structure():
    """Verify consistent solid tangent operator shape and elastic symmetry."""
    mat = law35_kelvinmax.build_law35({
        "MAT_E": 25.0,
        "MAT_NU": 0.3,
        "MAT_ETAN": 5.0,
        "MAT_NUt": 0.3,
        "MAT_ETA2": 6.0,
        "rho": 1.1e-3,
    })

    sig = np.zeros((2, 6))
    C = law35_kelvinmax.consistent_solid_tangent(mat, sig, dt=1.0e-3)
    assert C.shape == (2, 6, 6)

    # Check symmetry C_ijkl = C_jikl
    for el in range(2):
        assert pytest.approx(C[el, 0, 1]) == C[el, 1, 0]
        assert pytest.approx(C[el, 0, 2]) == C[el, 2, 0]
        assert pytest.approx(C[el, 1, 2]) == C[el, 2, 1]
        assert C[el, 3, 3] > 0.0
        assert C[el, 4, 4] > 0.0
        assert C[el, 5, 5] > 0.0
