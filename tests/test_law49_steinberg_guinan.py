"""Unit tests for LAW49 (Steinberg-Guinan shock plasticity model).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat049\\sigeps49.F
  (engine/source/materials/mat/mat049/m49law.F)
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\materials\\mat\\mat049\\hm_read_mat49.F
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials.law49_steinb import (
    Law49Params,
    build_law49,
    extra_shapes,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)


def test_steinberg_guinan_shear_modulus_scaling():
    """Verify G(P, T) = G0 * (1 + A * P / V^(1/3) - B * (T - 300)) where V = rho0 / rho."""
    G0 = 28.6e9  # Copper shear modulus ~ 28.6 GPa
    A = 1.0e-11   # G'_P / G0 [1/Pa]
    B = 3.5e-4    # -G'_T / G0 [1/K]
    T0 = 300.0

    p = Law49Params(
        rho0=8960.0,
        g0=G0,
        bulk=140.0e9,
        sig0=120.0e6,
        b1=A,
        b2=A,
        h=B,
        t0=T0,
        tmelt=1356.0,
    )

    # Check property aliases
    assert p.A == A
    assert p.B == B
    assert p.y0 == 120.0e6
    assert p.Y0 == 120.0e6

    # Apply pressure P = 10 GPa (1e10 Pa), compression ratio df = rho/rho0 = 1.2 -> V = 1/1.2
    # Temperature T = 500 K -> delta_T = 200 K
    P_val = 10.0e9
    df_val = 1.2
    V_cbrt = (1.0 / df_val) ** (1.0 / 3.0)
    T_val = 500.0

    expected_G = G0 * (1.0 + A * P_val / V_cbrt - B * (T_val - T0))

    # Old stress corresponds to hydrostatic pressure P_val
    sig0 = np.array([-P_val, -P_val, -P_val, 0.0, 0.0, 0.0])
    deps = np.zeros(6)
    extra = {
        "df": df_val,
        "theta": T_val,
    }

    # Small elastic strain increment
    deps[3] = 1.0e-5
    sig_new, epsp, c = solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6, extra=extra, return_tuple=True)

    # In pure shear, dtau = G * dgamma
    dtau = sig_new[3]
    computed_G = dtau / deps[3]

    assert math.isclose(computed_G, expected_G, rel_tol=1e-5)


def test_steinberg_guinan_yield_strength_scaling():
    """Verify Y(eps_p, P, T) = Y0 * (1 + beta * eps_p)^n * (G(P, T) / G0)."""
    G0 = 28.6e9
    Y0 = 120.0e6
    A = 1.0e-11
    B = 3.5e-4
    beta = 36.0
    n = 0.45
    T0 = 300.0

    p = Law49Params(
        rho0=8960.0,
        g0=G0,
        bulk=140.0e9,
        sig0=Y0,
        beta=beta,
        n=n,
        b1=A,
        b2=A,
        h=B,
        t0=T0,
        tmelt=1356.0,
    )

    # Ambient conditions P = 0, T = 300 -> G(P, T) = G0
    # Yield strength with cold work eps_p = 0.05
    epsp_val = 0.05
    expected_Y = Y0 * ((1.0 + beta * epsp_val) ** n)

    sig0 = np.zeros(6)
    # Apply large tensile strain to plastic regime
    deps = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
    sig_new, epsp_new, c = solid_update(p, sig0, deps, epsp=epsp_val, dt=1.0e-6, return_tuple=True)

    # Compute von Mises stress: sigma_eq
    s = sig_new[:3] - np.mean(sig_new[:3])
    svm = math.sqrt(1.5 * np.sum(s ** 2))

    # At plastic yield, von Mises stress reaches Y(eps_p)
    assert svm >= expected_Y
    assert epsp_new > epsp_val


def test_steinberg_guinan_johnson_cook_rate_dependence():
    """Verify high strain rate Johnson-Cook multiplier: 1 + C * ln(eps_dot / eps_dot0)."""
    p_static = Law49Params(
        rho0=8960.0,
        g0=28.6e9,
        bulk=140.0e9,
        sig0=120.0e6,
        c_rate=0.0,
    )
    p_rate = Law49Params(
        rho0=8960.0,
        g0=28.6e9,
        bulk=140.0e9,
        sig0=120.0e6,
        c_rate=0.025,
        eps0=1.0,
    )

    sig0 = np.zeros(6)
    deps = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    # dt = 1e-6 -> strain rate ~ 1e4 s^-1
    dt = 1.0e-6

    sig_static, _, _ = solid_update(p_static, sig0.copy(), deps.copy(), epsp=0.0, dt=dt, return_tuple=True)
    sig_rate, _, _ = solid_update(p_rate, sig0.copy(), deps.copy(), epsp=0.0, dt=dt, return_tuple=True)

    svm_static = math.sqrt(1.5 * np.sum((sig_static[:3] - np.mean(sig_static[:3])) ** 2))
    svm_rate = math.sqrt(1.5 * np.sum((sig_rate[:3] - np.mean(sig_rate[:3])) ** 2))

    # Dynamic yield must be greater than static yield due to positive rate coefficient C
    assert svm_rate > svm_static


def test_steinberg_guinan_group_interface():
    """Verify group-style solid_update and tangent compliance."""
    p = Law49Params(
        rho0=7800.0,
        g0=80.0e9,
        bulk=160.0e9,
        sig0=250.0e6,
    )
    group = SimpleNamespace(
        nel=1,
        elements=[1],
        mat=p,
    )
    fint_in = np.zeros(24)
    res = solid_update(group, x=None, u=None, ur=None, dt=1.0e-6, fint=fint_in)
    assert res is fint_in

    tang = tangent(group)
    assert tang is not None
    assert tang.shape in ((6, 6), (1, 6, 6))
