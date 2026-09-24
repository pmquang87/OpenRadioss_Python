"""Unit tests for LAW74 (3D Hill Orthotropic Plasticity Model with Temperature Dependence).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat074\\sigeps74.F
  starter/source/materials/mat/mat074/hm_read_mat74.F
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials.law74_hill_3d import (
    Law74Params,
    build_law74,
    consistent_solid_tangent,
    extra_shapes,
    solid_tangent,
    solid_update,
    sound_speed_solid,
    tangent,
)
from pyradioss.model.entities import Material


def test_law74_isotropic_mises_limit():
    """Verify that when all directional yield ratios are 1.0, Hill 1948 reduces to isotropic von Mises."""
    sig0 = 250.0
    p = Law74Params(
        e=200000.0,
        nu=0.3,
        sigy0=sig0,
        s11y=1.0,
        s22y=1.0,
        s33y=1.0,
        s12y=1.0,
        s23y=1.0,
        s31y=1.0,
        chard=0.0,
    )

    # In isotropic mode: ff = gg = hh = ll = mm = nn = 0.5
    assert pytest.approx(p.F) == 0.5
    assert pytest.approx(p.G) == 0.5
    assert pytest.approx(p.H) == 0.5
    assert pytest.approx(p.L) == 0.5
    assert pytest.approx(p.M) == 0.5
    assert pytest.approx(p.N) == 0.5

    # Uniaxial tension in x exceeding yield
    sig = np.zeros(6)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig_out, epsp = solid_update(p, sig, deps, epsp=0.0, dt=1.0e-5)

    # Plastic yielding should cap equivalent stress at sig0
    s_dev = sig_out[:3] - np.mean(sig_out[:3])
    cri = math.sqrt(p.F * (s_dev[1] - s_dev[2]) ** 2
                    + p.G * (s_dev[2] - s_dev[0]) ** 2
                    + p.H * (s_dev[0] - s_dev[1]) ** 2)
    assert pytest.approx(cri, rel=1e-2) == sig0
    assert epsp > 0.0


def test_law74_adiabatic_plastic_heating():
    """Verify adiabatic thermal heating: delta_T = yld * delta_epsp / (rho * Cp)."""
    rho0 = 7800.0
    cp = 450.0
    rhocp = rho0 * cp
    sig0 = 300.0e6
    t0 = 293.15

    p = Law74Params(
        rho0=rho0,
        e=210.0e9,
        nu=0.3,
        sigy0=sig0,
        t0=t0,
        rhocp=rhocp,
    )

    extra = {"temp": np.array([t0])}
    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
    sig_out, epsp = solid_update(p, np.zeros(6), deps, epsp=0.0, dt=1.0e-5, extra=extra)

    assert epsp > 0.0
    # Temperature must have increased
    assert extra["temp"][0] > t0
    expected_dT = (sig0 * epsp) / rhocp
    actual_dT = extra["temp"][0] - t0
    assert pytest.approx(actual_dT, rel=0.1) == expected_dT


def test_law74_element_group_interface():
    """Verify element group calling convention compliance: solid_update and tangent."""
    p = Law74Params(e=200000.0, nu=0.3, sigy0=300.0)
    group = SimpleNamespace(
        nel=1,
        elements=[1],
        mat=p,
    )
    fint_in = np.zeros(24)

    res = solid_update(group, x=None, u=None, ur=None, dt=1.0e-5, fint=fint_in)
    assert res is fint_in

    tang = tangent(group)
    assert tang is not None
    assert tang.shape in ((6, 6), (1, 6, 6))
