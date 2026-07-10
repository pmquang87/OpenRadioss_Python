"""Unit tests: /FUNCT tables and the material law kernels against
closed-form solutions."""

import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials import law01_elastic, law02_johnson_cook
from pyradioss.model.entities import Material


def test_funct_interpolation_and_extrapolation():
    f = FunctTable(1, [0.0, 1.0, 2.0], [0.0, 10.0, 10.0])
    assert f.eval(0.5) == pytest.approx(5.0)
    assert f.eval(1.5) == pytest.approx(10.0)
    # slope extrapolation beyond both ends (Radioss FINTER behaviour)
    assert f.eval(3.0) == pytest.approx(10.0)     # last slope = 0
    assert f.eval(-1.0) == pytest.approx(-10.0)   # first slope = 10


def _steel(law=2):
    params = {"E": 210.0, "nu": 0.3}
    if law == 2:
        params.update(A=0.4, B=0.5, n=0.5, eps_p_max=1e30, sig_max=1e30,
                      c=0.0, eps_dot_0=1.0)
    return Material(id=1, law=law, rho0=7.8e-6, params=params)


def test_law1_solid_uniaxial_strain():
    """Uniaxial STRAIN (confined): sigma_xx = (K + 4G/3) * eps."""
    mat = _steel(law=1)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = 1e-4
    law01_elastic.solid_update(mat, sig, deps)
    expected = (mat.K + 4.0 * mat.G / 3.0) * 1e-4
    assert sig[0, 0] == pytest.approx(expected, rel=1e-12)
    # lateral stress = lambda * eps
    lam = mat.K - 2.0 * mat.G / 3.0
    assert sig[0, 1] == pytest.approx(lam * 1e-4, rel=1e-12)


def test_law2_solid_uniaxial_stress_follows_johnson_cook():
    """Drive a single point in uniaxial STRESS (with the exact elastic
    lateral contraction, then plastic incompressibility handled by the
    return): after sustained plastic flow the axial stress must track
    sigma_y = A + B * eps_p^n."""
    mat = _steel()
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    dt = 1e-3
    deps = np.zeros((1, 6))
    d = 1e-4  # axial strain increment per step
    for _ in range(3000):
        # lateral increments enforcing (approximately) zero lateral stress:
        # elastic: -nu*d; plastic: -d/2. Blend by current regime — simplest
        # robust choice: correct lateral strain each step to cancel the
        # lateral stress accumulated so far.
        lat = -(sig[0, 1] / mat.E) - 0.45 * d
        deps[0] = [d, lat, lat, 0, 0, 0]
        law02_johnson_cook.solid_update(mat, sig, deps, epsp, dt)
    assert epsp[0] > 0.1  # well into the plastic regime
    sy = 0.4 + 0.5 * epsp[0] ** 0.5
    seq = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                         + (sig[0, 1] - sig[0, 2]) ** 2
                         + (sig[0, 2] - sig[0, 0]) ** 2))
    assert seq == pytest.approx(sy, rel=1e-3)


def test_law2_shell_plane_stress_yield():
    """Plane-stress point in equibiaxial tension: von Mises = |sigma|,
    so the stress must saturate at the (hardening) yield curve."""
    mat = _steel()
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)
    deps = np.tile([1e-4, 1e-4, 0.0], (1, 1))
    for _ in range(2000):
        law02_johnson_cook.shell_update(mat, sig, deps, epsp, 1e-3)
    seq = np.sqrt(sig[0, 0] ** 2 - sig[0, 0] * sig[0, 1] + sig[0, 1] ** 2
                  + 3 * sig[0, 2] ** 2)
    sy = 0.4 + 0.5 * epsp[0] ** 0.5
    assert epsp[0] > 0.05
    assert seq == pytest.approx(sy, rel=1e-3)


def test_law2_strain_rate_hardening():
    """The Johnson-Cook rate term must raise the flow stress by
    1 + c*ln(rate/rate0)."""
    mat = _steel()
    mat.params.update(c=0.02, eps_dot_0=1e-3)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    deps[0] = [1e-4, -5e-5, -5e-5, 0, 0, 0]  # deviatoric push, rate 0.1/ms
    for _ in range(200):
        law02_johnson_cook.solid_update(mat, sig, deps, epsp, 1e-3)
    sy_static = 0.4 + 0.5 * epsp[0] ** 0.5
    # the flow stress (von Mises) must exceed the static curve
    vm2 = 0.5 * ((sig[0, 0] - sig[0, 1]) ** 2 + (sig[0, 1] - sig[0, 2]) ** 2
                 + (sig[0, 2] - sig[0, 0]) ** 2)
    assert np.sqrt(vm2) > sy_static * 1.02
