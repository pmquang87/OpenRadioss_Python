"""Tests for LAW36 tabulated elastic-plastic constitutive model and BUG-MAT-03 verification.

Fortran reference: sigeps36.F, sigeps36c.F, plas36.F.
Verifies:
- Curve weight clipping w = np.clip(w, 0.0, 1.0) when strain rate exceeds highest tabulated rate (BUG-MAT-03)
- Clamping to first curve when strain rate is below lowest tabulated rate
- 3D solid and 2D shell plane-stress plasticity updates
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials import law36_tabulated
from pyradioss.model.entities import Material


def _make_law36_material(curves=None, rates=None):
    if curves is None:
        curves = [
            ([0.0, 0.1, 0.5], [200.0, 250.0, 300.0]),
            ([0.0, 0.1, 0.5], [300.0, 350.0, 400.0]),
        ]
    if rates is None:
        rates = [0.0, 100.0]

    params = {"E": 210000.0, "nu": 0.3}
    cxs, cys, css = [], [], []
    for x, y in curves:
        f = FunctTable(1, x, y)
        cxs.append(f.x)
        cys.append(f.y)
        css.append(f.slope)
    params.update(curve_x=cxs, curve_y=cys, curve_s=css,
                  rates=np.asarray(rates, dtype=float))
    return Material(id=1, law=36, rho0=7.8e-9, params=params)


def test_law36_curve_weight_clipping_above_last_rate():
    """BUG-MAT-03: w = np.clip(w, 0.0, 1.0) ensures clamping to last curve at high rates."""
    mat = _make_law36_material()

    epsp = np.array([0.0])
    # Very high strain rate: 10000.0 >> 100.0
    rate_high = np.array([10000.0])

    sy, H = law36_tabulated._yield_stress(mat, epsp, rate_high)

    # Without clipping, w = (10000 - 0) / 100 = 100.0, sy would extrapolate to 200 + 100*(300-200) = 10200 MPa!
    # Under BUG-MAT-03 fix, w is clipped to 1.0, so sy is clamped to the last curve (300.0 MPa).
    assert sy[0] == pytest.approx(300.0, rel=1e-5)


def test_law36_curve_weight_below_first_rate():
    """Verify that strain rate below first rate clamps to the first curve."""
    mat = _make_law36_material()

    epsp = np.array([0.0])
    rate_low = np.array([0.0])

    sy, H = law36_tabulated._yield_stress(mat, epsp, rate_low)
    assert sy[0] == pytest.approx(200.0, rel=1e-5)


def test_law36_solid_plastic_step():
    """Verify solid element update under plastic deformation with LAW36."""
    mat = _make_law36_material()

    sig = np.zeros((1, 6))
    deps = np.array([[0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    dt = 1e-4

    sig_out, epsp_out = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=dt)

    assert epsp_out[0] > 0.0
    s_dev = sig_out[0, :3] - np.mean(sig_out[0, :3])
    vm = math.sqrt(1.5 * np.sum(s_dev**2))
    assert vm >= 200.0
    assert np.all(np.isfinite(sig_out))


def test_law36_shell_plane_stress():
    """Verify shell plane-stress update with LAW36."""
    mat = _make_law36_material()

    sig = np.zeros((1, 3))
    deps = np.array([[0.003, 0.0, 0.0]])
    epsp = np.zeros(1)
    dt = 1e-4

    sig_out, epsp_out = law36_tabulated.shell_update(mat, sig, deps, epsp, dt=dt)

    assert epsp_out[0] > 0.0
    vm = math.sqrt(sig_out[0, 0]**2 - sig_out[0, 0]*sig_out[0, 1] + sig_out[0, 1]**2 + 3.0*sig_out[0, 2]**2)
    assert vm >= 200.0
    assert np.all(np.isfinite(sig_out))
