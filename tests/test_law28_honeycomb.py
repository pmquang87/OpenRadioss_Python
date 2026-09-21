"""
Unit tests for LAW28 (orthotropic honeycomb material, /MAT/LAW28, /MAT/HONEYCOMB).

Fortran origin:
- engine/source/materials/mat/mat028/sigeps28.F
- starter/source/materials/mat/mat028/hm_read_mat28.F

Physics verified:
1. Orthotropic uncoupled elasticity:
   - Independent moduli: E11, E22, E33, G12, G23, G31
   - Zero Poisson coupling (nu = 0)
2. Bilinear crush curve (elastic loading then crush plateau):
   - Linear elastic stress up to crush stress / yield limit
   - Plateau clamping under continuing compression
3. Rate dependence:
   - Scaling of crush / yield stress under high strain rate
4. Element rupture / deletion:
   - Strain exceeding EPS_MAX zeros off and stress
5. Tangents, sound speed, and history variables:
   - Diagonal orthotropic elastic tangent, c = sqrt(max(E, G)/rho0)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials.law28_honeycomb import (
    build_law28,
    solid_update,
    solid_tangent,
    tangent,
    sound_speed,
    extra_shapes,
    needs_defgrad,
)


def _make_honeycomb(
    e11: float = 1000.0,
    e22: float = 500.0,
    e33: float = 200.0,
    g12: float = 300.0,
    g23: float = 150.0,
    g31: float = 80.0,
    rho0: float = 0.5,
    **kwargs,
) -> Material:
    """Create a LAW28 Material object."""
    rec = {
        "id": 28,
        "density": rho0,
        "title": "HONEYCOMB_TEST",
        "params": {
            "E11": e11,
            "E22": e22,
            "E33": e33,
            "G12": g12,
            "G23": g23,
            "G31": g31,
            **kwargs,
        },
    }
    return build_law28(rec)


def test_orthotropic_uncoupled_elasticity():
    """
    Verify orthotropic uncoupled elasticity:
    Different E11, E22, E33, G12, G23, G31 with zero Poisson coupling.
    """
    e11, e22, e33 = 1000.0, 500.0, 200.0
    g12, g23, g31 = 300.0, 150.0, 80.0
    mat = _make_honeycomb(e11=e11, e22=e22, e33=e33, g12=g12, g23=g23, g31=g31)

    sig = np.zeros(6, dtype=float)
    # Apply strain increments in all 6 components
    deps = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006], dtype=float)

    sig_new, _, c = solid_update(mat, sig, deps, dt=1.0e-4, return_tuple=True)

    # Verify uncoupled Hooke's law: sigma_i = E_i * deps_i
    assert sig_new[0] == pytest.approx(e11 * 0.001, rel=1e-5)
    assert sig_new[1] == pytest.approx(e22 * 0.002, rel=1e-5)
    assert sig_new[2] == pytest.approx(e33 * 0.003, rel=1e-5)
    assert sig_new[3] == pytest.approx(g12 * 0.004, rel=1e-5)
    assert sig_new[4] == pytest.approx(g23 * 0.005, rel=1e-5)
    assert sig_new[5] == pytest.approx(g31 * 0.006, rel=1e-5)

    # Sound speed is governed by max modulus: sqrt(E11 / rho0)
    assert c == pytest.approx(math.sqrt(e11 / 0.5))


def test_bilinear_crush_plateau():
    """
    Verify bilinear pressure/stress curve:
    Elastic response followed by a crush plateau at sig_crush.
    """
    e11 = 1000.0
    sig_crush_11 = 20.0  # crush plateau at 20 MPa (yield strain = 20/1000 = 0.02)
    mat = _make_honeycomb(e11=e11, sig_crush_0=sig_crush_11, h_crush=0.0)

    # Step 1: Small strain below crush limit (eps = 0.01 -> expected sig = 10.0 MPa < 20.0)
    sig = np.zeros(6, dtype=float)
    deps1 = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig1, _, _ = solid_update(mat, sig, deps1, dt=1.0e-3)
    assert sig1[0] == pytest.approx(10.0, abs=1e-5)

    # Step 2: Large strain well past crush limit (eps = 0.05 -> trial sig = 50.0 MPa)
    # Must be clamped to crush plateau 20.0 MPa
    deps2 = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig2, _, _ = solid_update(mat, sig1, deps2, dt=1.0e-3)
    assert sig2[0] == pytest.approx(20.0, abs=1e-5)


def test_rate_dependence():
    """
    Verify rate dependence:
    At high strain rate, the crush limit increases by rate factor.
    """
    # Base crush stress = 20.0 MPa
    # Rate sensitivity: c_rate = 0.2, eps_dot_0 = 1.0 /s
    mat = _make_honeycomb(
        e11=1000.0,
        sig_crush_0=20.0,
        c_rate=0.2,
        eps_dot_0=1.0,
    )

    sig = np.zeros(6, dtype=float)
    # Apply strain = 0.05 (well into crush plateau)
    deps = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Case A: Low strain rate (dt = 0.05 -> rate = 1.0 /s -> rate_factor = 1 + 0.2*ln(1) = 1.0)
    sig_low, _, _ = solid_update(mat, sig, deps, dt=0.05)

    # Case B: High strain rate (dt = 0.0005 -> rate = 100.0 /s -> rate_factor = 1 + 0.2*ln(100) = 1.921)
    sig_high, _, _ = solid_update(mat, sig, deps, dt=0.0005)

    # High rate stress should be greater than low rate stress
    assert sig_high[0] > sig_low[0]
    expected_rate_factor = 1.0 + 0.2 * math.log(100.0)
    assert sig_high[0] == pytest.approx(20.0 * expected_rate_factor, rel=1e-3)


def test_element_rupture_and_deletion():
    """
    Verify element deletion when maximum strain is exceeded:
    eps_xx > eps_max11 sets off = 0.0 and zeroes out stress.
    """
    eps_max11 = 0.05
    mat = _make_honeycomb(e11=1000.0, eps_max11=eps_max11)

    sig = np.zeros(6, dtype=float)
    extra = {"off": np.array([1.0]), "eps28": np.zeros(6, dtype=float)}

    # Exceed failure strain: deps_xx = 0.06 > 0.05
    deps = np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_new, _, _ = solid_update(mat, sig, deps, dt=1.0e-3, extra=extra)

    # Element must be deleted and stress zeroed
    assert extra["off"][0] == 0.0
    assert np.all(sig_new == 0.0)


def test_tangents_and_utilities():
    """Verify tangent stiffness matrix, alias, sound speed, and history shapes."""
    mat = _make_honeycomb(
        e11=800.0,
        e22=400.0,
        e33=200.0,
        g12=150.0,
        g23=100.0,
        g31=50.0,
        rho0=0.2,
    )

    D = solid_tangent(mat)
    assert D.shape == (1, 6, 6)
    # Check diagonal moduli
    assert D[0, 0, 0] == pytest.approx(800.0)
    assert D[0, 1, 1] == pytest.approx(400.0)
    assert D[0, 2, 2] == pytest.approx(200.0)
    assert D[0, 3, 3] == pytest.approx(150.0)
    assert D[0, 4, 4] == pytest.approx(100.0)
    assert D[0, 5, 5] == pytest.approx(50.0)

    # Tangent alias
    D_alias = tangent(mat)
    assert np.allclose(D, D_alias)

    # Sound speed
    c = sound_speed(mat)
    assert c == pytest.approx(math.sqrt(800.0 / 0.2))

    # History shapes
    shapes = extra_shapes(mat, nip=1)
    assert "eps28" in shapes
    assert "off28" in shapes
    assert not needs_defgrad(mat)
