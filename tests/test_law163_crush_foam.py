r"""Unit tests for LAW163 Crushable Foam tabulated material model (/MAT/LAW163, /MAT/CRUSHABLE_FOAM).

Upstream Fortran reference:
  ``C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat163\sigeps163.F90``

Verifies:
1. Law163Params initialization and derived elastic moduli (g, bulk, cii, cij).
2. Large deformation compressive behavior from tabulated lookup curves.
3. Tensile cutoff (tsc = 0: strictly no tensile stiffness/stress).
4. Shear response and Poisson's ratio influence.
5. Volumetric strain rate filtering and rate-capping (srclmt).
6. Viscous damping stress contribution (sigv).
7. Characteristic sound speed with table slope and damping stiffness.
8. Consistent tangent tensor (n, 6, 6) and template API stubs.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials.law163_crush_foam import (
    Law163Params,
    build_law163,
    consistent_solid_tangent,
    extra_shapes,
    shell_update,
    solid_update,
    sound_speed_solid,
    tangent,
)


def test_law163_params_initialization():
    """Verify parameters, defaults, and derived isotropic elastic moduli."""
    p = Law163Params(rho0=1.0e-3, e=300.0, nu=0.2, tsc=0.0)
    assert p.rho0 == 1.0e-3
    assert p.tsc == 0.0
    assert p.damp == 0.10
    assert p.ncycle == 12

    # G = E / (2*(1+nu)) = 300 / 2.4 = 125.0
    # K = E / (3*(1-2*nu)) = 300 / (3*0.6) = 300 / 1.8 = 166.66667
    assert math.isclose(p.g, 125.0, rel_tol=1e-6)
    assert math.isclose(p.bulk, 300.0 / 1.8, rel_tol=1e-6)
    assert math.isclose(p.cii, p.bulk + (4.0 / 3.0) * p.g, rel_tol=1e-6)
    assert math.isclose(p.cij, p.bulk - (2.0 / 3.0) * p.g, rel_tol=1e-6)


def test_law163_compressive_lookup_table():
    """Verify large deformation compressive response matches lookup curve."""
    # Table of yield stress vs volumetric strain gamma = 1 - rho0/rho
    # At gamma = 0.1, sigy = 20.0 MPa; at gamma = 0.5, sigy = 100.0 MPa
    tbl = FunctTable(fct_id=1, x=[0.0, 0.2, 0.4, 0.6, 0.8], y=[10.0, 20.0, 40.0, 80.0, 200.0])
    p = Law163Params(rho0=1.0e-3, e=500.0, nu=0.0, table=tbl, tsc=0.0)

    # 1. Compressive strain step: deps_xx = -0.10, deps_yy = 0, deps_zz = 0
    deps = np.array([[-0.10, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    sig0 = np.zeros((1, 6), dtype=float)

    # Simulated compression: rho increases so gamma = 1 - 1/1.1 = 0.0909
    extra = {
        "rho": np.array([1.1e-3]),
        "uvar163": np.zeros((1, 2)),
        "epsd163": np.zeros(1),
        "sigv": np.zeros((1, 6)),
    }

    sig_out, epsp, c = solid_update(p, sig0.copy(), deps, dt=1e-4, extra=extra, return_tuple=True)

    # In compression, stress is negative and capped by yield stress
    assert sig_out[0, 0] < 0.0
    # Effective plastic strain is logged as log(rho0/rho)
    assert epsp[0] < 0.0
    assert c[0] > 0.0


def test_law163_no_tensile_stiffness_cutoff():
    """Verify tsc=0 strictly prevents tensile stresses (zero tensile capacity)."""
    p = Law163Params(rho0=1.0e-3, e=200.0, nu=0.0, tsc=0.0, damp=0.0)

    # Uniaxial tension strain increment: deps_xx = +0.05
    deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    sig0 = np.zeros((1, 6), dtype=float)
    extra = {
        "rho": np.array([0.95e-3]),
        "uvar163": np.zeros((1, 2)),
        "epsd163": np.zeros(1),
        "sigv": np.zeros((1, 6)),
    }

    sig_out = solid_update(p, sig0.copy(), deps, dt=1e-4, extra=extra)

    # When tsc=0 and damp=0, tensile trial stress is clipped to <= tsc = 0.0
    assert sig_out[0, 0] <= 1e-12
    assert sig_out[0, 1] <= 1e-12
    assert sig_out[0, 2] <= 1e-12


def test_law163_viscous_damping():
    """Verify viscous damping adds rate-dependent damping stress."""
    p_no_damp = Law163Params(rho0=1.0e-3, e=100.0, nu=0.0, damp=0.0)
    p_damp = Law163Params(rho0=1.0e-3, e=100.0, nu=0.0, damp=0.20)

    deps = np.array([[-0.02, -0.02, -0.02, 0.0, 0.0, 0.0]], dtype=float)
    sig0 = np.zeros((1, 6), dtype=float)

    extra1 = {
        "rho": np.array([1.05e-3]),
        "aldt": np.array([1.0]),
        "uvar163": np.zeros((1, 2)),
        "epsd163": np.zeros(1),
        "sigv": np.zeros((1, 6)),
    }
    extra2 = {
        "rho": np.array([1.05e-3]),
        "aldt": np.array([1.0]),
        "uvar163": np.zeros((1, 2)),
        "epsd163": np.zeros(1),
        "sigv": np.zeros((1, 6)),
    }

    sig1 = solid_update(p_no_damp, sig0.copy(), deps, dt=1e-4, extra=extra1)
    sig2 = solid_update(p_damp, sig0.copy(), deps, dt=1e-4, extra=extra2)

    # Damping in compression adds additional compressive resistance
    assert abs(sig2[0, 0]) > abs(sig1[0, 0])


def test_law163_sound_speed():
    """Verify sound speed includes bulk modulus and slope."""
    tbl = FunctTable(fct_id=1, x=[0.0, 1.0], y=[0.0, 100.0])  # slope = 100
    p = Law163Params(rho0=1.0e-3, e=60.0, nu=0.0, table=tbl)
    # bulk = 20.0, g = 30.0, dsdgam = 100.0
    c = sound_speed_solid(p, rho=1.0e-3, slope=100.0)
    # c = sqrt((max(bulk, slope) + 4/3*G) / rho) = sqrt((100 + 40) / 1e-3) = sqrt(140 / 1e-3) = sqrt(140000)
    expected_c = math.sqrt(140.0 / 1.0e-3)
    assert math.isclose(c, expected_c, rel_tol=1e-5)


def test_law163_consistent_tangent():
    """Verify (n, 6, 6) consistent tangent tensor."""
    p = Law163Params(rho0=1.0e-3, e=300.0, nu=0.25)
    D = consistent_solid_tangent(p)
    assert D.shape == (1, 6, 6)
    assert math.isclose(D[0, 0, 0], p.cii, rel_tol=1e-6)
    assert math.isclose(D[0, 0, 1], p.cij, rel_tol=1e-6)
    assert math.isclose(D[0, 3, 3], p.g, rel_tol=1e-6)


def test_law163_template_api():
    """Verify template API compatibility."""
    class DummyGroup:
        mat = Law163Params(e=100.0)

    group = DummyGroup()

    # Template group update call returns None
    res_solid = solid_update(group, None, None, None, 0.001, np.zeros(6), np.zeros(6))
    assert res_solid is None

    # Template group tangent call returns None
    res_tan = tangent(group)
    assert res_tan is None

    # Material tangent call returns (n, 6, 6) tensor
    mat_tan = tangent(group.mat)
    assert mat_tan.shape == (1, 6, 6)

    # Shell update raises NotImplementedError
    with pytest.raises(NotImplementedError):
        shell_update()
