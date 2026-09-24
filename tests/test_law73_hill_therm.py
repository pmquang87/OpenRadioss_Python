"""
Unit tests for LAW73 (Hill 1948 thermo-elasto-plastic model for shells, /MAT/LAW73, /MAT/HILL_THERM).

Tests:
  - 3D Hill 1948 yield criterion: F*(s22-s33)^2 + G*(s33-s11)^2 + H*(s11-s22)^2 + 2L*s23^2 + 2M*s31^2 + 2N*s12^2 = sigma_y^2
  - Plane-stress Hill sheet reduction: A01*sxx^2 + A02*syy^2 - A03*sxx*syy + A12*sxy^2 = sigma_y^2
  - Lankford coefficients (R00, R45, R90) to Hill anisotropy parameters
  - Reduction to isotropic von Mises criterion
  - Temperature-dependent flow stress with thermal softening
  - Strain-rate dependent yield scaling
  - 2D shell constitutive update with adiabatic plastic heating and mixed hardening
  - Solid element rejection (NotImplementedError)
  - Consistent shell tangent
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials.law73_hill_therm import (
    Law73Params,
    build_law73,
    shell_update,
    solid_update,
    hill48_yield_criterion_3d,
    hill48_yield_criterion_plane_stress,
    hill48_lankford_to_anisotropy,
    hill48_lankford_to_3d_coefficients,
    hill48_thermal_yield_stress,
    consistent_shell_tangent,
    shell_membrane_tangent,
)


# ============================================================================
# 1. 3D Hill 1948 Yield Criterion
# ============================================================================

def test_hill48_yield_criterion_3d_formula():
    # Hill constants: F, G, H, L, M, N
    F, G, H = 0.4, 0.5, 0.6
    L, M, N = 1.2, 1.3, 1.8

    # Uniaxial tension in direction 1 (x-axis): sig = [100, 0, 0, 0, 0, 0]
    # G*(0-s11)^2 + H*(s11-0)^2 = (G+H)*s11^2
    sig_x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    seq_x = hill48_yield_criterion_3d(sig_x, F, G, H, L, M, N)
    assert seq_x == pytest.approx(math.sqrt(G + H) * 100.0)

    # Uniaxial tension in direction 2 (y-axis): sig = [0, 100, 0, 0, 0, 0]
    # F*(s22-0)^2 + H*(0-s22)^2 = (F+H)*s22^2
    sig_y = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    seq_y = hill48_yield_criterion_3d(sig_y, F, G, H, L, M, N)
    assert seq_y == pytest.approx(math.sqrt(F + H) * 100.0)

    # Uniaxial tension in direction 3 (z-axis): sig = [0, 0, 100, 0, 0, 0]
    # F*(0-s33)^2 + G*(s33-0)^2 = (F+G)*s33^2
    sig_z = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    seq_z = hill48_yield_criterion_3d(sig_z, F, G, H, L, M, N)
    assert seq_z == pytest.approx(math.sqrt(F + G) * 100.0)

    # Pure shear in 1-2 plane: sig = [0, 0, 0, 50, 0, 0]
    # 2*N*s12^2
    sig_xy = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    seq_xy = hill48_yield_criterion_3d(sig_xy, F, G, H, L, M, N)
    assert seq_xy == pytest.approx(math.sqrt(2.0 * N) * 50.0)


def test_hill48_3d_isotropic_von_mises_reduction():
    # For isotropic material: F = G = H = 0.5, L = M = N = 1.5
    F = G = H = 0.5
    L = M = N = 1.5

    sig = np.array([120.0, 40.0, -30.0, 25.0, 15.0, -10.0])
    s11, s22, s33, s12, s23, s31 = sig

    # Von Mises: sqrt(0.5*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2) + 3*(s12^2 + s23^2 + s31^2))
    vm_expected = math.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2)
    )
    vm_hill = hill48_yield_criterion_3d(sig, F, G, H, L, M, N)
    assert vm_hill == pytest.approx(vm_expected, rel=1e-12)


# ============================================================================
# 2. Plane-Stress Hill 1948 Sheet Reduction
# ============================================================================

def test_hill48_plane_stress_formula():
    A01, A02, A03, A12 = 1.2, 1.4, 0.9, 2.8

    # Uniaxial x: sqrt(A01) * sxx
    sig_x = np.array([100.0, 0.0, 0.0])
    assert hill48_yield_criterion_plane_stress(sig_x, A01, A02, A03, A12) == pytest.approx(
        math.sqrt(A01) * 100.0
    )

    # Pure shear: sqrt(A12) * sxy
    sig_xy = np.array([0.0, 0.0, 50.0])
    assert hill48_yield_criterion_plane_stress(sig_xy, A01, A02, A03, A12) == pytest.approx(
        math.sqrt(A12) * 50.0
    )

    # Isotropic plane stress (A01=A02=A03=1.0, A12=3.0)
    sig_gen = np.array([80.0, 30.0, 20.0])
    vm_plane_expected = math.sqrt(80.0 ** 2 + 30.0 ** 2 - 80.0 * 30.0 + 3.0 * 20.0 ** 2)
    vm_plane_calc = hill48_yield_criterion_plane_stress(sig_gen, 1.0, 1.0, 1.0, 3.0)
    assert vm_plane_calc == pytest.approx(vm_plane_expected)


def test_hill48_lankford_conversions():
    # Isotropic R00 = R45 = R90 = 1.0
    a01, a02, a03, a12, r, h = hill48_lankford_to_anisotropy(1.0, 1.0, 1.0)
    assert a01 == pytest.approx(1.0)
    assert a02 == pytest.approx(1.0)
    assert a03 == pytest.approx(1.0)
    assert a12 == pytest.approx(3.0)
    assert r == pytest.approx(1.0)
    assert h == pytest.approx(0.5)

    # 3D coefficients for isotropic material
    f, g, h_val, l, m, n = hill48_lankford_to_3d_coefficients(1.0, 1.0, 1.0)
    assert f == pytest.approx(0.5)
    assert g == pytest.approx(0.5)
    assert h_val == pytest.approx(0.5)
    assert l == pytest.approx(1.5)
    assert m == pytest.approx(1.5)
    assert n == pytest.approx(1.5)


# ============================================================================
# 3. Temperature & Rate Dependent Yield Flow Stress
# ============================================================================

def test_hill48_thermal_and_rate_flow_stress():
    # Johnson-Cook style thermal softening & strain-rate hardening:
    # sigma_y = (A + B*pla^n) * (1 + C*ln(1 + rate/rate0)) * (1 - (T - T0)/(Tm - T0))
    sigma0 = 250.0
    T0 = 293.0
    Tm = 900.0

    def thermo_table(pla, rate, temp):
        hard = sigma0 + 300.0 * (pla ** 0.5)
        rate_factor = 1.0 + 0.05 * math.log(1.0 + max(rate, 0.0) / 1.0)
        temp_factor = max(0.05, 1.0 - (temp - T0) / (Tm - T0))
        yld = hard * rate_factor * temp_factor
        slope = 150.0 * rate_factor * temp_factor
        return yld, slope

    # Baseline at T0, rate=0
    yld_0, _ = hill48_thermal_yield_stress(0.0, 0.0, T0, yield_table=thermo_table)
    assert yld_0 == pytest.approx(sigma0)

    # Elevated temperature T = 500 K causes thermal softening
    yld_hot, _ = hill48_thermal_yield_stress(0.0, 0.0, 500.0, yield_table=thermo_table)
    assert yld_hot < yld_0
    expected_hot = sigma0 * (1.0 - (500.0 - T0) / (Tm - T0))
    assert yld_hot == pytest.approx(expected_hot)

    # High strain rate rate = 100/s causes rate hardening
    yld_dyn, _ = hill48_thermal_yield_stress(0.0, 100.0, T0, yield_table=thermo_table)
    assert yld_dyn > yld_0


# ============================================================================
# 4. 2D Shell Constitutive Update with Adiabatic Heating
# ============================================================================

def test_law73_shell_plastic_loading_and_adiabatic_heating():
    # Set up material with heat capacity rhocp = 2.4e-3
    T0 = 293.0
    rhocp = 2.4e-3
    mat = build_law73(Law73Params(
        e=200000.0,
        nu=0.3,
        yield_table=300.0,
        t0=T0,
        rhocp=rhocp,
    ))

    n = 1
    sig = np.zeros((n, 3))
    deps = np.array([[0.005, 0.0, 0.0]])
    extra = {
        "pla73": np.zeros(n),
        "off73": np.ones(n),
        "uvar73": np.zeros((n, 7)),
        "temp73": np.full(n, T0),
        "vol73": np.ones(n),
        "eint73": np.zeros((n, 2)),
    }

    sig_out, pla_out = shell_update(mat, sig, deps, dt=1e-5, extra=extra)

    # Plastic strain must have developed
    assert pla_out[0] > 0.0
    # Equivalent stress must lie near yield stress 300.0
    seq = math.sqrt(sig_out[0, 0] ** 2 + sig_out[0, 1] ** 2 - sig_out[0, 0] * sig_out[0, 1] + 3.0 * sig_out[0, 2] ** 2)
    assert seq == pytest.approx(300.0, rel=1e-2)


# ============================================================================
# 5. Kinematic Hardening & Bauschinger Effect
# ============================================================================

def test_law73_kinematic_hardening_bauschinger():
    def hard_func(pla, *args):
        return 200.0 + 4000.0 * pla, 4000.0

    mat = build_law73(Law73Params(
        e=200000.0, nu=0.3, yield_table=hard_func, chard=1.0  # pure kinematic
    ))

    extra = {
        "pla73": np.zeros(1),
        "off73": np.ones(1),
        "uvar73": np.zeros((1, 7)),
    }

    # Forward plastic loading
    deps_fwd = np.array([[0.003, 0.0, 0.0]])
    sig_fwd, pla1 = shell_update(mat, np.zeros((1, 3)), deps_fwd, extra=extra)
    assert pla1[0] > 0.0
    alpha_xx = extra["uvar73"][0, 1]
    assert alpha_xx > 0.0  # positive backstress developed

    # Reverse plastic loading: compressive stress yields earlier due to positive backstress
    deps_rev = np.array([[-0.002, 0.0, 0.0]])
    sig_rev, pla2 = shell_update(mat, sig_fwd, deps_rev, epsp=pla1, extra=extra)
    # Bauschinger effect: compressive yield magnitude is reduced by backstress
    assert abs(sig_rev[0, 0]) < abs(sig_fwd[0, 0])
    assert sig_rev[0, 0] - extra["uvar73"][0, 1] < sig_rev[0, 0]


# ============================================================================
# 6. Rejection of Solid Elements (OpenRadioss Shell-Only Contract)
# ============================================================================

def test_law73_solid_rejection_not_implemented():
    mat = build_law73()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update(mat, sig, deps)


# ============================================================================
# 7. Consistent Shell Tangent
# ============================================================================

def test_law73_consistent_shell_tangent_elastic_and_symmetry():
    mat = build_law73(Law73Params(e=210000.0, nu=0.3, yield_table=500.0))
    sig = np.zeros((1, 3))

    # Elastic tangent
    C = consistent_shell_tangent(mat, sig, epsp=np.zeros(1))
    assert C.shape == (1, 3, 3)

    # Check symmetry C_ij = C_ji
    np.testing.assert_allclose(C[0], C[0].T, atol=1e-10)

    # Check plane stress values
    a11 = 210000.0 / (1.0 - 0.3 ** 2)
    a21 = 0.3 * a11
    g = 0.5 * 210000.0 / (1.0 + 0.3)

    assert C[0, 0, 0] == pytest.approx(a11)
    assert C[0, 1, 1] == pytest.approx(a11)
    assert C[0, 0, 1] == pytest.approx(a21)
    assert C[0, 2, 2] == pytest.approx(g)
