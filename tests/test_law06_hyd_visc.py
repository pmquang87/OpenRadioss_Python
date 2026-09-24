"""
Unit tests for LAW06 (hydrodynamic viscous fluid / Newtonian fluid, /MAT/LAW6, /MAT/HYD_VISC).

Tests:
  - 1D Newtonian viscous shear stress relation (tau = eta * gamma_dot)
  - 3D Newtonian viscous deviatoric stress tensor (s_ij = 2 * eta * e_dot_ij)
  - Trace-free property of deviatoric stress (tr(s) = 0)
  - Pure hydrostatic strain rate yielding zero deviatoric stress
  - Linear Equation of State pressure and cavitation limit (Pmin)
  - Polynomial Equation of State pressure in compression and expansion
  - Total Cauchy stress assembly (sigma_ij = s_ij - P * delta_ij)
  - 3D solid constitutive update for water / oil fluid simulation
  - Shell element rejection (NotImplementedError)
  - Consistent solid algorithmic tangent
  - Material physics registry registration
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material, EquationOfState
from pyradioss.materials.law06_hyd_visc import (
    build_law6,
    solid_update,
    shell_update,
    consistent_solid_tangent,
    newtonian_shear_stress,
    newtonian_deviatoric_stress,
    linear_eos_pressure,
    polynomial_eos_pressure,
    hydrodynamic_fluid_stress,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. 1D Newtonian Shear Stress
# ============================================================================

def test_newtonian_shear_stress_1d():
    visc = 1.5e-3  # Pa.s
    rate = 200.0   # 1/s

    # Dynamic viscosity: tau = visc * rate
    tau = newtonian_shear_stress(rate, visc)
    assert tau == pytest.approx(visc * rate)

    # Kinematic viscosity scaling: eta = visc * rho
    rho = 1000.0
    tau_kin = newtonian_shear_stress(rate, visc, rho=rho, kinematic=True)
    assert tau_kin == pytest.approx(visc * rho * rate)

    # Vectorized strain rates
    rates = np.array([0.0, 50.0, 100.0, -200.0])
    taus = newtonian_shear_stress(rates, visc)
    np.testing.assert_allclose(taus, visc * rates)


# ============================================================================
# 2. 3D Newtonian Deviatoric Stress Tensor
# ============================================================================

def test_newtonian_deviatoric_stress_3d_normal_components():
    eta = 2.0  # Pa.s
    # Uniaxial extension: D1 = 3.0, D2 = 0, D3 = 0 -> tr(D) = 3.0, tr(D)/3 = 1.0
    # e1 = 3.0 - 1.0 = 2.0 -> s11 = 2 * eta * 2.0 = 4.0 * eta
    # e2 = 0.0 - 1.0 = -1.0 -> s22 = 2 * eta * (-1.0) = -2.0 * eta
    # e3 = 0.0 - 1.0 = -1.0 -> s33 = 2 * eta * (-1.0) = -2.0 * eta
    edot = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = newtonian_deviatoric_stress(edot, eta)

    assert s[0] == pytest.approx(4.0 * eta)
    assert s[1] == pytest.approx(-2.0 * eta)
    assert s[2] == pytest.approx(-2.0 * eta)
    assert s[3] == pytest.approx(0.0)

    # Deviatoric stress trace must be identically zero
    assert pytest.approx(0.0, abs=1e-12) == (s[0] + s[1] + s[2])


def test_newtonian_deviatoric_stress_shear_components():
    eta = 1.2
    # Pure shear: gamma_xy_dot = 10.0, gamma_yz_dot = -5.0, gamma_zx_dot = 8.0
    edot = np.array([0.0, 0.0, 0.0, 10.0, -5.0, 8.0])
    s = newtonian_deviatoric_stress(edot, eta)

    assert s[0] == pytest.approx(0.0)
    assert s[1] == pytest.approx(0.0)
    assert s[2] == pytest.approx(0.0)
    assert s[3] == pytest.approx(eta * 10.0)
    assert s[4] == pytest.approx(eta * -5.0)
    assert s[5] == pytest.approx(eta * 8.0)


def test_newtonian_deviatoric_stress_pure_volumetric_zero():
    eta = 5.0
    # Pure volumetric expansion: D1 = D2 = D3 = 4.0 -> deviatoric rate = 0
    edot = np.array([4.0, 4.0, 4.0, 0.0, 0.0, 0.0])
    s = newtonian_deviatoric_stress(edot, eta)
    np.testing.assert_allclose(s, np.zeros(6), atol=1e-12)


# ============================================================================
# 3. Equation of State (EOS) Pressure
# ============================================================================

def test_linear_eos_pressure():
    rho0 = 1000.0
    bulk = 2.2e9  # bulk modulus of water (2.2 GPa)

    # Compression: rho = 1010 -> mu = 0.01 -> P = 2.2e7 Pa
    p_comp = linear_eos_pressure(1010.0, rho0, bulk)
    assert p_comp == pytest.approx(2.2e7)

    # Expansion: rho = 990 -> mu = -0.01 -> P = -2.2e7 Pa
    p_exp = linear_eos_pressure(990.0, rho0, bulk)
    assert p_exp == pytest.approx(-2.2e7)

    # Cavitation limit Pmin = -1e5 Pa
    p_cav = linear_eos_pressure(990.0, rho0, bulk, pmin=-1e5)
    assert p_cav == pytest.approx(-1e5)


def test_polynomial_eos_pressure():
    c0 = 0.0
    c1 = 2.2e9
    c2 = 1.5e8
    c3 = 0.0
    c4 = 0.4
    c5 = 0.0
    e0 = 100.0

    # Compression: mu = 0.02
    mu_comp = 0.02
    expected_comp = c0 + c1 * mu_comp + c2 * (mu_comp ** 2) + c4 * e0
    p_comp = polynomial_eos_pressure(mu_comp, c0, c1, c2, c3, c4, c5, e0)
    assert p_comp == pytest.approx(expected_comp)

    # Expansion: mu = -0.01
    mu_exp = -0.01
    expected_exp = c0 + c1 * mu_exp + c3 * (mu_exp ** 2) + c4 * e0
    p_exp = polynomial_eos_pressure(mu_exp, c0, c1, c2, c3, c4, c5, e0)
    assert p_exp == pytest.approx(expected_exp)


# ============================================================================
# 4. Total Hydrodynamic Stress Assembly
# ============================================================================

def test_hydrodynamic_fluid_stress_cauchy():
    visc = 1.0e-3
    rho = 1000.0
    dt = 0.01
    deps = np.array([0.02, 0.0, 0.0, 0.05, 0.0, 0.0])
    p_eos = 1.5e6  # 1.5 MPa pressure

    # Total Cauchy stress: sig_ij = s_ij - P * delta_ij
    sig = hydrodynamic_fluid_stress(deps, dt, visc, rho, p_eos=p_eos)

    # Mean normal stress: (sig_xx + sig_yy + sig_zz) / 3 = -P
    mean_stress = (sig[0] + sig[1] + sig[2]) / 3.0
    assert mean_stress == pytest.approx(-p_eos)

    # Shear stress unaffected by hydrostatic pressure
    assert sig[3] == pytest.approx(visc * rho * (0.05 / dt))


# ============================================================================
# 5. 3D Solid Constitutive Update (Water / Oil Simulation)
# ============================================================================

def test_solid_update_water_properties():
    # Water: rho0 = 1000 kg/m^3, DAMP1 = 1.0e-3, bulk c1 = 2.2e9
    mat = build_law6({
        "id": 1,
        "density": 1000.0,
        "params": {
            "DAMP1": 1.0e-3,
            "MAT_C1": 2.2e9,
        },
    })

    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.001, -0.0005, -0.0005, 0.002, 0.0, 0.0]])
    dt = 1e-4
    extra = {"rho": np.full(n, 1000.0)}

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, None, dt=dt, extra=extra)

    # Deviatoric stress:
    # Rate: D1 = 10.0, D2 = -5.0, D3 = -5.0, D4 = 20.0
    # eta = 1.0e-3 * 1000.0 = 1.0 Pa.s
    # s_xx = 2 * 1.0 * (10.0 - 0) = 20.0 Pa
    # s_xy = 1.0 * 20.0 = 20.0 Pa
    assert sig_out[0, 0] == pytest.approx(20.0)
    assert sig_out[0, 1] == pytest.approx(-10.0)
    assert sig_out[0, 2] == pytest.approx(-10.0)
    assert sig_out[0, 3] == pytest.approx(20.0)

    # Sound speed for water: c = sqrt(c1 / rho0) = sqrt(2.2e9 / 1000) ~ 1483.24 m/s
    expected_c = math.sqrt(2.2e9 / 1000.0)
    assert c_out[0] == pytest.approx(expected_c, rel=1e-5)


# ============================================================================
# 6. Shell Element Rejection
# ============================================================================

def test_law06_shell_rejection():
    mat = build_law6({"id": 1, "density": 1000.0, "params": {"visc": 1.0}})
    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        shell_update(mat, sig, deps)


# ============================================================================
# 7. Consistent Solid Tangent Operator
# ============================================================================

def test_law06_consistent_solid_tangent():
    bulk = 2.0e9
    visc = 1.0e-3
    rho = 1000.0
    dt = 1e-4

    mat = build_law6({
        "id": 1,
        "density": rho,
        "params": {
            "visc": visc,
            "c1": bulk,
        },
    })

    sig = np.zeros((1, 6))
    extra = {"rho": np.array([rho])}
    D = consistent_solid_tangent(mat, sig, dt=dt, extra=extra)

    # Tangent symmetry: D_ij = D_ji
    np.testing.assert_allclose(D[0], D[0].T, atol=1e-12)

    # Shear tangent: G_visc = (visc * rho) / dt = 1.0 / 1e-4 = 10000.0
    g_visc = (visc * rho) / dt
    assert D[0, 3, 3] == pytest.approx(g_visc)
    assert D[0, 4, 4] == pytest.approx(g_visc)
    assert D[0, 5, 5] == pytest.approx(g_visc)

    # Volumetric diagonal: bulk + 4/3 * g_visc
    assert D[0, 0, 0] == pytest.approx(bulk + (4.0 / 3.0) * g_visc)
    # Off-diagonal: bulk - 2/3 * g_visc
    assert D[0, 0, 1] == pytest.approx(bulk - (2.0 / 3.0) * g_visc)


# ============================================================================
# 8. Material Registry
# ============================================================================

def test_law06_registry():
    assert "LAW6" in MAT_PHYSICS_REGISTRY
    assert "HYD_VISC" in MAT_PHYSICS_REGISTRY
    assert "HYDRO" in MAT_PHYSICS_REGISTRY
    fn = MAT_PHYSICS_REGISTRY["LAW6"]
    mat = fn({"id": 10, "density": 1000.0, "params": {"visc": 1.0}})
    assert isinstance(mat, Material)
    assert mat.law == 6
