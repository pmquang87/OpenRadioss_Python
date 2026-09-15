"""Tests for /MAT/LAW101 Bouvard viscoplastic polymer constitutive physics.

Verifies:
  - Rate- and temperature-dependent modulus (EMOD_TPU)
  - Initial user variable allocation (UVAR 1..42)
  - Hyperbolic sine viscoplastic flow rule with Arrhenius thermal activation
  - Hydrostatic pressure sensitivity on yield threshold
  - Isotropic defect hardening (zeta_1) and orientation backstress (beta)
  - Non-affine network locking stretch singularity (mu_B)
  - Adiabatic heating and dissipation temperature rise
  - Dynamic acoustic sound speed and algorithmic tangent stiffness
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law101_plas_poly import (
    BouvardParams,
    build_law101,
    emod_tpu,
    init_history,
    solid_update,
    solid_update_single,
    sound_speed,
    solid_tangent,
    extra_shapes,
)


def _sample_params(**kwargs) -> BouvardParams:
    """Create typical polypropylene material parameters for testing."""
    defaults = dict(
        rho0=0.9e-6,        # kg/mm^3
        e_ref=1500.0,       # MPa
        e1=-2.5,            # MPa/K
        nu=0.38,
        ve1=0.25,
        ve2=1.5,
        edot_ref=1.0e-4,    # 1/s
        gamma0_ref=1.0e-3,  # 1/s
        alpha_p=0.08,       # Pressure sensitivity
        deltah=45000.0,     # J/mol
        vol=2.0e-28,        # Activation volume
        m=1.8,
        c3=-0.05,
        c4=25.0,            # Yield stress at T0
        alphak1=0.15,
        alphak2=0.05,
        h0=50.0,
        zeta1_0=0.1,
        c5=-0.001,
        c6=1.0,
        c7=-0.002,
        c8=2.0,
        c9=-0.01,
        c10=10.0,
        h1=20.0,
        zeta2_0=0.05,
        c11=-0.001,
        c12=1.5,
        c13=0.01,
        c14=5.0,
        c1=-0.05,
        c2=8.0,
        lambda_l=5.0,
        rho_p=0.9e-6,
        cv=1800.0,
        theta0=293.15,
        beta0=1.2e-4,
        theta_g=250.0,
        factor=0.9,
        temp_opt=0.0,
        theta_i=293.15,
    )
    defaults.update(kwargs)
    return build_law101(**defaults)


def test_emod_tpu_rate_and_temp():
    """Verify EMOD_TPU temperature softening and strain-rate stiffening."""
    p = _sample_params()

    # At reference temperature and reference rate:
    # log_diff = log10(1e-4) - log10(1e-4) = 0 -> exp(0) = 1 -> rate_term = ve1 / 2
    e_ref_val = emod_tpu(p.e_ref, p.e1, p.theta0, p.theta0, p.ve1, p.ve2, p.edot_ref, p.edot_ref)
    expected_ref = p.e_ref * (1.0 + p.ve1 / 2.0)
    assert pytest.approx(e_ref_val, rel=1e-6) == expected_ref

    # Higher strain rate -> stiffer modulus
    e_high_rate = emod_tpu(p.e_ref, p.e1, p.theta0, p.theta0, p.ve1, p.ve2, 10.0, p.edot_ref)
    assert e_high_rate > e_ref_val

    # Higher temperature -> softer modulus (since e1 < 0)
    e_hot = emod_tpu(p.e_ref, p.e1, p.theta0 + 50.0, p.theta0, p.ve1, p.ve2, p.edot_ref, p.edot_ref)
    assert e_hot < e_ref_val


def test_init_history_state_variables():
    """Verify 42-element user variable initialization matching sigeps101.F."""
    p = _sample_params()
    uvar = init_history(p)
    assert len(uvar) == 42

    # Viscoplastic F^v -> Identity
    assert pytest.approx(uvar[0]) == 1.0
    assert pytest.approx(uvar[1]) == 1.0
    assert pytest.approx(uvar[2]) == 1.0
    assert pytest.approx(uvar[3]) == 0.0
    assert pytest.approx(uvar[4]) == 0.0
    assert pytest.approx(uvar[5]) == 0.0

    # Internal state variables
    assert pytest.approx(uvar[6]) == p.zeta1_0
    assert pytest.approx(uvar[8]) == p.zeta2_0

    # Backstress orientation tensor beta -> Identity
    assert pytest.approx(uvar[9]) == 1.0
    assert pytest.approx(uvar[10]) == 1.0
    assert pytest.approx(uvar[11]) == 1.0

    # Temperature
    assert pytest.approx(uvar[19]) == p.theta_i

    # Kinematic memory F0, U0 -> Identity
    assert pytest.approx(uvar[21]) == 1.0
    assert pytest.approx(uvar[22]) == 1.0
    assert pytest.approx(uvar[23]) == 1.0
    assert pytest.approx(uvar[30]) == 1.0
    assert pytest.approx(uvar[31]) == 1.0
    assert pytest.approx(uvar[32]) == 1.0


def test_elastic_step_small_strain():
    """Verify small elastic strain response below yield stress."""
    p = _sample_params()
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    # Very small strain step: 1e-5 in xx
    deps = np.array([1.0e-5, -0.38e-5, -0.38e-5, 0.0, 0.0, 0.0], dtype=np.float64)
    dt = 1.0e-6

    sig_out, epsp_out, c_out = solid_update(p, sig, deps, dt=dt, extra=extra)

    # Stress should be uniaxial tension in xx
    assert sig_out[0] > 0.0
    assert abs(sig_out[1]) < 1e-4 * sig_out[0]
    assert abs(sig_out[2]) < 1e-4 * sig_out[0]

    # No viscoplastic strain accumulated below yield
    assert epsp_out == pytest.approx(0.0, abs=1e-12)
    assert c_out > 0.0


def test_viscoplastic_flow_above_yield():
    """Verify hyperbolic sine flow activation and plastic strain growth above yield."""
    p = _sample_params()
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    dt = 1.0e-3  # moderate time step

    # Explicit solver advances state from t_n to t_{n+1}; run 2 cycles to build stress and flow
    deps = np.array([0.05, -0.015, -0.015, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, epsp, c = solid_update(p, sig, deps, dt=dt, extra=extra)
    sig, epsp, c = solid_update(p, sig, deps, dt=dt, extra=extra)

    # Plasticity must be triggered on cycle 2
    assert epsp > 0.0
    assert sig[0] > p.c4  # Exceeded initial yield stress

    # Check that uvar stored in extra evolved
    uvar = extra["uvar101"]
    assert uvar[18] == epsp      # Equivalent plastic strain
    assert uvar[0] > 1.0         # F^v_11 stretched in tension


def test_pressure_sensitivity():
    """Verify alpha_p pressure dependence delays yielding in compression and accelerates in tension."""
    p = _sample_params(alpha_p=0.2)  # strong pressure sensitivity
    dt = 1.0e-3

    # Case A: Tensile step with positive hydrostatic tension (negative pressure)
    extra_ten = {}
    sig_ten = np.zeros(6)
    deps_ten = np.array([0.04, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(2):
        sig_ten, epsp_ten, _ = solid_update(p, sig_ten, deps_ten, dt=dt, extra=extra_ten)

    # Case B: Compressive step with equal magnitude shear
    extra_comp = {}
    sig_comp = np.zeros(6)
    deps_comp = np.array([-0.04, 0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(2):
        sig_comp, epsp_comp, _ = solid_update(p, sig_comp, deps_comp, dt=dt, extra=extra_comp)

    # Tension lowers yield stress -> more plastic flow than compression
    assert epsp_ten > epsp_comp


def test_adiabatic_heating():
    """Verify temperature increase under adiabatic plastic deformation."""
    p_iso = _sample_params(temp_opt=0.0, factor=0.9)  # Isothermal
    p_adiab = _sample_params(temp_opt=2.0, factor=0.9)  # Adiabatic

    dt = 1.0e-3
    deps = np.array([0.06, -0.02, -0.02, 0.0, 0.0, 0.0], dtype=np.float64)

    extra_iso = {}
    sig_iso = np.zeros(6)
    for _ in range(2):
        sig_iso, _, _ = solid_update(p_iso, sig_iso, deps, dt=dt, extra=extra_iso)
    theta_iso = extra_iso["uvar101"][19]

    extra_adiab = {}
    sig_adiab = np.zeros(6)
    for _ in range(2):
        sig_adiab, _, _ = solid_update(p_adiab, sig_adiab, deps, dt=dt, extra=extra_adiab)
    theta_adiab = extra_adiab["uvar101"][19]

    # In isothermal mode, temperature remains initial
    assert pytest.approx(theta_iso) == p_iso.theta_i
    # In adiabatic mode, plastic work converts to heat
    assert theta_adiab > p_adiab.theta_i


def test_backstress_locking_singularity():
    """Verify non-affine network locking singularity as trace(beta) approaches lambda_L."""
    p = _sample_params(lambda_l=2.0, c2=10.0)
    extra = {}
    sig = np.zeros(6)
    dt = 1.0e-3
    mu_b_vals = []

    # Apply successive tensile increments
    for _ in range(4):
        deps = np.array([0.04, -0.015, -0.015, 0.0, 0.0, 0.0], dtype=np.float64)
        sig, _, _ = solid_update(p, sig, deps, dt=dt, extra=extra)
        mu_b_vals.append(extra["uvar101"][15])

    # As stretch increases, mu_B increases beyond reference mu_R
    assert max(mu_b_vals) > p.c2


def test_solid_tangent_consistency():
    """Verify 6x6 algorithmic tangent matrix is symmetric and positive-definite."""
    p = _sample_params()
    tan = solid_tangent(p, np.zeros(6))

    assert tan.shape == (6, 6)
    # Symmetry check
    assert np.allclose(tan, tan.T, rtol=1e-6, atol=1e-8)
    # Positive definiteness check
    eigvals = np.linalg.eigvalsh(tan)
    assert np.all(eigvals > 0.0)


def test_extra_shapes():
    """Verify extra_shapes returns 42-element uvar101 storage."""
    shapes = extra_shapes()
    assert "uvar101" in shapes
    assert shapes["uvar101"] == (42,)
