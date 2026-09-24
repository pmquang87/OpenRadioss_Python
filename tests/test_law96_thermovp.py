"""Unit tests for LAW96 (Thermo-viscoplastic polymer model).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat096\\sigeps96.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law96_thermovp import (
    Law96Params,
    build_law96,
    calc_yield,
    extra_shapes,
    needs_defgrad,
    resolve,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)
from pyradioss.materials import (
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    solid_tangent as mat_solid_tangent,
)


class DummyMaterial:
    """Mock Material entity for dispatcher integration."""
    def __init__(self, law: int = 96, **kwargs):
        self.law = law
        self.law_name = f"LAW{law}"
        self.params = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_law96_params_defaults_and_derived():
    """Verify default parameters and derived elastic/acoustic constants."""
    p = Law96Params(
        young=2100.0,
        nu=0.35,
        rho0=1.2,
        tanb=0.25,
        tanp=0.15,
        sigy=40.0,
    )
    # Check derived shear modulus: G = E / (2 * (1 + nu))
    expected_g = 2100.0 / (2.0 * 1.35)
    assert math.isclose(p.g, expected_g, rel_tol=1e-6)

    # Check bulk modulus: K = E / (3 * (1 - 2*nu))
    expected_bulk = 2100.0 / (3.0 * (1.0 - 2.0 * 0.35))
    assert math.isclose(p.bulk, expected_bulk, rel_tol=1e-6)

    # Check acoustic speed: c = sqrt((K + 4/3*G) / rho0)
    expected_c = math.sqrt((expected_bulk + 4.0 / 3.0 * expected_g) / 1.2)
    assert math.isclose(p.c_solid, expected_c, rel_tol=1e-6)
    assert math.isclose(sound_speed(p), expected_c, rel_tol=1e-6)

    # Check history state variable shapes
    shapes = extra_shapes(p)
    assert "uvar96" in shapes
    assert shapes["uvar96"] == (4,)
    assert needs_defgrad(p) is False


def test_law96_calc_yield_evolution():
    """Verify hardening / softening / rubbery modulus formulation matching sigeps96.F."""
    p = Law96Params(
        sigy=30.0,
        siga=15.0,
        sigb=10.0,
        sigr=5.0,
        ra1=40.0,
        ra2=10.0,
        rb=20.0,
        rc=0.5,
    )
    # At epsp = 0, yield stress must equal initial yield sigy
    yld_0, h_0 = calc_yield(p, epsp=0.0)
    assert math.isclose(yld_0, 30.0, rel_tol=1e-6)

    # At small epsp = 0.05, hardening peak gain dominates
    yld_small, h_small = calc_yield(p, epsp=0.05)
    assert yld_small > 30.0

    # Temperature coupling: when na != 0 and temp is varied
    p_temp = Law96Params(
        sigy=30.0,
        siga=15.0,
        sigb=10.0,
        sigr=5.0,
        ra1=40.0,
        ra2=10.0,
        na=1.5,
        tref=300.0,
        jthe=1,
    )
    yld_t1, _ = calc_yield(p_temp, epsp=0.02, temp=300.0)
    yld_t2, _ = calc_yield(p_temp, epsp=0.02, temp=400.0)
    # Higher temperature changes Ra and thus modifies the hardening response
    assert yld_t1 != yld_t2


def test_law96_elastic_trial_below_yield():
    """Verify purely elastic response below the Drucker-Prager yield surface."""
    p = Law96Params(
        young=1000.0,
        nu=0.25,
        sigy=100.0,  # High yield stress
    )
    sig0 = np.zeros(6, dtype=float)
    # Small uniaxial strain increment: deps_xx = 1e-4
    deps = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])

    sig_new, epsp_new, c = solid_update(p, sig=sig0, deps=deps)

    # For uniaxial strain in isotropic elasticity:
    # sig_xx = (K + 4/3*G) * deps_xx
    c11 = p.bulk + 4.0 / 3.0 * p.g
    c12 = p.bulk - 2.0 / 3.0 * p.g
    assert math.isclose(sig_new[0], c11 * 1.0e-4, rel_tol=1e-5)
    assert math.isclose(sig_new[1], c12 * 1.0e-4, rel_tol=1e-5)
    assert math.isclose(sig_new[2], c12 * 1.0e-4, rel_tol=1e-5)
    # No plastic deformation should have accumulated
    assert float(epsp_new) == 0.0


def test_law96_elastic_only_fallback():
    """Verify elastic-only fallback mode when requested (stub / placeholder behavior)."""
    p = Law96Params(
        young=1000.0,
        nu=0.25,
        sigy=10.0,       # Low yield stress
        elastic_only=True, # Force elastic fallback
    )
    sig0 = np.zeros(6, dtype=float)
    # Large strain increment that would normally cause huge plastic yielding
    deps = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    sig_new, epsp_new, c = solid_update(p, sig=sig0, deps=deps)

    # Response should remain linear elastic because of elastic_only fallback
    c11 = p.bulk + 4.0 / 3.0 * p.g
    assert math.isclose(sig_new[0], c11 * 0.1, rel_tol=1e-5)
    assert float(epsp_new) == 0.0


def test_law96_plastic_yielding():
    """Verify plastic flow and return mapping when yield condition is exceeded."""
    p = Law96Params(
        young=1000.0,
        nu=0.30,
        sigy=20.0,
        siga=5.0,
        sigb=2.0,
        tanb=0.2,
        tanp=0.1,
        elastic_only=False,
    )
    sig0 = np.zeros(6, dtype=float)
    # Pure shear strain increment: deps_xy = 0.08
    # Elastic trial shear stress = G * deps_xy
    trial_tau = p.g * 0.08
    # For pure shear: Q_trial = sqrt(3) * trial_tau, P_trial = 0
    assert trial_tau * math.sqrt(3.0) > p.sigy

    deps = np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.0])
    extra = {"uvar96": np.zeros(4, dtype=float)}

    sig_new, epsp_new, c = solid_update(p, sig=sig0, deps=deps, extra=extra)

    # Plastic strain must have accumulated
    assert float(epsp_new) > 0.0
    # State variable UVAR should be updated
    assert extra["uvar96"][0, 0] > 0.0   # epsp_dev
    assert extra["uvar96"][0, 2] >= p.sigy # current yield stress

    # Effective stress Q must be bounded close to the yield surface
    p_final = - (sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    s_dev = sig_new[:3] + p_final
    j2 = 0.5 * np.sum(s_dev ** 2) + np.sum(sig_new[3:] ** 2)
    q_final = math.sqrt(3.0 * j2)
    current_yld, _ = calc_yield(p, float(epsp_new))
    f_surface = q_final - max(0.0, p_final * p.tanb + current_yld)
    assert abs(f_surface) < 1.0  # Converged on or very near yield surface


def test_law96_shell_update():
    """Verify 2D plane-stress shell update for LAW96."""
    p = Law96Params(
        young=2000.0,
        nu=0.3,
        sigy=50.0,
    )
    sig0 = np.zeros(3, dtype=float)
    deps = np.array([1.0e-4, 0.0, 0.0])  # in-plane deps_xx

    sig_new, epsp_new, c = shell_update(p, sig=sig0, deps=deps)

    # Plane stress elastic stiffness: E / (1 - nu^2)
    a11 = p.young / (1.0 - p.nu ** 2)
    a21 = p.nu * a11
    assert math.isclose(sig_new[0], a11 * 1.0e-4, rel_tol=1e-5)
    assert math.isclose(sig_new[1], a21 * 1.0e-4, rel_tol=1e-5)
    assert float(epsp_new) == 0.0


def test_law96_dispatcher_integration():
    """Verify LAW96 resolves correctly through central pyradioss.materials dispatcher."""
    mat = DummyMaterial(
        law=96,
        MAT_E=1500.0,
        MAT_NU=0.32,
        MAT_RHO=1.1,
        SIGY=25.0,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])

    sig_out, ep_out, c = mat_solid_update(mat, sig, deps)
    assert c is not None
    assert sig_out[0] > 0.0

    # Test tangent matrix shape
    c_tan = mat_solid_tangent(mat, sig)
    assert c_tan.shape == (1, 6, 6)
    assert c_tan[0, 0, 0] > 0.0

    # Test module-level tangent alias
    c_alias = tangent(mat)
    assert c_alias.shape == (1, 6, 6)
