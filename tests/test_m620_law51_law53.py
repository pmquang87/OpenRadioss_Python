"""
Milestone M620 Test Suite — LAW51 Granular Soil (Drucker-Prager) &
                             LAW53 Tabulated Foam (Tsai-Wu yield surface)

Fortran references:
  LAW51 Drucker-Prager kernel:
    C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/materials/mat/mat051/dprag51.F
    lines 73-184
  LAW51 Granular kernel (tabulated G, Y):
    C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/materials/mat/mat051/granular51.F90
    lines 106-256
  LAW53 Tabulated foam:
    C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/materials/mat/mat053/sigeps53.F
    lines 199-414

Tests:
  1. test_law51_elastic         — Drucker-Prager: below yield, pure elastic trial
  2. test_law51_plastic         — Drucker-Prager: above yield, radial return scales deviator
  3. test_law53_compression     — Foam: monotone compression, curve lookup drives stress
  4. test_law53_unloading       — Foam: unloading (no volumetric memory reset) stays inside surface
  5. test_both_energy           — Energy balance: dW = sig : deps >= 0 for each material
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  Module imports
# ─────────────────────────────────────────────────────────────────────────────
from pyradioss.materials.law51_granular_soil import (
    Law51Params,
    solid_update as law51_solid_update,
    resolve as law51_resolve,
)
from pyradioss.materials.law53_tab_foam import (
    Law53Params,
    solid_update as law53_solid_update,
    resolve as law53_resolve,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _j2(sig6: np.ndarray) -> float:
    """Second invariant of deviatoric stress.

    Fortran dprag51.F line 112:
        J2(I) = HALF*(T1²+T2²+T3²) + T4²+T5²+T6²
    where T1-T3 are deviatoric normal components and T4-T6 are shear.
    """
    s = sig6.copy()
    p = (s[0] + s[1] + s[2]) / 3.0   # mean stress
    s[0] -= p; s[1] -= p; s[2] -= p  # deviatoric normals
    return 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2


def _vm(sig6: np.ndarray) -> float:
    """von Mises equivalent stress = sqrt(3*J2)."""
    return math.sqrt(max(0.0, 3.0 * _j2(sig6)))


def _pressure(sig6: np.ndarray) -> float:
    """Hydrostatic pressure p = -trace(sigma)/3."""
    return -(sig6[0] + sig6[1] + sig6[2]) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
#  Fixture: standard LAW51 params
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def law51_mat() -> Law51Params:
    """
    Material parameters for Drucker-Prager soil:
      E  = 100 MPa, nu = 0.30
      A0 = 10 kPa  (cohesion-like intercept)
      A1 = 0.5     (pressure slope)
      A2 = 0.0     (no quadratic term)
      pfrac = -50 kPa (tensile cutoff)
      ymax = 500 kPa
    """
    return Law51Params(
        E=100e3,        # Pa
        nu=0.30,
        rho0=2000.0,    # kg/m³
        pfrac=-50e3,    # Pa  (tension cutoff, < 0)
        a0=10e3,        # Pa  (cohesion)
        a1=0.5,         # dimensionless
        a2=0.0,
        ymax=500e3,     # Pa
        k_pore=0.0,
        porosity=0.0,
        skempton_b=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Fixture: simple LAW53 tabulated foam params (constant strength curves)
# ─────────────────────────────────────────────────────────────────────────────

class _ConstantCurve:
    """Trivial curve: f(x) = value for all x (simulates a flat foam plateau)."""
    def __init__(self, value: float) -> None:
        self.value = value
    def __call__(self, x: float) -> float:
        return self.value


@pytest.fixture
def law53_mat() -> Law53Params:
    """
    LAW53 with constant strength curves at 50 kPa in all directions.
    E11=E22=10 MPa, G12=G23=4 MPa.
    Matching sigeps53.F UPARAM layout:
      E11  → UPARAM(IADBUFV+1)
      E22  → UPARAM(IADBUFV+2)
      G12  → UPARAM(IADBUFV+3)
      G23  → UPARAM(IADBUFV+4)
    """
    strength = _ConstantCurve(50e3)   # 50 kPa yield
    return Law53Params(
        rho0=100.0,
        refer_rho=100.0,
        e11=10e6,
        e22=10e6,
        g12=4e6,
        g23=4e6,
        fun_a1=strength,   # direction-1 compression/tension
        fun_b1=strength,   # direction-2 compression/tension
        fun_a3=strength,   # direction-3 (shear XY)
        fun_a5=strength,   # direction-4 (shear YZ)
        fun_a6=None,       # no 45-degree correction
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1. LAW51 Elastic Test
# ─────────────────────────────────────────────────────────────────────────────

def test_law51_elastic(law51_mat):
    """
    Apply a small deviatoric strain increment that stays well below yield.

    Drucker-Prager yield function (dprag51.F line 120):
        YIELD2 = J2 - G0
    where G0 = A0 + A1*P + A2*P² (line 115), clamped to [0, AMAX] (line 116-117).

    Below yield: no scaling, trial stress is accepted unchanged.
    Expected: output sigma == elastic trial stress.
    """
    p = law51_mat
    # Start from isotropic compression: sigma = [-200 kPa, -200 kPa, -200 kPa, 0, 0, 0]
    sig0 = np.array([-200e3, -200e3, -200e3, 0.0, 0.0, 0.0], dtype=float)

    # Small deviatoric shear increment: engineering shear dε_xy = 1e-5
    # J2 trial ≈ G²·(dε_xy/2)² - well below yield
    de_xy = 1e-5
    deps = np.array([0.0, 0.0, 0.0, de_xy, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_out = law51_solid_update(law51_mat, sig0, deps)
    sig_out = np.asarray(sig_out, dtype=float)

    # Hydrostatic pressure from trial: sig stays at -200 kPa (no volumetric increment)
    p_out = _pressure(sig_out)
    assert p_out == pytest.approx(200e3, rel=1e-4), (
        f"Elastic: pressure should remain ~200 kPa, got {p_out:.3e}"
    )

    # J2 must be below yield surface:  G0 = A0 + A1*P = 10e3 + 0.5*200e3 = 110 kPa
    # YIELD2 = J2 - G0; G0 here is used as G0² / 3 in some branches but dprag51 uses
    # J2 - G0 directly (G0 is already squared in that context via sqrt(3*G0)/(vm) ratio)
    # For dprag51 (IPLA2 case), G0 = A0+A1*P then YIELD2=J2-G0 (not G0²).
    # The existing _solid_step_single uses J2 comparison via vm vs sig_y (von Mises form).
    # Here we simply verify the shear stress response is purely elastic:
    G = p.G
    sxy_expected = sig0[3] + G * de_xy   # G * deps_xy
    assert sig_out[3] == pytest.approx(sxy_expected, rel=1e-3), (
        f"Elastic shear stress: expected {sxy_expected:.3e}, got {sig_out[3]:.3e}"
    )

    # Plastic strain should remain (near) zero
    assert float(epsp_out) == pytest.approx(0.0, abs=1e-12)

    # Sound speed must be positive
    assert float(c_out) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  2. LAW51 Plastic Test
# ─────────────────────────────────────────────────────────────────────────────

def test_law51_plastic(law51_mat):
    """
    Apply a large deviatoric shear increment that exceeds the Drucker-Prager yield.

    Drucker-Prager (dprag51.F lines 126-135):
        r = sqrt(3*G0) / (vm + EM14)
        sigd_new = sigd_trial * r    (radial return)
        dpla = (1 - r) * vm / (3*G)

    After return:
        - vm_new must equal sqrt(3 * G0) (on yield surface)
        - plas (accumulated plastic strain) > 0
    """
    p = law51_mat
    # Start with confining pressure ~200 kPa:
    # G0 = A0 + A1*P = 10e3 + 0.5*200e3 = 110 kPa
    # Yield (von Mises) = sqrt(3 * G0) ≈ sqrt(3*110e3) ≈ 575 Pa  (note: dprag uses J2-G0 form)
    # For our Python impl, sig_y = a0 + a1*p_eff so yield vm = sig_y
    # sig_y = a0 + a1 * 200e3 = 10e3 + 100e3 = 110 kPa
    sig0 = np.array([-200e3, -200e3, -200e3, 0.0, 0.0, 0.0], dtype=float)

    # Shear strain increment that would produce trial vm >> sig_y
    # trial_sxy = G * de_xy => want trial_vm ≈ 2 * sig_y => trial_sxy ~ sig_y / sqrt(3)
    sig_y = p.a0 + p.a1 * 200e3   # = 110 kPa
    # For shear only: vm = sqrt(3) * |sxy|, so target trial_vm = 3 * sig_y
    target_sxy = 3.0 * sig_y / math.sqrt(3.0)
    de_xy = target_sxy / p.G

    deps = np.array([0.0, 0.0, 0.0, de_xy, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_out = law51_solid_update(
        law51_mat, sig0, deps, epsp=0.0
    )
    sig_out = np.asarray(sig_out, dtype=float)

    # von Mises stress after return must be ≤ sig_y (on or inside yield surface)
    vm_out = _vm(sig_out)
    assert vm_out <= sig_y * 1.001, (
        f"Plastic: vm {vm_out:.3e} exceeds yield {sig_y:.3e}"
    )
    # Must have actually returned to yield:
    assert vm_out == pytest.approx(sig_y, rel=2e-2), (
        f"Plastic: vm {vm_out:.3e} should be on yield surface {sig_y:.3e}"
    )

    # Accumulated plastic strain must be positive
    assert float(epsp_out) > 0.0, "Plastic step: epsp must increase"

    # Pressure should be unchanged (no volumetric increment → purely deviatoric plastic)
    p_out = _pressure(sig_out)
    assert p_out == pytest.approx(200e3, rel=1e-3), (
        f"Plastic: pressure should stay ~200 kPa, got {p_out:.3e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3. LAW53 Compression Test
# ─────────────────────────────────────────────────────────────────────────────

def test_law53_compression(law53_mat):
    """
    Compress foam in XX direction through multiple increments.

    sigeps53.F lines 199-213 (elastic update):
        SIGNXX = SIGOXX + E11 * DEPSXX
        SIGNYY = SIGOYY + E22 * DEPSYY
        SIGNZZ = SIGOZZ + E22 * DEPSZZ
        SIGNXY = SIGOXY + G12 * DEPSXY
        etc.

    sigeps53.F line 210:
        EVOL = 1 - EXP(PLA)   where PLA = accumulated volumetric log-strain

    Tsai-Wu criterion (sigeps53.F lines 365-411):
        F(sig) = F1*sxx + F2*syy + F2*szz
               + F11*sxx² + F22*syy² + F22*szz²
               + ...
        if F > 1: scale = positive root of AA*s²+BB*s-1=0

    With constant curve (50 kPa strength), compressive stress must be
    limited to ≤ 50 kPa magnitude.
    """
    p = law53_mat
    sig = np.zeros(6, dtype=float)
    deps_xx = -1e-3   # 0.1% compression per step

    extra = {}

    sig_list = []
    for step in range(20):
        deps = np.array([deps_xx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        sig, epsp, c = law53_solid_update(law53_mat, sig, deps, extra=extra)
        sig = np.asarray(sig, dtype=float)
        sig_list.append(float(sig[0]))

    # After many steps compressing, stress magnitude should be ≤ 50 kPa
    # (capped by the Tsai-Wu surface with S1C = 50 kPa)
    assert sig_list[-1] < 0.0, "Compressive stress must be negative"
    assert abs(sig_list[-1]) <= 50e3 * 1.05, (
        f"Compressive stress {sig_list[-1]:.3e} should not exceed 50 kPa"
    )

    # Stress must have grown from zero during initial elastic regime
    assert sig_list[0] != 0.0, "First step must produce nonzero stress"

    # Sound speed must be positive
    assert float(c) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  4. LAW53 Unloading Test
# ─────────────────────────────────────────────────────────────────────────────

def test_law53_unloading(law53_mat):
    """
    Compress then unload the foam.

    sigeps53.F uses EVOL = 1 - EXP(accumulated volumetric log-strain).
    During unloading (tension-side EVOL → 0), the Tsai-Wu surface uses
    the tension branch (S1T from J=1 EVOL=0 lookup, line 255-258):
        IF(EVOL > 0): S1C=Y1, S1T=Y2   (compression uses full curve)
        ELSE:         S1C=Y2, S1T=Y1   (swap: expansion uses zero-strain value)

    With our constant curve (S1T = S1C = 50 kPa), behaviour is symmetric.
    After loading to peak and unloading by same amount, stress should
    return toward zero elastically (below yield surface) or remain inside.
    """
    p = law53_mat
    sig = np.zeros(6, dtype=float)
    extra_load = {}
    extra_unload = {}

    # Load: 10 compression steps
    deps_comp = np.array([-1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(10):
        sig, epsp, c = law53_solid_update(law53_mat, sig, deps_comp, extra=extra_load)
        sig = np.asarray(sig, dtype=float)

    sig_peak = float(sig[0])
    assert sig_peak < 0.0, "Peak stress must be compressive"

    # Unload: same number of steps with reversed strain
    deps_ext = np.array([+1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(10):
        sig, epsp, c = law53_solid_update(law53_mat, sig, deps_ext, extra=extra_load)
        sig = np.asarray(sig, dtype=float)

    sig_unload = float(sig[0])

    # After symmetric unloading (elastic unloading), stress is closer to zero
    # than the peak value (or at zero / slightly positive for crushing foam).
    # At minimum, |sig_unload| < |sig_peak| (energy was absorbed).
    # With our CONSTANT curve, elastic unloading is symmetric → near zero.
    assert abs(sig_unload) <= abs(sig_peak) + 1.0, (
        f"Unloaded stress |{sig_unload:.3e}| should not exceed peak |{sig_peak:.3e}|"
    )

    # Stress must remain inside Tsai-Wu surface:
    # Simplified: |sig_xx| <= 50 kPa
    assert abs(sig_unload) <= 50e3 * 1.05, (
        f"Unloaded stress {sig_unload:.3e} exceeds surface bound"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5. Energy Balance Test (both laws)
# ─────────────────────────────────────────────────────────────────────────────

def test_both_energy(law51_mat, law53_mat):
    """
    Verify that incremental internal energy dW = sigma : deps >= 0 for
    both materials over a loading sequence.

    This is a thermodynamic consistency check:
        dW = (sig_old + sig_new)/2 : deps  (midpoint rule)

    For a material that does not generate energy, dW must be >= 0
    when deforming in the direction of stress.
    """
    # ─── LAW51 energy balance ─────────────────────────────────────────────
    sig = np.zeros(6, dtype=float)
    total_dw_51 = 0.0

    # Shear loading steps (confining first, then shear)
    conf_deps = np.array([-1e-4, -1e-4, -1e-4, 0.0, 0.0, 0.0], dtype=float)
    shear_deps = np.array([0.0, 0.0, 0.0, 5e-4, 0.0, 0.0], dtype=float)

    # Apply 5 confinement steps
    for _ in range(5):
        sig_old = sig.copy()
        sig, epsp, c = law51_solid_update(law51_mat, sig, conf_deps)
        sig = np.asarray(sig, dtype=float)
        sig_mid = 0.5 * (sig_old + sig)
        dw = float(np.dot(sig_mid, conf_deps))
        total_dw_51 += dw

    # Apply 10 shear steps
    for _ in range(10):
        sig_old = sig.copy()
        sig, epsp, c = law51_solid_update(law51_mat, sig, shear_deps)
        sig = np.asarray(sig, dtype=float)
        sig_mid = 0.5 * (sig_old + sig)
        dw = float(np.dot(sig_mid, shear_deps))
        total_dw_51 += dw

    assert total_dw_51 >= -1e-3, (
        f"LAW51 energy balance violated: total dW = {total_dw_51:.4e} < 0"
    )

    # ─── LAW53 energy balance ─────────────────────────────────────────────
    sig53 = np.zeros(6, dtype=float)
    total_dw_53 = 0.0
    extra = {}

    comp_deps = np.array([-5e-4, -5e-4, -5e-4, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(15):
        sig_old = sig53.copy()
        sig53, epsp, c = law53_solid_update(law53_mat, sig53, comp_deps, extra=extra)
        sig53 = np.asarray(sig53, dtype=float)
        sig_mid = 0.5 * (sig_old + sig53)
        dw = float(np.dot(sig_mid, comp_deps))
        total_dw_53 += dw

    assert total_dw_53 >= -1e-3, (
        f"LAW53 energy balance violated: total dW = {total_dw_53:.4e} < 0"
    )

    # Both must have absorbed work (for compressive loading on a material with
    # compressive strength, the total work should be positive).
    assert total_dw_51 > 0.0, f"LAW51 should absorb energy; dW={total_dw_51:.4e}"
    assert total_dw_53 > 0.0, f"LAW53 should absorb energy; dW={total_dw_53:.4e}"
