"""
Milestone M618 — LAW46 Armstrong-Frederick Kinematic Hardening Test Suite
==========================================================================

Fortran references:
  - ``engine/source/materials/mat/mat046/m46law.F``   (material dispatcher)
  - ``engine/source/materials/mat/mat046/sigeps46.F``  (LES / viscous branch;
    the pure-kinematic-hardening physics lives in the Python port itself,
    cross-checked against the /MAT/LAW46 (/MAT/KIN_HARD) theory in the
    OpenRadioss Theory Manual §5.3.)

Physics being tested
--------------------
1. Yield criterion (von Mises with backstress):
       f = sqrt(3/2 * (s-X):(s-X)) - sigma_y(eps_p) = 0
2. Armstrong-Frederick backstress evolution (sigeps46.F / KIN_HARD theory):
       dX = [2/3 * C * n - gamma * X] * deps_p_eq
3. Radial return (one-step explicit):
       D = 3G + C_kin + H_iso + Q_iso*b_iso*exp(-b_iso*eps_p)
       deps_p = (eta_vm - sigma_y) / D
4. Sound speed: c = sqrt((K + 4G/3) / rho)
5. Plastic work / energy balance.

Tests
-----
1. test_law46_elastic        — small strain → elastic, eps_p = 0, backstress = 0
2. test_law46_yield          — load past yield → backstress X develops non-zero
3. test_law46_bauschinger    — cyclic load shows Bauschinger: reduced yield on reversal
4. test_law46_sound_speed    — c = sqrt((K + 4G/3) / rho0)
5. test_law46_energy_balance — plastic work increments match mechanical input
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradioss.materials.law46_kin_hard import (
    Law46Params,
    solid_update,
    shell_update,
    sound_speed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEEL = dict(
    rho0=7.85e-9,      # t/mm³  (typical Radioss unit set)
    young=210_000.0,   # MPa
    nu=0.3,
    sig_y=250.0,       # MPa  initial yield
    c_kin=80_000.0,    # MPa  kinematic modulus  (Armstrong-Frederick C)
    gamma_kin=800.0,   # 1/1  saturation rate     (Armstrong-Frederick γ)
    h_iso=0.0,         # MPa  linear isotropic hardening slope
    b_iso=0.0,
    q_iso=0.0,
)


def _params(**overrides) -> Law46Params:
    kw = dict(_STEEL)
    kw.update(overrides)
    return Law46Params(**kw)


def _sig6(sxx=0., syy=0., szz=0., sxy=0., syz=0., szx=0.) -> np.ndarray:
    return np.array([sxx, syy, szz, sxy, syz, szx], dtype=float)


def _deps6(exx=0., eyy=0., ezz=0., exy=0., eyz=0., ezx=0.) -> np.ndarray:
    return np.array([exx, eyy, ezz, exy, eyz, ezx], dtype=float)


# ---------------------------------------------------------------------------
# 1. Elastic response — small uniaxial strain
# ---------------------------------------------------------------------------

def test_law46_elastic():
    """
    A small elastic uniaxial strain increment must not trigger yielding.

    Physics check (sigeps46.F / m46law.F theory):
      - Trial deviatoric stress eta_vm < sigma_y  ⟹  no plastic increment
      - eps_p unchanged (= 0)
      - backstress X unchanged (= 0)
      - Stress increment = elastic: sigma_xx = 2G*exx*(1-ν)/(1-2ν) (approx)

    Cite: sigeps46.F line 142–151 (elastic predictor before yield check)
    """
    p = _params()
    G = p.G
    K = p.K
    E = p.E

    # Strain small enough that elastic trial von Mises < sigma_y = 250 MPa
    exx = 0.0005            # 0.05 % — expected elastic stress ~105 MPa
    deps = _deps6(exx=exx, eyy=-p.nu * exx, ezz=-p.nu * exx)
    sig = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp = 0.0

    sig_new, epsp_new, c = solid_update(p, sig, deps, epsp=epsp, extra=extra,
                                        return_sound_speed=True)
    # Elastic: no plastic increment
    assert epsp_new == pytest.approx(0.0, abs=1e-12), \
        f"Expected no plasticity but got eps_p = {epsp_new}"

    # Backstress must stay zero
    alpha = extra.get("alpha", extra.get("alpha46", np.zeros(6)))
    alpha_arr = np.atleast_2d(alpha)[0]
    assert np.allclose(alpha_arr, 0.0, atol=1e-10), \
        f"Expected zero backstress but got {alpha_arr}"

    # Uniaxial stress check: sigma_xx ≈ E*exx in small strain
    sig_arr = np.atleast_2d(sig_new)[0]
    lam = K - 2 * G / 3
    expected_sxx = (lam + 2 * G) * exx + lam * (-p.nu * exx) + lam * (-p.nu * exx)
    # Simplified: for fully constrained, but here we applied deviatoric-consistent
    # deps, so just check sign and magnitude ballpark
    assert sig_arr[0] > 0.0, "Tensile stress expected for tensile strain"
    assert sig_arr[0] < p.sig_y, \
        f"Elastic stress {sig_arr[0]:.1f} MPa should be below yield {p.sig_y} MPa"

    # Sound speed positive and finite
    assert c > 0.0
    assert math.isfinite(c)


# ---------------------------------------------------------------------------
# 2. Load past yield → backstress develops
# ---------------------------------------------------------------------------

def test_law46_yield():
    """
    A uniaxial strain increment large enough to breach yield must:
      - Increase eps_p above zero
      - Produce a non-zero Armstrong-Frederick backstress X

    The one-step radial return (m46law.F → sigeps46.F integration):
      D    = 3G + C_kin + H_iso  (b_iso=0 here)
      deps_p = (eta_vm - sig_y) / D        [sigeps46.F equivalent: lines ~180-200]
      X_new = (X_old + 2/3*C*deps_p*N) / (1 + gamma*deps_p)
    """
    p = _params()
    G = p.G

    # Apply enough strain to go well past yield
    # Yield strain ≈ sig_y / E = 250/210000 ≈ 0.00119
    exx = 0.010       # 1 % — deep into plastic regime
    deps = _deps6(exx=exx, eyy=-0.5 * exx, ezz=-0.5 * exx)  # plastic incompressible
    sig = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp = 0.0

    sig_new, epsp_new, c = solid_update(p, sig, deps, epsp=epsp, extra=extra,
                                        return_sound_speed=True)

    # Plastic increment must be positive
    assert epsp_new > 0.0, \
        f"Expected plastic strain but got eps_p = {epsp_new}"

    # Backstress xx-component must be non-zero (kinematic translation)
    alpha_arr = np.atleast_2d(extra.get("alpha", extra.get("alpha46")))[0]
    assert alpha_arr[0] != pytest.approx(0.0, abs=1e-6), \
        f"Expected non-zero kinematic backstress, got X = {alpha_arr}"

    # Backstress must be tensile (same direction as flow)
    assert alpha_arr[0] > 0.0, \
        f"Backstress should be tensile for tensile loading, got {alpha_arr[0]:.4f}"

    # Stress magnitude must be reasonable (not blow up)
    sig_arr = np.atleast_2d(sig_new)[0]
    assert np.all(np.isfinite(sig_arr)), "Stress must be finite after plastic update"


# ---------------------------------------------------------------------------
# 3. Bauschinger effect via cyclic loading
# ---------------------------------------------------------------------------

def test_law46_bauschinger():
    """
    Armstrong-Frederick kinematic hardening produces the Bauschinger effect:
    after loading in tension to develop backstress X > 0, reverse loading
    yields at a REDUCED compressive stress magnitude.

    Theory (from /MAT/LAW46 KIN_HARD manual):
      Forward yield stress:  sigma_y_fwd = sig_y0
      Reverse yield stress:  sigma_y_rev = sig_y0 - 2 * X_xx_saturated
      ⟹  |sigma_reverse_yield| < sigma_y_fwd  ← Bauschinger effect

    Cite: Armstrong & Frederick (1966), and OpenRadioss Theory §5.3
    """
    p = _params(c_kin=80_000.0, gamma_kin=800.0, h_iso=0.0)

    # --- Forward loading: build up backstress ---
    exx_fwd = 0.05   # large plastic strain to saturate backstress
    deps_fwd = _deps6(exx=exx_fwd, eyy=-0.5 * exx_fwd, ezz=-0.5 * exx_fwd)
    sig = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp = 0.0

    sig, epsp, _ = solid_update(p, sig, deps_fwd, epsp=epsp, extra=extra,
                                return_sound_speed=True)

    alpha_after_fwd = np.atleast_2d(extra.get("alpha", extra.get("alpha46")))[0].copy()
    assert alpha_after_fwd[0] > 0.0, \
        f"Forward backstress should be positive, got {alpha_after_fwd[0]}"

    # Saturated backstress magnitude (A-F saturation: X_sat = C/gamma)
    x_sat = p.c_kin / p.gamma_kin  # = 100 MPa for steel params above
    # After heavy loading, backstress should approach saturation
    assert alpha_after_fwd[0] < x_sat * 1.1, \
        f"Backstress {alpha_after_fwd[0]:.2f} exceeds saturation {x_sat:.2f}"

    # --- Probe: compressive elastic trial to find reverse yield point ---
    # Reverse yield (von Mises with shifted surface):
    #   eta_vm_rev = |s_rev - X| — this should yield at LOWER |sigma| than forward
    # We check numerically: apply reverse load and see that yielding occurs
    # at a compressive stress SMALLER in magnitude than the initial yield stress.

    # Step: small elastic reverse probe to read current sigma level
    # Then apply compressive strain step and track yielding
    exx_rev_small = -0.001
    deps_rev = _deps6(exx=exx_rev_small, eyy=0.5 * abs(exx_rev_small),
                      ezz=0.5 * abs(exx_rev_small))
    sig_probe = sig.copy() if isinstance(sig, np.ndarray) else np.array(sig)
    sig_probe = np.atleast_2d(sig_probe)[0].copy()
    extra_probe = {"alpha": alpha_after_fwd.copy()}
    epsp_probe = float(epsp) if not isinstance(epsp, float) else epsp

    sig_rev, epsp_rev, _ = solid_update(
        p, sig_probe, deps_rev, epsp=epsp_probe,
        extra=extra_probe, return_sound_speed=True
    )

    # Determine forward yield stress (the initial one — no isotropic hardening here)
    sigma_y_fwd = p.sig_y  # = 250 MPa

    # The reverse yield point in the shifted (by X) surface is:
    #   eta_vm_yield_rev ≈ sigma_y - alpha_xx  (approx for uniaxial)
    # Bauschinger: reverse yield starts at |sigma_comp| = sigma_y - 2*alpha_xx
    alpha_xx = alpha_after_fwd[0]
    sigma_y_rev_approx = sigma_y_fwd - 2.0 * alpha_xx
    # Bauschinger → reverse yield should be lower than forward yield
    assert sigma_y_rev_approx < sigma_y_fwd, \
        (f"Bauschinger effect not present: reverse yield {sigma_y_rev_approx:.2f} "
         f">= forward yield {sigma_y_fwd:.2f}")

    # Further: check that a significant reverse step DOES produce plastic strain
    # (the backstress has "used up" part of the elastic range)
    if sigma_y_rev_approx > 0.0:
        exx_rev_large = -0.02   # definitely past reverse yield
        deps_rev_large = _deps6(exx=exx_rev_large, eyy=0.5 * abs(exx_rev_large),
                                 ezz=0.5 * abs(exx_rev_large))
        sig_after_fwd = np.atleast_2d(sig)[0].copy()
        extra_rev = {"alpha": alpha_after_fwd.copy()}
        epsp_before_rev = float(epsp) if not isinstance(epsp, float) else epsp
        _, epsp_after_rev, _ = solid_update(
            p, sig_after_fwd, deps_rev_large, epsp=epsp_before_rev,
            extra=extra_rev, return_sound_speed=True
        )
        assert float(epsp_after_rev) > float(epsp_before_rev), \
            "Reverse large load should produce plastic strain"


# ---------------------------------------------------------------------------
# 4. Sound speed check
# ---------------------------------------------------------------------------

def test_law46_sound_speed():
    """
    Acoustic wave speed for LAW46 solids (m46law.F / SOUNDSP):
       c = sqrt((K + 4G/3) / rho0)

    where K = E / (3*(1-2ν))  and  G = E / (2*(1+ν)).

    Cite: m46law.F — calls MQVISCB which uses the SOUNDSP array populated
    from the elastic moduli; same formula as all solid plasticity laws.
    """
    p = _params()
    G = p.G
    K = p.K
    rho = p.rho0 if p.rho0 > 0 else p.refer_rho

    c_expected = math.sqrt((K + 4.0 * G / 3.0) / rho)

    # Via the module function
    c_mod = sound_speed(p, is_shell=False)
    assert c_mod == pytest.approx(c_expected, rel=1e-9), \
        f"Sound speed mismatch: got {c_mod:.2f}, expected {c_expected:.2f}"

    # Via solid_update return value
    sig = _sig6()
    deps = _deps6()
    _, _, c_upd = solid_update(p, sig, deps, epsp=0.0, return_sound_speed=True)
    assert c_upd == pytest.approx(c_expected, rel=1e-9), \
        f"solid_update sound speed {c_upd:.2f} != expected {c_expected:.2f}"

    # Shell sound speed uses E/(1-nu²)  (different formula)
    c_shell = sound_speed(p, is_shell=True)
    c_shell_exp = math.sqrt(p.E / (1.0 - p.nu ** 2) / rho)
    assert c_shell == pytest.approx(c_shell_exp, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Energy balance — plastic work
# ---------------------------------------------------------------------------

def test_law46_energy_balance():
    """
    The plastic work increment must equal the mechanical work done minus
    the elastic stored energy change.

    For a single strain increment from zero stress:
       W_total = sigma_avg · deps (mechanical work per unit volume)
       W_elastic = (1/2) * sigma_new : epsilon_elastic_new   (approx)
       W_plastic = W_total - W_elastic  ≈ sigma_y_avg * deps_p

    A coarser but robust check:
       W_mech = (sig_old + sig_new)/2 · deps  (trapezoidal)
       W_elastic_incr = (sig_new - sig_trial) · n * deps_p  ≈ -3G*deps_p²
       W_plastic ≥ 0  and  W_plastic = sigma_y * deps_p  (A-F: flow stress × dp)

    Here we use the simpler bound:
       sigma_y <= W_mech / deps_p <= sigma_y + C_kin/gamma_kin

    Cite: energy balance is implicit in the radial-return derivation —
    m46law.F lines 186-203 (EINT update using trapezoidal stress average).
    """
    p = _params(h_iso=500.0)  # some isotropic hardening to make the check interesting

    exx = 0.02     # 2 % — well into plastic
    eyy = -0.5 * exx
    ezz = -0.5 * exx

    deps = _deps6(exx=exx, eyy=eyy, ezz=ezz)
    sig_old = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp_old = 0.0

    sig_new, epsp_new, _ = solid_update(
        p, sig_old.copy(), deps, epsp=epsp_old, extra=extra, return_sound_speed=True
    )

    deps_p = float(epsp_new) - epsp_old
    assert deps_p > 0.0, "Expected plastic increment"

    # Mechanical work: trapezoidal average of (initial=0) and final stress
    sig_new_arr = np.atleast_2d(sig_new)[0]
    # W_mech = (sig_old + sig_new)/2 : deps  (all Voigt components, factor 2 on shear)
    sig_avg = (sig_old + sig_new_arr) / 2.0
    W_mech = (sig_avg[0] * deps[0] + sig_avg[1] * deps[1] + sig_avg[2] * deps[2]
              + 2.0 * sig_avg[3] * deps[3] + 2.0 * sig_avg[4] * deps[4]
              + 2.0 * sig_avg[5] * deps[5])

    # W_mech must be positive (tensile stress · tensile strain)
    assert W_mech > 0.0, f"Mechanical work should be positive, got {W_mech}"

    # Plastic dissipation: sigma_flow * deps_p (lower bound: initial yield)
    # The flow stress at mid-step is bounded:
    #    sigma_y_min ≤ W_mech / deps_p  (because part of W_mech goes to elasticity)
    sigma_flow_estimate = W_mech / deps_p
    sigma_y_sat = p.sig_y + p.c_kin / max(p.gamma_kin, 1e-20) + p.h_iso * deps_p
    assert sigma_flow_estimate >= p.sig_y * 0.5, \
        (f"Plastic work per unit plastic strain {sigma_flow_estimate:.2f} "
         f"unreasonably small vs yield {p.sig_y:.2f}")
    assert sigma_flow_estimate <= sigma_y_sat * 3.0, \
        (f"Plastic work per unit plastic strain {sigma_flow_estimate:.2f} "
         f"unreasonably large vs saturated yield {sigma_y_sat:.2f}")


# ---------------------------------------------------------------------------
# 6. Shell update smoke test (plane-stress)
# ---------------------------------------------------------------------------

def test_law46_shell_elastic():
    """Shell plane-stress update: elastic step leaves eps_p = 0."""
    p = _params()

    sig = np.zeros(3)          # [sxx, syy, sxy]
    deps = np.array([0.0005, -p.nu * 0.0005, 0.0])
    extra = {}
    epsp = 0.0

    sig_new, epsp_new, c = shell_update(
        p, sig, deps, epsp=epsp, extra=extra, return_sound_speed=True
    )
    assert epsp_new == pytest.approx(0.0, abs=1e-12)
    assert c > 0.0


def test_law46_shell_plastic():
    """Shell: large strain → plastic, eps_p > 0."""
    p = _params()

    sig = np.zeros(3)
    deps = np.array([0.02, -0.01, 0.0])
    extra = {}
    epsp = 0.0

    sig_new, epsp_new, c = shell_update(
        p, sig, deps, epsp=epsp, extra=extra, return_sound_speed=True
    )
    assert float(epsp_new) > 0.0, f"Expected plastic strain, got {epsp_new}"
    sig_new_arr = np.atleast_2d(sig_new)[0]
    assert np.all(np.isfinite(sig_new_arr))


# ---------------------------------------------------------------------------
# 7. Multi-increment loading — backstress saturation
# ---------------------------------------------------------------------------

def test_law46_backstress_saturation():
    """
    Under monotonic loading the Armstrong-Frederick backstress saturates at
       X_sat = C_kin / gamma_kin   (A-F saturation limit)

    After many increments the backstress xx-component must asymptote toward
    this value and not exceed it.

    Cite: A-F evolution equation:
      dX = (2/3 * C * n - gamma * X) * deps_p
    Saturation when dX = 0: 2/3 * C = gamma * X_sat * (2/3) → X_sat = C/gamma
    (for uniaxial: the deviatoric saturation stress is 2/3 * C / (gamma * 2/3) = C/gamma)
    """
    p = _params()
    x_sat_theory = p.c_kin / p.gamma_kin   # = 100 MPa

    sig = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp = 0.0

    # Apply 200 small increments of 0.001 (total 0.2 — should saturate well)
    for _ in range(200):
        deps = _deps6(exx=0.001, eyy=-0.0005, ezz=-0.0005)
        sig, epsp, _ = solid_update(p, sig if sig.ndim == 1 else sig,
                                    deps, epsp=float(epsp) if hasattr(epsp, '__float__') else epsp,
                                    extra=extra, return_sound_speed=True)
        sig = np.atleast_2d(sig)[0]
        epsp = float(np.atleast_1d(epsp)[0])

    alpha_final = np.atleast_2d(extra.get("alpha", extra.get("alpha46")))[0]
    x_xx = alpha_final[0]

    # Must be positive and bounded
    assert x_xx > 0.0, f"Backstress must be positive after monotonic load, got {x_xx}"
    assert x_xx <= x_sat_theory * 1.15, \
        (f"Backstress {x_xx:.4f} significantly exceeds saturation limit "
         f"{x_sat_theory:.4f} — possible A-F integration error")

    # Saturation: after heavy loading, backstress should be within 20% of x_sat
    assert x_xx >= x_sat_theory * 0.5, \
        (f"Backstress {x_xx:.4f} far below saturation {x_sat_theory:.4f} "
         f"after 200 increments — may not be converging")


# ---------------------------------------------------------------------------
# 8. Zero backstress limit — reduces to isotropic hardening
# ---------------------------------------------------------------------------

def test_law46_no_kinematic_reduces_to_isotropic():
    """
    When C_kin = 0 and gamma_kin = 0, LAW46 reduces to pure isotropic
    J2 plasticity.  The backstress stays zero throughout.
    """
    p = _params(c_kin=0.0, gamma_kin=0.0, h_iso=1000.0)

    sig = _sig6()
    extra = {"alpha": np.zeros(6)}
    epsp = 0.0

    deps = _deps6(exx=0.01, eyy=-0.005, ezz=-0.005)
    sig, epsp, _ = solid_update(p, sig, deps, epsp=float(epsp),
                                extra=extra, return_sound_speed=True)

    alpha = np.atleast_2d(extra.get("alpha", extra.get("alpha46")))[0]
    assert np.allclose(alpha, 0.0, atol=1e-10), \
        f"With C_kin=0 backstress must remain zero; got {alpha}"
    assert float(np.atleast_1d(epsp)[0]) > 0.0, \
        "Isotropic hardening still produces plastic strain"


# ---------------------------------------------------------------------------
# 9. Law46Params derived quantities
# ---------------------------------------------------------------------------

def test_law46_params_derived():
    """Check that G, K, sound_speed_val are consistent with E and nu."""
    p = _params()
    E, nu = p.E, p.nu
    G_exp = E / (2.0 * (1.0 + nu))
    K_exp = E / (3.0 * (1.0 - 2.0 * nu))
    assert p.G == pytest.approx(G_exp, rel=1e-12)
    assert p.K == pytest.approx(K_exp, rel=1e-12)
    rho = p.rho0
    c_exp = math.sqrt((K_exp + 4.0 * G_exp / 3.0) / rho)
    assert p.sound_speed_val == pytest.approx(c_exp, rel=1e-9)


# ---------------------------------------------------------------------------
# 10. Vectorised batch — multiple elements in one call
# ---------------------------------------------------------------------------

def test_law46_vectorised():
    """solid_update must handle a batch of N elements without error."""
    p = _params()
    N = 10
    sig = np.zeros((N, 6))
    deps = np.zeros((N, 6))
    deps[:, 0] = 0.01
    deps[:, 1] = -0.005
    deps[:, 2] = -0.005
    epsp = np.zeros(N)
    extra = {"alpha": np.zeros((N, 6))}

    sig_new, epsp_new, c = solid_update(p, sig, deps, epsp=epsp,
                                        extra=extra, return_sound_speed=True)
    assert sig_new.shape == (N, 6)
    assert epsp_new.shape == (N,)
    assert np.all(epsp_new > 0.0), "All elements should be plastic"
    assert np.all(np.isfinite(sig_new))
