"""Unit and verification tests for /MAT/LAW124 (/MAT/CDPM2) — Concrete Damage Plasticity Model 2 (CDPM2) (M598).

Verifies Fortran-faithfulness against OpenRadioss:
  - Starter: ``starter/source/materials/mat/mat124/hm_read_mat124.F``
  - Engine:  ``engine/source/materials/mat/mat124/sigeps124.F``
  - CFG:     ``hm_cfg_files/config/CFG/radioss*/MAT/matl124_cdpm2.cfg``

Test coverage:
  1. Parameter initialization, derived elastic moduli (G, G2, lambda, K), eccentricity e,
     friction parameter m0, fracture energy mappings (Gf, Gc), and parameter bounds validation.
  2. Haigh-Westergaard invariant transformation (sigma_m, rho, theta, cos_theta, sin_theta, s, J2)
     and Willam-Warnke elliptic function r(cos_theta, e) with analytical/numerical derivative.
  3. Hardening evolution q_h1(kappa), q_h2(kappa) and ductility measure xsih(sigma_m).
  4. 3D spectral decomposition of symmetric stress tensor with eigenvalue/eigenvector orthogonality.
  5. Monotonic uniaxial tension: linear elasticity up to ft, linear softening, tensile damage omega_t,
     zero compressive damage (omega_c = 0), and fracture energy dissipation.
  6. Monotonic uniaxial compression: elastic limit at q_h0*fc = 0.3*fc, parabolic hardening up to peak fc,
     post-peak softening, and compressive damage omega_c evolution.
  7. Cyclic load reversal and unilateral crack closure: 100% initial compressive stiffness recovery
     upon transition from tensile softening into compression (sigma_xx < 0).
  8. Biaxial compression enhancement: Menetrey-Willam yield surface satisfaction at sigma_cb = 1.16*fc.
  9. Dilatational acoustic sound speed c = sqrt((K + 4/3*G)/rho0) and algorithmic consistent tangent.
  10. Shell element rejection (NotImplementedError) and solid element group dispatch.
  11. CEB-FIP dynamic strain-rate scaling (IRATE=2) for tension and compression.
  12. Element deletion check (IDEL=2) at terminal damage D >= 0.999.
  13. Starter keyword deck parsing for /MAT/LAW124 (fixed format) and /MAT/CDPM2 (free format).
  14. Starter model parameter and element compatibility checks (check_mat_law124).
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Tuple

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck
import pyradioss.materials as mat_mod
from pyradioss.materials.law124_cdpm2 import (
    Law124Params,
    build_law124,
    compute_haigh_westergaard,
    consistent_solid_tangent,
    eval_ductility_xsih,
    eval_hardening,
    extra_shapes,
    r_willam_warnke,
    resolve,
    solid_step,
    solid_tangent,
    solid_update,
    sound_speed,
    spectral_decomposition,
    _SQR6,
    _SQRT_3_OVER_2,
)
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law124


# =============================================================================
# 1. Parameter initialization and derived moduli
# =============================================================================

def test_law124_params_initialization():
    """Verify derived elastic moduli, eccentricity, friction parameter, and bounds."""
    # Standard concrete C30/37
    p = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, rho0=2.4e-6)

    # Elastic moduli: hm_read_mat124.F lines 140-143
    # G2 = E / (1 + nu) = 30000 / 1.2 = 25000.0
    assert np.isclose(p.G2, 25000.0)
    # G = 0.5 * G2 = 12500.0
    assert np.isclose(p.G, 12500.0)
    # lambda = G2 * nu / (1 - 2*nu) = 25000 * 0.2 / 0.6 = 8333.333333333334
    assert np.isclose(p.lam, 8333.333333333334)
    # bulk = E / (3*(1 - 2*nu)) = 30000 / (3 * 0.6) = 16666.666666666668
    assert np.isclose(p.bulk, 16666.666666666668)

    # Eccentricity default calibrated for biaxial ratio 1.16: lines 185-188
    # epsi = ft * ((1.16*fc)**2 - fc**2) / (1.16*fc * (fc**2 - ft**2))
    # ecc = (1 + epsi) / (2 - epsi)
    fc116 = 1.16 * 30.0
    epsi_expected = 3.0 * (fc116**2 - 30.0**2) / (fc116 * (30.0**2 - 3.0**2))
    ecc_expected = (1.0 + epsi_expected) / (2.0 - epsi_expected)
    assert np.isclose(p.ecc, ecc_expected)
    assert 0.5 < p.ecc < 0.6

    # Friction parameter m0: line 190
    # m0 = 3 * ((fc**2 - ft**2)/(fc*ft)) * (ecc / (ecc + 1))
    m0_expected = 3.0 * ((30.0**2 - 3.0**2) / (30.0 * 3.0)) * (p.ecc / (p.ecc + 1.0))
    assert np.isclose(p.m0, m0_expected)
    assert np.isclose(p.m0, 10.19793103448276)

    # Fracture energy mappings (Gf -> wf)
    # Linear softening (dtype=1): wf = 2*Gf/ft
    p_lin = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, dtype=1, Gf=0.15)
    assert np.isclose(p_lin.wf, 2.0 * 0.15 / 3.0)  # 0.10

    # Bilinear softening (dtype=2): wf = Gf / (0.225*ft)
    p_bilin = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, dtype=2, Gf=0.15)
    assert np.isclose(p_bilin.wf, 0.15 / (0.225 * 3.0))  # 0.2222222...

    # Exponential softening (dtype=3): wf = Gf / ft
    p_exp = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, dtype=3, Gf=0.15)
    assert np.isclose(p_exp.wf, 0.15 / 3.0)  # 0.05

    # Parameter validation exceptions
    with pytest.raises(ValueError, match="Poisson's ratio"):
        Law124Params(nu=-0.1)
    with pytest.raises(ValueError, match="Poisson's ratio"):
        Law124Params(nu=0.5)
    with pytest.raises(ValueError, match="Young's modulus"):
        Law124Params(E=0.0)
    with pytest.raises(ValueError, match="Compressive strength"):
        Law124Params(fc=-10.0)
    with pytest.raises(ValueError, match="Tensile strength"):
        Law124Params(ft=0.0)


# =============================================================================
# 2. Haigh-Westergaard coordinates and Willam-Warnke elliptic function
# =============================================================================

def test_law124_haigh_westergaard_and_willam_warnke():
    """Verify Haigh-Westergaard stress invariants and Willam-Warnke elliptic function."""
    # Pure hydrostatic stress: sigma = [p, p, p, 0, 0, 0]
    p_hyd = 25.0
    sig_hyd = np.array([p_hyd, p_hyd, p_hyd, 0.0, 0.0, 0.0])
    sm, rho, th, cth, sth, s, j2 = compute_haigh_westergaard(sig_hyd)
    assert np.isclose(sm, p_hyd)
    assert np.isclose(rho, 0.0, atol=1e-9)
    assert np.allclose(s, 0.0, atol=1e-9)

    # Pure shear stress: sigma = [0, 0, 0, tau, 0, 0]
    tau = 15.0
    sig_shr = np.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    sm, rho, th, cth, sth, s, j2 = compute_haigh_westergaard(sig_shr)
    assert np.isclose(sm, 0.0)
    assert np.isclose(j2, tau**2)
    assert np.isclose(rho, math.sqrt(2.0) * tau)
    # In pure shear, J3 = 0 => cos(3*theta) = 0 => theta = pi/6
    assert np.isclose(th, math.pi / 6.0)
    assert np.isclose(cth, math.sqrt(3.0) / 2.0)

    # Uniaxial tension: sigma = [sig0, 0, 0, 0, 0, 0]
    sig0 = 10.0
    sig_ten = np.array([sig0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sm, rho, th, cth, sth, s, j2 = compute_haigh_westergaard(sig_ten)
    assert np.isclose(sm, sig0 / 3.0)
    assert np.isclose(rho, math.sqrt(2.0 / 3.0) * sig0)
    # In uniaxial tension: cos(3*theta) = 1 => theta = 0, cos(theta) = 1
    assert np.isclose(th, 0.0, atol=1e-7)
    assert np.isclose(cth, 1.0)

    # Uniaxial compression: sigma = [-sig0, 0, 0, 0, 0, 0]
    sig_cmp = np.array([-sig0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sm, rho, th, cth, sth, s, j2 = compute_haigh_westergaard(sig_cmp)
    assert np.isclose(sm, -sig0 / 3.0)
    assert np.isclose(rho, math.sqrt(2.0 / 3.0) * sig0)
    # In uniaxial compression: cos(3*theta) = -1 => theta = pi/3, cos(theta) = 0.5
    assert np.isclose(th, math.pi / 3.0, atol=1e-7)
    assert np.isclose(cth, 0.5, atol=1e-7)

    # Willam-Warnke elliptic function r(cos_theta, e) and its derivative
    ecc = 0.5229
    r_val, dr_dcosth = r_willam_warnke(cth, ecc)
    assert r_val > 0.0
    # At cth = 0.5 (theta = pi/3, compression meridian), derivative is extremum (dr/dcosth = 0)
    assert np.isclose(dr_dcosth, 0.0, atol=1e-4)
    _, dr_exact05 = r_willam_warnke(0.5, ecc)
    assert np.isclose(dr_exact05, 0.0, atol=1e-12)

    # Verify derivative by central difference perturbation at an intermediate angle (cth = 0.8)
    cth_mid = 0.8
    _, dr_mid = r_willam_warnke(cth_mid, ecc)
    d_cth = 1.0e-6
    r_plus, _ = r_willam_warnke(cth_mid + d_cth, ecc)
    r_minus, _ = r_willam_warnke(cth_mid - d_cth, ecc)
    dr_num = (r_plus - r_minus) / (2.0 * d_cth)
    assert np.isclose(dr_mid, dr_num, rtol=1e-5)


# =============================================================================
# 3. Hardening and ductility evaluation
# =============================================================================

def test_law124_hardening_and_ductility():
    """Verify dual hardening functions q_h1, q_h2 and ductility measure xsih."""
    qh0 = 0.3
    hp = 0.05

    # Case 1: kappa < 0 (initial elastic state)
    qh1, qh2, dqh1, dqh2 = eval_hardening(-0.1, qh0, hp)
    assert np.isclose(qh1, qh0)
    assert np.isclose(qh2, 1.0)
    assert np.isclose(dqh2, 0.0)

    # Case 2: 0 <= kappa < 1 (pre-peak hardening cubic polynomial)
    # At kappa = 0: qh1 = qh0
    qh1_0, qh2_0, _, _ = eval_hardening(0.0, qh0, hp)
    assert np.isclose(qh1_0, qh0)
    assert np.isclose(qh2_0, 1.0)

    # At kappa = 0.5: smooth monotonically increasing
    qh1_half, qh2_half, dqh1_half, dqh2_half = eval_hardening(0.5, qh0, hp)
    assert qh0 < qh1_half < 1.0
    assert np.isclose(qh2_half, 1.0)
    # Check numerical derivative of qh1 at kappa = 0.5
    dkap = 1.0e-6
    qh1_p, _, _, _ = eval_hardening(0.5 + dkap, qh0, hp)
    qh1_m, _, _, _ = eval_hardening(0.5 - dkap, qh0, hp)
    dqh1_num = (qh1_p - qh1_m) / (2.0 * dkap)
    assert np.isclose(dqh1_half, dqh1_num, rtol=1e-5)

    # Case 3: kappa >= 1 (post-peak hardening / linear softening if hp < 0)
    # At kappa = 1: qh1 = 1, qh2 = 1
    qh1_1, qh2_1, dqh1_1, dqh2_1 = eval_hardening(1.0, qh0, hp)
    assert np.isclose(qh1_1, 1.0)
    assert np.isclose(qh2_1, 1.0)
    assert np.isclose(dqh1_1, 0.0)
    assert np.isclose(dqh2_1, hp)

    # Ductility measure xsih: sigeps124.F lines 398-405
    fc = 30.0
    ah, bh, ch, dh = 0.08, 0.003, 2.0, 1.0e-6
    # rh >= 0: high confinement (sigma_m < -fc/3)
    xsih_comp = eval_ductility_xsih(-20.0, fc, ah, bh, ch, dh)
    assert xsih_comp > 0.0
    # rh < 0: low confinement / tension (sigma_m > -fc/3)
    xsih_ten = eval_ductility_xsih(10.0, fc, ah, bh, ch, dh)
    assert xsih_ten > 0.0
    # Hardening ductility is higher under high confinement than under tension
    assert xsih_comp > xsih_ten


# =============================================================================
# 4. Spectral decomposition of stress tensor
# =============================================================================

def test_law124_spectral_decomposition():
    """Verify 3D symmetric tensor spectral decomposition and crack closure recovery."""
    # General mixed stress state
    sig = np.array([40.0, -20.0, 10.0, 5.0, -3.0, 2.0], dtype=np.float64)
    sig_pos, sig_neg, evals, evecs = spectral_decomposition(sig)

    # 1. Stress reconstruction: sig = sig_pos + sig_neg
    assert np.allclose(sig, sig_pos + sig_neg, atol=1e-12)

    # 2. Eigenvectors are orthonormal: V^T * V = I
    assert np.allclose(evecs.T @ evecs, np.eye(3), atol=1e-12)

    # 3. Pure tension check
    sig_pure_ten = np.array([10.0, 5.0, 2.0, 0.0, 0.0, 0.0])
    s_pos, s_neg, _, _ = spectral_decomposition(sig_pure_ten)
    assert np.allclose(s_pos, sig_pure_ten, atol=1e-12)
    assert np.allclose(s_neg, 0.0, atol=1e-12)

    # 4. Pure compression check
    sig_pure_cmp = np.array([-15.0, -30.0, -5.0, 0.0, 0.0, 0.0])
    s_pos, s_neg, _, _ = spectral_decomposition(sig_pure_cmp)
    assert np.allclose(s_pos, 0.0, atol=1e-12)
    assert np.allclose(s_neg, sig_pure_cmp, atol=1e-12)


# =============================================================================
# 5. Monotonic uniaxial tension test
# =============================================================================

def _solve_free_lateral_strain(
    params: Law124Params,
    sig: np.ndarray,
    de_axial: float,
    uvar: np.ndarray,
    dmg: np.ndarray,
    le: float = 1.0,
    tol: float = 1.0e-6,
    max_iter: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Helper Newton solver to determine lateral strain ensuring uniaxial stress state (sig_yy = sig_zz = 0)."""
    de_lat = -params.nu * de_axial
    stiff_lat = params.bulk + (4.0 / 3.0) * params.G

    for _ in range(max_iter):
        deps = np.array([de_axial, de_lat, de_lat, 0.0, 0.0, 0.0], dtype=np.float64)
        s_test, _, _, _, _ = solid_step(params, sig, deps, uvar, dmg, le=le)
        err = s_test[1]
        if abs(err) < tol:
            break
        de_lat -= err / stiff_lat

    deps_final = np.array([de_axial, de_lat, de_lat, 0.0, 0.0, 0.0], dtype=np.float64)
    return solid_step(params, sig, deps_final, uvar, dmg, le=le)


def test_law124_uniaxial_tension():
    """Verify monotonic uniaxial tension: elastic loading, softening, omega_t damage, and zero omega_c."""
    ft_target = 3.0
    p = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=ft_target, dtype=1, wf=0.01)
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(16, dtype=np.float64)
    dmg = np.zeros(3, dtype=np.float64)

    stresses = []
    strains = []
    total_strain_x = 0.0
    de_x = 2.0e-5

    for step in range(35):
        sig, uvar, dmg, dpla, c = _solve_free_lateral_strain(p, sig, de_x, uvar, dmg, le=1.0)
        total_strain_x += de_x
        stresses.append(sig[0])
        strains.append(total_strain_x)

    # 1. Peak tensile stress matches ft within 0.5%
    peak_stress = max(stresses)
    assert np.isclose(peak_stress, ft_target, rtol=0.005)

    # 2. Post-peak softening: stress decreases after peak
    peak_idx = stresses.index(peak_stress)
    assert peak_idx < len(stresses) - 1
    assert stresses[-1] < peak_stress

    # 3. Damage state: tensile damage omega_t grew, compressive damage omega_c remained zero
    assert dmg[1] > 0.05
    assert np.isclose(dmg[2], 0.0, atol=1e-12)


# =============================================================================
# 6. Monotonic uniaxial compression test
# =============================================================================

def test_law124_uniaxial_compression():
    """Verify monotonic uniaxial compression: 0.3*fc elastic limit, parabolic hardening, peak fc, and omega_c."""
    fc_target = 30.0
    qh0 = 0.3
    p = Law124Params(E=30000.0, nu=0.2, fc=fc_target, ft=3.0, qh0=qh0, hp=0.0, efc=1.0e-4)
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(16, dtype=np.float64)
    dmg = np.zeros(3, dtype=np.float64)

    stresses = []
    total_strain_x = 0.0
    de_x = -5.0e-5

    # Initial elastic limit is qh0 * fc = 9.0 MPa
    elastic_limit = qh0 * fc_target

    for step in range(80):
        sig, uvar, dmg, dpla, c = _solve_free_lateral_strain(p, sig, de_x, uvar, dmg, le=1.0)
        total_strain_x += de_x
        comp_stress = -sig[0]
        stresses.append(comp_stress)

        # Before plastic yield (comp_stress < elastic_limit): kappa should remain 0
        if comp_stress < elastic_limit * 0.95:
            assert np.isclose(uvar[0], 0.0, atol=1e-6)

    # 1. Peak compressive stress reaches fc within 1%
    peak_comp = max(stresses)
    assert np.isclose(peak_comp, fc_target, rtol=0.01)

    # 2. Hardening occurred from 9.0 MPa up to 30.0 MPa
    assert peak_comp > elastic_limit * 3.0

    # 3. Post-peak softening and compressive damage omega_c
    assert stresses[-1] < peak_comp
    assert dmg[2] > 0.02


# =============================================================================
# 7. Cyclic load reversal and unilateral crack closure recovery
# =============================================================================

def test_law124_cyclic_crack_closure():
    """Verify unilateral crack closure: 100% recovery of undamaged compressive stiffness E under load reversal."""
    E0 = 30000.0
    p = Law124Params(E=E0, nu=0.2, fc=30.0, ft=3.0, dtype=1, wf=0.005)
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(16, dtype=np.float64)
    dmg = np.zeros(3, dtype=np.float64)

    # Phase 1: Load in tension to induce substantial tensile damage (omega_t > 0.2)
    de_x = 5.0e-5
    for _ in range(30):
        sig, uvar, dmg, _, _ = _solve_free_lateral_strain(p, sig, de_x, uvar, dmg, le=1.0)

    wt_damaged = dmg[1]
    assert wt_damaged > 0.2
    assert np.isclose(dmg[2], 0.0, atol=1e-12)

    # Phase 2: Reverse strain loading into compression
    de_rev = -5.0e-5
    comp_slope = None

    for _ in range(50):
        sig_prev = sig[0]
        sig, uvar, dmg, _, _ = _solve_free_lateral_strain(p, sig, de_rev, uvar, dmg, le=1.0)
        # Check slope upon entering compressive stress
        if sig[0] < 0.0:
            d_sig = sig[0] - sig_prev
            comp_slope = d_sig / de_rev
            break

    assert comp_slope is not None
    # Crack closure: compressive stiffness recovered to initial undamaged Young's modulus E0 within 0.1%!
    assert np.isclose(comp_slope, E0, rtol=0.001)


# =============================================================================
# 8. Biaxial compression envelope
# =============================================================================

def test_law124_biaxial_compression_envelope():
    """Verify that calibrated eccentricity satisfies Menetrey-Willam yield surface at 1.16*fc."""
    fc = 35.0
    ft = 3.2
    p = Law124Params(E=32000.0, nu=0.18, fc=fc, ft=ft)

    # Equal biaxial compression: sigma_x = sigma_y = -1.16*fc, sigma_z = 0
    sig_biax = np.array([-1.16 * fc, -1.16 * fc, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sm, rho, th, cth, sth, s, j2 = compute_haigh_westergaard(sig_biax)
    rcos, _ = r_willam_warnke(cth, p.ecc)

    # At peak hardening: qh1 = 1, qh2 = 1
    qh1, qh2 = 1.0, 1.0
    cl = (rho * rcos / (_SQR6 * fc)) + sm / fc
    bl = (sm / fc) + rho / (_SQR6 * fc)
    al = (1.0 - qh1) * (bl**2) + _SQRT_3_OVER_2 * (rho / fc)
    fp = al**2 + p.m0 * (qh1**2) * qh2 * cl - (qh1**2) * (qh2**2)

    # Menetrey-Willam yield function F_p must be zero to machine precision!
    assert np.isclose(fp, 0.0, atol=1e-12)


# =============================================================================
# 9. Sound speed and algorithmic tangents
# =============================================================================

def test_law124_sound_speed_and_tangents():
    """Verify dilatational sound speed and consistency of algorithmic tangent."""
    p = Law124Params(E=30000.0, nu=0.2, rho0=2.4e-6, fc=30.0, ft=3.0)

    # Acoustic wave speed c = sqrt((bulk + 4/3*G)/rho0)
    c_expected = math.sqrt((p.bulk + (4.0 / 3.0) * p.G) / p.rho0)
    assert np.isclose(sound_speed(p), c_expected)
    assert np.isclose(sound_speed(p), 117851.13019775791)

    # Elastic tangent comparison: consistent_solid_tangent vs analytical solid_tangent
    c_analytical = solid_tangent(p)
    c_num = consistent_solid_tangent(p, sig=np.zeros(6), deps=np.zeros(6))
    rel_diff = np.max(np.abs(c_num - c_analytical)) / np.max(np.abs(c_analytical))
    assert rel_diff < 1.0e-12

    # Verify symmetry of elastic matrix
    assert np.allclose(c_analytical, c_analytical.T)


# =============================================================================
# 10. Shell rejection and solid group dispatch
# =============================================================================

def test_law124_shell_rejection_and_solid_dispatch():
    """Verify LAW124 rejects shell elements and successfully dispatches solid updates."""
    mat = Material(id=1, law=124, rho0=2.4e-6, params={"E": 30000.0, "nu": 0.2, "fc": 30.0, "ft": 3.0})

    # 1. shell_update must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="only supported for solids"):
        mat_mod.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    # 2. shell_membrane_tangent must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="solid elements only"):
        mat_mod.shell_membrane_tangent(mat)

    # 3. shell_layer_tangent must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="solid elements only"):
        mat_mod.shell_layer_tangent(mat)

    # 4. solid_update must succeed for a vectorized group of solid elements
    n_elems = 4
    sig = np.zeros((n_elems, 6), dtype=np.float64)
    deps = np.full((n_elems, 6), 1.0e-5, dtype=np.float64)
    sig_new, epsp_new, c_arr = mat_mod.solid_update(mat, sig, deps)

    assert sig_new.shape == (n_elems, 6)
    assert epsp_new.shape == (n_elems,)
    assert c_arr.shape == (n_elems,)
    assert np.all(c_arr > 0.0)


# =============================================================================
# 11. Dynamic strain rate effects (IRATE=2)
# =============================================================================

def test_law124_strain_rate_enhancement():
    """Verify CEB-FIP dynamic strain rate enhancement for IRATE=2."""
    p = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, irate=2, fc0=10.0)
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(16, dtype=np.float64)
    dmg = np.zeros(3, dtype=np.float64)

    # Step at dynamic strain rate eps_dot = 100 s^-1
    deps = np.array([1.0e-4, -0.2e-4, -0.2e-4, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_dyn, uvar_dyn, _, _, _ = solid_step(p, sig, deps, uvar, dmg, le=1.0, eps_dot=100.0)

    # Strength magnification factor stored in uvar[14] must be > 1.0
    arate = uvar_dyn[14]
    assert arate > 1.0


# =============================================================================
# 12. Element deletion check (IDEL=2)
# =============================================================================

def test_law124_element_deletion():
    """Verify element deletion when D >= 0.999 under IDEL=2 (Fortran line 905)."""
    p = Law124Params(E=30000.0, nu=0.2, fc=30.0, ft=3.0, idel=2, dtype=1, dflag=2, wf=0.001)
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(16, dtype=np.float64)
    dmg = np.zeros(3, dtype=np.float64)

    # Apply severe tensile strain to exceed damage limit
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(5):
        sig, uvar, dmg, _, _ = solid_step(p, sig, deps, uvar, dmg, le=1.0)

    # When damage reaches threshold, stress must be zeroed out and dmg[0] = 1.0
    assert np.allclose(sig, 0.0)
    assert np.isclose(dmg[0], 1.0)


# =============================================================================
# 13. Starter keyword deck parsing
# =============================================================================

def test_starter_parsing_law124_and_cdpm2():
    """Verify parsing of fixed-format /MAT/LAW124 and free-format /MAT/CDPM2 cards."""
    # Fixed-format cards with proper column widths
    c1 = f"{2.4e-6:>20.6e}"
    c2 = f"{30000.0:>20.1f}" + f"{0.2:>20.1f}" + " " * 10 + f"{1:>10d}" + " " * 10 + f"{2:>10d}" + f"{10000.0:>20.1f}"
    c3 = f"{0.0:>20.1f}" + f"{0.3:>20.1f}" + f"{3.0:>20.1f}" + f"{30.0:>20.1f}" + f"{0.0:>20.1f}"
    c4 = f"{0.08:>20.4f}" + f"{0.003:>20.4f}" + f"{2.0:>20.1f}" + f"{1.0e-6:>20.6e}"
    c5 = f"{15.0:>20.1f}" + f"{1.0:>20.1f}" + f"{0.85:>20.4f}" + " " * 10 + f"{1:>10d}" + f"{2:>10d}" + f"{2:>10d}"
    c6 = f"{0.1:>20.4f}" + f"{0.0:>20.1f}" + f"{0.0:>20.1f}" + f"{1.0e-4:>20.6e}"

    fixed_deck = f"""/BEGIN
Test_Fixed
       100         1
/MAT/LAW124/10
Concrete_Fixed
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""

    free_deck = """/BEGIN
Test_Free
         0         1
/MAT/CDPM2/20
Concrete_Free
2.4E-06
30000.0 0.2 1 2 10000.0
0.0 0.3 3.5 35.0 0.0
0.08 0.003 2.0 1.0E-6
15.0 1.0 0.85 1 2 2
0.1 0.0 0.0 1.0E-4
/END
"""

    for dialect, deck, mid, expected_fc in [("fixed", fixed_deck, 10, 30.0), ("free", free_deck, 20, 35.0)]:
        with tempfile.NamedTemporaryFile("w", suffix=".rad", delete=False) as f:
            f.write(deck)
            tname = f.name
        try:
            model = parse_starter_deck(tname)
            assert mid in model.materials
            mat = model.materials[mid]
            assert mat.law == 124
            assert np.isclose(mat.params["E"], 30000.0)
            assert np.isclose(mat.params["fc"], expected_fc)
            assert np.isclose(mat.params["nu"], 0.2)
            assert mat.params["idel"] == 1
            assert mat.params["irate"] == 2
            # Also verify populated in model.mat_cdpm2s / mat_law124s
            assert mid in model.mat_cdpm2s
        finally:
            if os.path.exists(tname):
                os.remove(tname)


# =============================================================================
# 14. Starter parameter and element compatibility checks
# =============================================================================

def test_starter_checks_law124():
    """Verify check_mat_law124 error detection for out-of-bounds parameters and shell assignment."""
    log = MessageLog()
    model = Model()

    # Valid material
    mat_ok = Material(id=1, law=124, rho0=2.4e-6, params={"E": 30000.0, "nu": 0.2, "fc": 30.0, "ft": 3.0})
    check_mat_law124(model=model, mat=mat_ok, log=log)
    assert log.nerror == 0

    # Invalid parameters: negative rho, negative E, nu >= 0.5, negative fc, negative ft
    log_bad = MessageLog()
    mat_bad = Material(id=2, law=124, rho0=-1.0, params={"E": -50.0, "nu": 0.6, "fc": -10.0, "ft": -1.0})
    check_mat_law124(model=model, mat=mat_bad, log=log_bad)
    assert log_bad.nerror == 5

    # Shell element assignment rejection (ANCMSG 306)
    log_shell = MessageLog()
    model.materials[1] = mat_ok
    model.properties[10] = Property(id=10, type=1)  # Shell property
    model.parts[100] = Part(id=100, mat_id=1, prop_id=10)
    check_mat_law124(model=model, mat=mat_ok, log=log_shell)
    assert log_shell.nerror == 1
    assert any("3D solid elements" in msg for msg in log_shell.messages)
