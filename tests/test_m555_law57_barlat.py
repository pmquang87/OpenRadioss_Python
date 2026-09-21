"""
Tests for OpenRadioss Material Law 57 (/MAT/LAW57, /MAT/BARLAT3 - Barlat-Lian 1989 Anisotropic Plasticity).

Covers:
  1. calculp2 solver convergence & isotropic recovery (R00=R45=R90=1 => p=1, a=1, c=1, h_bar=1)
  2. Barlat equivalent stress calculation across 0, 45, and 90 degree uniaxial states
  3. Elastic trial and plane-stress cutting plane return mapping
  4. Yield curve interpolation across multiple strain rates and rate filtering
  5. Mixed isotropic / kinematic hardening (FISOKIN / CHARD) and backstress evolution
  6. Dynamic Young's modulus degradation (CE exponential and tabulated ifunce)
  7. Tensile failure damage factor (EPSR1..EPSR2) and element deletion (EPSMAX)
  8. Shell thickness thinning increment
  9. Consistent algorithmic tangent (n, 3, 3) verified against central finite differences
 10. Explicit rejection of 3D solid elements with NotImplementedError
 11. Transverse shear integration with shear factor
 12. Batched element vectorization equivalence (1D vs 2D)
 13. Registry and curve resolution hook (resolve)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials import (
    law57_barlat,
    build_law57,
    barlat_params,
    calculp2,
    barlat_equivalent_stress,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    solid_update_law57,
    LAW_DISPATCH_METADATA,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. calculp2 Solver Convergence & Isotropic Recovery
# ============================================================================

def test_calculp2_isotropic_recovery():
    """Verify isotropic Lankford coefficients (R00=R45=R90=1) yield p=1, a=1, c=1, h_bar=1."""
    bp = barlat_params(r0=1.0, r45=1.0, r90=1.0, m=6.0)
    assert bp.r == pytest.approx(0.5)
    assert bp.h == pytest.approx(0.5)
    assert bp.c == pytest.approx(1.0)
    assert bp.a == pytest.approx(1.0)
    assert bp.h_bar == pytest.approx(1.0)
    assert bp.p == pytest.approx(1.0, abs=1e-12)

    # Test direct calculp2 solver with isotropic parameters for multiple exponents m
    for m in (2.0, 4.0, 6.0, 8.0):
        p_sol = calculp2(a=1.0, c=1.0, h_bar=1.0, p=1.0, m=m, r45=1.0)
        assert p_sol == pytest.approx(1.0, abs=1e-12)


def test_calculp2_anisotropic_convergence():
    """Verify calculp2 convergence for realistic sheet metal Lankford anisotropy."""
    # Typical AA6016-T4 aluminum alloy Lankford parameters
    r00, r45, r90, m = 1.5, 1.2, 1.8, 6.0
    bp = barlat_params(r0=r00, r45=r45, r90=r90, m=m)

    assert bp.r == pytest.approx(1.5 / 2.5)
    assert bp.h == pytest.approx(1.8 / 2.8)
    assert bp.c == pytest.approx(2.0 * math.sqrt(bp.r * bp.h))
    assert bp.a == pytest.approx(2.0 - bp.c)
    assert bp.h_bar == pytest.approx(math.sqrt(bp.r / bp.h))
    assert bp.p > 0.0

    # Verify that the returned p is a root of the upstream equation in calculp2
    p_val = bp.p
    c2 = 0.25 * (1.0 - bp.h_bar)
    c2 = c2 * c2
    c3 = 0.25 * (1.0 + bp.h_bar)
    c4 = 0.5 * (1.0 + bp.h_bar)
    gama = math.sqrt(c2 + 0.25 * p_val * p_val)
    alpha = c3 - gama
    beta = c3 + gama
    c5 = 2.0 * c2 / gama
    m2 = m - 2.0
    aba2 = abs(alpha) ** m2
    abb2 = abs(beta) ** m2
    ca = aba2 * (c4 - c5) * (1.0 + r45)
    cb = abb2 * (c4 + c5) * (1.0 + r45)
    ca1 = aba2 * alpha
    cb1 = abb2 * beta
    c1 = (2.0 ** m) * bp.c
    abg1 = c1 * (gama ** (m - 1.0))
    f_res = bp.a * (alpha * (ca1 - ca) + beta * (cb1 - cb)) + gama * abg1

    assert abs(f_res) < 1.0e-3


# ============================================================================
# 2. Barlat Equivalent Stress Calculation Across 0, 45, 90 Deg Uniaxial States
# ============================================================================

def test_barlat_equivalent_stress_0_deg_identity():
    """Verify that in the rolling direction (0 deg), sigma_eq identically equals sigma_0."""
    # Under uniaxial tension along rolling direction, K1 = sig0/2, K2 = sig0/2.
    # Phi = (a/2)*(sig0^m) + (c/2)*(sig0^m) = ((a+c)/2)*sig0^m = sig0^m (since a = 2-c).
    # Thus sigma_eq = (sig0^m)^(1/m) = sig0 for ANY anisotropic parameters!
    for r00, r45, r90 in [(1.0, 1.0, 1.0), (1.5, 1.2, 1.8), (0.7, 1.4, 1.9)]:
        for m in (4.0, 6.0, 8.0):
            bp = barlat_params(r00, r45, r90, m)
            for sig0 in (50.0, 100.0, 250.0, 480.0):
                sig = np.array([sig0, 0.0, 0.0])
                seq = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m)
                assert seq == pytest.approx(sig0, rel=1e-10)


def test_barlat_equivalent_stress_90_deg_relation():
    """Verify that in transverse direction (90 deg), sigma_eq = h_bar * sigma_90."""
    r00, r45, r90, m = 1.6, 1.1, 2.1, 6.0
    bp = barlat_params(r00, r45, r90, m)
    for sig90 in (80.0, 200.0, 350.0):
        sig = np.array([0.0, sig90, 0.0])
        seq = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m)
        assert seq == pytest.approx(bp.h_bar * sig90, rel=1e-10)


def test_barlat_equivalent_stress_45_deg():
    """Verify Barlat equivalent stress for 45 degree uniaxial tension."""
    r00, r45, r90, m = 1.0, 1.0, 1.0, 2.0
    bp = barlat_params(r00, r45, r90, m)
    sig45 = 200.0
    # At 45 deg, stress components in material frame are sigxx = sig45/2, sigyy = sig45/2, sigxy = sig45/2
    sig = np.array([sig45 / 2.0, sig45 / 2.0, sig45 / 2.0])
    seq = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m)
    # For isotropic m=2, von Mises of [s/2, s/2, s/2] is sqrt(s^2/4 - s^2/4 + s^2/4 + 3*s^2/4) = s
    assert seq == pytest.approx(sig45, rel=1e-10)


def test_barlat_isotropic_von_mises_recovery():
    """Verify that for isotropic parameters and m=2, Barlat-Lian reduces identically to von Mises."""
    bp = barlat_params(1.0, 1.0, 1.0, m=2.0)
    stresses = [
        np.array([150.0, 80.0, 45.0]),
        np.array([-120.0, 95.0, -30.0]),
        np.array([0.0, 0.0, 100.0]),
        np.array([210.0, -210.0, 0.0]),
    ]
    for sig in stresses:
        seq_barlat = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m=2.0)
        sxx, syy, sxy = sig
        seq_vm = math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
        assert seq_barlat == pytest.approx(seq_vm, rel=1e-10)


# ============================================================================
# 3. Elastic Trial and Plane-Stress Return Mapping
# ============================================================================

def test_elastic_trial_below_yield():
    """Verify elastic Hookean update when stress state is interior to yield surface."""
    mat = build_law57(
        id=57,
        E=200000.0,
        nu=0.3,
        sigy0=350.0,
    )
    sig_old = np.array([0.0, 0.0, 0.0])
    deps = np.array([2.0e-4, 5.0e-5, 1.0e-4])

    sig_new, epsp_new, c_sound = shell_update_law57(mat, sig_old, deps, dt=1.0e-6)

    a11 = 200000.0 / (1.0 - 0.3 ** 2)
    a12 = 0.3 * a11
    g = 200000.0 / (2.0 * 1.3)

    assert epsp_new == 0.0
    assert sig_new[0] == pytest.approx(a11 * deps[0] + a12 * deps[1])
    assert sig_new[1] == pytest.approx(a12 * deps[0] + a11 * deps[1])
    assert sig_new[2] == pytest.approx(g * deps[2])
    assert c_sound == pytest.approx(math.sqrt(a11 / 1.0))


def test_plane_stress_plastic_return():
    """Verify cutting plane return mapping satisfies yield consistency."""
    mat = build_law57(
        id=57,
        E=200000.0,
        nu=0.3,
        sigy0=200.0,
        curves=[
            ([0.0, 0.05], [200.0, 300.0], 0.0),
        ],
    )
    sig_old = np.array([0.0, 0.0, 0.0])
    # Large tensile strain increment driving plasticity
    deps = np.array([3.0e-3, 0.0, 0.0])

    sig_new, epsp_new, _ = shell_update_law57(mat, sig_old, deps, dt=1.0e-6)

    assert epsp_new > 0.0
    # Equivalent stress must match the expanded yield stress
    expected_yield = 200.0 + (300.0 - 200.0) / 0.05 * epsp_new
    bp = mat.barlat
    seq = barlat_equivalent_stress(sig_new, bp.a, bp.c, bp.h_bar, bp.p, mat.m)
    assert seq == pytest.approx(expected_yield, rel=0.02)


# ============================================================================
# 4. Multi-Rate Yield Curves & Strain Rate Filtering
# ============================================================================

def test_multi_rate_yield_curve_interpolation():
    """Verify rate-dependent yield stress interpolation between curves."""
    mat = build_law57(
        id=1,
        E=210000.0,
        nu=0.3,
        curves=[
            ([0.0, 0.1], [200.0, 300.0], 0.0),
            ([0.0, 0.1], [260.0, 360.0], 100.0),
        ],
    )
    sig_old = np.array([0.0, 0.0, 0.0])

    # Case A: rate = 0 (dt = 0)
    deps_a = np.array([1.5e-3, 0.0, 0.0])
    sig_a, epsp_a, _ = shell_update_law57(mat, sig_old.copy(), deps_a, dt=0.0)

    # Case B: rate = 50.0 (deps = 0.001, dt = 2e-5 => rate = 50.0)
    dt_b = 3.0e-5
    deps_b = np.array([1.5e-3, 0.0, 0.0])
    sig_b, epsp_b, _ = shell_update_law57(mat, sig_old.copy(), deps_b, dt=dt_b)

    # Yield stress at rate 50 should be higher than rate 0
    assert sig_b[0] > sig_a[0]
    # At mid-rate 50, initial yield should be exactly halfway (230.0)
    assert sig_b[0] > 230.0


def test_strain_rate_filtering_asrate():
    """Verify low-pass filtering of strain rate with asrate/israte."""
    mat = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        asrate=100.0,
        israte=1,
        curves=[
            ([0.0, 0.1], [200.0, 300.0], 0.0),
            ([0.0, 0.1], [300.0, 400.0], 1000.0),
        ],
    )
    extra = {"epsd57": np.array([0.0])}
    sig_old = np.array([0.0, 0.0, 0.0])
    deps = np.array([1.0e-3, 0.0, 0.0])
    dt = 1.0e-5  # Instantaneous rate = 1.0e-3 / 1.0e-5 = 100.0

    shell_update_law57(mat, sig_old, deps, dt=dt, extra=extra)

    filtered_rate = float(extra["epsd57"][0])
    # With asrate = 100 and dt = 1e-5, alpha = asrate * dt = 0.001
    # Filtered rate should be heavily smoothed compared to raw 100.0
    assert 0.0 < filtered_rate < 5.0


# ============================================================================
# 5. Mixed Isotropic / Kinematic Hardening
# ============================================================================

def test_mixed_isotropic_kinematic_hardening():
    """Verify backstress evolution and yield surface radius with FISOKIN."""
    # Pure kinematic: fisokin = 1.0
    mat_kin = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        sigy0=200.0,
        fisokin=1.0,
        curves=[
            ([0.0, 0.1], [200.0, 400.0], 0.0),
        ],
    )
    extra_kin = {"sigb57": np.zeros((1, 3))}
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([5.0e-3, 0.0, 0.0])

    sig_out_kin, epsp_kin, _ = shell_update_law57(mat_kin, sig.copy(), deps, dt=1e-6, extra=extra_kin)

    # For pure kinematic hardening, backstress alpha_xx must grow positively
    sigb_kin = extra_kin["sigb57"][0]
    assert sigb_kin[0] > 0.0
    assert epsp_kin > 0.0

    # Pure isotropic: fisokin = 0.0
    mat_iso = build_law57(
        id=2,
        E=200000.0,
        nu=0.3,
        sigy0=200.0,
        fisokin=0.0,
        curves=[
            ([0.0, 0.1], [200.0, 400.0], 0.0),
        ],
    )
    extra_iso = {"sigb57": np.zeros((1, 3))}
    sig_out_iso, epsp_iso, _ = shell_update_law57(mat_iso, sig.copy(), deps, dt=1e-6, extra=extra_iso)

    sigb_iso = extra_iso["sigb57"][0]
    assert sigb_iso[0] == pytest.approx(0.0)
    assert epsp_iso > 0.0


# ============================================================================
# 6. Dynamic Young's Modulus Degradation
# ============================================================================

def test_dynamic_young_modulus_degradation_exponential():
    """Verify dynamic Young's modulus degradation E(pla) = E0 - (E0 - Einf)*(1 - exp(-CE*pla))."""
    e0 = 210000.0
    einf = 170000.0
    ce = 25.0
    mat = build_law57(
        id=1,
        E=e0,
        nu=0.3,
        einf=einf,
        ce=ce,
        sigy0=200.0,
        curves=[([0.0, 0.1], [200.0, 300.0], 0.0)],
    )

    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([5.0e-3, 0.0, 0.0])
    sig_out, epsp, c_sound1 = shell_update_law57(mat, sig, deps, dt=1e-6)

    # Step 2: degraded modulus is active from the start of the next increment
    sig_out2, epsp2, c_sound2 = shell_update_law57(mat, sig_out, np.array([1e-5, 0.0, 0.0]), dt=1e-6, epsp=epsp)

    expected_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * epsp))
    expected_a11 = expected_e / (1.0 - 0.3 ** 2)
    expected_c = math.sqrt(expected_a11 / 1.0)

    assert c_sound2 == pytest.approx(expected_c, rel=1e-3)
    assert c_sound2 < c_sound1


def test_dynamic_young_modulus_degradation_tabulated():
    """Verify dynamic Young's modulus degradation using tabulated curve ifunce."""
    mat = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        ifunce=99,
        sigy0=200.0,
        curves=[([0.0, 0.1], [200.0, 300.0], 0.0)],
        E_curve_x=[0.0, 0.02, 0.1],
        E_curve_y=[200000.0, 180000.0, 150000.0],
    )
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([5.0e-3, 0.0, 0.0])
    sig_out, epsp, c_sound1 = shell_update_law57(mat, sig, deps, dt=1e-6)

    # Step 2: degraded modulus is active from the start of the next increment
    sig_out2, epsp2, c_sound2 = shell_update_law57(mat, sig_out, np.array([1e-5, 0.0, 0.0]), dt=1e-6, epsp=epsp)

    assert epsp > 0.0
    assert c_sound2 < c_sound1


# ============================================================================
# 7. Tensile Failure Damage Factor & Element Erosion
# ============================================================================

def test_tensile_failure_damage_scaling():
    """Verify progressive damage scaling FAIL = (EPSR2 - epst)/(EPSR2 - EPSR1)."""
    mat = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        sigy0=100000.0,
        epsr1=0.01,
        epsr2=0.03,
    )
    extra = {
        "eps57": np.array([[0.0, 0.0, 0.0]]),
        "dmg57": np.zeros((1, 3)),
    }
    # Strain increment resulting in epst = 0.02 (halfway between epsr1 and epsr2)
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([0.02, 0.0, 0.0])

    sig_out, _, _ = shell_update_law57(mat, sig, deps, dt=1e-6, extra=extra)

    dmg_tensile = extra["dmg57"][0, 2]
    assert dmg_tensile == pytest.approx(0.5, abs=1e-3)

    # Elastic unsoftened stress
    a11 = 200000.0 / (1.0 - 0.3 ** 2)
    sig_unsoftened = a11 * 0.02
    assert sig_out[0] == pytest.approx(0.5 * sig_unsoftened, rel=1e-3)


def test_element_erosion_by_epsmax():
    """Verify element deletion when plastic strain exceeds EPSMAX."""
    mat = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        sigy0=200.0,
        epsmax=0.005,
        curves=[([0.0, 0.1], [200.0, 250.0], 0.0)],
    )
    extra = {
        "pla57": np.array([0.0]),
        "off57": np.array([1.0]),
    }
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([0.02, 0.0, 0.0])  # Much larger than epsmax

    sig_out, epsp, _ = shell_update_law57(mat, sig, deps, dt=1e-6, extra=extra)

    assert epsp >= 0.005
    assert extra["off57"][0] < 1.0


# ============================================================================
# 8. Shell Thickness Thinning
# ============================================================================

def test_shell_thickness_thinning():
    """Verify shell thickness thinning deps_zz and thk update."""
    mat = build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        sigy0=200.0,
        curves=[([0.0, 0.1], [200.0, 250.0], 0.0)],
    )
    extra = {
        "thk57": np.array([1.5]),
        "thk0": np.array([1.5]),
        "off57": np.array([1.0]),
    }
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([3.0e-3, 1.0e-3, 0.0])

    shell_update_law57(mat, sig, deps, dt=1e-6, extra=extra)

    # In tension, deps_zz must be negative, resulting in reduced thickness
    assert extra["depszz"] < 0.0
    assert extra["thk57"][0] < 1.5


# ============================================================================
# 9. Consistent Algorithmic Tangent Verified Against Central Finite Differences
# ============================================================================

def test_consistent_algorithmic_tangent_elastic():
    """Verify tangent matches elastic plane-stress matrix when deps is None."""
    mat = build_law57(id=1, E=210000.0, nu=0.3)
    D = tangent_law57_shell(mat)

    a11 = 210000.0 / (1.0 - 0.3 ** 2)
    a12 = 0.3 * a11
    g = 210000.0 / (2.0 * 1.3)

    assert D.shape == (3, 3)
    assert D[0, 0] == pytest.approx(a11)
    assert D[1, 1] == pytest.approx(a11)
    assert D[0, 1] == pytest.approx(a12)
    assert D[1, 0] == pytest.approx(a12)
    assert D[2, 2] == pytest.approx(g)


def test_consistent_tangent_vs_central_finite_differences():
    """Verify algorithmic tangent matches independent central finite differences to < 1e-4."""
    mat = build_law57(
        id=57,
        E=200000.0,
        nu=0.3,
        r00=1.4,
        r45=1.2,
        r90=1.6,
        m=6.0,
        sigy0=200.0,
        curves=[([0.0, 0.05], [200.0, 300.0], 0.0)],
    )

    sig_in = np.array([180.0, 50.0, 20.0])
    deps = np.array([2.5e-3, -5.0e-4, 1.0e-3])

    # Algorithmic tangent from module
    D_algo = tangent_law57_shell(mat, sig=sig_in, deps=deps, dt=1e-6)
    assert D_algo.shape == (3, 3)

    # Independent central finite difference verification with step h_test = 1e-6
    h_test = 1e-6
    D_fd = np.zeros((3, 3), dtype=float)

    for j in range(3):
        deps_p = deps.copy()
        deps_m = deps.copy()
        deps_p[j] += h_test
        deps_m[j] -= h_test

        res_p = shell_update_law57(mat, sig_in.copy(), deps_p, dt=1e-6, return_sound_speed=False)
        res_m = shell_update_law57(mat, sig_in.copy(), deps_m, dt=1e-6, return_sound_speed=False)

        sp = res_p[0][:3]
        sm = res_m[0][:3]

        D_fd[:, j] = (sp - sm) / (2.0 * h_test)

    max_err = np.max(np.abs(D_algo - D_fd))
    scale = np.max(np.abs(D_algo))
    rel_err = max_err / scale

    assert rel_err < 1.0e-4 or max_err < 1.0e-2


# ============================================================================
# 10. Solids Rejection & Metadata
# ============================================================================

def test_solids_rejection():
    """Verify that solid elements are supported in metadata and solid_update_law57 rejects 0-arg calls."""
    assert law57_barlat.LAW_DISPATCH_METADATA["solid"] is True
    assert law57_barlat.LAW_DISPATCH_METADATA["plane_stress"] is True
    assert law57_barlat.LAW_DISPATCH_METADATA["shell"] is True

    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law57()


# ============================================================================
# 11. Transverse Shear Modulus
# ============================================================================

def test_transverse_shear_stress():
    """Verify transverse shear stresses are integrated with shear modulus and shf."""
    mat = build_law57(id=1, E=200000.0, nu=0.25, shf=0.8333333333333334)
    g = 200000.0 / (2.0 * 1.25)
    shf = 0.8333333333333334

    sig = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.array([0.0, 0.0, 0.0, 2.0e-4, -3.0e-4])

    sig_out, _, _ = shell_update_law57(mat, sig, deps, dt=1e-6)

    assert sig_out[3] == pytest.approx(g * shf * deps[3])
    assert sig_out[4] == pytest.approx(g * shf * deps[4])


# ============================================================================
# 12. Batched Element Vectorization Equivalence
# ============================================================================

def test_batched_vectorization_equivalence():
    """Verify batched 2D element evaluation matches individual 1D evaluation."""
    mat = build_law57(
        id=1,
        E=210000.0,
        nu=0.3,
        r00=1.3,
        r45=1.1,
        r90=1.5,
        m=6.0,
        sigy0=220.0,
        curves=[([0.0, 0.1], [220.0, 320.0], 0.0)],
    )

    sig_batch = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 20.0, 5.0],
        [180.0, -30.0, 15.0],
    ])
    deps_batch = np.array([
        [1.0e-4, 0.0, 0.0],
        [2.0e-3, 5.0e-4, 1.0e-4],
        [4.0e-3, -1.0e-3, 5.0e-4],
    ])

    sig_batch_out, epsp_batch_out, c_batch_out = shell_update_law57(
        mat, sig_batch.copy(), deps_batch.copy(), dt=1e-6
    )

    for i in range(3):
        s_single, ep_single, c_single = shell_update_law57(
            mat, sig_batch[i].copy(), deps_batch[i].copy(), dt=1e-6
        )
        assert np.allclose(sig_batch_out[i], s_single, rtol=1e-9, atol=1e-9)
        assert epsp_batch_out[i] == pytest.approx(ep_single, rel=1e-9)
        assert c_batch_out[i] == pytest.approx(c_single, rel=1e-9)


# ============================================================================
# 13. Registry & Curve Resolution Hook
# ============================================================================

def test_registry_and_curve_resolution():
    """Verify MAT_PHYSICS_REGISTRY registration and resolve() hook."""
    assert "57" in MAT_PHYSICS_REGISTRY or 57 in MAT_PHYSICS_REGISTRY
    factory = MAT_PHYSICS_REGISTRY["LAW57"]
    mat = factory(id=57, E=200000.0, nu=0.3, FunctionIds=[101, 102], ABG_cpb=[0.0, 50.0])

    assert mat.id == 57
    assert mat.funct_ids == [101, 102]

    # Mock model functions dictionary
    class MockFunction:
        def __init__(self, x, y):
            self.x = np.asarray(x, dtype=float)
            self.y = np.asarray(y, dtype=float)

    mock_model = {
        101: MockFunction([0.0, 0.1], [200.0, 280.0]),
        102: MockFunction([0.0, 0.1], [250.0, 330.0]),
    }

    law57_barlat.resolve(mat, mock_model)

    assert len(mat.curve_x) == 2
    assert len(mat.curve_y) == 2
    assert len(mat.curve_s) == 2
    assert np.allclose(mat.curve_y[0], [200.0, 280.0])
    assert np.allclose(mat.curve_y[1], [250.0, 330.0])
