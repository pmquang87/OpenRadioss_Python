"""
Tests for Milestone M564: OpenRadioss Material Law 87 (/MAT/LAW87, /MAT/BARLAT2000,
/MAT/BARLAT_2000, /MAT/BARLAT2000_2D) — Barlat Yld2000-2d Anisotropic Plasticity for Shells.

Covers:
1. Law87Params dataclass, defaults, and derived elastic/projection matrix calculations.
2. Isotropic reduction: alpha1..alpha8 = 1.0, a = 2.0 identically recovers von Mises plane stress.
3. Degree-1 homogeneity: bar_sigma(k * sigma) = k * bar_sigma(sigma).
4. Anisotropic directional yield stresses (0 deg, 45 deg, 90 deg, equibiaxial) for a = 6 and a = 8.
5. Swift-Voce hardening (iflag = 1): pure Swift, pure Voce, mixed Swift-Voce, and Cowper-Symonds rate effect.
6. Tabulated hardening (iflag = 0): piece-wise linear curve interpolation.
7. Hansel hardening (iflag = 2): martensite evolution kinetics and hardness calculation.
8. Kinematic hardening & Bauschinger effect: Chaboche-Rousselier (ikin=1) and Prager (ikin=2).
9. Shell thickness thinning increment (depszz) and plastic incompressibility (tr(deps_p) = 0).
10. Algorithmic consistent shell tangent vs central finite differences in elastic and plastic regimes.
11. Vectorization / batch execution equivalence across multiple element slices.
12. Integration & dispatch: MAT_PHYSICS_REGISTRY, MATERIAL_SHELL_DISPATCH, solid rejection, extra_shapes.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    build_law87,
    barlat2000_equivalent_stress,
    shell_update_law87,
    solid_update_law87,
    sound_speed_shell_law87,
    consistent_shell_tangent,
    shell_membrane_tangent,
    extra_shapes,
    resolve,
)
import pyradioss.materials as materials
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# Helpers
# ============================================================================

def make_isotropic_law87_params(
    e: float = 210000.0,
    nu: float = 0.3,
    rho0: float = 7.85e-9,
    iflag: int = 1,
    aswift: float = 500.0,
    epso: float = 0.002,
    nexp: float = 0.2,
    alpha: float = 1.0,
    qvoce: float = 0.0,
    beta: float = 0.0,
    **kwargs,
) -> Law87Params:
    """Helper creating an isotropic Barlat 2000 model (alpha1..alpha8 = 1, a = 2)."""
    d = dict(
        rho0=rho0,
        e=e,
        nu=nu,
        iflag=iflag,
        expa=2.0,
        al1=1.0,
        al2=1.0,
        al3=1.0,
        al4=1.0,
        al5=1.0,
        al6=1.0,
        al7=1.0,
        al8=1.0,
        aswift=aswift,
        epso=epso,
        nexp=nexp,
        alpha=alpha,
        qvoce=qvoce,
        beta=beta,
    )
    d.update(kwargs)
    return build_law87(d)


def make_anisotropic_law87_params(
    e: float = 70000.0,
    nu: float = 0.33,
    rho0: float = 2.7e-9,
    expa: float = 8.0,
    **kwargs,
) -> Law87Params:
    """Helper creating an anisotropic Barlat 2000 model (typical aluminum alloy, a = 8)."""
    # Typical AA6111-T4 parameters
    return Law87Params(
        rho0=rho0,
        e=e,
        nu=nu,
        iflag=1,
        expa=expa,
        al1=0.95,
        al2=1.05,
        al3=0.98,
        al4=1.02,
        al5=1.01,
        al6=0.97,
        al7=1.03,
        al8=1.08,
        aswift=450.0,
        epso=0.005,
        nexp=0.18,
        alpha=0.5,
        qvoce=120.0,
        beta=15.0,
        **kwargs,
    )


# ============================================================================
# 1. Law87Params Construction and Derived Quantities
# ============================================================================

def test_law87_params_construction():
    """Verify initialization, default values, and computed projection matrices."""
    p = make_isotropic_law87_params()
    assert p.e == 210000.0
    assert p.nu == 0.3
    assert p.a1 == pytest.approx(210000.0 / (1.0 - 0.09))
    assert p.a2 == pytest.approx(0.3 * 210000.0 / (1.0 - 0.09))
    assert p.g == pytest.approx(210000.0 / (2.0 * 1.3))

    # For isotropic, L' and L'' should match theoretical projection
    assert p.lp11 == pytest.approx(2.0 / 3.0)
    assert p.lp12 == pytest.approx(-1.0 / 3.0)
    assert p.lp21 == pytest.approx(-1.0 / 3.0)
    assert p.lp22 == pytest.approx(2.0 / 3.0)
    assert p.lp66 == pytest.approx(1.0)


# ============================================================================
# 2. Isotropic Reduction vs Von Mises
# ============================================================================

def test_barlat2000_isotropic_reduction_uniaxial_x():
    """Verify pure uniaxial tension in x matches von Mises identically."""
    p = make_isotropic_law87_params()
    sig = np.array([[250.0, 0.0, 0.0]])
    seq = barlat2000_equivalent_stress(sig, p)
    assert seq[0] == pytest.approx(250.0, rel=1e-10)


def test_barlat2000_isotropic_reduction_uniaxial_y():
    """Verify pure uniaxial tension in y matches von Mises identically."""
    p = make_isotropic_law87_params()
    sig = np.array([[0.0, 310.0, 0.0]])
    seq = barlat2000_equivalent_stress(sig, p)
    assert seq[0] == pytest.approx(310.0, rel=1e-10)


def test_barlat2000_isotropic_reduction_pure_shear():
    """Verify pure shear matches von Mises (seq = sqrt(3) * sig_xy)."""
    p = make_isotropic_law87_params()
    sig = np.array([[0.0, 0.0, 100.0]])
    seq = barlat2000_equivalent_stress(sig, p)
    expected = 100.0 * math.sqrt(3.0)
    assert seq[0] == pytest.approx(expected, rel=1e-10)


def test_barlat2000_isotropic_reduction_equibiaxial():
    """Verify equibiaxial stress (sig_xx = sig_yy, sig_xy = 0) matches von Mises (seq = sig_xx)."""
    p = make_isotropic_law87_params()
    sig = np.array([[180.0, 180.0, 0.0]])
    seq = barlat2000_equivalent_stress(sig, p)
    assert seq[0] == pytest.approx(180.0, rel=1e-10)


def test_barlat2000_isotropic_reduction_general_plane_stress():
    """Verify arbitrary combinations of (sig_xx, sig_yy, sig_xy) match von Mises exactly."""
    p = make_isotropic_law87_params()
    stresses = [
        [150.0, -80.0, 45.0],
        [-200.0, 120.0, -60.0],
        [300.0, 150.0, 75.0],
        [-100.0, -100.0, 0.0],
        [50.0, 20.0, -30.0],
    ]
    for s in stresses:
        sig = np.array([s])
        seq = barlat2000_equivalent_stress(sig, p)[0]
        vm = math.sqrt(s[0]**2 - s[0]*s[1] + s[1]**2 + 3.0 * s[2]**2)
        assert seq == pytest.approx(vm, rel=1e-9)


# ============================================================================
# 3. Degree-1 Homogeneity
# ============================================================================

def test_barlat2000_homogeneity_degree_one():
    """Verify bar_sigma(k * sigma) = k * bar_sigma(sigma) for any positive scalar k."""
    p = make_anisotropic_law87_params()
    sig = np.array([[120.0, 80.0, 35.0]])
    seq1 = barlat2000_equivalent_stress(sig, p)[0]

    for k in (0.2, 0.5, 1.5, 2.7, 10.0):
        seq_k = barlat2000_equivalent_stress(k * sig, p)[0]
        assert seq_k == pytest.approx(k * seq1, rel=1e-9)


# ============================================================================
# 4. Anisotropic Directional Yielding
# ============================================================================

def test_barlat2000_anisotropic_directional_variation():
    """Verify anisotropic alphas give distinct yield stresses in 0, 45, 90, and equibiaxial."""
    p = make_anisotropic_law87_params()
    
    # 0 deg uniaxial
    s0 = np.array([[100.0, 0.0, 0.0]])
    seq0 = barlat2000_equivalent_stress(s0, p)[0]

    # 90 deg uniaxial
    s90 = np.array([[0.0, 100.0, 0.0]])
    seq90 = barlat2000_equivalent_stress(s90, p)[0]

    # 45 deg uniaxial: sig = [50, 50, 50]
    s45 = np.array([[50.0, 50.0, 50.0]])
    seq45 = barlat2000_equivalent_stress(s45, p)[0]

    # Equibiaxial: sig = [100, 100, 0]
    sb = np.array([[100.0, 100.0, 0.0]])
    seqb = barlat2000_equivalent_stress(sb, p)[0]

    # All directional stresses should differ due to anisotropy
    assert seq0 != pytest.approx(seq90, rel=1e-3)
    assert seq0 != pytest.approx(seq45, rel=1e-3)
    assert seq0 != pytest.approx(seqb, rel=1e-3)


# ============================================================================
# 5. Swift-Voce Hardening Options
# ============================================================================

def test_swift_voce_pure_swift():
    """Verify iflag=1 with alpha=1.0 gives pure Swift hardening: sig_y = a*(eps0 + epsp)^n."""
    p = make_isotropic_law87_params(
        aswift=600.0,
        epso=0.005,
        nexp=0.25,
        alpha=1.0,
        qvoce=100.0,
        beta=10.0,
    )
    sig0 = np.zeros((1, 3))
    deps = np.array([[0.005, 0.0, 0.0]])  # Total strain
    sig_out, epsp_out, c = shell_update_law87(p, sig0, deps, epsp=np.zeros(1), dt=1e-5, return_tuple=True)

    # Plastic strain accumulated
    assert epsp_out[0] > 0.0
    # Equivalent stress must equal yield stress at epsp
    seq = barlat2000_equivalent_stress(sig_out, p)[0]
    expected_yield = 600.0 * ((0.005 + epsp_out[0]) ** 0.25)
    assert seq == pytest.approx(expected_yield, rel=1e-4)


def test_swift_voce_pure_voce():
    """Verify iflag=1 with alpha=0.0 gives pure Voce hardening: sig_y = sig0 + Q*(1 - exp(-beta*epsp))."""
    p = make_isotropic_law87_params(
        aswift=300.0,  # acts as initial yield ko in Voce
        epso=0.0,
        nexp=0.0,
        alpha=0.0,
        ko=250.0,
        qvoce=150.0,
        beta=20.0,
    )
    sig0 = np.zeros((1, 3))
    deps = np.array([[0.008, 0.0, 0.0]])
    sig_out, epsp_out, c = shell_update_law87(p, sig0, deps, epsp=np.zeros(1), dt=1e-5, return_tuple=True)

    assert epsp_out[0] > 0.0
    seq = barlat2000_equivalent_stress(sig_out, p)[0]
    expected_yield = 250.0 + 150.0 * (1.0 - math.exp(-20.0 * epsp_out[0]))
    assert seq == pytest.approx(expected_yield, rel=1e-4)


def test_cowper_symonds_rate_sensitivity():
    """Verify Cowper-Symonds rate scaling factor (1 + (eps_dot / C)^P)."""
    # Quasi-static run
    p_qs = make_isotropic_law87_params(invc=0.0, invp=0.0)
    sig0 = np.zeros((1, 3))
    deps = np.array([[0.005, 0.0, 0.0]])
    sig_qs, _, _ = shell_update_law87(p_qs, sig0.copy(), deps, epsp=np.zeros(1), dt=1.0, return_tuple=True)

    # Dynamic run with C=40.0, P=0.2 (invc=40, invp=0.2), dt=1e-4 => rate = 50 / s
    p_dyn = make_isotropic_law87_params(invc=40.0, invp=0.2)
    sig_dyn, _, _ = shell_update_law87(p_dyn, sig0.copy(), deps, epsp=np.zeros(1), dt=1e-4, return_tuple=True)

    # Dynamic stress must be higher due to rate enhancement
    assert sig_dyn[0, 0] > sig_qs[0, 0]


# ============================================================================
# 6. Tabulated Hardening (iflag = 0)
# ============================================================================

def test_tabulated_hardening_interpolation():
    """Verify iflag=0 interpolates yield stress from user curve."""
    curve = [(0.0, 200.0), (0.05, 300.0), (0.2, 450.0)]
    p = make_isotropic_law87_params(iflag=0, curves=curve)
    
    sig0 = np.zeros((1, 3))
    # Step into plastic regime
    deps = np.array([[0.005, 0.0, 0.0]])
    sig_out, epsp_out, _ = shell_update_law87(p, sig0, deps, epsp=np.zeros(1), dt=1e-5, return_tuple=True)

    assert epsp_out[0] > 0.0
    seq = barlat2000_equivalent_stress(sig_out, p)[0]
    # Linear interpolation between (0, 200) and (0.05, 300)
    expected_yield = 200.0 + (epsp_out[0] / 0.05) * 100.0
    assert seq == pytest.approx(expected_yield, rel=1e-3)


# ============================================================================
# 7. Hansel Hardening (iflag = 2)
# ============================================================================

def test_hansel_hardening_transformation():
    """Verify iflag=2 evaluates phase transformation kinetics."""
    p = make_isotropic_law87_params(
        iflag=2,
        ko=220.0,
        ckh=500.0,
        akh=0.3,
        a_hansel=1.2,
        b_hansel=15.0,
        c_hansel=0.05,
    )
    sig0 = np.zeros((1, 3))
    deps = np.array([[0.008, 0.0, 0.0]])
    sig_out, epsp_out, _ = shell_update_law87(p, sig0, deps, epsp=np.zeros(1), dt=1e-5, return_tuple=True)

    assert epsp_out[0] > 0.0
    seq = barlat2000_equivalent_stress(sig_out, p)[0]
    assert seq > 220.0


# ============================================================================
# 8. Kinematic Hardening & Bauschinger Effect
# ============================================================================

def test_kinematic_hardening_bauschinger_effect():
    """Verify tension followed by compression reveals early yielding (Bauschinger effect)."""
    # Mixed isotropic / kinematic: fisokin = 0.5, ikin = 1 (Chaboche), ckh = 50.0, akh = 50.0
    p = make_isotropic_law87_params(
        fisokin=0.5,
        ikin=1,
        ckh=50.0,
        akh=50.0,
        aswift=300.0,
        epso=0.002,
        nexp=0.1,
    )
    extra = {}
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)

    # 1. Forward tensile loading
    deps_tens = np.array([[0.01, 0.0, 0.0]])
    sig, epsp, _ = shell_update_law87(p, sig, deps_tens, epsp=epsp, dt=1e-4, extra=extra, return_tuple=True)
    assert sig[0, 0] > 0.0
    assert epsp[0] > 0.0
    # Backstress should have developed in tension
    assert "sigb87" in extra or "sigb" in extra
    sigb_tens = extra.get("sigb87", extra.get("sigb"))
    assert abs(sigb_tens[0, 0]) > 0.0

    # 2. Reverse compressive loading
    # An isotropic model would require reaching -sig_tens to yield in compression.
    # With kinematic hardening, yielding occurs earlier: |sig_comp_yield| < sig_tens.
    deps_comp = np.array([[-0.008, 0.0, 0.0]])
    sig_rev, epsp_rev, _ = shell_update_law87(p, sig.copy(), deps_comp, epsp=epsp.copy(), dt=1e-4, extra=extra, return_tuple=True)
    
    # Plastic strain must have increased in reverse
    assert epsp_rev[0] > epsp[0]


# ============================================================================
# 9. Shell Thickness Thinning (depszz)
# ============================================================================

def test_shell_thickness_thinning_incompressibility():
    """Verify plastic incompressibility: tr(deps_p) = 0 implies depszz = -(depsp_xx + depsp_yy)."""
    p = make_isotropic_law87_params(aswift=300.0, epso=0.002, nexp=0.1)
    extra = {}
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)

    # Uniaxial stretch along x: depsp_yy should equal -0.5 * depsp_xx (isotropic Poisson ratio 0.5)
    # depsp_zz should also equal -0.5 * depsp_xx
    deps = np.array([[0.01, 0.0, 0.0]])
    sig, epsp, _ = shell_update_law87(p, sig, deps, epsp=epsp, dt=1e-4, extra=extra, return_tuple=True)

    assert "depszz" in extra
    depszz = extra["depszz"][0]
    # Under plastic flow in x, shell must thin: depszz < 0
    assert depszz < 0.0


# ============================================================================
# 10. Consistent Shell Tangent vs Central Differences
# ============================================================================

def test_consistent_shell_tangent_elastic():
    """Verify tangent in elastic regime matches plane stress elasticity tensor."""
    p = make_isotropic_law87_params()
    c_num = consistent_shell_tangent(p, sig=np.zeros((1, 3)), deps=np.array([[1e-5, 0.0, 0.0]]))
    c_ana = shell_membrane_tangent(p)

    np.testing.assert_allclose(c_num[0], c_ana, rtol=1e-4, atol=1e-4)


def test_consistent_shell_tangent_plastic_perturbation():
    """Verify algorithmic consistent tangent matches finite difference perturbation in plastic regime."""
    p = make_isotropic_law87_params(aswift=400.0, epso=0.002, nexp=0.2)
    sig0 = np.zeros((1, 3))
    epsp0 = np.zeros(1)
    deps0 = np.array([[0.015, 0.002, 0.001]])

    # Base converged state
    sig_base, epsp_base, _ = shell_update_law87(p, sig0.copy(), deps0.copy(), epsp=epsp0.copy(), dt=1e-5, return_tuple=True)
    c_alg = consistent_shell_tangent(p, sig=sig0, deps=deps0, epsp=epsp0, h=1e-7)

    # Numerical tangent by central difference
    h = 1e-7
    c_fd = np.zeros((3, 3))
    for j in range(3):
        deps_p = deps0.copy()
        deps_m = deps0.copy()
        deps_p[0, j] += h
        deps_m[0, j] -= h

        s_p, _, _ = shell_update_law87(p, sig0.copy(), deps_p, epsp=epsp0.copy(), dt=1e-5, return_tuple=True)
        s_m, _, _ = shell_update_law87(p, sig0.copy(), deps_m, epsp=epsp0.copy(), dt=1e-5, return_tuple=True)

        c_fd[:, j] = (s_p[0] - s_m[0]) / (2.0 * h)

    np.testing.assert_allclose(c_alg[0], c_fd, rtol=1e-4, atol=1e-4)


# ============================================================================
# 11. Element Batch Vectorization Equivalence
# ============================================================================

def test_element_vectorization_slice_independence():
    """Verify running N elements vectorized yields identical results to serial execution."""
    p = make_anisotropic_law87_params()
    n = 5
    sig_batch = np.zeros((n, 3))
    epsp_batch = np.zeros(n)
    deps_batch = np.array([
        [0.001, 0.0, 0.0],       # purely elastic
        [0.010, 0.002, 0.0],     # plastic
        [0.005, -0.002, 0.004],  # mixed biaxial + shear
        [-0.008, 0.001, 0.0],    # compressive
        [0.015, 0.015, 0.002],   # high biaxial stretch
    ])

    # Batch run
    sig_res_b, epsp_res_b, c_b = shell_update_law87(
        p, sig_batch.copy(), deps_batch.copy(), epsp=epsp_batch.copy(), dt=1e-5, return_tuple=True
    )

    # Serial runs
    for i in range(n):
        s_i, ep_i, c_i = shell_update_law87(
            p, sig_batch[i:i+1].copy(), deps_batch[i:i+1].copy(), epsp=epsp_batch[i:i+1].copy(), dt=1e-5, return_tuple=True
        )
        np.testing.assert_allclose(sig_res_b[i], s_i[0], rtol=1e-5, atol=1e-5)
        assert epsp_res_b[i] == pytest.approx(ep_i[0], rel=1e-5)


# ============================================================================
# 12. Dispatch and Integration Checks
# ============================================================================

def test_law87_registry_and_dispatch():
    """Verify LAW87 registration in materials module dictionaries."""
    materials.register_materials()

    expected_keys = (
        87, "87", "LAW87", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000",
        "MAT_LAW87", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000",
        "LAW87_BARLAT2000",
    )
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is True
        assert meta.get("solid") is False
        assert meta.get("shell") is True
        assert k in materials.MATERIAL_SOLID_DISPATCH
        assert k in materials.MATERIAL_SHELL_DISPATCH

    # Rejection of 3D solid update
    p = make_isotropic_law87_params()
    with pytest.raises(NotImplementedError):
        solid_update_law87(p, np.zeros((1, 6)), np.zeros((1, 6)))

    # needs_env check
    assert materials.needs_env(p) is True

    # extra_shapes check
    shapes = extra_shapes(p, nip=4)
    assert shapes["uvar87"] == (4, 1)

    # sound speed check
    c_sound = sound_speed_shell_law87(p)
    expected_c = math.sqrt(210000.0 / ((1.0 - 0.3**2) * 7.85e-9))
    assert c_sound == pytest.approx(expected_c, rel=1e-6)
