"""Unit tests for Milestone M558: /MAT/LAW79 (Johnson-Holmquist JH-2 ceramic/brittle damage model).

Verifies:
- Law79Params dataclass fields, defaults, and derived properties (shel, tstar, young, nu)
- build_law79 factory from dictionary, MatLaw79, and Material
- Factory defaults: eps0=1.0 when 0, sigfmax=1e20 when 0, epsmax=1e20 when 0
- Validation errors: phel > hel, shear <= 0, k1 <= 0, eps0 <= 0, beta not in [0, 1]
- Pure elastic shear and volumetric compression response
- Yield stress calculation: intact and fractured envelopes, pressure dependence (n, m exponents)
- Strain rate enhancement: logarithmic scaling with c and eps0
- Damage accumulation: epfail(P*, T*), plastic strain dpla, and progressive degradation
- Bulking pressure increment: internal shear energy conversion via beta
- Element deletion modes: IDEL = 0, 1 (tension), 2 (plastic strain), 3 (full damage)
- Longitudinal sound speed: compression (mu > 0 with K2, K3) and tension (mu <= 0)
- Consistent algorithmic solid tangent tensor: analytical vs numerical perturbation
- Vectorized group execution across multiple elements
- Plane-stress shell update raising NotImplementedError
- Dispatcher integration in pyradioss.materials
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material, MatLaw79
from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update,
    solid_update_law79,
    sound_speed_solid,
    consistent_solid_tangent,
    shell_update,
)
import pyradioss.materials as materials


# ---------------------------------------------------------------------------
# Test fixtures and basic parameter checks
# ---------------------------------------------------------------------------

def test_law79_params_dataclass():
    """Test Law79Params instantiation, properties, and derived physical quantities."""
    # Typical Silicon Carbide (SiC) JH-2 parameters
    p = Law79Params(
        rho0=3.21e-3,       # g/mm^3
        refer_rho=3.21e-3,
        shear=193.0,        # GPa
        a=0.96,
        b=0.35,
        m=1.0,
        n=0.65,
        c=0.009,
        eps0=1.0,
        sigfmax=0.8,
        fcut=0.0,
        t=0.37,             # T0 (GPa)
        hel=14.5,           # HEL (GPa)
        phel=5.13,          # PHEL (GPa)
        d1=0.48,
        d2=0.48,
        idel=0,
        epsmax=1e20,
        k1=220.0,           # K1 (GPa)
        k2=0.0,
        k3=0.0,
        beta=1.0,
        title="SiC-JH2",
    )

    # Derived shel = 1.5 * (hel - phel) = 1.5 * (14.5 - 5.13) = 14.055
    assert math.isclose(p.shel, 1.5 * (14.5 - 5.13), rel_tol=1e-12)
    # Derived tstar = t / phel = 0.37 / 5.13
    assert math.isclose(p.tstar, 0.37 / 5.13, rel_tol=1e-12)
    # Young's modulus E = 9 * K1 * G / (3 * K1 + G)
    expected_e = 9.0 * 220.0 * 193.0 / (3.0 * 220.0 + 193.0)
    assert math.isclose(p.young, expected_e, rel_tol=1e-12)
    assert math.isclose(p.E, expected_e, rel_tol=1e-12)
    # Poisson's ratio nu = (3 * K1 - 2 * G) / (6 * K1 + 2 * G)
    expected_nu = (3.0 * 220.0 - 2.0 * 193.0) / (6.0 * 220.0 + 2.0 * 193.0)
    assert math.isclose(p.nu, expected_nu, rel_tol=1e-12)

    # Convenience aliases
    assert p.G == 193.0
    assert p.K == 220.0
    assert p.bulk == 220.0
    assert p.rho == 3.21e-3
    assert p.t0 == 0.37
    assert p.sig0 == 0.96
    assert p.title == "SiC-JH2"


def test_build_law79_defaults_and_factories():
    """Verify build_law79 factory defaults from dict and MatLaw79."""
    mat_dict = {
        "id": 1,
        "rho": 3.2,
        "shear": 190.0,
        "a": 0.9,
        "b": 0.3,
        "m": 1.0,
        "n": 0.6,
        "t": 0.35,
        "hel": 14.0,
        "phel": 5.0,
        "k1": 210.0,
        # eps0, sigfmax, epsmax left 0 to test defaults
        "eps0": 0.0,
        "sigfmax": 0.0,
        "epsmax": 0.0,
    }

    mat = build_law79(mat_dict)
    assert isinstance(mat, Material)
    assert mat.id == 1
    assert mat.law == 79
    assert mat.rho0 == 3.2

    p = mat.params
    # Fortran defaults
    assert p["eps0"] == 1.0
    assert p["sigfmax"] == 1.0e20
    assert p["epsmax"] == 1.0e20
    assert p["refer_rho"] == 3.2
    assert math.isclose(p["shel"], 1.5 * (14.0 - 5.0), rel_tol=1e-12)
    assert math.isclose(p["tstar"], 0.35 / 5.0, rel_tol=1e-12)

    # Factory from MatLaw79 entity
    m79_entity = MatLaw79(
        id=79,
        rho=2.5,
        tau_shear=100.0,
        a=0.8,
        b=0.4,
        m=1.0,
        n=0.7,
        c=0.01,
        eps0=0.0,
        sigfmax=0.0,
        t=0.2,
        hel=10.0,
        phel=4.0,
        k1=150.0,
        epsmax=0.0,
        title="Glass-JH2",
    )
    mat2 = build_law79(m79_entity)
    assert mat2.id == 79
    assert mat2.rho0 == 2.5
    assert mat2.params["shear"] == 100.0
    assert mat2.params["eps0"] == 1.0
    assert mat2.params["sigfmax"] == 1.0e20
    assert mat2.params["epsmax"] == 1.0e20
    assert math.isclose(mat2.params["shel"], 1.5 * (10.0 - 4.0), rel_tol=1e-12)
    assert math.isclose(mat2.params["tstar"], 0.2 / 4.0, rel_tol=1e-12)


def test_build_law79_validation_errors():
    """Verify validation errors from hm_read_mat79.F."""
    base = {
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
        "eps0": 1.0,
        "beta": 0.5,
    }

    # 1. PHEL > HEL
    with pytest.raises(ValueError, match="PHEL.*cannot exceed HEL"):
        bad = dict(base, phel=12.0)
        build_law79(bad)

    # 2. SHEAR <= 0
    with pytest.raises(ValueError, match="Shear modulus must be positive"):
        bad = dict(base, shear=0.0)
        build_law79(bad)

    # 3. K1 <= 0
    with pytest.raises(ValueError, match="Bulk modulus K1 must be positive"):
        bad = dict(base, k1=-10.0)
        build_law79(bad)

    # 4. EPS0 <= 0 (when explicitly negative)
    with pytest.raises(ValueError, match="Reference strain rate eps0 must be positive"):
        bad = dict(base, eps0=-1.0)
        build_law79(bad)

    # 5. BETA not in [0, 1]
    with pytest.raises(ValueError, match="Bulking coefficient beta must be in"):
        bad = dict(base, beta=1.5)
        build_law79(bad)
    with pytest.raises(ValueError, match="Bulking coefficient beta must be in"):
        bad = dict(base, beta=-0.1)
        build_law79(bad)


# ---------------------------------------------------------------------------
# Elastic regimes
# ---------------------------------------------------------------------------

def test_elastic_pure_shear():
    """Verify small shear strain below yield strength behaves elastically."""
    mat = build_law79({
        "rho": 3.2,
        "shear": 190.0,
        "a": 1.0,
        "b": 0.5,
        "m": 1.0,
        "n": 1.0,
        "t": 0.3,
        "hel": 14.0,
        "phel": 5.0,
        "k1": 220.0,
    })
    G = mat.params["shear"]
    sig_old = np.zeros(6, dtype=float)
    # Small engineering shear strain: gamma_xy = 1e-4
    deps = np.array([0.0, 0.0, 0.0, 1e-4, 0.0, 0.0], dtype=float)

    sig_new, epsp_new, c_val = solid_update(mat, sig_old, deps, epsp=0.0, return_tuple=True)

    # s_xy = G * gamma_xy
    expected_sxy = G * 1e-4
    assert math.isclose(sig_new[3], expected_sxy, rel_tol=1e-10)
    assert np.allclose(sig_new[[0, 1, 2, 4, 5]], 0.0)
    assert epsp_new == 0.0
    assert c_val > 0.0


def test_elastic_hydrostatic_compression_and_eos():
    """Verify nonlinear EOS pressure P = K1*mu + K2*mu^2 + K3*mu^3 for mu > 0."""
    k1 = 200.0
    k2 = 150.0
    k3 = 100.0
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "hel": 10.0,
        "phel": 5.0,
        "k1": k1,
        "k2": k2,
        "k3": k3,
    })

    mu = 0.05
    # Isotropic volumetric compression: tr(deps) = -mu => deps_ii = -mu/3
    deps = np.array([-mu / 3.0, -mu / 3.0, -mu / 3.0, 0.0, 0.0, 0.0])
    sig_old = np.zeros(6, dtype=float)
    extra = {"mu": mu}

    sig_new = solid_update(mat, sig_old, deps, extra=extra)

    # Normal stress should be pure hydrostatic pressure: sig_ii = -P
    expected_p = k1 * mu + k2 * (mu ** 2) + k3 * (mu ** 3)
    for i in range(3):
        assert math.isclose(sig_new[i], -expected_p, rel_tol=1e-10)
    assert np.allclose(sig_new[3:], 0.0)


def test_elastic_tension_cutoff_pmin():
    """Verify tensile cutoff P >= -T* * PHEL * (1 - D) when IDEL != 1."""
    t0 = 0.4
    phel = 5.0
    k1 = 200.0
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "t": t0,
        "hel": 10.0,
        "phel": phel,
        "k1": k1,
        "idel": 0,
    })

    # Large volumetric expansion (mu < 0) that would exceed tensile cutoff
    mu = -0.01  # K1*mu = -2.0, but pmin = -T0 = -0.4
    extra = {"mu": mu, "dmg": 0.0}
    deps = np.zeros(6)

    sig_new = solid_update(mat, np.zeros(6), deps, extra=extra)

    # Expected: P = max(-2.0, -0.4) = -0.4 => sig_ii = -(-0.4) = +0.4
    expected_p = -t0
    for i in range(3):
        assert math.isclose(sig_new[i], -expected_p, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Yield functions and pressure dependence
# ---------------------------------------------------------------------------

def test_yield_intact_pressure_dependence():
    """Verify intact strength sigyi = A * (P* + T*)^N."""
    a = 0.95
    n_exp = 0.65
    t0 = 0.35
    phel = 5.0
    hel = 14.0
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": a,
        "n": n_exp,
        "t": t0,
        "hel": hel,
        "phel": phel,
        "k1": 200.0,
    })
    shel = 1.5 * (hel - phel)
    tstar = t0 / phel

    # Apply pressure: P = 5.0 => P* = 1.0
    p_applied = 5.0
    pstar = p_applied / phel
    extra = {"amu": p_applied / 200.0}  # mu = P / K1

    # Large shear strain to force plastic yielding
    deps = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    sig_old = np.array([-p_applied, -p_applied, -p_applied, 0.0, 0.0, 0.0])

    sig_new = solid_update(mat, sig_old, deps, extra=extra)

    # Expected intact normalized strength
    expected_sigyi = a * ((pstar + tstar) ** n_exp)
    expected_yield = expected_sigyi * shel

    # For pure shear, von Mises equivalent stress is sqrt(3)*tau_xy
    vm_stress = math.sqrt(3.0 * (sig_new[3] ** 2))
    assert math.isclose(vm_stress, expected_yield, rel_tol=1e-4)


def test_yield_fractured_pressure_dependence_and_cap():
    """Verify fractured strength sigyf = min(B * (P*)^M, sigfmax) when damaged."""
    b = 0.35
    m_exp = 1.0
    sigfmax = 0.25  # Cap
    phel = 5.0
    hel = 14.0
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "b": b,
        "m": m_exp,
        "sigfmax": sigfmax,
        "hel": hel,
        "phel": phel,
        "k1": 200.0,
    })
    shel = 1.5 * (hel - phel)

    # Case 1: P = 2.5 => P* = 0.5. B * P*^M = 0.35 * 0.5 = 0.175 < 0.25
    p1 = 2.5
    extra1 = {"amu": p1 / 200.0, "dmg": 1.0}  # Fully fractured D = 1
    deps = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    sig_new1 = solid_update(mat, np.array([-p1, -p1, -p1, 0, 0, 0], dtype=float), deps, extra=extra1)
    vm1 = math.sqrt(3.0 * (sig_new1[3] ** 2))
    expected_yield1 = 0.175 * shel
    assert math.isclose(vm1, expected_yield1, rel_tol=1e-4)

    # Case 2: P = 5.0 => P* = 1.0. B * P*^M = 0.35 > 0.25 (capped at sigfmax)
    p2 = 5.0
    extra2 = {"amu": p2 / 200.0, "dmg": 1.0}
    sig_new2 = solid_update(mat, np.array([-p2, -p2, -p2, 0, 0, 0], dtype=float), deps, extra=extra2)
    vm2 = math.sqrt(3.0 * (sig_new2[3] ** 2))
    expected_yield2 = sigfmax * shel
    assert math.isclose(vm2, expected_yield2, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Strain rate scaling
# ---------------------------------------------------------------------------

def test_strain_rate_scaling():
    """Verify strain rate factor ce = 1 + c * ln(epsd / eps0) for epsd > eps0."""
    c_rate = 0.02
    eps0 = 1.0
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "n": 0.0,   # Constant intact strength A
        "c": c_rate,
        "eps0": eps0,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
    })
    shel = 1.5 * (10.0 - 4.0)

    # Test 1: Strain rate epsd = 100 s^-1 > eps0 = 1 s^-1
    epsd = 100.0
    extra = {"epsd": epsd, "dmg": 0.0}
    deps = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    sig_new = solid_update(mat, np.zeros(6), deps, extra=extra)

    expected_ce = 1.0 + c_rate * math.log(100.0 / 1.0)
    expected_yield = expected_ce * 1.0 * shel
    vm = math.sqrt(3.0 * (sig_new[3] ** 2))
    assert math.isclose(vm, expected_yield, rel_tol=1e-4)

    # Test 2: Strain rate epsd = 0.1 s^-1 <= eps0 => ce = 1.0
    extra2 = {"epsd": 0.1, "dmg": 0.0}
    sig_new2 = solid_update(mat, np.zeros(6), deps, extra=extra2)
    vm2 = math.sqrt(3.0 * (sig_new2[3] ** 2))
    assert math.isclose(vm2, 1.0 * shel, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Damage evolution
# ---------------------------------------------------------------------------

def test_damage_evolution_and_softening():
    """Verify plastic strain increment dpla and progressive damage accumulation."""
    d1 = 0.1
    d2 = 1.0
    t0 = 0.2
    phel = 4.0
    hel = 10.0
    g = 100.0
    mat = build_law79({
        "rho": 3.0,
        "shear": g,
        "a": 1.0,
        "b": 0.2,
        "n": 0.0,
        "m": 0.0,
        "d1": d1,
        "d2": d2,
        "t": t0,
        "hel": hel,
        "phel": phel,
        "k1": 150.0,
    })
    shel = 1.5 * (hel - phel)

    # Zero pressure: P* = 0, P* + T* = T* = 0.2 / 4.0 = 0.05
    # epfail = D1 * (T*)^D2 = 0.1 * (0.05)^1.0 = 0.005
    epfail_expected = 0.1 * 0.05

    extra = {"dmg": 0.0, "sigy_old": 1.0}
    # Apply shear strain causing plastic flow
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])  # s_trial = G * deps_xy = 2.0
    # Trial VM = sqrt(3)*2.0 = 3.464, Yield = 1.0 * shel = 1.0 * 9.0 = 9.0 (too large!)
    # Let's adjust strain so trial VM exceeds yield:
    # Yield = 1.0 * 9.0 = 9.0 => gamma_xy > 9.0 / (sqrt(3) * 100) = 0.05196
    deps = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    s_tr_xy = g * 0.1  # 10.0
    vm_tr = math.sqrt(3.0 * (s_tr_xy ** 2))  # 17.3205
    scale_expected = shel / vm_tr
    dpla_expected = (1.0 - scale_expected) * vm_tr / (3.0 * math.sqrt(3.0) * g)

    sig_new, epsp_new, _ = solid_update(mat, np.zeros(6), deps, epsp=0.0, extra=extra, return_tuple=True)

    assert math.isclose(epsp_new, dpla_expected, rel_tol=1e-4)
    expected_dmg = min(1.0, dpla_expected / epfail_expected)
    assert math.isclose(float(extra["dmg"]), expected_dmg, rel_tol=1e-4)
    assert float(extra["dmg"]) > 0.0


# ---------------------------------------------------------------------------
# Bulking pressure increment
# ---------------------------------------------------------------------------

def test_bulking_pressure_increment():
    """Verify internal shear strain energy converted to bulking pressure via beta."""
    g = 100.0
    k1 = 150.0
    beta = 0.8
    hel = 10.0
    phel = 4.0
    mat = build_law79({
        "rho": 3.0,
        "shear": g,
        "a": 1.0,
        "b": 0.2,
        "n": 0.0,
        "m": 0.0,
        "d1": 0.05,
        "d2": 1.0,
        "t": 0.2,
        "hel": hel,
        "phel": phel,
        "k1": k1,
        "beta": beta,
    })
    shel = 1.5 * (hel - phel)

    # Setup state with compaction mu > 0 and previous step intact strength sigy_old = 1.0
    mu = 0.02
    extra = {
        "mu": mu,
        "amu": mu,
        "dmg": 0.0,
        "deltap": 0.0,
        "sigy_old": 1.0,  # intact
        "off": 1.0,
    }

    # Apply plastic strain causing damage to increase
    deps = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    solid_update(mat, np.zeros(6), deps, extra=extra)

    # After update, dmg > 0, so yield_curr = (1 - D)*1.0 + D*0.2 < sigy_old
    # deltau = (1.0^2 - yield_curr^2) / (6 * G) * shel^2 > 0
    # deltap should be positive!
    assert extra["deltap"] > 0.0

    # Verify formula: P1 = K1 * mu
    p1 = k1 * mu
    dmg = float(extra["dmg"])
    yield_curr = (1.0 - dmg) * 1.0 + dmg * 0.2
    deltau = (1.0 ** 2 - yield_curr ** 2) / (6.0 * g) * (shel ** 2)
    expected_deltap = -p1 + math.sqrt(p1 ** 2 + 2.0 * beta * k1 * deltau)
    assert math.isclose(float(extra["deltap"]), expected_deltap, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Element deletion modes (IDEL)
# ---------------------------------------------------------------------------

def test_element_deletion_idel_modes():
    """Verify element deletion under IDEL=1 (tension), IDEL=2 (epsmax), IDEL=3 (damage)."""
    # 1. IDEL = 1: Deletion in hydrostatic tension (P* + T* < 0)
    mat_idel1 = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "t": 0.2,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
        "idel": 1,
    })
    # mu < 0 large enough so P < -T0 => P* + T* < 0
    extra1 = {"mu": -0.01, "off": 1.0}  # K1*mu = -1.5 < -0.2
    solid_update(mat_idel1, np.zeros(6), np.zeros(6), extra=extra1)
    assert extra1["off"] == 0.8  # Element deletion triggered

    # 2. IDEL = 2: Deletion if plastic strain > epsmax
    mat_idel2 = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "n": 0.0,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
        "idel": 2,
        "epsmax": 0.005,
        "d1": 0.5,
    })
    extra2 = {"off": 1.0}
    deps_large = np.array([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])
    solid_update(mat_idel2, np.zeros(6), deps_large, epsp=0.01, extra=extra2)
    assert extra2["off"] == 0.8

    # 3. IDEL = 3: Deletion if damage == 1.0
    mat_idel3 = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
        "idel": 3,
        "d1": 0.001,  # very fragile
    })
    extra3 = {"off": 1.0, "dmg": 0.999}
    solid_update(mat_idel3, np.zeros(6), deps_large, extra=extra3)
    assert extra3["dmg"] == 1.0
    assert extra3["off"] == 0.8

    # 4. Stress becomes 0 when off <= 0.0 (completely deleted)
    extra_deleted = {"off": 0.0}
    sig_del = solid_update(mat_idel3, np.zeros(6), deps_large, extra=extra_deleted)
    assert np.allclose(sig_del, 0.0)


# ---------------------------------------------------------------------------
# Sound speed calculation
# ---------------------------------------------------------------------------

def test_sound_speed_solid():
    """Verify longitudinal wave speed under tension and compression."""
    rho0 = 3.2
    g = 190.0
    k1 = 220.0
    k2 = 150.0
    k3 = 80.0
    mat = build_law79({
        "rho": rho0,
        "shear": g,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "hel": 14.0,
        "phel": 5.0,
    })

    # Tension (mu <= 0): dpdmu = K1
    c_tension = sound_speed_solid(mat, extra={"mu": -0.01})
    expected_c_tension = math.sqrt((k1 + 4.0 / 3.0 * g) / rho0)
    assert math.isclose(c_tension, expected_c_tension, rel_tol=1e-12)

    # Compression (mu > 0): dpdmu = K1 + 2*K2*mu + 3*K3*mu^2
    mu_val = 0.04
    dpdmu = k1 + 2.0 * k2 * mu_val + 3.0 * k3 * (mu_val ** 2)
    expected_c_comp = math.sqrt((dpdmu + 4.0 / 3.0 * g) / rho0)
    c_comp = sound_speed_solid(mat, extra={"mu": mu_val})
    assert math.isclose(c_comp, expected_c_comp, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Tangent stiffness tensor
# ---------------------------------------------------------------------------

def test_consistent_solid_tangent_elastic():
    """Verify consistent solid tangent matches elastic tensor C_el."""
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "t": 0.35,
        "k1": 150.0,
        "hel": 10.0,
        "phel": 4.0,
    })
    G = 100.0
    K = 150.0

    # Analytical tangent
    D_ana = consistent_solid_tangent(mat, np.zeros(6), analytical=True)
    assert D_ana.shape == (6, 6)

    # Check normal components: C_11 = K + 4/3*G
    assert math.isclose(D_ana[0, 0], K + 4.0 / 3.0 * G, rel_tol=1e-10)
    # Off-diagonal normal: C_12 = K - 2/3*G
    assert math.isclose(D_ana[0, 1], K - 2.0 / 3.0 * G, rel_tol=1e-10)
    # Shear component: C_44 = G
    assert math.isclose(D_ana[3, 3], G, rel_tol=1e-10)

    # Numerical perturbation tangent with small strain increment
    deps_small = np.array([1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0])
    D_num = consistent_solid_tangent(mat, np.zeros(6), deps=deps_small)
    assert np.allclose(D_num, D_ana, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Vectorization and package integration
# ---------------------------------------------------------------------------

def test_vectorization_multi_elements():
    """Verify vectorized execution on multiple elements simultaneously."""
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "a": 1.0,
        "hel": 10.0,
        "phel": 4.0,
        "k1": 150.0,
    })

    nel = 4
    sig_arr = np.zeros((nel, 6), dtype=float)
    deps_arr = np.array([
        [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1e-5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1e-5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1e-5],
    ], dtype=float)

    sig_new, epsp_new, c_arr = solid_update(mat, sig_arr, deps_arr, epsp=np.zeros(nel), return_tuple=True)
    assert sig_new.shape == (nel, 6)
    assert epsp_new.shape == (nel,)
    assert len(c_arr) == nel


def test_shell_update_not_implemented():
    """Verify shell_update raises NotImplementedError for brittle solids."""
    mat = build_law79({
        "rho": 3.0,
        "shear": 100.0,
        "k1": 150.0,
        "hel": 10.0,
        "phel": 4.0,
    })
    with pytest.raises(NotImplementedError, match="3D solid and SPH"):
        shell_update(mat, np.zeros(3), np.zeros(3))


def test_pyradioss_materials_dispatcher():
    """Verify pyradioss.materials dispatching for LAW79."""
    mat = build_law79({
        "id": 79,
        "rho": 3.2,
        "shear": 190.0,
        "a": 1.0,
        "t": 0.35,
        "k1": 220.0,
        "hel": 14.0,
        "phel": 5.0,
    })

    # Dispatch solid_update
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.0, 0.0, 0.0, 1e-4, 0.0, 0.0])
    sig_out, epsp_out, c_out = materials.solid_update(mat, sig, deps, return_tuple=True)
    assert math.isclose(sig_out[3], 190.0 * 1e-4, rel_tol=1e-8)
    assert c_out > 0.0

    # Dispatch sound_speed
    c_disp = materials.sound_speed(mat)
    assert c_disp > 0.0

    # Dispatch solid_tangent
    D_disp = materials.solid_tangent(mat, sig)
    assert D_disp.shape == (6, 6)

    # Shell update raises NotImplementedError
    with pytest.raises(NotImplementedError):
        materials.shell_update(mat, np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# Starter Reader Tests (Fixed & Free format, Synonyms)
# ---------------------------------------------------------------------------

def test_starter_keyword_reader_fixed(tmp_path):
    """Verify fixed-format /MAT/LAW79 reading into Model.mat_law79s."""
    from pathlib import Path
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import read_mat_law79
    from pyradioss.common.messages import MessageLog
    from pyradioss.model.model import Model

    deck_text = """# OpenRadioss Starter Deck
/BEGIN
TEST_LAW79
/MAT/LAW79/101
Silicon-Carbide-Brittle
#        Init. dens.          Ref. dens.
               3.21e-3               3.21e-3
#                  G
                 193.0
#                  a                   b                   m                   n
                0.96                0.35                 1.0                0.65
#                  c                EPS0          SIGMA_FMAX                FCUT
               0.009                 1.0                 0.8                 0.0
#                  T                 HEL                PHEL
                0.37                14.5                5.13
#                 D1                  D2                IDEL              EPSMAX
                0.48                0.48                   2              1.0e20
#                 K1                  K2                  K3                BETA
               220.0                 0.0                 0.0                 1.0
/END
"""
    deck_file = tmp_path / "deck_fixed_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law79(block, model, log)

    assert not log.has_errors
    assert 101 in model.mat_law79s
    m = model.mat_law79s[101]
    assert m.id == 101
    assert m.title.strip() == "Silicon-Carbide-Brittle"
    assert math.isclose(m.rho, 3.21e-3, rel_tol=1e-8)
    assert math.isclose(m.refer_rho, 3.21e-3, rel_tol=1e-8)
    assert math.isclose(m.tau_shear, 193.0, rel_tol=1e-8)
    assert math.isclose(m.a, 0.96, rel_tol=1e-8)
    assert math.isclose(m.b, 0.35, rel_tol=1e-8)
    assert math.isclose(m.m, 1.0, rel_tol=1e-8)
    assert math.isclose(m.n, 0.65, rel_tol=1e-8)
    assert math.isclose(m.c, 0.009, rel_tol=1e-8)
    assert math.isclose(m.eps0, 1.0, rel_tol=1e-8)
    assert math.isclose(m.sigfmax, 0.8, rel_tol=1e-8)
    assert math.isclose(m.t, 0.37, rel_tol=1e-8)
    assert math.isclose(m.hel, 14.5, rel_tol=1e-8)
    assert math.isclose(m.phel, 5.13, rel_tol=1e-8)
    assert math.isclose(m.d1, 0.48, rel_tol=1e-8)
    assert math.isclose(m.d2, 0.48, rel_tol=1e-8)
    assert m.idel == 2
    assert math.isclose(m.k1, 220.0, rel_tol=1e-8)
    assert math.isclose(m.beta, 1.0, rel_tol=1e-8)

    # Derived properties on parsed entity
    assert math.isclose(m.shel, 1.5 * (14.5 - 5.13), rel_tol=1e-8)
    assert math.isclose(m.tstar, 0.37 / 5.13, rel_tol=1e-8)
    assert m.young > 0.0
    assert 0.0 < m.nu < 0.5


def test_starter_keyword_reader_free_and_synonyms(tmp_path):
    """Verify free-format and synonyms (/MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2)."""
    from pathlib import Path
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import read_mat_law79
    from pyradioss.common.messages import MessageLog
    from pyradioss.model.model import Model

    deck_text = """# OpenRadioss Starter Deck
/BEGIN
TEST_SYNONYMS
/MAT/JOHN_HOLM/1
JH_Free
2.5, 2.5
120.0
0.9, 0.4, 1.0, 0.7
0.01, 1.0, 1.0e30, 0.0
0.25, 12.0, 4.5
0.3, 0.6, 0, 1.0e20
160.0, 50.0, 20.0, 0.9
/MAT/JOHNSON_HOLMQUIST/2
JH_Syn1
3.0
150.0
0.8, 0.3, 1.0, 0.6
0.0, 1.0, 1.0e30, 0.0
0.3, 10.0, 4.0
0.5, 1.0, 1, 1.0e20
180.0, 0.0, 0.0, 1.0
/MAT/JH2/3
JH_Syn2
2.8
140.0
0.85, 0.32, 1.0, 0.62
0.005, 1.0, 0.7, 0.0
0.28, 11.0, 4.2
0.4, 0.5, 3, 1.0e20
175.0, 0.0, 0.0, 0.85
/END
"""
    deck_file = tmp_path / "deck_syn_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law79(block, model, log)

    assert not log.has_errors
    assert 1 in model.mat_john_holms
    assert 2 in model.mat_law79s
    assert 3 in model.mat_jh2s

    m1 = model.mat_john_holms[1]
    assert m1.tau_shear == 120.0
    assert m1.k1 == 160.0
    assert m1.k2 == 50.0
    assert m1.k3 == 20.0
    assert m1.beta == 0.9

    m2 = model.mat_law79s[2]
    assert m2.idel == 1
    assert m2.hel == 10.0

    m3 = model.mat_jh2s[3]
    assert m3.idel == 3
    assert m3.sigfmax == 0.7


# ---------------------------------------------------------------------------
# Deck Writer Tests
# ---------------------------------------------------------------------------

def test_deck_writer_roundtrip(tmp_path):
    """Verify StarterDeck().mat_law79() writes format that Starter reads back identically."""
    from pathlib import Path
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import read_mat_law79
    from pyradioss.common.messages import MessageLog
    from pyradioss.model.model import Model

    writer = StarterDeck("ROUNDTRIP_LAW79")
    writer.mat_law79(
        mat_id=1,
        title="Ceramic-JH2",
        rho=3.2,
        refer_rho=3.2,
        tau_shear=190.0,
        a=0.92,
        b=0.36,
        m=1.0,
        n=0.64,
        c=0.008,
        eps0=1.0,
        sigfmax=0.75,
        fcut=0.0,
        t=0.36,
        hel=14.2,
        phel=5.1,
        d1=0.45,
        d2=0.45,
        idel=2,
        epsmax=1.0e20,
        k1=215.0,
        k2=12.0,
        k3=5.0,
        beta=0.95,
    )
    deck_str = writer.write()

    deck_file = tmp_path / "deck_wt_0000.rad"
    deck_file.write_text(deck_str, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law79(block, model, log)

    assert not log.has_errors
    assert 1 in model.mat_law79s
    m = model.mat_law79s[1]
    assert m.title.strip() == "Ceramic-JH2"
    assert math.isclose(m.rho, 3.2, rel_tol=1e-6)
    assert math.isclose(m.tau_shear, 190.0, rel_tol=1e-6)
    assert math.isclose(m.a, 0.92, rel_tol=1e-6)
    assert math.isclose(m.b, 0.36, rel_tol=1e-6)
    assert math.isclose(m.m, 1.0, rel_tol=1e-6)
    assert math.isclose(m.n, 0.64, rel_tol=1e-6)
    assert math.isclose(m.c, 0.008, rel_tol=1e-6)
    assert math.isclose(m.eps0, 1.0, rel_tol=1e-6)
    assert math.isclose(m.sigfmax, 0.75, rel_tol=1e-6)
    assert math.isclose(m.t, 0.36, rel_tol=1e-6)
    assert math.isclose(m.hel, 14.2, rel_tol=1e-6)
    assert math.isclose(m.phel, 5.1, rel_tol=1e-6)
    assert math.isclose(m.d1, 0.45, rel_tol=1e-6)
    assert math.isclose(m.d2, 0.45, rel_tol=1e-6)
    assert m.idel == 2
    assert math.isclose(m.k1, 215.0, rel_tol=1e-6)
    assert math.isclose(m.k2, 12.0, rel_tol=1e-6)
    assert math.isclose(m.k3, 5.0, rel_tol=1e-6)
    assert math.isclose(m.beta, 0.95, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Starter Checks Tests (ANCMSG errors)
# ---------------------------------------------------------------------------

def test_starter_checks_valid():
    """Verify check_mat_law79 passes cleanly on valid material."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law79

    mat = build_law79({
        "id": 1,
        "rho": 3.2,
        "shear": 190.0,
        "a": 0.95,
        "t": 0.35,
        "hel": 14.0,
        "phel": 5.0,
        "k1": 220.0,
        "eps0": 1.0,
        "beta": 0.9,
    })
    log = MessageLog()
    check_mat_law79(mat=mat, log=log)
    assert not log.has_errors


def test_starter_checks_errors():
    """Verify check_mat_law79 catches ANCMSG errors."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law79

    # 1. PHEL > HEL (ANCMSG 907)
    bad_phel = MatLaw79(id=1, rho=3.2, tau_shear=190.0, hel=10.0, phel=12.0, k1=200.0, eps0=1.0, beta=1.0)
    log = MessageLog()
    check_mat_law79(mat=bad_phel, log=log)
    assert log.has_errors
    assert any("ANCMSG 907" in str(msg) or "cannot exceed HEL" in str(msg) for msg in log.errors)

    # 2. SHEAR <= 0 (ANCMSG 908)
    bad_shear = MatLaw79(id=1, rho=3.2, tau_shear=-5.0, hel=14.0, phel=5.0, k1=200.0, eps0=1.0, beta=1.0)
    log = MessageLog()
    check_mat_law79(mat=bad_shear, log=log)
    assert log.has_errors
    assert any("ANCMSG 908" in str(msg) or "shear modulus" in str(msg) for msg in log.errors)

    # 3. K1 <= 0 (ANCMSG 909)
    bad_k1 = MatLaw79(id=1, rho=3.2, tau_shear=190.0, hel=14.0, phel=5.0, k1=0.0, eps0=1.0, beta=1.0)
    log = MessageLog()
    check_mat_law79(mat=bad_k1, log=log)
    assert log.has_errors
    assert any("ANCMSG 909" in str(msg) or "bulk modulus" in str(msg) for msg in log.errors)

    # 4. EPS0 <= 0 (ANCMSG 910)
    bad_eps0 = MatLaw79(id=1, rho=3.2, tau_shear=190.0, hel=14.0, phel=5.0, k1=200.0, eps0=-1.0, beta=1.0)
    log = MessageLog()
    check_mat_law79(mat=bad_eps0, log=log)
    assert log.has_errors
    assert any("ANCMSG 910" in str(msg) or "reference strain rate" in str(msg) for msg in log.errors)

    # 5. BETA not in [0, 1] (ANCMSG 911)
    bad_beta = MatLaw79(id=1, rho=3.2, tau_shear=190.0, hel=14.0, phel=5.0, k1=200.0, eps0=1.0, beta=1.5)
    log = MessageLog()
    check_mat_law79(mat=bad_beta, log=log)
    assert log.has_errors
    assert any("ANCMSG 911" in str(msg) or "BETA" in str(msg) for msg in log.errors)


# ---------------------------------------------------------------------------
# Extra Shapes Registration
# ---------------------------------------------------------------------------

def test_extra_shapes():
    """Verify extra_shapes exposes persistent state variables for solids."""
    mat = build_law79({
        "id": 1,
        "rho": 3.2,
        "shear": 190.0,
        "k1": 220.0,
        "hel": 14.0,
        "phel": 5.0,
    })
    shapes = materials.extra_shapes(mat)
    assert "deltap" in shapes
    assert "sigy_old" in shapes
    assert "dmg" in shapes
    assert "off" in shapes
    assert "off79" in shapes
    assert "mu" in shapes
    assert "amu" in shapes
    assert "uvar" in shapes
    assert shapes["uvar"] == (2,)


# ---------------------------------------------------------------------------
# Compatibility & Element Checks
# ---------------------------------------------------------------------------

def test_starter_checks_incompatible_elements():
    """Verify check_mat_law79 rejects shells, 1D elements, and 2D analysis."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law79
    from pyradioss.model.model import Model

    mat = build_law79({
        "id": 1,
        "rho": 3.2,
        "shear": 190.0,
        "k1": 220.0,
        "hel": 14.0,
        "phel": 5.0,
    })

    class DummyGroup:
        def __init__(self, name: str, mid: int):
            self.name = name
            self.mid = mid
            class DummyEl:
                mat_id = mid
            self.els = [DummyEl()]
        def values(self):
            return self.els

    # 1. Shell elements (ANCMSG 305)
    model_shell = Model()
    model_shell.materials[1] = mat
    grp_shell = DummyGroup("shells", 1)
    model_shell.element_groups = lambda: [("shells", grp_shell)]
    log = MessageLog()
    check_mat_law79(model=model_shell, mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 305" in str(msg) or "not supported for shell" in str(msg) for msg in log.errors)

    # 2. 1D elements (ANCMSG 306)
    model_1d = Model()
    model_1d.materials[1] = mat
    grp_1d = DummyGroup("beams", 1)
    model_1d.element_groups = lambda: [("beams", grp_1d)]
    log2 = MessageLog()
    check_mat_law79(model=model_1d, mat=mat, log=log2)
    assert log2.has_errors
    assert any("ANCMSG 306" in str(msg) or "not supported for 1D" in str(msg) for msg in log2.errors)

    # 3. 2D analysis (N2D > 0, ANCMSG 305)
    model_2d = Model()
    model_2d.n2d = 1
    model_2d.materials[1] = mat
    log3 = MessageLog()
    check_mat_law79(model=model_2d, mat=mat, log=log3)
    assert log3.has_errors
    assert any("ANCMSG 305" in str(msg) or "2D analysis" in str(msg) for msg in log3.errors)


# ---------------------------------------------------------------------------
# Solid Hexa8 Kernel Integration
# ---------------------------------------------------------------------------

def test_hexa8_solid_integration():
    """Verify Hexa8 brick element initialization, sound speed, and force evaluation with LAW79."""
    from pyradioss.elements import solid_hexa8
    from pyradioss.model.model import Model

    mat = build_law79({
        "id": 1,
        "rho": 3.2,
        "shear": 190.0,
        "a": 0.95,
        "b": 0.35,
        "m": 1.0,
        "n": 0.65,
        "t": 0.35,
        "hel": 14.0,
        "phel": 5.0,
        "k1": 220.0,
        "beta": 1.0,
        "idel": 0,
    })

    class DummyProp:
        id = 1
        params = {"qa": 1.1, "qb": 0.05, "hm": 0.1, "hf": 0.1, "hr": 0.1}

    class DummyGroup:
        def __init__(self, conn):
            self.conn = np.asarray(conn, dtype=np.int64)
            self.n = len(self.conn)
            self.ids = np.arange(1, self.n + 1, dtype=np.int64)
            self.state = {"slices": [(slice(0, 1), mat, DummyProp())]}
            self._model = None

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    group = DummyGroup(conn)
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)

    # Initial time step and wave speed
    fint = np.zeros((8, 3), dtype=float)
    mint = np.zeros((8, 3), dtype=float)
    dt_arr = solid_hexa8.forces(group, coords, None, None, 0.0, fint, mint)

    c_expected = math.sqrt((220.0 + (4.0 / 3.0) * 190.0) / 3.2)
    assert len(dt_arr) == 1
    assert dt_arr[0] > 0.0
    dtfac = group.state.get("dtfac", np.array([0.85]))[0]
    # For unit cube, lc = 1.0, so dt_expected = dtfac * (1.0 / c_expected)
    dt_expected = dtfac * (1.0 / c_expected)
    assert math.isclose(dt_arr[0], dt_expected, rel_tol=0.01)

    # Run one step with velocity gradient (compression)
    ve = np.zeros((8, 3), dtype=float)
    ve[4:, 2] = -1.0
    dt_step = 1.0e-3

    solid_hexa8.forces(group, coords + ve * dt_step, ve, None, dt_step, fint, mint)

    assert fint[4:, 2].sum() > 0.0
    assert fint[:4, 2].sum() < 0.0
    st = group.state
    assert st["off"][0] == 1.0
    assert not np.allclose(st["sig"][0], 0.0)



