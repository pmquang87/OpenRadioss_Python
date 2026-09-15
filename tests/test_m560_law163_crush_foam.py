"""Unit tests for Milestone M560: /MAT/LAW163 (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM).

Constitutive physics kernel tests verifying:
1. Law163Params dataclass fields, defaults, and derived moduli (g, bulk, cii, cij).
2. build_law163 factory supporting dict, MatLaw163, Material, and Law163Params.
3. Elastic trial stress tensor calculations (uniaxial, hydrostatic, pure shear).
4. Principal stress spectral decomposition and scaling with compressive yield stress.
5. Tensile stress cutoff (tsc) clamping.
6. Volumetric strain & strain rate filtering (alpha, nrs=0/1, srclmt change limit).
7. Yield table and curve lookup (1D FunctTable, 2D table grid, fscale factor, slope dsdgam).
8. Viscous damping stress tensor and damping parameter influence.
9. Sound speed computation with bulk modulus, table slope, and damping stiffness.
10. Algorithmic consistent solid tangent tensor (n, 6, 6).
11. Shell update raising NotImplementedError.
12. Integration with pyradioss.materials dispatchers (solid_update, sound_speed, solid_tangent).
13. Vectorized multi-element group consistency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import Material, MatLaw163
import pyradioss.materials as materials
from pyradioss.materials.law163_crush_foam import (
    Law163Params,
    build_law163,
    solid_update,
    sound_speed_solid,
    consistent_solid_tangent,
    shell_update,
    extra_shapes,
    resolve,
)


# ---------------------------------------------------------------------------
# 1. Parameter checks and factory tests
# ---------------------------------------------------------------------------

def test_law163_params_dataclass_and_derived():
    """Verify Law163Params defaults, derived elastic properties, and nu clamping."""
    # Default parameters
    p = Law163Params(rho0=1.2e-3, e=100.0, nu=0.25)
    assert p.rho0 == 1.2e-3
    assert p.refer_rho == 1.2e-3
    assert p.damp == 0.10
    assert p.ncycle == 12
    assert p.srclmt == 1.0e20
    assert p.fscale == 1.0
    assert p.nrs == 0

    # Derived moduli for E=100, nu=0.25:
    # G = E / (2*(1+nu)) = 100 / 2.5 = 40.0
    # K = E / (3*(1-2*nu)) = 100 / (3*0.5) = 66.6666667
    # cii = K + 4/3*G = 66.6666667 + 53.3333333 = 120.0
    # cij = K - 2/3*G = 66.6666667 - 26.6666667 = 40.0
    assert math.isclose(p.g, 40.0, rel_tol=1e-12)
    assert math.isclose(p.bulk, 100.0 / 1.5, rel_tol=1e-12)
    assert math.isclose(p.cii, 120.0, rel_tol=1e-12)
    assert math.isclose(p.cij, 40.0, rel_tol=1e-12)
    assert math.isclose(p.cii - p.cij, 2.0 * p.g, rel_tol=1e-12)

    # nu clamping: nu > 0.499 is clamped to 0.499 (hm_read_mat163 check)
    p_high_nu = Law163Params(e=100.0, nu=0.5)
    assert p_high_nu.nu == 0.499

    # nu = 0 (typical ideal crushable foam without lateral expansion)
    p_zero_nu = Law163Params(e=60.0, nu=0.0)
    assert math.isclose(p_zero_nu.g, 30.0, rel_tol=1e-12)
    assert math.isclose(p_zero_nu.bulk, 20.0, rel_tol=1e-12)
    assert math.isclose(p_zero_nu.cii, 60.0, rel_tol=1e-12)
    assert math.isclose(p_zero_nu.cij, 0.0, rel_tol=1e-12)


def test_build_law163_factory_inputs():
    """Verify build_law163 factory supports dict, MatLaw163, Material, and Law163Params."""
    # From dict with standard names
    d1 = {
        "id": 10,
        "title": "Foam-A",
        "rho": 0.05,
        "e": 200.0,
        "nu": 0.2,
        "tsc": 5.0,
        "damp": 0.15,
        "ncycle": 16,
        "fscale": 1.25,
    }
    m1 = build_law163(d1)
    assert isinstance(m1, Material)
    assert m1.id == 10
    assert m1.law == 163
    assert m1.rho0 == 0.05
    assert m1.title == "Foam-A"
    assert m1.params["tsc"] == 5.0
    assert m1.params["damp"] == 0.15
    assert m1.params["ncycle"] == 16
    assert m1.params["fscale"] == 1.25

    # From dict with CFG starter token keys
    d2 = {
        "id": 20,
        "MAT_RHO": 0.08,
        "MAT_E": 150.0,
        "MAT_NU": 0.1,
        "LSDYNA_TSC": 8.0,
        "LSD_MAT_DAMP": 0.2,
        "LSD_NCYCLE": 20,
        "LSD_TID": 101,
        "FSCALE": 2.0,
        "NRSFlag": 1,
    }
    m2 = build_law163(d2)
    assert m2.id == 20
    assert m2.rho0 == 0.08
    assert m2.params["tsc"] == 8.0
    assert m2.params["damp"] == 0.2
    assert m2.params["ncycle"] == 20
    assert m2.params["tab_id"] == 101
    assert m2.params["fscale"] == 2.0
    assert m2.params["nrs"] == 1

    # From MatLaw163 entity
    ent = MatLaw163(
        id=30,
        rho=0.04,
        e=80.0,
        nu=0.15,
        tsc=3.0,
        damp=0.08,
        ncycle=10,
        tab_id=202,
        title="MatLaw163-Entity",
    )
    m3 = build_law163(ent)
    assert m3.id == 30
    assert m3.rho0 == 0.04
    assert m3.title == "MatLaw163-Entity"
    assert m3.params["tsc"] == 3.0
    assert m3.params["damp"] == 0.08
    assert m3.params["tab_id"] == 202

    # From Law163Params instance
    lp = Law163Params(rho0=0.03, e=50.0, nu=0.0, tsc=1.5)
    m4 = build_law163(lp)
    assert m4.rho0 == 0.03
    assert m4.params["cii"] == 50.0


# ---------------------------------------------------------------------------
# 2. Elastic trial stress & principal decomposition
# ---------------------------------------------------------------------------

def test_elastic_trial_stress_uniaxial_and_shear():
    """Verify exact elastic trial stress integration before plasticity / cutoff."""
    p = Law163Params(rho0=1.0, e=120.0, nu=0.2, tsc=1e20)  # very high tsc, no table
    mat = build_law163(p)

    sig_old = np.zeros(6)
    # Uniaxial strain increment deps_xx = 1e-3
    deps = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    # With dt=0 -> no damping
    sig_new = solid_update(mat, sig_old, deps, dt=0.0, return_tuple=False)

    # st_xx = cii * deps_xx, st_yy = cij * deps_xx, st_zz = cij * deps_xx
    assert math.isclose(sig_new[0], p.cii * 1e-3, rel_tol=1e-10)
    assert math.isclose(sig_new[1], p.cij * 1e-3, rel_tol=1e-10)
    assert math.isclose(sig_new[2], p.cij * 1e-3, rel_tol=1e-10)
    assert math.isclose(sig_new[3], 0.0, abs_tol=1e-15)

    # Pure shear strain increment deps_xy = 2e-3
    deps_shear = np.array([0.0, 0.0, 0.0, 2e-3, 0.0, 0.0])
    sig_shear = solid_update(mat, sig_old, deps_shear, dt=0.0, return_tuple=False)
    assert math.isclose(sig_shear[3], p.g * 2e-3, rel_tol=1e-10)
    assert math.isclose(sig_shear[0], 0.0, abs_tol=1e-15)


def test_elastic_trial_stress_hydrostatic():
    """Verify hydrostatic trial stress matches 3 * bulk * deps_0."""
    p = Law163Params(rho0=1.0, e=300.0, nu=0.25, tsc=1e20)
    mat = build_law163(p)

    sig_old = np.zeros(6)
    deps_vol = np.array([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0])
    sig_new = solid_update(mat, sig_old, deps_vol, dt=0.0, return_tuple=False)

    # Hydrostatic stress = (cii + 2*cij) * deps_0 = (3*bulk) * deps_0
    expected_p = 3.0 * p.bulk * 1e-4
    assert math.isclose(sig_new[0], expected_p, rel_tol=1e-10)
    assert math.isclose(sig_new[1], expected_p, rel_tol=1e-10)
    assert math.isclose(sig_new[2], expected_p, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 3. Principal stress scaling (compressive yield & tensile cutoff)
# ---------------------------------------------------------------------------

def test_principal_stress_compressive_yield_scaling():
    """Verify principal stresses are clamped to negative yield stress in compression."""
    # Yield stress = 40.0 -> sigy = -40.0
    p = Law163Params(rho0=1.0, e=1000.0, nu=0.0, tsc=1e20, table=40.0)
    mat = build_law163(p)

    sig_old = np.zeros(6)
    # Severe compressive strain in x: deps_xx = -0.1 -> trial st_xx = 1000 * (-0.1) = -100.0
    # Trial principal stresses: (-100, 0, 0)
    deps = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    sig_new = solid_update(mat, sig_old, deps, dt=0.0, return_tuple=False)
    # sigp[0] was -100.0, clamped to sigy = -40.0
    assert math.isclose(sig_new[0], -40.0, rel_tol=1e-10)
    assert math.isclose(sig_new[1], 0.0, abs_tol=1e-10)
    assert math.isclose(sig_new[2], 0.0, abs_tol=1e-10)


def test_principal_stress_tensile_cutoff():
    """Verify principal stresses are clamped to tensile cutoff (tsc)."""
    p = Law163Params(rho0=1.0, e=1000.0, nu=0.0, tsc=25.0, table=100.0)
    mat = build_law163(p)

    sig_old = np.zeros(6)
    # Large tensile strain in y: deps_yy = +0.05 -> trial st_yy = +50.0
    deps = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])

    sig_new = solid_update(mat, sig_old, deps, dt=0.0, return_tuple=False)
    # Clamped to tsc = 25.0
    assert math.isclose(sig_new[1], 25.0, rel_tol=1e-10)
    assert math.isclose(sig_new[0], 0.0, abs_tol=1e-10)
    assert math.isclose(sig_new[2], 0.0, abs_tol=1e-10)


def test_rotated_principal_stress_reconstruction():
    """Verify eigendecomposition, principal clamping, and reconstruction for rotated stress state."""
    # Yield stress = 50.0
    p = Law163Params(rho0=1.0, e=200.0, nu=0.0, tsc=1e20, table=50.0)
    mat = build_law163(p)

    # Pure shear in xy: deps_xy = 0.8 -> trial st_xy = G * 0.8 = 100.0 * 0.8 = 80.0
    # Symmetric 3x3 matrix in xy plane has eigenvalues +80.0 (tension at 45 deg) and -80.0 (compression at 135 deg)
    sig_old = np.zeros(6)
    deps = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0])

    # Compressive principal stress (-80) clamped to -50. Tensile principal stress (+80) stays +80 (tsc=1e20).
    sig_new = solid_update(mat, sig_old, deps, dt=0.0, return_tuple=False)

    # Reconstruct 2D tensor:
    # Principal stresses: p1 = +80, p2 = -50
    # Eigenvectors: v1 = [1/sqrt(2), 1/sqrt(2)], v2 = [-1/sqrt(2), 1/sqrt(2)]
    # S_xx = 0.5 * (+80) + 0.5 * (-50) = +15.0
    # S_yy = 0.5 * (+80) + 0.5 * (-50) = +15.0
    # S_xy = 0.5 * (+80) - 0.5 * (-50) = +65.0
    assert math.isclose(sig_new[0], 15.0, rel_tol=1e-10)
    assert math.isclose(sig_new[1], 15.0, rel_tol=1e-10)
    assert math.isclose(sig_new[3], 65.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 4. Volumetric strain & strain rate filtering
# ---------------------------------------------------------------------------

def test_strain_rate_filtering_formula_and_ncycle():
    """Verify volumetric strain rate filter alpha = 2*pi / (2*pi + ncycle)."""
    ncycle = 12
    alpha_expected = 2.0 * math.pi / (2.0 * math.pi + 12.0)
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, ncycle=ncycle, nrs=0)
    mat = build_law163(p)

    # Step 1: True strain rate from deps: deps = [-0.01, -0.01, -0.01, 0, 0, 0], dt=0.001
    # raw_dgamdt = -(-0.03) / 0.001 = 30.0
    # epsd_old = 0.0
    # dgamdt_filtered = alpha * 30.0 + (1 - alpha) * 0.0 = alpha * 30.0
    dt = 0.001
    deps = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0])
    extra = {}
    solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra)

    epsd_1 = extra["epsd"]
    assert math.isclose(float(epsd_1[0]), alpha_expected * 30.0, rel_tol=1e-10)

    # Step 2: raw_dgamdt = 30.0 again
    # dgamdt_2 = alpha * 30.0 + (1 - alpha) * epsd_1
    solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra)
    epsd_2 = extra["epsd"]
    expected_2 = alpha_expected * 30.0 + (1.0 - alpha_expected) * float(epsd_1[0])
    assert math.isclose(float(epsd_2[0]), expected_2, rel_tol=1e-10)


def test_strain_rate_change_limit_srclmt():
    """Verify rate change capping by srclmt * dt."""
    # srclmt = 1000.0, dt = 0.01 -> max allowed change is 10.0
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, ncycle=1, srclmt=1000.0, nrs=0)
    mat = build_law163(p)

    # raw_dgamdt = 100.0, epsd_old = 0.0
    # Without cap, filtered rate would be ~86.2. With cap, it must be capped at 10.0!
    dt = 0.01
    deps = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # raw rate = 1.0 / 0.01 = 100.0
    extra = {}
    solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra)

    assert math.isclose(float(extra["epsd"][0]), 10.0, rel_tol=1e-10)


def test_strain_rate_nrs_flag():
    """Verify nrs=0 (true strain rate) vs nrs=1 (engineering strain rate)."""
    p0 = Law163Params(rho0=1.0, e=100.0, nu=0.0, nrs=0, ncycle=1000000)  # alpha small
    p1 = Law163Params(rho0=1.0, e=100.0, nu=0.0, nrs=1, ncycle=1000000)
    mat0 = build_law163(p0)
    mat1 = build_law163(p1)

    # Compression with rho = 1.25 -> gama = 1 - 1.0/1.25 = 0.2
    # uvar1_old = 0.0, dt = 0.05 -> engineering rate = 0.2 / 0.05 = 4.0
    # deps = [-0.1, -0.05, -0.05] -> true rate = -(-0.2) / 0.05 = 4.0
    dt = 0.05
    deps = np.array([-0.1, -0.05, -0.05, 0.0, 0.0, 0.0])
    extra0 = {"rho": np.array([1.25])}
    extra1 = {"rho": np.array([1.25])}

    solid_update(mat0, np.zeros(6), deps, dt=dt, extra=extra0)
    solid_update(mat1, np.zeros(6), deps, dt=dt, extra=extra1)

    # Both produce matching volumetric strain rate when consistent
    assert math.isclose(float(extra0["epsd"][0]), float(extra1["epsd"][0]), rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 5. Yield table / curve lookup and fscale
# ---------------------------------------------------------------------------

def test_yield_curve_funct_table():
    """Verify yield lookup with a 1D FunctTable curve and end-slope extrapolation."""
    # Curve: (0.0, 20.0), (0.1, 40.0), (0.3, 100.0)
    # Slope segment 1: (40-20)/0.1 = 200.0
    # Slope segment 2: (100-40)/0.2 = 300.0
    fct = FunctTable(fct_id=1, x=[0.0, 0.1, 0.3], y=[20.0, 40.0, 100.0])
    p = Law163Params(rho0=1.0, e=1000.0, nu=0.0, tsc=1e20, table=fct, fscale=1.5)
    mat = build_law163(p)

    # Test point 1: gama = 0.05 -> sigy_val = 20 + 200 * 0.05 = 30.0
    # Scaled by fscale=1.5 -> sigy = -45.0, dsdgam = 200 * 1.5 = 300.0
    # Compressive trial stress -100.0 -> clamped to -45.0
    extra = {"rho": np.array([1.0 / (1.0 - 0.05)])}  # gama = 0.05
    sig = solid_update(mat, np.zeros(6), np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]), dt=0.0, extra=extra)
    assert math.isclose(sig[0], -45.0, rel_tol=1e-10)

    # Test point 2: extrapolation beyond end: gama = 0.4 -> 100 + 300 * 0.1 = 130.0
    # Scaled by 1.5 -> sigy = -195.0
    extra2 = {"rho": np.array([1.0 / (1.0 - 0.4)])}
    sig2 = solid_update(mat, np.zeros(6), np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0]), dt=0.0, extra=extra2)
    assert math.isclose(sig2[0], -195.0, rel_tol=1e-10)


def test_yield_2d_table_grid():
    """Verify yield lookup with 2D table grid (strain x strain rate)."""
    # 2D table grid:
    # xg = [0.0, 0.2]
    # rates = [0.0, 100.0]
    # Y = [[10.0, 20.0],
    #      [30.0, 60.0]]
    xg = np.array([0.0, 0.2])
    rates = np.array([0.0, 100.0])
    Y = np.array([[10.0, 20.0],
                  [30.0, 60.0]])
    tbl2d = (xg, rates, Y)

    p = Law163Params(rho0=1.0, e=2000.0, nu=0.0, tsc=1e20, damp=0.0, table=tbl2d, nrs=0, ncycle=1)
    mat = build_law163(p)

    # Evaluate at gama = 0.1, dgamdt = 50.0:
    # At rate 0: y(0.1) = 10 + 0.5*(30-10) = 20.0
    # At rate 100: y(0.1) = 20 + 0.5*(60-20) = 40.0
    # Interpolated at rate 50: 20 + 0.5*(40-20) = 30.0
    # dt = 0.001, deps_xx = -0.05 -> raw rate = 50.0
    extra = {"rho": np.array([1.0 / (1.0 - 0.1)]), "epsd": np.array([50.0])}
    sig = solid_update(mat, np.zeros(6), np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), dt=0.001, extra=extra)
    assert math.isclose(sig[0], -30.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 6. Viscous damping
# ---------------------------------------------------------------------------

def test_viscous_damping_stresses():
    """Verify viscous damping contribution sigv to total stress."""
    # E=100, nu=0.0 -> bulk=100/3, g=50, ssp0 = sqrt(100) = 10.0
    # damp = 0.2, le = 2.0, rho = 1.0, gama = 0.0 -> denom = 1.0
    # a = ssp0 * rho * damp * le / denom = 10.0 * 1.0 * 0.2 * 2.0 / 1.0 = 4.0
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, damp=0.20, tsc=1e20)
    mat = build_law163(p)

    dt = 0.01
    deps = np.array([0.001, 0.0, 0.0, 0.002, 0.0, 0.0])
    extra = {"le": np.array([2.0]), "rho": np.array([1.0])}

    sig_tot, epsp, c = solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra, return_tuple=True)

    # Elastic trial stress:
    # st_xx = 100.0 * 0.001 = 0.1
    # st_xy = 50.0 * 0.002 = 0.1
    # Strain rates: epsp_xx = 0.001 / 0.01 = 0.1, epsp_xy = 0.002 / 0.01 = 0.2
    # ldav = 0.1 / 3.0
    # sigv_xx = a * ((0.1 - ldav)/(1.0) + ldav/(1.0)) = a * 0.1 = 4.0 * 0.1 = 0.4
    # sigv_xy = a * epsp_xy / (2 * 1.0) = 4.0 * 0.2 / 2.0 = 0.4
    # Total:
    # sig_tot[0] = 0.1 + 0.4 = 0.5
    # sig_tot[3] = 0.1 + 0.4 = 0.5
    assert math.isclose(sig_tot[0], 0.5, rel_tol=1e-10)
    assert math.isclose(sig_tot[3], 0.5, rel_tol=1e-10)


def test_zero_viscous_damping_at_zero_dt():
    """Verify zero viscous stress when dt=0."""
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, damp=0.5, tsc=1e20)
    mat = build_law163(p)

    deps = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig = solid_update(mat, np.zeros(6), deps, dt=0.0, return_tuple=False)

    # Exactly purely elastic
    assert math.isclose(sig[0], 0.1, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 7. Sound speed
# ---------------------------------------------------------------------------

def test_sound_speed_solid_formulations():
    """Verify sound speed in elastic state, with table slope, and with viscous stiffness."""
    # E=100.0, nu=0.0, rho0=1.0 -> bulk=33.333, g=50.0 -> mod_eff = 33.333 + 4/3*50 = 100.0
    # c0 = sqrt(100.0 / 1.0) = 10.0
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, damp=0.10)
    mat = build_law163(p)

    c_static = sound_speed_solid(mat)
    assert math.isclose(c_static, 10.0, rel_tol=1e-12)

    # With array rho
    rhos = np.array([1.0, 4.0])
    c_arr = sound_speed_solid(mat, rho=rhos)
    assert math.isclose(c_arr[0], 10.0, rel_tol=1e-12)
    assert math.isclose(c_arr[1], 5.0, rel_tol=1e-12)

    # Sound speed with dynamic damping from solid_update:
    # When dt=0.01, a = 10.0 * 1.0 * 0.1 * 1.0 = 1.0
    # abs(a)/dt = 1.0 / 0.01 = 100.0
    # Total modulus = 100.0 + 100.0 = 200.0 -> c = sqrt(200) = 14.1421...
    sig, epsp, c_dyn = solid_update(mat, np.zeros(6), np.zeros(6), dt=0.01, return_tuple=True)
    assert math.isclose(c_dyn, math.sqrt(200.0), rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 8. Consistent solid tangent & shell update error
# ---------------------------------------------------------------------------

def test_consistent_solid_tangent_structure():
    """Verify consistent solid tangent tensor (n, 6, 6) matches cii, cij, and g."""
    p = Law163Params(rho0=1.0, e=120.0, nu=0.25)
    mat = build_law163(p)

    # 1D stress input
    sig_1d = np.zeros(6)
    D = consistent_solid_tangent(mat, sig_1d)
    assert D.shape == (1, 6, 6)

    # Check symmetry
    assert np.allclose(D[0], D[0].T)

    # Check diagonal and off-diagonal normal blocks
    assert math.isclose(D[0, 0, 0], p.cii, rel_tol=1e-12)
    assert math.isclose(D[0, 1, 1], p.cii, rel_tol=1e-12)
    assert math.isclose(D[0, 2, 2], p.cii, rel_tol=1e-12)

    assert math.isclose(D[0, 0, 1], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 0, 2], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 1, 2], p.cij, rel_tol=1e-12)

    # Check shear diagonal
    assert math.isclose(D[0, 3, 3], p.g, rel_tol=1e-12)
    assert math.isclose(D[0, 4, 4], p.g, rel_tol=1e-12)
    assert math.isclose(D[0, 5, 5], p.g, rel_tol=1e-12)

    # Batch (3, 6) input
    sig_batch = np.zeros((3, 6))
    D_batch = consistent_solid_tangent(mat, sig_batch)
    assert D_batch.shape == (3, 6, 6)
    assert np.allclose(D_batch[0], D_batch[1])


def test_shell_update_raises_not_implemented():
    """Verify shell_update raises NotImplementedError (solids only)."""
    p = Law163Params(rho0=1.0, e=100.0, nu=0.2)
    mat = build_law163(p)
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update(mat, np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# 9. Materials module dispatch integration
# ---------------------------------------------------------------------------

def test_materials_dispatcher_integration():
    """Verify pyradioss.materials top-level dispatchers wire LAW163 properly."""
    p = Law163Params(rho0=1.5, e=150.0, nu=0.2, tsc=10.0, damp=0.0)
    mat = build_law163(p)

    # 1. extra_shapes
    shapes = materials.extra_shapes(mat)
    assert "uvar163" in shapes
    assert "epsd163" in shapes

    # 2. needs_env
    assert materials.needs_env(mat) is True

    # 3. solid_update dispatcher
    sig_old = np.zeros(6)
    deps = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_new, epsp, c = materials.solid_update(mat, sig_old, deps, dt=0.0)
    assert math.isclose(sig_new[0], p.cii * 1e-3, rel_tol=1e-10)
    assert c is not None
    assert math.isclose(c, math.sqrt((p.bulk + 4.0/3.0 * p.g) / 1.5), rel_tol=1e-10)

    # 4. sound_speed dispatcher
    c_disp = materials.sound_speed(mat)
    assert math.isclose(c_disp, math.sqrt((p.bulk + 4.0/3.0 * p.g) / 1.5), rel_tol=1e-10)

    # 5. solid_tangent dispatcher
    D_disp = materials.solid_tangent(mat, sig_old)
    assert D_disp.shape == (1, 6, 6)
    assert math.isclose(D_disp[0, 0, 0], p.cii, rel_tol=1e-12)

    # 6. shell_update dispatcher raises
    with pytest.raises(NotImplementedError, match="solid elements only"):
        materials.shell_update(mat, np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# 10. Vectorized group evaluation consistency
# ---------------------------------------------------------------------------

def test_vectorized_multielement_consistency():
    """Verify vectorized group evaluation matches serial element-by-element evaluation."""
    p = Law163Params(rho0=1.0, e=200.0, nu=0.2, tsc=20.0, damp=0.15, table=60.0)
    mat = build_law163(p)

    np.random.seed(42)
    nel = 8
    sig_batch = np.random.uniform(-10.0, 10.0, (nel, 6))
    deps_batch = np.random.uniform(-0.02, 0.02, (nel, 6))
    rho_batch = np.random.uniform(0.8, 1.2, nel)
    le_batch = np.random.uniform(0.5, 2.0, nel)

    dt = 0.005
    extra_batch = {
        "rho": rho_batch.copy(),
        "le": le_batch.copy(),
        "epsd163": np.zeros(nel),
        "uvar163": np.zeros((nel, 2)),
    }

    # Vectorized execution
    sig_res, epsp_res, c_res = solid_update(
        mat, sig_batch.copy(), deps_batch.copy(), dt=dt, extra=extra_batch, return_tuple=True
    )

    # Element-by-element execution
    for i in range(nel):
        extra_elem = {
            "rho": np.array([rho_batch[i]]),
            "le": np.array([le_batch[i]]),
            "epsd163": np.array([0.0]),
            "uvar163": np.zeros((1, 2)),
        }
        sig_i, epsp_i, c_i = solid_update(
            mat, sig_batch[i].copy(), deps_batch[i].copy(), dt=dt, extra=extra_elem, return_tuple=True
        )
        assert np.allclose(sig_res[i], sig_i, atol=1e-12)
        assert math.isclose(epsp_res[i], epsp_i, abs_tol=1e-12)
        assert math.isclose(c_res[i], c_i, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 11. Additional thorough tests: perturbation tangent, plastic strain, table types
# ---------------------------------------------------------------------------

def test_plastic_strain_evolution():
    """Verify epsp tracks effective volumetric true strain plas = log(rho0/rho)."""
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0)
    mat = build_law163(p)

    # Initial state rho = rho0 -> plas = log(1.0) = 0.0
    sig, epsp, c = solid_update(mat, np.zeros(6), np.zeros(6), dt=0.0, return_tuple=True)
    assert math.isclose(float(epsp), 0.0, abs_tol=1e-12)

    # Compressed state rho = 2.0 -> plas = log(0.5) = -0.693147...
    extra = {"rho": np.array([2.0])}
    sig, epsp, c = solid_update(mat, np.zeros(6), np.zeros(6), dt=0.0, extra=extra, return_tuple=True)
    assert math.isclose(float(epsp), math.log(0.5), rel_tol=1e-10)


def test_consistent_solid_tangent_numerical_perturbation():
    """Verify consistent solid tangent matches finite-difference perturbation of solid_update."""
    p = Law163Params(rho0=1.0, e=150.0, nu=0.22, tsc=1e20, damp=0.0)
    mat = build_law163(p)

    sig_0 = np.zeros(6)
    deps_0 = np.array([1e-4, -5e-5, 2e-5, 3e-5, -1e-5, 4e-5])
    D_ana = consistent_solid_tangent(mat, sig_0)[0]

    delta = 1e-7
    D_num = np.zeros((6, 6))
    for j in range(6):
        deps_plus = deps_0.copy()
        deps_plus[j] += delta
        sig_plus = solid_update(mat, sig_0, deps_plus, dt=0.0, return_tuple=False)

        deps_minus = deps_0.copy()
        deps_minus[j] -= delta
        sig_minus = solid_update(mat, sig_0, deps_minus, dt=0.0, return_tuple=False)

        D_num[:, j] = (sig_plus - sig_minus) / (2.0 * delta)

    assert np.allclose(D_ana, D_num, rtol=1e-6, atol=1e-8)


def test_various_table_types_in_lookup():
    """Verify lookup with callable, dict with x/y, and resolved table."""
    # 1. Callable returning (sigy, dsdgam)
    def custom_table(gama, dgamdt):
        return 45.0 + 10.0 * gama, 10.0

    p_call = Law163Params(rho0=1.0, e=1000.0, nu=0.0, tsc=1e20, table=custom_table, damp=0.0)
    mat_call = build_law163(p_call)
    extra_call = {"rho": np.array([1.0 / (1.0 - 0.2)])}  # gama = 0.2 -> sigy = 45 + 2 = 47.0
    sig_call = solid_update(mat_call, np.zeros(6), np.array([-0.1, 0, 0, 0, 0, 0]), dt=0.0, extra=extra_call)
    assert math.isclose(sig_call[0], -47.0, rel_tol=1e-10)

    # 2. Dict with "x" and "y"
    dict_curve = {"x": [0.0, 0.1, 0.5], "y": [15.0, 35.0, 75.0]}
    p_dict = Law163Params(rho0=1.0, e=1000.0, nu=0.0, tsc=1e20, table=dict_curve, damp=0.0)
    mat_dict = build_law163(p_dict)
    extra_dict = {"rho": np.array([1.0 / (1.0 - 0.05)])}  # gama = 0.05 -> sigy = 15 + 200*0.05 = 25.0
    sig_dict = solid_update(mat_dict, np.zeros(6), np.array([-0.1, 0, 0, 0, 0, 0]), dt=0.0, extra=extra_dict)
    assert math.isclose(sig_dict[0], -25.0, rel_tol=1e-10)


def test_resolve_function_from_model():
    """Verify resolve hook maps tab_id from model.functions or model.tables."""
    class DummyModel:
        def __init__(self):
            self.functions = {
                55: FunctTable(55, [0.0, 0.2, 0.4], [10.0, 50.0, 90.0])
            }
            self.tables = {}

    model = DummyModel()
    p = Law163Params(rho0=1.0, e=500.0, tab_id=55)
    mat = build_law163(p)

    assert mat.params.get("table") is None
    resolve(mat, model)
    assert mat.params.get("table") is model.functions[55]


def test_viscous_damping_components_yz_zx():
    """Verify viscous damping on shear components yz and zx."""
    p = Law163Params(rho0=1.0, e=100.0, nu=0.0, damp=0.20, tsc=1e20)
    mat = build_law163(p)

    dt = 0.01
    deps = np.array([0.0, 0.0, 0.0, 0.0, 0.002, 0.004])
    extra = {"le": np.array([2.0]), "rho": np.array([1.0]), "sigv": np.zeros(6)}

    sig_tot, epsp, c = solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra, return_tuple=True)

    # Elastic: st_yz = 50 * 0.002 = 0.1, st_zx = 50 * 0.004 = 0.2
    # a = 10.0 * 1.0 * 0.2 * 2.0 = 4.0
    # epsp_yz = 0.002 / 0.01 = 0.2 -> sigv_yz = 4.0 * 0.2 / 2.0 = 0.4 -> sig_tot = 0.1 + 0.4 = 0.5
    # epsp_zx = 0.004 / 0.01 = 0.4 -> sigv_zx = 4.0 * 0.4 / 2.0 = 0.8 -> sig_tot = 0.2 + 0.8 = 1.0
    assert math.isclose(sig_tot[4], 0.5, rel_tol=1e-10)
    assert math.isclose(sig_tot[5], 1.0, rel_tol=1e-10)
    assert "sigv" in extra
    assert math.isclose(extra["sigv"][0, 4], 0.4, rel_tol=1e-10)
    assert math.isclose(extra["sigv"][0, 5], 0.8, rel_tol=1e-10)

