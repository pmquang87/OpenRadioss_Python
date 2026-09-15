"""
Comprehensive test suite for /MAT/LAW83 (/MAT/CONNECT, /MAT/SPR_JOU).

Verifies:
- Defensive empty array handling across solid_update and consistent_solid_tangent
- Shell update rejection (solids-only for CONNECT spotwelds)
- CFG vs direct dictionary parameter extraction and validation
- Error codes for nonpositive E and density
- Shear modulus default calculation G = E / 2.6
- Curve resolution tolerance and function evaluation
- Normal tension (E) and compression (E_comp under icomp=0, 1)
- Transverse shears (G in YZ and ZX)
- Pure normal yielding (sig_zz = fy * Rn)
- Pure shear yielding (tau = fy * Rs)
- Combined normal/shear anisotropic quadratic yield surface
- Tabulated work hardening (curve_y / FUN_A1)
- Rate-dependent scaling (FUN_A2, FUN_A3)
- Damage softening (1 - D)
- Cycle 0 static stability (dt <= 0)
- Sound speed evaluation c = sqrt(E / rho0)
- Consistent solid tangent symmetry and positive semidefiniteness
- Directional derivative consistency of tangent with solid_update
- Batched element vectorization equivalence.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from types import SimpleNamespace

from pyradioss.materials import law83_spotweld, solid_tangent, shell_update, solid_update
from pyradioss.model.entities import Material


def _make_law83(E=200000.0, G=0.0, nu=0.3, rho0=7800.0, sig_y=400.0,
                rn=1.0, rs=1.0, icomp=1, e_comp=220000.0, alpha=0.0, beta=2.0) -> Material:
    rec = {
        "id": 83,
        "title": "SPOTWELD_LAW83",
        "MAT_RHO": rho0,
        "MAT_E": E,
        "MAT_G": G,
        "MAT_ECOMP": e_comp,
        "COMP_OPT": icomp,
        "MAT_R00": rn,
        "MAT_R45": rs,
        "MAT_ALPHA": alpha,
        "MAT_Beta": beta,
        "sig_y": sig_y,
        "nu": nu,
    }
    return law83_spotweld.build_law83(rec)


# ============================================================================
# 1. Empty arrays & shell rejection
# ============================================================================

def test_law83_empty_arrays():
    mat = _make_law83()
    sig = np.empty((0, 6))
    deps = np.empty((0, 6))
    s_out, epsp_out, c = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    assert s_out.shape == (0, 6)
    assert epsp_out.shape == (0,)
    assert c.shape == (0,)


def test_law83_tangent_empty_arrays():
    mat = _make_law83()
    sig = np.empty((0, 6))
    D = law83_spotweld.consistent_solid_tangent(mat, sig=sig)
    assert D.shape == (0, 6, 6)


def test_law83_shell_update_rejection():
    mat = _make_law83()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="solid elements only"):
        law83_spotweld.shell_update(mat, sig, deps, None, 1.0)
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update(mat, sig, deps, None, 1.0)


# ============================================================================
# 2. CFG vs Direct parameter construction & validation
# ============================================================================

def test_law83_build_from_direct_dict():
    rec = {
        "id": 1,
        "rho": 7850.0,
        "E": 210000.0,
        "G": 80000.0,
        "nu": 0.3,
        "sig_y": 350.0,
        "rn": 1.2,
        "rs": 0.8,
        "icomp": 0,
        "e_comp": 210000.0,
    }
    mat = law83_spotweld.build_law83(rec)
    assert mat.law == 83
    assert mat.rho0 == 7850.0
    assert mat.params["E"] == 210000.0
    assert mat.params["G"] == 80000.0
    assert mat.params["rn"] == 1.2
    assert mat.params["rs"] == 0.8
    assert mat.params["icomp"] == 0


def test_law83_build_validation_e_nonpositive():
    rec = {"E": 0.0, "rho": 7800.0}
    with pytest.raises(ValueError, match="Young modulus E must be positive"):
        law83_spotweld.build_law83(rec)

    rec_neg = {"E": -100.0, "rho": 7800.0}
    with pytest.raises(ValueError, match="Young modulus E must be positive"):
        law83_spotweld.build_law83(rec_neg)


def test_law83_build_validation_density_nonpositive():
    rec = {"E": 200000.0, "rho": 0.0}
    with pytest.raises(ValueError, match="Density rho0 must be positive"):
        law83_spotweld.build_law83(rec)


def test_law83_default_shear_modulus():
    # When G <= 0, G defaults to E / (2 * 1.3) = E / 2.6
    mat = _make_law83(E=260000.0, G=0.0)
    expected_G = 260000.0 / 2.6
    assert math.isclose(mat.params["G"], expected_G)


# ============================================================================
# 3. Curve resolution tolerance
# ============================================================================

def test_law83_resolve_tolerance_missing_functions():
    mat = _make_law83()
    # resolve with None or empty functions dict should not crash
    law83_spotweld.resolve(mat, None, None)

    class DummyModel:
        functions = {}

    law83_spotweld.resolve(mat, DummyModel(), None)


def test_law83_resolve_valid_curves():
    mat = _make_law83()
    mat.params["id_yield"] = 101

    class DummyModel:
        functions = {
            101: SimpleNamespace(id=101, title="YLD", x=np.array([0.0, 0.05, 0.2]), y=np.array([400.0, 500.0, 650.0]))
        }

    law83_spotweld.resolve(mat, DummyModel(), None)
    assert "curve_y_x" in mat.params
    assert "curve_y_y" in mat.params
    assert np.allclose(mat.params["curve_y_x"], [0.0, 0.05, 0.2])


# ============================================================================
# 4. Elastic kinematics: normal tension, compression, and shears
# ============================================================================

def test_law83_linear_elastic_normal_tension():
    mat = _make_law83(E=200000.0, sig_y=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.001  # dz = 0.001
    sig, epsp, c = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    expected_szz = 200000.0 * 0.001
    assert math.isclose(sig[0, 2], expected_szz)
    assert epsp[0] == 0.0  # no plastic strain


def test_law83_linear_elastic_normal_compression_icomp1():
    # icomp=1: compression uses E_comp
    mat = _make_law83(E=200000.0, e_comp=250000.0, icomp=1, sig_y=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = -0.001  # compressive increment
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    expected_szz = -250000.0 * 0.001
    assert math.isclose(sig[0, 2], expected_szz)


def test_law83_linear_elastic_normal_compression_icomp0():
    # icomp=0: compression uses Young's modulus E symmetrically
    mat = _make_law83(E=200000.0, e_comp=250000.0, icomp=0, sig_y=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = -0.001
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    expected_szz = -200000.0 * 0.001
    assert math.isclose(sig[0, 2], expected_szz)


def test_law83_linear_elastic_transverse_shears():
    G = 75000.0
    mat = _make_law83(E=200000.0, G=G, sig_y=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 4] = 0.002  # YZ
    deps[0, 5] = 0.003  # ZX
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    assert math.isclose(sig[0, 4], G * 0.002)
    assert math.isclose(sig[0, 5], G * 0.003)
    assert epsp[0] == 0.0


def test_law83_3component_strain_tensor_input():
    # Calling solid_update with (n, 3) deps [zz, yz, zx]
    mat = _make_law83(E=200000.0, G=75000.0, sig_y=1000.0)
    sig = np.zeros((1, 6))
    deps3 = np.array([[0.001, 0.002, 0.003]])
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps3, None, 1.0)
    assert math.isclose(sig[0, 2], 200000.0 * 0.001)
    assert math.isclose(sig[0, 4], 75000.0 * 0.002)
    assert math.isclose(sig[0, 5], 75000.0 * 0.003)


# ============================================================================
# 5. Plasticity return mapping & yield surface
# ============================================================================

def test_law83_pure_normal_yield():
    # Pure normal tension yielding: rn = 1.0, sig_y = 300.0
    mat = _make_law83(E=200000.0, sig_y=300.0, rn=1.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.005  # trial stress = 1000 MPa >> 300 MPa
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    # Stress should be returned onto yield surface sig_zz = sig_y = 300
    assert math.isclose(sig[0, 2], 300.0, rel_tol=1e-3)
    assert epsp[0] > 0.0  # plastic strain accumulated


def test_law83_pure_shear_yield():
    # Pure shear yielding: rs = 1.0, sig_y = 200.0
    mat = _make_law83(E=200000.0, G=80000.0, sig_y=200.0, rs=1.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 4] = 0.01  # trial shear = 800 MPa >> 200 MPa
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    assert math.isclose(sig[0, 4], 200.0, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_combined_normal_and_shear_yield():
    # Mixed normal and shear: an * szz^2 + ast * (syz^2 + szx^2) = fy^2
    mat = _make_law83(E=200000.0, G=80000.0, sig_y=400.0, rn=1.0, rs=1.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.004  # trial szz = 800
    deps[0, 4] = 0.008  # trial syz = 640
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0)
    # Effective stress must equal 400.0 to high accuracy
    eff_stress = math.sqrt(sig[0, 2]**2 + sig[0, 4]**2)
    assert math.isclose(eff_stress, 400.0, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_tabulated_yield_hardening():
    # Plastic hardening using curve_y
    mat = _make_law83(E=200000.0)
    mat.params["curve_y_x"] = np.array([0.0, 0.01, 0.05])
    mat.params["curve_y_y"] = np.array([300.0, 400.0, 500.0])
    mat.params["curve_y_s"] = np.array([10000.0, 2500.0, 2500.0])

    extra = {"epsp": np.zeros(1), "asrate": np.zeros(1)}
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    # Step 1: yield at initial yield stress ~ 300
    deps[0, 2] = 0.002  # trial = 400
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0, extra=extra)
    sig1 = sig[0, 2]
    assert sig1 >= 300.0

    # Step 2: further plastic loading expands yield surface
    deps[0, 2] = 0.003
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0, extra=extra)
    sig2 = sig[0, 2]
    assert sig2 > sig1  # hardening increases flow stress


def test_law83_rate_dependent_scaling():
    # Rate functions curve_n and curve_t
    mat = _make_law83(E=200000.0, sig_y=300.0)
    # Double strength at rate 100
    mat.params["curve_n_x"] = np.array([0.0, 100.0])
    mat.params["curve_n_y"] = np.array([1.0, 2.0])

    extra = {"epsp": np.zeros(1), "asrate": np.zeros(1)}
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.01
    dt = 0.0001  # strain rate = 0.01 / 0.0001 = 100
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt, extra=extra)
    # rn is scaled by factor 2 -> normal strength increases
    assert sig[0, 2] > 300.0


def test_law83_damage_softening():
    # Damage reduces yield stress by (1 - D)
    mat = _make_law83(E=200000.0, sig_y=400.0)
    extra = {"epsp": np.zeros(1), "asrate": np.zeros(1), "dmg": np.array([0.5])}
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.005  # trial = 1000
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, 1.0, extra=extra)
    # Effective yield stress is 400 * (1 - 0.5) = 200
    assert math.isclose(sig[0, 2], 200.0, rel_tol=1e-3)


def test_law83_cycle_0_static_stability():
    mat = _make_law83(E=200000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # dt <= 0 should safely evaluate without crashing or dividing by zero
    sig, epsp, c = law83_spotweld.solid_update(mat, sig, deps, None, dt=0.0)
    assert np.allclose(sig, 0.0)
    assert epsp[0] == 0.0
    assert c[0] > 0.0


def test_law83_sound_speed_evaluation():
    E = 200000.0
    rho0 = 8000.0
    mat = _make_law83(E=E, rho0=rho0)
    _, _, c = law83_spotweld.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), None, 1.0)
    expected_c = math.sqrt(E / rho0)
    assert math.isclose(c[0], expected_c)


# ============================================================================
# 6. Tangents (consistent_solid_tangent)
# ============================================================================

def test_law83_consistent_solid_tangent_symmetry():
    mat = _make_law83(E=200000.0, G=77000.0, e_comp=250000.0, icomp=1)
    # Test both positive and negative normal stress
    sig_tens = np.array([[0.0, 0.0, 100.0, 0.0, 50.0, 50.0]])
    D_tens = law83_spotweld.consistent_solid_tangent(mat, sig=sig_tens)[0]
    assert np.allclose(D_tens, D_tens.T)

    sig_comp = np.array([[0.0, 0.0, -100.0, 0.0, 50.0, 50.0]])
    D_comp = law83_spotweld.consistent_solid_tangent(mat, sig=sig_comp)[0]
    assert np.allclose(D_comp, D_comp.T)
    # Check that D_comp[2, 2] uses E_comp
    assert math.isclose(D_comp[2, 2], 250000.0)
    assert math.isclose(D_tens[2, 2], 200000.0)


def test_law83_consistent_solid_tangent_positive_semidefinite():
    mat = _make_law83(E=200000.0, G=77000.0)
    sig = np.array([[0.0, 0.0, 200.0, 0.0, 100.0, 100.0]])
    D = law83_spotweld.consistent_solid_tangent(mat, sig=sig)[0]
    eigvals = np.linalg.eigvalsh(D)
    assert np.all(eigvals >= -1e-12)


def test_law83_consistent_solid_tangent_directional_derivative():
    mat = _make_law83(E=200000.0, G=80000.0, sig_y=500.0)
    sig0 = np.array([[0.0, 0.0, 100.0, 0.0, 50.0, 30.0]])
    D = law83_spotweld.consistent_solid_tangent(mat, sig=sig0)[0]

    # Small perturbation d_eps in [zz, yz, zx]
    deps = np.zeros((1, 6))
    deps[0, 2] = 1e-6
    deps[0, 4] = 2e-6
    deps[0, 5] = 1.5e-6

    sig_new, _, _ = law83_spotweld.solid_update(mat, sig0.copy(), deps, None, 1.0)
    d_sig_actual = sig_new[0] - sig0[0]
    d_sig_tangent = D @ deps[0]
    assert np.allclose(d_sig_actual, d_sig_tangent, atol=1e-8)


def test_law83_solid_tangent_dispatch():
    mat = _make_law83()
    sig = np.zeros((2, 6))
    D = solid_tangent(mat, sig, None, None)
    assert D.shape == (2, 6, 6)


def test_law83_batched_vectorization_equivalence():
    mat = _make_law83(E=200000.0, G=75000.0, sig_y=350.0)
    n = 4
    sig_batch = np.array([
        [0.0, 0.0, 50.0, 0.0, 20.0, 10.0],
        [0.0, 0.0, -80.0, 0.0, 40.0, 30.0],
        [0.0, 0.0, 400.0, 0.0, 200.0, 150.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    deps_batch = np.array([
        [0.0, 0.0, 0.001, 0.0, 0.0005, 0.0002],
        [0.0, 0.0, -0.001, 0.0, 0.001, 0.0005],
        [0.0, 0.0, 0.002, 0.0, 0.002, 0.001],
        [0.0, 0.0, 0.0005, 0.0, 0.0002, 0.0001],
    ])

    extra_batch = {"epsp": np.zeros(n), "asrate": np.zeros(n)}
    s_batch, epsp_batch, c_batch = law83_spotweld.solid_update(
        mat, sig_batch.copy(), deps_batch, None, 1.0, extra=extra_batch
    )
    D_batch = law83_spotweld.consistent_solid_tangent(mat, sig=sig_batch)

    for i in range(n):
        extra_single = {"epsp": np.zeros(1), "asrate": np.zeros(1)}
        s_single, epsp_single, c_single = law83_spotweld.solid_update(
            mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1], None, 1.0, extra=extra_single
        )
        D_single = law83_spotweld.consistent_solid_tangent(mat, sig=sig_batch[i:i+1])

        assert np.allclose(s_batch[i], s_single[0], atol=1e-12)
        assert math.isclose(epsp_batch[i], epsp_single[0], abs_tol=1e-12)
        assert math.isclose(c_batch[i], c_single[0], rel_tol=1e-12)
        assert np.allclose(D_batch[i], D_single[0], atol=1e-12)
