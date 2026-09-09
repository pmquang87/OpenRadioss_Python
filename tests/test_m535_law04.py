"""Tests for M535 Wave 1: LAW4 (Hydrodynamic Johnson-Cook).

Validates pyradioss/materials/law04_hyd_jcook.py against:
- engine/source/materials/mat/mat004/m4law.F
- starter/source/materials/mat/mat004/hm_read_mat04.F
- config/CFG/radioss110/MAT/matl4_hyd_jcook.cfg
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law04_hyd_jcook as law04
from pyradioss.materials import law01_elastic


def _make_law04(E=210000.0, nu=0.3, rho0=7.8e-6, A=400.0, B=500.0, N=0.5,
                sig_max=0.0, eps_max=0.0, C=0.0, eps0=1.0, m=1.0,
                Tmelt=0.0, Tmax=0.0, rho_cp=0.0, T0=300.0, pmin=0.0, **kwargs):
    rec = {
        "id": 1,
        "title": "Steel_LAW4",
        "density": rho0,
        "params": {
            "E": E,
            "nu": nu,
            "A": A,
            "B": B,
            "N": N,
            "sig_max": sig_max,
            "eps_max": eps_max,
            "C": C,
            "eps0": eps0,
            "m": m,
            "Tmelt": Tmelt,
            "Tmax": Tmax,
            "rho_cp": rho_cp,
            "T0": T0,
            "pmin": pmin,
            **kwargs,
        },
    }
    return law04.build_law04(rec)


# ============================================================================
# 1. Empty array handling
# ============================================================================

def test_law04_empty_array_solid_update():
    mat = _make_law04()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    s_out, epsp_out, c_out = law04.solid_update(mat, sig, deps, None, 0.01)
    assert s_out.shape == (0, 6)
    assert c_out is None


def test_law04_empty_array_tangent():
    mat = _make_law04()
    sig = np.zeros((0, 6))
    D = law04.consistent_solid_tangent(mat, sig, None, None)
    assert D.shape == (0, 6, 6)


# ============================================================================
# 2. Validation in build_law04
# ============================================================================

def test_law04_invalid_density():
    with pytest.raises(ValueError, match="Initial density rho0 must be > 0"):
        _make_law04(rho0=0.0)
    with pytest.raises(ValueError, match="Initial density rho0 must be > 0"):
        _make_law04(rho0=-1.0)


def test_law04_invalid_young_modulus():
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        _make_law04(E=0.0)
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        _make_law04(E=-1000.0)


def test_law04_invalid_poisson_ratio():
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        _make_law04(nu=-0.1)
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        _make_law04(nu=0.5)
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        _make_law04(nu=0.6)


# ============================================================================
# 3. Defaults & parameter clamps
# ============================================================================

def test_law04_parameter_defaults_and_clamps():
    # Test N clamping: N=0 or N=1 -> 1.0001
    mat_n0 = _make_law04(N=0.0)
    assert mat_n0.params["N"] == pytest.approx(1.0001)
    assert mat_n0.params["MAT_HARD"] == pytest.approx(1.0001)

    mat_n1 = _make_law04(N=1.0)
    assert mat_n1.params["N"] == pytest.approx(1.0001)

    # Test defaults when 0
    mat = _make_law04(eps_max=0.0, sig_max=0.0, C=0.0, eps0=0.0, m=0.0,
                      Tmelt=0.0, Tmax=0.0, T0=0.0, pmin=0.0)
    assert mat.params["eps_max"] == 1e20
    assert mat.params["sig_max"] == 1e20
    assert mat.params["eps0"] == 1.0  # CC == 0 -> eps0 = 1.0
    assert mat.params["m"] == 1.0     # CM == 0 -> m = 1.0
    assert mat.params["Tmelt"] == 1e20
    assert mat.params["Tmax"] == 1e20
    assert mat.params["T0"] == 300.0  # T0 <= 0 -> 300.0
    assert mat.params["pmin"] == -1e30

    # Computed elastic constants
    E = 210000.0
    nu = 0.3
    G_expected = E / (2.0 * (1.0 + nu))
    K_expected = E / (3.0 * (1.0 - 2.0 * nu))
    assert mat.params["G"] == pytest.approx(G_expected)
    assert mat.params["K"] == pytest.approx(K_expected)


# ============================================================================
# 4. Parameter synchronization (_ensure_params & CFG keys)
# ============================================================================

def test_law04_ensure_params_cfg_keys():
    # Build from CFG-style keys
    rec = {
        "id": 4,
        "title": "CFG_Material",
        "density": 7.85e-6,
        "params": {
            "MAT_E": 200000.0,
            "MAT_NU": 0.28,
            "MAT_SIGY": 350.0,
            "MAT_BETA": 450.0,
            "MAT_HARD": 0.45,
            "MAT_EPS": 0.35,
            "MAT_SIG": 800.0,
            "MAT_SRC": 0.02,
            "MAT_SRP": 0.001,
            "MAT_M": 0.9,
            "MAT_TMELT": 1800.0,
            "MAT_TMAX": 1200.0,
            "MAT_SPHEAT": 3.5e6,
            "MAT_T0": 293.15,
            "MAT_PC": -500.0,
        },
    }
    mat = law04.build_law04(rec)
    p = law04._ensure_params(mat)

    # Check that direct keys match CFG keys
    assert p["E"] == pytest.approx(200000.0)
    assert p["nu"] == pytest.approx(0.28)
    assert p["A"] == pytest.approx(350.0)
    assert p["B"] == pytest.approx(450.0)
    assert p["N"] == pytest.approx(0.45)
    assert p["eps_max"] == pytest.approx(0.35)
    assert p["sig_max"] == pytest.approx(800.0)
    assert p["C"] == pytest.approx(0.02)
    assert p["eps0"] == pytest.approx(0.001)
    assert p["m"] == pytest.approx(0.9)
    assert p["Tmelt"] == pytest.approx(1800.0)
    assert p["Tmax"] == pytest.approx(1200.0)
    assert p["rho_cp"] == pytest.approx(3.5e6)
    assert p["T0"] == pytest.approx(293.15)
    assert p["pmin"] == pytest.approx(-500.0)


# ============================================================================
# 5. Embedded Polynomial EOS support
# ============================================================================

def test_law04_embedded_polynomial_eos():
    rec = {
        "id": 5,
        "title": "LAW4_with_EOS",
        "density": 8.9e-6,
        "params": {
            "E": 110000.0,
            "nu": 0.34,
            "A": 100.0,
            "B": 300.0,
            "N": 0.3,
            "c0": 0.0,
            "c1": 130000.0,
            "c2": 5000.0,
            "c3": 1000.0,
            "c4": 1.5,
            "c5": 0.2,
            "e0": 10.0,
            "psh": 1.0,
            "pmin": -2000.0,
        },
    }
    mat = law04.build_law04(rec)
    assert mat.eos is not None
    assert mat.eos.kind == "POLYNOMIAL"
    assert mat.eos.rho0 == pytest.approx(8.9e-6)
    assert mat.eos.params["c1"] == pytest.approx(130000.0)
    assert mat.eos.params["c4"] == pytest.approx(1.5)
    assert mat.eos.params["e0"] == pytest.approx(10.0)
    assert mat.eos.params["psh"] == pytest.approx(1.0)


# ============================================================================
# 6. Registry check
# ============================================================================

def test_law04_registry():
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    assert MAT_PHYSICS_REGISTRY["LAW4"] is law04.build_law04
    assert MAT_PHYSICS_REGISTRY["HYD_JCOOK"] is law04.build_law04
    assert MAT_PHYSICS_REGISTRY["JCOOK_HYD"] is law04.build_law04
    assert MAT_PHYSICS_REGISTRY["4"] is law04.build_law04


# ============================================================================
# 7. Shell update Not implemented
# ============================================================================

def test_law04_shell_update_not_implemented():
    mat = _make_law04()
    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        law04.shell_update(mat, sig, deps)


# ============================================================================
# 8. Elastic increment & Sound speed
# ============================================================================

def test_law04_elastic_step():
    mat = _make_law04(E=210000.0, nu=0.3, rho0=7.8e-6, A=500.0, B=0.0)
    G = mat.params["G"]
    K = mat.params["K"]

    # Small shear strain increment: gamma_xy = 0.0005
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.0005, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_out, epsp_out, c_out = law04.solid_update(mat, sig, deps, epsp, dt=1e-5)

    # Elastic: sigma_xy = G * gamma_xy
    assert sig_out[0, 3] == pytest.approx(G * 0.0005)
    assert epsp_out[0] == 0.0

    # Sound speed: c = sqrt((K + 4/3*G)/rho0)
    c_expected = math.sqrt((K + (4.0 / 3.0) * G) / 7.8e-6)
    assert c_out[0] == pytest.approx(c_expected)


# ============================================================================
# 9. Plastic yield and power-law hardening (IPLA=0)
# ============================================================================

def test_law04_plastic_flow_and_hardening():
    # Pure shear loading causing yield
    A = 300.0
    B = 400.0
    N = 0.5
    mat = _make_law04(E=200000.0, nu=0.3, A=A, B=B, N=N)
    G = mat.params["G"]

    sig = np.zeros((1, 6))
    # Imposed large shear strain: trial s_xy = G * 0.01 = 769.2 MPa > 300 / sqrt(3)
    gamma_xy = 0.01
    deps = np.array([[0.0, 0.0, 0.0, gamma_xy, 0.0, 0.0]])
    epsp = np.zeros(1)

    # Step 1: initial epsp = 0.0 -> yield stress = A = 300.0
    sig_out, epsp_out, c_out = law04.solid_update(mat, sig, deps, epsp, dt=1e-5)

    # Plastic strain must have increased
    assert epsp_out[0] > 0.0

    # In IPLA=0 (m4law.F), stress is projected to sigma_y at start of step
    s_xy = sig_out[0, 3]
    svm = math.sqrt(3.0) * abs(s_xy)
    assert svm == pytest.approx(A, rel=1e-5)

    # Step 2: start from updated epsp_out -> yield stress should now be A + B * epsp_out^N
    sig_out2, epsp_out2, _ = law04.solid_update(mat, sig_out, deps, epsp_out, dt=1e-5)
    s_xy2 = sig_out2[0, 3]
    svm2 = math.sqrt(3.0) * abs(s_xy2)
    expected_sy2 = A + B * (epsp_out[0] ** N)
    assert svm2 == pytest.approx(expected_sy2, rel=1e-5)


def test_law04_sig_max_cap():
    # Maximum stress cap
    A = 400.0
    B = 200.0
    N = 0.5
    sig_max = 450.0
    mat = _make_law04(A=A, B=B, N=N, sig_max=sig_max)

    # Initial epsp = 0.09 -> A + B * sqrt(0.09) = 400 + 60 = 460 > sig_max (450)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    epsp = np.array([0.09])

    sig_out, epsp_out, _ = law04.solid_update(mat, sig, deps, epsp, dt=1e-5)

    # Verify von Mises equivalent stress is capped at sig_max
    p = (sig_out[0, 0] + sig_out[0, 1] + sig_out[0, 2]) / 3.0
    s_dev = sig_out[0, :3] - p
    j2 = 0.5 * np.sum(s_dev ** 2)
    svm = math.sqrt(3.0 * j2)
    assert svm == pytest.approx(sig_max, rel=1e-5)


def test_law04_eps_max_failure():
    # Hardening drops to 0 when epsp > eps_max
    A = 300.0
    B = 500.0
    N = 0.5
    eps_max = 0.02
    mat = _make_law04(A=A, B=B, N=N, eps_max=eps_max)

    sig = np.zeros((1, 6))
    deps = np.array([[0.001, -0.0005, -0.0005, 0.0, 0.0, 0.0]])
    # Already beyond failure plastic strain
    epsp = np.array([0.025])

    sig_out, epsp_out, _ = law04.solid_update(mat, sig, deps, epsp, dt=1e-5)
    # When epsp > eps_max, C_H = 0 -> sigma_y = 0 -> deviatoric stress is 0
    p = (sig_out[0, 0] + sig_out[0, 1] + sig_out[0, 2]) / 3.0
    s_dev = sig_out[0, :3] - p
    assert np.allclose(s_dev, 0.0)


# ============================================================================
# 10. Strain rate sensitivity (C_E)
# ============================================================================

def test_law04_strain_rate_sensitivity():
    A = 300.0
    C = 0.05
    eps0 = 10.0
    mat = _make_law04(A=A, B=0.0, C=C, eps0=eps0)

    # Use pure shear strain gamma_xy = 0.01 (trial stress ~ 800 MPa >> 300)
    deps = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])

    # 1. dt with rate <= eps0 -> C_E = 1.0
    dt_slow = 0.01  # eps_dot = 0.5 * 0.01 / 0.01 = 0.5 <= eps0 (10.0)
    sig1 = np.zeros((1, 6))
    sig1_out, epsp1, _ = law04.solid_update(mat, sig1, deps, np.zeros(1), dt=dt_slow)

    # 2. dt with rate > eps0 -> C_E = 1.0 + C * ln(eps_dot / eps0)
    dt_fast = 5e-7  # eps_dot = 0.5 * 0.01 / 5e-7 = 10000.0 > 10.0
    sig2 = np.zeros((1, 6))
    sig2_out, epsp2, _ = law04.solid_update(mat, sig2, deps, np.zeros(1), dt=dt_fast)

    rate = (0.5 * 0.01) / 5e-7
    expected_ce = 1.0 + C * math.log(rate / eps0)

    # Compare yield stresses (via von Mises deviator)
    svm1 = math.sqrt(3.0) * abs(sig1_out[0, 3])
    svm2 = math.sqrt(3.0) * abs(sig2_out[0, 3])

    assert svm1 == pytest.approx(A, rel=1e-4)
    assert svm2 == pytest.approx(A * expected_ce, rel=1e-4)


# ============================================================================
# 11. Temperature & Thermal softening (C_T) & Melting
# ============================================================================

def test_law04_thermal_softening():
    A = 400.0
    T0 = 300.0
    Tmelt = 1500.0
    Tmax = 1000.0
    m = 0.8
    mat = _make_law04(A=A, B=0.0, T0=T0, Tmelt=Tmelt, Tmax=Tmax, m=m)

    # Element at 600 K: T0 < T < Tmax
    T_curr = 600.0
    tstar = (T_curr - T0) / (Tmelt - T0)
    expected_ct = 1.0 - (tstar ** m)
    expected_sy = A * expected_ct

    deps = np.array([[0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0]])
    extra = {"temp": np.array([T_curr])}
    sig = np.zeros((1, 6))
    sig_out, _, _ = law04.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    p = np.mean(sig_out[0, :3])
    svm = math.sqrt(1.5 * np.sum((sig_out[0, :3] - p) ** 2))
    assert svm == pytest.approx(expected_sy, rel=1e-4)


def test_law04_thermal_softening_above_tmax():
    A = 400.0
    T0 = 300.0
    Tmelt = 1500.0
    Tmax = 1000.0
    m = 0.8
    mat = _make_law04(A=A, B=0.0, T0=T0, Tmelt=Tmelt, Tmax=Tmax, m=m)

    # Element at 1200 K: T > Tmax -> m_eff = 1.0
    T_curr = 1200.0
    tstar = (T_curr - T0) / (Tmelt - T0)
    expected_ct = 1.0 - tstar  # exponent 1.0
    expected_sy = A * expected_ct

    deps = np.array([[0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0]])
    extra = {"temp": np.array([T_curr])}
    sig = np.zeros((1, 6))
    sig_out, _, _ = law04.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    p = np.mean(sig_out[0, :3])
    svm = math.sqrt(1.5 * np.sum((sig_out[0, :3] - p) ** 2))
    assert svm == pytest.approx(expected_sy, rel=1e-4)


def test_law04_melting():
    A = 400.0
    T0 = 300.0
    Tmelt = 1500.0
    mat = _make_law04(A=A, B=100.0, T0=T0, Tmelt=Tmelt)

    # Element at or above Tmelt (1550 K)
    extra = {"temp": np.array([1550.0])}
    deps = np.array([[0.005, -0.0025, -0.0025, 0.002, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    sig_out, epsp_out, _ = law04.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    # Deviatoric stress is completely relaxed to 0
    p = np.mean(sig_out[0, :3])
    s_dev = sig_out[0, :3] - p
    assert np.allclose(s_dev, 0.0)
    assert np.allclose(sig_out[0, 3:], 0.0)


def test_law04_adiabatic_temperature_rise():
    A = 400.0
    rho_cp = 3.5e6
    mat = _make_law04(A=A, B=0.0, rho_cp=rho_cp, T0=300.0)

    # Plastic step
    deps = np.array([[0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0]])
    temp_arr = np.array([300.0])
    extra = {"temp": temp_arr}
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)

    sig_out, epsp_out, _ = law04.solid_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)

    dpla = epsp_out[0]
    expected_dT = (A * dpla) / rho_cp
    assert extra["temp"][0] == pytest.approx(300.0 + expected_dT, rel=1e-5)


# ============================================================================
# 12. Hydrodynamic pressure & Cutoff
# ============================================================================

def test_law04_pressure_cutoff():
    pmin = -100.0
    mat = _make_law04(pmin=pmin)
    K = mat.params["K"]

    # Tensile volumetric strain causing pressure < pmin
    # tr(deps) = -0.01 -> p_new = -0.01 * K < -100.0
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))

    sig_out, _, _ = law04.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5)
    p_out = np.mean(sig_out[0, :3])
    assert p_out == pytest.approx(pmin)


def test_law04_density_based_pressure():
    mat = _make_law04(rho0=8000.0)
    K = mat.params["K"]

    # Compressed density: rho = 8080.0 -> mu = 8080/8000 - 1 = 0.01
    extra = {"rho": np.array([8080.0])}
    deps = np.zeros((1, 6))
    sig = np.zeros((1, 6))

    sig_out, _, _ = law04.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)
    p_out = np.mean(sig_out[0, :3])
    expected_p = -K * (8080.0 / 8000.0 - 1.0)
    assert p_out == pytest.approx(expected_p)


# ============================================================================
# 13. Consistent Solid Algorithmic Tangent (Simo & Hughes)
# ============================================================================

def test_law04_consistent_solid_tangent_elastic():
    mat = _make_law04(E=210000.0, nu=0.3)
    G = mat.params["G"]
    K = mat.params["K"]
    lam = K - 2.0 * G / 3.0

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    epsp_incr = np.zeros(1)

    D = law04.consistent_solid_tangent(mat, sig, epsp, epsp_incr)[0]

    # Verify isotropic elastic matrix
    assert D[0, 0] == pytest.approx(lam + 2.0 * G)
    assert D[0, 1] == pytest.approx(lam)
    assert D[3, 3] == pytest.approx(G)
    assert D[0, 3] == pytest.approx(0.0)


def test_law04_consistent_solid_tangent_plastic():
    mat = _make_law04(E=210000.0, nu=0.3, A=300.0, B=400.0, N=0.5)
    sig = np.array([[200.0, -100.0, -100.0, 50.0, 0.0, 0.0]])
    epsp = np.array([0.01])
    epsp_incr = np.array([0.001])

    D = law04.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
    assert D.shape == (1, 6, 6)

    # Check plastic softening: diagonal component is softer than elastic
    C_el = law01_elastic.solid_tangent(mat)
    assert D[0, 0, 0] < C_el[0, 0]

    # Check symmetry of the tangent tensor
    assert np.allclose(D[0], D[0].T, atol=1e-8)


# ============================================================================
# 14. Batch Vectorization
# ============================================================================

def test_law04_batch_vectorization():
    mat = _make_law04(A=300.0, B=200.0, N=0.5, Tmelt=1500.0)

    # 3 elements:
    # 0: elastic
    # 1: plastic
    # 2: melted
    nel = 3
    sig = np.zeros((nel, 6))
    deps = np.array([
        [0.0001, -0.00005, -0.00005, 0.0, 0.0, 0.0],
        [0.005, -0.0025, -0.0025, 0.001, 0.0, 0.0],
        [0.005, -0.0025, -0.0025, 0.001, 0.0, 0.0],
    ])
    epsp = np.zeros(nel)
    extra = {"temp": np.array([300.0, 300.0, 1600.0])}

    sig_out, epsp_out, c_out = law04.solid_update(mat, sig, deps, epsp, dt=1e-4, extra=extra)

    # Element 0: elastic
    assert epsp_out[0] == 0.0

    # Element 1: plastic
    assert epsp_out[1] > 0.0

    # Element 2: melted
    p2 = np.mean(sig_out[2, :3])
    s2 = sig_out[2, :3] - p2
    assert np.allclose(s2, 0.0)
    assert np.allclose(sig_out[2, 3:], 0.0)

    # Tangents for all 3
    D_batch = law04.consistent_solid_tangent(mat, sig_out, epsp_out, epsp_out - epsp)
    assert D_batch.shape == (3, 6, 6)
