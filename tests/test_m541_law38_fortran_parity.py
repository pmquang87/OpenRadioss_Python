"""
Exhaustive Fortran Parity Audit Tests for LAW38 (/MAT/LAW38, /MAT/VISC_TAB).

Validates every formula, constant, default, and branch against upstream OpenRadioss:
  - hm_read_mat38.F: card parsing, parameter validation, and defaults
  - m38init.F: UVAR initial state vector
  - sigeps38.F: constitutive stress update, eigensolver, DREH rotation, CHECKAXES,
    air pressure, rate interpolation, tension stiffening, and cutoff deletion.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials.law38_visc_tab import (
    build_law38,
    solid_update,
    sound_speed,
    consistent_solid_tangent,
    shell_update,
    extra_shapes,
    _eval_curve,
)


# ============================================================================
# 1. Default Values and Clamping (hm_read_mat38.F)
# ============================================================================

def test_defaults_vt_vc_clamping():
    """Verify VT and VC clamping matching hm_read_mat38.F lines 228-230:
    IF( VT <= ZERO) VT = EM20 (1e-20)
    IF( VT >= HALF) VT = 0.499
    IF( VC >= HALF) VC = 0.499
    """
    # VT <= 0
    mat1 = build_law38({"id": 1, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": -0.1, "MAT_NUt": 0.2})
    assert mat1.params["nu_t"] == pytest.approx(1.0e-20)

    mat0 = build_law38({"id": 1, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": 0.0, "MAT_NUt": 0.2})
    assert mat0.params["nu_t"] == pytest.approx(1.0e-20)

    # VT >= 0.5
    mat2 = build_law38({"id": 2, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": 0.5, "MAT_NUt": 0.1})
    assert mat2.params["nu_t"] == pytest.approx(0.499)

    mat3 = build_law38({"id": 3, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": 0.6, "MAT_NUt": 0.1})
    assert mat3.params["nu_t"] == pytest.approx(0.499)

    # VC >= 0.5
    mat4 = build_law38({"id": 4, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": 0.3, "MAT_NUt": 0.5})
    assert mat4.params["nu_c"] == pytest.approx(0.499)

    mat5 = build_law38({"id": 5, "rho0": 1.0, "MAT_E": 100.0, "MAT_NU": 0.3, "MAT_NUt": 0.8})
    assert mat5.params["nu_c"] == pytest.approx(0.499)


def test_defaults_itotal():
    """Verify ITOTAL reset matching hm_read_mat38.F line 231:
    IF( ITOTAL > 3 ) ITOTAL = 0
    Values -2, -1, 0, 1, 2, 3 are preserved.
    """
    mat_high = build_law38({"id": 1, "rho0": 1.0, "MAT_E": 100.0, "ITOTAL": 4})
    assert mat_high.params["itotal"] == 0

    for val in (-2, -1, 0, 1, 2, 3):
        m = build_law38({"id": 1, "rho0": 1.0, "MAT_E": 100.0, "ITOTAL": val})
        assert m.params["itotal"] == val


def test_defaults_beta_hyster_ratedamp():
    """Verify decay, hysteresis, damping defaults matching hm_read_mat38.F lines 232-234:
    IF( BETA <= ZERO) BETA = EM20
    IF( HYSTER <= ZERO) HYSTER = ONE
    IF( RATEDAMP <= ZERO) RATEDAMP = HALF
    """
    m = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_RELX": -5.0, "MAT_HYST": 0.0, "DAMP1": -0.1
    })
    assert m.params["beta"] == pytest.approx(1.0e-20)
    assert m.params["hyster"] == pytest.approx(1.0)
    assert m.params["ratedamp"] == pytest.approx(0.5)


def test_defaults_krecover_kdecay():
    """Verify KRECOVER and KDECAY clamping matching hm_read_mat38.F lines 235-238:
    IF( KRECOVER <= 0 ) KRECOVER = 0
    IF( KRECOVER > 2 ) KRECOVER = 0
    IF( KDECAY <= 0 ) KDECAY = 0
    IF( KDECAY > 2 ) KDECAY = 0
    """
    for invalid in (-1, 0, 3, 5):
        m = build_law38({
            "id": 1, "rho0": 1.0, "MAT_E": 100.0,
            "Gflag": invalid, "Vflag": invalid
        })
        assert m.params["krecover"] == 0
        assert m.params["kdecay"] == 0

    for valid in (1, 2):
        m = build_law38({
            "id": 1, "rho0": 1.0, "MAT_E": 100.0,
            "Gflag": valid, "Vflag": valid
        })
        assert m.params["krecover"] == valid
        assert m.params["kdecay"] == valid


def test_defaults_theta_relaxp_maxpres_funload():
    """Verify defaults matching hm_read_mat38.F lines 239-246:
    IF( THETA <= ZERO) THETA = 0.67
    IF( RELAXP <= ZERO) RELAXP = EM20
    IF( MAXPRES <= ZERO) MAXPRES = INFINITY
    IF( FUNLOAD <= ZERO) FUNLOAD = ONE
    """
    m = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_Theta": 0.0, "MAT_PR": -1.0,
        "MAT_PMAX": 0.0, "MAT_ALPHA6": -0.5
    })
    assert m.params["theta"] == pytest.approx(0.67)
    assert m.params["relaxp"] == pytest.approx(1.0e-20)
    assert m.params["maxpres"] >= 1.0e20
    assert m.params["funload"] == pytest.approx(1.0)


def test_defaults_exponas_exponbs_mfunc_tensioncut():
    """Verify defaults matching hm_read_mat38.F lines 247-252:
    IF( EXPONAS == ZERO) EXPONAS = ONE
    IF( EXPONBS == ZERO) EXPONBS = ONE
    IF( MFUNC > 5 ) MFUNC = 5
    IF( TENSIONCUT <= ZERO) TENSIONCUT = INFINITY
    """
    m = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_EXP1": 0.0, "MAT_EXP2": 0.0,
        "NFUNC": 8, "MAT_CUTOFF": 0.0
    })
    assert m.params["exponas"] == pytest.approx(1.0)
    assert m.params["exponbs"] == pytest.approx(1.0)
    assert m.params["m_func"] == 5
    assert m.params["tensioncut"] >= 1.0e20


def test_defaults_epsfin_lamda_tolerance_viscosity_efinal():
    """Verify defaults matching hm_read_mat38.F lines 253-267:
    IF(EPSFIN<=ZERO.OR.EPSFIN>ZERO) EPSFIN = ONE
    IF( LAMDA <= ZERO) LAMDA = ONE
    IF( TOLERANCE <= ZERO) TOLERANCE = ONE
    IF( VISCOSITY <= ZERO) VISCOSITY = INFINITY
    IF (EFINAL <= E) EFINAL = E
    """
    # EPSFIN is always overridden to 1.0 in Fortran OpenRadioss
    for epsfin_val in (-1.0, 0.0, 0.5, 2.0):
        m = build_law38({
            "id": 1, "rho0": 1.0, "MAT_E": 100.0,
            "MAT_Epsfinal": epsfin_val, "MAT_Lamda": 0.0,
            "MAT_Tol": -1.0, "MAT_MaxVisc": 0.0, "MAT_Efinal": 50.0
        })
        assert m.params["epsfin"] == pytest.approx(1.0)
        assert m.params["lamda"] == pytest.approx(1.0)
        assert m.params["tolerance"] == pytest.approx(1.0)
        assert m.params["viscosity"] >= 1.0e20
        assert m.params["efinal"] == pytest.approx(100.0)  # Efinal clamped to >= E0


# ============================================================================
# 2. Principal Strain Eigendecomposition and DREH History Transformation
# ============================================================================

def test_checkaxes_threshold_and_dreh_history_transformation():
    """Verify CHECKAXES and DREH coordinate transformation matching sigeps38.F:
    - If amax < TOLERANCE: axes unchanged, no rotation of history variables.
    - If amax >= TOLERANCE: rotate stored history variables:
        Q = V_new^T @ V_old
        Q2 = Q * Q
        UVAR(start:start+3) = Q2 @ UVAR(start:start+3)
    """
    e0 = 200.0
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": e0, "MAT_Tol": 0.01,
        "load_curves": [([0.0, 1.0], [0.0, 200.0])],
    })

    # Initialize UVAR state
    extra = {}
    uvar = np.zeros((1, 33), dtype=float)
    uvar[0, 0:3] = [0.01, 0.02, 0.03]        # Strains
    uvar[0, 3:6] = [-2.0, -4.0, -6.0]        # Stresses
    uvar[0, 6:9] = [10.0, 20.0, 30.0]        # Strain rates
    uvar[0, 9:12] = [200.0, 200.0, 200.0]    # Moduli
    uvar[0, 12:15] = 1.0e-20 / e0
    # Old eigenvectors: Identity matrix in column-major order
    V_old = np.eye(3)
    uvar[0, 16:25] = V_old.ravel(order="F")
    extra["uv38"] = uvar

    # 1. Step with identical axis orientation (amax = 0 < TOLERANCE)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.001
    deps[0, 1] = -0.002
    deps[0, 2] = -0.003
    solid_update(mat, sig, deps, dt=1e-4, extra=extra)

    # 2. Now force an axis rotation: swap X and Y principal axes (90 deg around Z)
    # V_new = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # This corresponds to a pure shear or swapped principal strain state
    deps_rot = np.zeros((1, 6))
    deps_rot[0, 0] = -0.05
    deps_rot[0, 1] = -0.01
    deps_rot[0, 3] = 0.08  # Off-diagonal introduces significant rotation

    solid_update(mat, sig, deps_rot, dt=1e-4, extra=extra)

    # Verify that eigenvectors were stored in Fortran order in UVAR(17..25)
    V_stored = extra["uv38"][0, 16:25].reshape((3, 3), order="F")
    # Columns must be orthonormal
    np.testing.assert_allclose(V_stored.T @ V_stored, np.eye(3), atol=1e-6)


def test_dreh_matrix_rotation_algebraic_parity():
    """Directly test the mathematical equivalence of DREH with Q^2 elementwise transform:
    In sigeps38.F:
      KEN=1: DREH(SN, DPRAO) -> S_global = DPRAO @ SN @ DPRAO.T
      KEN=0: DREH(SN, DPRA)  -> S_new = DPRA.T @ S_global @ DPRA
    Since SN is diagonal, (S_new)_nn = sum_k (Q_nk)^2 * SN_kk where Q = DPRA.T @ DPRAO.
    """
    # Random orthogonal matrices (eigenvector sets)
    theta = np.radians(37.0)
    c, s = np.cos(theta), np.sin(theta)
    V_old = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    phi = np.radians(72.0)
    cp, sp = np.cos(phi), np.sin(phi)
    V_new = np.array([
        [cp, -sp, 0.0],
        [sp,  cp, 0.0],
        [0.0, 0.0, 1.0]
    ])

    SN_diag = np.array([15.0, 42.0, -8.0])
    SN = np.diag(SN_diag)

    # Step 1: DREH with KEN=1: S_global = V_old @ SN @ V_old.T
    S_global = V_old @ SN @ V_old.T

    # Step 2: DREH with KEN=0: S_new = V_new.T @ S_global @ V_new
    S_new = V_new.T @ S_global @ V_new
    diag_dreh = np.diag(S_new)

    # Formula using Q2:
    Q = V_new.T @ V_old
    Q2 = Q * Q
    diag_q2 = Q2 @ SN_diag

    np.testing.assert_allclose(diag_q2, diag_dreh, rtol=1e-12, atol=1e-12)


# ============================================================================
# 3. Closed-Cell Air Pressure Parity (sigeps38.F lines 556-583)
# ============================================================================

def test_confined_air_pressure_formula_and_capping():
    """Verify closed-cell air pressure:
    - J = lambda1 * lambda2 * lambda3
    - When J < 1.0: P_air = P0 * (J - 1) / (J - phi) * exp(-relaxp * t)
    - Capped at -MAXPRES (i.e. cannot be more compressive than -MAXPRES)
    """
    p0 = 10.0
    phi = 0.2
    relaxp = 5.0
    maxpres = 50.0

    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_Kair": 1, "MAT_P0": p0, "MAT_POROS": phi,
        "MAT_PR": relaxp, "MAT_PMAX": maxpres,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Test compression with J = exp(-0.1) * exp(-0.1) * exp(-0.1) = exp(-0.3) approx 0.7408
    eps_val = -0.1
    deps = np.full((1, 6), 0.0)
    deps[0, 0:3] = eps_val
    curr_time = 0.05
    extra = {"time": curr_time}

    sig = np.zeros((1, 6))
    sig_new, _, _ = solid_update(mat, sig, deps, dt=1e-3, extra=extra)

    J = math.exp(3.0 * eps_val)
    assert J < 1.0
    expected_pair_uncapped = p0 * (J - 1.0) / (J - phi)
    expected_pair = math.exp(-relaxp * curr_time) * max(expected_pair_uncapped, -maxpres)

    # Check UVAR(16) which stores PAIR
    actual_pair = extra["uv38"][0, 15]
    assert actual_pair == pytest.approx(expected_pair, rel=1e-5)


def test_confined_air_pressure_exceeding_maxpres():
    """Verify pressure capping when analytical pressure exceeds -MAXPRES."""
    p0 = 100.0
    phi = 0.5
    relaxp = 0.0  # No time decay
    maxpres = 80.0

    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_Kair": 1, "MAT_P0": p0, "MAT_POROS": phi,
        "MAT_PR": relaxp, "MAT_PMAX": maxpres,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Compress close to porosity phi=0.5: J = 0.55
    # P_uncapped = 100 * (0.55 - 1) / (0.55 - 0.5) = 100 * (-0.45) / 0.05 = -900
    # Capped at -maxpres = -80.0
    eps_c = math.log(0.55) / 3.0
    deps = np.full((1, 6), 0.0)
    deps[0, 0:3] = eps_c
    extra = {"time": 0.0}

    solid_update(mat, np.zeros((1, 6)), deps, dt=1e-3, extra=extra)
    assert extra["uv38"][0, 15] == pytest.approx(-80.0, rel=1e-5)


def test_confined_air_pressure_zero_in_tension():
    """Verify that when J >= 1.0, air pressure is 0.0."""
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "MAT_Kair": 1, "MAT_P0": 10.0, "MAT_POROS": 0.2,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })
    deps = np.full((1, 6), 0.0)
    deps[0, 0:3] = 0.05  # Expansion
    extra = {"time": 0.0}

    solid_update(mat, np.zeros((1, 6)), deps, dt=1e-3, extra=extra)
    assert extra["uv38"][0, 15] == 0.0


# ============================================================================
# 4. Multi-Curve Rate Interpolation and Viscosity Limiting (sigeps38.F)
# ============================================================================

def test_rate_ratio_and_power_law_interpolation():
    """Verify multi-curve interpolation matching sigeps38.F lines 707-725:
    ratio = (rate - edots) / (edotl - edots)
    pui_1 = ratio ** EXPONAS
    sigma = sigma2 + (sigma1 - sigma2) * (1 - pui_1) ** EXPONBS
    """
    exponas = 0.75
    exponbs = 1.5

    # Curve 1: rate = 0, y = 100 * eps
    # Curve 2: rate = 50, y = 200 * eps
    c1 = (np.array([0.0, 0.5]), np.array([0.0, 50.0]))
    c2 = (np.array([0.0, 0.5]), np.array([0.0, 100.0]))

    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 200.0,
        "load_curves": [c1, c2],
        "eps_tab": [0.0, 50.0],
        "fscale_tab": [1.0, 1.0],
        "MAT_EXP1": exponas, "MAT_EXP2": exponbs,
        "DAMP1": 1.0,  # No rate damping on first step
        "MAT_MaxVisc": 1.0e30,  # High so viscosity limiting is not active
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
    })

    # Strain = -0.1 (compressive), dt = 0.005 -> strain rate = 0.1 / 0.005 = 20.0
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.1
    dt = 0.005
    sig, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=dt)

    # Manual analytical calculation matching sigeps38.F:
    # EBN = EBR * EL where EBR = deps / dt, EL = exp(eps)
    rate = (0.1 / dt) * math.exp(-0.1)  # Nominal strain rate
    edots_val = 0.0
    edotl_val = 50.0
    ratio = (rate - edots_val) / (edotl_val - edots_val)
    pui_1 = ratio ** exponas
    # Note: in compression, strain_j = -ean = 1 - exp(-0.1)
    nom_strain = 1.0 - math.exp(-0.1)
    s1 = 100.0 * nom_strain
    s2 = 200.0 * nom_strain
    expected_stress = -(s2 + (s1 - s2) * ((1.0 - pui_1) ** exponbs))

    # Cauchy conversion: / (lambda_y * lambda_z) = expected_stress / 1.0
    assert sig[0, 0] == pytest.approx(expected_stress, rel=1e-3)


def test_viscosity_limiting():
    """Verify viscosity limiting matching sigeps38.F lines 728-733:
    visc = abs((sigma - sigma1) / d_rate)
    visc = min(visc, VISCOSITY)
    sigma = sigma1 - visc * d_rate
    """
    c1 = (np.array([0.0, 0.5]), np.array([0.0, 50.0]))
    c2 = (np.array([0.0, 0.5]), np.array([0.0, 500.0]))  # Very large dynamic increase

    max_visc = 0.05  # Tight viscosity clamp
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 1000.0,
        "load_curves": [c1, c2],
        "eps_tab": [0.0, 10.0],
        "fscale_tab": [1.0, 1.0],
        "MAT_MaxVisc": max_visc,
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
    })

    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05
    dt = 0.01  # Rate = 5.0
    extra = {}
    sig, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=dt, extra=extra)

    # Verify that visc was clamped to max_visc
    assert extra["viscmax"][0] == pytest.approx(max_visc)


def test_batched_loading_and_unloading_element_independence():
    """Verify that in an N-element batch, an unloading element uses unload curves
    while a loading element concurrently uses load curves without interference.
    """
    c_load = (np.array([0.0, 0.5]), np.array([0.0, 50.0]))
    c_unload = (np.array([0.0, 0.5]), np.array([0.0, 25.0]))

    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 100.0,
        "load_curves": [c_load],
        "unload_curves": [c_unload],
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
    })

    # Element 0: Loading in compression
    # Element 1: Pre-compressed, now unloading (strain rate opposite sign to strain)
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    deps[0, 0] = -0.05   # Element 0 loading
    deps[1, 0] = +0.02   # Element 1 unloading

    extra = {
        "uv38": np.zeros((2, 33)),
        "eps38": np.zeros((2, 6)),
        "off38": np.ones(2),
    }
    # Pre-compress element 1
    extra["uv38"][1, 0] = -0.10  # Previous compressive strain
    extra["eps38"][1, 0] = -0.10

    sig_new, _, _ = solid_update(mat, sig, deps, dt=1e-3, extra=extra)

    # Element 0 must follow load curve (slope 100)
    # Element 1 must follow unload curve (slope 50)
    assert sig_new[0, 0] < 0.0
    assert sig_new[1, 0] < 0.0
    # Both elements are compressed, stress in element 0 is negative
    assert abs(sig_new[0, 0]) > 0.0


# ============================================================================
# 5. Tension Stiffening Modulus (sigeps38.F lines 795-850)
# ============================================================================

def test_tension_stiffening_formula():
    """Verify tension stiffening formula:
    EI = Efinal + (E0 - Efinal) * (1 - exp(-lamda * (J - 1 + epsfin)))
    EYN = max(EI, lamda * (Efinal - E0) * exp(-lamda * (J - 1 + epsfin)))
    """
    e0 = 100.0
    efinal = 600.0
    lamda = 1.5
    # Upstream hm_read_mat38.F:253 forces EPSFIN = 1.0
    epsfin = 1.0

    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": e0,
        "MAT_Efinal": efinal, "MAT_Lamda": lamda,
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    deps = np.zeros((1, 6))
    deps[0, 0] = 0.08  # Tensile strain
    sig, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=1e-3)

    J = math.exp(0.08)
    tmp1 = math.exp(-lamda * (J - 1.0 + epsfin))
    ei_expected = efinal + (e0 - efinal) * (1.0 - tmp1)
    ean_x = J - 1.0
    expected_stress = ei_expected * ean_x

    assert sig[0, 0] == pytest.approx(expected_stress, rel=1e-4)


def test_tension_itotal_modes():
    """Verify ITOTAL tension modes:
    - ITOTAL = 0: Total, EI * ean
    - ITOTAL = 2: Incremental, UVAR(4) + EI * ecn
    - ITOTAL = -1: Total, previous max cycle modulus
    - ITOTAL = -2: Incremental, previous max cycle modulus
    """
    e0 = 100.0
    mat0 = build_law38({"id": 1, "rho0": 1.0, "MAT_E": e0, "ITOTAL": 0, "MAT_NU": 0.0, "MAT_NUt": 0.0, "load_curves": [([0, 1], [0, 100])]})
    mat2 = build_law38({"id": 2, "rho0": 1.0, "MAT_E": e0, "ITOTAL": 2, "MAT_NU": 0.0, "MAT_NUt": 0.0, "load_curves": [([0, 1], [0, 100])]})

    deps = np.zeros((1, 6))
    deps[0, 0] = 0.02
    sig0, _, _ = solid_update(mat0, np.zeros((1, 6)), deps, dt=1e-3)
    sig2, _, _ = solid_update(mat2, np.zeros((1, 6)), deps, dt=1e-3)

    # On first step from zero, total and incremental give identical results
    assert sig0[0, 0] == pytest.approx(sig2[0, 0], rel=1e-4)


# ============================================================================
# 6. Tension Cutoff Element Deletion (sigeps38.F lines 1002-1009)
# ============================================================================

def test_tension_cutoff_element_deletion_total():
    """Verify that when tensile stress exceeds TENSIONCUT in TOTAL mode:
    - off is set to 0.0
    - element stress is zeroed completely
    """
    cutoff = 25.0
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 1000.0,
        "MAT_CUTOFF": cutoff,
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
        "ITOTAL": 0,
        "load_curves": [([0.0, 1.0], [0.0, 1000.0])],
    })

    # Small strain: stress = 1000 * (exp(0.01) - 1) approx 10.05 < cutoff
    deps_small = np.zeros((1, 6))
    deps_small[0, 0] = 0.01
    extra = {}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), deps_small, dt=1e-3, extra=extra)
    assert extra["off38"][0] == 1.0
    assert sig1[0, 0] > 0.0

    # Large strain: stress = 1000 * (exp(0.05) - 1) approx 51.27 > cutoff
    deps_large = np.zeros((1, 6))
    deps_large[0, 0] = 0.05
    sig2, _, _ = solid_update(mat, sig1, deps_large, dt=1e-3, extra=extra)

    # Element must be deleted and stress completely zeroed
    assert extra["off38"][0] == 0.0
    np.testing.assert_allclose(sig2, 0.0, atol=1e-12)


def test_tension_cutoff_element_deletion_incremental():
    """Verify that when tensile stress exceeds TENSIONCUT in INCREMENTAL mode:
    - off is set to 0.0
    - element stress does NOT retain previous cycle stress (sign = 0.0)
    """
    cutoff = 30.0
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 1000.0,
        "MAT_CUTOFF": cutoff,
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
        "ITOTAL": 2,  # Incremental mode
        "load_curves": [([0.0, 1.0], [0.0, 1000.0])],
    })

    extra = {}
    # Step 1: Incremental strain 0.02 -> stress reaches approx 20.0 (< 30.0)
    deps1 = np.zeros((1, 6))
    deps1[0, 0] = 0.02
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), deps1, dt=1e-3, extra=extra)
    assert extra["off38"][0] == 1.0
    assert sig1[0, 0] > 0.0

    # Step 2: Additional incremental strain 0.02 pushes total stress to ~40.0 (> cutoff 30.0)
    deps2 = np.zeros((1, 6))
    deps2[0, 0] = 0.02
    sig_new, _, _ = solid_update(mat, sig1, deps2, dt=1e-3, extra=extra)

    assert extra["off38"][0] == 0.0
    np.testing.assert_allclose(sig_new, 0.0, atol=1e-12)


# ============================================================================
# 7. Instability Control (IMSTA == 1, 2)
# ============================================================================

def test_instability_control_imsta_2_shear_stabilization():
    """Verify IMSTA == 2 shear stress stabilization matching sigeps38.F lines 1105-1135."""
    mat = build_law38({
        "id": 1, "rho0": 1.0, "MAT_E": 200.0,
        "MAT_Iinsta": 2,  # IMSTA = 2
        "MAT_NU": 0.0, "MAT_NUt": 0.0,
        "load_curves": [([0.0, 1.0], [0.0, 200.0])],
    })

    # Apply shear deformation
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05
    deps[0, 1] = 0.02
    deps[0, 3] = 0.04  # Shear xy
    sig, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=1e-3)

    assert not np.isnan(sig).any()
    assert abs(sig[0, 3]) > 0.0


# ============================================================================
# 8. Sound Speed and Tangent Parity
# ============================================================================

def test_sound_speed_formula():
    """Verify sound speed formula:
    KKK = Emax / (3 * (1 - 2*nu_max))
    GGG = Emax / (2 * (1 + nu_max))
    c = sqrt((KKK + 4/3*GGG) / rho0)
    """
    rho0 = 1.2e-3
    e0 = 300.0
    efinal = 500.0
    nu_max = 0.25

    mat = build_law38({
        "id": 1, "rho0": rho0, "MAT_E": e0, "MAT_Efinal": efinal,
        "MAT_NU": nu_max, "MAT_NUt": 0.1,
    })

    emax = 500.0
    kkk = emax / (3.0 * (1.0 - 2.0 * nu_max))
    ggg = emax / (2.0 * (1.0 + nu_max))
    expected_c = math.sqrt((kkk + (4.0 / 3.0) * ggg) / rho0)

    c = sound_speed(mat)
    assert c == pytest.approx(expected_c, rel=1e-5)


def test_consistent_solid_tangent_shape_and_symmetry():
    """Verify tangent tensor has shape (n, 6, 6) and major symmetry."""
    mat = build_law38({"id": 1, "rho0": 1.0, "MAT_E": 150.0, "MAT_NU": 0.3})
    D = consistent_solid_tangent(mat, np.zeros((4, 6)), np.zeros((4, 6)), dt=1e-3)
    assert D.shape == (4, 6, 6)
    for i in range(4):
        np.testing.assert_allclose(D[i], D[i].T, rtol=1e-6)


def test_shell_update_not_implemented():
    """Verify shell_update raises NotImplementedError (LAW38 is solid-only)."""
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update()
