"""Tests for Milestone M539: LAW34 Boltzmann linear viscoelastic relaxation model.

Upstream Fortran origins:
- 3D Solids: engine/source/materials/mat/mat034/sigeps34.F
- 2D Shells (Plane Stress): engine/source/materials/mat/mat034/sigeps34c.F
- Starter Reader: starter/source/materials/mat/mat034/hm_read_mat34.F
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
import pyradioss.materials as pm
from pyradioss.materials import law34_boltzmann as l34
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# =============================================================================
# Fixtures & Helpers
# =============================================================================

def _make_law34(
    bulk=1e7,
    g0=3e6,
    gi=1e6,
    beta=10.0,
    p0=0.0,
    phi=0.0,
    gama0=0.0,
    rho0=1000.0,
    rhor=0.0,
    mat_id=1,
    title="BOLTZMANN_TEST",
):
    rec = {
        "id": mat_id,
        "density": rho0,
        "title": title,
        "params": {
            "MAT_BULK": bulk,
            "MAT_G0": g0,
            "MAT_GI": gi,
            "MAT_DECAY": beta,
            "MAT_P0": p0,
            "MAT_PHI": phi,
            "MAT_GAMA0": gama0,
            "Refer_Rho": rhor,
        },
    }
    return l34.build_law34(rec)


# =============================================================================
# 1. Parameter Extraction & Initialization Tests
# =============================================================================

def test_build_law34_defaults():
    mat = _make_law34()
    assert mat.id == 1
    assert mat.law == 34
    assert mat.rho0 == 1000.0
    assert mat.params["rhor"] == 1000.0
    assert mat.params["bulk"] == 1e7
    assert mat.params["g0"] == 3e6
    assert mat.params["gi"] == 1e6
    assert mat.params["beta"] == 10.0
    assert mat.params["p0"] == 0.0
    assert mat.params["phi"] == 0.0
    assert mat.params["gama0"] == 0.0

    # Derived Young's modulus & Poisson ratio
    # 9 * K * G0 / (3 * K + G0) = 9 * 1e7 * 3e6 / (3e7 + 3e6) = 2.7e14 / 3.3e7 = 8.181818e6
    expected_E = (9.0 * 1e7 * 3e6) / (3.0 * 1e7 + 3e6)
    expected_nu = (3.0 * 1e7 - 2.0 * 3e6) / (2.0 * (3.0 * 1e7 + 3e6))
    assert math.isclose(mat.params["E"], expected_E, rel_tol=1e-12)
    assert math.isclose(mat.params["nu"], expected_nu, rel_tol=1e-12)
    assert math.isclose(mat.E, expected_E, rel_tol=1e-12)
    assert math.isclose(mat.nu, expected_nu, rel_tol=1e-12)
    assert math.isclose(mat.G, 3e6, rel_tol=1e-12)
    assert math.isclose(mat.K, 1e7, rel_tol=1e-12)


def test_build_law34_aliases():
    rec = {
        "mat_id": 42,
        "rho": 500.0,
        "refer_rho": 600.0,
        "K": 2e7,
        "G_INS": 4e6,
        "G_INF": 2e6,
        "BETA": 5.0,
        "P0": 1e5,
        "PHI": 0.2,
        "gamma0": 0.05,
    }
    mat = l34.build_law34(rec)
    assert mat.id == 42
    assert mat.rho0 == 500.0
    assert mat.params["rhor"] == 600.0
    assert mat.params["bulk"] == 2e7
    assert mat.params["g0"] == 4e6
    assert mat.params["gi"] == 2e6
    assert mat.params["beta"] == 5.0
    assert mat.params["p0"] == 1e5
    assert mat.params["phi"] == 0.2
    assert mat.params["gama0"] == 0.05


def test_registry_hooks():
    for key in (34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        assert key in MAT_PHYSICS_REGISTRY
        assert MAT_PHYSICS_REGISTRY[key] is l34.build_law34


# =============================================================================
# 2. Sound Speed Tests
# =============================================================================

def test_sound_speed():
    mat = _make_law34(bulk=1e7, g0=3e6, rho0=1000.0)
    # c = sqrt((BULK + 4/3 * G0) / rho0) = sqrt((1e7 + 4e6) / 1000) = sqrt(1.4e4) = 118.32159566
    c_expected = math.sqrt((1e7 + (4.0 / 3.0) * 3e6) / 1000.0)
    assert math.isclose(l34.sound_speed(mat), c_expected, rel_tol=1e-12)

    # Array rho
    rho_arr = np.array([1000.0, 1000.0])
    c_arr = l34.sound_speed(mat, rho=rho_arr)
    assert isinstance(c_arr, np.ndarray)
    assert np.allclose(c_arr, c_expected)

    # Materials dispatcher
    assert math.isclose(pm.sound_speed(mat), c_expected, rel_tol=1e-12)


# =============================================================================
# 3. Solid Update Tests (sigeps34.F)
# =============================================================================

def test_solid_update_hydrostatic():
    """Pure hydrostatic strain increment producing volumetric DP."""
    bulk = 1e7
    g0 = 3e6
    gi = 1e6
    mat = _make_law34(bulk=bulk, g0=g0, gi=gi, p0=0.0)

    # Hydrostatic strain: tr(deps) = 3e-3, deviatoric strain = 0
    deps = np.array([[1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    extra = {}
    sign, _, c = l34.solid_update(mat, sig, deps, dt=1e-4, extra=extra)

    # DP = 3 * BULK * deps_m = 3 * 1e7 * 1e-3 = 3e4
    expected_p = bulk * 3e-3
    assert math.isclose(sign[0, 0], expected_p, rel_tol=1e-12)
    assert math.isclose(sign[0, 1], expected_p, rel_tol=1e-12)
    assert math.isclose(sign[0, 2], expected_p, rel_tol=1e-12)
    assert np.allclose(sign[0, 3:], 0.0)


def test_solid_update_air_pressure():
    """Volumetric strain coupled with air pressure (P0 > 0, PHI > 0)."""
    bulk = 1e7
    p0 = 1e5
    phi = 0.1
    gama0 = 0.0
    mat = _make_law34(bulk=bulk, p0=p0, phi=phi, gama0=gama0, rho0=1000.0)

    deps = np.array([[1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    extra = {"rho": np.array([1000.0])}
    sign, _, _ = l34.solid_update(mat, sig, deps, dt=1e-4, extra=extra)

    # GAMA = 1000/1000 - 1 + 0 = 0
    # DPDGAMA = -P0 * (1 - PHI) / (1 + GAMA - PHI) = -1e5 * 0.9 / 0.9 = -1e5
    # DP = (3 * BULK - DPDGAMA) * deps_m = (3e7 - (-1e5)) * 1e-3 = 30.1e6 * 1e-3 = 30100
    expected_p = (3.0 * bulk - (-p0 * (1.0 - phi) / (1.0 - phi))) * 1e-3
    assert math.isclose(sign[0, 0], expected_p, rel_tol=1e-12)


def test_solid_update_pure_shear_relaxation():
    """Pure shear strain increment showing instantaneous stiffness vs long-term decay."""
    g0 = 4e6
    gi = 1e6
    beta = 100.0
    mat = _make_law34(bulk=1e7, g0=g0, gi=gi, beta=beta)

    # Very small dt: response should approach instantaneous shear modulus G0
    dt_fast = 1e-8
    deps = np.array([[0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    extra = {}
    sign_fast, _, _ = l34.solid_update(mat, sig, deps, dt=dt_fast, extra=extra)

    # tau_xy ~ G0 * gamma_xy = 4e6 * 1e-3 = 4000
    assert math.isclose(sign_fast[0, 3], g0 * 1e-3, rel_tol=1e-4)

    # Now step with dt = 0.0 (dt -> 0 limit)
    sig0 = np.zeros((1, 6))
    extra0 = {}
    sign0, _, _ = l34.solid_update(mat, sig0, deps, dt=0.0, extra=extra0)
    assert math.isclose(sign0[0, 3], g0 * 1e-3, rel_tol=1e-4)


def test_solid_update_relaxation_time_series():
    """Step strain held constant relaxing toward long-term modulus GI."""
    g0 = 4e6
    gi = 1e6
    beta = 10.0  # relaxation time tau = 0.1 s
    mat = _make_law34(bulk=1e7, g0=g0, gi=gi, beta=beta)

    gamma = 1e-3
    deps = np.array([[0.0, 0.0, 0.0, gamma, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    extra = {}

    # Step 1: instantaneous loading with tiny dt
    dt = 1e-5
    sign, _, _ = l34.solid_update(mat, sig, deps, dt=dt, extra=extra)
    # Initial stress is approximately G0 * gamma = 4000
    assert math.isclose(sign[0, 3], 4000.0, rel_tol=1e-3)

    # Now hold strain constant (deps = 0) for several time constants
    dt_hold = 0.01
    deps_zero = np.zeros((1, 6))
    for _ in range(500):  # 5.0 seconds total (50 * tau)
        sign, _, _ = l34.solid_update(mat, sign, deps_zero, dt=dt_hold, extra=extra)

    # Relaxed shear stress should converge to GI * gamma = 1e6 * 1e-3 = 1000
    assert math.isclose(sign[0, 3], gi * gamma, rel_tol=1e-3)


def test_solid_update_calling_conventions():
    mat = _make_law34()
    sig = np.zeros((1, 6))
    deps = np.array([[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # 1. (mat, sig, deps, dt, extra)
    s1, _, _ = l34.solid_update(mat, sig.copy(), deps, 1e-4, {})
    # 2. (mat, sig, deps, epsp, dt, extra)
    s2, _, _ = l34.solid_update(mat, sig.copy(), deps, None, 1e-4, {})
    # 3. (mat, sig, deps, dt=..., extra=...)
    s3, _, _ = l34.solid_update(mat, sig.copy(), deps, dt=1e-4, extra={})
    # 4. dispatch via pyradioss.materials.solid_update
    s4, _, _ = pm.solid_update(mat, sig.copy(), deps, None, 1e-4, {})

    assert np.allclose(s1, s2)
    assert np.allclose(s1, s3)
    assert np.allclose(s1, s4)


# =============================================================================
# 4. Shell Update Tests (Plane Stress, sigeps34c.F)
# =============================================================================

def test_shell_update_plane_stress_exact():
    """Verify that the analytical solution satisfies sigma_zz = 0 to machine precision."""
    bulk = 1e7
    g0 = 3e6
    gi = 1e6
    beta = 10.0
    mat = _make_law34(bulk=bulk, g0=g0, gi=gi, beta=beta)

    # Biaxial tension in-plane
    deps_xx = 1.2e-3
    deps_yy = -0.5e-3
    deps_xy = 0.8e-3
    deps_sh = np.array([[deps_xx, deps_yy, deps_xy]])
    sig_sh = np.zeros((1, 3))
    extra = {}
    dt = 1e-4

    sign_sh, _ = l34.shell_update(mat, sig_sh, deps_sh, dt=dt, extra=extra)

    # Now, test with the exact 3D solid kernel using the analytical deps_zz
    # to verify sigma_zz is identically 0:
    ge = gi
    gv = g0 - gi
    ge2 = 2.0 * ge
    gv2 = 2.0 * gv
    bulk3 = 3.0 * bulk
    c1 = 1.0 - math.exp(-beta * dt)
    c2 = -c1 / beta
    cc2 = gv2 * (c1 + c2 / dt)
    aa = (1.0 / 3.0) * (ge2 - cc2 - bulk3) * (deps_xx + deps_yy)
    bb = (2.0 / 3.0) * ge2 + bulk - (2.0 / 3.0) * cc2
    deps_zz = aa / bb

    deps_3d = np.array([[deps_xx, deps_yy, deps_zz, deps_xy, 0.0, 0.0]])
    sig_3d = np.zeros((1, 6))
    extra_3d = {}
    sign_3d, _, _ = l34.solid_update(mat, sig_3d, deps_3d, dt=dt, extra=extra_3d)

    # sigma_zz must be 0 to machine precision
    assert abs(sign_3d[0, 2]) < 1e-10
    # in-plane stresses must match
    assert np.allclose(sign_sh[0, :3], sign_3d[0, [0, 1, 3]], atol=1e-10)


def test_shell_update_thickness_change():
    """Verify thickness update if thk is present in extra."""
    mat = _make_law34()
    sig_sh = np.zeros((1, 3))
    deps_sh = np.array([[1e-3, 1e-3, 0.0]])
    extra = {"thk": np.array([2.0]), "thkly": np.array([1.0]), "off": np.array([1.0])}

    l34.shell_update(mat, sig_sh, deps_sh, dt=1e-4, extra=extra)

    # Since biaxial tension in-plane produces negative thickness strain (Poisson contraction),
    # thickness should decrease
    assert extra["thk"][0] < 2.0


def test_shell_dispatcher():
    mat = _make_law34()
    sig = np.zeros((1, 3))
    deps = np.array([[1e-3, 0.0, 0.0]])
    s1, _ = l34.shell_update(mat, sig.copy(), deps, dt=1e-4)
    s2, _ = pm.shell_update(mat, sig.copy(), deps, None, 1e-4)
    assert np.allclose(s1, s2)


# =============================================================================
# 5. Tangent Tests (Algorithmic / Consistent Tangents)
# =============================================================================

def test_consistent_solid_tangent_vs_finite_differences():
    """Central finite differences verification of consistent_solid_tangent."""
    mat = _make_law34(bulk=1e7, g0=3e6, gi=1e6, beta=10.0, p0=1e5, phi=0.1)
    dt = 1e-4
    h = 1e-7

    C_ana = l34.consistent_solid_tangent(mat, dt=dt)
    C_num = np.zeros((6, 6))

    for j in range(6):
        deps_p = np.zeros(6)
        deps_p[j] += h
        extra_p = {"uv34": np.zeros((1, 6)), "eps34": np.zeros((1, 6)), "rho": np.array([1000.0])}
        s_p, _, _ = l34.solid_update(mat, np.zeros((1, 6)), deps_p.reshape(1, 6), dt=dt, extra=extra_p)

        deps_m = np.zeros(6)
        deps_m[j] -= h
        extra_m = {"uv34": np.zeros((1, 6)), "eps34": np.zeros((1, 6)), "rho": np.array([1000.0])}
        s_m, _, _ = l34.solid_update(mat, np.zeros((1, 6)), deps_m.reshape(1, 6), dt=dt, extra=extra_m)

        C_num[:, j] = (s_p[0] - s_m[0]) / (2.0 * h)

    diff = np.abs(C_ana - C_num)
    rel_err = np.max(diff / (np.abs(C_ana) + 1.0))
    assert rel_err < 1e-6, f"Solid tangent finite difference relative error {rel_err} exceeds threshold"


def test_shell_membrane_tangent_vs_finite_differences():
    """Central finite differences verification of shell_membrane_tangent."""
    mat = _make_law34(bulk=1e7, g0=3e6, gi=1e6, beta=10.0)
    dt = 1e-4
    h = 1e-7

    C_sh_ana = l34.shell_membrane_tangent(mat, dt=dt)
    C_sh_num = np.zeros((3, 3))

    for j in range(3):
        deps_p = np.zeros(3)
        deps_p[j] += h
        extra_p = {"uv34": np.zeros((1, 7)), "eps34": np.zeros((1, 3))}
        s_p, _ = l34.shell_update(mat, np.zeros((1, 3)), deps_p.reshape(1, 3), dt=dt, extra=extra_p)

        deps_m = np.zeros(3)
        deps_m[j] -= h
        extra_m = {"uv34": np.zeros((1, 7)), "eps34": np.zeros((1, 3))}
        s_m, _ = l34.shell_update(mat, np.zeros((1, 3)), deps_m.reshape(1, 3), dt=dt, extra=extra_m)

        C_sh_num[:, j] = (s_p[0] - s_m[0]) / (2.0 * h)

    diff = np.abs(C_sh_ana - C_sh_num)
    rel_err = np.max(diff / (np.abs(C_sh_ana) + 1.0))
    assert rel_err < 1e-6, f"Shell tangent finite difference relative error {rel_err} exceeds threshold"


def test_tangent_dispatchers():
    mat = _make_law34()
    sig = np.zeros((2, 6))
    sig_sh = np.zeros((2, 3))

    C_sol = pm.solid_tangent(mat, sig, None, None, extra={"dt": 1e-4})
    assert C_sol.shape == (2, 6, 6)

    C_sh_m = pm.shell_membrane_tangent(mat)
    assert C_sh_m.shape == (3, 3)

    C_sh_l = pm.shell_layer_tangent(mat, sig_sh, None, None, extra={"dt": 1e-4})
    assert C_sh_l.shape == (2, 3, 3)


def test_solid_update_1d_input():
    """Verify 1D array handling for solid_update."""
    mat = _make_law34()
    sig = np.zeros(6)
    deps = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    extra = {}
    sign, epsp, c = l34.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    assert sign.ndim == 1
    assert sign.shape == (6,)
    assert isinstance(c, (float, np.floating)) or (isinstance(c, np.ndarray) and c.ndim == 0)


def test_shell_update_1d_input():
    """Verify 1D array handling for shell_update."""
    mat = _make_law34()
    sig = np.zeros(3)
    deps = np.array([1e-3, 0.0, 0.0])
    extra = {}
    sign, epsp = l34.shell_update(mat, sig, deps, dt=1e-4, extra=extra)
    assert sign.ndim == 1
    assert sign.shape == (3,)


def test_shell_update_5_components():
    """Verify 5-component stress/strain tensor (in-plane + transverse shears)."""
    mat = _make_law34()
    sig = np.zeros((2, 5))
    deps = np.array([
        [1e-3, 0.0, 1e-3, 0.5e-3, 0.2e-3],
        [0.0, 1e-3, 0.0, 0.1e-3, 0.4e-3],
    ])
    extra = {}
    sign, _ = l34.shell_update(mat, sig, deps, dt=1e-4, extra=extra)
    # Output returns in-plane components
    assert sign.shape == (2, 3)
    # sig updated in place has all 5 components updated
    assert sig.shape == (2, 5)
    assert not np.allclose(sig[:, 3:], 0.0)


def test_empty_inputs():
    """Verify robust handling of empty (0 elements) inputs."""
    mat = _make_law34()
    sig_sol = np.zeros((0, 6))
    deps_sol = np.zeros((0, 6))
    s_out, _, c_out = l34.solid_update(mat, sig_sol, deps_sol)
    assert s_out.shape == (0, 6)
    assert c_out.shape == (0,)

    sig_sh = np.zeros((0, 3))
    deps_sh = np.zeros((0, 3))
    s_sh_out, _ = l34.shell_update(mat, sig_sh, deps_sh)
    assert s_sh_out.shape == (0, 3)


def test_starter_deck_roundtrip(tmp_path):
    """Verify full deck formatting, file write, and parsing through Starter."""
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    deck = StarterDeck("BOLTZ_0000")
    deck.title("BOLTZMANN VISCOELASTIC RELAXATION DECK")
    deck.mat_law34(
        id=7,
        rho=1200.0,
        bulk=5e7,
        g0=1.5e7,
        gi=5e6,
        beta=25.0,
        p0=1.2e5,
        phi=0.15,
        gamma0=0.01,
        title="FOAM_VISC_TEST",
    )

    p = tmp_path / "BOLTZ_0000.rad"
    deck.write(str(p))

    raw = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(raw, model, log)

    assert len(model.materials) == 1
    mat = model.materials[7]
    assert mat.law == 34
    assert mat.rho0 == 1200.0
    assert mat.params["bulk"] == 5e7
    assert mat.params["g0"] == 1.5e7
    assert mat.params["gi"] == 5e6
    assert mat.params["beta"] == 25.0
    assert mat.params["p0"] == 1.2e5
    assert mat.params["phi"] == 0.15
    assert mat.params.get("gama0", mat.params.get("gamma0")) == 0.01

