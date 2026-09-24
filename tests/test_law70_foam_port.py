"""
Tests for LAW70 tabulated visco-elastic foam model (pyradioss/materials/law70_tabfoam.py).
Fortran reference:
  - engine/source/materials/mat/mat070/sigeps70.F
  - starter/source/materials/mat/mat070/hm_read_mat70.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law70_tabfoam
from pyradioss.model.entities import Material


def _create_law70_mat(
    e0: float = 1200.0,
    nu: float = 0.0,
    rho0: float = 1.0e-9,
    emax: float = 6000.0,
    epsmax: float = 0.5,
    iflag: int = 0,
    shape: float = 1.0,
    hys: float = 0.5,
    itens: int = 0,
    fcut: float = 1e30,
    xg: np.ndarray | None = None,
    yl: np.ndarray | None = None,
    **kwargs,
) -> Material:
    """Helper to create a configured LAW70 material entity."""
    if xg is None:
        xg = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0], dtype=float)
    if yl is None:
        # Default yield curve: elastic up to 0.05 (slope 1200), plateau around 60-100
        yl = np.array([[0.0], [60.0], [70.0], [80.0], [100.0], [150.0], [300.0]], dtype=float)

    aa = (emax - e0) / epsmax

    # Compute YLD_EMAX
    stat = yl[:, 0]
    k = int(np.searchsorted(xg, epsmax, side="left"))
    k = min(max(k, 1), len(xg) - 1)
    deri = (stat[k] - stat[k - 1]) / (xg[k] - xg[k - 1])
    yld_emax = float(stat[k - 1] + deri * (epsmax - xg[k - 1]))

    params = {
        "E": e0,
        "nu": nu,
        "E0": e0,
        "EMAX": emax,
        "EPSMAX": epsmax,
        "AA": aa,
        "iflag": iflag,
        "shape": shape,
        "hys": hys,
        "itens": itens,
        "fcut": fcut,
        "ismooth": 0,
        "xg_load": xg,
        "r_load": np.array([0.0]),
        "y_load": yl,
        "YLD_EMAX": yld_emax,
    }

    if iflag in (0, 1, 2):
        params["xg_un"] = xg
        params["r_un"] = np.array([0.0])
        params["y_un"] = yl * 0.45
    else:
        params["xg_un"] = None

    if itens > 0:
        params["tens_fid"] = 1
        params["tens_scale"] = 1.0
        params["tens_x"] = np.array([0.0, 0.1, 0.3, 0.5])
        params["tens_y"] = np.array([1.0, 0.6, 0.3, 0.1])

    params.update(kwargs)
    return Material(id=70, law=70, rho0=rho0, title="LAW70_TEST", params=params)


def _create_extra(n: int = 1, e0: float = 1200.0, rho0: float = 1.0e-9) -> dict:
    """Initialize state variables dict for n integration points."""
    uv = np.zeros((n, 10))
    uv[:, 2] = e0
    return {
        "eps70": np.zeros((n, 6)),
        "uv70": uv,
        "epsd70": np.zeros(n),
        "rho": np.full(n, rho0),
    }


# ============================================================================
# Test 1: Elastic loading with static curve
# ============================================================================

def test_law70_elastic_loading_static_curve():
    """Verify loading along uniaxial axes conforms exactly to static table projection."""
    mat = _create_law70_mat(e0=1200.0, nu=0.0)
    extra = _create_extra(1, e0=1200.0)
    sig = np.zeros((1, 6))

    # Strain exx = 0.1 -> table yl[2] = 70.0
    deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_out, c_out = law70_tabfoam.solid_update(mat, sig, deps, 1e-4, extra)

    # Normal stress must match 70.0 with nu=0
    assert s_out[0, 0] == pytest.approx(70.0, rel=1e-8)
    assert abs(s_out[0, 1]) < 1e-12
    assert abs(s_out[0, 2]) < 1e-12
    assert abs(s_out[0, 3]) < 1e-12

    # Verify Frobenius stress norm matches 70.0
    s_norm = np.sqrt(s_out[0, 0]**2 + s_out[0, 1]**2 + s_out[0, 2]**2
                     + 2.0 * (s_out[0, 3]**2 + s_out[0, 4]**2 + s_out[0, 5]**2))
    assert s_norm == pytest.approx(70.0, rel=1e-8)

    # Next step: further compression exx += 0.1 -> total 0.2 -> yl[3] = 80.0
    deps2 = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_out2, _ = law70_tabfoam.solid_update(mat, s_out, deps2, 1e-4, extra)
    assert s_out2[0, 0] == pytest.approx(80.0, rel=1e-8)


# ============================================================================
# Test 2: Unloading with hysteresis (Iflag 0, 1, 2, 3, 4)
# ============================================================================

def test_law70_unloading_iflag0_bracket():
    """Iflag 0: Unloading yield stress is bounded between YLDMIN and previous peak YLD."""
    mat = _create_law70_mat(iflag=0, nu=0.0)
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    # Step 1: Load to exx = 0.2 -> yld = 80.0
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    peak_yld = extra["uv70"][0, 5]
    assert peak_yld == pytest.approx(80.0, rel=1e-8)

    # Step 2: Slight unload to exx = 0.18
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    un_yld = extra["uv70"][0, 5]
    assert un_yld <= peak_yld
    # Unload curve is 0.45 * yl
    assert un_yld >= 0.45 * 70.0


def test_law70_unloading_iflag1_deviator_damage():
    """Iflag 1: Damage scales deviator only, preserving hydrostatic pressure."""
    mat = _create_law70_mat(iflag=1, nu=0.25)
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    # Step 1: Load in 3D
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.1, 0.05, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Step 2: Unload
    sig_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, -0.02, -0.01, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Mean hydrostatic pressure P = (sxx + syy + szz)/3 is non-zero
    p_mean = (sig_un[0, 0] + sig_un[0, 1] + sig_un[0, 2]) / 3.0
    assert abs(p_mean) > 0.0
    assert extra["uv70"][0, 4] == -1.0  # unloading flag set


def test_law70_unloading_iflag2_full_tensor():
    """Iflag 2: Damage scales the entire stress tensor."""
    mat = _create_law70_mat(iflag=2, nu=0.25)
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.1, 0.05, 0.0, 0.0, 0.0]]), 1e-4, extra)
    sig_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, -0.02, -0.01, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert not np.isnan(sig_un).any()
    assert np.all(np.isfinite(sig_un))


def test_law70_unloading_iflag3_energy_deviator():
    """Iflag 3: Hysteretic dissipation scales deviator only based on accumulated energy."""
    mat = _create_law70_mat(iflag=3, shape=1.5, hys=0.6, nu=0.2)
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    # Load
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Unload
    sig_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Accumulated energy in uv70[:, 7] > 0
    assert extra["uv70"][0, 7] > 0.0
    assert not np.isnan(sig_un).any()


def test_law70_unloading_iflag4_energy_full_tensor():
    """Iflag 4: Hysteretic dissipation scales full stress tensor."""
    mat = _create_law70_mat(iflag=4, shape=2.0, hys=0.4, nu=0.2)
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    # Load and unload
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    sig_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 7] > 0.0
    assert not np.isnan(sig_un).any()


# ============================================================================
# Test 3: Hydrostatic/tensile EOS response (Itens > 0 with mu = 1 - rho/rho0)
# ============================================================================

def test_law70_tensile_eos_scaling():
    """When in net volumetric expansion (mu = 1 - rho/rho0 > 0), stresses are scaled by alpha_1(mu)."""
    mat = _create_law70_mat(itens=1, nu=0.0)
    extra = _create_extra(1)

    # Set density expansion: rho = 0.9 * rho0 -> mu = 1 - 0.9 = 0.1
    extra["rho"] = np.array([0.9 * mat.rho0])
    sig = np.zeros((1, 6))

    # Uniaxial exx = 0.1 -> baseline yield is 70.0.
    # At mu = 0.1, tens_y[1] = 0.6 -> alpha1 = 0.6. Expected sxx = 70.0 * 0.6 = 42.0!
    s, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert s[0, 0] == pytest.approx(42.0, rel=1e-5)
    assert extra["uv70"][0, 9] == pytest.approx(0.6, rel=1e-5)

    # Under compression (rho > rho0, mu <= 0), alpha1 must remain 1.0 (no reduction)
    extra_comp = _create_extra(1)
    extra_comp["rho"] = np.array([1.1 * mat.rho0])  # mu = -0.1 < 0
    s_comp, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra_comp)
    assert s_comp[0, 0] == pytest.approx(70.0, rel=1e-5)
    assert extra_comp["uv70"][0, 9] == pytest.approx(1.0, rel=1e-5)


def test_law70_tensile_eos_callable():
    """Optional callable EOS/tensile scaling function in mat.params['eos']."""
    mat = _create_law70_mat(itens=1, nu=0.0)
    # Define an exponential tensile cutoff function f(mu) = exp(-10 * mu)
    mat.params["eos"] = lambda mu: np.exp(-10.0 * mu)
    # Remove tens_x/tens_y so it uses eos
    del mat.params["tens_x"]
    del mat.params["tens_y"]

    extra = _create_extra(1)
    extra["rho"] = np.array([0.95 * mat.rho0])  # mu = 0.05
    sig = np.zeros((1, 6))

    s, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    expected_alpha = math.exp(-10.0 * 0.05)
    assert s[0, 0] == pytest.approx(70.0 * expected_alpha, rel=1e-5)


# ============================================================================
# Test 4: Orthotropic / uncoupled 3-direction deformation test
# ============================================================================

def test_law70_uncoupled_poisson_behavior():
    """With nu = 0, deformation along any axis produces zero Poisson coupling in other axes."""
    mat = _create_law70_mat(nu=0.0)

    # Test X-direction compression
    ex1 = _create_extra(1)
    s1, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, ex1)
    assert s1[0, 0] == pytest.approx(70.0, rel=1e-8)
    assert abs(s1[0, 1]) < 1e-12
    assert abs(s1[0, 2]) < 1e-12

    # Test Y-direction compression
    ex2 = _create_extra(1)
    s2, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.0, 0.1, 0.0, 0.0, 0.0, 0.0]]), 1e-4, ex2)
    assert s2[0, 1] == pytest.approx(70.0, rel=1e-8)
    assert abs(s2[0, 0]) < 1e-12
    assert abs(s2[0, 2]) < 1e-12

    # Test Z-direction compression
    ex3 = _create_extra(1)
    s3, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]), 1e-4, ex3)
    assert s3[0, 2] == pytest.approx(70.0, rel=1e-8)
    assert abs(s3[0, 0]) < 1e-12
    assert abs(s3[0, 1]) < 1e-12


def test_law70_directional_curves_orthotropic():
    """Separate directional curves for X, Y, Z evaluate distinct yield stresses along each axis."""
    # Define 3 distinct directional stress-strain curves
    # Direction X: plateau at 50 MPa
    # Direction Y: plateau at 100 MPa
    # Direction Z: plateau at 150 MPa
    curve_x = (np.array([0.0, 0.1, 0.5]), np.array([0.0, 50.0, 50.0]))
    curve_y = (np.array([0.0, 0.1, 0.5]), np.array([0.0, 100.0, 100.0]))
    curve_z = (np.array([0.0, 0.1, 0.5]), np.array([0.0, 150.0, 150.0]))

    mat = _create_law70_mat(
        nu=0.0,
        directional_curves=[curve_x, curve_y, curve_z]
    )
    extra = _create_extra(1)
    sig = np.zeros((1, 6))

    # Triaxial compression: exx=0.1, eyy=0.1, ezz=0.1
    deps = np.array([[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]])
    s, _ = law70_tabfoam.solid_update(mat, sig, deps, 1e-4, extra)

    # Each axis must reach its own distinct directional plateau!
    assert s[0, 0] == pytest.approx(50.0, rel=1e-6)
    assert s[0, 1] == pytest.approx(100.0, rel=1e-6)
    assert s[0, 2] == pytest.approx(150.0, rel=1e-6)
    assert abs(s[0, 3]) < 1e-12
    assert abs(s[0, 4]) < 1e-12
    assert abs(s[0, 5]) < 1e-12


# ============================================================================
# Test 5: Tangent matrix calculation (tangent(mat) and solid_tangent)
# ============================================================================

def test_law70_tangent_dispatcher_interface():
    """Verify tangent(mat) returns a valid 6x6 elastic matrix conforming to dispatcher."""
    e0 = 1500.0
    nu = 0.2
    mat = _create_law70_mat(e0=e0, nu=nu)

    # Base call: tangent(mat)
    D = law70_tabfoam.tangent(mat)
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(D) > 0.0)

    # Theoretical coefficients
    aa1 = e0 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    aa2 = aa1 * nu / (1.0 - nu)
    g = 0.5 * e0 / (1.0 + nu)

    assert D[0, 0] == pytest.approx(aa1, rel=1e-10)
    assert D[1, 1] == pytest.approx(aa1, rel=1e-10)
    assert D[2, 2] == pytest.approx(aa1, rel=1e-10)
    assert D[0, 1] == pytest.approx(aa2, rel=1e-10)
    assert D[3, 3] == pytest.approx(g, rel=1e-10)

    # Call with extra: returns (n, 6, 6) evolving tangent
    extra = _create_extra(2, e0=e0)
    D_batch = law70_tabfoam.tangent(mat, extra=extra)
    assert D_batch.shape == (2, 6, 6)


def test_law70_tangent_uncoupled_poisson():
    """When nu = 0, off-diagonal normal couplings aa2 are identically 0."""
    mat = _create_law70_mat(e0=2000.0, nu=0.0)
    D = law70_tabfoam.tangent(mat)

    assert D[0, 0] == pytest.approx(2000.0, rel=1e-10)
    assert D[0, 1] == 0.0
    assert D[0, 2] == 0.0
    assert D[1, 2] == 0.0
    assert D[3, 3] == pytest.approx(1000.0, rel=1e-10)  # G = 0.5 * E


# ============================================================================
# Test 6: Sound speed & shell rejection
# ============================================================================

def test_law70_sound_speed():
    """Instantaneous P-wave sound speed c = sqrt(AA1 / rho0)."""
    e0 = 1000.0
    nu = 0.25
    rho0 = 1.5e-9
    mat = _create_law70_mat(e0=e0, nu=nu, rho0=rho0)
    extra = _create_extra(1, e0=e0, rho0=rho0)

    _, c = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), 1e-4, extra)
    aa1 = e0 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    expected_c = math.sqrt(aa1 / rho0)
    assert c[0] == pytest.approx(expected_c, rel=1e-10)


def test_law70_shell_rejection():
    """Shell elements are rejected with NotImplementedError matching starter hm_read_mat70.F."""
    mat = _create_law70_mat()
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        law70_tabfoam.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 1e-4)
