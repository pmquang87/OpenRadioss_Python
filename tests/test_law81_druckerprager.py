r"""
Unit tests for LAW81 (Drucker-Prager elastoplastic model for soil/concrete).

Fortran origins:
- C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat081\\sigeps81.F
- starter\\source\\materials\\mat\\mat081\\hm_read_mat81.F90
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law81_druckerprager
from pyradioss.materials.law81_druckerprager import (
    build_law81,
    solid_update,
    consistent_solid_tangent,
    solid_tangent,
    sound_speed,
    shell_update,
    extra_shapes,
)
from pyradioss.model.entities import Material


class DummyRec:
    def __init__(self, id=1, density=2500.0, title="geomaterial", params=None):
        self.id = id
        self.density = density
        self.title = title
        self.params = params or {}


def _make_mat(**kwargs) -> Material:
    density = kwargs.pop("density", 2500.0)
    defaults = {
        "K0": 20000.0,
        "MAT_G0": 12000.0,
        "MAT_COH0": 10.0,
        "MAT_PB0": 1e6,       # very high cap so pure Drucker-Prager cone is tested
        "MAT_Beta": 30.0,     # friction angle phi = 30 deg
        "Psi": 0.0,           # dilation angle psi = 0 deg (non-associated)
        "MAT_ALPHA": 0.5,
        "MAT_EPS": 0.0,
        "MAT_SRP": 0.0,
        "Iflag": 0,
    }
    defaults.update(kwargs)
    return build_law81(DummyRec(density=density, params=defaults))


# -----------------------------------------------------------------------------
# 1. Elastic Response Verification
# -----------------------------------------------------------------------------

def test_law81_elastic_hydrostatic_compression():
    """Verify linear elastic response under pure hydrostatic compression (Hooke bulk)."""
    k0 = 25000.0
    g0 = 15000.0
    mat = _make_mat(K0=k0, MAT_G0=g0, MAT_COH0=20.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    ev = -0.003  # volumetric strain increment
    deps[0, :3] = ev / 3.0
    extra = {}
    s, epsp, c = solid_update(mat, sig, deps, extra=extra)

    p = -(s[0, 0] + s[0, 1] + s[0, 2]) / 3.0
    expected_p = -k0 * ev  # 25000 * 0.003 = 75.0
    assert p == pytest.approx(expected_p, rel=1e-7)
    assert s[0, 0] == pytest.approx(-expected_p, rel=1e-7)
    assert s[0, 1] == pytest.approx(-expected_p, rel=1e-7)
    assert s[0, 2] == pytest.approx(-expected_p, rel=1e-7)
    assert np.all(s[0, 3:] == 0.0)
    # Pure elastic: no plastic strains
    assert extra["epspd81"][0] == 0.0
    assert extra["epspv81"][0] == 0.0


def test_law81_elastic_pure_shear():
    """Verify linear elastic shear response below yield (tau = G * gamma)."""
    g0 = 12000.0
    c0 = 50.0  # high cohesion so yield is not reached
    mat = _make_mat(MAT_G0=g0, MAT_COH0=c0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    gamma = 0.001
    deps[0, 3] = gamma  # engineering shear strain
    extra = {}
    s, epsp, c = solid_update(mat, sig, deps, extra=extra)

    expected_tau = g0 * gamma  # 12.0
    q = math.sqrt(3.0) * expected_tau  # ~20.78 < 50.0
    assert s[0, 3] == pytest.approx(expected_tau, rel=1e-7)
    assert np.all(s[0, :3] == 0.0)
    assert extra["epspd81"][0] == 0.0
    assert extra["epspv81"][0] == 0.0


def test_law81_elastic_uniaxial_strain():
    """Verify 3D uniaxial strain: sig_xx = (lambda + 2G)*deps_xx, sig_yy = sig_zz = lambda*deps_xx."""
    k0 = 20000.0
    g0 = 10000.0
    lame = k0 - (2.0 / 3.0) * g0  # 13333.333
    c11 = lame + 2.0 * g0         # 33333.333
    c12 = lame                    # 13333.333
    mat = _make_mat(K0=k0, MAT_G0=g0, MAT_COH0=100.0)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    exx = 0.0005
    deps[0, 0] = exx
    extra = {}
    s, epsp, c = solid_update(mat, sig, deps, extra=extra)

    assert s[0, 0] == pytest.approx(c11 * exx, rel=1e-6)
    assert s[0, 1] == pytest.approx(c12 * exx, rel=1e-6)
    assert s[0, 2] == pytest.approx(c12 * exx, rel=1e-6)
    assert extra["epspd81"][0] == 0.0


# -----------------------------------------------------------------------------
# 2. Yield at Correct Stress State & Pressure Confinement Dependence
# -----------------------------------------------------------------------------

def test_law81_yield_confinement_dependence():
    """Verify that yield shear stress q increases with hydrostatic pressure p:
    q_yield = p * tan(phi) + c.
    """
    phi_deg = 30.0
    c0 = 15.0
    k0 = 20000.0
    g0 = 10000.0
    tgphi = math.tan(math.radians(phi_deg))
    mat = _make_mat(K0=k0, MAT_G0=g0, MAT_COH0=c0, MAT_Beta=phi_deg, Psi=0.0)

    confinements = [20.0, 50.0, 100.0]
    yield_q_measured = []

    for conf in confinements:
        sig = np.zeros((1, 6))
        # Pre-compress to confinement pressure conf
        deps_conf = np.zeros((1, 6))
        deps_conf[0, :3] = -conf / (3.0 * k0)
        extra = {}
        solid_update(mat, sig, deps_conf, extra=extra)

        # Apply large shear strain well into plastic flow
        deps_shear = np.zeros((1, 6))
        deps_shear[0, 3] = 0.02
        solid_update(mat, sig, deps_shear, extra=extra)

        p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
        q = math.sqrt(3.0) * abs(sig[0, 3])
        expected_q = conf * tgphi + c0

        assert p == pytest.approx(conf, rel=1e-5)
        assert q == pytest.approx(expected_q, rel=1e-4)
        assert extra["epspd81"][0] > 0.0
        yield_q_measured.append(q)

    # Check linear slope dq / dp = tan(phi)
    slope1 = (yield_q_measured[1] - yield_q_measured[0]) / (confinements[1] - confinements[0])
    slope2 = (yield_q_measured[2] - yield_q_measured[1]) / (confinements[2] - confinements[1])
    assert slope1 == pytest.approx(tgphi, rel=1e-4)
    assert slope2 == pytest.approx(tgphi, rel=1e-4)


def test_law81_yield_with_cos_phi():
    """Verify yield criterion when using Mohr-Coulomb cos(phi) cohesion:
    F = q - p*tan(phi) - c*cos(phi) <= 0.
    """
    phi_deg = 30.0
    c_mc = 20.0
    k0 = 20000.0
    g0 = 10000.0
    tgphi = math.tan(math.radians(phi_deg))
    cosphi = math.cos(math.radians(phi_deg))
    c_eff = c_mc * cosphi  # 20 * sqrt(3)/2 ~ 17.3205

    mat = _make_mat(K0=k0, MAT_G0=g0, MAT_COH0=c_mc, MAT_Beta=phi_deg, Psi=0.0, use_cos_phi=True)

    conf = 40.0
    sig = np.zeros((1, 6))
    deps_conf = np.zeros((1, 6))
    deps_conf[0, :3] = -conf / (3.0 * k0)
    extra = {}
    solid_update(mat, sig, deps_conf, extra=extra)

    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 0.015
    solid_update(mat, sig, deps_shear, extra=extra)

    p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    q = math.sqrt(3.0) * abs(sig[0, 3])
    expected_q = conf * tgphi + c_eff

    assert p == pytest.approx(conf, rel=1e-5)
    assert q == pytest.approx(expected_q, rel=1e-4)
    assert extra["epspd81"][0] > 0.0


def test_law81_tri_traction_apex_return():
    """Verify apex return mapping (hydrostatic tension cutoff p = -c / tan(phi))."""
    phi_deg = 30.0
    c0 = 12.0
    k0 = 20000.0
    tgphi = math.tan(math.radians(phi_deg))
    apex_p = -c0 / tgphi  # -12 / tan(30) = -20.7846

    mat = _make_mat(K0=k0, MAT_COH0=c0, MAT_Beta=phi_deg)

    # Large tensile volumetric strain: trial p = -100.0 << apex_p
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, :3] = (100.0 / (3.0 * k0))  # positive tension strain
    extra = {}
    solid_update(mat, sig, deps, extra=extra)

    p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert p == pytest.approx(apex_p, rel=1e-6)
    # Volumetric plastic strain absorbed excess tension
    assert extra["epspv81"][0] < 0.0


def test_law81_tri_compression_cap_return():
    """Verify tri-compression return mapping onto cap pressure Pb."""
    k0 = 20000.0
    pb0 = 150.0
    mat = _make_mat(K0=k0, MAT_PB0=pb0)

    # Compress beyond cap: trial p = 400.0 > Pb0 = 150.0
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, :3] = -400.0 / (3.0 * k0)
    extra = {}
    solid_update(mat, sig, deps, extra=extra)

    p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert p == pytest.approx(pb0, rel=1e-6)
    # Plastic volumetric compaction
    assert extra["epspv81"][0] == pytest.approx((400.0 - pb0) / k0, rel=1e-6)


# -----------------------------------------------------------------------------
# 3. Plastic Flow: Associated vs Non-Associated Rule & Dilatancy
# -----------------------------------------------------------------------------

def test_law81_non_associated_flow_zero_dilatancy():
    """With psi = 0 (non-associated isochoric flow), shear plastic straining produces NO volumetric plastic strain."""
    mat = _make_mat(MAT_Beta=35.0, Psi=0.0, MAT_COH0=10.0)

    sig = np.zeros((1, 6))
    # Pre-compress to p = 30.0
    deps_conf = np.zeros((1, 6))
    deps_conf[0, :3] = -30.0 / (3.0 * 20000.0)
    extra = {}
    solid_update(mat, sig, deps_conf, extra=extra)

    # Apply shear well past yield
    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 0.02
    solid_update(mat, sig, deps_shear, extra=extra)

    # Deviatoric plastic strain accumulated
    assert extra["epspd81"][0] > 0.0
    # Volumetric plastic strain remains zero
    assert abs(extra["epspv81"][0]) < 1e-12


def test_law81_associated_flow_dilatancy():
    """With associated flow (psi = phi), plastic shear deformation induces volumetric dilation (epspv < 0)."""
    phi_deg = 25.0
    mat = _make_mat(MAT_Beta=phi_deg, associated=True, MAT_COH0=10.0)
    assert mat.params["TGPSI"] == pytest.approx(math.tan(math.radians(phi_deg)))

    sig = np.zeros((1, 6))
    deps_conf = np.zeros((1, 6))
    deps_conf[0, :3] = -50.0 / (3.0 * 20000.0)
    extra = {}
    solid_update(mat, sig, deps_conf, extra=extra)

    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 0.01
    solid_update(mat, sig, deps_shear, extra=extra)

    assert extra["epspd81"][0] > 0.0
    # Dilation: volumetric plastic strain moves in the expansion direction
    assert extra["epspv81"][0] != 0.0


def test_law81_dilatancy_increases_with_psi():
    """Verify that volumetric plastic deformation increases with dilation angle psi."""
    psi_angles = [5.0, 15.0, 25.0]
    epspv_results = []

    for psi in psi_angles:
        mat = _make_mat(MAT_Beta=30.0, Psi=psi, MAT_COH0=15.0)
        sig = np.zeros((1, 6))
        deps_conf = np.zeros((1, 6))
        deps_conf[0, :3] = -40.0 / (3.0 * 20000.0)
        extra = {}
        solid_update(mat, sig, deps_conf, extra=extra)

        deps_shear = np.zeros((1, 6))
        deps_shear[0, 3] = 0.01
        solid_update(mat, sig, deps_shear, extra=extra)

        epspv_results.append(abs(extra["epspv81"][0]))

    # Larger psi should cause larger plastic volumetric change
    assert epspv_results[0] < epspv_results[1] < epspv_results[2]


# -----------------------------------------------------------------------------
# 4. Isotropic Hardening via Hardening Modulus H
# -----------------------------------------------------------------------------

def test_law81_isotropic_hardening_modulus():
    """Verify isotropic work hardening: c(epspd) = c0 + H * epspd.
    After plastic flow, the yield surface expands and subsequent loading is elastic
    until the new yield stress is reached.
    """
    c0 = 10.0
    h_mod = 50000.0  # hardening modulus H
    k0 = 20000.0
    g0 = 10000.0
    mat = _make_mat(K0=k0, MAT_G0=g0, MAT_COH0=c0, MAT_Beta=30.0, Psi=0.0, H=h_mod)
    tgphi = math.tan(math.radians(30.0))

    conf = 50.0
    sig = np.zeros((1, 6))
    deps_conf = np.zeros((1, 6))
    deps_conf[0, :3] = -conf / (3.0 * k0)
    extra = {}
    solid_update(mat, sig, deps_conf, extra=extra)

    # Initial yield stress at p=50 is q0 = 50*tan(30) + 10 ~ 38.8675
    q0 = conf * tgphi + c0

    # Step 1: Shear causing plastic straining
    deps_shear1 = np.zeros((1, 6))
    deps_shear1[0, 3] = 0.008
    solid_update(mat, sig, deps_shear1, extra=extra)

    epspd_1 = extra["epspd81"][0]
    assert epspd_1 > 0.0

    # Stress at end of step 1 should match hardened yield stress
    q1 = math.sqrt(3.0) * abs(sig[0, 3])
    expected_q1 = conf * tgphi + (c0 + h_mod * epspd_1)
    assert q1 == pytest.approx(expected_q1, rel=1e-4)

    # Step 2: Small elastic unloading then reloading
    deps_unload = np.zeros((1, 6))
    deps_unload[0, 3] = -0.0005
    solid_update(mat, sig, deps_unload, extra=extra)

    # Plastic strain unchanged during unloading
    assert extra["epspd81"][0] == pytest.approx(epspd_1)

    # Reload below hardened yield
    deps_reload = np.zeros((1, 6))
    deps_reload[0, 3] = 0.0003
    solid_update(mat, sig, deps_reload, extra=extra)
    # Still elastic
    assert extra["epspd81"][0] == pytest.approx(epspd_1)


# -----------------------------------------------------------------------------
# 5. Direct Dict and Material Construction Interfaces
# -----------------------------------------------------------------------------

def test_law81_direct_dict_interface():
    """Verify solid_update, consistent_solid_tangent, and sound_speed work with a plain dictionary."""
    mat_dict = {
        "E": 25000.0,
        "nu": 0.25,
        "phi": 30.0,
        "psi": 0.0,
        "c": 15.0,
        "H": 2000.0,
        "density": 2400.0,
    }
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = 0.0001
    extra = {}

    s_out, ep_out, ssp = solid_update(mat_dict, sig, deps, extra=extra)
    assert s_out.shape == (1, 6)
    assert ssp[0] > 0.0

    tang = solid_tangent(mat_dict, s_out, ep_out, np.zeros(1), extra=extra)
    assert tang.shape == (1, 6, 6)

    c_spd = sound_speed(mat_dict)
    assert c_spd > 0.0


# -----------------------------------------------------------------------------
# 6. Consistent Tangents and Sound Speed
# -----------------------------------------------------------------------------

def test_law81_tangents_and_sound_speed():
    """Verify consistent solid tangent properties and acoustic sound speed."""
    k0 = 20000.0
    g0 = 12000.0
    rho0 = 2200.0
    mat = _make_mat(K0=k0, MAT_G0=g0, density=rho0)

    # Elastic tangent
    sig = np.zeros((1, 6))
    C_el = consistent_solid_tangent(mat, sig, np.zeros(1), np.zeros(1))
    assert C_el.shape == (1, 6, 6)
    # Symmetric
    assert np.allclose(C_el[0], C_el[0].T)

    # Plastic tangent shows softened shear
    C_pl = consistent_solid_tangent(mat, sig, np.array([0.01]), np.array([0.005]))
    assert C_pl[0, 3, 3] < C_el[0, 3, 3]

    # Sound speed
    expected_c = math.sqrt((k0 + (4.0 / 3.0) * g0) / rho0)
    assert sound_speed(mat) == pytest.approx(expected_c, rel=1e-7)


def test_law81_shell_rejection_error():
    """Verify that shell_update raises NotImplementedError."""
    mat = _make_mat()
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))


def test_law81_extra_shapes_helper():
    """Verify extra_shapes helper returns epspd81 and epspv81."""
    shapes = extra_shapes()
    assert "epspd81" in shapes
    assert "epspv81" in shapes
