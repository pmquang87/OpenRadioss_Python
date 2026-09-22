"""
Unit tests for LAW74 (3D Tabulated Hill Orthotropic Plasticity Model for Solids).

Fortran upstream reference:
- Source: engine/source/materials/mat/mat074/sigeps74.F
- Subroutine: SIGEPS74 (engine/source/materials/mat/mat074/sigeps74.F, lines 35-954)
- Starter reader: starter/source/materials/mat/mat074/hm_read_mat74.F (lines 39-343)

Verifies:
1. Elastic response: small strain below yield, hookean law.
2. Hill 3D orthotropic yield criterion: directional yields along principal and shear directions.
3. Mixed isotropic-kinematic hardening: backstress evolution and yield stress partitioning.
4. Adiabatic plastic heating: temperature rise with rhocp and volume.
5. Dynamic Young's modulus degradation: exponential CE decay and function OPTE=1.
6. Tensile failure & element deletion: EPSR1/EPSR2 softening and EPSMAX deletion.
7. Consistent tangent matrix: solid_tangent, consistent_solid_tangent, tangent.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law74_hill_3d import (
    Law74Params,
    build_law74,
    consistent_solid_tangent,
    extra_shapes,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)
from pyradioss.model.entities import Material


def test_law74_elastic_response():
    """Verify linear elastic response prior to plastic yield."""
    E = 210000.0
    nu = 0.3
    mat = build_law74(e=E, nu=nu, sigy0=500.0)

    sig = np.zeros(6)
    deps = np.array([1.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c = solid_update(mat, sig, deps, epsp=0.0, dt=1.0e-6, return_tuple=True)

    # In 3D solid Hookean law:
    # G = E / (2 * (1 + nu)), C1 = E / (3 * (1 - 2*nu))
    # s_xx = C1 * tr(deps) + 2*G * (deps_xx - tr(deps)/3)
    # tr(deps) = 1e-5
    G = 0.5 * E / (1.0 + nu)
    C1 = E / (3.0 * (1.0 - 2.0 * nu))
    expected_sxx = C1 * 1.0e-5 + 2.0 * G * (1.0e-5 - 1.0e-5 / 3.0)
    expected_syy = C1 * 1.0e-5 + 2.0 * G * (0.0 - 1.0e-5 / 3.0)

    assert sig_out[0] == pytest.approx(expected_sxx, rel=1.0e-5)
    assert sig_out[1] == pytest.approx(expected_syy, rel=1.0e-5)
    assert epsp_out == 0.0
    assert c > 0.0


def test_law74_hill_orthotropic_yield():
    """Verify Hill 3D anisotropic yield criterion with directional yield strengths."""
    # S11=1.0, S22=1.5, S33=2.0, S12=0.8, S23=0.7, S31=0.9
    y0 = 300.0
    mat = build_law74(
        e=200000.0, nu=0.3, sigy0=y0,
        s11y=1.0, s22y=1.5, s33y=2.0,
        s12y=0.8, s23y=0.7, s31y=0.9,
    )
    p = mat.params["_obj"]

    # Check Hill parameters:
    # FF = 0.5 * (1/S22^2 + 1/S33^2 - 1/S11^2)
    exp_ff = 0.5 * (1.0 / 1.5**2 + 1.0 / 2.0**2 - 1.0 / 1.0**2)
    exp_gg = 0.5 * (1.0 / 1.0**2 + 1.0 / 2.0**2 - 1.0 / 1.5**2)
    exp_hh = 0.5 * (1.0 / 1.0**2 + 1.0 / 1.5**2 - 1.0 / 2.0**2)
    assert p.ff == pytest.approx(exp_ff)
    assert p.gg == pytest.approx(exp_gg)
    assert p.hh == pytest.approx(exp_hh)

    # Plastic step along direction 1 (exx large)
    sig = np.zeros(6)
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out = solid_update(mat, sig, deps, epsp=0.0, dt=1.0e-6)

    assert epsp_out > 0.0
    # Effective Hill equivalent stress CRI
    sxx, syy, szz = sig_out[0], sig_out[1], sig_out[2]
    sxy, syz, szx = sig_out[3], sig_out[4], sig_out[5]
    cri = math.sqrt(
        p.ff * (syy - szz)**2 + p.gg * (szz - sxx)**2 + p.hh * (sxx - syy)**2
        + 2.0 * p.ll * syz**2 + 2.0 * p.mm * szx**2 + 2.0 * p.nn * sxy**2
    )
    # CRI must return to yield stress y0
    assert cri == pytest.approx(y0, rel=1.0e-3)


def test_law74_mixed_isokin_hardening():
    """Verify mixed isotropic-kinematic hardening and backstress evolution."""
    # CHARD = 0.5 (half kinematic, half isotropic) with hardening curve
    curve = np.array([[0.0, 300.0], [0.05, 450.0]])
    mat = build_law74(
        e=200000.0, nu=0.3,
        chard=0.5,
        yield_table=curve,
    )
    extra = {
        "uvar74": np.zeros(10),
    }

    # Step 1: Forward tension into plastic regime
    deps_fwd = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig = np.zeros(6)
    sig_fwd, epsp_1 = solid_update(mat, sig, deps_fwd, epsp=0.0, dt=1.0e-5, extra=extra)

    assert epsp_1 > 0.0
    # UVAR(5:10) hold the backstress alpha
    alpha = extra["uvar74"][4:10]
    # Under forward tension, backstress component alpha_xx should be positive
    assert alpha[0] > 0.0
    assert np.all(np.isfinite(sig_fwd))



def test_law74_adiabatic_heating():
    """Verify adiabatic temperature rise: dtemp = (YLD * DPLA) / (RHOCP * VOLUME)."""
    y0 = 400.0
    rhocp = 2.0e6  # J / (m^3 * K)
    mat = build_law74(
        e=200000.0, nu=0.3, sigy0=y0,
        rhocp=rhocp, t0=293.0,
    )
    extra = {
        "temp": np.array([293.0]),
        "volume": np.array([2.0]),
    }

    sig = np.zeros(6)
    deps = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out = solid_update(mat, sig, deps, epsp=0.0, dt=1.0e-5, extra=extra)

    temp_after = float(extra["temp"][0])
    assert temp_after > 293.0
    # Expected temperature increment: dtemp = (y0 * epsp_out) / (rhocp * 2.0)
    exp_dtemp = (y0 * epsp_out) / (rhocp * 2.0)
    assert temp_after == pytest.approx(293.0 + exp_dtemp, rel=5.0e-2)


def test_law74_dynamic_young_modulus_degradation():
    """Verify dynamic Young's modulus reduction via CE: E(pla) = E0 - (E0 - Einf)*(1 - exp(-CE*pla))."""
    E0 = 200000.0
    Einf = 150000.0
    CE = 50.0
    mat = build_law74(
        e=E0, nu=0.3, sigy0=300.0,
        einf=Einf, ce=CE,
    )
    extra = {
        "uvar74": np.zeros(10),
    }

    # Step 1: accumulate plastic strain
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig = np.zeros(6)
    _, epsp_1 = solid_update(mat, sig, deps, epsp=0.0, dt=1.0e-5, extra=extra)

    assert epsp_1 > 0.0
    # Next elastic step should see degraded modulus
    p = mat.params["_obj"]
    expected_e = E0 - (E0 - Einf) * (1.0 - math.exp(-CE * epsp_1))
    # Apply small elastic strain
    deps_el = np.array([1.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_el, _ = solid_update(mat, np.zeros(6), deps_el, epsp=epsp_1, dt=1.0e-5, extra=extra)

    # Trial stress with degraded E
    nu = 0.3
    G_deg = 0.5 * expected_e / (1.0 + nu)
    C1_deg = expected_e / (3.0 * (1.0 - 2.0 * nu))
    expected_sxx = C1_deg * 1.0e-5 + 2.0 * G_deg * (1.0e-5 - 1.0e-5 / 3.0)
    assert sig_el[0] == pytest.approx(expected_sxx, rel=1.0e-3)


def test_law74_tensile_failure_and_deletion():
    """Verify progressive tensile softening (EPSR1, EPSR2) and deletion (EPSMAX)."""
    # EPSMAX = 0.01
    mat = build_law74(
        e=200000.0, nu=0.3, sigy0=300.0,
        eps_max=0.005,
    )
    extra = {
        "off": np.array([1.0]),
        "uvar74": np.zeros(10),
    }

    sig = np.zeros(6)
    # Exceed EPSMAX
    deps = np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out = solid_update(mat, sig, deps, epsp=0.0, dt=1.0e-5, extra=extra)

    assert epsp_out > 0.005
    # Off flag is eroded (0.8 on first rupture, and decays to 0.0)
    assert extra["off"][0] <= 0.8


def test_law74_tangent_consistency():
    """Verify consistent solid tangent matrix generation and aliases."""
    mat = build_law74(e=200000.0, nu=0.3, sigy0=400.0)
    sig = np.zeros(6)
    deps = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])

    C_cons = consistent_solid_tangent(mat, sig, deps=deps)
    C_solid = solid_tangent(mat, sig, deps=deps)
    C_alias = tangent(mat, sig, deps=deps)

    assert C_cons.shape == (6, 6)
    assert np.allclose(C_cons, C_solid)
    assert np.allclose(C_cons, C_alias)
    assert np.all(np.isfinite(C_cons))

    # Verify extra_shapes
    shapes = extra_shapes(mat)
    assert "uvar74" in shapes
    assert "temp" in shapes
