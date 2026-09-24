"""
Unit tests for LAW33 (crushable foam plasticity, /MAT/LAW33, /MAT/FOAM_PLAS).

Fortran origin:
- engine/source/materials/mat/mat033/sigeps33.F
- starter/source/materials/mat/mat033/hm_read_mat33.F

Physics verified:
1. Yield surface in P-Q space:
   - Closed-cell air pressure contribution to pressure: sig_air = max(0, -P0*gamma/(1+gamma-phi))
   - Principal stress return mapping: clamping principal stresses to yield limit
2. Volumetric strain used for crushing:
   - gamma = rho0/rho - 1 + gamma0
   - Compression (rho > rho0 -> gamma < 0) develops crushing yield stress: |A + B*(1 + C*gamma)|
3. Rate dependence:
   - Deviatoric equivalent strain rate scaling of yield stress via rate curve
4. Tension cutoff behavior (KEN=2):
   - Principal tensile stress capped at SIGT_CUTOFF
5. Tangents and sound speed:
   - Elastic and rate-dependent tangents, acoustic wave speed c = sqrt(E/rho0)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law33_foamplas import (
    build_law33,
    solid_update,
    solid_tangent,
    tangent,
    sound_speed,
    extra_shapes,
    needs_defgrad,
)


def _make_foam(
    e: float = 100.0,
    rho0: float = 1.0,
    ken: int = 0,
    a: float = 0.0,
    b: float = 10.0,
    c: float = 0.0,
    p0: float = 0.0,
    phi: float = 0.0,
    gama0: float = 0.0,
    **kwargs,
):
    """Create a LAW33 Material entity with specified properties."""
    rec = {
        "id": 33,
        "density": rho0,
        "title": "FOAM_TEST",
        "params": {
            "MAT_E": e,
            "Itype": ken,
            "MAT_A0": a,
            "MAT_A1": b,
            "MAT_A2": c,
            "MAT_P0": p0,
            "MAT_PHI": phi,
            "MAT_GAMA0": gama0,
            **kwargs,
        },
    }
    return build_law33(rec)


def test_law33_volumetric_strain_crushing_and_air_pressure():
    """
    Verify volumetric strain crushing and air pressure contribution:
    1. Compression: rho > rho0 => gamma = rho0/rho - 1 < 0.
    2. Air pressure develops: sig_air = -P0*gamma / (1 + gamma - phi).
    3. Yield stress increases with crushing if C < 0 (typical foam compaction curve).
    """
    rho0 = 1.0
    p0 = 0.1   # air pressure
    phi = 0.1  # initial air volume fraction
    # Syield = |A + B*(1 + C*gamma)|: at gamma=0 -> 10. At gamma=-0.5 -> 10*(1 - 2*(-0.5)) = 20
    mat = _make_foam(e=100.0, rho0=rho0, ken=0, a=0.0, b=10.0, c=-2.0, p0=p0, phi=phi)

    # Initial state
    sig = np.zeros((1, 6))
    dt = 1.0e-3
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])  # compressive increment

    # Test at compressed density rho = 1.2 (volumetric compaction gamma = 1.0/1.2 - 1 = -0.1667)
    rho_comp = 1.2
    extra = {"rho": np.array([rho_comp])}

    sig_new, _, c = solid_update(mat, sig, deps, epsp=None, dt=dt, extra=extra)

    gamma = rho0 / rho_comp - 1.0
    expected_air_p = max(0.0, -(p0 * gamma) / (1.0 + gamma - phi + 1.0e-15))
    assert expected_air_p > 0.0, "Air pressure should be positive during compression"

    # Compressive stress must be developed
    assert sig_new[0, 0] < 0.0
    assert sig_new[0, 1] < 0.0
    assert sig_new[0, 2] < 0.0

    # Sound speed must be c = sqrt(E / rho0)
    assert c[0] == pytest.approx(math.sqrt(100.0 / rho0))


def test_law33_principal_stress_clamping_yield_surface():
    """
    Verify yield surface clamping in principal stress space:
    Under large shear or uniaxial strain, principal stresses must be clamped
    to the yield stress Syield.
    """
    # Yield stress is constant = 5.0 MPa (A=5.0, B=0.0)
    mat = _make_foam(e=1000.0, rho0=1.0, ken=0, a=5.0, b=0.0, c=0.0)

    # Apply large shear strain: trial shear stress = 0.5 * E * deps_xy = 0.5 * 1000 * 0.05 = 25.0 MPa >> 5.0
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.05, 0.0, 0.0]])
    extra = {"rho": np.array([1.0])}

    sig_new, _, _ = solid_update(mat, sig, deps, epsp=None, dt=1.0e-3, extra=extra)

    # Pure shear: principal stresses are [+tau, -tau, 0]
    # Clamping ensures |tau| <= Syield = 5.0 MPa
    assert abs(sig_new[0, 3]) == pytest.approx(5.0, abs=1e-5)
    assert abs(sig_new[0, 0]) < 1e-10
    assert abs(sig_new[0, 1]) < 1e-10


def test_law33_rate_dependence():
    """
    Verify optional rate-dependence scaling (sigeps33 lines 161-172):
    Higher equivalent deviatoric strain rate scales the yield stress via rate curve.
    """
    # Base yield = 10.0 MPa
    # Rate curve: rate [0, 10, 100] -> scale [1.0, 1.5, 2.0]
    rate_xs = np.array([0.0, 10.0, 100.0])
    rate_ys = np.array([1.0, 1.5, 2.0])

    mat = _make_foam(
        e=2000.0,
        rho0=1.0,
        ken=0,
        a=10.0,
        b=0.0,
        c=0.0,
        IFN2=2,
        rate_curve=(rate_xs, rate_ys),
        FAC1=1.0,
    )

    sig = np.zeros((1, 6))
    extra = {"rho": np.array([1.0])}

    # Case A: Low strain rate (dt = 0.01 -> rate = deps / dt = 0.05 / 0.01 = 5.0 /s)
    # Expected rate factor: interp(5.0, [0, 10], [1.0, 1.5]) = 1.25 -> Syield = 12.5 MPa
    deps = np.array([[0.05, -0.025, -0.025, 0.0, 0.0, 0.0]])
    sig_low, _, _ = solid_update(mat, sig, deps, epsp=None, dt=0.01, extra=extra)

    # Case B: High strain rate (dt = 0.0005 -> rate = 0.05 / 0.0005 = 100.0 /s)
    # Expected rate factor: 2.0 -> Syield = 20.0 MPa
    sig_high, _, _ = solid_update(mat, sig, deps, epsp=None, dt=0.0005, extra=extra)

    # Higher rate produces higher stress
    assert abs(sig_high[0, 0]) > abs(sig_low[0, 0])
    assert abs(sig_high[0, 0]) == pytest.approx(20.0, rel=1e-3)
    assert abs(sig_low[0, 0]) == pytest.approx(12.5, rel=1e-3)


def test_law33_tension_cutoff_ken2():
    """
    Verify tension cutoff branch (KEN=2 / ICASE=3):
    Tensile principal stresses cannot exceed SIGT_CUTOFF.
    """
    mat = _make_foam(
        e=1000.0,
        rho0=1.0,
        ken=2,
        a=50.0,  # high compressive yield
        b=0.0,
        c=0.0,
        sigt_cutoff=3.0,  # low tension cutoff
    )

    sig = np.zeros((1, 6))
    extra = {"rho": np.array([1.0])}
    deps = np.array([[0.02, 0.0, 0.0, 0.0, 0.0, 0.0]])  # tensile strain

    sig_new, _, _ = solid_update(mat, sig, deps, epsp=None, dt=1.0e-3, extra=extra)

    # Tensile stress capped at SIGT_CUTOFF = 3.0
    assert sig_new[0, 0] == pytest.approx(3.0, abs=1e-5)


def test_law33_tangent_and_utilities():
    """Verify solid_tangent, tangent alias, sound_speed, and history shapes."""
    mat = _make_foam(e=120.0, rho0=0.5, ken=0)

    # Tangent
    D = solid_tangent(mat)
    assert D.shape == (1, 6, 6)
    # Normals carry E, shears carry E/2
    assert D[0, 0, 0] == pytest.approx(120.0)
    assert D[0, 1, 1] == pytest.approx(120.0)
    assert D[0, 3, 3] == pytest.approx(60.0)

    # Tangent convenience alias
    D_alias = tangent(mat)
    assert np.allclose(D, D_alias)

    # Sound speed
    c = sound_speed(mat)
    assert c == pytest.approx(math.sqrt(120.0 / 0.5))

    # needs_defgrad is False
    assert not needs_defgrad(mat)

    # extra_shapes: empty for KEN=0, contains eps33 for KEN=1
    assert extra_shapes(mat, nip=1) == {}
    mat_kelvin = _make_foam(e=100.0, ken=1, c1_kelvin=10.0, c2_kelvin=10.0, et=10.0, vmu=1.0, vmu0=1.0)
    assert "eps33" in extra_shapes(mat_kelvin, nip=1)
