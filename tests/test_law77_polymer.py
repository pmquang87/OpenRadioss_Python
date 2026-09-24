"""Unit tests for LAW77 (Thermoplastic Viscoplastic Polymer / Foam-Air) ported from sigeps77.F."""

import math
import numpy as np
import pytest

from pyradioss.materials import law77_polymer
from pyradioss.materials.law77_polymer import (
    Law77Params,
    build_law77,
    extra_shapes,
    needs_defgrad,
    resolve,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)
from pyradioss.model.entities import Material


def test_law77_params_from_dict():
    """Verify parameter parsing and derived constants from dictionary."""
    mat_data = {
        "id": 77,
        "title": "TEST_POLYMER",
        "MAT_RHO": 0.05,
        "MAT_E": 10.0,
        "MAT_NU": 0.25,
        "E_Max": 100.0,
        "MAT_EPS": 0.8,
        "MAT_P0": 0.1013,
        "GAMMA": 1.4,
        "MAT_POROS": 0.85,
    }
    p = build_law77(mat_data)
    assert isinstance(p, Law77Params)
    assert p.id == 77
    assert p.e0 == 10.0
    assert p.nu == 0.25
    assert p.emax == 100.0
    assert p.epsmax == 0.8
    assert p.p0 == 0.1013
    assert p.gamma == 1.4
    assert p.frac == 0.85

    # Check derived moduli
    assert p.g > 0.0
    assert p.bulk > 0.0
    assert p.aa1 > p.bulk
    assert p.c_solid > 0.0
    assert p.c_shell > 0.0
    assert needs_defgrad(p) is False


def test_law77_params_from_material_entity():
    """Verify parameter parsing from pyradioss Material entity."""
    mat = Material(
        id=5,
        law=77,
        law_name="LAW77",
        params={
            "MAT_RHO": 0.08,
            "MAT_E": 12.5,
            "MAT_NU": 0.2,
            "E_Max": 75.0,
        },
    )
    p = resolve(mat)
    assert isinstance(p, Law77Params)
    assert p.id == 5
    assert p.e0 == 12.5
    assert p.nu == 0.2
    assert p.emax == 75.0
    assert p.rho0 == 0.08


def test_law77_extra_shapes():
    """Verify history variable allocation returns 23 state variables."""
    shapes = extra_shapes(None, nip=1)
    assert "uvar77" in shapes
    assert shapes["uvar77"] == (23,)
    assert shapes["uvar"] == (23,)

    shapes_multi = extra_shapes(None, nip=4)
    assert shapes_multi["uvar77"] == (4, 23)


def test_law77_solid_update_single_element():
    """Verify 3D solid continuum update on a single element under compression."""
    p = build_law77(e0=10.0, nu=0.2, emax=80.0, rho0=0.05)
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.02, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = solid_update(
        p, sig0, deps, dt=1.0e-6, extra=extra, return_sound_speed=True
    )

    assert sig_out.shape == (6,)
    # Compressive strain in x produces compressive stress in x (negative)
    assert sig_out[0] < 0.0
    assert sig_out[1] < 0.0
    assert sig_out[2] < 0.0
    assert c_out > 0.0

    # Verify history variables were initialized and updated
    assert "uvar77" in extra
    uvar = extra["uvar77"]
    assert uvar.shape == (1, 23)
    assert uvar[0, 11] == p.e0  # UVAR(12) = current Young's modulus
    assert uvar[0, 12] > 0.0  # UVAR(13) = EPST equivalent strain
    assert uvar[0, 13] == 1.0  # UVAR(14) = ILOAD (+1 loading)


def test_law77_solid_update_vectorized():
    """Verify multi-element vectorized batch solid update."""
    p = build_law77(e0=20.0, nu=0.3, emax=120.0, rho0=0.1)
    nel = 5
    sig0 = np.zeros((nel, 6), dtype=np.float64)
    deps = np.full((nel, 6), -0.01, dtype=np.float64)
    deps[:, 3:] = 0.002
    extra = {}

    sig_out, epsp_out, c_out = solid_update(
        p, sig0, deps, dt=1.0e-5, extra=extra, return_sound_speed=True
    )

    assert sig_out.shape == (nel, 6)
    assert len(c_out) == nel
    assert np.all(sig_out[:, 0] < 0.0)
    assert extra["uvar77"].shape == (nel, 23)


def test_law77_shell_update():
    """Verify 2D plane-stress shell constitutive update."""
    p = build_law77(e0=15.0, nu=0.25, rho0=0.04)
    sig0 = np.zeros(3, dtype=np.float64)
    deps = np.array([-0.01, -0.002, 0.005], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = shell_update(
        p, sig0, deps, dt=1.0e-6, extra=extra
    )

    assert sig_out.shape == (3,)
    assert sig_out[0] < 0.0
    assert c_out > 0.0
    assert extra["uvar77"].shape == (1, 23)


def test_law77_tangents():
    """Verify consistent solid and shell tangent stiffness matrices."""
    p = build_law77(e0=50.0, nu=0.3, rho0=0.08)

    # Solid tangent
    c_sol = solid_tangent(p)
    assert c_sol.shape == (1, 6, 6)
    # Check symmetry
    np.testing.assert_allclose(c_sol[0], c_sol[0].T, rtol=1e-12)
    # Diagonal components must be positive
    assert np.all(np.diag(c_sol[0]) > 0.0)

    # Shell tangent
    c_sh = shell_tangent(p)
    assert c_sh.shape == (1, 3, 3)
    np.testing.assert_allclose(c_sh[0], c_sh[0].T, rtol=1e-12)
    assert np.all(np.diag(c_sh[0]) > 0.0)

    # Elemental template tangent interface
    t_res = tangent(p)
    assert t_res is not None
    assert t_res.shape == (1, 6, 6)


def test_law77_sound_speed():
    """Verify acoustic sound speed calculation."""
    p = build_law77(e0=100.0, nu=0.2, rho0=0.05)
    c_sol = sound_speed(p, is_shell=False)
    c_sh = sound_speed(p, is_shell=True)
    assert c_sol > 0.0
    assert c_sh > 0.0
    assert isinstance(c_sol, float)


def test_law77_elemental_template_signature():
    """Verify template signature compliance: solid_update(group, x, u, ur, dt, fint, mint)."""
    class MockGroup:
        elements = [1, 2, 3]

    grp = MockGroup()
    x = np.zeros((3, 3))
    u = np.zeros((3, 3))
    ur = np.zeros((3, 3))
    dt = 1.0e-6
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    res = solid_update(grp, x, u, ur, dt, fint, mint)
    assert res is None

    t_res = tangent(grp)
    assert t_res is None or isinstance(t_res, np.ndarray)


def test_law77_dispatcher():
    """Verify dispatch through pyradioss.materials central dispatcher."""
    import pyradioss.materials as pmat

    mat = Material(
        id=77,
        law=77,
        law_name="LAW77",
        params={
            "MAT_RHO": 0.05,
            "MAT_E": 10.0,
            "MAT_NU": 0.2,
        },
    )

    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.01, -0.002, -0.002, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_out, epsp_out, c_out = pmat.solid_update(mat, sig0, deps, dt=1.0e-6)
    assert sig_out[0] < 0.0
    assert c_out > 0.0


def test_law77_plastic_yielding_and_spherical_projection():
    """Verify spherical yield projection bounds stress when exceeding yield."""
    p = build_law77(e0=20.0, nu=0.2, emax=100.0, rho0=0.05, yield_init=0.5, aa=0.0, p0=0.0, pext=0.0)
    sig0 = np.zeros(6, dtype=np.float64)
    # Large compressive strain exceeding yield
    deps = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = solid_update(p, sig0, deps, dt=1.0e-6, extra=extra)

    # In 1D uniaxial strain with zero transverse strain:
    # With yield projection, SVM should be capped at YLD = 0.5
    svm = np.sqrt(
        sig_out[0] ** 2
        + sig_out[1] ** 2
        + sig_out[2] ** 2
        + 2.0 * (sig_out[3] ** 2 + sig_out[4] ** 2 + sig_out[5] ** 2)
    )
    assert svm == pytest.approx(0.5, rel=1e-3)
    assert epsp_out > 0.0


def test_law77_tabulated_rate_dependent_yield():
    """Verify tabulated yield stress interpolates between loading curves at different strain rates."""
    # Curve 1 at rate=0.0 (quasi-static): yield stress = 1.0
    # Curve 2 at rate=100.0 (dynamic): yield stress = 2.0
    c1 = (np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.0, 1.0]))
    c2 = (np.array([0.0, 0.5, 1.0]), np.array([2.0, 2.0, 2.0]))
    load_curves = [{"rate": 0.0, "curve": c1}, {"rate": 100.0, "curve": c2}]

    p = build_law77(
        e0=50.0,
        nu=0.2,
        emax=100.0,
        rho0=0.05,
        load_curves=load_curves,
        aa=0.0,
        p0=0.0,
        pext=0.0,
    )

    deps = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Case A: quasi-static (rate ~ 0) -> yield ~ 1.0
    sig_qs, _, _ = solid_update(p, np.zeros(6), deps, dt=1.0)
    svm_qs = np.sqrt(np.sum(sig_qs[:3] ** 2) + 2.0 * np.sum(sig_qs[3:] ** 2))
    assert svm_qs == pytest.approx(1.0, rel=1e-2)

    # Case B: high rate (deps_norm = 0.1, dt = 0.001 -> rate = 100.0) -> yield ~ 2.0
    sig_dyn, _, _ = solid_update(p, np.zeros(6), deps, dt=1.0e-3)
    svm_dyn = np.sqrt(np.sum(sig_dyn[:3] ** 2) + 2.0 * np.sum(sig_dyn[3:] ** 2))
    assert svm_dyn == pytest.approx(2.0, rel=1e-2)

    # Case C: intermediate rate (deps_norm = 0.1, dt = 0.002 -> rate = 50.0) -> yield ~ 1.5
    sig_mid, _, _ = solid_update(p, np.zeros(6), deps, dt=2.0e-3)
    svm_mid = np.sqrt(np.sum(sig_mid[:3] ** 2) + 2.0 * np.sum(sig_mid[3:] ** 2))
    assert svm_mid == pytest.approx(1.5, rel=1e-2)


def test_law77_modulus_evolution():
    """Verify modulus E evolution (sigeps77.F lines 404-420) increases E towards EMAX."""
    p = build_law77(e0=10.0, nu=0.2, emax=50.0, rho0=0.05, yield_init=0.2, aa=5.0)
    sig = np.zeros(6, dtype=np.float64)
    extra = {}

    # Step 1: initial compression causing plastic flow
    deps = np.array([-0.05, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, epsp, _ = solid_update(p, sig, deps, dt=1.0e-5, extra=extra)

    uvar = extra["uvar77"]
    e_evolved = uvar[0, 11]
    # Modulus must have grown above E0 = 10.0 due to plastic strain
    assert e_evolved > 10.0
    assert e_evolved <= 50.0


def test_law77_unloading_damage_modes():
    """Verify loading followed by unloading detects delta < 0 and tests damage models."""
    for idamage in (1, 2, 3):
        p = build_law77(
            e0=20.0,
            nu=0.2,
            emax=100.0,
            rho0=0.05,
            yield_init=1.0,
            iunload=idamage,
            hys=0.5,
            expo=1.5,
            aa=0.0,
        )
        sig = np.zeros(6, dtype=np.float64)
        extra = {}

        # Loading step
        deps_load = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sig, epsp, _ = solid_update(p, sig, deps_load, dt=1.0e-5, extra=extra)
        assert extra["uvar77"][0, 13] == 1.0  # ILOAD = 1 (loading)

        # Unloading step (strain reversed)
        deps_unload = np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sig, epsp, _ = solid_update(p, sig, deps_unload, dt=1.0e-5, extra=extra)
        # Under unloading, ILOAD must flip to -1
        assert extra["uvar77"][0, 13] == -1.0


def test_law77_pore_gas_pressure_coupling():
    """Verify pore air compression generates positive P_AIR and adjusts stress."""
    p = build_law77(
        e0=20.0,
        nu=0.2,
        rho0=0.05,
        p0=0.1013,
        rhoa=1.2e-3,
        gamma=1.4,
        frac=0.8,
    )
    sig = np.zeros(6, dtype=np.float64)
    extra = {}

    # Negative volumetric strain (compression): increases air pressure
    deps = np.array([-0.05, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, epsp, c_sound = solid_update(p, sig, deps, dt=1.0e-5, extra=extra)

    uvar = extra["uvar77"]
    # Net air pressure P_AIR stored in UVAR(19)
    p_air = uvar[0, 18]
    assert p_air > 0.0
    # Acoustic wave speed accounts for pore air compressibility EF
    assert c_sound > p.c_solid


def test_law77_shell_tabulated_plasticity():
    """Verify 2D plane-stress shell update with tabulated rate-dependent curve."""
    c_load = (np.array([0.0, 0.5, 1.0]), np.array([2.5, 2.5, 2.5]))
    p = build_law77(
        e0=25.0,
        nu=0.2,
        load_curves=[{"rate": 0.0, "curve": c_load}],
        aa=0.0,
        p0=0.0,
        pext=0.0,
    )

    sig0 = np.zeros(3, dtype=np.float64)
    deps = np.array([0.2, 0.1, 0.0], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = shell_update(p, sig0, deps, dt=1.0e-5, extra=extra)

    # In plane stress: svm = sqrt(sxx^2 + syy^2 - sxx*syy + 3*sxy^2)
    svm = np.sqrt(sig_out[0] ** 2 + sig_out[1] ** 2 - sig_out[0] * sig_out[1] + 3.0 * sig_out[2] ** 2)
    assert svm == pytest.approx(2.5, rel=1e-2)
    assert epsp_out > 0.0
    assert c_out > 0.0


