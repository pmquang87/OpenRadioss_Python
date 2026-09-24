"""Tests for LAW27 brittle material model and BUG-MAT-04 verification.

Fortran reference: m27plas.F, m27law.F, el27.F, dam27.F.
Verifies:
- Cutting-plane return mapping convergence with sig_max and updated hardening modulus (BUG-MAT-04)
- Shell plane-stress update: elasto-plastic yield before cracking
- Tensile cracking damage evolution
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law27_brittle import (
    consistent_shell_tangent,
    extra_shapes,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    tangent,
)
from pyradioss.model.entities import Material


def test_law27_cutting_plane_return_convergence():
    """BUG-MAT-04: cutting-plane return mapping with sig_max and hardening update."""
    mat = Material(1, law=27, rho0=7.8e-9, params={
        "E": 210000.0, "nu": 0.3,
        "eps_t1": 0.5, "eps_m1": 0.6, "dmax1": 0.9, "eps_f1": 1.0,
        "eps_t2": 0.5, "eps_m2": 0.6, "dmax2": 0.9, "eps_f2": 1.0,
        "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 350.0  # Capped at sig_max
    })

    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }

    sig = np.zeros((1, 3))
    # Strain increment large enough to reach sig_max cap
    deps = np.array([[0.01, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_out, epsp_out = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)

    assert epsp_out[0] > 0.0
    vm = math.sqrt(sig_out[0, 0]**2 - sig_out[0, 0]*sig_out[0, 1] + sig_out[0, 1]**2 + 3.0*sig_out[0, 2]**2)
    # Under BUG-MAT-04 fix, cutting-plane converges properly and vm is capped at sig_max (350.0)
    assert vm <= 350.0 * 1.01
    assert np.all(np.isfinite(sig_out))


def test_law27_plasticity_yields_before_cracking():
    """Verify that LAW27 applies Johnson-Cook plasticity to the trial stress."""
    mat = Material(1, law=27, rho0=7.8e-9, params={
        "E": 210000.0, "nu": 0.3,
        "eps_t1": 0.1, "eps_m1": 0.2, "dmax1": 0.9, "eps_f1": 1.0,
        "eps_t2": 0.1, "eps_m2": 0.2, "dmax2": 0.9, "eps_f2": 1.0,
        "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0
    })

    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }

    sig = np.zeros((1, 3))
    deps = np.array([[0.001, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_out, epsp_out = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)

    sxx, syy, sxy = sig_out[0, 0], sig_out[0, 1], sig_out[0, 2]
    vm = math.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

    assert epsp_out[0] > 0.0
    sy = 200.0 + 300.0 * epsp_out[0]**0.5
    assert vm == pytest.approx(sy, rel=5e-3)
    assert extra["crk27"][0] == 0.0


def test_law27_cracking_and_softening():
    """Verify that LAW27 initiates cracking and damage when tensile strain exceeds eps_t1."""
    mat = Material(1, law=27, rho0=7.8e-9, params={
        "E": 210000.0, "nu": 0.3,
        "eps_t1": 0.001, "eps_m1": 0.005, "dmax1": 0.9, "eps_f1": 0.010,
        "eps_t2": 0.001, "eps_m2": 0.005, "dmax2": 0.9, "eps_f2": 0.010,
        "A": 500.0, "B": 0.0, "n": 1.0, "sig_max": 1000.0
    })

    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }

    sig = np.zeros((1, 3))
    deps = np.array([[0.003, 0.0, 0.0]])  # Exceeds eps_t1 (0.001)
    epsp = np.zeros(1)

    sig_out, epsp_out = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)

    # Crack initiated
    assert extra["crk27"][0] > 0.0
    assert extra["dmg27"][0, 0] > 0.0


def test_law27_elastic_prior_to_cracking():
    """Verify linear elastic response prior to reaching initiation strain eps_t1."""
    E = 200000.0
    nu = 0.3
    mat = Material(1, law=27, rho0=2.5e-9, params={
        "E": E, "nu": nu,
        "eps_t1": 0.01, "eps_m1": 0.03, "dmax1": 0.9, "eps_f1": 0.05,
        "eps_t2": 0.01, "eps_m2": 0.03, "dmax2": 0.9, "eps_f2": 0.05,
    })
    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }
    sig = np.zeros((1, 3))
    # Strain well below eps_t1 (0.01)
    deps = np.array([[0.002, 0.001, 0.0005]])
    s_out, _ = shell_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    cps = E / (1.0 - nu ** 2)
    G = E / (2.0 * (1.0 + nu))
    expected_sxx = cps * (0.002 + nu * 0.001)
    expected_syy = cps * (0.001 + nu * 0.002)
    expected_sxy = G * 0.0005

    assert s_out[0, 0] == pytest.approx(expected_sxx, rel=1e-8)
    assert s_out[0, 1] == pytest.approx(expected_syy, rel=1e-8)
    assert s_out[0, 2] == pytest.approx(expected_sxy, rel=1e-8)
    assert extra["crk27"][0] == 0.0
    assert extra["dmg27"][0, 0] == 0.0
    assert extra["dmg27"][0, 1] == 0.0


def test_law27_rankine_initiation_and_frozen_angle():
    """Rankine criterion: initiation when e1 > eps_t1, crack angle frozen at theta."""
    E = 200000.0
    nu = 0.3
    mat = Material(1, law=27, rho0=2.5e-9, params={
        "E": E, "nu": nu,
        "eps_t1": 0.01, "eps_m1": 0.03, "dmax1": 0.9, "eps_f1": 0.05,
    })
    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }
    sig = np.zeros((1, 3))

    # Apply strain with major principal direction at theta = pi / 4
    theta = np.pi / 4.0
    em = 0.015
    rad = 0.008
    exx = em + rad * np.cos(2.0 * theta)
    eyy = em - rad * np.cos(2.0 * theta)
    gxy = 2.0 * rad * np.sin(2.0 * theta)
    deps = np.array([[exx, eyy, gxy]])

    shell_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    assert extra["crk27"][0] == 1.0
    assert extra["ang27"][0] == pytest.approx(theta, abs=1e-5)

    # Subsequent strain at different angle does not change frozen crack angle
    frozen_angle = extra["ang27"][0]
    deps2 = np.array([[0.001, -0.003, 0.002]])
    shell_update(mat, sig, deps2, np.zeros(1), dt=1e-5, extra=extra)
    assert extra["ang27"][0] == pytest.approx(frozen_angle, abs=1e-8)


def test_law27_directional_softening_and_closure():
    """Verify progressive softening in tension and unilateral closure in compression."""
    E = 200000.0
    nu = 0.0  # zero Poisson ratio for clean 1D checks
    eps_t1 = 0.005
    eps_m1 = 0.025
    dmax1 = 0.8
    mat = Material(1, law=27, rho0=2.5e-9, params={
        "E": E, "nu": nu,
        "eps_t1": eps_t1, "eps_m1": eps_m1, "dmax1": dmax1, "eps_f1": 0.05,
    })
    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }
    sig = np.zeros((1, 3))

    # Tension step: exx = 0.015 (halfway between eps_t1 and eps_m1)
    deps_t = np.array([[0.015, 0.0, 0.0]])
    s_out, _ = shell_update(mat, sig, deps_t, np.zeros(1), dt=1e-5, extra=extra)

    expected_d1 = dmax1 * (0.015 - eps_t1) / (eps_m1 - eps_t1)  # 0.8 * 0.010 / 0.020 = 0.4
    assert extra["dmg27"][0, 0] == pytest.approx(expected_d1, rel=1e-8)
    expected_sxx = (1.0 - expected_d1) * E * 0.015
    assert s_out[0, 0] == pytest.approx(expected_sxx, rel=1e-8)

    # Compression step: reverse strain to exx = -0.005
    deps_c = np.array([[-0.020, 0.0, 0.0]])
    s_comp, _ = shell_update(mat, sig, deps_c, np.zeros(1), dt=1e-5, extra=extra)

    # Damage does not decrease (irreversibility)
    assert extra["dmg27"][0, 0] == pytest.approx(expected_d1, rel=1e-8)
    # Under compression, crack is closed: full stiffness E * (-0.005)
    expected_sxx_comp = E * (-0.005)
    assert s_comp[0, 0] == pytest.approx(expected_sxx_comp, rel=1e-8)


def test_law27_rupture_layer_failure():
    """When normal strain reaches eps_f1, the layer ruptures and layfail is set to 0."""
    mat = Material(1, law=27, rho0=2.5e-9, params={
        "E": 200000.0, "nu": 0.3,
        "eps_t1": 0.005, "eps_m1": 0.020, "dmax1": 0.9, "eps_f1": 0.040,
    })
    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }
    sig = np.zeros((1, 3))
    # Strain exceeding rupture strain eps_f1 = 0.040
    deps = np.array([[0.050, 0.0, 0.0]])
    s_out, _ = shell_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)

    assert extra["layfail"][0] == 0.0
    # Broken layer carries zero stress
    assert np.allclose(s_out[0], 0.0)


def test_law27_solid_guard_and_tangent_aliases():
    """Verify solid_update and solid_tangent raise NotImplementedError and aliases work."""
    mat = Material(1, law=27, rho0=2.5e-9, params={"E": 200000.0, "nu": 0.3})
    with pytest.raises(NotImplementedError):
        solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), np.zeros(1), 1e-5)

    with pytest.raises(NotImplementedError):
        solid_tangent(mat)

    shapes = extra_shapes(mat, nip=3)
    assert shapes["eps27"] == (3, 3)
    assert shapes["crk27"] == (3,)
    assert shapes["ang27"] == (3,)
    assert shapes["dmg27"] == (3, 2)

    extra = {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1),
    }
    C = tangent(mat, extra)
    assert C.shape == (1, 3, 3)
    C_shell = shell_tangent(mat, extra)
    assert np.allclose(C, C_shell)

