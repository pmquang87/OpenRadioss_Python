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

from pyradioss.materials.law27_brittle import shell_update
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
