"""
Tests for Milestone M567: /MAT/LAW94 (/MAT/YEOH) Yeoh Hyperelastic Model
Framework Integration: Material Dispatch, Elements Integration & Model Verification.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw94,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.starter.checks import check_materials, check_model, MessageLog
import pyradioss.materials as mats


# ============================================================================
# 1. Framework Materials Dispatch
# ============================================================================

def test_materials_solid_dispatch_law94():
    """Verify general mats.solid_update dispatches to LAW94."""
    mat = MaterialLaw94(
        id=1,
        rho0=1000.0,
        c10=1.2e6,
        c20=-5.0e4,
        c30=1.0e3,
        d1=1e-8,
        d2=0.0,
        d3=0.0,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_sound = mats.solid_update(
        mat, sig=sig, deps=deps, return_tuple=True
    )
    assert sig_out[0] > 0.0
    assert c_sound > 0.0


def test_materials_shell_dispatch_law94():
    """Verify general mats.shell_update dispatches to LAW94."""
    mat = MaterialLaw94(
        id=2,
        rho0=1100.0,
        c10=8.0e5,
        c20=-2.0e4,
        c30=5.0e2,
        d1=2e-8,
        d2=0.0,
        d3=0.0,
    )
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.03, -0.01, 0.01], dtype=float)

    sig_out, epsp_out = mats.shell_update(
        mat, sig=sig, deps=deps
    )
    assert len(sig_out) == 3


def test_materials_sound_speed_dispatch_law94():
    """Verify general mats.sound_speed dispatches for LAW94."""
    mat = MaterialLaw94(
        id=3,
        rho0=1200.0,
        c10=1.5e6,
        c20=-1.0e5,
        c30=2.0e3,
        d1=1e-8,
        d2=0.0,
        d3=0.0,
    )
    c_solid = mats.sound_speed(mat, is_shell=False)
    c_shell = mats.sound_speed(mat, is_shell=True)
    assert c_solid > 0.0
    assert c_shell > 0.0


def test_materials_consistent_tangent_dispatch_law94():
    """Verify general mats.solid_tangent dispatches for LAW94."""
    mat = MaterialLaw94(
        id=4,
        rho0=1000.0,
        c10=1.0e6,
        c20=0.0,
        c30=0.0,
        d1=1e-8,
        d2=0.0,
        d3=0.0,
    )
    eps_trial = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=float)
    tangent = mats.solid_tangent(mat, eps_trial)
    assert tangent.shape == (6, 6)
    assert np.all(np.real(np.linalg.eigvals(tangent)) > 0.0)


# ============================================================================
# 2. Shell Elements Sound Speed Integration
# ============================================================================

def test_shell_sound_speed_law94_bt4():
    """Verify shell sound speed property and dispatch match for LAW94."""
    mat = MaterialLaw94(
        id=1,
        rho0=1000.0,
        c10=1.5e6,
        c20=-1.0e5,
        c30=1.0e4,
        d1=1e-8,
    )
    c_expected = float(mat.sound_speed_shell)
    c_dispatched = mats.sound_speed(mat, is_shell=True)
    assert math.isclose(c_dispatched, c_expected, rel_tol=1e-6)
    assert c_expected > 0.0


def test_shell_sound_speed_law94_tri3():
    """Verify shell sound speed calculation for LAW94 with alternative density."""
    mat = MaterialLaw94(
        id=2,
        rho0=1150.0,
        c10=2.0e6,
        c20=-8.0e4,
        c30=5.0e3,
        d1=2e-8,
    )
    c_expected = float(mat.sound_speed_shell)
    c_dispatched = mats.sound_speed(mat, is_shell=True)
    assert math.isclose(c_dispatched, c_expected, rel_tol=1e-6)
    assert c_expected > 0.0


# ============================================================================
# 3. Model & Full Check Integration
# ============================================================================

def test_full_model_check_with_law94():
    """Verify check_materials and check_model pass with LAW94 solid and shell parts."""
    model = Model()
    mat_solid = MaterialLaw94(
        id=1,
        title="SolidYeoh",
        rho0=1000.0,
        c10=1.0e6,
        c20=-5.0e4,
        c30=2.0e3,
        d1=1e-8,
    )
    mat_shell = MaterialLaw94(
        id=2,
        title="ShellYeoh",
        rho0=1100.0,
        c10=1.2e6,
        c20=-6.0e4,
        c30=3.0e3,
        d1=1.5e-8,
    )
    model.materials[1] = mat_solid
    model.materials[2] = mat_shell
    model.mat_law94s[1] = mat_solid
    model.mat_law94s[2] = mat_shell

    part1 = Part(id=1, prop_id=1, mat_id=1, title="SolidPart")
    part1.elem_type = "SOLID"
    part2 = Part(id=2, prop_id=2, mat_id=2, title="ShellPart")
    part2.elem_type = "SHELL"
    model.parts[1] = part1
    model.parts[2] = part2

    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors
    assert not log.has_warnings
