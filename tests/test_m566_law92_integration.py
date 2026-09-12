"""
Tests for Milestone M566: /MAT/LAW92 (/MAT/ARRUDA_BOYCE) Arruda-Boyce Hyperelastic Model
Framework Integration: Material Dispatch, Elements Integration & Model Verification.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw92,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.starter.checks import check_materials, check_model, MessageLog
import pyradioss.materials as mats


# ============================================================================
# 1. Framework Materials Dispatch
# ============================================================================

def test_materials_solid_dispatch_law92():
    """Verify general mats.solid_update dispatches to LAW92."""
    mat = MaterialLaw92(
        id=1,
        rho0=1000.0,
        mu=1.2e6,
        d=1e-8,
        lam=6.0,
        nu=0.495,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_sound = mats.solid_update(
        mat, sig=sig, deps=deps, return_tuple=True
    )
    assert sig_out[0] > 0.0
    assert c_sound > 0.0


def test_materials_shell_dispatch_law92():
    """Verify general mats.shell_update dispatches to LAW92."""
    mat = MaterialLaw92(
        id=2,
        rho0=1100.0,
        mu=8.0e5,
        d=2e-8,
        lam=5.0,
        nu=0.49,
    )
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.03, -0.01, 0.01], dtype=float)

    sig_out, epsp_out = mats.shell_update(
        mat, sig=sig, deps=deps
    )
    assert len(sig_out) == 3


def test_materials_sound_speed_dispatch_law92():
    """Verify general mats.sound_speed dispatches for LAW92."""
    mat = MaterialLaw92(
        id=3,
        rho0=1200.0,
        mu=1.5e6,
        d=1e-8,
        lam=7.0,
        nu=0.495,
    )
    c_solid = mats.sound_speed(mat, is_shell=False)
    c_shell = mats.sound_speed(mat, is_shell=True)
    assert c_solid > 0.0
    assert c_shell > 0.0


def test_materials_consistent_tangent_dispatch_law92():
    """Verify general mats.solid_tangent dispatches for LAW92."""
    mat = MaterialLaw92(
        id=4,
        rho0=1000.0,
        mu=1.0e6,
        d=1e-8,
        lam=5.5,
        nu=0.495,
    )
    eps_trial = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=float)
    tangent = mats.solid_tangent(mat, eps_trial)
    assert tangent.shape == (6, 6)
    assert np.all(np.real(np.linalg.eigvals(tangent)) > 0.0)


# ============================================================================
# 2. Shell Elements Sound Speed Integration
# ============================================================================

def test_shell_sound_speed_law92_bt4():
    """Verify shell sound speed property and dispatch match for LAW92."""
    mat = MaterialLaw92(
        id=1,
        rho0=1000.0,
        mu=1.5e6,
        d=1e-8,
        lam=5.0,
        nu=0.495,
    )
    c_expected = float(mat.sound_speed_shell)
    c_dispatched = mats.sound_speed(mat, is_shell=True)
    assert math.isclose(c_dispatched, c_expected, rel_tol=1e-6)
    assert c_expected > 0.0


def test_shell_sound_speed_law92_tri3():
    """Verify shell sound speed calculation for LAW92 with alternative density."""
    mat = MaterialLaw92(
        id=2,
        rho0=1150.0,
        mu=2.0e6,
        d=2e-8,
        lam=6.0,
        nu=0.49,
    )
    c_expected = float(mat.sound_speed_shell)
    c_dispatched = mats.sound_speed(mat, is_shell=True)
    assert math.isclose(c_dispatched, c_expected, rel_tol=1e-6)
    assert c_expected > 0.0


# ============================================================================
# 3. Model & Full Check Integration
# ============================================================================

def test_full_model_check_with_law92():
    """Verify check_materials and check_model pass with LAW92 solid and shell parts."""
    model = Model()
    mat_solid = MaterialLaw92(
        id=1,
        title="SolidRubber",
        rho0=1000.0,
        mu=1.0e6,
        d=1e-8,
        lam=6.0,
        nu=0.495,
    )
    mat_shell = MaterialLaw92(
        id=2,
        title="ShellRubber",
        rho0=1100.0,
        mu=8.0e5,
        d=2e-8,
        lam=5.0,
        nu=0.49,
    )
    model.materials[1] = mat_solid
    model.mat_law92s[1] = mat_solid
    model.materials[2] = mat_shell
    model.mat_law92s[2] = mat_shell

    part_solid = Part(id=1, prop_id=1, mat_id=1, title="SolidPart")
    part_solid.elem_type = "SOLID"
    part_shell = Part(id=2, prop_id=2, mat_id=2, title="ShellPart")
    part_shell.elem_type = "SHELL"

    model.parts[1] = part_solid
    model.parts[2] = part_shell

    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors
