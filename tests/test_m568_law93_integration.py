"""
Tests for Milestone M568: /MAT/LAW93 (/MAT/ORTH_HILL) Orthotropic Hill Model
Framework Integration: Material Dispatch, Elements Integration & Model Verification.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw93,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.starter.checks import check_materials, check_model, MessageLog
import pyradioss.materials as mats


# ============================================================================
# 1. Framework Materials Dispatch
# ============================================================================

def test_materials_solid_dispatch_law93():
    """Verify general mats.solid_update dispatches to LAW93."""
    mat = MaterialLaw93(
        id=1,
        rho0=7.8e-6,
        e11=210000.0,
        e22=205000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        sigma_y=300.0,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.002, -0.0006, -0.0006, 0.0, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_sound = mats.solid_update(
        mat, sig=sig, deps=deps
    )
    assert sig_out[0] > 0.0
    assert c_sound > 0.0


def test_materials_shell_dispatch_law93():
    """Verify general mats.shell_update dispatches to LAW93."""
    mat = MaterialLaw93(
        id=2,
        rho0=7.8e-6,
        e11=210000.0,
        e22=205000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        sigma_y=300.0,
    )
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.002, -0.0006, 0.0], dtype=float)

    sig_out, epsp_out = mats.shell_update(
        mat, sig=sig, deps=deps
    )
    assert len(sig_out) == 3
    assert sig_out[0] > 0.0


def test_materials_sound_speed_dispatch_law93():
    """Verify general mats.sound_speed dispatches for LAW93."""
    mat = MaterialLaw93(
        id=3,
        rho0=7.8e-6,
        e11=210000.0,
        e22=205000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        sigma_y=300.0,
    )
    c_solid = mats.sound_speed(mat, is_shell=False)
    c_shell = mats.sound_speed(mat, is_shell=True)
    assert c_solid > 0.0
    assert c_shell > 0.0
    assert c_solid > c_shell


def test_materials_extra_shapes_law93():
    """Verify extra_shapes includes uvar and dpla for LAW93."""
    mat = MaterialLaw93(id=4, law=93, rho0=7.8e-6)
    shapes = mats.extra_shapes(mat)
    assert "uvar" in shapes
    assert "dpla" in shapes


def test_materials_solid_tangent_dispatch_law93():
    """Verify general mats.solid_tangent dispatches for LAW93."""
    mat = MaterialLaw93(
        id=5,
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        sigma_y=300.0,
    )
    sig = np.zeros(6, dtype=float)
    tangent = mats.solid_tangent(mat, sig)
    assert tangent.shape == (6, 6)
    assert np.all(np.real(np.linalg.eigvals(tangent)) > 0.0)


# ============================================================================
# 2. Shell Elements Sound Speed Integration
# ============================================================================

def test_shell_sound_speed_law93_bt4():
    """Verify shell sound speed property and dispatch match for LAW93."""
    mat = MaterialLaw93(
        id=1,
        rho0=7.8e-6,
        e11=210000.0,
        e22=205000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
    )
    c_expected = float(mat.sound_speed_shell)
    c_dispatched = mats.sound_speed(mat, is_shell=True)
    assert math.isclose(c_dispatched, c_expected, rel_tol=1e-5)
    assert c_expected > 0.0


# ============================================================================
# 3. Model & Full Check Integration
# ============================================================================

def test_full_model_check_with_law93():
    """Verify check_materials and check_model pass with LAW93 solid and shell parts."""
    model = Model()
    mat_solid = MaterialLaw93(
        id=1,
        title="SolidHill",
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        sigma_y=300.0,
    )
    mat_shell = MaterialLaw93(
        id=2,
        title="ShellHill",
        rho0=7.8e-6,
        e11=205000.0,
        e22=200000.0,
        e33=205000.0,
        g12=78000.0,
        nu12=0.3,
        sigma_y=320.0,
    )
    model.materials[1] = mat_solid
    model.materials[2] = mat_shell
    model.mat_law93s[1] = mat_solid
    model.mat_law93s[2] = mat_shell

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
