"""
Tests for Milestone M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) Bergstrom-Boyce Visco-Hyperelastic Model
Framework Integration: Material Dispatch, Elements Integration & Model Verification.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw95,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.starter.checks import check_materials, check_model, MessageLog
import pyradioss.materials as mats


# ============================================================================
# 1. Framework Materials Dispatch
# ============================================================================

def test_materials_solid_dispatch_law95():
    """Verify general mats.solid_update dispatches to LAW95."""
    mat = MaterialLaw95(
        id=1,
        rho0=1050.0,
        c10=1.2e6,
        c01=1.5e5,
        sb=0.5,
        d1=1e-8,
        a=0.01,
        expc=-0.7,
        expm=1.0,
        ksi=0.01,
        tauref=1.0e5,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=float)

    sig_out, epsp_out, c_sound = mats.solid_update(
        mat, sig=sig, deps=deps, dt=1.0e-5, return_tuple=True
    )
    assert sig_out[0] > 0.0
    assert c_sound > 0.0


def test_materials_shell_dispatch_law95_rejected():
    """Verify general mats.shell_update raises NotImplementedError for LAW95."""
    mat = MaterialLaw95(
        id=2,
        rho0=1100.0,
        c10=8.0e5,
        c01=1.0e5,
        sb=0.5,
        d1=2e-8,
    )
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.03, -0.01, 0.01], dtype=float)

    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mats.shell_update(mat, sig=sig, deps=deps)


def test_materials_sound_speed_dispatch_law95():
    """Verify general mats.sound_speed dispatches for LAW95."""
    mat = MaterialLaw95(
        id=3,
        rho0=1200.0,
        c10=1.5e6,
        c01=2.0e5,
        sb=0.8,
        d1=1e-8,
    )
    c_solid = mats.sound_speed(mat, is_shell=False)
    assert c_solid > 0.0


def test_materials_consistent_tangent_dispatch_law95():
    """Verify general mats.solid_tangent dispatches for LAW95."""
    mat = MaterialLaw95(
        id=4,
        rho0=1000.0,
        c10=1.0e6,
        c01=2.0e5,
        sb=0.5,
        d1=1e-8,
    )
    eps_trial = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=float)
    tangent = mats.solid_tangent(mat, sig=None, extra={"eps": eps_trial})
    assert tangent.shape == (6, 6)
    assert np.all(np.real(np.linalg.eigvals(tangent)) > 0.0)


def test_materials_shell_layer_tangent_law95_rejected():
    """Verify general mats.shell_layer_tangent raises NotImplementedError for LAW95."""
    mat = MaterialLaw95(id=5, rho0=1000.0, c10=1.0e6)
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mats.shell_layer_tangent(mat)


# ============================================================================
# 2. History Variable Evolution Across Steps
# ============================================================================

def test_law95_history_evolution_solid():
    """Verify extra['uvar'] (F_p and dgamma) evolves and is retained across time steps."""
    mat = MaterialLaw95(
        id=1,
        rho0=1000.0,
        c10=1.0e6,
        c01=2.0e5,
        sb=1.0,
        d1=1e-8,
        a=0.1,
        expc=-0.7,
        expm=1.0,
        ksi=0.01,
        tauref=1.0e5,
    )
    extra = {}
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=float)
    dt = 1.0e-4

    # Run 5 time steps
    for step in range(5):
        sig, epsp, c = mats.solid_update(
            mat, sig=sig, deps=deps, dt=dt, extra=extra, return_tuple=True
        )

    # Check that uvar is populated
    assert "uvar" in extra
    uvar = extra["uvar"]
    assert len(uvar) == 10
    # Diagonal components of F_p
    fp11 = uvar[0]
    fp22 = uvar[1]
    fp33 = uvar[2]
    # Under uniaxial tensile strain, F_p11 should have grown above 1.0
    assert fp11 > 1.0
    # And F_p22 / F_p33 should have contracted below 1.0
    assert fp22 < 1.0
    assert fp33 < 1.0
    # Inelastic creep strain dgamma
    assert uvar[9] > 0.0


# ============================================================================
# 3. Model & Full Check Integration
# ============================================================================

def test_full_model_check_with_law95_solid():
    """Verify check_materials and check_model pass with LAW95 solid parts."""
    model = Model()
    mat_solid = MaterialLaw95(
        id=1,
        title="SolidBB",
        rho0=1000.0,
        c10=1.0e6,
        c01=2.0e5,
        sb=0.5,
        d1=1e-8,
        a=0.01,
        expc=-0.7,
        expm=1.0,
        ksi=0.01,
        tauref=1.0e5,
    )
    model.materials[1] = mat_solid
    model.mat_law95s[1] = mat_solid

    part1 = Part(id=1, prop_id=1, mat_id=1, title="SolidPart")
    part1.elem_type = "SOLID"
    model.parts[1] = part1

    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors
    assert not log.has_warnings


def test_full_model_check_with_law95_shell_rejected():
    """Verify check_materials catches and rejects LAW95 shell parts with ANCMSG 305."""
    model = Model()
    mat_shell = MaterialLaw95(
        id=2,
        title="ShellBB",
        rho0=1100.0,
        c10=1.2e6,
        c01=1.0e5,
        sb=0.5,
        d1=1.5e-8,
    )
    model.materials[2] = mat_shell
    model.mat_law95s[2] = mat_shell

    part2 = Part(id=2, prop_id=2, mat_id=2, title="ShellPart")
    part2.elem_type = "SHELL"
    model.parts[2] = part2

    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("ANCMSG 305" in str(err) for err in log.errors)
