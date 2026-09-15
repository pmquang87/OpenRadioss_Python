"""Tests for /MAT/LAW103 Continuum Integration, Element Compatibility, and Materials Dispatch.

Verifies:
  - Vectorized solid update across batches of 3D continuum elements
  - Materials dispatch: solid_update, sound_speed, solid_tangent
  - Rejection of shell elements in materials dispatch (ANCMSG 305)
  - Rejection of shell tangent in materials dispatch
  - Hydrostatic compression elasticity vs deviatoric plastic flow
  - Thermomechanical coupling and adiabatic heating across multiple steps
  - Viscoplastic strain rate sensitivity
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import MatLaw103
from pyradioss.materials.law103_hensel_spittel import (
    HenselSpittelParams,
    build_law103,
    solid_update,
    solid_update_array,
    sound_speed,
    solid_tangent,
)
from pyradioss.materials import (
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    solid_tangent as mat_solid_tangent,
    sound_speed as mat_sound_speed,
    consistent_shell_tangent as mat_shell_tangent,
)


def _sample_material() -> MatLaw103:
    return MatLaw103(
        id=103,
        title="Ti-6Al-4V Hot Forging",
        rho=4.43e-9,
        e=110000.0,
        nu=0.34,
        a0=800.0,
        m1=-0.002,
        m2=0.15,
        m3=0.05,
        m4=-0.0005,
        m5=-0.0001,
        m7=0.02,
        fsmooth=0,
        fcut=0.0,
        eps_0=0.001,
        pmin=-1.0e30,
        rhocp=2.4e-3,
        t0=973.15,
        eta=0.9,
    )


def test_vectorized_solid_update_batch():
    """Verify solid_update_array processes batches of 3D solid elements simultaneously."""
    mat = _sample_material()
    params = build_law103(mat)
    n_elems = 16

    # Random small strain increments
    np.random.seed(42)
    deps = np.random.uniform(-1e-4, 1e-4, size=(n_elems, 6))
    sig_old = np.zeros((n_elems, 6))
    hist_old = np.zeros((n_elems, 2))
    hist_old[:, 1] = 973.15  # T0

    sig_new, hist_new, ssp = solid_update_array(params, deps, sig_old, hist_old, dt=1.0e-5)

    assert sig_new.shape == (n_elems, 6)
    assert hist_new.shape == (n_elems, 2)
    assert ssp.shape == (n_elems,)
    assert np.all(ssp > 0.0)
    assert np.all(hist_new[:, 0] >= 0.0)
    assert np.all(hist_new[:, 1] >= 973.15)


def test_materials_dispatch_solid_update():
    """Verify central materials.solid_update dispatches to LAW103."""
    mat = _sample_material()
    n_elems = 4
    sig = np.zeros((n_elems, 6))
    deps = np.full((n_elems, 6), 1.0e-4)

    sig_out, epsp_out, c_out = mat_solid_update(mat, sig, deps, dt=1.0e-5)

    assert sig_out.shape == (n_elems, 6)
    assert epsp_out.shape == (n_elems,)
    assert c_out is not None
    assert np.all(c_out > 0.0)


def test_materials_dispatch_sound_speed_and_tangent():
    """Verify central materials.sound_speed and solid_tangent dispatches to LAW103."""
    mat = _sample_material()

    # Sound speed dispatch
    c_val = mat_sound_speed(mat)
    expected_c = math.sqrt((mat.K + 4.0 * mat.G / 3.0) / mat.rho)
    assert pytest.approx(c_val) == expected_c

    # Solid tangent dispatch
    sig = np.zeros((2, 6))
    c_tangent = mat_solid_tangent(mat, sig)
    assert c_tangent.shape == (6, 6)
    # Check elastic symmetry
    assert np.allclose(c_tangent, c_tangent.T)


def test_shell_rejection_in_materials_dispatch():
    """Verify LAW103 raises NotImplementedError for shell formulations (ANCMSG 305)."""
    mat = _sample_material()
    sig_shell = np.zeros(3)
    deps_shell = np.zeros(3)

    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mat_shell_update(mat, sig_shell, deps_shell)

    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mat_shell_tangent(mat, sig_shell)


def test_hydrostatic_compression_vs_shear_flow():
    """Verify pure hydrostatic compression remains elastic while large shear yields."""
    mat = _sample_material()
    params = build_law103(mat)

    # 1. Pure hydrostatic compression: deps_xx = deps_yy = deps_zz = -0.002
    deps_comp = np.array([-0.002, -0.002, -0.002, 0.0, 0.0, 0.0])
    sig_old = np.zeros(6)
    hist_old = np.array([0.0, 973.15, 0.0, 0.0])

    sig_comp, hist_comp, _ = solid_update_array(
        params, deps_comp[np.newaxis, :], sig_old[np.newaxis, :], hist_old[np.newaxis, :], dt=1.0e-5
    )
    # Deviatoric stress and plastic strain remain 0
    p_comp = -(sig_comp[0, 0] + sig_comp[0, 1] + sig_comp[0, 2]) / 3.0
    assert p_comp > 0.0  # Positive pressure
    assert hist_comp[0, 0] == 0.0  # Zero plastic strain
    assert hist_comp[0, 1] == 973.15  # No heating without plastic dissipation

    # 2. Large shear strain causing plastic flow
    deps_shear = np.array([0.0, 0.0, 0.0, 0.04, 0.0, 0.0])
    sig_shear, hist_shear, _ = solid_update_array(
        params, deps_shear[np.newaxis, :], sig_old[np.newaxis, :], hist_old[np.newaxis, :], dt=1.0e-5
    )
    # Plastic flow must occur
    assert hist_shear[0, 0] > 0.0
    # Temperature should rise due to Taylor-Quinney adiabatic heating
    assert hist_shear[0, 1] > 973.15


def test_rate_sensitivity():
    """Verify that a higher strain rate increases the flow stress under viscoplasticity."""
    mat = _sample_material()
    params = build_law103(mat)

    # Prescribe a shear strain increment deps = 0.02
    deps = np.array([[0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])
    sig_0 = np.zeros((1, 6))
    hist_0 = np.array([[0.01, 973.15, 0.0, 0.0]])

    # Case A: Low strain rate (dt = 1.0 s -> eps_dot ~ 0.02 s^-1)
    sig_slow, _, _ = solid_update_array(params, deps, sig_0, hist_0, dt=1.0)

    # Case B: High strain rate (dt = 1.0e-4 s -> eps_dot ~ 200 s^-1)
    sig_fast, _, _ = solid_update_array(params, deps, sig_0, hist_0, dt=1.0e-4)

    # Von Mises stress should be higher at the higher strain rate
    vm_slow = math.sqrt(3.0) * abs(sig_slow[0, 3])
    vm_fast = math.sqrt(3.0) * abs(sig_fast[0, 3])

    assert vm_fast > vm_slow
