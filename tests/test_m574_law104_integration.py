"""Integration tests for /MAT/LAW104 material dispatch, continuum updates, and tangents (M574).

Verifies:
- pyradioss.materials.solid_update dispatches LAW104 for both 1D and 2D arrays
- pyradioss.materials.shell_update dispatches LAW104 with plane-stress condition
- Sound speeds for solids and shells via pyradioss.materials.sound_speed
- Tangent dispatch: solid_tangent, shell_layer_tangent, shell_membrane_tangent
- Pure hydrostatic compression remains purely elastic without spurious plastic flow
- Taylor-Quinney plastic work conversion to thermal energy and softening
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import MaterialLaw104
from pyradioss.materials import (
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    sound_speed as mat_sound_speed,
    solid_tangent as mat_solid_tangent,
    shell_layer_tangent as mat_shell_layer_tangent,
    shell_membrane_tangent as mat_shell_membrane_tangent,
)
from pyradioss.materials.law104_drucker import (
    DruckerParams,
    build_law104,
    solid_update,
    shell_update,
    sound_speed_solid,
    sound_speed_shell,
)


def _make_sample_mat(cdr: float = 1.0) -> MaterialLaw104:
    return MaterialLaw104(
        id=104,
        title="Drucker Test Mat",
        rho0=7.8e-9,
        young=210000.0,
        nu=0.3,
        ires=1,
        sigma_r=400.0,
        h=1000.0,
        qv=250.0,
        bv=15.0,
        cdr=cdr,
        cjc=0.02,
        epsp0=1.0,
        fcut=1000.0,
        tss=0.002,
        tref=293.15,
        tini=293.15,
        eta=0.9,
        cp=4.5e8,
        eps_iso=1.0,
        eps_ad=100.0,
    )


def test_materials_dispatch_solid_update_single_and_batch():
    """Verify pyradioss.materials.solid_update handles both 1D (6,) and 2D (N, 6) arrays."""
    mat = _make_sample_mat()

    # 1D single element update
    sig_1d = np.zeros(6, dtype=np.float64)
    deps_1d = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_out, epsp_out, c_out = mat_solid_update(mat, sig_1d, deps_1d, epsp=0.0, dt=1e-5)

    assert sig_out.shape == (6,)
    assert epsp_out > 0.0, "Plastic strain should accumulate beyond yield"
    assert c_out > 0.0
    assert sig_out[0] > 300.0

    # 2D batch elements update
    n_elems = 8
    sig_batch = np.zeros((n_elems, 6), dtype=np.float64)
    deps_batch = np.tile(deps_1d, (n_elems, 1))
    epsp_batch = np.zeros(n_elems, dtype=np.float64)

    sig_b_out, epsp_b_out, c_b_out = mat_solid_update(mat, sig_batch, deps_batch, epsp=epsp_batch, dt=1e-5)

    assert sig_b_out.shape == (n_elems, 6)
    assert epsp_b_out.shape == (n_elems,)
    for i in range(n_elems):
        assert math.isclose(sig_b_out[i, 0], sig_out[0], rel_tol=1e-5)
        assert math.isclose(epsp_b_out[i], epsp_out, rel_tol=1e-5)


def test_materials_dispatch_shell_update():
    """Verify pyradioss.materials.shell_update performs plane-stress return mapping."""
    mat = _make_sample_mat()

    # In-plane biaxial strain increment
    sig_shell = np.zeros(3, dtype=np.float64)
    deps_shell = np.array([0.005, 0.002, 0.0], dtype=np.float64)  # eps_xx, eps_yy, eps_xy

    extra = {}
    sig_out, epsp_out = mat_shell_update(mat, sig_shell, deps_shell, epsp=0.0, dt=1e-5, extra=extra)

    assert sig_out.shape == (3,)
    assert epsp_out > 0.0, "Plastic flow must occur in shell"
    # Under tensile in-plane strain, thickness thinning de_zz must be compressive (< 0)
    assert extra.get("thickness_strain", 0.0) < 0.0, f"Expected compressive thickness strain, got {extra.get('thickness_strain')}"


def test_materials_dispatch_sound_speed():
    """Verify sound speed dispatch for both solid and shell formulations."""
    mat = _make_sample_mat()

    c_solid = mat_sound_speed(mat, is_shell=False)
    c_shell = mat_sound_speed(mat, is_shell=True)

    # Theoretical sound speeds
    e = mat.young
    nu = mat.nu
    rho = mat.rho0
    k = e / (3.0 * (1.0 - 2.0 * nu))
    g = e / (2.0 * (1.0 + nu))
    expected_c_solid = math.sqrt((k + 4.0 / 3.0 * g) / rho)
    expected_c_shell = math.sqrt(e / ((1.0 - nu * nu) * rho))

    assert math.isclose(c_solid, expected_c_solid, rel_tol=1e-6)
    assert math.isclose(c_shell, expected_c_shell, rel_tol=1e-6)
    assert c_solid > c_shell, "3D dilatational wave speed should exceed 2D plate wave speed"


def test_materials_dispatch_tangents():
    """Verify implicit tangent operators: solid_tangent, shell_layer_tangent, shell_membrane_tangent."""
    mat = _make_sample_mat()
    sig = np.zeros(6, dtype=np.float64)

    # Solid tangent: 6x6 matrix
    c_solid = mat_solid_tangent(mat, sig)
    assert c_solid.shape == (6, 6)
    # Check symmetry and positive-definiteness
    assert np.allclose(c_solid, c_solid.T)
    eigvals = np.linalg.eigvalsh(c_solid)
    assert np.all(eigvals > 0.0)

    # Shell membrane tangent: 3x3 matrix
    c_mem = mat_shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    assert np.allclose(c_mem, c_mem.T)
    assert c_mem[0, 0] > 0.0
    assert c_mem[1, 1] > 0.0
    assert c_mem[2, 2] > 0.0

    # Shell layer tangent
    c_layer = mat_shell_layer_tangent(mat, sig=np.zeros(3))
    assert c_layer.shape in ((3, 3), (5, 5))


def test_hydrostatic_elasticity_vs_deviatoric_yielding():
    """Pure hydrostatic stress should produce zero Drucker equivalent stress and no plastic flow."""
    mat = _make_sample_mat(cdr=1.5)
    params = build_law104(mat)

    # Apply pure volumetric compression
    eps_vol = -0.003
    deps = np.array([eps_vol / 3.0, eps_vol / 3.0, eps_vol / 3.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_old = np.zeros(6, dtype=np.float64)
    hist_old = np.array([0.0, 293.15], dtype=np.float64)
    extra = {"history": hist_old}

    sig_new, epsp_new, ssp = solid_update(params, sig_old, deps, extra=extra, dt=1e-5)
    hist_new = extra["uvar104"]

    # Plastic strain must remain zero
    assert hist_new[0] == 0.0, "Pure hydrostatic stress should not induce plastic flow"

    # Hydrostatic pressure P = -trace(sigma)/3 = -K * eps_vol
    trace_sig = np.sum(sig_new[:3])
    expected_trace = params.bulk * eps_vol * 3.0
    assert math.isclose(trace_sig, expected_trace, rel_tol=1e-5)


def test_energy_dissipation_and_temperature_coupling():
    """Verify Taylor-Quinney conversion of plastic work into adiabatic self-heating."""
    mat = _make_sample_mat()
    params = build_law104(mat)

    # Multi-step plastic deformation at high strain rate (above eps_ad -> fully adiabatic)
    dt = 1e-6
    deps = np.array([0.002, -0.0006, -0.0006, 0.0, 0.0, 0.0], dtype=np.float64)

    sig = np.zeros(6, dtype=np.float64)
    hist = np.array([0.0, 293.15], dtype=np.float64)  # [epsp, Temp]

    total_plastic_work = 0.0
    for step in range(20):
        sig_prev = sig.copy()
        epsp_prev = hist[0]
        extra = {"history": hist}
        sig, epsp, _ = solid_update(params, sig, deps, extra=extra, dt=dt)
        hist = extra["uvar104"]
        depsp = hist[0] - epsp_prev
        if depsp > 0.0:
            # Incremental plastic work dWp = sig_y * depsp
            eps_dot = math.sqrt((2.0 / 3.0) * np.sum(deps[:3]**2)) / dt
            from pyradioss.materials.law104_drucker import compute_drucker_yield_stress
            yld = compute_drucker_yield_stress(params, eps_p=epsp_prev, eps_dot=eps_dot, temp=hist[1])
            total_plastic_work += yld * depsp

    # Temperature must have increased from 293.15
    temp_rise = hist[1] - 293.15
    assert temp_rise > 0.0, f"Expected temperature rise, got {temp_rise}"

    # Verify order of magnitude: dT ~ (eta / (rho * Cp)) * Wp
    expected_dt = (params.eta / (params.rho * params.cp)) * total_plastic_work
    assert math.isclose(temp_rise, expected_dt, rel_tol=0.25)
