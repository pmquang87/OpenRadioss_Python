"""Unit tests for LAW50 (Rate-Dependent Viscoelastic Honeycomb Material).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat050\\sigeps50.F
  (engine/source/materials/mat/mat050/sigeps50s.F90)
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\materials\\mat\\mat050\\hm_read_mat50.F90
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials.law50_visc_honey import (
    Law50Params,
    build_law50,
    consistent_solid_tangent,
    extra_shapes,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
    sound_speed_solid,
    tangent,
)
from pyradioss.model.entities import Material


def _make_honeycomb_mat(
    rho0: float = 1.0e-3,
    ea: float = 100.0,
    eb: float = 200.0,
    ec: float = 300.0,
    gab: float = 40.0,
    gbc: float = 50.0,
    gca: float = 60.0,
    fcut: float = 0.0,
    irate: int = 2,
    ecomp: float = 0.0,
    et: float = 0.0,
    sigy: float = 0.0,
    pr: float = 0.0,
    vcomp: float = 0.0,
) -> Material:
    """Helper to build a Material entity with LAW50 parameters."""
    mat = Material(id=50, law="50", rho0=rho0)
    mat.params = {
        "MAT_RHO": rho0,
        "MAT_EA": ea,
        "MAT_EB": eb,
        "MAT_EC": ec,
        "MAT_GAB": gab,
        "MAT_GBC": gbc,
        "MAT_GCA": gca,
        "MAT_asrate": fcut,
        "asrate": fcut,
        "fcut": fcut,
        "Irate": irate,
        "irate": irate,
        "MAT_ECOMP": ecomp,
        "ecomp": ecomp,
        "MAT_ET": et,
        "et": et,
        "MAT_SIGY": sigy,
        "sigy": sigy,
        "MAT_PR": pr,
        "pr": pr,
        "MAT_VCOMP": vcomp,
        "vcomp": vcomp,
    }
    return mat


def test_law50_orthotropic_elasticity_uncoupled():
    """Verify orthotropic elasticity with distinct Ea, Eb, Ec, Gab, Gbc, Gca and zero Poisson coupling."""
    ea, eb, ec = 150.0, 250.0, 350.0
    gab, gbc, gca = 45.0, 55.0, 65.0
    mat = _make_honeycomb_mat(ea=ea, eb=eb, ec=ec, gab=gab, gbc=gbc, gca=gca)

    sig_init = np.zeros(6)

    # 1. Direct normal strain in 11 (x) direction
    deps_11 = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_11, dt=1.0e-5, return_tuple=True)

    assert pytest.approx(sig_out[0], rel=1e-5) == ea * deps_11[0]
    # Zero Poisson coupling in honeycomb state: transverse stresses must be 0
    assert pytest.approx(sig_out[1], abs=1e-12) == 0.0
    assert pytest.approx(sig_out[2], abs=1e-12) == 0.0
    assert np.allclose(sig_out[3:], 0.0, atol=1e-12)

    # 2. Direct normal strain in 22 (y) direction
    deps_22 = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_22, dt=1.0e-5, return_tuple=True)
    assert pytest.approx(sig_out[1], rel=1e-5) == eb * deps_22[1]
    assert pytest.approx(sig_out[0], abs=1e-12) == 0.0
    assert pytest.approx(sig_out[2], abs=1e-12) == 0.0

    # 3. Direct normal strain in 33 (z) direction
    deps_33 = np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_33, dt=1.0e-5, return_tuple=True)
    assert pytest.approx(sig_out[2], rel=1e-5) == ec * deps_33[2]
    assert pytest.approx(sig_out[0], abs=1e-12) == 0.0
    assert pytest.approx(sig_out[1], abs=1e-12) == 0.0

    # 4. Pure shear strains (engineering shear: tau = G * gamma)
    deps_12 = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_12, dt=1.0e-5, return_tuple=True)
    assert pytest.approx(sig_out[3], rel=1e-5) == gab * deps_12[3]

    deps_23 = np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.0])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_23, dt=1.0e-5, return_tuple=True)
    assert pytest.approx(sig_out[4], rel=1e-5) == gbc * deps_23[4]

    deps_31 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.03])
    sig_out, _, _ = solid_update(mat, sig_init.copy(), deps_31, dt=1.0e-5, return_tuple=True)
    assert pytest.approx(sig_out[5], rel=1e-5) == gca * deps_31[5]


def test_law50_strain_rate_filtering():
    """Verify strain rate filtering: asrate = min(1.0, fcut * dt) with directional (irate=2) and equivalent (irate=1)."""
    fcut = 500.0  # Cutoff frequency
    dt = 1.0e-4   # asrate = min(1.0, 500.0 * 1e-4) = 0.05
    mat_dir = _make_honeycomb_mat(ea=100.0, fcut=fcut, irate=2)
    mat_eq = _make_honeycomb_mat(ea=100.0, fcut=fcut, irate=1)

    # Directional rate filtering (irate=2)
    extra_dir = {"uvar50": np.zeros((1, 6))}
    deps = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    solid_update(mat_dir, np.zeros((1, 6)), deps, dt=dt, extra=extra_dir)

    raw_rate = 0.001 / dt  # 10.0
    expected_rate = 0.05 * raw_rate  # filtered = 0.05 * 10.0 = 0.5
    assert pytest.approx(extra_dir["uvar50"][0, 0], rel=1e-4) == expected_rate

    # Equivalent rate filtering (irate=1)
    extra_eq = {"uvar50": np.zeros((1, 6))}
    solid_update(mat_eq, np.zeros((1, 6)), deps, dt=dt, extra=extra_eq)
    assert extra_eq["uvar50"][0, 0] > 0.0


def test_law50_compaction_transition_and_coupling():
    """Verify transition to compacted state when rvol <= vcomp, engaging bulk modulus and coupling."""
    ecomp = 1000.0
    pr = 0.25
    vcomp = 0.5
    sigy = 200.0
    mat = _make_honeycomb_mat(
        ea=50.0, eb=50.0, ec=50.0,
        ecomp=ecomp, pr=pr, vcomp=vcomp, sigy=sigy,
    )

    # Relative volume below vcomp triggers full compaction
    extra = {"amu": np.array([1.5])}  # mu = 1.5 -> rvol = 1 / (1 + 1.5) = 0.4 <= vcomp = 0.5
    sig_init = np.zeros((1, 6))
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 1.0e-5

    sig_out, epsp, c = solid_update(mat, sig_init, deps, dt=dt, extra=extra, return_tuple=True)

    # In compacted state, bulk modulus couples normal directions: sig_yy and sig_zz become non-zero
    assert abs(sig_out[0, 1]) > 0.0
    assert abs(sig_out[0, 2]) > 0.0
    assert extra.get("compacted") is not None
    assert bool(np.all(extra["compacted"])) is True


def test_law50_sound_speed():
    """Verify acoustic sound speed calculation in uncompacted and compacted regimes."""
    rho0 = 1.0e-3
    ea, eb, ec = 100.0, 200.0, 300.0
    ecomp = 2000.0
    mat = _make_honeycomb_mat(rho0=rho0, ea=ea, eb=eb, ec=ec, ecomp=ecomp, vcomp=0.5, sigy=100.0)

    # Uncompacted sound speed: sqrt(max(E) / rho) = sqrt(300 / 1e-3) = sqrt(300000)
    c_uncomp = sound_speed_solid(mat)
    expected_c_uncomp = math.sqrt(300.0 / rho0)
    assert pytest.approx(c_uncomp, rel=1e-5) == expected_c_uncomp

    # Compacted sound speed: sqrt(ecomp / rho) = sqrt(2000 / 1e-3)
    extra_comp = {"compacted": np.array([True])}
    c_comp = sound_speed_solid(mat, extra=extra_comp)
    expected_c_comp = math.sqrt(ecomp / rho0)
    assert pytest.approx(c_comp, rel=1e-5) == expected_c_comp


def test_law50_element_group_interface():
    """Verify element group calling convention: solid_update(group, x, u, ur, dt, fint, mint) and tangent(group)."""
    p = Law50Params(
        rho0=1.0e-3,
        ea=100.0,
        eb=200.0,
        ec=300.0,
        gab=40.0,
        gbc=50.0,
        gca=60.0,
    )
    group = SimpleNamespace(
        nel=1,
        elements=[1],
        mat=p,
    )
    fint_in = np.zeros(24)

    # Element group style solid_update
    res = solid_update(group, x=None, u=None, ur=None, dt=1.0e-5, fint=fint_in)
    assert res is fint_in

    # Element group style tangent
    tang = tangent(group)
    assert tang is not None
    assert tang.shape == (1, 6, 6)
    # Check orthotropic diagonal stiffness values
    assert pytest.approx(tang[0, 0, 0]) == 100.0
    assert pytest.approx(tang[0, 1, 1]) == 200.0
    assert pytest.approx(tang[0, 2, 2]) == 300.0
    assert pytest.approx(tang[0, 3, 3]) == 40.0
    assert pytest.approx(tang[0, 4, 4]) == 50.0
    assert pytest.approx(tang[0, 5, 5]) == 60.0


def test_law50_consistent_tangent_uncoupled():
    """Verify consistent solid tangent produces block diagonal uncoupled tensor in uncompacted state."""
    mat = _make_honeycomb_mat(ea=120.0, eb=220.0, ec=320.0, gab=42.0, gbc=52.0, gca=62.0)
    c_algo = consistent_solid_tangent(mat)
    assert c_algo.shape == (1, 6, 6)

    d = c_algo[0]
    expected_diag = [120.0, 220.0, 320.0, 42.0, 52.0, 62.0]
    for i in range(6):
        assert pytest.approx(d[i, i]) == expected_diag[i]
        for j in range(6):
            if i != j:
                assert d[i, j] == 0.0


def test_law50_shell_rejection():
    """Verify shell_update raises NotImplementedError (LAW50 is 3D solid only)."""
    mat = _make_honeycomb_mat()
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update(mat, np.zeros(3), np.zeros(3))


def test_law50_extra_shapes():
    """Verify extra_shapes returns required persistent state fields."""
    mat = _make_honeycomb_mat()
    shapes = extra_shapes(mat)
    assert "eps50" in shapes
    assert "off50" in shapes
    assert "uvar50" in shapes
    assert shapes["eps50"] == (6,)
    assert shapes["uvar50"] == (6,)
