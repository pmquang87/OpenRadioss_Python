"""Unit and regression tests for solid element formulations (HEXA8, HEPH, TETRA4, TETRA10, BRIC20, QUAD)."""

import types
import numpy as np
import pytest

from pyradioss.model.model import ElementGroup, Model
from pyradioss.elements import (
    solid_hexa8,
    solid_heph,
    solid_tetra4,
    solid_tetra10,
    solid_bric20,
    solid_quad,
)


def _make_elastic_mat_and_prop(E=2.1e11, nu=0.3, rho0=7800.0):
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    mat = types.SimpleNamespace(
        id=1, law=1, rho0=rho0, E=E, nu=nu, K=K, G=G,
        eos=None, fail=None, params={"E": E, "nu": nu, "rho": rho0}
    )
    prop = types.SimpleNamespace(
        id=1, type=14, qa=1.1, qb=0.05,
        params={"qa": 1.1, "qb": 0.05}
    )
    return mat, prop


def test_hexa8_solid_execution():
    """Verify standard 8-node brick element initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.arange(8, dtype=np.int64).reshape(1, 8)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    x0 = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_hexa8.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_hexa8.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_heph_solid_execution():
    """Verify 8-node physically-stabilized brick element initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.arange(8, dtype=np.int64).reshape(1, 8)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    x0 = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_heph.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_heph.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_tetra4_solid_execution():
    """Verify 4-node tetrahedral solid initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_tetra4.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_tetra4.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_tetra10_solid_execution():
    """Verify 10-node quadratic tetrahedral solid initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    x4 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    x0 = np.zeros((10, 3), dtype=np.float64)
    x0[:4] = x4
    for m, (n1, n2) in enumerate([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]):
        x0[4 + m] = 0.5 * (x4[n1] + x4[n2])

    conn = np.arange(10, dtype=np.int64).reshape(1, 10)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None, warning=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_tetra10.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_tetra10.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_bric20_solid_execution():
    """Verify 20-node quadratic brick solid initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    x0 = solid_bric20._XI_NODES.copy()
    conn = np.arange(20, dtype=np.int64).reshape(1, 20)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_bric20.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_bric20.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_quad_solid_execution():
    """Verify 4-node 2D solid quad initialization and forces."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.arange(4, dtype=np.int64).reshape(1, 4)
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=x0)
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    solid_quad.init_group(group, model, log)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)

    dt_crit = solid_quad.forces(group, x0, v, vr, dt=1e-5, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0
