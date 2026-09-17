"""Unit and regression tests for shell element formulations (BT4, QEPH, QBAT, TRI3, THICK16, SHEL16)."""

import types
import numpy as np
import pytest

from pyradioss.model.model import ElementGroup, Model
from pyradioss.elements import (
    shell_bt4,
    shell_qeph,
    shell_qbat,
    shell_tri3,
    shell_thick16,
)


def _make_elastic_mat_and_prop(thick=0.01, nu=0.3, E=2.1e11, rho0=7800.0):
    mat = types.SimpleNamespace(
        id=1, law=1, rho0=rho0, E=E, nu=nu,
        G=E / (2.0 * (1.0 + nu)), K=E / (3.0 * (1.0 - 2.0 * nu)),
        eos=None, fail=None, params={"E": E, "nu": nu, "rho": rho0}
    )
    mat.sound_speed_shell = lambda: np.sqrt(E / (rho0 * (1.0 - nu**2)))
    prop = types.SimpleNamespace(
        id=1, type=1, thick=thick,
        params={"thick": thick, "hm": 0.1, "hf": 0.1, "hr": 0.1, "nip": 3}
    )
    return mat, prop


def test_bt4_shell_basic_execution():
    """Verify BT4 shell initialization and force computation with positive dt."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64))
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    shell_bt4.init_group(group, model, log)
    assert "qshear" in group.state
    assert "hgq" in group.state
    assert "sig" in group.state

    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    dt_crit = shell_bt4.forces(group, x, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_qeph_shell_basic_execution():
    """Verify QEPH shell initialization and force computation."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64))
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    shell_qeph.init_group(group, model, log)
    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    dt_crit = shell_qeph.forces(group, x, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_qbat_shell_basic_execution():
    """Verify QBAT shell initialization and force computation."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64))
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    shell_qbat.init_group(group, model, log)
    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    dt_crit = shell_qbat.forces(group, x, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_tri3_shell_basic_execution():
    """Verify TRI3 triangular shell initialization and force computation."""
    mat, prop = _make_elastic_mat_and_prop()
    conn = np.array([[0, 1, 2]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64))
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    shell_tri3.init_group(group, model, log)
    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    dt_crit = shell_tri3.forces(group, x, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


def test_thick16_shell_basic_execution():
    """Verify THICK16 thick shell initialization and force computation."""
    mat, prop = _make_elastic_mat_and_prop(thick=1.0)
    conn = np.arange(16, dtype=np.int64).reshape(1, 16)
    x0 = np.zeros((16, 3))
    for i in range(8):
        x0[i] = [i % 4, i // 4, 0.0]
        x0[8 + i] = [i % 4, i // 4, 1.0]

    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    model = types.SimpleNamespace(x0=x0, numnod=len(x0))
    log = types.SimpleNamespace(error=lambda *args: None)
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    shell_thick16.init_group(group, model, log)
    v = np.zeros_like(x0)
    vr = np.zeros_like(x0)
    fint = np.zeros_like(x0)
    mint = np.zeros_like(x0)

    dt_crit = shell_thick16.forces(group, x0, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0
