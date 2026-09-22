"""
Unit tests for /PROP/TYPE26 (/PROP/SPR_TAB: Tabulated Nonlinear Spring).

Fortran origin:
  engine/source/elements/spring/r26def3.F
  engine/source/elements/spring/r26sig.F
  engine/source/elements/spring/rforc3.F
  starter/source/properties/spring/hm_read_prop26.F
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_advanced
from pyradioss.model.entities import Property, PropType26, PropType26Curve
from pyradioss.model.model import ElementGroup, Model


class MockFunction:
    """Mock Radioss function curve."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))


def test_type26_elastic_tension():
    """Verify linear elastic response in tension (DX >= 0).
    Fortran r26sig.F lines 125-126: FX = XK * DX.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=26, title="TabSpring_Tension",
        params={
            "mass": 5.0,
            "stiff0": 2000.0,
            "alpha": 1.0,
            "scale": 1.0,
            "ileng": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Displace node 2 by +0.1 along X (L = 2.1, DL = 0.1)
    x = model.x0.copy()
    x[1, 0] = 2.1
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt_crit = spring.forces(group, x, v, vr, dt, fint, mint)

    # Force: F = K0 * DL = 2000.0 * 0.1 = 200.0 N
    # Tension pulls nodes together: Node 0 gets +200 on X, Node 1 gets -200 on X
    assert group.state["force"][0] == pytest.approx(200.0)
    assert fint[0, 0] == pytest.approx(200.0)
    assert fint[1, 0] == pytest.approx(-200.0)
    # Energy: 0.5 * 0.1 * 200.0 = 10.0 J
    assert group.state["eint"][0] == pytest.approx(10.0)
    # Critical dt: omega = 2 * sqrt(2000 / 5) = 2 * 20 = 40 rad/s -> dt_crit = 2/40 = 0.05 s
    assert dt_crit[0] == pytest.approx(0.05)


def test_type26_compressive_loading_curve():
    """Verify compression yielding clamped to loading curve FMAX.
    Fortran r26sig.F lines 131-175, 222-228.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    # Function 10: F_yield(x) = 500.0 * x
    model.functions[10] = MockFunction(10, [0.0, 0.5, 1.0], [0.0, 250.0, 500.0])

    prop = Property(
        id=1, type=26, title="TabSpring_Compress",
        params={
            "mass": 4.0,
            "stiff0": 50000.0,  # High unloading stiffness
            "alpha": 1.0,
            "scale": 1.0,
            "ileng": 0,
            "load_curves": [{"fun_load": 10, "scale_load": 1.0, "strainrate_load": 0.0}],
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Compress by 0.1: L = 1.9, DL = -0.1
    # Elastic trial force: 50000 * (-0.1) = -5000 N
    # Yield limit: -f10(0.1) = -50.0 N
    # Clamped to FMAX = -50.0 N
    x = model.x0.copy()
    x[1, 0] = 1.9
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, mint)

    assert group.state["force"][0] == pytest.approx(-50.0)
    # Compression pushes nodes apart: Node 0 gets -50 on X, Node 1 gets +50 on X
    assert fint[0, 0] == pytest.approx(-50.0)
    assert fint[1, 0] == pytest.approx(50.0)


def test_type26_compressive_unloading_curve():
    """Verify compression unloading clamped to unloading curve FMIN.
    Fortran r26sig.F lines 176-228.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    # Loading curve 10: F_load(x) = 1000 * x
    model.functions[10] = MockFunction(10, [0.0, 1.0], [0.0, 1000.0])

    # Unloading curve 20: F_unload(x) = 200 * x
    model.functions[20] = MockFunction(20, [0.0, 1.0], [0.0, 200.0])

    prop = Property(
        id=1, type=26, title="TabSpring_Unload",
        params={
            "mass": 4.0,
            "stiff0": 20000.0,
            "alpha": 1.0,
            "scale": 1.0,
            "ileng": 0,
            "load_curves": [{"fun_load": 10, "scale_load": 1.0, "strainrate_load": 0.0}],
            "unload_curves": [{"fun_unload": 20, "scale_unload": 1.0, "strainrate_load": 0.0}],
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: Compress to DL = -0.2
    # FMAX = -1000 * 0.2 = -200 N
    x = model.x0.copy()
    x[1, 0] = 1.8
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, None)
    assert group.state["force"][0] == pytest.approx(-200.0)

    # Step 2: Unload partially to DL = -0.1 (DDX = +0.1)
    # Trial force: -200 + 20000 * 0.1 = +1800 N
    # Unload curve FMIN = -200 * 0.1 = -20.0 N
    # Clamped to FMIN = -20.0 N
    x[1, 0] = 1.9
    fint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, None)
    assert group.state["force"][0] == pytest.approx(-20.0)


def test_type26_strain_rate_interpolation():
    """Verify rate interpolation between two yield curves at different strain rates.
    Fortran r26sig.F lines 135-167: fac = (DV - RATE1) / (RATE2 - RATE1).
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    # Rate 1: 100 * x
    model.functions[1] = MockFunction(1, [0.0, 1.0], [0.0, 100.0])

    # Rate 10: 300 * x
    model.functions[2] = MockFunction(2, [0.0, 1.0], [0.0, 300.0])

    prop = Property(
        id=1, type=26, title="Rate_Spring",
        params={
            "mass": 1.0,
            "stiff0": 10000.0,
            "alpha": 1.0,
            "scale": 1.0,
            "ileng": 0,
            "load_curves": [
                {"fun_load": 1, "scale_load": 1.0, "strainrate_load": 1.0},
                {"fun_load": 2, "scale_load": 1.0, "strainrate_load": 10.0},
            ],
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Compress by 0.1 in dt = 0.01818 s -> velocity = 0.1 / 0.01818 = 5.5 m/s
    # Midway between rate 1.0 and 10.0: fac = (5.5 - 1.0) / 9.0 = 4.5 / 9.0 = 0.5
    # Interpolated yield curve: 0.5 * 100 * 0.1 + 0.5 * 300 * 0.1 = 5.0 + 15.0 = 20.0 N
    # FMAX = -20.0 N
    dt = 0.1 / 5.5
    x = model.x0.copy()
    x[1, 0] = 1.9

    fint = np.zeros((2, 3))
    spring.forces(group, x, None, None, dt, fint, None)
    assert group.state["force"][0] == pytest.approx(-20.0)


def test_type26_velocity_filtering():
    """Verify velocity filtering with alpha < 1.0.
    Fortran r26sig.F lines 109-110: DV = (1 - alpha) * DV0 + alpha * DV_new.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=26, title="Filter_Spring",
        params={
            "mass": 1.0,
            "stiff0": 1000.0,
            "alpha": 0.25,  # 25% weight on new velocity
            "scale": 1.0,
            "ileng": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: initial elongation 0.1 with dt = 0.01 -> velocity = 10.0
    # DV = (1 - 0.25) * 0.0 + 0.25 * 10.0 = 2.5
    x = model.x0.copy()
    x[1, 0] = 2.1
    spring.forces(group, x, None, None, 0.01, None, None)
    assert group.state["t26_dv0"][0] == pytest.approx(2.5)

    # Step 2: keep elongation at 0.1 -> ddx = 0, new velocity = 0.0
    # DV = (1 - 0.25) * 2.5 + 0.25 * 0 = 0.75 * 2.5 = 1.875
    spring.forces(group, x, None, None, 0.01, None, None)
    assert group.state["t26_dv0"][0] == pytest.approx(1.875)


def test_type26_rupture_limits():
    """Verify displacement rupture in tension (DMAX) and compression (DMIN).
    Fortran r26def3.F lines 156-170.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=26, title="Rupture_Spring",
        params={
            "mass": 1.0,
            "stiff0": 1000.0,
            "dmin": -0.15,
            "dmax": 0.25,
            "ileng": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    # Within limits: DL = 0.20 <= 0.25
    x[1, 0] = 2.20
    spring.forces(group, x, None, None, 0.01, None, None)
    assert group.state["off"][0] == 1.0
    assert group.state["force"][0] == pytest.approx(200.0)

    # Exceed tension limit: DL = 0.26 > 0.25 -> Rupture!
    x[1, 0] = 2.26
    spring.forces(group, x, None, None, 0.01, None, None)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0


def test_type26_engineering_strain_mode():
    """Verify ILENG = 1 engineering strain mode (scale by L0).
    Fortran r26def3.F line 141, r26sig.F lines 90-93.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Initial length L0 = 4.0
    model.x0 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=26, title="Strain_Spring",
        params={
            "mass": 2.0,
            "stiff0": 1000.0,
            "ileng": 1,  # Engineering strain mode
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Displace by 0.4: engineering strain epsilon = 0.4 / 4.0 = 0.1
    x = model.x0.copy()
    x[1, 0] = 4.4
    spring.forces(group, x, None, None, 0.01, None, None)

    # Force: K0 * epsilon = 1000 * 0.1 = 100.0 N
    assert group.state["force"][0] == pytest.approx(100.0)
