"""
Unit tests for /PROP/TYPE28 (/PROP/NSTRAND: Multi-strand cable / wire rope element).

Fortran origin:
  * engine/source/elements/xelem/xforc28.F (constitutive law, capstan friction, damping,
    rupture, internal energy, critical time step)
  * starter/source/properties/xelem/hm_read_prop28.F (property reader, scale factors)
  * starter/source/elements/xelem/xini28.F (geometry, nodal masses, initial dt)
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import nstrand, spring
from pyradioss.model.entities import Property, PropType28, PropStrandLayer
from pyradioss.model.model import ElementGroup, Model


class MockFunction:
    """Mock Radioss function curve with piecewise linear eval and slope."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)
        dx = np.diff(self.x)
        dx = np.where(np.abs(dx) < 1.0e-30, 1.0e-30, dx)
        self.slope = np.diff(self.y) / dx

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))

    def eval_and_slope(self, val: float) -> tuple[float, float]:
        if val <= self.x[0]:
            return float(self.y[0] + self.slope[0] * (val - self.x[0])), float(self.slope[0])
        elif val >= self.x[-1]:
            return float(self.y[-1] + self.slope[-1] * (val - self.x[-1])), float(self.slope[-1])
        else:
            i = int(np.clip(np.searchsorted(self.x, val, side="right") - 1, 0, len(self.x) - 2))
            return float(self.y[i] + self.slope[i] * (val - self.x[i])), float(self.slope[i])


def test_type28_elastic_loading_linear():
    """Verify linear elastic cable response: F = K * (L - L0) / L0 and sum(F) == 0.

    Fortran origin: xforc28.F lines 232-238, 394-426.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = PropType28(
        id=1,
        mass=0.02,     # rho = 0.02 kg/m
        k=1000.0,      # stiffness XK = 1000 N
        c=0.0,
        fun_a1=0,
        fun_b1=0,
        strain1=-1.0e30,
        strain2=1.0e30,
        mu1=0.0,
        mu2=0.0,
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )

    log = MessageLog()
    nstrand.init_nstrand_type28(group, model, log)

    # Initial checks: L0 = 2.0 m, mass = 0.02 * 2.0 = 0.04 kg
    assert group.state["L0"][0] == pytest.approx(2.0)
    assert group.state["mass"][0] == pytest.approx(0.04)

    # Displace nodes: node 1 at x=0, node 2 at x=1.05, node 3 at x=2.10
    # L = 2.10, L0 = 2.0, DL = 0.10, strain = 0.05
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0],
        [2.10, 0.0, 0.0],
    ], dtype=np.float64)
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.001

    fint = np.zeros((3, 3), dtype=np.float64)
    mint = np.zeros((3, 3), dtype=np.float64)

    dtc = nstrand.forces_nstrand_type28(group, x, v, vr, dt, fint, mint)

    # Expected force: F = 1000.0 * 0.05 = 50.0 N
    assert group.state["force"][0] == pytest.approx(50.0)

    # Nodal forces (tension pulls endpoints together):
    # Node 0 (start): +50.0 N in X
    # Node 1 (mid): 0.0 N in X (aligned segments cancel)
    # Node 2 (end): -50.0 N in X
    assert fint[0, 0] == pytest.approx(50.0)
    assert fint[1, 0] == pytest.approx(0.0)
    assert fint[2, 0] == pytest.approx(-50.0)

    # Equilibrium: sum(F) == 0
    assert np.sum(fint, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-12)

    # Critical dt > 0
    assert dtc[0] > 0.0


def test_type28_elastic_loading_nonlinear():
    """Verify nonlinear elastic curve evaluation F = FFAC * func(epstot).

    Fortran origin: xforc28.F lines 250-258.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    curve1 = MockFunction(10, [0.0, 0.10], [0.0, 500.0])
    model.functions = {10: curve1}

    prop = PropType28(
        id=1,
        mass=0.01,
        k=0.0,
        c=0.0,
        fun_a1=10,
        fun_b1=0,
        fscale11=2.0,  # FFAC = 2.0
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )

    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Strain = 0.05 -> f_curve = 250.0 N -> with FFAC=2.0 -> F = 500.0 N
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0],
        [2.10, 0.0, 0.0],
    ], dtype=np.float64)
    fint = np.zeros((3, 3))
    nstrand.forces_nstrand_type28(group, x, None, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(500.0)
    assert fint[0, 0] == pytest.approx(500.0)
    assert fint[2, 0] == pytest.approx(-500.0)
    assert np.sum(fint, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-12)


def test_type28_pulley_friction_bend():
    """Verify Euler-Eytelwein capstan pulley friction across a 90-degree corner:
    FMAX = FF * tanh(0.5 * FRIC * beta), with beta = pi / 2.

    Fortran origin: xforc28.F lines 341-390.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    # 90 degree corner: node 0 at (0, 1, 0), node 1 (pulley) at (0, 0, 0), node 2 at (1, 0, 0)
    # Segment 0: (0, 1, 0) -> (0, 0, 0), direction (0, -1, 0), length 1.0
    # Segment 1: (0, 0, 0) -> (1, 0, 0), direction (1, 0, 0), length 1.0
    # Wrap angle: vprev = (0, -1, 0), vnext = (-1, 0, 0), alpha = 0 -> beta = pi - pi/2 = pi/2
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    mu_pulley = 0.20
    prop = PropType28(
        id=1,
        mass=0.01,
        k=2000.0,
        c=0.0,
        mu1=mu_pulley,
        mu2=0.0,
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )

    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Step 1: Baseline stretch with equal elongation in both branches
    # Node 0 at (0, 1.05, 0), Node 1 at (0, 0, 0), Node 2 at (1.05, 0, 0)
    # L0 = 2.0, L = 2.10, DL = 0.10, strain = 0.05 -> FX = 2000 * 0.05 = 100.0 N
    x = np.array([
        [0.0, 1.05, 0.0],
        [0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0],
    ], dtype=np.float64)
    v = np.zeros_like(x)
    fint = np.zeros((3, 3))
    nstrand.forces_nstrand_type28(group, x, v, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(100.0)

    # Step 2: Now induce sliding across the pulley!
    # Node 2 moves in +x with velocity v_x = 20.0 (branch 1 stretching faster than branch 0)
    # Advance x[2, 0] by v[2, 0] * dt for physical consistency
    v[2, 0] = 20.0
    dt = 0.01
    x[2, 0] += v[2, 0] * dt

    fint = np.zeros((3, 3))
    nstrand.forces_nstrand_type28(group, x, v, None, dt, fint, None)

    # Check Euler-Eytelwein capstan tension limit:
    # beta = pi / 2
    # FF = 2.0 * FX + ...
    # FMAX = FF * tanh(0.5 * mu * beta)
    fx = group.state["force"][0]
    beta = 0.5 * math.pi
    dfs = group.state["t28_dfs"][0]
    ff = 2.0 * fx + dfs[0] + dfs[1]
    fmax_expected = ff * math.tanh(0.5 * mu_pulley * beta)

    # Tension difference between strand 0 and strand 1:
    delta_t = abs(dfs[0] - dfs[1])
    assert delta_t <= fmax_expected * (1.0 + 1.0e-6)
    assert delta_t == pytest.approx(fmax_expected, rel=1.0e-3)

    # Total nodal forces must always satisfy equilibrium sum(F) == 0
    assert np.sum(fint, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_type28_pulley_and_strand_layer_overrides():
    """Verify specific PULLEY and STRAND layer friction overrides.

    Fortran origin: hm_read_prop28.F lines 322-373, xforc28.F lines 352-366:
    FRIC = mu_pulley(k) + 0.5 * mu_strand(k-1) + 0.5 * mu_strand(k).
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Interior node is node 1 (in 1-based Fortran indexing: Pulley number 2)
    # Strands are 1 and 2
    layers = [
        PropStrandLayer(type_name="PULLEY", k_id=2, mu=0.25),
        PropStrandLayer(type_name="STRAND", k_id=1, mu=0.10),
        PropStrandLayer(type_name="STRAND", k_id=2, mu=0.20),
    ]

    prop = PropType28(
        id=1,
        mass=0.01,
        k=1000.0,
        mu1=0.05,
        mu2=0.05,
        layers=layers,
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )

    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Step 1: Baseline stretch
    x = np.array([
        [0.0, 1.05, 0.0],
        [0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0],
    ], dtype=np.float64)
    fint = np.zeros((3, 3))
    nstrand.forces_nstrand_type28(group, x, None, None, 0.001, fint, None)

    # Step 2: Induce sliding with velocity
    v = np.zeros_like(x)
    v[2, 0] = 50.0  # large sliding rate to saturate friction
    dt = 0.01
    x[2, 0] += v[2, 0] * dt

    fint.fill(0.0)
    nstrand.forces_nstrand_type28(group, x, v, None, dt, fint, None)

    fx = group.state["force"][0]
    beta = 0.5 * math.pi
    expected_fric = 0.25 + 0.5 * 0.10 + 0.5 * 0.20  # 0.40
    dfs = group.state["t28_dfs"][0]
    ff = 2.0 * fx + dfs[0] + dfs[1]
    fmax_expected = ff * math.tanh(0.5 * expected_fric * beta)

    delta_t = abs(dfs[0] - dfs[1])
    assert delta_t == pytest.approx(fmax_expected, rel=1.0e-3)


def test_type28_dynamic_damping():
    """Verify viscous damping force FX = F_elast * G(deps * xfac) + XC * deps.

    Fortran origin: xforc28.F lines 260-285.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    # 1. Linear damping only (XC = 20.0, K = 1000.0)
    prop = PropType28(
        id=1,
        mass=0.01,
        k=1000.0,
        c=20.0,  # XC = 20.0
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )
    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Step 1: t=0, at x0 (dx = 0, v = 0)
    x = model.x0.copy()
    fint = np.zeros((3, 3))
    nstrand.forces_nstrand_type28(group, x, None, None, 0.01, fint, None)
    assert group.state["force"][0] == 0.0

    # Step 2: extend by DL = 0.10 in dt = 0.01
    # deps = (0.10 / 2.0) / 0.01 = 0.05 / 0.01 = 5.0 s^-1
    # F_elast = 1000.0 * 0.05 = 50.0 N
    # F_damp = 20.0 * 5.0 = 100.0 N
    # FX = 50.0 + 100.0 = 150.0 N
    x[2, 0] = 2.10
    x[1, 0] = 1.05
    nstrand.forces_nstrand_type28(group, x, None, None, 0.01, fint, None)
    assert group.state["force"][0] == pytest.approx(150.0)

    # 2. Dynamic rate amplification curve G(deps * xfac) on elastic curve (IFUNCT > 0)
    elast_curve = MockFunction(10, [0.0, 0.10], [0.0, 100.0])  # slope = 1000.0 -> F(0.05) = 50.0
    rate_curve = MockFunction(20, [0.0, 10.0], [1.0, 2.0])    # slope = 0.10
    model.functions = {10: elast_curve, 20: rate_curve}
    prop_rate = PropType28(
        id=2,
        mass=0.01,
        k=0.0,
        c=0.0,
        fun_a1=10,
        fun_b1=20,
        fscale22=2.0,  # X_SCAL = 2.0 -> XFAC = 0.5
    )
    group_rate = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop_rate)], "model": model}
    )
    nstrand.init_nstrand_type28(group_rate, model, MessageLog())

    # Initial zero
    nstrand.forces_nstrand_type28(group_rate, model.x0, None, None, 0.01, fint, None)

    # Next step: deps = 5.0 -> deps * XFAC = 5.0 * 0.5 = 2.5
    # G(2.5) = 1.0 + 0.10 * 2.5 = 1.25
    # FX = F_elast * G = 50.0 * 1.25 = 62.5 N
    nstrand.forces_nstrand_type28(group_rate, x, None, None, 0.01, fint, None)
    assert group_rate.state["force"][0] == pytest.approx(62.5)

    # 3. Pure viscous branch (IFUNCT == 0, IFV > 0 -> F = FFAC, xforc28.F lines 240-248)
    prop_visc_only = PropType28(
        id=3,
        mass=0.01,
        fun_a1=0,
        fun_b1=20,
        fscale11=2.0,  # FFAC = 2.0
        fscale22=2.0,  # XFAC = 0.5
    )
    group_visc = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop_visc_only)], "model": model}
    )
    nstrand.init_nstrand_type28(group_visc, model, MessageLog())
    nstrand.forces_nstrand_type28(group_visc, model.x0, None, None, 0.01, fint, None)
    # FX = FFAC * G = 2.0 * 1.25 = 2.5 N
    nstrand.forces_nstrand_type28(group_visc, x, None, None, 0.01, fint, None)
    assert group_visc.state["force"][0] == pytest.approx(2.5)


def test_type28_rupture_criteria():
    """Verify rupture on EPSTOT < EPSMIN and EPSTOT > EPSMAX.

    Fortran origin: xforc28.F lines 311-320, 452-458.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = PropType28(
        id=1,
        mass=0.01,
        k=1000.0,
        strain1=-0.05,  # EPSMIN = -5% compression failure
        strain2=0.10,   # EPSMAX = +10% tension failure
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )
    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Case A: Normal strain within limits (strain = +0.06 < 0.10)
    x = np.array([[0.0, 0.0, 0.0], [1.06, 0.0, 0.0], [2.12, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros((3, 3))
    dtc = nstrand.forces_nstrand_type28(group, x, None, None, 0.01, fint, None)
    assert group.state["off"][0] == 1.0
    assert group.state["force"][0] == pytest.approx(60.0)
    assert dtc[0] < 1.0e10

    # Case B: Exceed tension failure strain (strain = 0.12 > EPSMAX = 0.10)
    x[2, 0] = 2.24
    x[1, 0] = 1.12
    fint.fill(0.0)
    dtc = nstrand.forces_nstrand_type28(group, x, None, None, 0.01, fint, None)

    # Element must be deactivated (off = 0)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    assert np.all(fint == 0.0)
    assert dtc[0] >= 1.0e20

    # Next cycle should remain dead and silent
    nstrand.forces_nstrand_type28(group, x, None, None, 0.01, fint, None)
    assert group.state["off"][0] == 0.0
    assert np.all(fint == 0.0)


def test_type28_energy_conservation():
    """Verify trapezoidal work matches exact analytical strain energy E = 0.5 * K * DL^2 / L0.

    Fortran origin: xforc28.F lines 437-450.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    k_val = 5000.0
    prop = PropType28(
        id=1,
        mass=0.05,
        k=k_val,
        c=0.0,
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )
    nstrand.init_nstrand_type28(group, model, MessageLog())

    # Multi-step elongation
    dt = 0.005
    x = model.x0.copy()
    displacements = [0.02, 0.05, 0.09, 0.14]

    for dl in displacements:
        x[2, 0] = 2.0 + dl
        x[1, 0] = 1.0 + dl * 0.5
        fint = np.zeros((3, 3))
        nstrand.forces_nstrand_type28(group, x, None, None, dt, fint, None)

        # Exact strain energy: 0.5 * K * (DL)^2 / L0
        e_exact = 0.5 * k_val * (dl ** 2) / 2.0
        assert group.state["eint"][0] == pytest.approx(e_exact, rel=1.0e-6)


def test_type28_critical_time_step():
    """Verify explicit critical time step formula DTE = min(DTK, DTC).

    Fortran origin: xforc28.F lines 462-502.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    rho = 0.04
    k_val = 4000.0
    c_val = 10.0
    prop = PropType28(
        id=1,
        mass=rho,
        k=k_val,
        c=c_val,
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )
    nstrand.init_nstrand_type28(group, model, MessageLog())

    x = model.x0.copy()
    x[2, 0] = 2.04
    x[1, 0] = 1.02

    fint = np.zeros((3, 3))
    dtc = nstrand.forces_nstrand_type28(group, x, None, None, 0.001, fint, None)

    # Theoretical time step for strand of length L_k = 1.02, L = 2.04, L0 = 2.0
    # XM = rho * L0_k = 0.04 * 1.0 = 0.04
    # XKM = STIF * (NX - 1) / L = 4000.0 * (2.04 / 2.0) * 2 / 2.04 = 4000.0
    # XCM = XC * L / (L_k * L0) = 10.0 * 2.04 / (1.02 * 2.0) = 10.0
    xm = 0.04
    xkm = 4000.0
    xcm = 10.0
    dtk_expected = (math.sqrt(xcm ** 2 + xm * xkm) - xcm) / xkm
    dtc_expected = xm / xcm
    dte_expected = min(dtk_expected, dtc_expected)

    assert dtc[0] == pytest.approx(dte_expected, rel=1.0e-5)


def test_type28_via_spring_group_integration():
    """Verify integration when /SPRING group contains a /PROP/TYPE28 slice."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=28,
        type=28,
        title="StrandViaSpring",
        params={
            "mass": 0.05,
            "k": 2000.0,
            "c": 0.0,
            "strain1": -0.1,
            "strain2": 0.2,
        }
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)], "model": model}
    )

    node_idx, massn, inertn = spring.init_group(group, model, MessageLog())
    assert len(node_idx) == 3
    # Nodal mass distribution: node 0: 0.5*rho*L1 = 0.5*0.05*1.0 = 0.025
    # node 1: 0.5*rho*(L1+L2) = 0.050
    # node 2: 0.5*rho*L2 = 0.025
    assert massn[0] == pytest.approx(0.025)
    assert massn[1] == pytest.approx(0.050)
    assert massn[2] == pytest.approx(0.025)

    x = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [2.10, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    dtc = spring.forces(group, x, None, None, 0.001, fint, mint)

    # Force: 2000.0 * 0.05 = 100.0 N
    assert group.state["force"][0] == pytest.approx(100.0)
    assert fint[0, 0] == pytest.approx(100.0)
    assert fint[2, 0] == pytest.approx(-100.0)
    assert dtc[0] > 0.0
