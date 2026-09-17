"""Unit tests verifying Wave 2 Fix Subagent 3 bugs BUG-ENG-01 through BUG-ENG-07."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.model.model import Model
from pyradioss.model.entities import Sensor, RigidWall, RigidBody, ImposedVelocity, ImposedDisplacement, NodeGroup
from pyradioss.engine.rigid_wall import RigidWalls
from pyradioss.engine.mass_scaling import NodalTimeStep
from pyradioss.engine.rigid_body import RigidBodyEngine
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.engine.sensors import Sensors


class _Controls:
    def __init__(self, cst=False, dt_min=0.0, dt_scale=1.0):
        self.dt_noda = "CST" if cst else "DEL"
        self.dt_min = dt_min
        self.dt_scale = dt_scale


def test_bug_eng_01_rigid_wall_impact_work():
    """BUG-ENG-01: For stationary wall (wnode < 0), impact work -U goes to removed, NOT wext."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.mass = np.array([2.5, 2.5])
    model.x = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 1.0]])
    model.v = np.array([[3.0, 4.0, -1.0], [0.0, 0.0, 0.0]])
    v_old = model.v.copy()
    dt = 0.05

    rw = RigidWall(
        id=1, point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
        geom="PLANE", slide=0
    )
    model.rwalls = [rw]
    log = MessageLog()
    walls = RigidWalls(model, log)

    de, dw = walls.apply(model.x, model.v, v_old, model.mass, dt)
    assert de == pytest.approx(1.05, rel=1e-12)
    assert dw == 0.0


def test_bug_eng_02_mass_scaling_rbody_slave_stiffness_and_cst_update():
    """BUG-ENG-02: Slave stiffness accumulated to master, and CST updates rb.M and rb.J0."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.mass = np.array([1.0, 1.0])
    model.inertia = np.array([1.0, 1.0])
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    model.x = model.x0.copy()
    model.v = np.zeros((2, 3))

    class MockRB:
        def __init__(self):
            self.id = 1
            self.kind = "RBODY"
            self.master = 0
            self.slaves = np.array([1], dtype=np.int64)
            self.nodes = np.array([0, 1], dtype=np.int64)
            self.mass_total = 2.0
            self.M = 2.0
            self.J0 = np.eye(3) * 2.0

    rb = MockRB()
    controls = _Controls(cst=True, dt_min=0.1, dt_scale=1.0)
    log = MessageLog()
    noda = NodalTimeStep(model, controls, log)
    noda._rot = True
    noda.set_prescribed(np.array([1]))
    noda.add_rigid_body(rb.nodes, rb.master, rb.M, rb.J0, model.x0, rb=rb)

    # Put stiffness on slave node 1
    noda.stifn[1] = 1000.0
    noda.stifr[1] = 500.0

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    inv_inertia = 1.0 / model.inertia

    # Target dt = 0.1 / 1.0 = 0.1 -> m_req = K * dt^2 / 2 = 1000 * 0.01 / 2 = 5.0
    # rb.M is 2.0 -> dm = 3.0 added to rb.M and master node
    dt = noda.apply(mass_eff, inv_mass, model.v, t=0.0,
                    inertia=model.inertia, inv_inertia=inv_inertia)

    assert rb.M == pytest.approx(5.0, rel=1e-12)
    assert model.mass[0] == pytest.approx(4.0, rel=1e-12)  # 1.0 + 3.0
    assert noda.mass_added == pytest.approx(3.0, rel=1e-12)
    # Check that rb.J0 was also updated
    assert rb.J0[0, 0] > 2.0


def test_bug_eng_03_rigid_body_rotational_skew_drive():
    """BUG-ENG-03: Rotational skew drive applied after torque integration."""
    model = Model()
    model.node_ids = np.array([10], dtype=np.int64)
    model._id2idx = {10: 0}
    model.mass = np.array([5.0])
    model.x0 = np.zeros((1, 3))
    model.x = np.zeros((1, 3))

    class MockSkew:
        def __init__(self):
            self.axes = {
                1: np.array([[0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
            }

    model.skews = MockSkew()

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[10.0, 10.0])
    model.functions[1] = fct

    class MockLoads:
        fix_tra = np.zeros((1, 3), dtype=bool)
        fix_rot = np.zeros((1, 3), dtype=bool)
        impvel = []
        skew_impvel = [(1, np.array([0]), 5, fct, 1.0, 1.0, 0.0, 10.0, None)]  # dof 5 = ZZ (axis 2)
        skew_impdisp = []

    rb_entity = RigidBody(id=1, kind="RBODY", master_id=10, grnod_id=1, master=0, slaves=np.zeros(0, dtype=np.int64),
                          mass_total=5.0, xg=np.zeros(3), J=np.eye(3)*2.0)
    rb_engine = RigidBodyEngine(rb_entity, model, MockLoads(), MessageLog())

    fint = np.zeros((1, 3))
    fext = np.zeros((1, 3))
    fcont = np.zeros((1, 3))
    mint = np.zeros((1, 3))
    # Apply a moment about X
    mint[0, 0] = 20.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    x = np.zeros((1, 3))
    dt = 0.01

    wext = rb_engine.advance(fint, fext, fcont, mint, v, vr, x, dt, t_next=0.01)

    # The torque on X (mint[0,0] = 20) should have produced spin on X: wx = T * dt / J = 20 * 0.01 / 2 = 0.1
    assert rb_engine.w[0] == pytest.approx(0.1, rel=1e-12)
    # The skew drive on axis 2 (+Z) should set wz = 10.0
    assert rb_engine.w[2] == pytest.approx(10.0, rel=1e-12)
    assert wext > 0.0


def test_bug_eng_06_logical_sensor_fire_time_tdelay():
    """BUG-ENG-06: Logical sensors set fire_time[sid] = self.crit_time[sid] + tdelay."""
    model = Model()
    model.sensors = [
        Sensor(id=1, kind="TIME", tdelay=0.1),
        Sensor(id=2, kind="NOT", sens_id1=1, tdelay=0.05),
    ]
    log = MessageLog()
    sensors = Sensors(model, log)

    # At t=0.0: sensor 1 not active -> NOT condition is True -> crit_time[2] set to 0.0
    sensors.update(0.0, log)
    # fire_time[2] should be 0.0 + 0.05 = 0.05
    sensors.update(0.06, log)
    assert sensors.active(2)
    assert sensors.fire_time[2] == pytest.approx(0.05, rel=1e-12)


def test_bug_eng_07_impvel_impdisp_sensor_gating():
    """BUG-ENG-07: impvel and impdisp store sens_id and gate on sensors."""
    model = Model()
    model.node_ids = np.array([1], dtype=np.int64)
    model._id2idx = {1: 0}
    model.mass = np.array([2.0])
    model.x0 = np.array([[1.0, 0.0, 0.0]])
    model.x = model.x0.copy()

    fct = FunctTable(fct_id=1, x=[0.0, 10.0], y=[5.0, 5.0])
    model.functions[1] = fct

    # Imposed velocity gated by sensor 10
    imp = ImposedVelocity(id=1, funct_id=1, dof=0, grnod_id=1, scale=1.0,
                          xscale=1.0, tstart=0.0, tstop=10.0, sens_id=10)
    model.impvel = [imp]
    # Set group
    model.node_groups[1] = NodeGroup(id=1, node_ids=[1], node_idx=np.array([0]))

    log = MessageLog()
    loads = LoadsAndConstraints(model, log)

    class MockSensors:
        def active(self, sens_id):
            return sens_id == 10 and self.is_active

        def shifted_time(self, sens_id, t):
            if not self.active(sens_id):
                return None
            return t - 0.5

        is_active = False

    sensors = MockSensors()
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    dt = 0.01

    # Before sensor fires
    loads.apply_kinematic(t=0.1, v=v, vr=vr, mass=model.mass, x=model.x, dt=dt, sensors=sensors)
    assert v[0, 0] == 0.0

    # After sensor fires
    sensors.is_active = True
    loads.apply_kinematic(t=0.6, v=v, vr=vr, mass=model.mass, x=model.x, dt=dt, sensors=sensors)
    assert v[0, 0] == pytest.approx(5.0, rel=1e-12)
