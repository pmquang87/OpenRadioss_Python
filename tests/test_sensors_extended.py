"""Unit tests for extended sensor suite (/SENSOR/ACC, /SENSOR/ENERGY, /SENSOR/RWALL, /SENSOR/SECT, /SENSOR/TEMP)

Fortran references:
- starter/source/tools/sensor/read_sensor_acc.F & engine/source/tools/sensor/sensor_acc.F
- starter/source/tools/sensor/read_sensor_energy.F & engine/source/tools/sensor/sensor_energy.F
- starter/source/tools/sensor/read_sensor_rwall.F & engine/source/tools/sensor/sensor_rwall.F
- starter/source/tools/sensor/read_sensor_sect.F & engine/source/tools/sensor/sensor_section.F
- starter/source/tools/sensor/read_sensor_temp.F & engine/source/tools/sensor/sensor_temp.F
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.sensors import Sensors
from pyradioss.model.entities import Sensor
from pyradioss.model.model import Model


def _make_base_model() -> Model:
    """Construct minimal test model with 3 nodes."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=int)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    model.x = model.x0.copy()
    model.v = np.zeros((3, 3), dtype=float)
    model.a = np.zeros((3, 3), dtype=float)
    model.mass = np.array([1.0, 2.0, 1.0], dtype=float)
    return model


def test_sensor_acc_triggering():
    """Verify /SENSOR/ACC triggering on acceleration threshold and latched semantics."""
    model = _make_base_model()
    log = MessageLog()

    # Sensor 1: Triggers when node 2 resultant acceleration >= 50.0
    # Sensor 2: Triggers when node 1 X-acceleration >= 30.0, with tdelay=0.001
    model.sensors = [
        Sensor(id=1, kind="ACC", node_id=2, a_max=50.0),
        Sensor(id=2, kind="ACC", node_id=1, a_max=30.0, dir="X", tdelay=0.001),
    ]

    sensors = Sensors(model, log)
    assert len(sensors) == 2
    assert not sensors.active(1)
    assert not sensors.active(2)

    # Step 1: Low acceleration at t=0.0005
    model.a[1] = [10.0, 20.0, 0.0]  # norm ~ 22.36 < 50
    model.a[0] = [15.0, 0.0, 0.0]   # |ax| = 15 < 30
    sensors.update(0.0005, log)
    assert not sensors.active(1)
    assert not sensors.active(2)

    # Step 2: Node 2 acceleration exceeds 50.0 at t=0.001
    model.a[1] = [40.0, 35.0, 0.0]  # norm = sqrt(1600+1225) = sqrt(2825) ~ 53.15 > 50
    model.a[0] = [35.0, 0.0, 0.0]   # |ax| = 35 >= 30 (has tdelay 0.001)
    sensors.update(0.001, log)
    assert sensors.active(1)
    assert sensors.fire_time[1] == pytest.approx(0.001)
    # Sensor 2 entered critical state at t=0.001, but has tdelay 0.001 -> fires at 0.002
    assert not sensors.active(2)

    # Step 3: Acceleration drops to zero at t=0.0015 (test latching for sensor 1)
    model.a[1] = [0.0, 0.0, 0.0]
    sensors.update(0.0015, log)
    assert sensors.active(1)  # Latched!
    assert sensors.fire_time[1] == pytest.approx(0.001)
    assert not sensors.active(2)

    # Step 4: Time reaches 0.002, sensor 2 fires
    sensors.update(0.002, log)
    assert sensors.active(2)
    assert sensors.fire_time[2] == pytest.approx(0.002)


def test_sensor_energy_triggering():
    """Verify /SENSOR/ENERGY triggering on kinetic/internal/total energy threshold."""
    model = _make_base_model()
    log = MessageLog()

    # Sensor 10: Kinetic energy >= 100.0
    # Sensor 11: Internal energy >= 250.0
    # Sensor 12: Total energy >= 300.0
    model.sensors = [
        Sensor(id=10, kind="ENERGY", energy_type="KE", e_max=100.0),
        Sensor(id=11, kind="ENERGY", energy_type="IE", e_max=250.0),
        Sensor(id=12, kind="ENERGY", energy_type="TOT", e_max=300.0),
    ]

    sensors = Sensors(model, log)
    assert len(sensors) == 3

    # Step 1: Low velocities at t=0.0
    # KE = 0.5 * (1*4^2 + 2*5^2 + 1*0) = 0.5 * (16 + 50) = 33.0 < 100.0
    model.v[0] = [4.0, 0.0, 0.0]
    model.v[1] = [5.0, 0.0, 0.0]
    model.energies = {"IE": 50.0, "KE": 33.0, "TOT": 83.0}
    sensors.update(0.0, log)
    assert not sensors.active(10)
    assert not sensors.active(11)
    assert not sensors.active(12)

    # Step 2: Increase KE past threshold at t=0.002
    # Set model.energies: KE = 120.0 > 100.0
    model.energies = {"IE": 50.0, "KE": 120.0, "TOT": 170.0}
    sensors.update(0.002, log)
    assert sensors.active(10)
    assert sensors.fire_time[10] == pytest.approx(0.002)
    assert not sensors.active(11)
    assert not sensors.active(12)

    # Step 3: Increase IE past threshold at t=0.004
    model.energies = {"IE": 260.0, "KE": 80.0, "TOT": 340.0}
    sensors.update(0.004, log)
    assert sensors.active(10)  # Latched
    assert sensors.active(11)  # IE triggered
    assert sensors.active(12)  # TOT triggered (340 >= 300)
    assert sensors.fire_time[11] == pytest.approx(0.004)
    assert sensors.fire_time[12] == pytest.approx(0.004)

    # Step 4: Energy drops to zero, verify latching
    model.energies = {"IE": 0.0, "KE": 0.0, "TOT": 0.0}
    sensors.update(0.006, log)
    assert sensors.active(10)
    assert sensors.active(11)
    assert sensors.active(12)


def test_sensor_rwall_and_sect_force_triggering():
    """Verify /SENSOR/RWALL and /SENSOR/SECT force threshold triggering."""
    model = _make_base_model()
    log = MessageLog()

    # Sensor 20: RWALL id=1 resultant force >= 500.0
    # Sensor 21: RWALL id=1 normal force (FN) >= 400.0
    # Sensor 30: SECT id=5 resultant force >= 1000.0
    model.sensors = [
        Sensor(id=20, kind="RWALL", rwall_id=1, f_max=500.0, dir="TF"),
        Sensor(id=21, kind="RWALL", rwall_id=1, f_max=400.0, dir="FN"),
        Sensor(id=30, kind="SECT", sect_id=5, f_max=1000.0, dir="TF"),
    ]

    sensors = Sensors(model, log)

    # Initial state: low forces
    model.rwall_forces = {1: np.array([300.0, 200.0, 0.0])}  # FN=300 < 400, TF=sqrt(130000)~360.5 < 500
    model.section_forces = {5: (np.array([400.0, 300.0, 0.0]), np.zeros(3))}  # TF=500 < 1000
    sensors.update(0.001, log)
    assert not sensors.active(20)
    assert not sensors.active(21)
    assert not sensors.active(30)

    # At t=0.003: Wall force FN exceeds 400, TF exceeds 500
    model.rwall_forces = {1: np.array([450.0, 300.0, 0.0])}  # FN=450 >= 400, TF=sqrt(292500)~540.8 >= 500
    sensors.update(0.003, log)
    assert sensors.active(20)
    assert sensors.active(21)
    assert sensors.fire_time[20] == pytest.approx(0.003)
    assert sensors.fire_time[21] == pytest.approx(0.003)
    assert not sensors.active(30)

    # At t=0.005: Section force exceeds 1000
    model.section_forces = {5: (np.array([800.0, 700.0, 0.0]), np.zeros(3))}  # TF=sqrt(1130000)~1063.0 >= 1000
    sensors.update(0.005, log)
    assert sensors.active(30)
    assert sensors.fire_time[30] == pytest.approx(0.005)

    # Wall contact lost (forces drop to zero), verify latching
    model.rwall_forces = {1: np.zeros(3)}
    model.section_forces = {5: (np.zeros(3), np.zeros(3))}
    sensors.update(0.007, log)
    assert sensors.active(20)
    assert sensors.active(21)
    assert sensors.active(30)


def test_sensor_temp_triggering():
    """Verify /SENSOR/TEMP triggering on temperature threshold."""
    model = _make_base_model()
    log = MessageLog()

    # Sensor 40: Node 2 temperature >= 400.0
    model.sensors = [
        Sensor(id=40, kind="TEMP", node_id=2, t_max=400.0),
    ]

    sensors = Sensors(model, log)

    # Step 1: Initial ambient temperature
    model.temperatures = {1: 293.15, 2: 320.0, 3: 293.15}
    sensors.update(0.001, log)
    assert not sensors.active(40)

    # Step 2: Temperature heats up to 425.0 >= 400.0 at t=0.003
    model.temperatures[2] = 425.0
    sensors.update(0.003, log)
    assert sensors.active(40)
    assert sensors.fire_time[40] == pytest.approx(0.003)

    # Step 3: Temperature cools back down to 300.0, verify latching
    model.temperatures[2] = 300.0
    sensors.update(0.005, log)
    assert sensors.active(40)
    assert sensors.fire_time[40] == pytest.approx(0.003)


def test_sensor_logical_combinations_with_extended_sensors():
    """Verify logical sensors (AND, OR, NOT) interacting with new physical sensors."""
    model = _make_base_model()
    log = MessageLog()

    # Physical sensors:
    # Sensor 1: ACC on node 1 >= 50.0
    # Sensor 2: RWALL id=1 force >= 200.0
    # Sensor 3: TEMP on node 3 >= 350.0
    # Logical sensors:
    # Sensor 100: AND(1, 2)  - fires when both ACC and RWALL are active
    # Sensor 200: OR(1, 3)   - fires when either ACC or TEMP is active
    # Sensor 300: NOT(1)     - active while ACC is inactive, deactivates when ACC fires
    model.sensors = [
        Sensor(id=1, kind="ACC", node_id=1, a_max=50.0),
        Sensor(id=2, kind="RWALL", rwall_id=1, f_max=200.0),
        Sensor(id=3, kind="TEMP", node_id=3, t_max=350.0),
        Sensor(id=100, kind="AND", sens_id1=1, sens_id2=2),
        Sensor(id=200, kind="OR", sens_id1=1, sens_id2=3),
        Sensor(id=300, kind="NOT", sens_id1=1),
    ]

    sensors = Sensors(model, log)

    # Initial state at t=0: all physical sensors inactive
    model.a[0] = [10.0, 0.0, 0.0]
    model.rwall_forces = {1: np.array([50.0, 0.0, 0.0])}
    model.temperatures = {3: 300.0}
    sensors.update(0.0, log)

    assert not sensors.active(1)
    assert not sensors.active(2)
    assert not sensors.active(3)
    assert not sensors.active(100)  # AND(1,2) is False
    assert not sensors.active(200)  # OR(1,3) is False
    assert sensors.active(300)      # NOT(1) is True!

    # Step 2: At t=0.002, RWALL force fires (Sensor 2)
    model.rwall_forces = {1: np.array([250.0, 0.0, 0.0])}
    sensors.update(0.002, log)
    assert not sensors.active(1)
    assert sensors.active(2)
    assert not sensors.active(100)  # AND is still False because 1 is inactive
    assert not sensors.active(200)  # OR is still False
    assert sensors.active(300)      # NOT(1) remains True

    # Step 3: At t=0.004, ACC fires (Sensor 1)
    model.a[0] = [60.0, 0.0, 0.0]
    sensors.update(0.004, log)
    assert sensors.active(1)
    assert sensors.active(2)
    assert sensors.active(100)      # AND(1,2) is now True!
    assert sensors.active(200)      # OR(1,3) is now True!
    assert not sensors.active(300)  # NOT(1) dynamically becomes False!

    # Step 4: Verify shifted_time for gated evaluation
    # Gated by sensor 100: at t=0.006, shifted time is 0.006 - 0.004 = 0.002
    assert sensors.shifted_time(100, 0.006) == pytest.approx(0.002)
    # Sensor 3 has not fired, so shifted_time is None
    assert sensors.shifted_time(3, 0.006) is None
    # Sensor 0 (ungated) returns current time
    assert sensors.shifted_time(0, 0.006) == pytest.approx(0.006)
