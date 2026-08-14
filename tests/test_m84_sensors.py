"""Tests for Milestone M84: Logical and Velocity Sensors (/SENSOR/VEL, /SENSOR/NOT, /SENSOR/AND, /SENSOR/OR)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.sensors import Sensors
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Sensor
from pyradioss.model.model import Model


def test_starter_read_sensors():
    deck = """/BEGIN
TEST SENSORS
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/SENSOR/TIME/1
Time sensor
0.05
/SENSOR/DISP/2
Disp sensor
0.01
1 0.2
/SENSOR/VEL/3
Vel sensor
0.02
2 5.0 100.0
/SENSOR/NOT/4
Not sensor 1
0.0
1
/SENSOR/AND/5
And sensor 1 and 2
0.0
1 2
/SENSOR/OR/6
Or sensor 1 or 3
0.0
1 3
/END
"""
    with open("m84_test_sensors.rad", "w") as f:
        f.write(deck)

    log = MessageLog()
    model = Model()
    blocks = read_deck("m84_test_sensors.rad")
    parse_starter_deck(blocks, model, log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    assert len(model.sensors) == 6

    # Verify sensor 3 (VEL)
    s3 = next(s for s in model.sensors if s.id == 3)
    assert s3.kind == "VEL"
    assert s3.node_id == 2
    assert s3.vmax == 5.0
    assert s3.fcut == 100.0
    assert s3.tdelay == 0.02

    # Verify sensor 4 (NOT)
    s4 = next(s for s in model.sensors if s.id == 4)
    assert s4.kind == "NOT"
    assert s4.sens_id1 == 1

    # Verify sensor 5 (AND)
    s5 = next(s for s in model.sensors if s.id == 5)
    assert s5.kind == "AND"
    assert s5.sens_id1 == 1
    assert s5.sens_id2 == 2

    # Verify sensor 6 (OR)
    s6 = next(s for s in model.sensors if s.id == 6)
    assert s6.kind == "OR"
    assert s6.sens_id1 == 1
    assert s6.sens_id2 == 3


def test_engine_sensors_eval():
    model = Model()
    model.node_ids = np.array([1, 2], dtype=int)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    model.v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    model.sensors = [
        Sensor(id=1, kind="TIME", tdelay=0.1),
        Sensor(id=2, kind="VEL", tdelay=0.0, node_id=2, vmax=10.0),
        Sensor(id=3, kind="NOT", tdelay=0.0, sens_id1=1),
        Sensor(id=4, kind="AND", tdelay=0.0, sens_id1=1, sens_id2=2),
        Sensor(id=5, kind="OR", tdelay=0.0, sens_id1=1, sens_id2=2),
    ]

    log = MessageLog()
    sensors = Sensors(model, log)

    # t = 0.0: TIME(1) has not fired -> NOT(3) is active, AND(4) inactive, OR(5) inactive, VEL(2) inactive
    sensors.update(0.0, log)
    assert not sensors.active(1)
    assert not sensors.active(2)
    assert sensors.active(3)
    assert not sensors.active(4)
    assert not sensors.active(5)

    # t = 0.05: node 2 velocity reaches 12.0 -> VEL(2) fires
    model.v[1] = [12.0, 0.0, 0.0]
    sensors.update(0.05, log)
    assert not sensors.active(1)
    assert sensors.active(2)
    assert sensors.active(3)
    assert not sensors.active(4)
    assert sensors.active(5)  # OR(5) active because VEL(2) active

    # t = 0.12: TIME(1) fires -> NOT(3) goes inactive, AND(4) goes active
    sensors.update(0.12, log)
    assert sensors.active(1)
    assert sensors.active(2)
    assert not sensors.active(3)  # NOT(3) inactive because TIME(1) is active
    assert sensors.active(4)      # AND(4) active because both TIME(1) and VEL(2) active
    assert sensors.active(5)      # OR(5) active

    # Verify shifted time
    assert sensors.shifted_time(0, 0.12) == 0.12
    assert sensors.shifted_time(1, 0.12) == pytest.approx(0.12 - 0.1)
    assert sensors.shifted_time(3, 0.12) is None


def test_cascaded_logical_sensors():
    model = Model()
    model.sensors = [
        Sensor(id=1, kind="TIME", tdelay=0.1),
        Sensor(id=2, kind="TIME", tdelay=0.2),
        Sensor(id=3, kind="AND", sens_id1=1, sens_id2=2),
        Sensor(id=4, kind="NOT", sens_id1=3),
        Sensor(id=5, kind="OR", sens_id1=4, sens_id2=1),
    ]
    log = MessageLog()
    sensors = Sensors(model, log)

    # At t = 0.05: S1=F, S2=F -> S3=F -> S4=T -> S5=T
    sensors.update(0.05, log)
    assert not sensors.active(1)
    assert not sensors.active(2)
    assert not sensors.active(3)
    assert sensors.active(4)
    assert sensors.active(5)

    # At t = 0.15: S1=T, S2=F -> S3=F -> S4=T -> S5=T
    sensors.update(0.15, log)
    assert sensors.active(1)
    assert not sensors.active(2)
    assert not sensors.active(3)
    assert sensors.active(4)
    assert sensors.active(5)

    # At t = 0.25: S1=T, S2=T -> S3=T -> S4=F -> S5=T (because S1=T)
    sensors.update(0.25, log)
    assert sensors.active(1)
    assert sensors.active(2)
    assert sensors.active(3)
    assert not sensors.active(4)
    assert sensors.active(5)

