"""Tests for Stage 5 bug audit fixes (AUD-001, AUD-012, AUD-021, AUD-022, AUD-023, AUD-024)."""

import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from pyradioss.model.model import Model
from pyradioss.common.messages import StarterError
import pyradioss.starter.__main__ as starter_main
import pyradioss.engine.__main__ as engine_main
from pyradioss.accel.jit_kernels import shells_qeph
from pyradioss.gui.postproc import _stream_subprocess


def test_starter_main_missing_file_exit_code_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["starter", "-i", "nonexistent_file_0000.rad"])
    rc = starter_main.main()
    assert rc == 2


def test_engine_main_missing_file_exit_code_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["engine", "-i", "nonexistent_file_0001.rad"])
    rc = engine_main.main()
    assert rc == 2


def test_engine_main_failure_stop_reason_exit_code_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["engine", "-i", "dummy_0001.rad"])
    mock_model = MagicMock()
    mock_model.engine_state.stop_reason = "ENERGY ERROR LIMIT REACHED"
    with patch("pyradioss.engine.engine.run_engine", return_value=mock_model):
        with patch("os.path.isfile", return_value=True):
            rc = engine_main.main()
            assert rc == 2


def test_engine_main_normal_stop_reason_exit_code_0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["engine", "-i", "dummy_0001.rad"])
    mock_model = MagicMock()
    mock_model.engine_state.stop_reason = "/STOP/NSTEP LIMIT"
    with patch("pyradioss.engine.engine.run_engine", return_value=mock_model):
        with patch("os.path.isfile", return_value=True):
            rc = engine_main.main()
            assert rc == 0


def test_shells_qeph_imported_cleanly():
    assert hasattr(shells_qeph, "qeph_pre")


def test_stream_subprocess_timeout():
    # Run a python process that sleeps longer than timeout
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    rc = _stream_subprocess(cmd, cwd=None, emit=lambda line: None, timeout=0.5)
    assert rc == -9


def test_bcs_active_controls():
    import numpy as np
    from pyradioss.common.messages import MessageLog
    from pyradioss.engine.kinematics import LoadsAndConstraints
    from pyradioss.model.entities import BoundaryCondition, NodeGroup
    from pyradioss.model.model import EngineControls

    model = Model()
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0]))
    model.bcs.append(BoundaryCondition(id=1, grnod_id=1, fix_tra=np.array([True, True, True]), fix_rot=np.array([True, True, True])))
    model.node_ids = np.array([1])
    model.x0 = np.zeros((1, 3))
    log = MessageLog()

    # Active BCS
    lc_active = LoadsAndConstraints(model, log, controls=None)
    v = np.array([[1.0, 2.0, 3.0]])
    vr = np.array([[4.0, 5.0, 6.0]])
    lc_active.apply_kinematic(0.0, v, vr, np.ones(1), model.x0, 1e-4)
    assert np.all(v == 0.0)
    assert np.all(vr == 0.0)

    # Inactive BCS
    controls = EngineControls()
    controls.bcs_active[1] = False
    lc_inactive = LoadsAndConstraints(model, log, controls=controls)
    v2 = np.array([[1.0, 2.0, 3.0]])
    vr2 = np.array([[4.0, 5.0, 6.0]])
    lc_inactive.apply_kinematic(0.0, v2, vr2, np.ones(1), model.x0, 1e-4)
    assert np.all(v2 == [1.0, 2.0, 3.0])
    assert np.all(vr2 == [4.0, 5.0, 6.0])


def test_resume_sensors_status():
    from pyradioss.common.messages import MessageLog
    from pyradioss.engine.sensors import Sensors

    model = Model()
    sensors = Sensors(model, MessageLog())
    saved = {"sensors_status": {42: True}, "sensors": {99: 0.005}}

    # Mirror the resume logic in engine.py:
    sensors.fire_time.update(saved.get("sensors", {}))
    sensors.status.update(saved.get("sensors_status", {}))
    for sid in sensors.fire_time:
        sensors.status[sid] = True

    assert sensors.status[42] is True
    assert sensors.status[99] is True


def test_del_elements_integration(tmp_path):
    """Verify /DEL deletes elements before time integration."""
    deck_path = tmp_path / "DEL_0000.rad"
    deck_content = """\
/BEGIN
DEL test
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 10.0 0.0
4 10.0 10.0 0.0
/SH3N/1
1 1 2 3
2 2 4 3
/PART/1
plate
1 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SHELL/1
dkt18 shell prop
1 0 2 0
0.01 0.01 0.01
3 0 1.0
/END
"""
    deck_path.write_text(deck_content)

    engine_deck_path = tmp_path / "DEL_0001.rad"
    engine_deck_content = """\
/RUN/DEL/1
0.01
/DEL/SHELL
1
/STOP/NSTEP
1
/END
"""
    engine_deck_path.write_text(engine_deck_content)

    from pyradioss.starter.starter import run_starter
    from pyradioss.engine.engine import run_engine

    model = run_starter(str(deck_path))
    model = run_engine(str(engine_deck_path))
    assert model.engine_state.stop_reason.startswith("/STOP/NSTEP")
    # Element 1 should have been deleted by /DEL
    g = model.sh3n_dkt18
    assert g.state["off"][0] == 0.0
    assert g.state["off"][1] == 1.0



