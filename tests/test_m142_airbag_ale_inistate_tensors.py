"""Tests for Milestone M142: Airbag Injector & Venthole Models, Advanced ALE Solver Controls & Extended Element Initial State Tensors Suite
(/AIRBAG/INJECTOR, /INJECTOR, /AIRBAG/VENTHOLE, /VENTHOLE, /INIBRI/STRA_F, /INIBRI/FAIL, /INIBRI/AUX, /INISHE/STRA_F, /INISHE/EPSP_F, /INISHE/FAIL, /INISHE/AUX, /INISH3/STRA_F, /INISH3/EPSP_F).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m142_airbag_injector_and_venthole(tmp_path):
    deck = """/BEGIN
Test Airbag Injector and Venthole
/AIRBAG/INJECTOR/10
Primary Jet Injector
101 1 1 2 3
10 20 30 1.2 1.3 1.4
/VENTHOLE/20
Safety Burst Venthole
501 1 0.05 0.02
0.0 0.2 150.0 0.005 1
100 200 300 1.0 1.0 1.0
/PROP/INJECTOR/30
Secondary Jet Injector
102 0
/PROP/VENTHOLE/40
Secondary Porous Vent
502 2 0.08 0.03
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # Injector 10
    assert 10 in model.airbag_injectors
    inj = model.airbag_injectors[10]
    assert inj.sensor_id == 101
    assert inj.ijet == 1
    assert inj.node1 == 1
    assert inj.node2 == 2
    assert inj.node3 == 3
    assert inj.fct_pt == 10
    assert inj.fct_theta == 20
    assert inj.fct_delta == 30
    assert inj.fscale_pt == pytest.approx(1.2)
    assert inj.fscale_ptheta == pytest.approx(1.3)
    assert inj.fscale_pdelta == pytest.approx(1.4)

    # Venthole 20
    assert 20 in model.airbag_ventholes
    vh = model.airbag_ventholes[20]
    assert vh.surf_vent == 501
    assert vh.iform == 1
    assert vh.avent == pytest.approx(0.05)
    assert vh.bvent == pytest.approx(0.02)
    assert vh.tstart == pytest.approx(0.0)
    assert vh.tstop == pytest.approx(0.2)
    assert vh.dpdef == pytest.approx(150.0)
    assert vh.dtpdef == pytest.approx(0.005)
    assert vh.idtpdef == 1
    assert vh.fct_id_t == 100
    assert vh.fct_id_p == 200
    assert vh.fct_id_a == 300

    # Injector 30 & Venthole 40 via /PROP
    assert 30 in model.airbag_injectors
    assert 40 in model.airbag_ventholes
    assert model.airbag_ventholes[40].surf_vent == 502
    assert model.airbag_ventholes[40].iform == 2


def test_m142_solid_initial_state_tensors_and_modifiers(tmp_path):
    deck = """/BEGIN
Test Solid Initial State Tensors
/INIBRI/STRA_F
101 0.01 0.02 -0.005 0.003 0.001 -0.002
/INIBRI/FAIL
101 1.0
102 0.5
/INIBRI/AUX
101 42.0
/INIBRI/SCALE_YLD
101 1.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 101 in model.ini_bricks
    b101 = model.ini_bricks[101]
    assert np.allclose(b101.eps, [0.01, 0.02, -0.005, 0.003, 0.001, -0.002])
    assert b101.fail_flag == pytest.approx(1.0)
    assert b101.aux == pytest.approx(42.0)
    assert b101.scale_yld == pytest.approx(1.25)

    assert 102 in model.ini_bricks
    assert model.ini_bricks[102].fail_flag == pytest.approx(0.5)


def test_m142_shell_initial_state_tensors_layers_modifiers(tmp_path):
    deck = """/BEGIN
Test Shell Initial State Tensors & Layers
/INISHE/STRA_F
201 0.005 0.008 0.002
/INISHE/EPSP_F
201 0.01 0.02 0.03 0.04 0.05
/INISHE/FAIL
201 0.8
/INISHE/AUX
201 99.5
/INISH3/STRA_F
301 0.003 0.004 0.001
/INISH3/EPSP_F
301 0.015 0.025 0.035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # 4-node shell 201
    assert 201 in model.ini_shells
    sh201 = model.ini_shells[201]
    assert np.allclose(sh201.eps, [0.005, 0.008, 0.0, 0.002, 0.0, 0.0])
    assert np.allclose(sh201.epsp_layers, [0.01, 0.02, 0.03, 0.04, 0.05])
    assert sh201.fail_flag == pytest.approx(0.8)
    assert sh201.aux == pytest.approx(99.5)

    # 3-node shell 301
    assert 301 in model.ini_shells
    sh301 = model.ini_shells[301]
    assert np.allclose(sh301.eps, [0.003, 0.004, 0.0, 0.001, 0.0, 0.0])
    assert np.allclose(sh301.epsp_layers, [0.015, 0.025, 0.035])
