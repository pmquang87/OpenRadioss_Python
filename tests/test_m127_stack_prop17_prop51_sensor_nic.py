"""Tests for Milestone M127: Composite Laminate Stacks (/STACK, /PROP/TYPE17 STACK,
/PROP/TYPE51) and /SENSOR/NIC_NIJ Aliasing.
"""
from __future__ import annotations

from pathlib import Path
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


def test_m127_stack_free_format(tmp_path):
    deck = """/BEGIN
Test STACK Free Format
/PLY/1
Ply 1
1 0.25
/PLY/2
Ply 2
1 0.25
/STACK/100
Stack Title
1 1 1 1 -0.5
0.01 0.01 0.01 0.0 0.0
1 0.833 1 1
1.0 0.0 0.0 1 1 1 1
1 0.0 -0.25 0.0 1.0
2 45.0 0.25 0.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 100 in model.stacks
    st = model.stacks[100]
    assert st.id == 100
    assert st.title == "Stack Title"
    assert st.ishell == 1
    assert st.z0 == -0.5
    assert st.ashear == pytest.approx(0.833)
    assert st.vx == 1.0
    assert len(st.plies) == 2
    assert st.plies[0].ply_id == 1
    assert st.plies[0].phi == 0.0
    assert st.plies[0].zi == -0.25
    assert st.plies[1].ply_id == 2
    assert st.plies[1].phi == 45.0
    assert st.plies[1].zi == 0.25


def test_m127_stack_fixed_format(tmp_path):
    deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test Fixed Format
      2022         0
/STACK/101
Fixed Stack
         1         1         1         1                    -0.5
                0.01                0.01                0.01                 0.0                 0.0
                   1               0.833                   1                   1
                 1.0                 0.0                 0.0         1         1         1         1
         1                45.0               -0.25                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 101 in model.stacks
    st = model.stacks[101]
    assert st.id == 101
    assert st.z0 == pytest.approx(-0.5)
    assert len(st.plies) == 1
    assert st.plies[0].ply_id == 1
    assert st.plies[0].phi == pytest.approx(45.0)
    assert st.plies[0].zi == pytest.approx(-0.25)


def test_m127_prop17_stack(tmp_path):
    deck = """/BEGIN
Test PROP TYPE17
/PROP/TYPE17/10
Prop Type 17
1 1 1 1 -0.5
0.02 0.02 0.02 0.0 0.0
1 0.833 1 1
1.0 0.0 0.0 1 1 1 1
/PROP/STACK/11
Prop Stack Alias
1 1 1 1 -0.5
0.02 0.02 0.02 0.0 0.0
1 0.833 1 1
1.0 0.0 0.0 1 1 1 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.properties
    p17 = model.properties[10]
    assert p17.type == 17
    assert p17.params["ishell"] == 1
    assert p17.params["z0"] == -0.5
    assert p17.params["hm"] == pytest.approx(0.02)

    assert 11 in model.properties
    p_stack = model.properties[11]
    assert p_stack.type == 17


def test_m127_prop51(tmp_path):
    deck = """/BEGIN
Test PROP TYPE51
/PROP/TYPE51/20
Prop Type 51
1 1 1 1 -0.5
0.02 0.02 0.02 0.0 0.0
1 0.833 1
1.0 0.0 0.0 1 1 1 0.1 2.0 1
/PROP/P51/21
Prop P51 Alias
1 1 1 1 -0.5
0.02 0.02 0.02 0.0 0.0
1 0.833 1
1.0 0.0 0.0 1 1 1 0.1 2.0 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 20 in model.properties
    p51 = model.properties[20]
    assert p51.type == 51
    assert p51.params["ishell"] == 1
    assert p51.params["z0"] == -0.5
    assert p51.params["p_thick_fail"] == pytest.approx(0.1)
    assert p51.params["fexp"] == pytest.approx(2.0)

    assert 21 in model.properties
    p_p51 = model.properties[21]
    assert p_p51.type == 51


def test_m127_sensor_nic_nij(tmp_path):
    deck = """/BEGIN
Test SENSOR NIC_NIJ
/SENSOR/NIC_NIJ/1
Sensor NIC NIJ
0.05
1.0 500.0 600.0 200.0 300.0
10 1 X Y
0.01 2.0 600.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.sensors) == 1
    s = model.sensors[0]
    assert s.id == 1
    assert s.kind == "NIC"
    assert s.tdelay == pytest.approx(0.05)
    assert s.spring_id == 10
    assert s.nij_max == pytest.approx(1.0)
    assert s.fint_tens == pytest.approx(500.0)
    assert s.fint_comp == pytest.approx(600.0)
    assert s.mint_flex == pytest.approx(200.0)
    assert s.mint_ext == pytest.approx(300.0)
    assert s.ax_dir == "X"
    assert s.bend_dir == "Y"
