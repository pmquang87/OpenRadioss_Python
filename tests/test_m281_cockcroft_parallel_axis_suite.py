"""Tests for Milestone M281: Cockcroft Failure Model, EngXfemEnergy, ParallelAxisJoint, and SensorSpringShearWork."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m281_fail_cockcroft_fixed(tmp_path: Path):
    c1 = f"{450.0:>20.4f}{1.25:>20.4f}{1:>10d}"
    c2 = f"{870:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cockcroft Failure Model Fixed Format Test
2022 0
/FAIL/COCKCROFT/870
Cockcroft Latham Ductile Failure Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 870 in model.fail_cockcrofts
    fc = model.fail_cockcrofts[870]
    assert pytest.approx(fc.c0) == 450.0
    assert pytest.approx(fc.alpha) == 1.25
    assert fc.failip == 1
    assert fc.fail_id == 870
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "COCKCROFT"


def test_m281_fail_cockcroft_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cockcroft Free Format Test
/FAIL/COCKCROFT/871
550.0, 1.5, 2
871
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 871 in model.fail_cockcrofts
    fc = model.fail_cockcrofts[871]
    assert pytest.approx(fc.c0) == 550.0
    assert pytest.approx(fc.alpha) == 1.5
    assert fc.failip == 2
    assert fc.fail_id == 871


def test_m281_fail_cockcroft_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cockcroft Aliases Test
/FAIL/COCKCROFT/872
300.0, 1.0, 0
/FAIL/COCKCROFT_MODEL/873
300.0, 1.0, 0
/FAIL/COCKCROFT_LAW/874
300.0, 1.0, 0
/FAIL/COCKCROFT_DAMAGE/875
300.0, 1.0, 0
/FAIL/CL_DUCTILE/876
300.0, 1.0, 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 872 in model.fail_cockcrofts
    assert 873 in model.fail_cockcrofts
    assert 874 in model.fail_cockcrofts
    assert 875 in model.fail_cockcrofts
    assert 876 in model.fail_cockcrofts
    assert len(model.raw_fails) == 5


def test_m281_eng_xfem_energy(tmp_path: Path):
    c1 = f"{0.00015:>20.6f}{6:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
XFEM Energy Fixed and Free Format Test
2022 0
/ENG/XFEM_ENERGY/1
Fixed XFEM Energy
{c1}
/ENG/XFEM_ENERGY/2
Free XFEM Energy
0.0004, 12
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_xfem_energies
    assert 2 in model.eng_xfem_energies
    xe1 = model.eng_xfem_energies[1]
    assert pytest.approx(xe1.dt_xfem) == 0.00015
    assert xe1.sens_id == 6
    xe2 = model.eng_xfem_energies[2]
    assert pytest.approx(xe2.dt_xfem) == 0.0004
    assert xe2.sens_id == 12


def test_m281_eng_xfem_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
XFEM Energy Aliases Test
/ENG/XFEM_WORK/11
0.001, 10
/ENG/EXFEM/12
0.002, 11
/ENG/XFEM_ENER/13
0.003, 12
/ENG/XFEM_INTERNAL_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_xfem_energies
    assert 12 in model.eng_xfem_energies
    assert 13 in model.eng_xfem_energies
    assert 14 in model.eng_xfem_energies
    assert pytest.approx(model.eng_xfem_energies[11].dt_xfem) == 0.001
    assert pytest.approx(model.eng_xfem_energies[12].dt_xfem) == 0.002
    assert pytest.approx(model.eng_xfem_energies[13].dt_xfem) == 0.003
    assert pytest.approx(model.eng_xfem_energies[14].dt_xfem) == 0.004


def test_m281_parallel_axis_joint(tmp_path: Path):
    c1 = f"{121:>10d}{122:>10d}{123:>10d}{4.0e6:>20.4f}{8:>10d}{2.5e-6:>20.6e}"
    c2 = f"{0.0:>20.4f}{0.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Parallel Axis Joint Fixed and Free Format Test
2022 0
/LAGMUL/PARALLEL_AXIS_JOINT/1
Fixed Parallel Axis Joint
{c1}
{c2}
/LAGMUL/PARALLEL_AXIS_JOINT/2
Free Parallel Axis Joint
221, 222, 223, 3.5e6, 9, 4.0e-6
1.0, 0.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_parallel_axis_joints
    assert 2 in model.lagmul_parallel_axis_joints
    paj1 = model.lagmul_parallel_axis_joints[1]
    assert paj1.node1 == 121
    assert paj1.node2 == 122
    assert paj1.node3 == 123
    assert pytest.approx(paj1.stiff) == 4.0e6
    assert paj1.skew_id == 8
    assert pytest.approx(paj1.tol) == 2.5e-6
    assert pytest.approx(paj1.axis_x) == 0.0
    assert pytest.approx(paj1.axis_y) == 0.0
    assert pytest.approx(paj1.axis_z) == 1.0

    paj2 = model.lagmul_parallel_axis_joints[2]
    assert paj2.node1 == 221
    assert paj2.node2 == 222
    assert paj2.node3 == 223
    assert pytest.approx(paj2.stiff) == 3.5e6
    assert paj2.skew_id == 9
    assert pytest.approx(paj2.tol) == 4.0e-6
    assert pytest.approx(paj2.axis_x) == 1.0
    assert pytest.approx(paj2.axis_y) == 0.0
    assert pytest.approx(paj2.axis_z) == 0.0


def test_m281_parallel_axis_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Parallel Axis Joint Aliases Test
/PARALLEL_AXIS_JOINT/51
501, 502, 503, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/PARALLEL_AXIS_SLIDER/52
504, 505, 506, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/LAGMUL_PARALLEL_AXIS_SLIDER/53
507, 508, 509, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/PARALLEL_AXIS_MECHANISM/54
510, 511, 512, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.lagmul_parallel_axis_joints
    assert 52 in model.lagmul_parallel_axis_joints
    assert 53 in model.lagmul_parallel_axis_joints
    assert 54 in model.lagmul_parallel_axis_joints
    assert model.lagmul_parallel_axis_joints[51].node1 == 501
    assert model.lagmul_parallel_axis_joints[52].node1 == 504
    assert model.lagmul_parallel_axis_joints[53].node1 == 507
    assert model.lagmul_parallel_axis_joints[54].node1 == 510


def test_m281_sensor_spring_shear_work(tmp_path: Path):
    c1 = f"{751:>10d}{5200.0:>20.4f}{0.0025:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Shear Work Fixed and Free Format Test
2022 0
/SENSOR/SPRING_SHEAR_WORK/1
Fixed Spring Shear Work Sensor
{c1}
/SENSOR/SPRING_SHEAR_WORK/2
Free Spring Shear Work Sensor
752, 7900.0, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_shear_works
    assert 2 in model.sensor_spring_shear_works
    s1 = model.sensor_spring_shear_works[1]
    assert s1.spring_id == 751
    assert pytest.approx(s1.w_shear_max) == 5200.0
    assert pytest.approx(s1.t_delay) == 0.0025

    s2 = model.sensor_spring_shear_works[2]
    assert s2.spring_id == 752
    assert pytest.approx(s2.w_shear_max) == 7900.0
    assert pytest.approx(s2.t_delay) == 0.0045

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_SHEAR_WORK"
    assert model.sensors[1].kind == "SPRING_SHEAR_WORK"


def test_m281_sensor_spring_shear_work_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Shear Work Aliases Test
/SENSOR/SPRING_SHR_WORK/11
851, 8200.0, 0.001
/SENSOR/SPRING_WORK_SHEAR/12
852, 9300.0, 0.002
/SENSOR/SHEAR_WORK_SPRING/13
853, 10400.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_shear_works
    assert 12 in model.sensor_spring_shear_works
    assert 13 in model.sensor_spring_shear_works
    assert model.sensor_spring_shear_works[11].spring_id == 851
    assert pytest.approx(model.sensor_spring_shear_works[11].w_shear_max) == 8200.0
    assert model.sensor_spring_shear_works[12].spring_id == 852
    assert model.sensor_spring_shear_works[13].spring_id == 853
