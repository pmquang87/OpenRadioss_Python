"""Tests for Milestone M278: Tab1 Failure Model, EngSphEnergy, ClevisJoint, and SensorSpringTotalWork."""

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


def test_m278_fail_tab1_fixed(tmp_path: Path):
    c1 = f"{1:>10d}{1:>10d}{0.05:>20.4f}{0.02:>20.4f}{0:>10d}"
    c2 = f"{0.8:>20.4f}{0.1:>20.4f}{1.5:>20.4f}{0.05:>20.4f}{101:>10d}"
    c3 = f"{10:>10d}{1.0:>20.4f}{1.0:>20.4f}{20:>10d}{1.0:>20.4f}{1.0:>20.4f}"
    c4 = f"{30:>10d}{1.2:>20.4f}{5.0:>20.4f}{0.01:>20.4f}{1.5:>20.4f}{0.0:>20.4f}"
    c5 = f"{40:>10d}{1.1:>20.4f}{2:>10d}{0.4:>20.4f}{1.0:>20.4f}"
    c6 = f"{840:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tab1 Tabulated Failure Model Fixed Format Test
2022 0
/FAIL/TAB1/840
Tab1 Failure Model
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 840 in model.fail_tab1s
    ft = model.fail_tab1s[840]
    assert ft.ifail_sh == 1
    assert ft.ifail_so == 1
    assert pytest.approx(ft.p_thickfail) == 0.05
    assert pytest.approx(ft.p_thinfail) == 0.02
    assert ft.ixfem == 0
    assert pytest.approx(ft.dcrit) == 0.8
    assert pytest.approx(ft.d) == 0.1
    assert pytest.approx(ft.n) == 1.5
    assert pytest.approx(ft.dadv) == 0.05
    assert ft.fct_idd == 101
    assert ft.table1_id == 10
    assert pytest.approx(ft.xscale1) == 1.0
    assert ft.table2_id == 20
    assert ft.fct_id_el == 30
    assert pytest.approx(ft.fscale_el) == 1.2
    assert pytest.approx(ft.el_ref) == 5.0
    assert pytest.approx(ft.inst_start) == 0.01
    assert pytest.approx(ft.fad_exp) == 1.5
    assert ft.fct_id_t == 40
    assert pytest.approx(ft.fscale_t) == 1.1
    assert ft.ifunc == 2
    assert pytest.approx(ft.eps_max) == 0.4
    assert pytest.approx(ft.scale) == 1.0
    assert ft.fail_id == 840
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "TAB1"


def test_m278_fail_tab1_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Tab1 Free Format Test
/FAIL/TAB1/841
1, 1, 0.04, 0.01, 0
0.75, 0.05, 1.2, 0.02, 102
15, 1.0, 1.0, 25, 1.0, 1.0
35, 1.0, 4.0, 0.02, 1.0, 0.0
45, 1.0, 1, 0.35, 1.0
841
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 841 in model.fail_tab1s
    ft = model.fail_tab1s[841]
    assert ft.ifail_sh == 1
    assert ft.ifail_so == 1
    assert pytest.approx(ft.p_thickfail) == 0.04
    assert pytest.approx(ft.dcrit) == 0.75
    assert ft.table1_id == 15
    assert ft.table2_id == 25
    assert ft.fct_id_el == 35
    assert pytest.approx(ft.el_ref) == 4.0
    assert ft.fct_id_t == 45
    assert ft.ifunc == 1
    assert pytest.approx(ft.eps_max) == 0.35
    assert ft.fail_id == 841


def test_m278_fail_tab1_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Tab1 Aliases Test
/FAIL/TABULATED1/842
1, 1, 0.0, 0.0, 0
1.0, 0.0, 1.0, 0.0, 0
/FAIL/TAB_1D/843
1, 1, 0.0, 0.0, 0
/FAIL/TAB1_MODEL/844
1, 1, 0.0, 0.0, 0
/FAIL/TAB1_LAW/845
1, 1, 0.0, 0.0, 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 842 in model.fail_tab1s
    assert 843 in model.fail_tab1s
    assert 844 in model.fail_tab1s
    assert 845 in model.fail_tab1s
    assert len(model.raw_fails) == 4


def test_m278_eng_sph_energy(tmp_path: Path):
    c1 = f"{0.0005:>20.6f}{10:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
SPH Energy Fixed and Free Format Test
2022 0
/ENG/SPH_ENERGY/1
Fixed SPH Energy
{c1}
/ENG/SPH_ENERGY/2
Free SPH Energy
0.001, 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_sph_energies
    assert 2 in model.eng_sph_energies
    se1 = model.eng_sph_energies[1]
    assert pytest.approx(se1.dt_sph) == 0.0005
    assert se1.sens_id == 10
    se2 = model.eng_sph_energies[2]
    assert pytest.approx(se2.dt_sph) == 0.001
    assert se2.sens_id == 20


def test_m278_eng_sph_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
SPH Energy Aliases Test
/ENG/SPH_WORK/11
0.002, 10
/ENG/ESPH/12
0.003, 11
/ENG/SPH_ENER/13
0.004, 12
/ENG/SPH_INTERNAL_ENERGY/14
0.005, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_sph_energies
    assert 12 in model.eng_sph_energies
    assert 13 in model.eng_sph_energies
    assert 14 in model.eng_sph_energies
    assert pytest.approx(model.eng_sph_energies[11].dt_sph) == 0.002
    assert pytest.approx(model.eng_sph_energies[12].dt_sph) == 0.003
    assert pytest.approx(model.eng_sph_energies[13].dt_sph) == 0.004
    assert pytest.approx(model.eng_sph_energies[14].dt_sph) == 0.005


def test_m278_clevis_joint(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.5e6:>20.4f}{5:>10d}{1.0e-5:>20.6f}"
    c2 = f"{0.0:>20.4f}{1.0:>20.4f}{0.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Clevis Joint Fixed and Free Format Test
2022 0
/LAGMUL/CLEVIS_JOINT/1
Fixed Clevis Joint
{c1}
{c2}
/LAGMUL/CLEVIS_JOINT/2
Free Clevis Joint
201, 202, 203, 1.8e6, 6, 2.0e-5
1.0, 0.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_clevis_joints
    assert 2 in model.lagmul_clevis_joints
    cj1 = model.lagmul_clevis_joints[1]
    assert cj1.node1 == 101
    assert cj1.node2 == 102
    assert cj1.node3 == 103
    assert pytest.approx(cj1.stiff) == 2.5e6
    assert cj1.skew_id == 5
    assert pytest.approx(cj1.tol) == 1.0e-5
    assert pytest.approx(cj1.axis_x) == 0.0
    assert pytest.approx(cj1.axis_y) == 1.0
    assert pytest.approx(cj1.axis_z) == 0.0

    cj2 = model.lagmul_clevis_joints[2]
    assert cj2.node1 == 201
    assert cj2.node2 == 202
    assert cj2.node3 == 203
    assert pytest.approx(cj2.stiff) == 1.8e6
    assert cj2.skew_id == 6
    assert pytest.approx(cj2.tol) == 2.0e-5
    assert pytest.approx(cj2.axis_x) == 1.0
    assert pytest.approx(cj2.axis_y) == 0.0
    assert pytest.approx(cj2.axis_z) == 0.0


def test_m278_clevis_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Clevis Joint Aliases Test
/CLEVIS_JOINT/31
301, 302, 303, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/LAGMUL/CLEVIS_PIN/32
304, 305, 306, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/CLEVIS_PIN/33
307, 308, 309, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/CLEVIS_MECHANISM/34
310, 311, 312, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.lagmul_clevis_joints
    assert 32 in model.lagmul_clevis_joints
    assert 33 in model.lagmul_clevis_joints
    assert 34 in model.lagmul_clevis_joints
    assert model.lagmul_clevis_joints[31].node1 == 301
    assert model.lagmul_clevis_joints[32].node1 == 304
    assert model.lagmul_clevis_joints[33].node1 == 307
    assert model.lagmul_clevis_joints[34].node1 == 310


def test_m278_sensor_spring_total_work(tmp_path: Path):
    c1 = f"{501:>10d}{4500.0:>20.4f}{0.003:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Work Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_WORK/1
Fixed Spring Total Work Sensor
{c1}
/SENSOR/SPRING_TOTAL_WORK/2
Free Spring Total Work Sensor
502, 6000.0, 0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_works
    assert 2 in model.sensor_spring_total_works
    s1 = model.sensor_spring_total_works[1]
    assert s1.spring_id == 501
    assert pytest.approx(s1.w_tot_max) == 4500.0
    assert pytest.approx(s1.t_delay) == 0.003

    s2 = model.sensor_spring_total_works[2]
    assert s2.spring_id == 502
    assert pytest.approx(s2.w_tot_max) == 6000.0
    assert pytest.approx(s2.t_delay) == 0.005

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_WORK"
    assert model.sensors[1].kind == "SPRING_TOTAL_WORK"


def test_m278_sensor_spring_total_work_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Work Aliases Test
/SENSOR/SPRING_TOT_WORK/11
601, 7500.0, 0.001
/SENSOR/SPRING_WORK_TOTAL/12
602, 8500.0, 0.002
/SENSOR/TOTAL_WORK_SPRING/13
603, 9500.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_total_works
    assert 12 in model.sensor_spring_total_works
    assert 13 in model.sensor_spring_total_works
    assert model.sensor_spring_total_works[11].spring_id == 601
    assert pytest.approx(model.sensor_spring_total_works[11].w_tot_max) == 7500.0
    assert model.sensor_spring_total_works[12].spring_id == 602
    assert model.sensor_spring_total_works[13].spring_id == 603
