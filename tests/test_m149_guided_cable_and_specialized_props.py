"""
Unit tests for Milestone M149:
- /INTER/GUIDED_CABLE (guided cable sliding contact interface)
- /INTER/SUB and /SUBINTER (sub-interface definitions)
- /PROP/STITCH (TYPE35 stitch connection property)
- /PROP/PREDIT (TYPE36 progressive damage interface property)
- /PROP/SPR_MUSCLE (TYPE46 Hill-type active muscle spring property)
- Extended Time-History (/TH) channels for interfaces and sub-interfaces
"""

from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.output.time_history import TimeHistory
from pyradioss.starter.checks import check_model


def _parse_starter(tmp_path: Path, text: str, name: str = "TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_guided_cable_parsing_fixed_and_free(tmp_path):
    deck_fixed = (
        "# RADIOSS STARTER DECK\n"
        "/BEGIN\n"
        "GUIDED CABLE TEST\n"
        "                  2020         0\n"
        "                  1.000000000000000e-03                 1.000000000000000e-03\n"
        "/INTER/GUIDED_CABLE/101\n"
        "Sliding Guided Cable Interface\n"
        "#  GRNOD_ID  GRPART_ID    ISTIFF               STFAC                FRIC\n"
        "        10       200         2                 1.5                 0.1\n"
        "/END\n"
    )
    model_f, log_f = _parse_starter(tmp_path, deck_fixed, "gc_fixed_0000.rad")
    assert len(log_f.errors) == 0
    assert 101 in model_f.guided_cables
    gc = model_f.guided_cables[101]
    assert gc.id == 101
    assert gc.title == "Sliding Guided Cable Interface"
    assert gc.grnod_id == 10
    assert gc.grpart_id == 200
    assert gc.istiff == 2
    assert gc.stfac == pytest.approx(1.5)
    assert gc.fric == pytest.approx(0.1)

    # Free format deck
    deck_free = (
        "/BEGIN\n"
        "GUIDED CABLE FREE\n"
        "/GUIDED_CABLE/202\n"
        "Free Guided Cable\n"
        "15 300 1 2.0 0.05\n"
        "/END\n"
    )
    model_fr, log_fr = _parse_starter(tmp_path, deck_free, "gc_free_0000.rad")
    assert len(log_fr.errors) == 0
    assert 202 in model_fr.guided_cables
    gc2 = model_fr.guided_cables[202]
    assert gc2.id == 202
    assert gc2.title == "Free Guided Cable"
    assert gc2.grnod_id == 15
    assert gc2.grpart_id == 300
    assert gc2.istiff == 1
    assert gc2.stfac == pytest.approx(2.0)
    assert gc2.fric == pytest.approx(0.05)


def test_subinterface_parsing_and_aliases(tmp_path):
    deck = (
        "/BEGIN\n"
        "SUBINTER TEST\n"
        "/INTER/SUB/1\n"
        "SubInterface 1\n"
        "# INTER_ID   MAIN_ID1  SECOND_ID   MAIN_ID2\n"
        "        10         20         30         40\n"
        "/SUBINTER/2\n"
        "SubInterface 2\n"
        "        10         25         35          0\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck, "sub_0000.rad")
    assert len(log.errors) == 0
    assert len(model.sub_interfaces) == 2
    sub1 = model.sub_interfaces[0]
    assert sub1.id == 1
    assert sub1.title == "SubInterface 1"
    assert sub1.inter_id == 10
    assert sub1.main_id1 == 20
    assert sub1.second_id == 30
    assert sub1.main_id2 == 40

    sub2 = model.sub_interfaces[1]
    assert sub2.id == 2
    assert sub2.title == "SubInterface 2"
    assert sub2.inter_id == 10
    assert sub2.main_id1 == 25
    assert sub2.second_id == 35
    assert sub2.main_id2 == 0


def test_prop_stitch_parsing(tmp_path):
    deck = (
        "/BEGIN\n"
        "PROP STITCH TEST\n"
        "/PROP/STITCH/50\n"
        "Stitch Fastener Prop\n"
        "#             K_TENS              K_COMP             K_SHEAR              F_TENS             F_SHEAR\n"
        "               1.0e5               2.0e5               0.5e5               500.0               250.0\n"
        "#   SKEW_ID     IFLAG      IPEN     IFAIL            DIST_MAX                AREA\n"
        "          2         1         0         3                12.5                 2.5\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck, "stitch_0000.rad")
    assert len(log.errors) == 0
    assert 50 in model.properties
    prop = model.properties[50]
    assert prop.type == 35
    assert prop.title == "Stitch Fastener Prop"
    assert prop.params["k_tens"] == pytest.approx(1.0e5)
    assert prop.params["k_comp"] == pytest.approx(2.0e5)
    assert prop.params["k_shear"] == pytest.approx(0.5e5)
    assert prop.params["f_tens"] == pytest.approx(500.0)
    assert prop.params["f_shear"] == pytest.approx(250.0)
    assert prop.params["skew_id"] == 2
    assert prop.params["iflag"] == 1
    assert prop.params["ipen"] == 0
    assert prop.params["ifail"] == 3
    assert prop.params["dist_max"] == pytest.approx(12.5)
    assert prop.params["area"] == pytest.approx(2.5)


def test_prop_predit_parsing(tmp_path):
    # Card layout: hm_cfg_files/config/CFG/radioss110/PROP/prop_p36_predit.cfg
    # (FORMAT radioss51) and starter/source/properties/spring/hm_read_prop36.F
    # lines 103-166: Iutype=1 -> skew_ID prop_ID1 prop_ID2 / Xk ;
    #                Iutype=2 -> MAT_ID / Area Ixx Iyy Izz Ray.
    # Iutype = 1 (interface referring to two shell properties)
    deck_1 = (
        "/BEGIN\n"
        "PROP PREDIT TEST 1\n"
        "/PROP/PREDIT/60\n"
        "Predit Delamination Interface Prop\n"
        "#   Iutype\n"
        "         1\n"
        "#  skew_ID  prop_ID1  prop_ID2\n"
        "         2       101       102\n"
        "#                 Xk\n"
        "               5.5e4\n"
        "/END\n"
    )
    model_1, log_1 = _parse_starter(tmp_path, deck_1, "predit1_0000.rad")
    assert len(log_1.errors) == 0
    p1 = model_1.properties[60]
    assert p1.type == 36
    assert p1.params["lutype"] == 1
    assert p1.params["skew_id"] == 2
    assert p1.params["prop_id1"] == 101
    assert p1.params["prop_id2"] == 102
    assert p1.params["xk"] == pytest.approx(5.5e4)

    # Iutype = 2 (beam-like section referring to a material)
    deck_2 = (
        "/BEGIN\n"
        "PROP PREDIT TEST 2\n"
        "/PROP/TYPE36/61\n"
        "Predit Delamination Section Prop\n"
        "#   Iutype\n"
        "         2\n"
        "#   MAT_ID\n"
        "         7\n"
        "#               Area                 Ixx                 Iyy                 Izz                 Ray\n"
        "                 1.1                 2.2                 3.3                 4.4                 5.5\n"
        "/END\n"
    )
    model_2, log_2 = _parse_starter(tmp_path, deck_2, "predit2_0000.rad")
    assert len(log_2.errors) == 0
    p2 = model_2.properties[61]
    assert p2.type == 36
    assert p2.params["lutype"] == 2
    assert p2.params["mat_id"] == 7
    assert p2.params["area"] == pytest.approx(1.1)
    assert p2.params["ixx"] == pytest.approx(2.2)
    assert p2.params["iyy"] == pytest.approx(3.3)
    assert p2.params["izz"] == pytest.approx(4.4)
    assert p2.params["ray"] == pytest.approx(5.5)


def test_prop_spr_muscle_parsing(tmp_path):
    deck = (
        "/BEGIN\n"
        "PROP SPR_MUSCLE TEST\n"
        "/PROP/SPR_MUSCLE/70\n"
        "Active Muscle Hill Model\n"
        "#               MASS               F_MAX               L_OPT               V_MAX                K_PE\n"
        "              0.0015               450.0               0.085                 1.2               350.0\n"
        "#      ITYPE  IFUNC_CE IFUNC_SEE  IFUNC_PE  IFUNC_DE IFUNC_ACT\n"
        "           1        11        12        13        14        15\n"
        "#             F_SEE0     IFLAG\n"
        "                25.0         1\n"
        "#            GAMMA_M              BETA_M            ACT_INIT             TAU_ACT\n"
        "                0.25                0.75                0.05               0.015\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck, "muscle_0000.rad")
    assert len(log.errors) == 0
    assert 70 in model.properties
    p = model.properties[70]
    assert p.type == 46
    assert p.title == "Active Muscle Hill Model"
    assert p.params["mass"] == pytest.approx(0.0015)
    assert p.params["f_max"] == pytest.approx(450.0)
    assert p.params["l_opt"] == pytest.approx(0.085)
    assert p.params["v_max"] == pytest.approx(1.2)
    assert p.params["k_pe"] == pytest.approx(350.0)
    assert p.params["itype"] == 1
    assert p.params["ifunc_ce"] == 11
    assert p.params["ifunc_see"] == 12
    assert p.params["ifunc_pe"] == 13
    assert p.params["ifunc_de"] == 14
    assert p.params["ifunc_act"] == 15
    assert p.params["f_see0"] == pytest.approx(25.0)
    assert p.params["iflag"] == 1
    assert p.params["gamma_m"] == pytest.approx(0.25)
    assert p.params["beta_m"] == pytest.approx(0.75)
    assert p.params["act_init"] == pytest.approx(0.05)
    assert p.params["tau_act"] == pytest.approx(0.015)


def test_time_history_and_cross_ref_checks(tmp_path):
    deck = (
        "/BEGIN\n"
        "TH CHANNELS TEST\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/PROP/SPRING/1\n"
        "Spring Prop\n"
        "0.0 100.0 0.0\n"
        "/MAT/PLAS_JOHNS/1/1\n"
        "Steel\n"
        "7.85e-6\n"
        "210.0 0.3\n"
        "0.2 0.1 0.5\n"
        "/SPRING/1/1/1\n"
        "1 1 2\n"
        "/GRNOD/NODE/1\n"
        "Nodes Group 1\n"
        "1 2\n"
        "/PART/1\n"
        "Part 1\n"
        "1 1\n"
        "/INTER/GUIDED_CABLE/1\n"
        "Cable 1\n"
        "1 1 1 1.0 0.0\n"
        "/TH/GUIDED_CABLE/1\n"
        "Cable TH\n"
        "FN FT\n"
        "1\n"
        "/TH/SUBS/1\n"
        "Sub TH\n"
        "FN FT\n"
        "1\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck, "th_0000.rad")
    assert len(log.errors) == 0
    assert len(model.th_requests) == 2

    # Initialize model groups and check model cross-references
    from pyradioss.starter.initialization import resolve_materials, build_element_groups, resolve_node_groups
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    check_model(model, log)
    assert len(log.errors) == 0

    # Verify TimeHistory writes columns properly without error
    th_file = tmp_path / "test_out.csv"
    th = TimeHistory(str(th_file), model, log)
    energies = {"IE": 1.0, "KE": 2.0, "HE": 0.0, "CE": 0.0, "EN": 0.0, "DE": 0.0, "EW": 3.0, "ERR": 0.0}
    th.write(0.001, energies, mass=1.0, momentum=[0.0, 0.0, 0.0])
    th.close()

    content = th_file.read_text()
    assert "GU1_FN" in content
    assert "SU1_FN" in content
