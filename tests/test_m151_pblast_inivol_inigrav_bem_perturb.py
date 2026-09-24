"""Tests for Milestone M151:
- Extended Contact Interfaces (/INTER/TYPE1, /INTER/TYPE3, /INTER/TYPE5, /INTER/TYPE6,
  /INTER/TYPE14, /INTER/TYPE20, /INTER/TYPE21, /INTER/TYPE23)
- Blast Surface Loading (/LOAD/PBLAST, /PBLAST)
- Initial Volume Fraction (/INIVOL)
- Initial Gravity Field (/INIGRAV)
- Initial State Import (/INISTA, /INISTATE)
- Boundary Element Method (/BEM/FLOW, /BEM/DAA)
- Model Perturbation Suite (/PERTURB/PART/SHELL, /PERTURB/PART/SOLID, /PERTURB/FAIL)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ---------------------------------------------------------------------------
# PBLAST Tests
# ---------------------------------------------------------------------------
def test_pblast_fixed_format(tmp_path):
    c1 = f"{101:>10}{1:>10}{0:>10}{100:>10}{0:>10}{0:>10}{0:>10}"
    c2 = f"{0.0:>20.4f}{0.0:>20.4f}{5.0:>20.4f}{0.0:>20.4f}{2.5:>20.4f}"
    c3 = f"{1.0e-3:>20.4e}{0.05:>20.4f}"
    c4 = f"{102:>10}{1:>10}"
    deck = (
        "/BEGIN\n"
        "TEST PBLAST FIXED\n"
        "/LOAD/PBLAST/1\n"
        "Blast on plate\n"
        f"{c1}\n"
        f"{c2}\n"
        f"{c3}\n"
        f"{c4}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.pblast_loads
    pb = model.pblast_loads[1]
    assert pb.id == 1
    assert pb.title == "Blast on plate"
    assert pb.surf_id == 101
    assert pb.iabac == 1
    assert pb.ita_shift == 0
    assert pb.ndt == 100
    assert pb.xdet == pytest.approx(0.0)
    assert pb.ydet == pytest.approx(0.0)
    assert pb.zdet == pytest.approx(5.0)
    assert pb.tdet == pytest.approx(0.0)
    assert pb.wtnt == pytest.approx(2.5)
    assert pb.pmin == pytest.approx(1.0e-3)
    assert pb.tstop == pytest.approx(0.05)
    assert pb.surf_ground_id == 102
    assert pb.ishape == 1


def test_pblast_free_format(tmp_path):
    c1 = f"{201:>10}{2:>10}{0:>10}{100:>10}{0:>10}{0:>10}{0:>10}"
    c2 = f"{1.0:>20.4f}{2.0:>20.4f}{3.0:>20.4f}{0.0:>20.4f}{10.0:>20.4f}"
    c3 = f"{0.0:>20.4f}{1.0:>20.4f}"
    c4 = f"{0:>10}{0:>10}"
    deck = (
        "/BEGIN\n"
        "TEST PBLAST FREE\n"
        "/PBLAST/2\n"
        "Free Format Blast\n"
        f"{c1}\n"
        f"{c2}\n"
        f"{c3}\n"
        f"{c4}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 2 in model.pblast_loads
    pb = model.pblast_loads[2]
    assert pb.surf_id == 201
    assert pb.iabac == 2
    assert pb.xdet == pytest.approx(1.0)
    assert pb.ydet == pytest.approx(2.0)
    assert pb.zdet == pytest.approx(3.0)
    assert pb.wtnt == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# INIVOL Tests
# ---------------------------------------------------------------------------
def test_inivol_fixed_format(tmp_path):
    c1 = f"{10:>10}{2:>10}"
    c2 = f"{501:>10}{1:>10}{0:>10}{0:>10}{0.6:>20.4f}"
    c3 = f"{502:>10}{2:>10}{1:>10}{1:>10}{0.4:>20.4f}"
    deck = (
        "/BEGIN\n"
        "TEST INIVOL FIXED\n"
        "/INIVOL/1\n"
        "Multi Material Fill\n"
        f"{c1}\n"
        f"{c2}\n"
        f"{c3}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.inivols
    iv = model.inivols[1]
    assert iv.part_id == 10
    assert len(iv.containers) == 2
    assert iv.containers[0].surf_id == 501
    assert iv.containers[0].submat_id == 1
    assert iv.containers[0].ireversed == 0
    assert iv.containers[0].icumu == 0
    assert iv.containers[0].vfrac == pytest.approx(0.6)
    assert iv.containers[1].surf_id == 502
    assert iv.containers[1].submat_id == 2
    assert iv.containers[1].ireversed == 1
    assert iv.containers[1].icumu == 1
    assert iv.containers[1].vfrac == pytest.approx(0.4)


def test_inivol_single_container(tmp_path):
    c1 = f"{20:>10}{1:>10}"
    c2 = f"{601:>10}{1:>10}{0:>10}{0:>10}{1.0:>20.4f}"
    deck = (
        "/BEGIN\n"
        "TEST INIVOL FREE\n"
        "/INIVOL/5\n"
        "Free Inivol\n"
        f"{c1}\n"
        f"{c2}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 5 in model.inivols
    iv = model.inivols[5]
    assert iv.part_id == 20
    assert len(iv.containers) == 1
    assert iv.containers[0].surf_id == 601
    assert iv.containers[0].vfrac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# INIGRAV Tests
# ---------------------------------------------------------------------------
def test_inigrav_fixed_format(tmp_path):
    c1 = f"{12:>10}{0:>10}{1:>10}"
    c2 = f"{101325.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}{-10.0:>20.4f}"
    deck = (
        "/BEGIN\n"
        "TEST INIGRAV FIXED\n"
        "/INIGRAV/1\n"
        "Hydrostatic Init\n"
        f"{c1}\n"
        f"{c2}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.inigrav_loads
    ig = model.inigrav_loads[1]
    assert ig.grpart_id == 12
    assert ig.surf_id == 0
    assert ig.grav_id == 1
    assert ig.pref == pytest.approx(101325.0)
    assert ig.bz == pytest.approx(-10.0)


def test_inigrav_with_surface(tmp_path):
    c1 = f"{15:>10}{301:>10}{2:>10}"
    c2 = f"{0.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}"
    deck = (
        "/BEGIN\n"
        "TEST INIGRAV FREE\n"
        "/INIGRAV/2\n"
        "Free Inigrav\n"
        f"{c1}\n"
        f"{c2}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 2 in model.inigrav_loads
    ig = model.inigrav_loads[2]
    assert ig.grpart_id == 15
    assert ig.surf_id == 301
    assert ig.grav_id == 2


# ---------------------------------------------------------------------------
# INISTA / INISTATE Tests
# ---------------------------------------------------------------------------
def test_inista_parsing(tmp_path):
    c1 = f"{'state_file_t01.sta':<80}{1:>10}{2:>10}{0:>10}"
    deck = (
        "/BEGIN\n"
        "TEST INISTA FIXED\n"
        "/INISTA/1\n"
        "Initial State File Import\n"
        f"{c1}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.inistas
    ist = model.inistas[1]
    assert ist.filename == "state_file_t01.sta"
    assert ist.ibal == 1
    assert ist.ioutyy == 2
    assert ist.ioutynn == 0


def test_inistate_keyword(tmp_path):
    c1 = f"{'custom_dyn.sta':<80}{3:>10}{1:>10}{1:>10}"
    deck = (
        "/BEGIN\n"
        "TEST INISTATE FREE\n"
        "/INISTATE/3\n"
        "Free Inistate\n"
        f"{c1}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 3 in model.inistas
    ist = model.inistas[3]
    assert ist.filename == "custom_dyn.sta"
    assert ist.ibal == 3
    assert ist.ioutyy == 1
    assert ist.ioutynn == 1


# ---------------------------------------------------------------------------
# BEM Tests
# ---------------------------------------------------------------------------
def test_bem_flow_and_daa(tmp_path):
    c_flow = f"{701:>10}{1:>10}{801:>10}"
    c_daa = f"{702:>10}{2:>10}"
    deck = (
        "/BEGIN\n"
        "TEST BEM\n"
        "/BEM/FLOW/1\n"
        "BEM Flow Control\n"
        f"{c_flow}\n"
        "/BEM/DAA/2\n"
        "BEM DAA Free Surface\n"
        f"{c_daa}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.bem_controls
    assert 2 in model.bem_controls
    b1 = model.bem_controls[1]
    b2 = model.bem_controls[2]

    assert b1.subtype == "FLOW"
    assert b1.surf_id == 701
    assert b1.nio == 1
    assert b1.grnod_aux_id == 801

    assert b2.subtype == "DAA"
    assert b2.surf_id == 702
    assert b2.freesurf == 2


# ---------------------------------------------------------------------------
# PERTURB Tests
# ---------------------------------------------------------------------------
def test_perturb_shell_and_solid(tmp_path):
    c1_s = f"{1.0:>20.4f}{0.05:>20.4f}{0.8:>20.4f}{1.2:>20.4f}{12345:>10}{2:>10}"
    c2_s = f"{101:>10}{'THICK':>10}"
    c1_sol = f"{200.0:>20.4f}{10.0:>20.4f}{150.0:>20.4f}{250.0:>20.4f}{54321:>10}{1:>10}"
    c2_sol = f"{102:>10}{'YOUNG':>10}"
    deck = (
        "/BEGIN\n"
        "TEST PERTURB\n"
        "/PERTURB/PART/SHELL/1\n"
        "Shell Perturbation\n"
        f"{c1_s}\n"
        f"{c2_s}\n"
        "/PERTURB/PART/SOLID/2\n"
        "Solid Perturbation\n"
        f"{c1_sol}\n"
        f"{c2_sol}\n"
    )
    model, log = _parse_starter(tmp_path, deck)

    assert 1 in model.perturb_shells
    assert 1 in model.perturb_controls
    assert 2 in model.perturbations
    assert 2 in model.perturb_controls

    ps = model.perturb_shells[1]
    assert ps.grpart_id == 101
    assert ps.chvar == "THICK"
    assert ps.deviation == pytest.approx(0.05)

    pctrl_sol = model.perturb_controls[2]
    assert pctrl_sol.subtype == "PART/SOLID"
    assert pctrl_sol.grpart_id == 102
    assert pctrl_sol.ityp == 1
    assert pctrl_sol.scale == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Extended Contact Interfaces Tests
# ---------------------------------------------------------------------------
def test_extended_interfaces(tmp_path):
    types_to_test = [
        ("TYPE1", 1),
        ("TYPE3", 3),
        ("TYPE5", 5),
        ("TYPE6", 6),
        ("TYPE14", 14),
        ("TYPE20", 20),
        ("TYPE21", 21),
        ("TYPE23", 23),
    ]

    deck_lines = ["/BEGIN", "TEST EXTENDED INTERFACES"]
    for typename, itype in types_to_test:
        c1 = f"{10:>10}{20:>10}{1:>10}"
        deck_lines.extend([
            f"/INTER/{typename}/{itype}",
            f"Interface {typename}",
            c1,
        ])
    deck = "\n".join(deck_lines) + "\n"
    model, log = _parse_starter(tmp_path, deck)

    for typename, itype in types_to_test:
        matching = [i for i in model.interfaces if i.id == itype and i.type == itype]
        assert len(matching) == 1, f"Failed for interface {typename} id {itype}"
        inter = matching[0]
        if itype in (5, 14):
            assert inter.grnod_id == 10
            assert inter.surf_id == 20
        elif itype in (21, 23):
            assert inter.surf_id == 20
            assert inter.surf_id1 == 10
        else:
            assert inter.surf_id == 10
            assert inter.surf_id1 == 20


# ---------------------------------------------------------------------------
# Cross-Reference Checks Tests
# ---------------------------------------------------------------------------
def test_m151_cross_reference_checks(tmp_path):
    from pyradioss.starter.initialization import resolve_materials, build_element_groups, resolve_node_groups

    deck = (
        "/BEGIN\n"
        "CROSS REF M151 TEST\n"
        "/NODE\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "/MAT/LAW1/1\n"
        "Mat1\n"
        "7.8e-9 210000.0 0.3\n"
        "\n"
        "/PROP/TYPE4/1\n"
        "Prop1\n"
        "1.0 1.0\n"
        "/SPRING/1/1/1\n"
        "1 1 2\n"
        "/PART/1\n"
        "Part 1\n"
        "1 1\n"
        "/LOAD/PBLAST/2\n"
        "Blast on missing surf\n"
        f"{9999:>10}{1:>10}{0:>10}{100:>10}{0:>10}{0:>10}{9999:>10}\n"
        f"{0.0:>20.4f}{0.0:>20.4f}{5.0:>20.4f}{0.0:>20.4f}{1.0:>20.4f}\n"
        f"{0.0:>20.4f}{1.0:>20.4f}\n"
        f"{9998:>10}{0:>10}\n"
        "/INIVOL/2\n"
        "Inivol missing part and surf\n"
        f"{9999:>10}{1:>10}\n"
        f"{9999:>10}{1:>10}{0:>10}{0:>10}{1.0:>20.4f}\n"
        "/INIGRAV/2\n"
        "Inigrav missing grpart and surf\n"
        f"{9999:>10}{9999:>10}{1:>10}\n"
        f"{0.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}\n"
        "/BEM/FLOW/2\n"
        "BEM Flow missing surf and grnod\n"
        f"{9999:>10}{1:>10}{9999:>10}\n"
        "/PERTURB/PART/SHELL/2\n"
        "Perturb missing grpart\n"
        f"{1.0:>20.4f}{0.05:>20.4f}{0.8:>20.4f}{1.2:>20.4f}{12345:>10}{2:>10}\n"
        f"{9999:>10}{'THICK':>10}\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    check_model(model, log)

    err_msgs = " ".join(log.errors)
    assert "/LOAD/PBLAST/2: surface 9999" in err_msgs
    assert "/LOAD/PBLAST/2: ground surface 9998" in err_msgs
    assert "/LOAD/PBLAST/2: node 9999" in err_msgs or "/LOAD/PBLAST/2: detonation node 9999" in err_msgs
    assert "/INIVOL/2: part 9999" in err_msgs
    assert "/INIVOL/2: container surface 9999" in err_msgs or "/INIVOL/2: container surface/group 9999" in err_msgs
    assert "/INIGRAV/2: part group 9999" in err_msgs or "/INIGRAV/2: part group/part 9999" in err_msgs
    assert "/INIGRAV/2: surface 9999" in err_msgs
    assert "/BEM/FLOW/2: surface 9999" in err_msgs
    assert "/BEM/FLOW/2: node group 9999" in err_msgs
    assert "/PERTURB/PART/SHELL/2: part group 9999" in err_msgs or "/PERTURB/PART/SHELL/2: part group/part 9999" in err_msgs

