"""Tests for Milestone M115: Entity Sets (/SET, /SETS), Extended Surface Boundary Modifiers,
Monitored Volume Area (/MONVOL/AREA), Time History Titles (/TH/TITLE), State DT Controls (/STATE/DT, /DYNAIN/DT),
and State Output Entity Filters (/STATE/*).
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.starter import StarterError


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M115
      2022         0
/MAT/LAW1/1
LinearElastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell_1
         1         1
/PART/2
Part_Shell_2
         2         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/BOX/RECTA/10
BoundingBox10
         0         0         0
                 0.0                 0.0                 0.0
                 5.0                 5.0                 5.0
"""


def _run(tmp_path: Path, deck: str, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    try:
        model = run_starter(str(p), log)
    except StarterError:
        model = None
    return model, log


def test_m115_sets_node_part_elem_surf(tmp_path):
    deck = _BOILERPLATE + """\
/SET/NODE/10
SetNodes
         1         2         3         4
/SET/NODE/GENE/11
SetNodeGene
         1         4
/SET/PART/20
SetParts
         1         2
/SET/SHELL/30
SetShells
         1
/SET/BRIC/PART/31
SetBricks
         1
/SET/BEAM/PART/32
SetBeams
         1
/SET/TRUSS/PART/33
SetTrusses
         1
/SET/SPRING/PART/34
SetSprings
         1
/SET/QUAD/PART/35
SetQuads
         1
/SET/SH3N/PART/36
SetSh3n
         1
/SET/LINE/37
SetLines
         1
/SETS/SURF/40
SetSurface
         1         2
/SURF/SEG/1
Surf1
         1         2         3         4
/SURF/SEG/2
Surf2
         1         2         4         3
/LINE/SEG/1
Line1
         1         2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.node_groups
    assert model.node_groups[10].node_ids == [1, 2, 3, 4]
    assert 11 in model.node_groups
    assert model.node_groups[11].gene_ranges == [(1, 4, 1)]
    assert 20 in model.egroups.get("PART", {})
    assert model.egroups["PART"][20].part_ids == [1, 2]
    assert 30 in model.egroups.get("SHEL", {})
    assert model.egroups["SHEL"][30].elem_ids == [1]
    assert 31 in model.egroups.get("BRIC", {})
    assert model.egroups["BRIC"][31].part_ids == [1]
    assert 32 in model.egroups.get("BEAM", {})
    assert model.egroups["BEAM"][32].part_ids == [1]
    assert 33 in model.egroups.get("TRUS", {})
    assert model.egroups["TRUS"][33].part_ids == [1]
    assert 34 in model.egroups.get("SPRI", {})
    assert model.egroups["SPRI"][34].part_ids == [1]
    assert 35 in model.egroups.get("QUAD", {})
    assert model.egroups["QUAD"][35].part_ids == [1]
    assert 36 in model.egroups.get("SH3N", {})
    assert model.egroups["SH3N"][36].part_ids == [1]
    assert 37 in model.lines
    assert 40 in model.surfaces
    assert model.surfaces[40].surf_ids == [1, 2]


def test_m115_surf_modifiers_and_types(tmp_path):
    deck = _BOILERPLATE + """\
/SURF/EXT/100/PART
ExtPartSurf
         1
/SURF/ALL/200/MAT
AllMatSurf
         1
/SURF/FREE/300/PROP
FreePropSurf
         1
/SURF/BOX/400
BoxBoundedSurf
        10
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 100 in model.surfaces
    s100 = model.surfaces[100]
    assert s100.modifier == "EXT"
    assert s100.part_ids == [1]

    assert 200 in model.surfaces
    s200 = model.surfaces[200]
    assert s200.modifier == "ALL"
    assert s200.mat_ids == [1]

    assert 300 in model.surfaces
    s300 = model.surfaces[300]
    assert s300.modifier == "FREE"
    assert s300.prop_ids == [1]

    assert 400 in model.surfaces
    s400 = model.surfaces[400]
    assert s400.box_ids == [10]


def test_m115_monvol_area(tmp_path):
    deck = _BOILERPLATE + """\
/SURF/SEG/1
MonvolOuterSurf
         1         2         3         4
/MONVOL/AREA/1
MonitoredVolumeArea1
         1
                 1.0                 1.0                 1.0                 1.0                 1.5
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.monvol_areas
    ma = model.monvol_areas[1]
    assert ma.id == 1
    assert ma.title == "MonitoredVolumeArea1"
    assert ma.surf_id_ext == 1
    assert ma.scale_t == 1.0
    assert ma.scale_d == 1.5


def test_m115_th_title_and_state_dt(tmp_path):
    deck = _BOILERPLATE + """\
/TH/TITLE
Time History Results for M115 Run
Channel Output Run Summary
/STATE/DT/1
                 0.0                0.01
         1         2
/DYNAIN/DT
                 0.0                0.05
/STATE/SHELL/STRES/FULL
/STATE/BRICK/STRAIN/FULL
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.th_titles) == 2
    assert model.th_titles[0] == "Time History Results for M115 Run"
    assert model.th_titles[1] == "Channel Output Run Summary"
    assert len(model.state_dts) == 2
    sdt1 = model.state_dts[0]
    assert sdt1.tstart == 0.0
    assert sdt1.tfreq == 0.01
    assert sdt1.component_ids == [1, 2]
    sdt2 = model.state_dts[1]
    assert sdt2.tfreq == 0.05


def test_m115_cross_reference_errors(tmp_path):
    deck = _BOILERPLATE + """\
/MONVOL/AREA/5
BadMonvolArea
       999
/SURF/MAT/500
BadMatSurf
       888
/SURF/PROP/600
BadPropSurf
       777
/SURF/BOX/700
BadBoxSurf
       666
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/MONVOL/AREA/5: surface 999 not defined" in err_text
    assert "/SURF/500: material 888 not defined" in err_text
    assert "/SURF/600: property 777 not defined" in err_text
    assert "/SURF/700: box 666 not defined" in err_text


def test_m115_free_format(tmp_path):
    free_deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Free_M115
/MAT/LAW1/1
LinearElastic
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell_1
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
/SET/NODE/10
FreeNodes
1 2 3 4
/SET/PART/20
FreeParts
1
/SURF/SEG/1
FreeSurf
1 2 3 4
/MONVOL/AREA/1
FreeMonvolArea
1
1.0 1.0 1.0 1.0 2.0
/STATE/DT
0.0 0.02
1
/END
"""
    model, log = _run(tmp_path, free_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.node_groups
    assert 20 in model.egroups.get("PART", {})
    assert 1 in model.monvol_areas
    assert model.monvol_areas[1].scale_d == 2.0
    assert len(model.state_dts) == 1
    assert model.state_dts[0].tfreq == 0.02
