"""
Tests for Milestone M104: Virtual Sensors, Output Clusters, Flexible Bodies & Advanced Initial Field Suites
(/GAUGE, /CLUSTER, /EXTLNK, /FXBODY, /INIGRAV, /INIMAP1D, /INIMAP2D, /INISTATE).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import (
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile
)
from pyradioss.starter.starter import run_starter


_BOILERPLATE_FIXED = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_Fixed
      2022         0
/MAT/LAW1/1
Elastic_Matrix
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell
         1         1
/PART/2
Part_Shell_2
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/SURF/SEG/1
Surface_1
         1         2         3         4
/GRNOD/NODE/10
Node_Group_10
         1         2
/FUNCT/1
Funct_Curve_1
                 0.0                 0.0
                 1.0                 1.0
/GRAV/1
Gravity_1
         1         Z         0         0        10                   1.0                 1.0
/SKEW/FIX/1
Skew_1
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 0.0                 1.0                 0.0
"""

_BOILERPLATE_FREE = """\
# Free format boilerplate
/BEGIN
Test_Deck_Free
/MAT/LAW1/1
Elastic_Matrix
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
/SURF/SEG/1
Surface_1
1 2 3 4
/GRNOD/NODE/10
Node_Group_10
1 2
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
/GRAV/1
Gravity_1
1 Z 10 1.0
/SKEW/MOV2/1
Skew_1
1 2 3
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_gauge_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/GAUGE/SHELL/1
Strain Gauge Shell
         1                                                 1                0.05
/GAUGE/SPH/2
SPH Gauge
         2                                   100.0         1                0.02
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.gauges
    g1 = model.gauges[1]
    assert g1.subtype == "SHELL"
    assert g1.title == "Strain Gauge Shell"
    assert g1.node_id == 1
    assert g1.elem_id == 1
    assert g1.dist == pytest.approx(0.05)

    assert 2 in model.gauges
    g2 = model.gauges[2]
    assert g2.subtype == "SPH"
    assert g2.node_id == 2
    assert g2.fcut == pytest.approx(100.0)
    assert g2.elem_id == 1
    assert g2.dist == pytest.approx(0.02)

    deck_free = _BOILERPLATE_FREE + """\
/GAUGE/BEAM/3
Beam Gauge
1 1 0.10
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 3 in model_free.gauges
    g3 = model_free.gauges[3]
    assert g3.subtype == "BEAM"
    assert g3.node_id == 1
    assert g3.elem_id == 1
    assert g3.dist == pytest.approx(0.10)


def test_cluster_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/CLUSTER/BRICK/1
Brick Failure Cluster
        10         1         1
              1000.0                 1.0                 1.0
               500.0                 1.0                 1.0
               200.0                 1.0                 1.0
               150.0                 1.0                 1.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.clusters
    c1 = model.clusters[1]
    assert c1.subtype == "BRICK"
    assert c1.title == "Brick Failure Cluster"
    assert c1.group_id == 10
    assert c1.skew_id == 1
    assert c1.ifail == 1
    assert c1.fn_fail == pytest.approx(1000.0)
    assert c1.fs_fail == pytest.approx(500.0)
    assert c1.mt_fail == pytest.approx(200.0)
    assert c1.mb_fail == pytest.approx(150.0)

    deck_free = _BOILERPLATE_FREE + """\
/CLUSTER/SPRING/2
Spring Cluster
10 0 0
2000.0 1.0 1.0
800.0 1.0 1.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.clusters
    c2 = model_free.clusters[2]
    assert c2.subtype == "SPRING"
    assert c2.fn_fail == pytest.approx(2000.0)
    assert c2.fs_fail == pytest.approx(800.0)


def test_extlnk_and_fxbody(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/EXTLNK/1
External Link Coupling
        10
/FXBODY/2
Flexible Body CMS
         1         1         1        10
modal_body_mesh.pch
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.ext_links
    el = model.ext_links[1]
    assert el.title == "External Link Coupling"
    assert el.grnod_id == 10

    assert 2 in model.fxbodies
    fx = model.fxbodies[2]
    assert fx.title == "Flexible Body CMS"
    assert fx.node_id == 1
    assert fx.ianim == 1
    assert fx.imin == 1
    assert fx.imax == 10
    assert fx.filename == "modal_body_mesh.pch"


def test_inigrav_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/INIGRAV/1
Initial Gravity Equilibrium
         1         1         1                   101325.0                 0.0                 0.0                -9.81
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.ini_gravs
    ig = model.ini_gravs[1]
    assert ig.title == "Initial Gravity Equilibrium"
    assert ig.grpart_id == 1
    assert ig.surf_id == 1
    assert ig.grav_id == 1
    assert ig.pref == pytest.approx(101325.0)
    assert ig.bz == pytest.approx(-9.81)

    deck_free = _BOILERPLATE_FREE + """\
/INIGRAV/2
Initial Gravity Free
1 1 1 0.0 0.0 0.0 -9.81
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.ini_gravs
    ig2 = model_free.ini_gravs[2]
    assert ig2.bz == pytest.approx(-9.81)


def test_inimap_and_inista(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/INIMAP1D/1
1D Blast Mapping
         1         1         2         0         0         0                 1.0
blast_1d_output.map
/INIMAP2D/2
2D Blast Mapping
         1         1         2         3         0                 1.0
blast_2d_output.map
/INISTATE/FILE
initial_state_record.sta
         1         2
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.ini_map1ds
    m1 = model.ini_map1ds[1]
    assert m1.title == "1D Blast Mapping"
    assert m1.map_type == 1
    assert m1.node_id1 == 1
    assert m1.node_id2 == 2
    assert m1.filename == "blast_1d_output.map"

    assert 2 in model.ini_map2ds
    m2 = model.ini_map2ds[2]
    assert m2.title == "2D Blast Mapping"
    assert m2.node_id1 == 1
    assert m2.node_id2 == 2
    assert m2.node_id3 == 3
    assert m2.filename == "blast_2d_output.map"

    assert model.ini_state_file is not None
    assert model.ini_state_file.filename == "initial_state_record.sta"
    assert model.ini_state_file.isigi == 1
    assert model.ini_state_file.ioutp_fmt == 2


def test_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/GAUGE/SHELL/1
Missing Node Gauge
       999                                                 1                0.05
/CLUSTER/BRICK/2
Missing Skew Cluster
        10       999         1
              1000.0                 1.0                 1.0
/EXTLNK/3
Missing Group ExtLink
       999
/FXBODY/4
Missing Node FxBody
       999         1         1        10
modal_body.pch
/INIGRAV/5
Missing Grav IniGrav
         1         1       999                   101325.0                 0.0                 0.0                -9.81
/INIMAP1D/6
Missing Node IniMap
         1       999         2         0         0         0                 1.0
blast.map
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
