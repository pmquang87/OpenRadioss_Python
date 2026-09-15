"""
Tests for Milestone M102: Advanced Boundary Conditions, Kinematic Joints & Special Initial Conditions Suite
(/BCS/NRF, /BCS/WALL, /RLINK, /CYL_JOINT, /GJOINT, /MERGE, /INICRACK, /LASER).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import (
    BcsNrf, BcsWall, RigidLink, CylJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, LaserLoad
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
/GRNOD/NODE/10
Node_Group_10
         1         2
/GRNOD/NODE/20
Node_Group_20
         3         4
/FUNCT/1
Funct_Curve_1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Funct_Curve_2
                 0.0                 1.0
                 1.0                 2.0
/SENSOR/TIME/1
Sensor_Time_1
                 0.5
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
/GRNOD/NODE/10
Node_Group_10
1 2
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_bcs_nrf_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/BCS/NRF/1
Non Reflecting Boundary
        10
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.bcs_nrf
    bcs = model.bcs_nrf[1]
    assert bcs.grnod_id == 10
    assert bcs.title == "Non Reflecting Boundary"

    deck_free = _BOILERPLATE_FREE + """\
/BCS/NRF/2
Non Reflecting Free
10
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.bcs_nrf
    assert model_free.bcs_nrf[2].grnod_id == 10


def test_bcs_wall_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/BCS/WALL/1
Sliding Wall Boundary
        10         1
                0.01                0.99
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.bcs_walls
    bw = model.bcs_walls[1]
    assert bw.grnod_id == 10
    assert bw.sensor_id == 1
    assert bw.tstart == pytest.approx(0.01)
    assert bw.tstop == pytest.approx(0.99)

    deck_free = _BOILERPLATE_FREE + """\
/BCS/WALL/2
Sliding Wall Free
10 0
0.05 1.50
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.bcs_walls
    bw2 = model_free.bcs_walls[2]
    assert bw2.grnod_id == 10
    assert bw2.tstart == pytest.approx(0.05)
    assert bw2.tstop == pytest.approx(1.50)


def test_rlink_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/RLINK/1
Rigid Link Fixed
   111 000         1        10         2
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.rlinks
    rl = model.rlinks[1]
    assert rl.dofs == (1, 1, 1, 0, 0, 0)
    assert rl.skew_id == 1
    assert rl.grnod_id == 10
    assert rl.ipol == 2

    deck_free = _BOILERPLATE_FREE + """\
/RLINK/2
Rigid Link Free
1 1 1 1 1 1 0 10 1
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.rlinks
    rl2 = model_free.rlinks[2]
    assert rl2.dofs == (1, 1, 1, 1, 1, 1)
    assert rl2.grnod_id == 10


def test_cyl_joint_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/CYL_JOINT/1
Cylindrical Joint Fixed
         1         2        10
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.cyl_joints
    cj = model.cyl_joints[1]
    assert cj.node_id1 == 1
    assert cj.node_id2 == 2
    assert cj.grnod_id == 10

    deck_free = _BOILERPLATE_FREE + """\
/CYL_JOINT/2
Cylindrical Joint Free
1 2 10
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.cyl_joints
    assert model_free.cyl_joints[2].node_id1 == 1
    assert model_free.cyl_joints[2].node_id2 == 2


def test_gjoint_and_diff(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/GJOINT/GEAR/1
Gear Joint 1
         1                1.25                0.10                0.05         2         3         4
                0.20                0.01                 1.0                 0.0                 0.0
                0.30                0.02                 0.0                 1.0                 0.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.gjoints
    gj = model.gjoints[1]
    assert gj.subtype == "GEAR"
    assert gj.node_id0 == 1
    assert gj.fscale == pytest.approx(1.25)
    assert gj.mass0 == pytest.approx(0.10)
    assert gj.inertia0 == pytest.approx(0.05)
    assert gj.node_id1 == 2
    assert gj.node_id2 == 3
    assert gj.node_id3 == 4
    assert gj.mass1 == pytest.approx(0.20)
    assert gj.r1 == (pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.0))
    assert gj.mass2 == pytest.approx(0.30)
    assert gj.r2 == (pytest.approx(0.0), pytest.approx(1.0), pytest.approx(0.0))

    deck_diff = _BOILERPLATE_FREE + """\
/GJOINT/DIFF/2
Differential Joint
1 1.0 0.5 0.1 2 3 4
0.2 0.05 1.0 0.0 0.0
0.2 0.05 0.0 1.0 0.0
0.2 0.05 0.0 0.0 1.0
"""
    model_diff, log_diff = _run(tmp_path, deck_diff)
    assert not log_diff.errors
    assert 2 in model_diff.gjoints
    gjd = model_diff.gjoints[2]
    assert gjd.subtype == "DIFF"
    assert gjd.mass3 == pytest.approx(0.2)
    assert gjd.r3 == (pytest.approx(0.0), pytest.approx(0.0), pytest.approx(1.0))


def test_merge_node_and_rbody(tmp_path):
    deck_node = _BOILERPLATE_FIXED + """\
/MERGE/NODE/1
Merge Nodes in Group 10
               0.001        10         1
"""
    model, log = _run(tmp_path, deck_node)
    assert not log.errors
    assert 1 in model.node_merges
    mn = model.node_merges[1]
    assert mn.tol == pytest.approx(0.001)
    assert mn.grnod_id == 10
    assert mn.merge_type == 1

    deck_rbody = _BOILERPLATE_FIXED + """\
/MERGE/RBODY/2
Merge Rigid Bodies
         2
         1         1         2         1         2
         3         1         4         1         2
"""
    model_rb, log_rb = _run(tmp_path, deck_rbody)
    assert not log_rb.errors
    assert 2 in model_rb.rbody_merges
    mrb = model_rb.rbody_merges[2]
    assert len(mrb.items) == 2
    assert mrb.items[0] == (1, 1, 2, 1, 2)
    assert mrb.items[1] == (3, 1, 4, 1, 2)


def test_inicrack_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/INICRACK/1
Initial Crack Line
         2
         1         2                0.50
         2         3                0.75
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.inicracks
    ic = model.inicracks[1]
    assert len(ic.segments) == 2
    assert ic.segments[0].node_id1 == 1
    assert ic.segments[0].node_id2 == 2
    assert ic.segments[0].ratio == pytest.approx(0.50)
    assert ic.segments[1].node_id1 == 2
    assert ic.segments[1].node_id2 == 3
    assert ic.segments[1].ratio == pytest.approx(0.75)


def test_laser_fixed_and_dfs(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/LASER/1
Laser Heating Beam
               100.0         1                   0.05         2
                10.0                 2.0                 0.1                 0.5                0.01
         2         1
         1         2
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.laser_loads
    las = model.laser_loads[1]
    assert las.magnitude == pytest.approx(100.0)
    assert las.curve_id == 1
    assert las.s_target == pytest.approx(0.05)
    assert las.fct_id_target == 2
    assert las.hn == pytest.approx(10.0)
    assert las.vcp == pytest.approx(2.0)
    assert las.np == 2
    assert las.nc == 1
    assert las.plasma_elements == [1, 2]

    deck_dfs = _BOILERPLATE_FREE + """\
/DFS/LASER/5
DFS Laser
50.0 1 0.02 1
5.0 1.0 0.05 0.2 0.01
1 1
1
"""
    model_dfs, log_dfs = _run(tmp_path, deck_dfs)
    assert not log_dfs.errors
    assert 5 in model_dfs.laser_loads
    las5 = model_dfs.laser_loads[5]
    assert las5.magnitude == pytest.approx(50.0)
    assert las5.plasma_elements == [1]


def test_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/BCS/NRF/1
Missing Group NRF
       999
/BCS/WALL/2
Missing Sensor Wall
        10       999
                0.01                0.99
/RLINK/3
Missing Skew RLINK
   111 000       999        10         1
/CYL_JOINT/4
Missing Node Joint
       999         2        10
/GJOINT/5
Missing Node GJoint
       888                1.00                0.10                0.05         2         3         4
                0.20                0.01                 1.0                 0.0                 0.0
                0.30                0.02                 0.0                 1.0                 0.0
/MERGE/NODE/6
Missing Group Merge
               0.001       999         1
/INICRACK/7
Missing Node Crack
         1
       999         2                0.50
/LASER/8
Missing Function Laser
               100.0       999                   0.05       888
                10.0                 2.0                 0.1                 0.5                0.01
         1         1
         1
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
