"""Tests for Milestone M85: /TRANSFORM suite (/TRANSFORM/ROT, /TRANSFORM/SYM, /TRANSFORM/SCA)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter


_MINIMAL_ELEM_DECK = """/PROP/TYPE4/1
Spring prop
100.0
/MAT/LAW1/1
Mat
7.8e-6
210000.0 0.3
/PART/1
Part
1 1
/SPRING/1
1 1 2
"""



def test_transform_rot(tmp_path):
    # 90-degree rotation of node 2 (1, 0, 0) around Z-axis passing through (0, 0, 0) -> (0, 1, 0)
    deck = f"""/BEGIN
TEST TRANSFORM ROT
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/GRNOD/NODE/10
All nodes
1 2
/TRANSFORM/ROT/1
Rotate 90 deg around Z
10 0.0 0.0 0.0 0 0 0
0.0 0.0 1.0 90.0
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_ROT_0000.rad"
    p.write_text(deck)

    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    idx1 = model.node_index(1)
    idx2 = model.node_index(2)
    np.testing.assert_allclose(model.x0[idx1], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(model.x0[idx2], [0.0, 1.0, 0.0], atol=1e-6)


def test_transform_sym(tmp_path):
    # Reflection across plane at x=1 with normal along X (point1=(1,0,0), point2=(2,0,0))
    # Node 1 at (0, 0, 0) -> reflected to (2, 0, 0)
    deck = f"""/BEGIN
TEST TRANSFORM SYM
/NODE
1 0.0 0.0 0.0
2 3.0 0.0 0.0
/GRNOD/NODE/10
Node 1
1
/TRANSFORM/SYM/1
Reflect across x=1 plane
10 1.0 0.0 0.0 0 0 0
2.0 0.0 0.0
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_SYM_0000.rad"
    p.write_text(deck)

    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    idx1 = model.node_index(1)
    np.testing.assert_allclose(model.x0[idx1], [2.0, 0.0, 0.0], atol=1e-6)


def test_transform_sca(tmp_path):
    # Scaling by (2, 3, 4) from origin (0, 0, 0)
    # Node 1 at (1, 1, 1) -> (2, 3, 4)
    deck = f"""/BEGIN
TEST TRANSFORM SCA
/NODE
1 1.0 1.0 1.0
2 0.0 0.0 0.0
/GRNOD/NODE/10
Node 1
1
/TRANSFORM/SCA/1
Scale
10 2.0 3.0 4.0 0 0
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_SCA_0000.rad"
    p.write_text(deck)

    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    idx1 = model.node_index(1)
    np.testing.assert_allclose(model.x0[idx1], [2.0, 3.0, 4.0], atol=1e-6)


def test_transform_rot_by_nodes(tmp_path):
    # Rotate around axis defined by node 1 (0,0,0) and node 2 (0,0,1)
    # Node 3 at (1,0,0) rotated by 180 degrees -> (-1,0,0)
    deck = f"""/BEGIN
TEST TRANSFORM ROT BY NODES
/NODE
1 0.0 0.0 0.0
2 0.0 0.0 1.0
3 1.0 0.0 0.0
/GRNOD/NODE/10
Node 3
3
/TRANSFORM/ROT/1
Rotate 180 deg around node1-node2
10 0.0 0.0 0.0 1 2 0
0.0 0.0 0.0 180.0
/PROP/TYPE4/1
Spring prop
100.0
/MAT/LAW1/1
Mat
7.8e-6
210000.0 0.3
/PART/1
Part
1 1
/SPRING/1
1 1 2
2 1 3
/END
"""
    p = tmp_path / "TEST_ROT_NODES_0000.rad"
    p.write_text(deck)

    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    idx3 = model.node_index(3)
    np.testing.assert_allclose(model.x0[idx3], [-1.0, 0.0, 0.0], atol=1e-6)


def test_transform_fixed_rot(tmp_path):
    # Fixed format test for /TRANSFORM/ROT
    deck = """/BEGIN
TEST TRANSFORM FIXED ROT
2022 0
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
/GRNOD/NODE/10
All nodes
         1         2
/TRANSFORM/ROT/1
Rotate 90 deg
        10                 0.0                 0.0                 0.0         0         0         0
                           0.0                 0.0                 1.0                90.0
/PROP/TYPE4/1
Spring prop
               100.0
/MAT/LAW1/1
Mat
              7.8e-6
            210000.0                 0.3
/PART/1
Part
         1         1
/SPRING/1
         1         1         2
/END
"""
    p = tmp_path / "TEST_FIXED_ROT_0000.rad"
    p.write_text(deck)

    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    idx1 = model.node_index(1)
    idx2 = model.node_index(2)
    np.testing.assert_allclose(model.x0[idx1], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(model.x0[idx2], [0.0, 1.0, 0.0], atol=1e-6)


