"""M91 – /DFS/DETPOINT and /DFS/DETPLAN detonation ignition (Starter)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import DetonatorPoint, DetonatorPlane
from pyradioss.starter.starter import run_starter


# ──────────────────── reusable boilerplate ────────────────────────────
_ELEM = """\
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_one
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /DFS/DETPOINT tests
# ══════════════════════════════════════════════════════════════════════

class TestDetpointFixed:
    """/DFS/DETPOINT in fixed-format decks."""

    def test_basic_detpoint(self, tmp_path):
        """Parse /DFS/DETPOINT with coordinates, time and material."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_DETPOINT
      2021         0
{_ELEM}
/DFS/DETPOINT/1
#               XDET                YDET                ZDET                TDET mat_IDDET
                 5.0                 5.0                 0.0              0.001         1
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.det_points) == 1
        dp = model.det_points[0]
        assert isinstance(dp, DetonatorPoint)
        assert dp.id == 1
        assert abs(dp.x - 5.0) < 1e-12
        assert abs(dp.y - 5.0) < 1e-12
        assert abs(dp.z - 0.0) < 1e-12
        assert abs(dp.tdet - 0.001) < 1e-12
        assert dp.mat_id == 1

    def test_multiple_detpoints(self, tmp_path):
        """Multiple /DFS/DETPOINT entries are all stored."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_MULTI_DET
      2021         0
{_ELEM}
/DFS/DETPOINT/1
                 1.0                 2.0                 3.0               0.01         1
/DFS/DETPOINT/2
                 4.0                 5.0                 6.0               0.02         1
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.det_points) == 2
        assert model.det_points[0].id == 1
        assert model.det_points[1].id == 2
        assert abs(model.det_points[1].x - 4.0) < 1e-12


class TestDetpointFree:
    """/DFS/DETPOINT in free-format decks."""

    def test_free_format(self, tmp_path):
        """Free-format /DFS/DETPOINT parsing."""
        deck = f"""\
/BEGIN
TEST_DETPOINT_FREE
{_ELEM}
/DFS/DETPOINT/5
10.0 20.0 30.0 0.005 1
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.det_points) == 1
        dp = model.det_points[0]
        assert dp.id == 5
        assert abs(dp.x - 10.0) < 1e-12
        assert abs(dp.tdet - 0.005) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /DFS/DETPLAN tests
# ══════════════════════════════════════════════════════════════════════

class TestDetplanFixed:
    """/DFS/DETPLAN in fixed-format decks."""

    def test_basic_detplan(self, tmp_path):
        """Parse /DFS/DETPLAN with base point, time, material, and normal."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_DETPLAN
      2021         0
{_ELEM}
/DFS/DETPLAN/10
#                 XP                  YP                  ZP                TDET mat_IDDET
                 0.0                 0.0                 0.0              0.001         1
#                 NX                  NY                  NZ
                 1.0                 0.0                 0.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.det_planes) == 1
        dp = model.det_planes[0]
        assert isinstance(dp, DetonatorPlane)
        assert dp.id == 10
        assert abs(dp.x - 0.0) < 1e-12
        assert abs(dp.tdet - 0.001) < 1e-12
        assert dp.mat_id == 1
        assert abs(dp.nx - 1.0) < 1e-12
        assert abs(dp.ny - 0.0) < 1e-12
        assert abs(dp.nz - 0.0) < 1e-12


class TestDetplanFree:
    """/DFS/DETPLAN in free-format decks."""

    def test_free_format(self, tmp_path):
        """Free-format /DFS/DETPLAN parsing."""
        deck = f"""\
/BEGIN
TEST_DETPLAN_FREE
{_ELEM}
/DFS/DETPLAN/20
0.0 0.0 0.0 0.002 1
0.0 1.0 0.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        dp = model.det_planes[0]
        assert dp.id == 20
        assert abs(dp.tdet - 0.002) < 1e-12
        assert abs(dp.ny - 1.0) < 1e-12


class TestDetplanZeroVector:
    """Warning on zero direction vector."""

    def test_zero_normal_warning(self, tmp_path):
        """Zero normal vector should produce a warning."""
        deck = f"""\
/BEGIN
TEST_DETPLAN_ZERO
{_ELEM}
/DFS/DETPLAN/30
0.0 0.0 0.0 0.001 1
0.0 0.0 0.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        warn_msgs = [w for w in log.warnings if "direction vector" in str(w).lower()]
        assert len(warn_msgs) > 0
