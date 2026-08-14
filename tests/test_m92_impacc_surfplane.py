"""M92 – /IMPACC (imposed acceleration) and /SURF/PLANE (infinite planar surface)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import ImposedAcceleration, Surface
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
/GRNOD/NODE/1
AllNodes
       101       102       103       104
/FUNCT/1
AccCurve
                 0.0                 0.0
                 1.0                10.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /IMPACC tests
# ══════════════════════════════════════════════════════════════════════

class TestImpaccFixed:
    """/IMPACC in fixed-format decks."""

    def test_basic_impacc_fixed(self, tmp_path):
        """Parse fixed-format /IMPACC with translational DOF X."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_IMPACC_FIXED
      2021         0
{_ELEM}
/IMPACC/1
Imposed_Acc_X
#  fct_IDT       Dir   skew_ID sensor_ID  grnod_ID  frame_ID     Icoor
         1         X         0         0         1         0         0
#           Ascale_x            Fscale_Y              Tstart               Tstop
                 1.0                 5.0                 0.0                 1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.impacc) == 1
        ia = model.impacc[0]
        assert isinstance(ia, ImposedAcceleration)
        assert ia.id == 1
        assert ia.funct_id == 1
        assert ia.grnod_id == 1
        assert ia.dof == 0  # X
        assert abs(ia.xscale - 1.0) < 1e-12
        assert abs(ia.scale - 5.0) < 1e-12
        assert abs(ia.tstart - 0.0) < 1e-12
        assert abs(ia.tstop - 1.0) < 1e-12
        assert ia.title == "Imposed_Acc_X"

    def test_rotational_impacc_fixed(self, tmp_path):
        """Parse fixed-format /IMPACC with rotational DOF ZZ (dof 5)."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_IMPACC_ROT
      2021         0
{_ELEM}
/IMPACC/2
Imposed_Rot_ZZ
         1        ZZ         0         0         1         0         0
                 2.0               100.0                 0.1                 0.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.impacc) == 1
        ia = model.impacc[0]
        assert ia.id == 2
        assert ia.dof == 5  # ZZ -> 5
        assert abs(ia.xscale - 2.0) < 1e-12
        assert abs(ia.scale - 100.0) < 1e-12


class TestImpaccFree:
    """/IMPACC in free-format decks."""

    def test_basic_impacc_free(self, tmp_path):
        """Parse free-format /IMPACC (historic compact format: fct Dir grnod scale)."""
        deck = f"""\
/BEGIN
TEST_IMPACC_FREE
{_ELEM}
/IMPACC/3
FreeAcc
1 Y 1 2.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.impacc) == 1
        ia = model.impacc[0]
        assert ia.id == 3
        assert ia.dof == 1  # Y
        assert ia.grnod_id == 1
        assert abs(ia.scale - 2.5) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /SURF/PLANE tests
# ══════════════════════════════════════════════════════════════════════

class TestSurfPlaneFixed:
    """/SURF/PLANE in fixed-format decks."""

    def test_basic_surf_plane_fixed(self, tmp_path):
        """Parse fixed-format /SURF/PLANE with P1 and P2 cards."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SURF_PLANE
      2021         0
{_ELEM}
/SURF/PLANE/1
Rigid_Plane
#                 Xm                  Ym                  Zm
                 0.0                 0.0                 0.0
#                Xm1                 Ym1                 Zm1
                 0.0                 0.0                 1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.surfaces
        s = model.surfaces[1]
        assert isinstance(s, Surface)
        assert s.id == 1
        assert s.title == "Rigid_Plane"
        assert s.plane_p1 is not None
        assert s.plane_p2 is not None
        np.testing.assert_allclose(s.plane_p1, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(s.plane_p2, [0.0, 0.0, 1.0])


class TestSurfPlaneFree:
    """/SURF/PLANE in free-format decks."""

    def test_basic_surf_plane_free(self, tmp_path):
        """Parse free-format /SURF/PLANE."""
        deck = f"""\
/BEGIN
TEST_SURF_PLANE_FREE
{_ELEM}
/SURF/PLANE/2
FreePlane
10.0 20.0 30.0
10.0 20.0 31.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.surfaces
        s = model.surfaces[2]
        np.testing.assert_allclose(s.plane_p1, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(s.plane_p2, [10.0, 20.0, 31.0])

    def test_zero_normal_error(self, tmp_path):
        """Identical P1 and P2 points should log an error (zero normal)."""
        from pyradioss.common.messages import StarterError
        deck = f"""\
/BEGIN
TEST_SURF_PLANE_ERR
{_ELEM}
/SURF/PLANE/3
ZeroNormal
5.0 5.0 5.0
5.0 5.0 5.0
/END
"""
        with pytest.raises(StarterError):
            _run(tmp_path, deck)
