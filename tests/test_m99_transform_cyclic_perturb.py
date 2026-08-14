"""
Tests for Milestone M99: Transformations, Cyclic Symmetry, Perturbations & Blast Load Suite
(/TRANSFORM/POS, /BCS/CYCLIC, /PERTURB/PART/SOLID, /LOAD/PBLAST, /DEF_INTER/TYPE25).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import CyclicBoundaryCondition, SolidPartPerturbation, PBlastLoad
from pyradioss.starter.starter import run_starter


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PROP/TYPE14/2
Solid_Prop
         1         1         0         0         0         0         0
/PART/1
Part_Shell
         1         1
/PART/2
Part_Solid
         2         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
       101                 0.0                 0.0                 0.0
       102                 1.0                 0.0                 0.0
       103                 0.0                 1.0                 0.0
       201                10.0                20.0                30.0
       202                10.0                20.0                31.0
       203                10.0                21.0                30.0
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/1
Target_Nodes
         1         2         3         4
/GRNOD/NODE/2
Group_Two
         1         2
/GRNOD/NODE/3
Group_Three
         3         4
/SKEW/FIX/1
Skew_Sys
                 0.0                 0.0                 0.0
                 0.0                 1.0                 0.0
                 0.0                 0.0                 1.0
/SURF/SEG/1
Blast_Surface
         1         2         3         4
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /TRANSFORM/POS tests
# ══════════════════════════════════════════════════════════════════════

class TestTransformPos:
    """/TRANSFORM/POS and /TRANSFORM/POSITION 6-point/6-node rigid positioning."""

    def test_transform_position_nodes_fixed(self, tmp_path):
        """Transform node group using 6 reference nodes in fixed format."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_TRANSFORM_POS_NODES
      2021         0
{_BOILERPLATE}
/TRANSFORM/POS/1
Transform_Title
# grnod_ID  node_ID1  node_ID2  node_ID3  node_ID4  node_ID5  node_ID6                        sub_ID
         1       101       102       103       201       202       203                             0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0

        # Frame 1: origin=(0,0,0), ex=(1,0,0), ey=(0,1,0), ez=(0,0,1)
        # Frame 2: origin=(10,20,30), ex=(0,0,1), ey=(0,1,0), ez=(-1,0,0)
        # Node 1 originally at (0,0,0) -> target (10, 20, 30)
        # Node 2 originally at (2,0,0) -> local (2,0,0) -> target (10, 20, 30) + 2*ex2 = (10, 20, 32)
        # Node 4 originally at (0,2,0) -> local (0,2,0) -> target (10, 20, 30) + 2*ey2 = (10, 22, 30)
        idx1 = model.node_index(1)
        idx2 = model.node_index(2)
        idx4 = model.node_index(4)

        assert np.allclose(model.x0[idx1], [10.0, 20.0, 30.0], atol=1e-10)
        assert np.allclose(model.x0[idx2], [10.0, 20.0, 32.0], atol=1e-10)
        assert np.allclose(model.x0[idx4], [10.0, 22.0, 30.0], atol=1e-10)

    def test_transform_position_coords_free(self, tmp_path):
        """Transform node group using explicit coordinates in free format."""
        deck = f"""\
/BEGIN
TEST_TRANSFORM_POS_COORDS
{_BOILERPLATE}
/TRANSFORM/POSITION/1
Transform_Free
1 0 0 0 0 0 0 0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
50.0 50.0 50.0
50.0 51.0 50.0
49.0 50.0 50.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0

        # Frame 1: origin=(0,0,0), ex=(1,0,0), ey=(0,1,0), ez=(0,0,1)
        # Frame 2: origin=(50,50,50), ex=(0,1,0), ey=(-1,0,0), ez=(0,0,1)
        # Node 1 at (0,0,0) -> (50,50,50)
        # Node 2 at (2,0,0) -> (50, 50+2, 50) = (50, 52, 50)
        # Node 4 at (0,2,0) -> (50-2, 50, 50) = (48, 50, 50)
        idx1 = model.node_index(1)
        idx2 = model.node_index(2)
        idx4 = model.node_index(4)

        assert np.allclose(model.x0[idx1], [50.0, 50.0, 50.0], atol=1e-10)
        assert np.allclose(model.x0[idx2], [50.0, 52.0, 50.0], atol=1e-10)
        assert np.allclose(model.x0[idx4], [48.0, 50.0, 50.0], atol=1e-10)


# ══════════════════════════════════════════════════════════════════════
#  /BCS/CYCLIC tests
# ══════════════════════════════════════════════════════════════════════

class TestBcsCyclic:
    """/BCS/CYCLIC cyclic symmetry boundary condition."""

    def test_bcs_cyclic_fixed(self, tmp_path):
        """Parse fixed-format /BCS/CYCLIC."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_BCS_CYCLIC
      2021         0
{_BOILERPLATE}
/BCS/CYCLIC/1
Cyclic_Boundary
#  skew_ID  grnd_ID1  grnd_ID2
         1         2         3
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.cyclic_bcs
        cb = model.cyclic_bcs[1]
        assert isinstance(cb, CyclicBoundaryCondition)
        assert cb.skew_id == 1
        assert cb.grnod1_id == 2
        assert cb.grnod2_id == 3

    def test_bcs_cyclic_invalid_refs(self, tmp_path):
        """Test cross-reference errors when cyclic boundary condition groups are invalid."""
        deck = f"""\
/BEGIN
TEST_BCS_CYCLIC_ERR
{_BOILERPLATE}
/BCS/CYCLIC/1
Cyclic_Error
99 2 999
/END
"""
        with pytest.raises(Exception):
            _run(tmp_path, deck)


# ══════════════════════════════════════════════════════════════════════
#  /PERTURB/PART/SOLID tests
# ══════════════════════════════════════════════════════════════════════

class TestPerturbPartSolid:
    """/PERTURB/PART/SOLID part parameter perturbation."""

    def test_perturb_solid_fixed(self, tmp_path):
        """Parse fixed-format /PERTURB/PART/SOLID."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PERTURB_FIXED
      2021         0
{_BOILERPLATE}
/PERTURB/PART/SOLID/1
Density_Perturbation
#             F_Mean           Deviation             Min_cut             Max_cut      Seed   Idistri
               1.000               0.050               0.850               1.150     12345         2
#grpart_ID           parameter
         2                DENS
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.perturbations
        pt = model.perturbations[1]
        assert isinstance(pt, SolidPartPerturbation)
        assert abs(pt.f_mean - 1.0) < 1e-12
        assert abs(pt.deviation - 0.05) < 1e-12
        assert abs(pt.min_cut - 0.85) < 1e-12
        assert abs(pt.max_cut - 1.15) < 1e-12
        assert pt.seed == 12345
        assert pt.idistri == 2
        assert pt.grpart_id == 2
        assert pt.var_name == "DENS"

    def test_perturb_solid_free(self, tmp_path):
        """Parse free-format /PERTURB/PART/SOLID."""
        deck = f"""\
/BEGIN
TEST_PERTURB_FREE
{_BOILERPLATE}
/PERTURB/PART/SOLID/2
Young_Modulus_Perturb
1.0 0.1 0.7 1.3 999 1
2 E
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.perturbations
        pt = model.perturbations[2]
        assert abs(pt.f_mean - 1.0) < 1e-12
        assert abs(pt.deviation - 0.1) < 1e-12
        assert pt.idistri == 1
        assert pt.var_name == "E"


# ══════════════════════════════════════════════════════════════════════
#  /LOAD/PBLAST and /DEF_INTER tests
# ══════════════════════════════════════════════════════════════════════

class TestPBlastAndDefInter:
    """/LOAD/PBLAST air/ground blast loads and /DEF_INTER defaults."""

    def test_pblast_fixed(self, tmp_path):
        """Parse fixed-format /LOAD/PBLAST."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PBLAST_FIXED
      2021         0
{_BOILERPLATE}
/LOAD/PBLAST/1
Explosion_Load
#  surf_ID  Exp_data  I_tshift       Ndt        IZ    Imodel                                 Node_id
         1         1         1         0         2         0                                     101
#               Xdet                Ydet                Zdet                Tdet                WTNT
                 0.0                 0.0                15.0               0.001               100.0
#               Pmin
                 0.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.pblast_loads
        pb = model.pblast_loads[1]
        assert isinstance(pb, PBlastLoad)
        assert pb.surf_id == 1
        assert pb.exp_data == 1
        assert pb.iz == 2
        assert pb.node_id == 101
        assert abs(pb.zdet - 15.0) < 1e-12
        assert abs(pb.tdet - 0.001) < 1e-12
        assert abs(pb.wtnt - 100.0) < 1e-12

    def test_def_inter(self, tmp_path):
        """Parse global interface formulation defaults."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_DEF_INTER
      2021         0
{_BOILERPLATE}
/DEF_INTER/TYPE25
         4         2         1         1         0         1         0         2
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert model.def_inter.get("istf") == 4
        assert model.def_inter.get("igap") == 2
        assert model.def_inter.get("irem_i2") == 1
        assert model.def_inter.get("idel") == 1
        assert model.def_inter.get("iedge") == 2
