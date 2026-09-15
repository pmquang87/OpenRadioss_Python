"""Tests for Milestone M113:
SPH Symmetry Boundaries, Madymo Multibody Coupling, ALE Grid Solvers,
Global Adaptivity, Stamping State Import, Stochastic Controls, and Accelerometer Sensors.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M113
      2022         0
/MAT/LAW1/1
Mat_1
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
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/1
All Nodes
         1         2         3         4
/FRAME/FIX/1
Frame 1
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
/SKEW/FIX/1
Skew 1
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
"""


def _run(tmp_path: Path, deck: str, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m113_sphbcs_fixed_and_free(tmp_path):
    fixed_deck = _BOILERPLATE + """\
/SPHBCS/SYM/1
SPH Symmetry Boundary
         X         1         1                   1
/SPHBCS/CYCL/2
SPH Cyclic Boundary
         Y         0         1                   0
/END
"""
    model, log = _run(tmp_path, fixed_deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.sph_bcs
    assert 2 in model.sph_bcs

    sb1 = model.sph_bcs[1]
    assert sb1.bcs_type == "SYM"
    assert sb1.dir == "X"
    assert sb1.frame_id == 1
    assert sb1.grnod_id == 1
    assert sb1.ilevel == 1

    sb2 = model.sph_bcs[2]
    assert sb2.bcs_type == "CYCL"
    assert sb2.dir == "Y"
    assert sb2.frame_id == 0
    assert sb2.grnod_id == 1
    assert sb2.ilevel == 0


def test_m113_madymo_link_and_exfem(tmp_path):
    fixed_deck = _BOILERPLATE + """\
/MADYMO/LINK/1
Madymo Link 1
       101         1
/MADYMO/EXFEM/2
Madymo Exchanged FEM
         1         2
/END
"""
    model, log = _run(tmp_path, fixed_deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.madymo_links
    ml = model.madymo_links[1]
    assert ml.mdref == 101
    assert ml.node_id == 1

    assert 2 in model.madymo_exfems
    me = model.madymo_exfems[2]
    assert me.part_ids == [1, 2]


def test_m113_ale_grid_solvers(tmp_path):
    fixed_deck = _BOILERPLATE + """\
/ALE/GRID/DONEA
                 0.5               100.0                 1.0                 1.0                 1.0
             1.0e-06
/ALE/GRID/SPRING
             1.0e-04                 0.2                 0.5                 1.0
             1.0e-06
/ALE/GRID/STANDARD
                 0.1                 0.2                 0.5                 2.5

/ALE/GRID/DISP
               100.0
             1.0e-06
/ALE/GRID/LAPLACIAN
                 0.1                 0.2                 0.5
/ALE/GRID/VOLUME
                 0.3                 0.4
/END
"""
    model, log = _run(tmp_path, fixed_deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"

    assert model.ale_grid_donea is not None
    assert abs(model.ale_grid_donea.alpha - 0.5) < 1e-6
    assert abs(model.ale_grid_donea.gamma - 100.0) < 1e-6
    assert abs(model.ale_grid_donea.v_min - 1.0e-6) < 1e-12

    assert model.ale_grid_spring is not None
    assert abs(model.ale_grid_spring.dt - 1.0e-4) < 1e-9
    assert abs(model.ale_grid_spring.gamma - 0.2) < 1e-6

    assert model.ale_grid_standard is not None
    assert abs(model.ale_grid_standard.alpha - 0.1) < 1e-6
    assert abs(model.ale_grid_standard.l_c - 2.5) < 1e-6

    assert model.ale_grid_disp is not None
    assert abs(model.ale_grid_disp.u_max - 100.0) < 1e-6
    assert abs(model.ale_grid_disp.v_min - 1.0e-6) < 1e-12

    assert model.ale_grid_laplacian is not None
    assert abs(model.ale_grid_laplacian.alpha - 0.1) < 1e-6
    assert abs(model.ale_grid_laplacian.damp - 0.5) < 1e-6

    assert model.ale_grid_volume is not None
    assert abs(model.ale_grid_volume.alpha - 0.3) < 1e-6
    assert abs(model.ale_grid_volume.gamma - 0.4) < 1e-6


def test_m113_admesh_stamping_random_accel_subset(tmp_path):
    fixed_deck = _BOILERPLATE + """\
/ADMESH/GLOBAL
         3         1                 0.1         0
/STAMPING
SHEET_THICKNESS_DATA_LINE_1
PLASTIC_STRAIN_DATA_LINE_2
/RANDOM/GRNOD/1
                0.05                42.0
/ACCEL/1
Accelerometer Sensor 1
         1         1                    1000.0
/SUBSET/1
Subset Assembly 1
         1         2
/END
"""
    model, log = _run(tmp_path, fixed_deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"

    assert model.admesh_global is not None
    assert model.admesh_global.level_max == 3
    assert model.admesh_global.iadm_rule == 1
    assert abs(model.admesh_global.t_delay - 0.1) < 1e-6

    assert len(model.stamping_inits) == 1
    st = model.stamping_inits[0]
    assert len(st.data_lines) == 2
    assert st.data_lines[0] == "SHEET_THICKNESS_DATA_LINE_1"
    assert st.data_lines[1] == "PLASTIC_STRAIN_DATA_LINE_2"

    assert len(model.random_noises) == 1
    rn = model.random_noises[0]
    assert rn.grnod_id == 1
    assert abs(rn.xalea - 0.05) < 1e-6
    assert abs(rn.seed - 42.0) < 1e-6

    assert 1 in model.accelerometers
    acc = model.accelerometers[1]
    assert acc.node_id == 1
    assert acc.skew_id == 1
    assert abs(acc.cutoff - 1000.0) < 1e-6

    assert 1 in model.subsets
    sub = model.subsets[1]
    assert sub.assembly_ids == [1, 2]


def test_m113_cross_reference_errors(tmp_path):
    bad_deck = _BOILERPLATE + """\
/SPHBCS/SYM/1
Bad SPHBCS
         X       999       888                   0
/MADYMO/LINK/1
Bad Link
         1       999
/MADYMO/EXFEM/2
Bad Exfem
       999
/RANDOM/GRNOD/999
                0.05                42.0
/ACCEL/1
Bad Accel
       999       999                    1000.0
/END
"""
    p = tmp_path / "BAD_0000.rad"
    p.write_text(bad_deck, encoding="ascii")
    log = MessageLog()
    with pytest.raises(Exception):
        run_starter(str(p), log)
    assert len(log.errors) >= 7
    err_text = " ".join(log.errors)
    assert "/SPHBCS/1" in err_text
    assert "/MADYMO/LINK/1" in err_text
    assert "/MADYMO/EXFEM/2" in err_text
    assert "/RANDOM/GRNOD/999" in err_text
    assert "/ACCEL/1" in err_text
