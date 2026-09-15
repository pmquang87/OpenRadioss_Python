"""Tests for Milestone M194: SPH Flow, Elements, FAIL/TAB1, LAW40/80/102/NLOCAL, PROP12/13, DEF_INTER/TYPE2, MID/PID directives, Engine TH & Python function keywords."""

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_fail_tab1_parsing(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test FAIL_TAB1
                  10
/FAIL/TAB1/101
         1         0         0         1         0
      0.05       1.0       0.0       0.0       0.0
# Free format
/FAIL/TAB1/102
2, 0, 0, 1, 0
0.08, 1.2, 0.0, 0.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    assert 101 in model.fail_tab1s
    f1 = model.fail_tab1s[101]
    assert f1.ifunc == 1
    assert np.isclose(f1.eps_max, 0.05)
    assert np.isclose(f1.scale, 1.0)

    assert 102 in model.fail_tab1s
    f2 = model.fail_tab1s[102]
    assert f2.ifunc == 2
    assert np.isclose(f2.eps_max, 0.08)
    assert np.isclose(f2.scale, 1.2)


def test_mat_law40_law80_law102_nlocal_parsing(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test Mats M194
                  10
/MAT/LAW40/1
Concrete Subgrade Law40
              7.8e-6                210.                 0.3                 0.0                 0.0
                10.0                 5.0                 2.0                 1.0                 0.5
/MAT/LAW80/2
Phase Transfo Law80
              7.8e-6              7.8e-6
            205000.0                0.29                 301                 1.1              3600.0
/MAT/LAW102/3
Hill48 Law102
              7.8e-6                210.                 0.3                 0.0                 0.0
                 1.5                 0.8                 1.2                 1.0                 1.0
/MAT/NLOCAL/4
NonLocal Regulation
              7.8e-6                210.                 0.3                 0.0                 0.0
                 5.0                 1.0                 0.5
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    assert 1 in model.mat_law40s
    m1 = model.mat_law40s[1]
    assert np.isclose(m1.rho, 7.8e-6)
    assert np.isclose(m1.e, 210.0)

    assert 2 in model.mat_law80s
    m2 = model.mat_law80s[2]
    assert np.isclose(m2.rho0, 7.8e-6)
    assert np.isclose(m2.e, 205000.0)

    assert 3 in model.mat_law102s
    m3 = model.mat_law102s[3]
    assert np.isclose(m3.rho, 7.8e-6)
    assert np.isclose(m3.e, 210.0)

    assert 4 in model.mat_nlocals
    m4 = model.mat_nlocals[4]
    assert np.isclose(m4.rho, 7.8e-6)
    assert np.isclose(m4.e, 210.0)


def test_prop_type12_sph_and_type13_spr_pull(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test Props M194
                  10
/PROP/TYPE34/10
SPH Property
                 0.1                 1.2                 0.0                 0.0                   1
/PROP/SPH/11
SPH Property Alias
                 0.2                 1.5                 0.0                 0.0                   2
/PROP/TYPE13/20
Spring Pull Property
                 1.0                 2.0                 3.0                 0.0                   1
/PROP/SPR_PULL/21
Spring Pull Alias
                 1.5                 2.5                 3.5                 0.0                   2
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    assert 10 in model.prop_sphs
    p10 = model.prop_sphs[10]
    assert np.isclose(p10.mass, 0.1)
    assert np.isclose(p10.h0, 1.2)

    assert 11 in model.prop_sphs
    p11 = model.prop_sphs[11]
    assert np.isclose(p11.mass, 0.2)
    assert np.isclose(p11.h0, 1.5)

    assert 20 in model.prop_spr_pulls
    p20 = model.prop_spr_pulls[20]
    assert np.isclose(p20.stiff, 1.0)
    assert np.isclose(p20.f_max, 2.0)

    assert 21 in model.prop_spr_pulls
    p21 = model.prop_spr_pulls[21]
    assert np.isclose(p21.stiff, 1.5)
    assert np.isclose(p21.f_max, 2.5)


def test_def_inter_type2_and_inter_type2(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test Def Inter Type 2
                  10
/DEF_INTER/TYPE2
                   1                   2                   3
/INTER/TYPE2/100
Tied Interface 100
                   1                   2
                   0                   0                   0                   0
                   1                   2
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    assert model.def_inter_type2 is not None
    assert model.def_inter_type2.istf == 1
    assert model.def_inter_type2.igap == 2
    assert model.def_inter_type2.iref == 3

    assert len(model.interfaces) > 0
    inter = model.interfaces[0]
    assert inter.id == 100
    assert inter.type == 2


def test_sph_flow_and_mid_pid_directives(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test SPH_FLOW, MID, PID
                  10
/SPH_FLOW/1
SPH Inlet Flow
                   5                  10                  15
              1.0e-3                 5.0                10.0
/MID/1
Material ID Redirection
10
/PID/2
Property ID Redirection
20
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    assert 1 in model.sph_flows
    sf = model.sph_flows[1]
    assert sf.surf_id == 5
    assert sf.part_id == 10
    assert sf.fct_id == 15
    assert np.isclose(sf.params["rho"], 1.0e-3)
    assert np.isclose(sf.params["p"], 5.0)
    assert np.isclose(sf.params["e"], 10.0)

    assert 1 in model.mid_directives
    assert model.mid_directives[1].mat_id == 10

    assert 2 in model.pid_directives
    assert model.pid_directives[2].prop_id == 20


def test_element_aliases_parsing(tmp_path: Path):
    deck_str = """# Starter Header
/BEGIN
Test Element Aliases
                  10
/NODE
         1       0.0       0.0       0.0
         2       1.0       0.0       0.0
         3       0.0       1.0       0.0
         4       0.0       0.0       1.0
         5       1.0       1.0       0.0
         6       1.0       0.0       1.0
         7       0.0       1.0       1.0
         8       1.0       1.0       1.0
/SHELL3N/1
         1         1         1         2         3
/SOLIDE/1
         2         1         1         2         5         3         4         6         8         7
/SOLID/1
         3         1         1         2         5         3         4         6         8         7
/TETRA/1
         4         1         1         2         3         4
/SPHCEL/1
         5         1         1
/SPHCELL/1
         6         1         2
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)

    sh3n_ids = [e[0] for e in model.raw_elems["SH3N"]]
    assert 1 in sh3n_ids
    brick_ids = [e[0] for e in model.raw_elems["BRICK"]]
    assert 2 in brick_ids
    assert 3 in brick_ids
    tetra_ids = [e[0] for e in model.raw_elems["TETRA4"]]
    assert 4 in tetra_ids
    sph_ids = [e[0] for e in model.raw_elems["SPH"]]
    assert 5 in sph_ids
    assert 6 in sph_ids


def test_engine_th_and_python_funct_keywords(tmp_path: Path):
    deck_str = """# Engine Deck
/VERS/140
/RUN/RUN1/1
0.1
/TH/CLUSTER/1
CLUSTER TH
1 2
/TH/SH3N/2
SH3N TH
3 4
/TH/SHEL/3
SHEL TH
5 6
/TH/SPH_FLOW/4
SPH FLOW TH
7 8
/FUNCT_PYTHON/10
Python Function Definition
a * x + b
/END
"""
    ec, log = _parse_engine(tmp_path, deck_str)

    assert ec.run_name == "RUN1"
    # Verify TH requests exist in ec.th_records
    th_types = [th.th_type for th in ec.th_records]
    assert "CLUSTER" in th_types
    assert "SH3N" in th_types
    assert "SHEL" in th_types
    assert "SPH_FLOW" in th_types
