"""Tests for Milestone M152:
- EBCS/INIP and EBCS/INIV Eulerian boundary conditions
- Dedicated /PROP readers (INJECT1/2, JOINT, TORSION, SPR_ELAS_PLAS,
  SPR_BEAM, SPOTWELD, BUSHING) — tested via direct reader invocation
  because the prop dispatcher routes them through the generic cfg-driven
  prop_reader (existing M38 tests cover that path).
- /RLINK DOF accessor properties
- /IMPFLUX alias and dict storage
- Element group extended aliases (GRSHELL, GRBRICK, GRTRUSS)
"""
import pytest
from pathlib import Path
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_prop_inject1, read_prop_inject2,
    read_prop_joint, read_prop_torsion,
    read_prop_spring_elas_plas, read_prop_spring_beam,
    read_prop_spotweld, read_prop_bushing,
)
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_starter(tmp_path, text):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _make_block(tmp_path, keyword_text):
    """Create a single keyword block from raw text for direct reader testing."""
    # Wrap in a minimal deck so read_deck can parse it
    full = f"/BEGIN\nTEST\n90\n90\n{keyword_text}\n/END\n"
    p = tmp_path / "DIRECT_0000.rad"
    p.write_text(full, encoding="ascii")
    blocks = read_deck(str(p))
    # Find the target block (skip /BEGIN, /END)
    for b in blocks:
        key0 = b.parts[0].upper() if b.parts else ""
        if key0 not in ("BEGIN", "END"):
            return b
    return None


# ---------------------------------------------------------------------------
# EBCS Tests (routed through dispatcher)
# ---------------------------------------------------------------------------
def test_ebcs_inip(tmp_path):
    text = """/BEGIN
TEST M152
90
90
/EBCS/INIP/123
pressure
         5              2.5              1.2              0.5
/END"""
    model, log = _parse_starter(tmp_path, text)
    assert 123 in model.ebcs_inips
    ent = model.ebcs_inips[123]
    assert ent.title == "pressure"
    assert ent.surf_id == 5
    assert ent.rho == 2.5
    assert ent.c == 1.2
    assert ent.lcar == 0.5

def test_ebcs_iniv(tmp_path):
    text = """/BEGIN
TEST M152
90
90
/EBCS/INIV/456
velocity
         6              3.5              2.2              1.5
/END"""
    model, log = _parse_starter(tmp_path, text)
    assert 456 in model.ebcs_inivs
    ent = model.ebcs_inivs[456]
    assert ent.title == "velocity"
    assert ent.surf_id == 6
    assert ent.rho == 3.5
    assert ent.c == 2.2
    assert ent.lcar == 1.5


# ---------------------------------------------------------------------------
# PROP readers — direct invocation (not dispatcher-routed)
# ---------------------------------------------------------------------------
def test_prop_inject1_direct(tmp_path):
    kw = ("/PROP/INJECT1/789\n"
          "my inject1\n"
          "         2         1                10.5\n"
          "         1         2         3                1.1                 2.2\n"
          "         4         5         6                3.3                 4.4")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_inject1(block, model, log)
    assert 789 in model.prop_inject1s
    ent = model.prop_inject1s[789]
    assert ent.n_gases == 2
    assert ent.iflow == 1
    assert ent.ascale_t == 10.5
    assert len(ent.gases) == 2
    assert ent.gases[0].mat_id == 1
    assert ent.gases[0].fun_id_m == 2
    assert ent.gases[0].fun_id_t == 3
    assert ent.gases[0].fscale_m == 1.1
    assert ent.gases[0].fscale_t == 2.2

def test_prop_inject2_direct(tmp_path):
    kw = ("/PROP/INJECT2/101\n"
          "my inject2\n"
          "         3         1\n"
          "         1         2                 1.1                 2.2                 3.3\n"
          "         4                0.5         5\n"
          "         6                0.2         7\n"
          "         8                0.3         9")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_inject2(block, model, log)
    assert 101 in model.prop_inject2s
    ent = model.prop_inject2s[101]
    assert ent.n_gases == 3
    assert ent.iflow == 1
    assert ent.fun_id_m == 1
    assert ent.fscale_m == 1.1
    assert len(ent.gases) == 3
    assert ent.gases[0].mat_id == 4
    assert ent.gases[0].molar_fraction == 0.5
    assert ent.gases[0].fun_id_mf == 5

def test_prop_joint_direct(tmp_path):
    kw = ("/PROP/JOINT/202\n"
          "my joint\n"
          "         1         2                 1.1                 2.2                 3.3\n"
          "                 4.4                 5.5                 6.6                 7.7                 8.8")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_joint(block, model, log)
    assert 202 in model.prop_joints
    ent = model.prop_joints[202]
    assert ent.joint_type == 1
    assert ent.skew_id == 2
    assert ent.params["p1"] == 1.1
    assert ent.params["p4"] == 4.4

def test_prop_torsion_direct(tmp_path):
    kw = ("/PROP/TORSION/303\n"
          "my torsion\n"
          "                 1.1                 2.2                 3.3                 4.4                 5.5\n"
          "                 6.6                 7.7                 8.8                 9.9\n"
          "         1         2         3         4")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_torsion(block, model, log)
    assert 303 in model.prop_torsions
    ent = model.prop_torsions[303]
    assert ent.mass == 1.1
    assert ent.k_elas == 2.2
    assert ent.d1 == 6.6
    assert ent.fct_id1 == 1
    assert ent.fct_id4 == 4

def test_prop_spr_elas_plas_direct(tmp_path):
    kw = ("/PROP/SPR_ELAS_PLAS/404\n"
          "spr ep\n"
          "         1         2         3         4                 5.5\n"
          "         5                 6.6                 7.7                 8.8                 9.9")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_spring_elas_plas(block, model, log)
    assert 404 in model.prop_spring_elas_plas
    ent = model.prop_spring_elas_plas[404]
    assert ent.skew_id == 1
    assert ent.k_stiff == 5.5
    assert ent.mid1 == 5
    assert ent.area == 6.6
    assert ent.izz == 9.9

def test_prop_spring_beam_direct(tmp_path):
    kw = ("/PROP/SPR_BEAM/505\n"
          "spr beam\n"
          "         1         2         3\n"
          "                 4.4                 5.5                 6.6                 7.7")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_spring_beam(block, model, log)
    assert 505 in model.prop_spring_beams
    ent = model.prop_spring_beams[505]
    assert ent.skew_id == 1
    assert ent.nc_filter == 3
    assert ent.params["p1"] == 4.4
    assert ent.params["p4"] == 7.7

def test_prop_spotweld_direct(tmp_path):
    kw = ("/PROP/SPOTWELD/606\n"
          "spotweld\n"
          "         1                 2.2                 3.3                 4.4         5\n"
          "                 6.6                 7.7                 8.8                 9.9")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_spotweld(block, model, log)
    assert 606 in model.prop_spotwelds
    ent = model.prop_spotwelds[606]
    assert ent.skew_id == 1
    assert ent.knn == 2.2
    assert ent.scf == 4.4
    assert ent.sensor_id == 5
    assert ent.params["p1"] == 6.6

def test_prop_bushing_direct(tmp_path):
    kw = ("/PROP/BUSHING/707\n"
          "bushing\n"
          "                 1.1                 2.2                 3.3                 4.4                 5.5\n"
          "                 6.6         7         8")
    block = _make_block(tmp_path, kw)
    model, log = Model(), MessageLog()
    read_prop_bushing(block, model, log)
    assert 707 in model.prop_bushings
    ent = model.prop_bushings[707]
    assert ent.mass == 1.1
    assert ent.damp == 6.6
    assert ent.epsi == 7
    assert ent.idens == 8


# ---------------------------------------------------------------------------
# RLINK, IMPFLUX, element group aliases
# ---------------------------------------------------------------------------
def test_rlink_dofs(tmp_path):
    text = """/BEGIN
TEST M152
90
90
/RLINK/808
my rlink
011 001         9        10        11
/END"""
    model, log = _parse_starter(tmp_path, text)
    assert 808 in model.rlinks
    ent = model.rlinks[808]
    assert ent.tx == 0
    assert ent.ty == 1
    assert ent.tz == 1
    assert ent.rx == 0
    assert ent.ry == 0
    assert ent.rz == 1
    assert ent.skew_id == 9
    assert ent.grnod_id == 10
    assert ent.ipol == 11

def test_impflux_alias(tmp_path):
    text = """/BEGIN
TEST M152
90
90
/IMPFLUX/909
my impflux
        10        11        12        13
                2.2               3.3               4.4               5.5
/END"""
    model, log = _parse_starter(tmp_path, text)
    assert 909 in model.impfluxes
    ent = model.impfluxes[909]
    assert ent.surf_id == 10
    assert ent.funct_id == 11
    assert ent.sens_id == 12
    assert ent.grbric_id == 13
    assert ent.xscale == 2.2
    assert ent.scale == 3.3
    assert ent.tstart == 4.4
    assert ent.tstop == 5.5

def test_group_aliases(tmp_path):
    text = """/BEGIN
TEST M152
90
90
/GRSHELL/1
my shell group
1 2 3
/GRBRICK/2
my brick group
4 5 6
/GRTRUSS/3
my truss group
7 8 9
/END"""
    model, log = _parse_starter(tmp_path, text)
    assert "SHEL" in model.egroups
    assert 1 in model.egroups["SHEL"]
    assert "BRIC" in model.egroups
    assert 2 in model.egroups["BRIC"]
    assert "TRUS" in model.egroups
    assert 3 in model.egroups["TRUS"]
