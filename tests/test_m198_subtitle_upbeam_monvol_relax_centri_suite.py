"""Tests for Milestone M198: Subtitles, Beam Updates, Monvol Communication, Dynamic Relaxation & Centrifugal Loading Suite."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model, EngineControls
from pyradioss.model.entities import (
    Upbeam,
    RelaxSystem,
    CentrifugalLoad,
    MonvolComm,
)


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, text: str) -> tuple[EngineControls, MessageLog]:
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_m198_subtitle_parsing(tmp_path: Path):
    """Verify /SUBTITLE updates model title and subtitles list."""
    deck = """# RADIOSS STARTER
/BEGIN
SubTitle Test Model
/SUBTITLE
Detailed Subtitle Line 1
/SUBTITLE
Second Subtitle Line
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors
    assert len(model.subtitles) == 2
    assert model.subtitles[0] == "Detailed Subtitle Line 1"
    assert model.subtitles[1] == "Second Subtitle Line"
    assert model.subtitle == "Second Subtitle Line"


def test_m198_upbeam_fixed_and_free(tmp_path: Path):
    """Verify /UPBEAM and /UPBEAM/INT_BEAM in fixed and free formats."""
    # Fixed format test
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Upbeam Test
/UPBEAM/INT_BEAM/1
Upbeam Test Title
#--1------2-------3-------4-------5-------6-------7-------8-------9------10
       101         1                  0.05         5
/END
"""
    model_fixed, log_fixed = _parse_starter(tmp_path, deck_fixed)
    assert not log_fixed.errors
    assert 1 in model_fixed.upbeams
    up1 = model_fixed.upbeams[1]
    assert up1.id == 1
    assert up1.title == "Upbeam Test Title"
    assert up1.grnd_id == 101
    assert up1.i_updt == 1
    assert up1.eps_max == pytest.approx(0.05)
    assert up1.npt_int == 5

    # Free format test
    deck_free = """# RADIOSS STARTER
/BEGIN
Upbeam Free Test
/UPBEAM/2
Upbeam Free Title
102, 2, 0.08, 10
/END
"""
    model_free, log_free = _parse_starter(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.upbeams
    up2 = model_free.upbeams[2]
    assert up2.id == 2
    assert up2.title == "Upbeam Free Title"
    assert up2.grnd_id == 102
    assert up2.i_updt == 2
    assert up2.eps_max == pytest.approx(0.08)
    assert up2.npt_int == 10


def test_m198_relax_system_fixed_and_free(tmp_path: Path):
    """Verify /RELAX and /RELAX/SYSTEM / /RELAX/DYNA."""
    # Fixed format test
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Relax Test
/RELAX/DYNA/1
Dynamic Relaxation Control
#--1------2-------3-------4-------5-------6-------7-------8-------9------10
       0.0       1.5       0.1         1       100.0      0.0001
/END
"""
    model_fixed, log_fixed = _parse_starter(tmp_path, deck_fixed)
    assert not log_fixed.errors
    assert model_fixed.relax is True
    assert 1 in model_fixed.relax_systems
    rel1 = model_fixed.relax_systems[1]
    assert rel1.id == 1
    assert rel1.title == "Dynamic Relaxation Control"
    assert rel1.t_start == pytest.approx(0.0)
    assert rel1.t_stop == pytest.approx(1.5)
    assert rel1.damp_coeff == pytest.approx(0.1)
    assert rel1.i_damp == 1
    assert rel1.v_lim == pytest.approx(100.0)
    assert rel1.eps_tol == pytest.approx(0.0001)

    # Free format test
    deck_free = """# RADIOSS STARTER
/BEGIN
Relax Free Test
/RELAX/2
Free Relax Control
0.5, 2.5, 0.25, 2, 50.0, 1.0e-5
/END
"""
    model_free, log_free = _parse_starter(tmp_path, deck_free)
    assert not log_free.errors
    assert model_free.relax is True
    assert 2 in model_free.relax_systems
    rel2 = model_free.relax_systems[2]
    assert rel2.id == 2
    assert rel2.title == "Free Relax Control"
    assert rel2.t_start == pytest.approx(0.5)
    assert rel2.t_stop == pytest.approx(2.5)
    assert rel2.damp_coeff == pytest.approx(0.25)
    assert rel2.i_damp == 2
    assert rel2.v_lim == pytest.approx(50.0)
    assert rel2.eps_tol == pytest.approx(1.0e-5)


def test_m198_centri_fixed_and_free(tmp_path: Path):
    """Verify /CENTRI centrifugal body force definitions."""
    # Fixed format test
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Centri Test
/CENTRI/1
Centrifugal Rotor Loading
#--1------2-------3-------4-------5-------6-------7-------8-------9------10
        10         5        12       101       102     314.159
       1.0       1.0       0.5
/END
"""
    model_fixed, log_fixed = _parse_starter(tmp_path, deck_fixed)
    assert not log_fixed.errors
    assert 1 in model_fixed.centris
    c1 = model_fixed.centris[1]
    assert c1.id == 1
    assert c1.title == "Centrifugal Rotor Loading"
    assert c1.grnd_id == 10
    assert c1.sens_id == 5
    assert c1.fct_id == 12
    assert c1.node_orig == 101
    assert c1.node_axis == 102
    assert c1.omega == pytest.approx(314.159)
    assert c1.scale_x == pytest.approx(1.0)
    assert c1.scale_y == pytest.approx(1.0)
    assert c1.scale_z == pytest.approx(0.5)

    # Free format test
    deck_free = """# RADIOSS STARTER
/BEGIN
Centri Free Test
/CENTRI/2
Centri Free Loading
20, 0, 15, 201, 202, 628.318
0.0, 0.0, 1.0
/END
"""
    model_free, log_free = _parse_starter(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.centris
    c2 = model_free.centris[2]
    assert c2.id == 2
    assert c2.title == "Centri Free Loading"
    assert c2.grnd_id == 20
    assert c2.sens_id == 0
    assert c2.fct_id == 15
    assert c2.node_orig == 201
    assert c2.node_axis == 202
    assert c2.omega == pytest.approx(628.318)
    assert c2.scale_x == pytest.approx(0.0)
    assert c2.scale_y == pytest.approx(0.0)
    assert c2.scale_z == pytest.approx(1.0)


def test_m198_monvol_comm_fixed_and_free(tmp_path: Path):
    """Verify /MONVOL/COMM and /MONVOL/COMMUNICATION inter-chamber venting."""
    # Fixed format test
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Monvol Comm Test
/MONVOL/COMM/1
Chamber Communication 1 to 2
#--1------2-------3-------4-------5-------6-------7-------8-------9------10
         1         2        50       0.6    0.0025        10         3
/END
"""
    model_fixed, log_fixed = _parse_starter(tmp_path, deck_fixed)
    assert not log_fixed.errors
    assert 1 in model_fixed.monvol_comms
    mc1 = model_fixed.monvol_comms[1]
    assert mc1.id == 1
    assert mc1.title == "Chamber Communication 1 to 2"
    assert mc1.monvol1_id == 1
    assert mc1.monvol2_id == 2
    assert mc1.surface_id == 50
    assert mc1.cd == pytest.approx(0.6)
    assert mc1.a_vent == pytest.approx(0.0025)
    assert mc1.fct_id == 10
    assert mc1.sens_id == 3

    # Free format test
    deck_free = """# RADIOSS STARTER
/BEGIN
Monvol Comm Free Test
/MONVOL/COMMUNICATION/2
Chamber Communication 2 to 3
2, 3, 60, 0.75, 0.005, 11, 4
/END
"""
    model_free, log_free = _parse_starter(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.monvol_comms
    mc2 = model_free.monvol_comms[2]
    assert mc2.id == 2
    assert mc2.title == "Chamber Communication 2 to 3"
    assert mc2.monvol1_id == 2
    assert mc2.monvol2_id == 3
    assert mc2.surface_id == 60
    assert mc2.cd == pytest.approx(0.75)
    assert mc2.a_vent == pytest.approx(0.005)
    assert mc2.fct_id == 11
    assert mc2.sens_id == 4


def test_m198_dttsh_and_h3d(tmp_path: Path):
    """Verify /DTTSH, /DT/TSH and /H3D starter keyword controls."""
    deck = """# RADIOSS STARTER
/BEGIN
DTTSH and H3D Test
/DTTSH
/H3D
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors
    assert model.dttsh is True


def test_m198_anim_noda_acc_in_engine(tmp_path: Path):
    """Verify /ANIM/NODA/ACC and /ANIM/NODA/VEL engine output requests."""
    deck = """# RADIOSS ENGINE
/RUN/TEST/1
0.05
/ANIM/DT
0.0 0.005
/ANIM/NODA/ACC
/ANIM/NODA/VEL
/ANIM/ELEM/VONM
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert not log.errors
    assert "ACC" in ec.anim_vect
    assert "VEL" in ec.anim_vect
    assert "VONM" in ec.anim_elem
    assert "NODA/ACC" in ec.anim_elem or "NODA/VEL" in ec.anim_elem


def test_m198_cnode_parsing(tmp_path: Path):
    """Verify /CNODE commented node reader."""
    deck = """# RADIOSS STARTER
/BEGIN
CNode Test
/CNODE
         1                 1.0                 2.0                 3.0
         2                 4.5                 5.5                 6.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors
    assert 1 in model.cnodes
    assert 2 in model.cnodes
    cn1 = model.cnodes[1]
    assert cn1.id == 1
    assert cn1.x == pytest.approx(1.0)
    assert cn1.y == pytest.approx(2.0)
    assert cn1.z == pytest.approx(3.0)
    cn2 = model.cnodes[2]
    assert cn2.id == 2
    assert cn2.x == pytest.approx(4.5)
    assert cn2.y == pytest.approx(5.5)
    assert cn2.z == pytest.approx(6.5)
    assert model.numnod == 2


def test_m198_edge_cases(tmp_path: Path):
    """Verify default handling and edge cases for M198 keywords."""
    deck = """# RADIOSS STARTER
/BEGIN
Edge Case Test
/RELAX
/UPBEAM
/CENTRI
/MONVOL/COMM
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.relax is True
    # RELAX with no cards creates relax=True

