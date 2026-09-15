"""Tests for Milestone M155: Real-Deck Parser & Starter Robustness.

- /MAT/LAW42: Ogden stability check based on net ground-state shear modulus GS = sum(mu*alpha) > 0
- /FAIL/BIQUAD: M_Flag material preset defaults and c3 fallback
- /INTER/TYPE11: Dedicated fixed-format reader (radioss51, 110, 120, 140+)
- /RBODY: Massless rigid body warning and 1e-20 floor
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import RigidBody
from pyradioss.starter.initialization import initialize_rigid_bodies


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ---------------------------------------------------------------------------
# LAW42 Ogden material tests
# ---------------------------------------------------------------------------

def test_law42_mixed_pairs_with_positive_shear_modulus(tmp_path):
    """Ogden model with mixed positive/negative mu/alpha where sum(mu*alpha) > 0.
    Matches real deck rubber_tension_v1 (RD-E-5600)."""
    text = """/BEGIN
LAW42_TEST
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/OGDEN/1/1
rubber LAW42
                1E-9
               .4997                   0                   0                   0         0         0
    -0.23974             -11.5758              11.5748                   0                   0

     -4.5483               5.7141               5.7141                   0                   0

/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 42
    # G0 = ( -0.23974 * -4.5483 + -11.5758 * 5.7141 + 11.5748 * 5.7141 ) / 2 > 0
    assert mat.params["E"] > 0.0


def test_law42_negative_shear_modulus_rejected(tmp_path):
    """Ogden model with net sum(mu*alpha) <= 0 must be rejected."""
    text = """/BEGIN
LAW42_FAIL
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/OGDEN/1/1
rubber bad
                1E-9
               .4997                   0                   0                   0         0         0
   -10.0                               0                   0                   0                   0

      5.0                              0                   0                   0                   0

/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 1
    assert "material stability" in log.errors[0]


# ---------------------------------------------------------------------------
# FAIL/BIQUAD preset tests
# ---------------------------------------------------------------------------

def test_fail_biquad_preset_defaults(tmp_path):
    """BIQUAD with only c3 given uses Mild Steel (M_flag=1) preset."""
    text = """/BEGIN
BIQUAD_TEST
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/FAIL/BIQUAD/1
                 0.0                 0.0                 0.5                 0.0                 0.0
                 0.0                   1                   2
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 1
    assert fm.type == "BIQUAD"
    # c1 = 3.5 * c3 = 1.75, c2 = 1.6 * c3 = 0.8, c4 = 0.6 * c3 = 0.3, c5 = 1.5 * c3 = 0.75
    assert np.isclose(fm.params["c1"], 1.75)
    assert np.isclose(fm.params["c2"], 0.8)
    assert np.isclose(fm.params["c3"], 0.5)
    assert np.isclose(fm.params["c4"], 0.3)
    assert np.isclose(fm.params["c5"], 0.75)


def test_fail_biquad_omitted_all_c_defaults(tmp_path):
    """BIQUAD with card 1 all zeros defaults c3=0.6 and M_flag=1."""
    text = """/BEGIN
BIQUAD_EMPTY
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/FAIL/BIQUAD/1
                 0.0                 0.0                 0.0                 0.0                 0.0
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 1
    assert np.isclose(fm.params["c3"], 0.6)
    assert np.isclose(fm.params["c1"], 3.5 * 0.6)


# ---------------------------------------------------------------------------
# /INTER/TYPE11 fixed format tests
# ---------------------------------------------------------------------------

def test_inter_type11_fixed_format_reading(tmp_path):
    """TYPE11 contact with fixed format (radioss120/140 card layout)."""
    text = """/BEGIN
TYPE11_FIXED
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/INTER/TYPE11/1
Line to Line Contact
# line_IDs  line_IDm      Istf      Ithe      Igap                          Idel
         1         2         2         0         1                             1
#              Stmin               Stmax   Percent_mesh_size               dtmin     Iform   sens_ID
                 0.0                 0.0                 0.3                 0.0         0         5
#              Stfac                Fric              GAPmin              Tstart               Tstop
                 1.2                 0.1                0.05                 0.0                 0.0
#      IBC                        Inacti               VIS_S               VIS_F              Bumult
                                       0                 0.0                 0.0                 0.0
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.type == 11
    assert itf.line_id1 == 1
    assert itf.line_id2 == 2
    assert itf.istf == 2
    assert itf.igap == 1
    assert itf.sens_id == 5
    assert np.isclose(itf.percent_mesh_size, 0.3)
    assert np.isclose(itf.stfac, 1.2)
    assert np.isclose(itf.fric, 0.1)
    assert np.isclose(itf.gap, 0.05)


# ---------------------------------------------------------------------------
# /RBODY mass flooring test
# ---------------------------------------------------------------------------

def test_rbody_massless_floored_with_warning():
    """RBODY on massless nodes floors to 1e-20 with a warning instead of error."""
    model = Model()
    model.node_ids = np.array([1, 2, 3])
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model.x = model.x0.copy()
    model.mass = np.zeros(3)
    model.inertia = np.zeros(3)
    model.node_groups = {10: type('Group', (), {'node_idx': np.array([1, 2])})}

    rb = RigidBody(id=1, kind="RBODY", master_id=1, grnod_id=10, added_mass=0.0)
    model.rbodies.append(rb)

    log = MessageLog()
    initialize_rigid_bodies(model, log)

    assert len(log.errors) == 0
    assert any("floored to 1e-20" in w for w in log.warnings)
    assert rb.added_mass == 1e-20
