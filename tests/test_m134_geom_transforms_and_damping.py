"""Tests for Milestone M134: Extended Geometric Entities, Spatial Transformations & Damping Models
(/LINE extended types, /SURF analytical cylinders & spheres, /TRANSFORM/PROJ, /TRANSFORM/FRAME, /DAMP/GLOBAL, /DAMP/PART).
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m134_extended_line_types(tmp_path):
    deck = """/BEGIN
Test Extended Lines
/LINE/BEAM/1
Line of beams
10 11 12
/LINE/TRUSS/2
Line of trusses
20 21
/LINE/SPRING/3
Line of springs
30 31 32
/LINE/BOX/4
Line from bounding box
5
/LINE/CIRC/5
Circular line
0.0 0.0 0.0 1.5
0.0 0.0 1.0
/LINE/ALL/6
All exterior boundary lines
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.lines
    l1 = model.lines[1]
    assert l1.beam_ids == [10, 11, 12]

    assert 2 in model.lines
    l2 = model.lines[2]
    assert l2.truss_ids == [20, 21]

    assert 3 in model.lines
    l3 = model.lines[3]
    assert l3.spring_ids == [30, 31, 32]

    assert 4 in model.lines
    l4 = model.lines[4]
    assert l4.box_ids == [5]

    assert 5 in model.lines
    l5 = model.lines[5]
    assert l5.circ_radius == pytest.approx(1.5)
    assert np.allclose(l5.circ_center, [0.0, 0.0, 0.0])
    assert np.allclose(l5.circ_axis, [0.0, 0.0, 1.0])

    assert 6 in model.lines
    l6 = model.lines[6]
    assert l6.all_boundary is True


def test_m134_analytical_surfaces(tmp_path):
    deck = """/BEGIN
Test Analytical and Composite Surfaces
/SURF/CYL/1
Cylindrical Surface 1
0 2.5 10.0
0.0 0.0 0.0
0.0 1.0 0.0
/SURF/SPHER/2
Spherical Surface 2
0 5.0
1.0 2.0 3.0
/SURF/SUB/3
Subset Surface 3
10 11 12
/SURF/ALL/4
Full External Surface 4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.surfaces
    s1 = model.surfaces[1]
    assert s1.cyl_radius == pytest.approx(2.5)
    assert s1.cyl_length == pytest.approx(10.0)
    assert np.allclose(s1.cyl_center, [0.0, 0.0, 0.0])
    assert np.allclose(s1.cyl_axis, [0.0, 1.0, 0.0])

    assert 2 in model.surfaces
    s2 = model.surfaces[2]
    assert s2.spher_radius == pytest.approx(5.0)
    assert np.allclose(s2.spher_center, [1.0, 2.0, 3.0])

    assert 3 in model.surfaces
    s3 = model.surfaces[3]
    assert s3.subset_surf_ids == [10, 11, 12]

    assert 4 in model.surfaces
    s4 = model.surfaces[4]
    assert s4.modifier == "ALL"


def test_m134_transform_proj_and_frame(tmp_path):
    deck = """/BEGIN
Test Spatial Transforms
/TRANSFORM/PROJ/1
Project onto surface
100 PLANE 50 0.01
0.0 0.0 1.0
/TRANSFORM/FRAME/2
Transform frame to frame
200 1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.transform_projections) == 1
    tp = model.transform_projections[0]
    assert tp.id == 1
    assert tp.grnod_id == 100
    assert tp.proj_type == "PLANE"
    assert tp.target_id == 50
    assert tp.dist == pytest.approx(0.01)
    assert tp.dir_vector == (0.0, 0.0, 1.0)

    assert len(model.transform_frames) == 1
    tf = model.transform_frames[0]
    assert tf.id == 2
    assert tf.grnod_id == 200
    assert tf.frame_orig == 1
    assert tf.frame_dest == 2


def test_m134_damp_global_and_part(tmp_path):
    deck = """/BEGIN
Test Global and Part Damping
/DAMP/GLOBAL/1
Global Rayleigh damping
0.05 0.001 0.0 10.0
/DAMP/PART/2
Part Specific damping
5 0.1 0.002 0.0 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.damp_globals) == 1
    dg = model.damp_globals[0]
    assert dg.id == 1
    assert dg.alpha == pytest.approx(0.05)
    assert dg.beta == pytest.approx(0.001)
    assert dg.tstart == pytest.approx(0.0)
    assert dg.tstop == pytest.approx(10.0)

    assert 2 in model.damp_parts
    dp = model.damp_parts[2]
    assert dp.id == 2
    assert dp.part_id == 5
    assert dp.alpha == pytest.approx(0.1)
    assert dp.beta == pytest.approx(0.002)
    assert dp.tstart == pytest.approx(0.0)
    assert dp.tstop == pytest.approx(5.0)
