"""Tests for starter checks and LAW4 integration.

Audits and verifies:
1. _ALLOWED_LAWS in pyradioss.starter.checks:
   - bricks has 4
   - tetras has 4
   - shells, beams, trusses do NOT have 4
2. In-memory model with /BRICK element referencing /MAT/LAW4 passes check_model cleanly
   with zero errors and zero warnings.
3. In-memory model with /TETRA element referencing /MAT/LAW4 passes check_model cleanly.
4. Shells / beams referencing /MAT/LAW4 produce expected checker errors.
5. pyradioss.materials dispatch for LAW4:
   - solid_update dispatches LAW4
   - shell_update dispatches LAW4 raising NotImplementedError
   - solid_tangent / consistent_solid_tangent dispatches LAW4
   - extra_shapes defines "temp": ()
   - register_materials registers LAW4 without errors
6. Round-trip deck generation and reading for LAW4.
"""

from __future__ import annotations

import tempfile
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.input import deck_writer as dw
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import Model
from pyradioss.starter import checks
from pyradioss.starter.checks import check_model
from pyradioss.starter.starter import build_element_groups


def test_starter_checks_allowed_laws():
    """Verify _ALLOWED_LAWS has 4 for bricks/tetras and NOT for shells/beams/trusses."""
    assert 4 in checks._ALLOWED_LAWS["bricks"]
    assert 4 in checks._ALLOWED_LAWS["tetras"]

    assert 4 not in checks._ALLOWED_LAWS["shells"]
    assert 4 not in checks._ALLOWED_LAWS["shells_qbat"]
    assert 4 not in checks._ALLOWED_LAWS["shells_qeph"]
    assert 4 not in checks._ALLOWED_LAWS["sh3n"]
    assert 4 not in checks._ALLOWED_LAWS["beams"]
    assert 4 not in checks._ALLOWED_LAWS["trusses"]


def test_in_memory_brick_model_check_model():
    """Build a small model in memory with a /BRICK referencing /MAT/LAW4 and run check_model."""
    model = Model()
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    model.add_nodes(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64), coords)

    mat = Material(
        id=1,
        law=4,
        rho0=7.85e-3,
        title="JCOOK_BRICK",
        params={"E": 210000.0, "nu": 0.3, "A": 250.0, "B": 400.0, "n": 0.4, "T0": 300.0}
    )
    prop = Property(id=1, type=14, title="SOLID_PROP", params={})
    model.materials[1] = mat
    model.properties[1] = prop
    model.parts[1] = Part(id=1, prop_id=1, mat_id=1, title="SOLID_PART")
    model.raw_elems["BRICK"] = [(1, 1, (1, 2, 3, 4, 5, 6, 7, 8))]

    log = MessageLog()
    build_element_groups(model, log)
    check_model(model, log)

    assert len(log.errors) == 0, f"Expected 0 errors, got: {log.errors}"
    assert len(log.warnings) == 0, f"Expected 0 warnings, got: {log.warnings}"


def test_in_memory_tetra_model_check_model():
    """Build a small model in memory with a /TETRA4 referencing /MAT/LAW4 and run check_model."""
    model = Model()
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    model.add_nodes(np.array([1, 2, 3, 4], dtype=np.int64), coords)

    mat = Material(
        id=1,
        law=4,
        rho0=7.85e-3,
        title="JCOOK_TETRA",
        params={"E": 210000.0, "nu": 0.3, "A": 250.0, "B": 400.0, "n": 0.4, "T0": 300.0}
    )
    prop = Property(id=1, type=14, title="SOLID_PROP", params={})
    model.materials[1] = mat
    model.properties[1] = prop
    model.parts[1] = Part(id=1, prop_id=1, mat_id=1, title="SOLID_PART")
    model.raw_elems["TETRA4"] = [(1, 1, (1, 2, 3, 4))]

    log = MessageLog()
    build_element_groups(model, log)
    check_model(model, log)

    assert len(log.errors) == 0, f"Expected 0 errors, got: {log.errors}"
    assert len(log.warnings) == 0, f"Expected 0 warnings, got: {log.warnings}"


def test_in_memory_shell_with_law4_errors():
    """Verify assigning LAW4 to a shell element causes check_model to log an error."""
    model = Model()
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    model.add_nodes(np.array([1, 2, 3, 4], dtype=np.int64), coords)

    mat = Material(
        id=1,
        law=4,
        rho0=7.85e-3,
        title="JCOOK_SHELL",
        params={"E": 210000.0, "nu": 0.3, "A": 250.0, "B": 400.0, "n": 0.4}
    )
    prop = Property(id=1, type=1, title="SHELL_PROP", params={})
    model.materials[1] = mat
    model.properties[1] = prop
    model.parts[1] = Part(id=1, prop_id=1, mat_id=1, title="SHELL_PART")
    model.raw_elems["SHELL"] = [(1, 1, (1, 2, 3, 4))]

    log = MessageLog()
    build_element_groups(model, log)
    check_model(model, log)

    assert len(log.errors) > 0
    assert any("not ported for shells" in err for err in log.errors)


def test_in_memory_beam_with_law4_errors():
    """Verify assigning LAW4 to a beam element causes check_model to log an error."""
    model = Model()
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    model.add_nodes(np.array([1, 2], dtype=np.int64), coords)

    mat = Material(
        id=1,
        law=4,
        rho0=7.85e-3,
        title="JCOOK_BEAM",
        params={"E": 210000.0, "nu": 0.3, "A": 250.0, "B": 400.0, "n": 0.4}
    )
    prop = Property(id=1, type=3, title="BEAM_PROP", params={})
    model.materials[1] = mat
    model.properties[1] = prop
    model.parts[1] = Part(id=1, prop_id=1, mat_id=1, title="BEAM_PART")
    model.raw_elems["BEAM"] = [(1, 1, (1, 2, 0))]

    log = MessageLog()
    build_element_groups(model, log)
    check_model(model, log)

    assert len(log.errors) > 0
    assert any("not ported for beams" in err for err in log.errors)


def test_materials_dispatch_law4():
    """Verify solid_update, shell_update, solid_tangent, extra_shapes, and registration."""
    # 1. Registration
    materials.register_materials()
    assert "LAW4" in MAT_PHYSICS_REGISTRY
    assert "HYD_JCOOK" in MAT_PHYSICS_REGISTRY

    mat = Material(
        id=1,
        law=4,
        rho0=7.85e-3,
        params={
            "E": 210000.0, "nu": 0.3, "A": 250.0, "B": 400.0, "n": 0.4,
            "T0": 300.0, "Tmelt": 1800.0, "rho_cp": 3.5e6
        }
    )

    # 2. extra_shapes defines "temp": ()
    shapes = materials.extra_shapes(mat)
    assert "temp" in shapes
    assert shapes["temp"] == ()

    # 3. solid_update dispatches LAW4
    sig = np.zeros((2, 6))
    deps = np.full((2, 6), 1e-4)
    epsp = np.zeros(2)
    extra = {"temp": np.array([300.0, 300.0])}
    sig_out, epsp_out, c = materials.solid_update(mat, sig, deps, epsp, dt=1e-6, extra=extra)
    assert sig_out.shape == (2, 6)
    assert epsp_out.shape == (2,)
    assert c is not None
    assert np.all(c > 0.0)

    # 4. shell_update dispatches LAW4 raising NotImplementedError
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        materials.shell_update(mat, sig[:, :3], deps[:, :3], epsp, dt=1e-6)

    # 5. solid_tangent dispatches LAW4
    epsp_incr = np.zeros(2)
    tangent = materials.solid_tangent(mat, sig_out, epsp_out, epsp_incr, extra=extra)
    assert tangent.shape == (2, 6, 6)


def test_deck_writer_and_roundtrip_law4():
    """Verify round-trip deck generation and reading works cleanly for LAW4."""
    d = dw.StarterDeck("TEST_LAW4")
    d.node([
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ])
    d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
    d.part(1, "solid_part", prop_id=1, mat_id=1)
    d.prop_solid(1, "solid_prop")
    d.mat_law4(
        1, "jcook_mat", rho=7.85e-3, e=210000.0, nu=0.3,
        a=250.0, b=400.0, n=0.4, eps_max=0.5, sig_max=800.0,
        p_min=-500.0, c=0.05, eps_dot_0=1.0, m=1.0,
        tmelt=1800.0, tmax=2000.0, rhocp=3.5e6, t0=300.0
    )

    rendered = d.render()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".rad") as f:
        f.write(rendered)
        path = f.name

    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(path), model, log)
    build_element_groups(model, log)
    check_model(model, log)

    assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 4
    assert mat.rho0 == pytest.approx(7.85e-3)
    assert mat.params["E"] == pytest.approx(210000.0)
    assert mat.params["nu"] == pytest.approx(0.3)
    assert mat.params["A"] == pytest.approx(250.0)
    assert mat.params["B"] == pytest.approx(400.0)
    assert mat.params["n"] == pytest.approx(0.4)
    assert mat.params["pmin"] == pytest.approx(-500.0)


def test_port_dialect_conversion_law4():
    """Verify write_starter_from_port_lines converts port LAW4 cards to valid fixed deck."""
    port_lines = [
        "/BEGIN",
        "TEST_PORT",
        "2022 0",
        "Mg mm s",
        "Mg mm s",
        "/NODE",
        "1 0.0 0.0 0.0",
        "2 1.0 0.0 0.0",
        "3 1.0 1.0 0.0",
        "4 0.0 1.0 0.0",
        "5 0.0 0.0 1.0",
        "6 1.0 0.0 1.0",
        "7 1.0 1.0 1.0",
        "8 0.0 1.0 1.0",
        "/BRICK/1",
        "1 1 2 3 4 5 6 7 8",
        "/PART/1",
        "solid_part",
        "1 1",
        "/PROP/SOLID/1",
        "solid_prop",
        "1.1 0.05 0.1",
        "/MAT/LAW4/1",
        "jcook_mat",
        "7.85e-3",
        "210000.0 0.3",
        "250.0 400.0 0.4 0.5 800.0",
        "-500.0",
        "0.05 1.0 1.0 1800.0 2000.0",
        "3.5e6 300.0",
        "/END"
    ]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".rad") as f:
        out_path = f.name

    dw.write_starter_from_port_lines(port_lines, out_path, runname="TEST_PORT")

    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(out_path), model, log)
    assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 4
    assert mat.rho0 == pytest.approx(7.85e-3)
    assert mat.params["E"] == pytest.approx(210000.0)
    assert mat.params["A"] == pytest.approx(250.0)
