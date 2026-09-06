"""M47 — DKT18 Discrete Kirchhoff Triangle shell element tests.

Tests the Ish3n=2 dispatch from /PROP/SH3N or /PROP/SHELL into model.sh3n_dkt18,
DKT18 initialization (mass, inertia, local frame, char length), forces under
membrane and bending deformation, layer failure and deletion, and an end-to-end
Starter-to-Engine explicit dynamic simulation.
"""

import os
import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_dkt18
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (
    build_element_groups,
    initialize_elements_and_mass,
    resolve_materials,
    resolve_node_groups,
    resolve_surfaces,
)
from pyradioss.starter.starter import run_starter

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

STEEL_LAW2 = """\
/MAT/LAW2/1
steel plastic
7.8e-6
210. 0.3
0.4 0.0 0.0
"""


def _make_dkt18_deck(ish3n=2, fail_card="", mat_card=None):
    mat = mat_card or STEEL_LAW1
    return f"""\
/BEGIN
dkt18 test deck
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 10.0 0.0
4 10.0 10.0 0.0
/SH3N/1
1 1 2 3
2 2 4 3
/PART/1
triangles
1 1
{mat}
/PROP/SHELL/1
dkt18 shell prop
1 0 {ish3n} 0
0.01 0.01 0.01
3 0 1.0
{fail_card}
/END
"""


def _build_model(tmp_path, ish3n=2, fail_card="", mat_card=None, init_mass=False):
    deck = _make_dkt18_deck(ish3n=ish3n, fail_card=fail_card, mat_card=mat_card)
    f = tmp_path / f"deck_{ish3n}.rad"
    f.write_text(deck)
    log = MessageLog()
    model = Model()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    resolve_node_groups(model, log)
    build_element_groups(model, log)
    if init_mass:
        initialize_elements_and_mass(model, log)
    return model, log


def test_dkt18_starter_dispatch(tmp_path):
    """Verify /SH3N with Ish3n=2 is routed to sh3n_dkt18 group."""
    model, _ = _build_model(tmp_path, ish3n=2)

    assert model.sh3n is None
    assert model.sh3n_dkt18 is not None
    assert model.sh3n_dkt18.n == 2
    assert model.sh3n_dkt18.conn.shape == (2, 3)

    # Ish3n=1 or default should stay in standard sh3n
    model_std, _ = _build_model(tmp_path, ish3n=1)

    assert model_std.sh3n is not None
    assert model_std.sh3n_dkt18 is None


def test_dkt18_init_group(tmp_path):
    """Verify DKT18 initialization calculates mass, inertia, and state buffers."""
    model, _ = _build_model(tmp_path, ish3n=2, init_mass=True)

    g = model.sh3n_dkt18
    st = g.state
    # Area of right triangle (0,0)-(10,0)-(0,10) = 0.5 * 10 * 10 = 50
    assert np.allclose(st["area0"], 50.0)
    # Mass = rho * thick * area = 7.8e-6 * 1.0 * 50 = 3.9e-4
    expected_mass = 7.8e-6 * 1.0 * 50.0
    assert np.allclose(st["mass"], expected_mass)
    # Nodal mass scattered: 4 nodes total
    assert np.all(model.mass[:4] > 0.0)
    assert np.isclose(model.mass[:4].sum(), expected_mass * 2.0)
    # Rotational inertia initialized
    assert np.all(model.inertia[:4] > 0.0)
    # State buffers initialized
    assert "sig" in st
    assert "epsp" in st
    assert "off" in st
    assert np.all(st["off"] == 1.0)
    assert np.abs(g.state["sig"]).max() < 1e-15

def test_dkt18_forces_membrane(tmp_path):
    """Verify DKT18 forces under in-plane tensile velocity."""
    model, _ = _build_model(tmp_path, ish3n=2, init_mass=True)

    g = model.sh3n_dkt18
    n = model.numnod
    fint = np.zeros((n, 3))
    mint = np.zeros((n, 3))

    # Pull node 2 in +X direction
    v = np.zeros((n, 3))
    vr = np.zeros((n, 3))
    v[1, 0] = 10.0  # 10 mm/s or unit/s
    dt = 1e-4

    dt_e = shell_dkt18.forces(g, model.x0, v, vr, dt, fint, mint)

    assert len(dt_e) == 2
    assert np.all(dt_e > 0.0)
    assert np.all(dt_e < EP30)
    # Internal forces should resist tension: fint is -f_int, so fint has non-zero values
    assert not np.allclose(fint, 0.0)
    assert g.state["eint"].sum() > 0.0


def test_dkt18_forces_bending(tmp_path):
    """Verify DKT18 forces and moments under out-of-plane bending velocity."""
    model, _ = _build_model(tmp_path, ish3n=2, init_mass=True)

    g = model.sh3n_dkt18
    n = model.numnod
    fint = np.zeros((n, 3))
    mint = np.zeros((n, 3))

    # Apply out-of-plane velocity on node 4 (Z direction)
    v = np.zeros((n, 3))
    vr = np.zeros((n, 3))
    v[3, 2] = 5.0
    dt = 1e-4

    dt_e = shell_dkt18.forces(g, model.x0, v, vr, dt, fint, mint)

    assert len(dt_e) == 2
    assert np.all(dt_e > 0.0)
    # Moments and Z forces should be generated
    assert not np.allclose(mint, 0.0)
    assert not np.allclose(fint[:, 2], 0.0)
    assert g.state["eint"].sum() > 0.0


def test_dkt18_failure_and_deletion(tmp_path):
    """Verify /FAIL/JOHNSON deletes DKT18 element when plastic strain threshold is reached."""
    fail_card = """\
/FAIL/JOHNSON/1
0.001 0.0 0.0 0.0 0.0
"""
    model, _ = _build_model(tmp_path, ish3n=2, fail_card=fail_card, mat_card=STEEL_LAW2, init_mass=True)

    g = model.sh3n_dkt18
    assert g.state["chk_fail"]

    n = model.numnod
    fint = np.zeros((n, 3))
    mint = np.zeros((n, 3))

    # Large strain to exceed 0.001 failure strain
    v = np.zeros((n, 3))
    vr = np.zeros((n, 3))
    v[1, 0] = 1000.0
    v[3, 0] = 1000.0
    dt = 1e-3

    # Force evaluation should cause layer failure and element deletion
    dt_e = shell_dkt18.forces(g, model.x0, v, vr, dt, fint, mint)

    # At least element 0 or 1 should be deleted
    assert (g.state["off"] == 0.0).any()
    dead_idx = np.where(g.state["off"] == 0.0)[0]
    assert np.all(dt_e[dead_idx] == EP30)


def test_dkt18_end_to_end_engine(tmp_path):
    """Run Starter and Engine end-to-end on a deck with DKT18 shells."""
    deck_path = tmp_path / "DKT18_0000.rad"
    deck_content = """\
/BEGIN
DKT18 run test
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 10.0 0.0
4 10.0 10.0 0.0
/SH3N/1
1 1 2 3
2 2 4 3
/PART/1
plate
1 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SHELL/1
dkt18 shell prop
1 0 2 0
0.01 0.01 0.01
3 0 1.0
/BCS/1
fixed_edge
111 111 0 1
/GRNOD/NODE/1
fixed_nodes
1 3
/END
"""
    deck_path.write_text(deck_content)

    engine_deck_path = tmp_path / "DKT18_0001.rad"
    engine_deck_content = """\
/RUN/DKT18/1
0.001
/TFILE
1e-4
/PRINT
-10
/STOP
99.0
/END
"""
    engine_deck_path.write_text(engine_deck_content)

    # Run Starter
    model = run_starter(str(deck_path))
    assert model.sh3n_dkt18 is not None
    assert model.sh3n_dkt18.n == 2

    # Run Engine
    model = run_engine(str(engine_deck_path))
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.cycle > 0
    assert model.engine_state.t >= 0.001 * (1.0 - 1e-12)
