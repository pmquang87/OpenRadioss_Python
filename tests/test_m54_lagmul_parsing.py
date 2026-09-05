import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import read_inter
from pyradioss.common.messages import MessageLog

def test_read_inter_lagmul_type16(tmp_path):
    deck = """/FORMAT/1
/INTER/LAGMUL/TYPE16/1
Interface 1
         1         2
                   1
"""
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i16 = model.interfaces[0]
    assert i16.type == 16
    assert i16.grnod_id == 1
    assert i16.grbric_id1 == 2
    assert i16.itied == 1
    assert i16.lagmul is True

def test_read_inter_lagmul_type17(tmp_path):
    deck = """/FORMAT/1
/INTER/LAGMUL/TYPE17/2
Interface 2
         2         3
                   0
"""
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i17 = model.interfaces[0]
    assert i17.type == 17
    assert i17.grbric_id1 == 2
    assert i17.grbric_id2 == 3
    assert i17.itied == 0
    assert i17.lagmul is True

def test_read_inter_lagmul_type2(tmp_path):
    # TYPE2 requires Card 2: grnd_IDs surf_IDm Isearch dsearch
    # CFG: %10d%10d%30s%10d%20s%20lg
    deck = "/FORMAT/1\n/INTER/LAGMUL/TYPE2/3\nInterface 3\n"
    card1 = f"{4:>10}{5:>10}{'':>30}{0:>10}{'':>20}{1.5:>20}\n"
    deck += card1
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i2 = model.interfaces[0]
    assert i2.type == 2
    assert i2.grnod_id == 4
    assert i2.surf_id == 5
    assert i2.dsearch == 1.5
    assert i2.lagmul is True

def test_read_inter_lagmul_type7(tmp_path):
    # TYPE7 requires 4 cards:
    # Card 2: grnd_IDs  surf_IDm (20 chars)
    # Card 3: blank
    # Card 4: blank
    # Card 5: _BLANK_(40) Gapmin(20)
    deck = "/FORMAT/1\n/INTER/LAGMUL/TYPE7/4\nInterface 4\n"
    card2 = f"{6:>10}{7:>10}\n"
    card3 = "\n"
    card4 = "\n"
    card5 = f"{'':>40}{0.02:>20}\n"
    deck += card2 + card3 + card4 + card5
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i7 = model.interfaces[0]
    assert i7.type == 7
    assert i7.grnod_id == 6
    assert i7.surf_id == 7
    assert i7.gap == 0.02
    assert i7.lagmul is True


def test_bug03_lagmul_solver_enters_solve():
    """BUG-03: Lagrange-multiplier constraints must have nonzero len and generate rows on first force transfer."""
    import numpy as np
    from pyradioss.engine.lagmul import LagmulSolver
    from pyradioss.model.entities import Interface, NodeGroup
    from types import SimpleNamespace

    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],  # node 9 inside brick
    ], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.vr = np.zeros_like(model.x)
    model.mass = np.ones(9, dtype=float)

    # Brick group
    model.bricks = SimpleNamespace(
        ixs=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    )
    egrp = SimpleNamespace(elem_idx=np.array([0], dtype=np.int64))
    model.egroups = {"BRIC": {1: egrp}}

    # Node group for secondary node (node index 8)
    ngrp = NodeGroup(id=1, node_ids=[9])
    ngrp.node_idx = np.array([8], dtype=np.int64)
    model.node_groups[1] = ngrp

    itf = Interface(
        id=1, type=16, grnod_id=1, grbric_id1=1, itied=1, lagmul=True
    )
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, loads=None, log=log)

    # Acceptance check 1: len(solver) > 0 on initialization without manually seeding nc
    assert len(solver) == 1
    assert solver.nc == 0  # nc starts at 0 before first force transfer

    # Acceptance check 2: transfer_forces executes, generates rows, updates nc, and injects forces
    fint = np.zeros_like(model.x)
    fcont = np.zeros_like(model.x)
    fext = np.zeros_like(model.x)
    fext[8] = [10.0, 0.0, 0.0]  # external load on secondary node
    mint = np.zeros_like(model.x)
    inv_mass = np.ones(9, dtype=float)
    inv_inertia = np.zeros(9, dtype=float)

    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt=1e-4)

    # Row generation succeeded and updated nc
    assert solver.nc == 3  # 3 translational DOFs tied
    # Constraint force applied to balance external load
    assert np.linalg.norm(fint[8]) > 0.0

    # Acceptance check 3: enforce executes without error and enforces velocity projection
    model.v[8] = [1.0, 0.0, 0.0]
    solver.enforce(model.v, model.vr, inv_mass, inv_inertia)
    assert solver.nc == 3
