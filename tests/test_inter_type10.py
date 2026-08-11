import numpy as np
import pytest

from pyradioss.input.deck_reader import KeywordBlock, Card
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.model import Model
from pyradioss.contact.inter_type10 import ContactType10
from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import NodeGroup, Surface

def test_inter_type10_parsing():
    """Verify parsing of /INTER/TYPE10 free and fixed formats."""
    model = Model()
    log = MessageLog()
    
    cards = [
        Card("Tied Penalty Contact"),
        Card(f"{21:>10}{22:>10}{'':>30}{5:>10}{'':>10}{2:>10}"),
        Card(f"{0.5:>20}{'':>20}{0.1:>20}{0.0:>20}{1e30:>20}"),
        Card(f"{'':>20}{0:>10}{1:>10}{0.15:>20}{'':>20}{0.2:>20}")
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "11"], 11, cards, fixed=True)
    
    read_inter(block, model, log)
    
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.type == 10
    assert itf.grnod_id == 21
    assert itf.surf_id == 22
    assert itf.multimp == 5
    assert itf.idel10 == 2
    assert itf.stfac == 0.5
    assert itf.gap == 0.1
    assert itf.tstart == 0.0
    assert itf.tstop == 1e30
    assert itf.itied == 0
    assert itf.inactiv == 1
    assert itf.stiff_dc == 0.15
    assert itf.sort_fact == 0.2

def test_inter_type10_physics(monkeypatch):
    """Verify mathematical integration of ContactType10 incremental force."""
    model = Model()
    log = MessageLog()
    
    # Mock stiffness to return constant 1000.0 for easier math
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(10, 1000.0), np.zeros(10)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))
    
    # 1. Create a TYPE10 interface (tied permanently Itied = 1)
    cards = [
        Card("Tied"),
        Card("21 22 5 2"),
        Card("1.0 0.2 0.0 1e30"),
        Card("1 0 0.0 0.2") # Itied = 1, inactiv = 0, visc = 0
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "11"], 11, cards, fixed=False)
    read_inter(block, model, log)
    
    # 2. Setup the model
    # Node 0 is the slave node
    model.node_groups[21] = NodeGroup(id=21, node_idx=[0])
    
    # Node 1, 2, 3, 4 forms a single quad segment on the XY plane
    model.surfaces[22] = Surface(
        id=22, segments=np.array([[1, 2, 3, 4]]),
        seg_gtype=np.array([""]), seg_elem=np.array([0])
    )
    
    # Coordinates
    x = np.array([
        [0.5, 0.5, 0.1],   # node 0 (slave): directly above center of segment, gap = 0.1 (within 0.2 gap)
        [0.0, 0.0, 0.0],   # node 1
        [1.0, 0.0, 0.0],   # node 2
        [1.0, 1.0, 0.0],   # node 3
        [0.0, 1.0, 0.0]    # node 4
    ], dtype=np.float64)
    model.x0 = x.copy()
    
    mass = np.ones(5)
    model.mass = mass
    model.mass0 = mass
    
    # Initialize Engine Interface
    engine_itf = ContactType10(model.interfaces[0], model, log)
    assert engine_itf.gap_bound == 0.2
    
    v = np.zeros_like(x)
    fcont = np.zeros_like(x)
    
    # Cycle 0: vrel = 0, no force should develop yet, but it should tie!
    engine_itf.forces(x, v, mass, fcont, cycle=0, dt=1e-4, t=0.0)
    assert len(engine_itf._hist_keys) == 1 # Tied!
    assert np.allclose(fcont, 0.0)
    
    # Cycle 1: slave node moving UP with v = 10.0
    v[0, 2] = 10.0
    
    # Expected: V_rel = [0, 0, 10.0]
    # F_new = F_old - V_rel * dt * K
    # F_new = 0 - [0, 0, 10] * 1e-4 * 1000 = [0, 0, -1.0] on slave node
    # The penalty acts downwards on slave to pull it back!
    dt = 1e-4
    
    fcont = np.zeros_like(x)
    engine_itf.forces(x, v, mass, fcont, cycle=1, dt=dt, t=1e-4)
    
    # Slave node force
    assert np.allclose(fcont[0], [0.0, 0.0, -1.0])
    
    # Segment nodes should get the opposite force distributed by shape functions.
    # The quad is split into two triangles: (0, 1, 2) and (0, 2, 3).
    # The diagonal is between local node 0 and local node 2 (which are global nodes 1 and 3).
    # The point (0.5, 0.5) is exactly on this diagonal. 
    # Therefore, the weights are 0.5 for node 1, 0.5 for node 3, and 0.0 for nodes 2 and 4.
    assert np.allclose(fcont[1], [0.0, 0.0, 0.5])
    assert np.allclose(fcont[2], [0.0, 0.0, 0.0])
    assert np.allclose(fcont[3], [0.0, 0.0, 0.5])
    assert np.allclose(fcont[4], [0.0, 0.0, 0.0])
    
    # Cycle 2: Move node further, v = 10.0 still
    fcont = np.zeros_like(x)
    engine_itf.forces(x, v, mass, fcont, cycle=2, dt=dt, t=2e-4)
    # Force accumulates! F = -1.0 - 1.0 = -2.0
    assert np.allclose(fcont[0], [0.0, 0.0, -2.0])
    
    # Test Itied = 0 (Rebound allowed)
    engine_itf.itf.itied = 0
    
    # Cycle 3: Slave node moves DOWN with v = -30.0
    v[0, 2] = -30.0
    # Current Fn (normal pointing towards master)
    # Master normal is +Z. So F_n = F_vec \cdot (0, 0, 1) = -2.0. (Pulling)
    # New V_rel = -30.0.
    # dF = - (-30) * 1e-4 * 1000 = +3.0.
    # F_new = -2.0 + 3.0 = +1.0.
    # Wait, master normal points from master to slave!
    # Slave is at z=0.1, master at z=0. d_vec = slave - master = (0,0,0.1). nvec = (0,0,1).
    # fn = fvec \cdot nvec.
    # old fvec = (0,0,-2). fn_old = -2.0.
    # new fvec = (0,0, 1). fn_new = 1.0.
    # Since fn_old * fn_new < 0 and pen < 0 (it's at gap 0.1, best_d 0.1, pen = 0.2-0.1 = 0.1 > 0 -> Wait! pen > 0 means IT IS PENETRATING the gap margin!)
    # Actually, in penalty contacts, pen > 0 means the gap is breached.
    # If it is penetrating, it shouldn't rebound! It rebounds only if pen <= 0.
    # Let's set node position so pen <= 0 (best_d >= 0.2)
    x[0, 2] = 0.3
    # Now best_d = 0.3. gap_bound = 0.2. pen = -0.1 (not penetrating the gap)
    fcont = np.zeros_like(x)
    engine_itf.forces(x, v, mass, fcont, cycle=3, dt=dt, t=3e-4)
    
    # Since pen <= 0 and force changed sign, it REBOUNDS! Force drops to 0!
    assert np.allclose(fcont[0], 0.0)
    assert len(engine_itf._hist_keys) == 0 # Untied!
