import numpy as np
import pytest

from pyradioss.input.deck_reader import KeywordBlock, Card
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.model import Model
from pyradioss.contact.inter_type10 import ContactType10
from pyradioss.common.messages import MessageLog

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

def test_inter_type10_engine_init(monkeypatch):
    """Verify ContactType10 initializes properly."""
    model = Model()
    log = MessageLog()
    
    # Mock node_stiffness_gap and segment_stiffness_gap to prevent crashes on dummy model
    monkeypatch.setattr("pyradioss.contact.inter_type7.node_stiffness_gap", lambda m, s: (np.zeros(10), np.zeros(10)))
    monkeypatch.setattr("pyradioss.contact.inter_type7.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.zeros(len(s)), np.zeros(len(s))))
    
    # Add fake interface
    cards = [
        Card("Tied"),
        Card("21 22 5 2"),
        Card("0.5 0.1 0.0 1e30"),
        Card("0 1 0.15 0.2")
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "11"], 11, cards, fixed=False)
    read_inter(block, model, log)
    
    # Dummy groups to prevent init crash
    from pyradioss.model.entities import NodeGroup, Surface
    model.node_groups[21] = NodeGroup(id=21, node_idx=[0, 1])
    model.surfaces[22] = Surface(
        id=22, segments=np.array([[2, 3, 4, 5]]),
        seg_gtype=np.array([""]), seg_elem=np.array([0])
    )
    
    # Needs valid coordinates for segments to compute initial area/stiffness
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    model.mass = np.ones(6)
    model.mass0 = np.ones(6)
    
    engine_itf = ContactType10(model.interfaces[0], model, log)
    assert engine_itf.gap_min == 0.1
    
    # Dummy step
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    v = np.zeros_like(x)
    mass = np.ones(6)
    fcont = np.zeros_like(x)
    
    # Run a step (should broad phase but find nothing within 0.1 since z=1.0 for node 1
    # and z=0.0 for segment. Wait, node 0 is at 0,0,0, segment is at z=0.
    # It might hit node 0!
    # Let's move node 0 away so it doesn't hit yet.
    x[0, 2] = 2.0
    engine_itf.forces(x, v, mass, 1e-4, fcont, 1)
    assert len(engine_itf.tied_state) == 0
