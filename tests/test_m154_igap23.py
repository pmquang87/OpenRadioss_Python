import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.model.entities import Interface
from pyradioss.input.starter_keywords import KeywordBlock, read_inter
from pyradioss.contact.stiffness import segment_mesh_gap, node_mesh_gap
from pyradioss.contact.inter_type7 import ContactType7

class DummyLog:
    def error(self, msg, src): pass
    def warning(self, msg, src): pass
    def info(self, msg): pass

class CardMock(str):
    def __new__(cls, text):
        obj = super().__new__(cls, text)
        obj.raw = text
        return obj
    def ints(self):
        return [int(x) if x.strip() else 0 for x in self.split()]
    def tokens(self):
        return self.split()

def test_igap2_parse():
    model = Model()
    block = KeywordBlock("INTER", ["INTER", "TYPE7", "1"], 1, [])
    block.cards = [
        CardMock("         1         2         0         0         2         0         0         0         0         0"),
        CardMock("                 1.5                 0.0                 0.0                 0.0         0"),
        CardMock("                 0.0                 0.0                 0.0                 0.0         0         0"),
        CardMock("                 0.0                 0.0                 0.0                 0.0                 0.0"),
        CardMock("      0000         0         0         0                 0.0         0                 0.0                 0.0                 0.0"),
        CardMock("         0         0                 0.0         0         0         0                 0.0         0")
    ]
    read_inter(block, model, DummyLog())
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.igap == 2
    assert itf.fscale_gap == 1.5

def test_igap3_parse():
    model = Model()
    block = KeywordBlock("INTER", ["INTER", "TYPE7", "1"], 1, [])
    block.cards = [
        CardMock("         1         2         0         0         3         0         0         0         0         0"),
        CardMock("                 0.0                 0.0                 0.0                 0.0         0"),
        CardMock("                 0.0                 0.0                 0.6                 0.0         0         0"),
        CardMock("                 0.0                 0.0                 0.0                 0.0                 0.0"),
        CardMock("      0000         0         0         0                 0.0         0                 0.0                 0.0                 0.0"),
        CardMock("         0         0                 0.0         0         0         0                 0.0         0")
    ]
    read_inter(block, model, DummyLog())
    itf = model.interfaces[0]
    assert itf.igap == 3
    assert itf.percent_mesh_size == 0.6
    assert itf.fscale_gap == 1.0

def test_igap2_no_error():
    model = Model()
    itf = Interface(id=1, type=7, igap=2, fscale_gap=1.5)
    model.interfaces.append(itf)
    model.x0 = np.zeros((4, 3))
    model.node_ids = np.array([1, 2, 3, 4])
    model.mass = np.ones(4)
    model.mass0 = np.ones(4)
    model.surfaces = {0: type('Surf', (), {'segments': np.array([[0,1,2,3]]), 'seg_gtype': np.array(['']), 'seg_elem': np.array([0])})}
    model.node_groups = {0: type('Group', (), {'node_idx': np.array([0])})}
    model.element_groups = lambda: []
    
    c7 = ContactType7(itf, model, DummyLog())
    assert c7.itf.igap == 2

def test_igap3_no_error():
    model = Model()
    itf = Interface(id=1, type=7, igap=3, percent_mesh_size=0.6)
    model.interfaces.append(itf)
    model.x0 = np.zeros((4, 3))
    model.node_ids = np.array([1, 2, 3, 4])
    model.mass = np.ones(4)
    model.mass0 = np.ones(4)
    model.surfaces = {0: type('Surf', (), {'segments': np.array([[0,1,2,3]]), 'seg_gtype': np.array(['']), 'seg_elem': np.array([0])})}
    model.node_groups = {0: type('Group', (), {'node_idx': np.array([0])})}
    model.element_groups = lambda: []
    
    c7 = ContactType7(itf, model, DummyLog())
    assert c7.itf.igap == 3

def test_segment_mesh_gap():
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    segs = np.array([[0, 1, 2, 2]])
    g_m_l = segment_mesh_gap(model, segs, percent_mesh_size=0.5)
    assert np.allclose(g_m_l, [0.5])

def test_igap2_gap_scaled():
    model = Model()
    itf = Interface(id=1, type=7, igap=2, fscale_gap=2.0)
    model.interfaces.append(itf)
    model.x0 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    model.node_ids = np.array([1, 2, 3, 4])
    model.mass = np.ones(4)
    model.mass0 = np.ones(4)
    model.surfaces = {0: type('Surf', (), {'segments': np.array([[0,1,2,2]]), 'seg_gtype': np.array(['']), 'seg_elem': np.array([0])})}
    model.node_groups = {0: type('Group', (), {'node_idx': np.array([3])})}
    model.element_groups = lambda: []
    
    c7 = ContactType7(itf, model, DummyLog())
    c7.gap_s = np.array([1.0])
    c7.gap_m = np.array([2.0])
    c7.gap_min = 0.0
    c7.gap_max = 10.0
    c7.nodes = np.array([3])
    
    ni = np.array([3])
    loc = np.searchsorted(c7.nodes, ni)
    gap = c7.gap_s[loc] + c7.gap_m[0]
    gap = np.clip(gap, c7.gap_min, c7.gap_max)
    assert gap[0] == 3.0

def test_igap3_gap_capped():
    model = Model()
    itf = Interface(id=1, type=7, igap=3)
    model.interfaces.append(itf)
    model.x0 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    model.node_ids = np.array([1, 2, 3, 4])
    model.mass = np.ones(4)
    model.mass0 = np.ones(4)
    model.surfaces = {0: type('Surf', (), {'segments': np.array([[0,1,2,2]]), 'seg_gtype': np.array(['']), 'seg_elem': np.array([0])})}
    model.node_groups = {0: type('Group', (), {'node_idx': np.array([3])})}
    model.element_groups = lambda: []
    
    c7 = ContactType7(itf, model, DummyLog())
    c7.gap_s = np.array([5.0])
    c7.gap_m = np.array([5.0])
    c7.gap_s_l = np.array([1.0])
    c7.gap_m_l = np.array([1.0])
    c7.gap_min = 0.0
    c7.gap_max = 100.0
    c7.nodes = np.array([3])
    
    ni = np.array([3])
    loc = np.searchsorted(c7.nodes, ni)
    gap = c7.gap_s[loc] + c7.gap_m[0]
    mesh_gap = c7.gap_s_l[loc] + c7.gap_m_l[0]
    gap = np.minimum(gap, mesh_gap)
    gap = np.clip(gap, c7.gap_min, c7.gap_max)
    assert gap[0] == 2.0
