"""Unit tests for Milestone M580: /DT/<ELEM>/DEL element time-step erosion.

OpenRadioss reference: engine/source/time_step/dtchk.F, dtbric.F, dtshell.F.
When an element's critical time step drops below dt_min in /DT/<elem>/DEL,
the element is eroded (off=0.0) so it stops constraining the global time step.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.model import Model, ElementGroup, EngineControls
from pyradioss.model.entities import Part, Property, Material
from pyradioss.engine.engine import _integrate


def _build_two_brick_model():
    """Build a model with two Hexa8 brick elements, one normal and one compressed/small."""
    m = Model()
    m.title = "TEST_DT_BRICK_DEL"
    
    # 12 nodes: elements 1 and 2 share 4 nodes
    # Element 1: normal unit cube [0, 1] x [0, 1] x [0, 1]
    # Element 2: very thin/compressed element [1, 1.0001] x [0, 1] x [0, 1]
    nodes_x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0001, 0.0, 0.0],  # node 9
        [1.0001, 1.0, 0.0],  # node 10
        [1.0001, 0.0, 1.0],  # node 11
        [1.0001, 1.0, 1.0],  # node 12
    ], dtype=np.float64)
    
    m.node_ids = np.arange(1, len(nodes_x) + 1, dtype=np.int64)
    m.x0 = nodes_x.copy()
    m.x = nodes_x.copy()
    m.v = np.zeros((m.numnod, 3), dtype=np.float64)
    m.vr = np.zeros((m.numnod, 3), dtype=np.float64)
    m.a = np.zeros((m.numnod, 3), dtype=np.float64)
    m.mass = np.ones(m.numnod, dtype=np.float64) * 0.1
    m.mass0 = m.mass.copy()
    m.inertia = np.zeros(m.numnod, dtype=np.float64)
    
    mat = Material(id=1, law=1, rho0=1000.0, params={"E": 1.0e7, "nu": 0.3})
    prop = Property(id=1, type=14, params={})
    part = Part(id=1, prop_id=1, mat_id=1)
    
    m.materials[1] = mat
    m.properties[1] = prop
    m.parts[1] = part
    m.parts_list = [part]
    
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],      # Element 1: normal
        [1, 8, 9, 2, 5, 10, 11, 6],    # Element 2: thin (dx=0.0001)
    ], dtype=np.int64)
    
    bg = ElementGroup(
        ids=np.array([1, 2], dtype=np.int64),
        conn=conn,
        part=np.zeros(2, dtype=np.int64),
        state={"slices": [(slice(0, 2), mat, prop)]}
    )
    from pyradioss.elements import solid_hexa8
    solid_hexa8.init_group(bg, m, MessageLog())
    m.bricks = bg
    return m


def test_dt_brick_del_erodes_element(tmp_path):
    """Verify that /DT/BRICK/DEL erodes elements whose dt drops below dt_min."""
    model = _build_two_brick_model()
    log = MessageLog()
    
    controls = EngineControls()
    controls.t_end = 1.0e-5
    controls.dt_scale = 0.9
    controls.dt_noda = False
    
    # Normal element has sound speed c = sqrt(E/rho) = sqrt(1e7/1000) = 100 m/s
    # Normal element L = 1.0 m -> dt ~ 1.0 / 100 = 0.01 s
    # Thin element L = 0.0001 m -> dt ~ 0.0001 / 100 = 1.0e-6 s
    # Setting dt_min = 1.0e-4 with action DEL should delete element 2!
    controls.dt_controls["BRICK"] = {
        "action": "DEL",
        "scale": 0.9,
        "dt_min": 1.0e-4,
    }
    
    out_dir = str(tmp_path)
    res_model = _integrate(model, controls, log, out_dir, "TEST", 1)
    
    # Element 2 should be deleted (off == 0)
    assert res_model.bricks.state["off"][0] == 1.0, "Element 1 should remain alive"
    assert res_model.bricks.state["off"][1] == 0.0, "Element 2 should be eroded by /DT/BRICK/DEL"
