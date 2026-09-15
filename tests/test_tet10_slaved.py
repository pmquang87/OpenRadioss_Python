import numpy as np
import pytest

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import Property, Material, Part
from pyradioss.elements.solid_tetra10 import forces


def test_tet10_slaved_force_redistribution():
    # 1. Create a dummy model with 4 corner nodes and 6 zeroed mid-side nodes
    model = Model()
    
    # 10 nodes (4 corners + 6 mid-sides)
    model.x = np.array([
        [0.0, 0.0, 0.0],  # 0: origin
        [1.0, 0.0, 0.0],  # 1: x
        [0.0, 1.0, 0.0],  # 2: y
        [0.0, 0.0, 1.0],  # 3: z
        # 6 mid-side nodes (indices 4..9), we'll initialize them to garbage
        # The forces kernel should overwrite them!
        [9.9, 9.9, 9.9],
        [9.9, 9.9, 9.9],
        [9.9, 9.9, 9.9],
        [9.9, 9.9, 9.9],
        [9.9, 9.9, 9.9],
        [9.9, 9.9, 9.9]
    ])
    
    model.v = np.zeros((10, 3))
    # Give the corners some velocity so there is internal force
    model.v[1] = [0.1, 0.0, 0.0]
    model.v[2] = [0.0, 0.1, 0.0]
    
    model.vr = np.zeros((10, 3))
    
    # Create the conn array with the special -1 marker for slaved nodes
    # Let's map nodes 4..9 to -1 to trigger the is_slaved flag
    conn = np.array([[0, 1, 2, 3, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    
    # Mock group state
    state = {
        "mass": np.array([1.0]),
        "sig": np.zeros((1, 4, 6)),
        "epsp": np.zeros((1, 4)),
        "off": np.array([1.0]),
        "qvw_pend": np.zeros(1),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "slices": []
    }
    
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = state
    
    # We must mock materials.solid_update since slices is empty
    # Wait, we need at least one slice to give it a material
    mat = Material(id=1, law=1, rho0=1.0, params={"Rho_initial": 1.0, "rho": 1.0, "E": 1e5, "e": 1e5, "Nu": 0.3, "nu": 0.3})
    prop = Property(id=1, type=14, params={"qa": 1.0, "qb": 1.0, "h": 0.1, "Itetra4": 1, "itetra4": 1})
    
    # We can't easily mock the material update because it's deeply integrated.
    # We'll just provide a valid slice.
    state["slices"] = [(slice(0, 1), mat, prop)]
    
    fint = np.zeros((10, 3))
    mint = np.zeros((10, 3))
    
    dt_crit = forces(group, model.x, model.v, model.vr, dt=1e-5, fint=fint, mint=mint)
    
    # The mid-side node forces should be perfectly zeroed because they were redistributed
    assert np.all(fint[-1] == 0.0), "Mid-side node forces were not zeroed in fint!"
    
    # Check that mid-side coordinates were dynamically updated inside the kernel
    # Actually, we passed model.x which was indexed. x[conn] made a copy.
    # The global model.x is untouched. We can only verify fint.
    
    # Let's ensure that the corner nodes actually received forces
    corner_forces = fint[0:4]
    assert np.any(corner_forces != 0.0), "Corner nodes did not receive any forces!"
    
    # Sum of forces should be near zero (equilibrium)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8), "Forces are not in equilibrium!"
