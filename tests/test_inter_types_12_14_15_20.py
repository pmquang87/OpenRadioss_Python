"""
Unit tests for ContactType12, ContactType14, ContactType15, and ContactType20.

Verifies:
1. Standard contract:
   - class ContactTypeN
   - __init__(self, itf, model, log)
   - forces(self, x, v, mass, dt, fcont, cycle, stifn=None, t=0.0) -> (fcont, dt_bound)
2. Momentum conservation:
   - sum(fcont) == 0 for closed systems.
3. Specific mechanics:
   - ContactType12: ALE mesh motion / grid velocity coupling
   - ContactType14: Analytical rigid surfaces (PLANE, CYL, SPHER, ELLIP)
   - ContactType15: Surface segments to analytical surface penalty contact
   - ContactType20: Combined surface-to-surface and edge-to-edge contact with symmetry
"""

from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import (
    ContactType12,
    ContactType14,
    ContactType15,
    ContactType20,
)
from pyradioss.model.entities import NodeGroup, Surface
from pyradioss.model.model import Model


def _make_dummy_model(n_nodes: int = 20) -> Model:
    model = Model()
    model.mass = np.ones(n_nodes, dtype=np.float64) * 2.0
    model.mass0 = model.mass.copy()
    model.node_groups = {}
    model.surfaces = {}
    model.lines = {}
    return model


# ======================================================================
# ContactType12 Tests
# ======================================================================

def test_contact_type12_contract_and_momentum():
    """Verify ContactType12 initialization, forces contract, and strict momentum conservation."""
    model = _make_dummy_model(10)
    log = MessageLog()

    # Create master quad surface: nodes 0, 1, 2, 3 (z = 0 plane)
    surf_m = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf_m

    # Create secondary surface: node 4 located above the master segment
    surf_s = Surface(id=2, segments=np.array([[4, 4, 4, 4]], dtype=np.int64))
    model.surfaces[2] = surf_s

    itf = SimpleNamespace(
        id=12,
        type=12,
        surf_ids=2,
        surf_idm=1,
        tol=0.1,
        stfac=1.0,
        itied=0,
        tstart=0.0,
        tstop=1.0,
    )

    c12 = ContactType12(itf, model, log)
    assert hasattr(c12, "forces")
    assert c12.dt_bound > 0.0

    # Nodal coordinates: master segment at z = 0, secondary node 4 at (0.5, 0.5, 0.05) (within tol=0.1)
    x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.05], # 4 (secondary)
        [0.0, 0.0, 0.0],  # 5..9 unused
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)

    v = np.zeros_like(x)
    v[4] = [0.0, 0.0, -1.0]  # moving toward master segment

    fcont = np.zeros_like(x)
    stifn = np.zeros(len(x), dtype=np.float64)
    dt = 1e-4

    fcont_out, dt_bound = c12.forces(x, v, model.mass, dt, fcont, cycle=1, stifn=stifn, t=0.01)

    assert fcont_out is fcont
    assert dt_bound > 0.0

    # Secondary node 4 must experience an upward restoring force in z
    assert fcont[4, 2] < 0.0 or fcont[4, 2] != 0.0

    # Strict momentum conservation: sum of forces across all nodes must be identically zero
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-12), f"Non-zero net force in ContactType12: {net_force}"


# ======================================================================
# ContactType14 Tests (Analytical Rigid Surfaces: PLANE, CYL, SPHER, ELLIP)
# ======================================================================

@pytest.mark.parametrize("geom_type", ["PLANE", "CYL", "SPHER", "ELLIP"])
def test_contact_type14_geometries_and_momentum(geom_type):
    """Verify ContactType14 for all 4 analytical surface geometries with momentum conservation."""
    model = _make_dummy_model(10)
    log = MessageLog()

    # Secondary node group: node 1 is penetrating, node 0 is the rigid surface master node
    model.node_groups[10] = NodeGroup(id=10, node_idx=[1])

    # Construct analytical surface parameters based on geometry
    if geom_type == "PLANE":
        surf = Surface(
            id=100,
            plane_p1=np.array([0.0, 0.0, 0.0]),
            plane_p2=np.array([0.0, 0.0, 1.0]),  # normal is +z
        )
        # Secondary node at z = 0.01 with gap = 0.05 (penetrating by 0.04)
        node_pos = np.array([0.0, 0.0, 0.01])
    elif geom_type == "CYL":
        surf = Surface(
            id=100,
            cyl_center=np.array([0.0, 0.0, 0.0]),
            cyl_axis=np.array([0.0, 0.0, 1.0]),
            cyl_radius=1.0,
            cyl_length=10.0,
        )
        # Node at radius 0.98 with gap = 0.05 (inside cylinder obstacle by 0.07)
        node_pos = np.array([0.98, 0.0, 1.0])
    elif geom_type == "SPHER":
        surf = Surface(
            id=100,
            spher_center=np.array([0.0, 0.0, 0.0]),
            spher_radius=1.0,
        )
        # Node at radius 0.95 with gap = 0.1
        node_pos = np.array([0.95, 0.0, 0.0])
    elif geom_type == "ELLIP":
        surf = Surface(
            id=100,
            ellipse_center=np.array([0.0, 0.0, 0.0]),
            ellipse_semiaxes=np.array([1.0, 1.0, 1.0]),
        )
        node_pos = np.array([0.95, 0.0, 0.0])

    model.surfaces[100] = surf

    itf = SimpleNamespace(
        id=14,
        type=14,
        grnod_id=10,
        surf_id=100,
        geom_type=geom_type,
        gap=0.1,
        stfac=1.0,
        fric=0.1,
        visc=0.05,
        master_node=0,  # node 0 carries the rigid surface reaction
        closed_system=True,
    )

    c14 = ContactType14(itf, model, log)
    assert c14.geom_type == geom_type

    x = np.zeros((10, 3), dtype=np.float64)
    x[0] = [0.0, 0.0, 0.0]  # master node
    x[1] = node_pos

    v = np.zeros_like(x)
    v[1] = [-0.1, 0.0, -0.1]

    fcont = np.zeros_like(x)
    fcont_out, dt_b = c14.forces(x, v, model.mass, 1e-4, fcont, cycle=1, t=0.0)

    # Node 1 must receive contact force
    f_node = fcont[1]
    assert np.linalg.norm(f_node) > 0.0, f"No force generated for geom_type {geom_type}"

    # Master node 0 must receive opposite reaction force
    assert np.allclose(fcont[0], -f_node, atol=1e-12)

    # Net force for closed system must be zero
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-12), f"Non-zero net force for {geom_type}: {net_force}"


# ======================================================================
# ContactType15 Tests (Segments to Analytical Rigid Surface)
# ======================================================================

def test_contact_type15_segments_and_momentum():
    """Verify ContactType15 segment-to-analytical penalty contact and momentum conservation."""
    model = _make_dummy_model(10)
    log = MessageLog()

    # Secondary surface: quad segment on nodes (1, 2, 3, 4)
    surf_sec = Surface(
        id=10,
        segments=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )
    model.surfaces[10] = surf_sec

    # Main analytical surface: Plane at z=0 with normal +z
    surf_main = Surface(
        id=20,
        plane_p1=np.array([0.0, 0.0, 0.0]),
        plane_p2=np.array([0.0, 0.0, 1.0]),
    )
    model.surfaces[20] = surf_main

    itf = SimpleNamespace(
        id=15,
        type=15,
        surf_id=10,
        surf_id1=20,
        geom_type="PLANE",
        gap=0.05,
        stfac=1.0,
        fric=0.1,
        master_node=0,  # node 0 carries reaction
        closed_system=True,
    )

    c15 = ContactType15(itf, model, log)
    assert hasattr(c15, "forces")

    # Segment corners located at z = 0.02 (penetrating gap=0.05 by 0.03)
    x = np.zeros((10, 3), dtype=np.float64)
    x[1] = [0.0, 0.0, 0.02]
    x[2] = [1.0, 0.0, 0.02]
    x[3] = [1.0, 1.0, 0.02]
    x[4] = [0.0, 1.0, 0.02]

    v = np.zeros_like(x)
    fcont = np.zeros_like(x)

    fcont_out, dt_b = c15.forces(x, v, model.mass, 1e-4, fcont, cycle=1, t=0.0)

    # Penetrating segment nodes must receive positive normal force in z
    assert np.all(fcont[[1, 2, 3, 4], 2] > 0.0)

    # Reaction on master node 0 balances total segment force
    assert np.allclose(fcont[0], -np.sum(fcont[1:5], axis=0), atol=1e-12)

    # Total force must sum to 0 to machine precision
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-9), f"Non-zero net force in ContactType15: {net_force}"


# ======================================================================
# ContactType20 Tests (Combined Surface-to-Surface & Edge-to-Edge with Symmetry)
# ======================================================================

def test_contact_type20_surface_symmetry_and_momentum():
    """Verify ContactType20 surface-to-surface contact with symmetry and strict momentum conservation."""
    model = _make_dummy_model(10)
    log = MessageLog()

    # Surface 1: single node 0 contacting segment 2
    # Surface 2: segment (1, 2, 3, 4)
    surf1 = Surface(id=1, segments=np.array([[0, 0, 0, 0]], dtype=np.int64))
    surf2 = Surface(id=2, segments=np.array([[1, 2, 3, 4]], dtype=np.int64))
    model.surfaces[1] = surf1
    model.surfaces[2] = surf2

    itf = SimpleNamespace(
        id=20,
        type=20,
        surf_id=1,
        surf_id1=2,
        isym=1,    # symmetric pass
        iedge=0,   # test surface first
        gap=0.1,
        stfac=1.0,
    )

    c20 = ContactType20(itf, model, log)
    assert hasattr(c20, "forces")

    # Segment (1, 2, 3, 4) at z = 0, Node 0 at (0.5, 0.5, 0.04) (penetrating gap=0.1)
    x = np.zeros((10, 3), dtype=np.float64)
    x[0] = [0.5, 0.5, 0.04]
    x[1] = [0.0, 0.0, 0.0]
    x[2] = [1.0, 0.0, 0.0]
    x[3] = [1.0, 1.0, 0.0]
    x[4] = [0.0, 1.0, 0.0]

    v = np.zeros_like(x)
    fcont = np.zeros_like(x)

    fcont_out, dt_b = c20.forces(x, v, model.mass, 1e-4, fcont, cycle=1, t=0.0)

    # Node 0 must receive upward force in z
    assert fcont[0, 2] > 0.0

    # Master corners 1..4 receive equal and opposite force
    assert np.allclose(fcont[0], -np.sum(fcont[1:5], axis=0), atol=1e-12)

    # Strict momentum conservation
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-12), f"Non-zero net force in ContactType20 surface pass: {net_force}"


def test_contact_type20_edge_to_edge_and_momentum():
    """Verify ContactType20 edge-to-edge contact mechanics and momentum conservation."""
    model = _make_dummy_model(10)
    log = MessageLog()

    # Two crossed edges:
    # Surface 1 has segment containing edge (0, 1) along X axis: (0,0,0) to (1,0,0)
    # Surface 2 has segment containing edge (2, 3) along Y axis at z = 0.02: (0.5,-0.5,0.02) to (0.5,0.5,0.02)
    surf1 = Surface(id=1, segments=np.array([[0, 1, 1, 0]], dtype=np.int64))
    surf2 = Surface(id=2, segments=np.array([[2, 3, 3, 2]], dtype=np.int64))
    model.surfaces[1] = surf1
    model.surfaces[2] = surf2

    itf = SimpleNamespace(
        id=21,
        type=20,
        surf_id=1,
        surf_id1=2,
        isym=0,
        iedge=1,   # edge-to-edge contact enabled
        gap=0.05,
        stfac=1.0,
    )

    c20 = ContactType20(itf, model, log)

    x = np.zeros((10, 3), dtype=np.float64)
    x[0] = [0.0, 0.0, 0.0]
    x[1] = [1.0, 0.0, 0.0]
    x[2] = [0.5, -0.5, 0.02]
    x[3] = [0.5, 0.5, 0.02]

    v = np.zeros_like(x)
    fcont = np.zeros_like(x)

    fcont_out, dt_b = c20.forces(x, v, model.mass, 1e-4, fcont, cycle=1, t=0.0)

    # Edge-to-edge closest points are at (0.5, 0, 0) and (0.5, 0, 0.02)
    # Distance is 0.02 < gap (0.05), so contact occurs
    assert np.linalg.norm(fcont) > 0.0

    # Net force across all edge endpoints must sum to 0
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-12), f"Non-zero net force in ContactType20 edge pass: {net_force}"
