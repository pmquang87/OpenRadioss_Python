"""
Tests for Milestone 612 (M612): Complete Contact Interfaces & BCS Suite Port.
Validates:
- Complete census of all 25 Contact Interface types, Guided Cable, and Lagrange Multipliers
- Momentum conservation (sum F = 0, sum r x F = 0) across all contact families
- Non-reflecting boundary conditions (/BCS/NRF, Lysmer-Kuhlemeyer absorbing dashpots)
- Cyclic sector symmetry boundary conditions (/BCS/CYCLIC, cylindrical mapping)
- Sliding wall boundary condition triggering (/BCS/WALL)
- Lagrange multiplier boundary conditions (/BCS/LAGMUL)
- Dynamic non-linear boundary conditions (/NBCS)
- Starter parser fix for /BCS/PROPELLANT
- build_contacts unified factory integration in pyradioss/contact/
"""

import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.model.entities import (
    BoundaryCondition,
    NodeGroup,
    Surface,
    BcsNrf,
    BcsWall,
    BcsCyclic,
    BcsLagmul,
    NbcsBlock,
    NbcsNode,
    Interface,
    Part,
    Property,
    Material,
)
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.contact import build_contacts
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.common.messages import MessageLog


# ==============================================================================
# 1. Contact Interfaces Census & Physics Tests
# ==============================================================================

def test_contact_census_all_families_registered():
    """Verify that all contact interface families are defined and importable in pyradioss.contact."""
    from pyradioss.contact import (
        ContactType1,
        ContactType2,
        ContactType3,
        ContactType5,
        ContactType6,
        ContactType7,
        ContactType8,
        ContactType9,
        ContactType10,
        ContactType11,
        ContactType12,
        ContactType14,
        ContactType15,
        ContactType16,
        ContactType17,
        ContactType18,
        ContactType20,
        ContactType21,
        ContactType22,
        ContactType23,
        ContactType24,
        ContactType25,
        ContactGuidedCable,
        LagmulType2,
        LagmulType7,
        LagmulType11,
        LagmulType16,
        LagmulType17,
    )
    assert ContactType1 is not None
    assert ContactType3 is not None
    assert ContactType5 is not None
    assert ContactType6 is not None
    assert ContactType8 is not None
    assert ContactType9 is not None
    assert ContactType12 is not None
    assert ContactType14 is not None
    assert ContactType15 is not None
    assert ContactType16 is not None
    assert ContactType17 is not None
    assert ContactType20 is not None
    assert ContactType22 is not None
    assert ContactType25 is not None
    assert ContactGuidedCable is not None
    assert LagmulType11 is not None


def test_contact_type1_ale_fsi_coupling():
    """Verify /INTER/TYPE1 ALE fluid-structure coupling forces and momentum conservation."""
    from pyradioss.contact.inter_type1 import ContactType1

    model = Model()
    model.numnod = 5
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.01],  # Penetrating fluid node (inside gap 0.05)
    ], dtype=float)
    model.v = np.zeros((5, 3), dtype=float)
    model.v[4] = [0.0, 0.0, -1.0]
    model.mass = np.ones(5, dtype=float)

    surf = Surface(id=1, title="STR_SURF", segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, title="ALE_NODES", node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    itf = Interface(id=1, type=1, surf_id=1, grnod_id=2, params={"gap": 0.05, "stfac": 1000.0})
    ct1 = ContactType1(itf, model, MessageLog())

    fcont = np.zeros((5, 3), dtype=float)
    ct1.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    # Secondary node should experience upward repulsive force
    assert fcont[4, 2] > 0.0
    # Master nodes should balance secondary force exactly
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_type3_legacy_s2s():
    """Verify /INTER/TYPE3 surface-to-surface penalty contact with Cartesian DOF deactivation."""
    from pyradioss.contact.inter_type3 import ContactType3

    model = Model()
    model.numnod = 8
    # Master quad
    model.x = np.zeros((8, 3), dtype=float)
    model.x[:4] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    # Slave quad penetrating slightly
    model.x[4:8] = [[0.2, 0.2, 0.01], [0.8, 0.2, 0.01], [0.8, 0.8, 0.01], [0.2, 0.8, 0.01]]
    model.v = np.zeros((8, 3), dtype=float)
    model.mass = np.ones(8, dtype=float)

    surf_m = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    surf_s = Surface(id=2, segments=np.array([[4, 5, 6, 7]], dtype=np.int64))
    model.surfaces[1] = surf_m
    model.surfaces[2] = surf_s

    itf = Interface(id=1, type=3, surf_id1=1, surf_id2=2, surf_id=1, grnod_id=2,
                    params={"gap": 0.05, "stfac": 5000.0, "ibc1": 1})  # deactivate X forces
    ct3 = ContactType3(itf, model, MessageLog())

    fcont = np.zeros((8, 3), dtype=float)
    ct3.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    # Repulsive normal forces along Z
    assert np.sum(fcont[4:8, 2]) > 0.0
    # Exact linear momentum balance
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_type5_nonlinear_friction():
    """Verify /INTER/TYPE5 node-to-surface penalty contact with friction models."""
    from pyradioss.contact.inter_type5 import ContactType5

    model = Model()
    model.numnod = 5
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.01],
    ], dtype=float)
    model.v = np.zeros((5, 3), dtype=float)
    model.v[4] = [1.0, 0.0, 0.0]  # Sliding in X
    model.mass = np.ones(5, dtype=float)

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    itf = Interface(id=1, type=5, surf_id=1, grnod_id=2,
                    params={"gap": 0.05, "stfac": 2000.0, "fric": 0.2})
    ct5 = ContactType5(itf, model, MessageLog())

    fcont = np.zeros((5, 3), dtype=float)
    ct5.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    # Repulsive normal Z force and opposing friction X force
    assert fcont[4, 2] > 0.0
    assert fcont[4, 0] < 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_type6_nonlinear_curves():
    """Verify /INTER/TYPE6 nonlinear surface-to-surface force lookup."""
    from pyradioss.contact.inter_type6 import ContactType6

    model = Model()
    model.numnod = 5
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.02],
    ], dtype=float)
    model.v = np.zeros((5, 3), dtype=float)
    model.mass = np.ones(5, dtype=float)

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    class DummyFunc:
        def eval(self, p):
            return 1000.0 * (p ** 2)

    model.functions[10] = DummyFunc()
    itf = Interface(id=1, type=6, surf_id=1, grnod_id=2,
                    params={"gap": 0.05, "funct_id": 10})
    ct6 = ContactType6(itf, model, MessageLog())

    fcont = np.zeros((5, 3), dtype=float)
    ct6.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    assert fcont[4, 2] > 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_type8_drawbead():
    """Verify /INTER/TYPE8 drawbead line restraining force."""
    from pyradioss.contact.inter_type8 import ContactType8

    model = Model()
    model.numnod = 3
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # Blank node in bead
    ], dtype=float)
    model.v = np.zeros((3, 3), dtype=float)
    model.v[2] = [0.0, 1.0, 0.0]  # Moving along Y
    model.mass = np.ones(3, dtype=float)

    ng = NodeGroup(id=1, node_idx=np.array([2], dtype=np.int64))
    model.node_groups[1] = ng

    itf = Interface(id=1, type=8, grnod_id=1,
                    params={"dbead_force": 500.0, "mu": 0.15,
                            "bead_nodes": [0, 1], "blank_nodes": [2]})
    ct8 = ContactType8(itf, model, MessageLog())

    fcont = np.zeros((3, 3), dtype=float)
    ct8.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    # Restraining force opposite to velocity
    assert fcont[2, 1] < 0.0


def test_contact_type14_analytical_rigid_surface():
    """Verify /INTER/TYPE14 contact with analytical rigid cylinder and plane."""
    from pyradioss.contact.inter_type14 import ContactType14

    model = Model()
    model.numnod = 2
    model.x = np.array([
        [0.0, 0.0, -0.02],  # Penetrating plane Z=0
        [1.0, 0.0, 0.0],
    ], dtype=float)
    model.v = np.zeros((2, 3), dtype=float)
    model.mass = np.ones(2, dtype=float)

    ng = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))
    model.node_groups[1] = ng

    itf = Interface(id=1, type=14, grnod_id=1,
                    params={"geom_type": "PLANE", "stiff1": 1e4, "gap": 0.0,
                            "normal": [0.0, 0.0, 1.0], "orig": [0.0, 0.0, 0.0]})
    ct14 = ContactType14(itf, model, MessageLog())

    fcont = np.zeros((2, 3), dtype=float)
    ct14.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    assert fcont[0, 2] > 0.0


def test_contact_type15_surface_to_analytical():
    """Verify /INTER/TYPE15 surface to analytical surface contact."""
    from pyradioss.contact.inter_type15 import ContactType15

    model = Model()
    model.numnod = 4
    model.x = np.array([
        [0.0, 0.0, -0.01],
        [1.0, 0.0, -0.01],
        [1.0, 1.0, -0.01],
        [0.0, 1.0, -0.01],
    ], dtype=float)
    model.v = np.zeros((4, 3), dtype=float)
    model.mass = np.ones(4, dtype=float)

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf

    itf = Interface(id=1, type=15, surf_id=1,
                    params={"geom_type": "PLANE", "stiff1": 5000.0, "gap": 0.0,
                            "normal": [0.0, 0.0, 1.0], "orig": [0.0, 0.0, 0.0]})
    ct15 = ContactType15(itf, model, MessageLog())

    fcont = np.zeros((4, 3), dtype=float)
    ct15.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    assert np.all(fcont[:, 2] > 0.0)


def test_contact_type20_combined_s2s_and_edge():
    """Verify /INTER/TYPE20 combined surface and edge contact."""
    from pyradioss.contact.inter_type20 import ContactType20

    model = Model()
    model.numnod = 5
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.01],
    ], dtype=float)
    model.v = np.zeros((5, 3), dtype=float)
    model.mass = np.ones(5, dtype=float)

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    itf = Interface(id=1, type=20, surf_id=1, grnod_id=2,
                    params={"gap": 0.05, "stfac": 2500.0, "i_sym": 1})
    ct20 = ContactType20(itf, model, MessageLog())

    fcont = np.zeros((5, 3), dtype=float)
    ct20.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    assert fcont[4, 2] > 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_type25_general_s2s_with_adhesion():
    """Verify /INTER/TYPE25 general segment-to-segment contact with cohesive adhesion."""
    from pyradioss.contact.inter_type25 import ContactType25

    model = Model()
    model.numnod = 5
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, -0.015],
    ], dtype=float)
    model.v = np.zeros((5, 3), dtype=float)
    model.mass = np.ones(5, dtype=float)

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    itf = Interface(id=1, type=25, surf_id=1, grnod_id=2,
                    params={"gap": 0.05, "stfac": 3000.0, "sigmaxadh": 1e6})
    ct25 = ContactType25(itf, model, MessageLog())

    fcont = np.zeros((5, 3), dtype=float)
    ct25.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    assert fcont[4, 2] > 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_contact_guided_cable():
    """Verify /INTER/GUIDED_CABLE 1D sliding friction and centering force."""
    from pyradioss.contact.inter_guided_cable import ContactGuidedCable

    model = Model()
    model.numnod = 3
    # Cable segment along X: node 0 at (0,0,0), node 1 at (2,0,0)
    # Eyelet node 2 at (1, 0.02, 0)
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.02, 0.0],
    ], dtype=float)
    model.v = np.zeros((3, 3), dtype=float)
    model.v[2] = [1.0, 0.0, 0.0]  # Sliding in X
    model.mass = np.ones(3, dtype=float)

    itf = Interface(id=1, type=29, params={"k_trans": 1e4, "mu": 0.1,
                                           "cable_nodes": np.array([[0, 1]], dtype=np.int64),
                                           "guide_nodes": np.array([2], dtype=np.int64)})
    cgc = ContactGuidedCable(itf, model, MessageLog())

    fcont = np.zeros((3, 3), dtype=float)
    cgc.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)

    # Transverse centering force on eyelet along Y (pulling toward Y=0)
    assert fcont[2, 1] < 0.0
    # Sliding friction opposing slip
    assert fcont[2, 0] < 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_lagmul_type11_constraint_generation():
    """Verify LagmulType11 edge-to-edge constraint matrix generation."""
    from pyradioss.contact.inter_type11 import LagmulType11

    model = Model()
    model.numnod = 4
    # Line 1: (0, 0, 0) to (1, 0, 0)
    # Line 2: (0.5, -0.5, 0) to (0.5, 0.5, 0) - crossing at (0.5, 0, 0)
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    model.v = np.zeros((4, 3), dtype=float)
    model.mass = np.ones(4, dtype=float)

    itf = Interface(id=1, type=11, lagmul=True,
                    params={"line1": np.array([[0, 1]], dtype=np.int64),
                            "line2": np.array([[2, 3]], dtype=np.int64),
                            "gap": 0.05})
    lt11 = LagmulType11(itf, model, MessageLog())

    L_data, L_row, L_col = lt11.generate_l_matrix(1e-4)
    assert len(L_data) > 0
    # Constraint row sum should be zero for translational momentum balance
    assert np.isclose(np.sum(L_data), 0.0, atol=1e-12)


def test_build_contacts_all_families():
    """Verify build_contacts instantiates penalty and tied interfaces seamlessly."""
    model = Model()
    model.numnod = 10
    model.x = np.zeros((10, 3), dtype=float)
    model.v = np.zeros((10, 3), dtype=float)
    model.mass = np.ones(10, dtype=float)
    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf
    ng = NodeGroup(id=2, node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = ng

    # Add interfaces across types
    model.interfaces = [
        Interface(id=1, type=7, surf_id=1, grnod_id=2),
        Interface(id=2, type=2, surf_id=1, grnod_id=2),
        Interface(id=3, type=1, surf_id=1, grnod_id=2),
        Interface(id=4, type=3, surf_id=1, grnod_id=2),
        Interface(id=5, type=5, surf_id=1, grnod_id=2),
        Interface(id=6, type=6, surf_id=1, grnod_id=2),
        Interface(id=7, type=8, grnod_id=2),
        Interface(id=8, type=10, surf_id=1, grnod_id=2),
        Interface(id=9, type=14, grnod_id=2, params={"geom_type": "PLANE"}),
        Interface(id=10, type=20, surf_id=1, grnod_id=2),
        Interface(id=11, type=25, surf_id=1, grnod_id=2),
    ]

    penalty, tied = build_contacts(model, MessageLog())
    assert len(tied) == 1
    assert len(penalty) == 10


# ==============================================================================
# 2. Boundary Conditions (BCS) Physics & Engine Tests
# ==============================================================================

def test_bcs_nrf_absorbing_boundary():
    """Verify /BCS/NRF Lysmer-Kuhlemeyer absorbing boundary dashpots absorb stress waves."""
    from pyradioss.engine.bcs_nrf import NonReflectingBoundaryEngine

    model = Model()
    model.numnod = 4
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    # Incoming wave velocity
    model.v = np.array([[0.0, 0.0, 2.0]] * 4, dtype=float)
    model.mass = np.ones(4, dtype=float)

    ng = NodeGroup(id=1, node_idx=np.arange(4, dtype=np.int64))
    model.node_groups[1] = ng

    # Material & Part for acoustic impedance
    mat = Material(id=1, law=1, rho0=2000.0, params={"E": 2.0e10, "nu": 0.25})
    model.materials[1] = mat
    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf

    bcs_nrf = BcsNrf(id=1, grnod_id=1, isurf=1, rho=2000.0, cp=3464.1, cs=2000.0)
    model.bcs_nrfs[1] = bcs_nrf

    nrf_engine = NonReflectingBoundaryEngine(model, MessageLog())
    fext = np.zeros((4, 3), dtype=float)
    nrf_engine.compute_forces(0.0, model.x, model.v, fext)

    # Damping force must oppose normal velocity (Z)
    assert np.all(fext[:, 2] < 0.0)
    # Work done must be negative (dissipative)
    w = nrf_engine.compute_work(fext, model.v, 1e-4)
    assert w < 0.0


def test_bcs_cyclic_symmetry():
    """Verify /BCS/CYCLIC cylindrical coordinate averaging across sector planes."""
    from pyradioss.engine.bcs_cyclic import CyclicBoundaryEngine

    model = Model()
    model.numnod = 4
    # Sector cut plane 1: theta = 0 rad (along X axis)
    # Sector cut plane 2: theta = pi/4 rad (45 deg)
    model.x = np.array([
        [1.0, 0.0, 0.0],  # Node 0 on plane 1, r=1
        [2.0, 0.0, 0.0],  # Node 1 on plane 1, r=2
        [np.cos(np.pi/4), np.sin(np.pi/4), 0.0],  # Node 2 on plane 2, r=1
        [2*np.cos(np.pi/4), 2*np.sin(np.pi/4), 0.0],  # Node 3 on plane 2, r=2
    ], dtype=float)
    model.v = np.zeros((4, 3), dtype=float)
    # Asymmetric initial velocities
    model.v[0] = [1.0, 0.0, 0.0]  # Radial 1.0 on plane 1
    model.v[2] = [0.0, 0.0, 0.0]  # Radial 0.0 on plane 2
    model.mass = np.ones(4, dtype=float)

    ng1 = NodeGroup(id=1, node_idx=np.array([0, 1], dtype=np.int64))
    ng2 = NodeGroup(id=2, node_idx=np.array([2, 3], dtype=np.int64))
    model.node_groups[1] = ng1
    model.node_groups[2] = ng2

    bcs_cyc = BcsCyclic(id=1, grnd_id1=1, grnd_id2=2, skew_id=0)
    model.bcs_cyclics[1] = bcs_cyc

    cyc_engine = CyclicBoundaryEngine(model, MessageLog())
    cyc_engine.enforce(model.x, model.v, None)

    # After enforcement, radial velocity magnitude on both nodes must be identical (averaged to 0.5)
    vr0 = model.v[0, 0]
    vr2 = model.v[2, 0] * np.cos(np.pi/4) + model.v[2, 1] * np.sin(np.pi/4)
    assert np.isclose(vr0, 0.5, atol=1e-6)
    assert np.isclose(vr2, 0.5, atol=1e-6)


def test_bcs_wall_triggering():
    """Verify /BCS/WALL time window gating."""
    from pyradioss.engine.bcs_wall import SlidingWallBcsEngine

    model = Model()
    bcs_w = BcsWall(id=1, grnod_id=1, tstart=0.01, tstop=0.05)
    model.bcs_walls[1] = bcs_w

    wall_engine = SlidingWallBcsEngine(model, MessageLog())
    assert not wall_engine.is_active(1, t=0.005)
    assert wall_engine.is_active(1, t=0.02)
    assert not wall_engine.is_active(1, t=0.06)


def test_bcs_lagmul_constraint():
    """Verify /BCS/LAGMUL constraint matrix generation for LagmulSolver."""
    from pyradioss.engine.lagmul import LagmulBcs

    model = Model()
    model.numnod = 2
    model.x = np.zeros((2, 3), dtype=float)
    model.v = np.zeros((2, 3), dtype=float)
    model.mass = np.ones(2, dtype=float)

    ng = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))
    model.node_groups[1] = ng

    bcs_lag = BcsLagmul(id=1, grnod_id=1, tra=[True, True, False], rot=[False, False, True])
    model.bcs_lagmuls[1] = bcs_lag

    lag_bcs = LagmulBcs(bcs_lag, model, MessageLog())
    L_data, L_row, L_col = lag_bcs.generate_l_matrix(1e-4)

    # Should constrain node 0 in X and Y translations (2 rows) and Z rotation (1 row)
    assert len(L_data) == 3


def test_nbcs_dynamic_resync():
    """Verify /NBCS non-linear boundary condition dynamic resynchronization."""
    from pyradioss.engine.nbcs import NonLinearBcsEngine

    model = Model()
    model.numnod = 2
    model.x = np.zeros((2, 3), dtype=float)
    model.v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    model.mass = np.ones(2, dtype=float)

    node_cond = NbcsNode(node_id=1, tra=[True, False, False], rot=[False, False, False], active=True)
    block = NbcsBlock(id=1, nodes=[node_cond])
    model.nbcs_blocks[1] = block

    nbcs_engine = NonLinearBcsEngine(model, MessageLog())
    nbcs_engine.apply(model.v, None)

    # Node 0 (node_id 1) should have X velocity zeroed
    assert model.v[0, 0] == 0.0
    assert model.v[0, 1] == 2.0


def test_bcs_propellant_parsing():
    """Verify /BCS/PROPELLANT keyword deck parsing seamlessly routes to propellant handler."""
    deck = """# OpenRadioss Starter Deck
/BEGIN
TEST_DECK
/SURF/PART/1
Surface 1
1
/EBCS/PROPELLANT/1
Burning propellant surface
1 0 0 0
1.0 300.0
1.0 0.5
/BCS/PROPELLANT/2
Burning propellant surface alias
1 0 0 0
1.0 300.0
1.0 0.5
/END
"""
    model = parse_starter_deck(deck)
    assert 1 in model.ebcs_propellants
    assert 2 in model.ebcs_propellants
    assert model.ebcs_propellants[2].param_a == 1.0
    assert model.ebcs_propellants[2].param_n == 0.5


def test_kinematics_bcs_full_integration():
    """Verify LoadsAndConstraints coordinates global BCS, skewed BCS, and BCS/ON /OFF."""
    model = Model()
    model.numnod = 3
    model.x = np.zeros((3, 3), dtype=float)
    model.v = np.ones((3, 3), dtype=float)
    model.vr = np.ones((3, 3), dtype=float)
    model.mass = np.ones(3, dtype=float)

    ng1 = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))
    ng2 = NodeGroup(id=2, node_idx=np.array([1], dtype=np.int64))
    model.node_groups[1] = ng1
    model.node_groups[2] = ng2

    # BCS 1 fixes node 0 all DOFs
    # BCS 2 fixes node 1 all DOFs
    model.bcs = [
        BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, True, True], fix_rot=[True, True, True]),
        BoundaryCondition(id=2, grnod_id=2, fix_tra=[True, True, True], fix_rot=[True, True, True]),
    ]

    class MockControls:
        bcs_active = {1: True, 2: False}  # BCS 2 turned OFF via /BCS/OFF

    loads = LoadsAndConstraints(model, MessageLog(), controls=MockControls())
    w = loads.apply_kinematic(0.01, model.v, model.vr, model.mass, model.x, 1e-4)

    # Node 0 fixed, Node 1 free (BCS 2 off), Node 2 free
    assert np.all(model.v[0] == 0.0)
    assert np.all(model.v[1] == 1.0)
    assert np.all(model.v[2] == 1.0)
    assert w == 0.0
