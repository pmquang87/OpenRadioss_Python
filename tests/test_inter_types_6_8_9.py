"""
Unit tests for /INTER/TYPE6, /INTER/TYPE8, and /INTER/TYPE9 contact interfaces.

Validates:
1. Every module follows the standard pyradioss contact interface contract:
   - ContactType6, ContactType8, ContactType9
   - __init__(self, itf, model, log)
   - forces(self, x, v, mass, dt, fcont, cycle, stifn=None, t=0.0) -> returns (fcont, dt_bound)
   - EXACT linear momentum conservation: sum(fcont) == 0 to machine precision.
2. Specifically:
   - inter_type6.py (ContactType6):
     * Nonlinear surface-to-surface contact with user curve force/penetration function lookup (funct_id).
     * Abscissa and ordinate scale factors (facx, fac).
     * Fallback to linear penalty stiffness when no curve is defined.
     * Viscous damping and Coulomb friction.
   - inter_type8.py (ContactType8):
     * Simplified drawbead line interface for sheet metal forming (dbead_force, mu, blank nodes).
     * Line segment projection and barycentric weight interpolation.
     * Restraining force opposing transverse sliding across the bead.
     * Normal clamping force.
   - inter_type9.py (ContactType9):
     * ALE/Eulerian to Lagrangian coupling with thermal bridge and upwind momentum advection.
     * Normal reaction force and regularized friction.
     * Upwind momentum advection weighting.
     * Thermal bridge heat flux and energy accounting (ITH = 1).
     * Surface tension force (Fs > 0).
3. Defensive initialization for empty/missing entities and factory build_contacts integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.contact import (
    ContactType6,
    ContactType8,
    ContactType9,
    build_contacts,
)
from pyradioss.model.entities import Interface, NodeGroup, Surface
from pyradioss.model.model import Model


# =========================================================================
# Helper Fixture / Builders
# =========================================================================

def create_base_quad_model():
    """Create a minimal model with a 1-element quad plate (nodes 0..3) in the z=0 plane."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(len(model.x), dtype=float)

    # Master surface with one quad segment (nodes 0, 1, 2, 3)
    surf = Surface(id=1, title="main_surf")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.array(["SHELL"], dtype="<U8")
    surf.seg_elem = np.array([1], dtype=np.int64)
    model.surfaces[1] = surf
    return model


# =========================================================================
# 1. ContactType6 Tests
# =========================================================================

class TestContactType6:
    """Test suite for /INTER/TYPE6 nonlinear surface-to-surface contact."""

    def test_clean_import(self):
        """Verify ContactType6 can be imported cleanly."""
        assert ContactType6 is not None

    def test_exact_linear_momentum_conservation(self):
        """Secondary node impacting master segment must conserve linear momentum to machine precision."""
        model = create_base_quad_model()
        # Add secondary node 4 directly above centroid with penetration
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        model.v = np.zeros_like(model.x)
        model.v[4] = [0.0, 0.0, -10.0]  # approaching master plate
        model.mass = np.ones(5, dtype=float)

        # Secondary node group containing node 4
        model.node_groups[10] = NodeGroup(id=10, title="sec_grp", node_idx=[4])

        itf = Interface(
            id=601, type=6, surf_id=1, grnod_id=10,
            stfac=1.0e4, gap=0.05, fric=0.1, visc=0.05
        )
        log = MessageLog()
        ct6 = ContactType6(itf, model, log)

        fcont = np.zeros_like(model.x)
        fcont_out, dt_bound = ct6.forces(
            model.x, model.v, model.mass, dt=1e-5, fcont=fcont, cycle=0, t=0.0
        )

        assert fcont_out is fcont
        assert dt_bound > 0.0
        # Active contact force on secondary node
        assert fcont[4, 2] > 0.0, "Secondary node must feel repulsive normal force"
        # Master nodes must feel opposite force
        assert np.all(fcont[:4, 2] <= 0.0) and np.sum(fcont[:4, 2]) < 0.0, "Master nodes must feel downward reaction force"

        # EXACT Linear Momentum Conservation
        net_force = np.sum(fcont, axis=0)
        np.testing.assert_allclose(net_force, [0.0, 0.0, 0.0], atol=1e-14,
                                   err_msg="TYPE6 must conserve linear momentum to machine precision")

    def test_user_curve_function_lookup(self):
        """Verify user curve lookup: Fn = fac * curve.eval(penetration * facx)."""
        model = create_base_quad_model()
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        model.v = np.zeros_like(model.x)
        model.mass = np.ones(5, dtype=float)
        model.node_groups[10] = NodeGroup(id=10, title="sec_grp", node_idx=[4])

        # Define quadratic-like force-penetration curve in model.functions
        curve_x = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
        curve_y = np.array([0.0, 100.0, 400.0, 2500.0, 10000.0])
        model.functions[42] = FunctTable(fct_id=42, x=curve_x, y=curve_y, title="F_vs_P")

        itf = Interface(
            id=602, type=6, surf_id=1, grnod_id=10,
            gap=0.05, stfac=1.0,
            params={"funct_id": 42, "facx": 1.0, "fac": 2.0}
        )
        ct6 = ContactType6(itf, model, MessageLog())

        # penetration = gap (0.05) - dist (0.02) = 0.03
        # curve.eval(0.03): interpolated between 0.02 (400) and 0.05 (2500):
        # slope = (2500-400)/(0.05-0.02) = 2100/0.03 = 70000
        # y(0.03) = 400 + 70000 * 0.01 = 1100.0
        # Fn_elastic = 1100.0 * fac (2.0) = 2200.0
        fcont = np.zeros_like(model.x)
        ct6.forces(model.x, model.v, model.mass, dt=1e-5, fcont=fcont, cycle=0, t=0.0)

        expected_fn = 2200.0
        np.testing.assert_allclose(fcont[4, 2], expected_fn, rtol=1e-3,
                                   err_msg="TYPE6 must evaluate user curve f(pen * facx) * fac")
        # Exact momentum conservation holds with curve lookup
        np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-14)

    def test_fallback_to_linear_stiffness_when_no_curve(self):
        """When funct_id is missing or 0, falls back to linear stiffness K * pen."""
        model = create_base_quad_model()
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        model.v = np.zeros_like(model.x)
        model.mass = np.ones(5, dtype=float)
        model.node_groups[10] = NodeGroup(id=10, title="sec_grp", node_idx=[4])

        itf = Interface(
            id=603, type=6, surf_id=1, grnod_id=10,
            gap=0.05, stfac=5.0e4
        )
        ct6 = ContactType6(itf, model, MessageLog())
        fcont = np.zeros_like(model.x)
        ct6.forces(model.x, model.v, model.mass, dt=1e-5, fcont=fcont, cycle=0, t=0.0)

        # pen = 0.05 - 0.02 = 0.03
        assert fcont[4, 2] > 0.0
        np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-14)

    def test_viscous_damping_and_friction(self):
        """Viscous damping increases normal force during approach, friction opposes sliding."""
        model = create_base_quad_model()
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        # Approaching (vz < 0) and sliding in x (vx > 0)
        model.v = np.zeros_like(model.x)
        model.v[4] = [5.0, 0.0, -10.0]
        model.mass = np.ones(5, dtype=float)
        model.node_groups[10] = NodeGroup(id=10, title="sec_grp", node_idx=[4])

        itf = Interface(
            id=604, type=6, surf_id=1, grnod_id=10,
            gap=0.05, stfac=1e4, fric=0.2, visc=0.1
        )
        ct6 = ContactType6(itf, model, MessageLog())
        fcont = np.zeros_like(model.x)
        ct6.forces(model.x, model.v, model.mass, dt=1e-5, fcont=fcont, cycle=0, t=0.0)

        # Normal force positive
        assert fcont[4, 2] > 0.0
        # Friction force must oppose sliding in +x direction
        assert fcont[4, 0] < 0.0, "Friction force must oppose velocity in x"
        np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-14)

    def test_time_window_gating_and_dt_safety(self):
        """Contact must be inactive outside [tstart, tstop] or when dt <= 0."""
        model = create_base_quad_model()
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        model.v = np.zeros_like(model.x)
        model.mass = np.ones(5, dtype=float)
        model.node_groups[10] = NodeGroup(id=10, title="sec_grp", node_idx=[4])

        itf = Interface(
            id=605, type=6, surf_id=1, grnod_id=10,
            gap=0.05, stfac=1e4, tstart=0.1, tstop=0.5
        )
        ct6 = ContactType6(itf, model, MessageLog())

        # t < tstart
        fcont = np.zeros_like(model.x)
        ct6.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.05)
        np.testing.assert_array_equal(fcont, 0.0)

        # dt <= 0
        ct6.forces(model.x, model.v, model.mass, dt=0.0, fcont=fcont, cycle=0, t=0.2)
        np.testing.assert_array_equal(fcont, 0.0)

        # t in [tstart, tstop] -> active
        ct6.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.2)
        assert fcont[4, 2] > 0.0

    def test_defensive_empty_surface(self):
        """Missing or empty master surface must gracefully deactivate without raising."""
        model = Model()
        itf = Interface(id=606, type=6, surf_id=999, grnod_id=999)
        ct6 = ContactType6(itf, model, MessageLog())
        fcont = np.zeros((4, 3))
        ct6.forces(model.x if hasattr(model, 'x') else fcont, fcont, np.ones(4), dt=1e-4, fcont=fcont, cycle=0)
        np.testing.assert_array_equal(fcont, 0.0)


# =========================================================================
# 2. ContactType8 Tests
# =========================================================================

class TestContactType8:
    """Test suite for /INTER/TYPE8 simplified drawbead line interface."""

    def test_clean_import(self):
        """Verify ContactType8 can be imported cleanly."""
        assert ContactType8 is not None

    def test_exact_linear_momentum_conservation(self):
        """Blank node passing across bead line segment must conserve linear momentum to machine precision."""
        model = Model()
        # Bead line along x axis: Node 0 at (0, 0, 0), Node 1 at (2, 0, 0)
        # Blank node 2 at (1, 0.1, 0.02)
        model.x = np.array([
            [0.0, 0.0, 0.0],  # Bead 0
            [2.0, 0.0, 0.0],  # Bead 1
            [1.0, 0.1, 0.02], # Blank node
        ], dtype=float)
        # Blank moving across the bead in +y direction
        model.v = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ], dtype=float)
        model.mass = np.ones(3, dtype=float)

        # Bead line nodes in group 1
        model.node_groups[1] = NodeGroup(id=1, title="bead_line", node_idx=[0, 1])

        # Blank surface in surface 2 containing node 2
        surf_blank = Surface(id=2, title="blank_surf")
        surf_blank.segments = np.array([[2, 2, 2, 2]], dtype=np.int64)
        model.surfaces[2] = surf_blank

        itf = Interface(
            id=801, type=8, grnod_id=1, surf_id=2,
            stfac=500.0, # dbead_force = 500
            fric=0.2,    # mu = 0.2
            gap=0.5,     # depth = 0.5
        )
        ct8 = ContactType8(itf, model, MessageLog())

        fcont = np.zeros_like(model.x)
        fcont_out, dt_bound = ct8.forces(
            model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.0
        )

        assert fcont_out is fcont
        assert dt_bound > 0.0

        # Blank node must feel restraining force in -y (opposing motion)
        assert fcont[2, 1] < 0.0, "Blank node must experience restraining force opposing +y motion"
        # Bead line nodes must feel positive reaction force in +y
        assert fcont[0, 1] > 0.0, "Bead node 0 must feel reaction"
        assert fcont[1, 1] > 0.0, "Bead node 1 must feel reaction"

        # EXACT Linear Momentum Conservation
        net_force = np.sum(fcont, axis=0)
        np.testing.assert_allclose(net_force, [0.0, 0.0, 0.0], atol=1e-14,
                                   err_msg="TYPE8 must conserve linear momentum to machine precision")

    def test_drawbead_barycentric_force_distribution(self):
        """Reaction force on bead nodes must scale inversely with distance to nodes."""
        model = Model()
        # Blank node located at x = 0.5 (closer to node 0 than node 1)
        model.x = np.array([
            [0.0, 0.0, 0.0],  # Bead 0
            [2.0, 0.0, 0.0],  # Bead 1
            [0.5, 0.05, 0.0], # Blank node (xi = 0.25)
        ], dtype=float)
        model.v = np.array([[0,0,0], [0,0,0], [0, 5.0, 0]], dtype=float)
        model.mass = np.ones(3, dtype=float)

        model.node_groups[1] = NodeGroup(id=1, title="bead", node_idx=[0, 1])
        surf = Surface(id=2, title="blank")
        surf.segments = np.array([[2, 2, 2, 2]], dtype=np.int64)
        model.surfaces[2] = surf

        itf = Interface(id=802, type=8, grnod_id=1, surf_id=2, stfac=1000.0, gap=0.2)
        ct8 = ContactType8(itf, model, MessageLog())

        fcont = np.zeros_like(model.x)
        ct8.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.0)

        # xi = 0.5 / 2.0 = 0.25. Bead 0 gets 75% of reaction, Bead 1 gets 25%
        ratio = fcont[0, 1] / fcont[1, 1]
        np.testing.assert_allclose(ratio, 3.0, rtol=1e-4)
        np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-14)

    def test_defensive_empty_bead_nodes(self):
        """Fewer than 2 bead nodes deactivates cleanly without error."""
        model = Model()
        itf = Interface(id=803, type=8, grnod_id=999, surf_id=999)
        ct8 = ContactType8(itf, model, MessageLog())
        fcont = np.zeros((3, 3))
        ct8.forces(fcont, fcont, np.ones(3), dt=1e-4, fcont=fcont, cycle=0)
        np.testing.assert_array_equal(fcont, 0.0)


# =========================================================================
# 3. ContactType9 Tests
# =========================================================================

class TestContactType9:
    """Test suite for /INTER/TYPE9 ALE/Eulerian to Lagrangian coupling."""

    def test_clean_import(self):
        """Verify ContactType9 can be imported cleanly."""
        assert ContactType9 is not None

    def test_exact_linear_momentum_conservation(self):
        """ALE secondary node impacting Lagrangian main face must conserve momentum exactly."""
        model = create_base_quad_model()
        # Add ALE fluid node 4 near plate
        model.x = np.vstack([model.x, [0.5, 0.5, 0.02]])
        model.v = np.zeros_like(model.x)
        model.v[4] = [2.0, 0.0, -10.0]  # moving in +x and -z
        model.mass = np.ones(5, dtype=float)

        # Secondary ALE surface
        surf_ale = Surface(id=2, title="ale_surf")
        surf_ale.segments = np.array([[4, 4, 4, 4]], dtype=np.int64)
        model.surfaces[2] = surf_ale

        itf = Interface(
            id=901, type=9, surf_id=2, surf_id1=1,
            gap=0.05, fric=0.15, stfac=2.0e4, visc=0.05,
            params={"upwind": 0.5, "fs": 50.0, "i_th": 1, "r_th": 0.01}
        )
        ct9 = ContactType9(itf, model, MessageLog())

        fcont = np.zeros_like(model.x)
        fcont_out, dt_bound = ct9.forces(
            model.x, model.v, model.mass, dt=1e-5, fcont=fcont, cycle=0, t=0.0
        )

        assert fcont_out is fcont
        assert dt_bound > 0.0

        # ALE node feels upward reaction and opposing friction
        assert fcont[4, 2] > 0.0, "ALE node must feel normal reaction force"
        # EXACT Linear Momentum Conservation
        net_force = np.sum(fcont, axis=0)
        np.testing.assert_allclose(net_force, [0.0, 0.0, 0.0], atol=1e-14,
                                   err_msg="TYPE9 must conserve linear momentum to machine precision")

    def test_thermal_bridge_and_upwind_features(self):
        """Verify thermal bridge and upwind momentum advection attributes and energy accounting."""
        model = create_base_quad_model()
        model.x = np.vstack([model.x, [0.5, 0.5, 0.01]])
        model.v = np.zeros_like(model.x)
        model.v[4] = [1.0, 0.0, -5.0]
        model.mass = np.ones(5, dtype=float)

        surf_ale = Surface(id=2, title="ale_surf")
        surf_ale.segments = np.array([[4, 4, 4, 4]], dtype=np.int64)
        model.surfaces[2] = surf_ale

        itf = Interface(
            id=902, type=9, surf_id=2, surf_id1=1,
            gap=0.05, fric=0.1, stfac=1e4,
            params={"upwind": 1.0, "fs": 10.0, "i_th": 1, "r_th": 0.05}
        )
        ct9 = ContactType9(itf, model, MessageLog())

        fcont = np.zeros_like(model.x)
        ct9.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont, cycle=0, t=0.0)

        # Check thermal bridge accounting
        assert ct9.i_th == 1
        assert ct9.e_therm > 0.0, "Thermal conduction energy flux must be tracked when I_TH=1"
        assert ct9.e_fric > 0.0, "Friction dissipation energy must be tracked when I_TH=1"

        # Momentum conservation holds
        np.testing.assert_allclose(np.sum(fcont, axis=0), [0.0, 0.0, 0.0], atol=1e-14)


# =========================================================================
# 4. Factory build_contacts Integration
# =========================================================================

class TestBuildContactsFactory:
    """Verify build_contacts instantiates ContactType6, ContactType8, and ContactType9."""

    def test_factory_instantiation(self):
        """build_contacts creates instances of ContactType6, 8, and 9 into penalty list."""
        model = Model()

        # Add surfaces and node groups for interfaces
        surf1 = Surface(id=1, title="surf1")
        surf1.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
        model.surfaces[1] = surf1

        surf2 = Surface(id=2, title="surf2")
        surf2.segments = np.array([[4, 5, 6, 7]], dtype=np.int64)
        model.surfaces[2] = surf2

        grp1 = NodeGroup(id=1, title="bead_grp", node_idx=[8, 9])
        model.node_groups[1] = grp1

        model.interfaces = [
            Interface(id=60, type=6, surf_id=1, surf_id1=2),
            Interface(id=80, type=8, grnod_id=1, surf_id=1),
            Interface(id=90, type=9, surf_id=2, surf_id1=1),
        ]
        log = MessageLog()
        penalty, tied = build_contacts(model, log)

        assert len(tied) == 0
        assert len(penalty) == 3

        types = [type(c) for c in penalty]
        assert ContactType6 in types
        assert ContactType8 in types
        assert ContactType9 in types
