"""Test suite for M595: Centrifugal Loads (/LOAD/CENTRI, /CENTRI) and Imposed Accelerations (/IMPACC).

Verifies Fortran-faithful behavior matching:
- ``starter/source/loads/general/load_centri/hm_read_load_centri.F``
- ``engine/source/loads/general/load_centri/cfield.F``
- ``starter/source/constraints/general/impvel/hm_read_impacc.F``
- ``engine/source/constraints/general/impvel/fixvel.F``

Key tests:
1. Steady centrifugal body force: exact analytical radial force F = m omega^2 r.
2. Dynamic centrifugal expansion with kinetic and external energy balance.
3. Angular acceleration tangential force: exact analytical tangential force F = m alpha r.
4. Imposed acceleration /IMPACC: exact kinematic motion x(t) = 1/2 a t^2, v(t) = a t,
   reaction force R = m * a, and work ledger balance Wext = 1/2 m v^2.
5. Starter parsing in both fixed and free formats for /LOAD/CENTRI, /CENTRI, and /IMPACC.
6. Combined multi-feature simulation.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import CentrifugalLoad, ImposedAcceleration, NodeGroup
from pyradioss.model.model import Model


def Function(id: int = 1, x=None, y=None, title: str = "") -> FunctTable:
    """Helper creating FunctTable with id keyword argument."""
    return FunctTable(fct_id=id, x=x, y=y, title=title)

from pyradioss.starter.starter import run_starter
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.engine.centri import CentrifugalLoadEngine
from pyradioss.engine.impacc import ImposedAccelerationEngine
from pyradioss.engine.engine import run_engine, EngineControls, EngineState


# ─────────────────────────────────────────────────────────────────────────────
# Helper to create a minimal valid starter deck
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DECK = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
M595_CENTRI_IMPACC
      2021         0
/MAT/LAW1/1
Steel
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part1
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                 2.0                 0.0                 0.0
       103                 0.0                 3.0                 0.0
       104                 3.0                 4.0                 0.0
/SHELL/1
         1       101       102       104       103
/GRNOD/NODE/1
AllNodes
       101       102       103       104
/GRNOD/NODE/2
Node102
       102
/FUNCT/1
ConstOmega
                 0.0                10.0
                10.0                10.0
/FUNCT/2
RampOmega
                 0.0                 0.0
                 1.0                50.0
/FUNCT/3
ConstAcc
                 0.0                25.0
                10.0                25.0
"""


def _parse_deck(tmp_path: Path, deck_str: str, name: str = "TEST_0000.rad") -> Tuple[Model, MessageLog]:
    p = tmp_path / name
    p.write_text(deck_str, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ═════════════════════════════════════════════════════════════════════════════
# 1. STARTER KEYWORD PARSING (Fixed and Free format)
# ═════════════════════════════════════════════════════════════════════════════

class TestStarterParsing:
    """Verify parsing of /LOAD/CENTRI, /CENTRI, and /IMPACC in fixed and free formats."""

    def test_load_centri_fixed(self, tmp_path: Path):
        """Parse /LOAD/CENTRI in fixed column format."""
        deck = f"""\
{_BASE_DECK}
/LOAD/CENTRI/1
CentriFixedLoad
#funct_IDT       Dir  frame_ID sensor_ID  grnod_ID      Ivar             Ascalex             Fscaley
         1        ZZ         0         0         1         2                 2.0                15.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert len(model.centri_loads) == 1
        cl = model.centri_loads[0]
        assert cl.id == 1
        assert cl.funct_id == 1
        assert cl.dir == "ZZ"
        assert cl.frame_id == 0
        assert cl.sens_id == 0
        assert cl.grnod_id == 1
        assert cl.ivar == 2
        assert cl.scale_x == pytest.approx(2.0)
        assert cl.scale_y == pytest.approx(15.0)

    def test_load_centri_free(self, tmp_path: Path):
        """Parse /LOAD/CENTRI in comma-separated free format."""
        deck = f"""\
{_BASE_DECK}
/LOAD/CENTRI/2
CentriFreeLoad
1, XX, 0, 0, 1, 1, 0.5, 30.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert len(model.centri_loads) >= 1
        cl = [c for c in model.centri_loads if c.id == 2][0]
        assert cl.id == 2
        assert cl.funct_id == 1
        assert cl.dir == "XX"
        assert cl.grnod_id == 1
        assert cl.ivar == 1
        assert cl.scale_x == pytest.approx(0.5)
        assert cl.scale_y == pytest.approx(30.0)

    def test_centri_fixed(self, tmp_path: Path):
        """Parse /CENTRI in fixed format (CENTRI_1 and CENTRI_2)."""
        deck = f"""\
{_BASE_DECK}
/CENTRI/10
CentriM198Fixed
#--grnd_id---sens_id----fct_id-node_orig-node_axis-----omega
         1         0         1       101       102      12.5
#--scale_x---scale_y---scale_z
       1.0       2.0       1.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert 10 in model.centris
        c = model.centris[10]
        assert c.id == 10
        assert c.grnd_id == 1
        assert c.node_orig == 101
        assert c.node_axis == 102
        assert c.omega == pytest.approx(12.5)
        assert c.scale_y == pytest.approx(2.0)

    def test_centri_free(self, tmp_path: Path):
        """Parse /CENTRI in comma-separated free format."""
        deck = f"""\
{_BASE_DECK}
/CENTRI/11
CentriFree
1, 0, 1, 101, 102, 25.0
1.5, 3.0, 1.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert 11 in model.centris
        c = model.centris[11]
        assert c.id == 11
        assert c.grnd_id == 1
        assert c.node_orig == 101
        assert c.node_axis == 102
        assert c.omega == pytest.approx(25.0)
        assert c.scale_x == pytest.approx(1.5)
        assert c.scale_y == pytest.approx(3.0)

    def test_impacc_fixed(self, tmp_path: Path):
        """Parse /IMPACC in fixed format (IMP_1 and IMP_2)."""
        deck = f"""\
{_BASE_DECK}
/IMPACC/1
ImposedAccFixed
#  fct_IDT       Dir   skew_ID sensor_ID  grnod_ID  frame_ID     Icoor
         3         X         0         0         1         0         0
#           Ascale_x            Fscale_Y              Tstart               Tstop
                 1.0                25.0                 0.0                 5.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert len(model.impacc) == 1
        ia = model.impacc[0]
        assert ia.id == 1
        assert ia.funct_id == 3
        assert ia.dof == 0  # X
        assert ia.grnod_id == 1
        assert ia.scale == pytest.approx(25.0)
        assert ia.xscale == pytest.approx(1.0)
        assert ia.tstart == pytest.approx(0.0)
        assert ia.tstop == pytest.approx(5.0)

    def test_impacc_free(self, tmp_path: Path):
        """Parse /IMPACC in comma-separated free format."""
        deck = f"""\
{_BASE_DECK}
/IMPACC/2
ImposedAccFree
3, Y, 0, 0, 2
2.0, 50.0, 0.0, 10.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert len(model.impacc) >= 1
        ia = [i for i in model.impacc if i.id == 2][0]
        assert ia.id == 2
        assert ia.funct_id == 3
        assert ia.dof == 1  # Y
        assert ia.grnod_id == 2
        assert ia.xscale == pytest.approx(2.0)
        assert ia.scale == pytest.approx(50.0)
        assert ia.tstop == pytest.approx(10.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. STEADY CENTRIFUGAL BODY FORCE: F = m * omega^2 * r
# ═════════════════════════════════════════════════════════════════════════════

class TestSteadyCentrifugalForce:
    """Verifies that steady centrifugal rotation produces exact analytical radial forces."""

    def test_analytical_radial_forces(self):
        """Test F = m * omega^2 * r on particles at various distances from axis."""
        model = Model()
        model.numnod = 4
        # Positions:
        # Node 0 at (0, 0, 0) - on axis (r = 0)
        # Node 1 at (2.0, 0, 0) - r = 2.0 along X
        # Node 2 at (0, 3.0, 5.0) - r = 3.0 along Y (z is along axis)
        # Node 3 at (3.0, 4.0, 1.0) - r = 5.0 in XY plane
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 5.0],
            [3.0, 4.0, 1.0],
        ], dtype=np.float64)
        model.x0 = model.x.copy()
        model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)

        # Masses:
        masses = np.array([10.0, 1.5, 2.0, 0.5], dtype=np.float64)
        model.mass = masses.copy()
        model.frozen = np.zeros(model.numnod, dtype=bool)

        # Node group with all nodes
        model.node_groups = {1: NodeGroup(id=1, node_ids=[1, 2, 3, 4])}

        # Angular velocity omega = 10.0 rad/s about Z-axis
        omega = 10.0
        model.functions = {
            1: Function(id=1, x=[0.0, 10.0], y=[omega, omega])
        }

        # /LOAD/CENTRI: axis ZZ through origin
        cl = CentrifugalLoad(
            id=1, funct_id=1, dir="ZZ", frame_id=0,
            grnod_id=1, ivar=1, scale_x=1.0, scale_y=1.0
        )
        model.centri_loads = [cl]

        loads = LoadsAndConstraints(model, log=None)
        fext = np.zeros((model.numnod, 3), dtype=np.float64)
        loads.external_forces(t=0.0, fext=fext, x=model.x)

        # Analytical predictions: F = m * omega^2 * r
        omega2 = omega ** 2

        # Node 0: r = 0 -> F = (0, 0, 0)
        assert np.allclose(fext[0], [0.0, 0.0, 0.0], atol=1e-12)

        # Node 1: r = (2, 0, 0) -> F_x = 1.5 * 100 * 2 = 300.0 N
        expected_F1 = np.array([1.5 * omega2 * 2.0, 0.0, 0.0])
        assert np.allclose(fext[1], expected_F1, atol=1e-12)

        # Node 2: r = (0, 3, 0) -> F_y = 2.0 * 100 * 3 = 600.0 N
        expected_F2 = np.array([0.0, 2.0 * omega2 * 3.0, 0.0])
        assert np.allclose(fext[2], expected_F2, atol=1e-12)

        # Node 3: r = (3, 4, 0) -> F_x = 0.5 * 100 * 3 = 150 N, F_y = 0.5 * 100 * 4 = 200 N
        expected_F3 = np.array([0.5 * omega2 * 3.0, 0.5 * omega2 * 4.0, 0.0])
        assert np.allclose(fext[3], expected_F3, atol=1e-12)
        # Total magnitude ||F|| = 0.5 * 100 * 5 = 250.0 N
        assert math.isclose(np.linalg.norm(fext[3]), 250.0, rel_tol=1e-12)

    def test_arbitrary_axis_and_center(self):
        """Test centrifugal force about an arbitrary axis passing through an offset center."""
        model = Model()
        model.numnod = 3
        # Center node 1 at (1.0, 1.0, 0.0)
        # Axis node 2 at (1.0, 1.0, 10.0) -> axis is parallel to Z, offset by (1, 1, 0)
        # Target node 3 at (4.0, 5.0, 2.0)
        model.x = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 10.0],
            [4.0, 5.0, 2.0],
        ], dtype=np.float64)
        model.x0 = model.x.copy()
        model.node_ids = np.array([101, 102, 103], dtype=np.int64)
        model.mass = np.array([1.0, 1.0, 2.0], dtype=np.float64)
        model.frozen = np.zeros(model.numnod, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[103])}

        omega = 20.0
        # Using /CENTRI definition with node_orig=101, node_axis=102, constant omega
        c = CentrifugalLoad(
            id=1, grnd_id=1, node_orig=101, node_axis=102,
            omega=omega, scale_x=1.0, scale_y=1.0
        )
        model.centris = {1: c}

        loads = LoadsAndConstraints(model, log=None)
        fext = np.zeros((model.numnod, 3), dtype=np.float64)
        loads.external_forces(t=0.0, fext=fext, x=model.x)

        # Delta x = (4, 5, 2) - (1, 1, 0) = (3, 4, 2)
        # Axis is along Z: n = (0, 0, 1)
        # r = Delta x - (Delta x . n) n = (3, 4, 0)
        # ||r|| = 5.0
        # F = m * omega^2 * r = 2.0 * 400.0 * (3, 4, 0) = (2400.0, 3200.0, 0.0)
        expected_F = np.array([2400.0, 3200.0, 0.0])
        assert np.allclose(fext[2], expected_F, atol=1e-12)
        assert math.isclose(np.linalg.norm(fext[2]), 4000.0, rel_tol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# 3. ANGULAR ACCELERATION TANGENTIAL FORCE: F = m * alpha * r
# ═════════════════════════════════════════════════════════════════════════════

class TestAngularAccelerationForce:
    """Verifies that angular acceleration alpha = domega/dt produces exact tangential forces."""

    def test_pure_angular_acceleration(self):
        """At t=0 with omega(0)=0 and domega/dt = alpha > 0, force is purely tangential."""
        model = Model()
        model.numnod = 2
        # Node 0 at origin, Node 1 at (2.0, 0.0, 0.0)
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        model.x0 = model.x.copy()
        model.node_ids = np.array([1, 2], dtype=np.int64)
        model.mass = np.array([1.0, 3.0], dtype=np.float64)
        model.frozen = np.zeros(2, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[2])}

        # Linear ramp: omega(t) = alpha * t with alpha = 40 rad/s^2
        alpha = 40.0
        model.functions = {
            1: Function(id=1, x=[0.0, 1.0], y=[0.0, alpha])
        }

        # /LOAD/CENTRI with ivar=2 (account for velocity variation)
        cl = CentrifugalLoad(
            id=1, funct_id=1, dir="ZZ", grnod_id=1,
            ivar=2, scale_x=1.0, scale_y=1.0
        )
        model.centri_loads = [cl]

        loads = LoadsAndConstraints(model, log=None)
        fext = np.zeros((2, 3), dtype=np.float64)

        # At t = 0: omega = 0 -> a_centri = 0
        # alpha = 40 rad/s^2, n = (0, 0, 1)
        # a_tan = alpha * (n x r) = 40 * ((0, 0, 1) x (2, 0, 0)) = 40 * (0, 2, 0) = (0, 80, 0)
        # F_tan = m * a_tan = 3.0 * (0, 80, 0) = (0, 240, 0) N along +Y
        loads.external_forces(t=0.0, fext=fext, x=model.x)
        assert np.allclose(fext[1], [0.0, 240.0, 0.0], atol=1e-5)

    def test_combined_centrifugal_and_tangential(self):
        """At t > 0, both radial centrifugal and tangential acceleration forces coexist."""
        model = Model()
        model.numnod = 2
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        model.x0 = model.x.copy()
        model.node_ids = np.array([1, 2], dtype=np.int64)
        m = 2.5
        model.mass = np.array([1.0, m], dtype=np.float64)
        model.frozen = np.zeros(2, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[2])}

        # Ramp: omega(t) = 10 * t -> at t=0.5: omega = 5.0 rad/s, alpha = 10.0 rad/s^2
        alpha = 10.0
        model.functions = {
            1: Function(id=1, x=[0.0, 1.0], y=[0.0, alpha])
        }

        cl = CentrifugalLoad(
            id=1, funct_id=1, dir="ZZ", grnod_id=1,
            ivar=2, scale_x=1.0, scale_y=1.0
        )
        model.centri_loads = [cl]

        loads = LoadsAndConstraints(model, log=None)
        fext = np.zeros((2, 3), dtype=np.float64)

        t = 0.5
        loads.external_forces(t=t, fext=fext, x=model.x)

        omega_t = 5.0
        r = 2.0
        # Radial force F_x = m * omega^2 * r = 2.5 * 25.0 * 2.0 = 125.0 N
        expected_Fx = m * (omega_t ** 2) * r
        # Tangential force F_y = m * alpha * r = 2.5 * 10.0 * 2.0 = 50.0 N
        expected_Fy = m * alpha * r

        assert math.isclose(fext[1, 0], expected_Fx, rel_tol=1e-5)
        assert math.isclose(fext[1, 1], expected_Fy, rel_tol=1e-5)
        assert abs(fext[1, 2]) < 1e-12


# ═════════════════════════════════════════════════════════════════════════════
# 4. IMPOSED ACCELERATION /IMPACC: KINEMATICS & ENERGY BALANCE
# ═════════════════════════════════════════════════════════════════════════════

class TestImposedAccelerationKinematics:
    """Verifies that /IMPACC produces exact kinematic motion x(t)=1/2 a t^2 and v(t)=a t."""

    def test_constant_acceleration_trajectory(self):
        """Verify leapfrog integration under constant imposed acceleration."""
        model = Model()
        model.numnod = 1
        model.x = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model.x0 = model.x.copy()
        model.v = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model.vr = np.zeros((1, 3), dtype=np.float64)
        model.node_ids = np.array([101], dtype=np.int64)
        m = 4.0
        model.mass = np.array([m], dtype=np.float64)
        model.frozen = np.zeros(1, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[101])}

        # Prescribed acceleration a_0 = 15.0 m/s^2 along X
        a0 = 15.0
        model.functions = {
            1: Function(id=1, x=[0.0, 10.0], y=[a0, a0])
        }

        ia = ImposedAcceleration(
            id=1, funct_id=1, dof=0, grnod_id=1,
            scale=1.0, xscale=1.0, tstart=0.0, tstop=10.0
        )
        model.impacc = [ia]

        loads = LoadsAndConstraints(model, log=None)

        dt = 0.001
        n_steps = 100
        total_work = 0.0

        for step in range(n_steps):
            t = step * dt
            v_old = model.v.copy()
            vr_old = model.vr.copy()

            # Free acceleration (no forces)
            acc = np.zeros((1, 3), dtype=np.float64)
            ar = np.zeros((1, 3), dtype=np.float64)

            # Apply /IMPACC
            dw = loads.apply_acceleration(
                t, dt, acc, ar, model.mass, None,
                model.v, model.vr, v_old, vr_old
            )
            total_work += dw

            # Leapfrog update
            model.a = acc
            model.v += acc * dt
            model.x += model.v * dt

        t_end = n_steps * dt
        # Kinematics:
        # In leapfrog with midstep velocity:
        # v(t) is staggered, after n steps v = a0 * n * dt = a0 * t_end
        assert math.isclose(model.v[0, 0], a0 * t_end, rel_tol=1e-7)
        # Position x(t) = sum v dt = a0 * dt^2 * sum_{k=1}^n k = a0 * dt^2 * n*(n+1)/2
        # ~ 1/2 a0 t^2 + 1/2 a0 dt t
        analytical_x = 0.5 * a0 * (t_end ** 2) + 0.5 * a0 * dt * t_end
        assert math.isclose(model.x[0, 0], analytical_x, rel_tol=1e-7)

        # Reaction force R = m * a0 = 4.0 * 15.0 = 60.0 N
        reactions = loads.impacc_reactions
        assert 1 in reactions
        assert math.isclose(reactions[1][0], m * a0, rel_tol=1e-7)

        # Kinetic energy KE = 1/2 m v^2
        final_ke = 0.5 * m * (model.v[0, 0] ** 2)
        # Energy balance: Total work should equal change in kinetic energy
        assert math.isclose(total_work, final_ke, rel_tol=1e-12)

    def test_rotational_impacc(self):
        """Verify rotational /IMPACC (dof 5 = ZZ) angular acceleration."""
        model = Model()
        model.numnod = 1
        model.x = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model.x0 = model.x.copy()
        model.v = np.zeros((1, 3), dtype=np.float64)
        model.vr = np.zeros((1, 3), dtype=np.float64)
        model.node_ids = np.array([1], dtype=np.int64)
        model.mass = np.array([2.0], dtype=np.float64)
        inertia_val = 0.05
        model.inertia = np.array([inertia_val], dtype=np.float64)
        model.frozen = np.zeros(1, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[1])}

        # Angular acceleration 120.0 rad/s^2 about ZZ (dof 5)
        alpha0 = 120.0
        model.functions = {
            1: Function(id=1, x=[0.0, 1.0], y=[alpha0, alpha0])
        }

        ia = ImposedAcceleration(
            id=1, funct_id=1, dof=5, grnod_id=1,
            scale=1.0, xscale=1.0
        )
        model.impacc = [ia]

        loads = LoadsAndConstraints(model, log=None)
        dt = 0.002
        acc = np.zeros((1, 3), dtype=np.float64)
        ar = np.zeros((1, 3), dtype=np.float64)
        v_old = model.v.copy()
        vr_old = model.vr.copy()

        dw = loads.apply_acceleration(
            0.0, dt, acc, ar, model.mass, model.inertia,
            model.v, model.vr, v_old, vr_old
        )

        assert ar[0, 2] == pytest.approx(alpha0)
        # Reaction torque = I * alpha0 = 0.05 * 120.0 = 6.0 N*m
        reactions = loads.impacc_reactions
        assert 1 in reactions
        assert reactions[1][0] == pytest.approx(inertia_val * alpha0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. DYNAMIC SIMULATION & ENERGY CONSERVATION
# ═════════════════════════════════════════════════════════════════════════════

class TestDynamicEnergyConservation:
    """Verifies complete explicit engine energy conservation with centrifugal and imposed acceleration loads."""

    def test_engine_impacc_energy_balance(self):
        """Run explicit time integration loop with /IMPACC and verify exact energy ledger booking."""
        model = Model()
        model.numnod = 2
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        model.x0 = model.x.copy()
        model.v = np.zeros((2, 3), dtype=np.float64)
        model.vr = np.zeros((2, 3), dtype=np.float64)
        model.node_ids = np.array([1, 2], dtype=np.int64)
        model.mass = np.array([2.5, 3.0], dtype=np.float64)
        model.frozen = np.zeros(2, dtype=bool)
        model.node_groups = {
            1: NodeGroup(id=1, node_ids=[1]),
            2: NodeGroup(id=2, node_ids=[2]),
        }

        # Prescribe acceleration a(t) = 20.0 m/s^2 on Node 1 (X direction)
        model.functions = {
            1: Function(id=1, x=[0.0, 1.0], y=[20.0, 20.0])
        }

        ia = ImposedAcceleration(
            id=1, funct_id=1, dof=0, grnod_id=1,
            scale=1.0, xscale=1.0
        )
        model.impacc = [ia]

        loads = LoadsAndConstraints(model, log=None)
        state = EngineState()

        dt = 0.0005
        steps = 50

        for _ in range(steps):
            v_old = model.v.copy()
            vr_old = model.vr.copy()

            # Step 4: acceleration update
            f_total = np.zeros((2, 3), dtype=np.float64)
            inv_mass = 1.0 / model.mass
            acc = f_total * inv_mass[:, None]
            ar = None

            # Enforce /IMPACC
            state.wext += loads.apply_acceleration(
                state.t, dt, acc, ar, model.mass, None,
                model.v, model.vr, v_old, vr_old
            )
            model.a = acc
            model.v += acc * dt
            model.x += model.v * dt
            state.t += dt
            state.cycle += 1

        # Current kinetic energy
        ke = 0.5 * np.sum(model.mass[:, None] * (model.v ** 2))
        # Total external work
        wext = state.wext
        assert math.isclose(wext, ke, rel_tol=1e-12)
        # Also verify state.e_ext alias
        assert state.e_ext == state.wext

    def test_engine_centrifugal_energy_balance(self):
        """Run explicit time integration with centrifugal load and verify work-kinetic energy balance."""
        model = Model()
        model.numnod = 1
        # Particle at r = 1.0 m from Z axis
        model.x = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        model.x0 = model.x.copy()
        model.v = np.zeros((1, 3), dtype=np.float64)
        model.vr = np.zeros((1, 3), dtype=np.float64)
        model.node_ids = np.array([1], dtype=np.int64)
        m = 2.0
        model.mass = np.array([m], dtype=np.float64)
        model.frozen = np.zeros(1, dtype=bool)
        model.node_groups = {1: NodeGroup(id=1, node_ids=[1])}

        # Constant rotation omega = 10.0 rad/s about Z-axis
        omega = 10.0
        model.functions = {
            1: Function(id=1, x=[0.0, 1.0], y=[omega, omega])
        }

        cl = CentrifugalLoad(
            id=1, funct_id=1, dir="ZZ", grnod_id=1,
            scale_x=1.0, scale_y=1.0
        )
        model.centri_loads = [cl]

        loads = LoadsAndConstraints(model, log=None)
        state = EngineState()

        # Integrate for a short time step
        dt = 0.0001
        steps = 40
        fext = np.zeros((1, 3), dtype=np.float64)

        for _ in range(steps):
            v_old = model.v.copy()

            # Step 3: external forces
            fext[:] = 0.0
            loads.external_forces(state.t, fext, model.x)

            # Step 4: acceleration update
            acc = fext / m
            model.a = acc
            model.v += acc * dt

            # Step 5b: external work of loads
            v_mid = 0.5 * (v_old + model.v)
            state.wext += float(np.sum(fext * v_mid)) * dt

            # Step 6: position update
            model.x += model.v * dt
            state.t += dt
            state.cycle += 1

        ke = 0.5 * m * np.sum(model.v ** 2)
        # External work of centrifugal force must equal the kinetic energy gained
        assert math.isclose(state.wext, ke, rel_tol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# 6. COMBINED MULTI-FEATURE TEST
# ═════════════════════════════════════════════════════════════════════════════

class TestCombinedCentriImpacc:
    """Verifies simultaneous centrifugal loading and imposed acceleration in a single simulation."""

    def test_combined_centri_and_impacc(self, tmp_path: Path):
        """Parse and run a deck with both /LOAD/CENTRI and /IMPACC."""
        deck = f"""\
{_BASE_DECK}
/LOAD/CENTRI/1
RotationLoad
         1        ZZ         0         0         1         1                 1.0                10.0
/IMPACC/1
ZAcceleration
         3         Z         0         0         1         0         0
                 1.0                15.0                 0.0                 1.0
/END
"""
        model, log = _parse_deck(tmp_path, deck)
        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        assert len(model.centri_loads) == 1
        assert len(model.impacc) == 1

        loads = LoadsAndConstraints(model, log=None)
        fext = np.zeros((model.numnod, 3), dtype=np.float64)
        loads.external_forces(0.0, fext, model.x)

        # Node 102 is at (2, 0, 0): centrifugal force is along X
        assert fext[1, 0] > 0.0
        # Centrifugal force has no Z component
        assert fext[1, 2] == pytest.approx(0.0)

        # Apply acceleration update
        acc = fext / model.mass[:, None]
        v_old = model.v.copy()
        dw = loads.apply_acceleration(
            0.0, 0.001, acc, None, model.mass, None,
            model.v, None, v_old, None
        )

        # /IMPACC enforces Z acceleration = 15.0 * 25.0 = 375.0 on all nodes
        assert np.allclose(acc[:, 2], 375.0, atol=1e-12)
        # X acceleration is from centrifugal force
        assert acc[1, 0] > 0.0
