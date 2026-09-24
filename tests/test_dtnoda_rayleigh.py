"""Unit tests for Rayleigh damping nodal time step reduction matching dtnodarayl.F.

Upstream Fortran reference:
    C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\time_step\\dtnodarayl.F
    SUBROUTINE DTNODARAYL(MS, IN, STIFN, STIFR, DT2T, IGRNOD, DAMPR)

Formulation:
    dt_0 = sqrt(2 * M_i / K_i)
    bb = beta / dt_0 + 0.5 * alpha * dt_0
    fac = sqrt(bb**2 + 1.0) - bb
    coeff = 1.0 / (fac**2)
    K_i = K_i * coeff
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.mass_scaling import (
    NodalTimeStep,
    apply_rayleigh_damping_nodal,
)
from pyradioss.model.model import EngineControls


class _MockGroup:
    def __init__(self, conn, mass, dt_iner=None):
        self.conn = conn
        self.state = {"mass": mass}
        if dt_iner is not None:
            self.state["dt_iner"] = dt_iner


class _MockModel:
    def __init__(self, numnod, mass, groups=None, node_ids=None, inertia=None):
        self.numnod = numnod
        self.mass = np.asarray(mass, dtype=np.float64)
        self._groups = groups or {}
        self.node_ids = node_ids
        self.inertia = np.zeros(numnod, dtype=np.float64) if inertia is None else np.asarray(inertia, dtype=np.float64)
        self.v = np.zeros((numnod, 3), dtype=np.float64)
        self.vr = np.zeros((numnod, 3), dtype=np.float64)
        self.x0 = np.zeros((numnod, 3), dtype=np.float64)
        self.node_groups = {}

    def element_groups(self):
        return self._groups.items()


def _make_noda(
    dt_noda: str = "NODA",
    dt_scale: float = 1.0,
    dt_min: float = 0.0,
    numnod: int = 4,
    node_mass: float = 1.0,
    node_ids: np.ndarray = None,
    node_inertia: np.ndarray = None,
) -> tuple[_MockModel, NodalTimeStep, MessageLog]:
    conn = np.array([[0, 1, 2, 3]])
    elem_mass = np.array([float(node_mass * 4)])
    elem_iner = np.array([1.0]) if node_inertia is not None else None
    group = _MockGroup(conn, elem_mass, dt_iner=elem_iner)
    mass = np.full(numnod, node_mass, dtype=np.float64)
    model = _MockModel(
        numnod=numnod,
        mass=mass,
        groups={"shells": group},
        node_ids=node_ids,
        inertia=node_inertia,
    )
    controls = EngineControls(dt_noda=dt_noda, dt_scale=dt_scale, dt_min=dt_min)
    log = MessageLog()
    noda = NodalTimeStep(model, controls, log)
    return model, noda, log


def test_rayleigh_damping_zero_damping_unchanged():
    """Test that without damping (alpha=0, beta=0), fac=1.0 and dt is unchanged."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    dt0 = math.sqrt(2.0 * model.mass[0] / noda.stifn[0])  # sqrt(4 / 10000) = 0.02

    # Apply alpha=0, beta=0
    noda.apply_rayleigh_damping(alpha=0.0, beta=0.0)

    # Stiffness should be strictly unchanged
    assert noda.stifn[0] == pytest.approx(10000.0, rel=1e-12)

    # Time step from apply() should match dt0 exactly
    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)
    assert dt == pytest.approx(dt0, rel=1e-12)


def test_rayleigh_damping_stiffness_reduction_factor():
    """Test that with stiffness damping beta > 0, dt is reduced by factor sqrt(beta^2 + 1) - beta.

    When alpha=0 and dt_0 = 1.0:
        bb = beta / 1.0 = beta
        fac = sqrt(beta^2 + 1.0) - beta
        dt_reduced = fac * dt_0 = sqrt(beta^2 + 1.0) - beta
    """
    # Create a node with mass=0.5 and stiffness=1.0 -> dt_0 = sqrt(2 * 0.5 / 1.0) = 1.0
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=0.5)
    noda.stifn[0] = 1.0
    noda.free[0] = True

    dt0 = math.sqrt(2.0 * model.mass[0] / noda.stifn[0])
    assert dt0 == pytest.approx(1.0, rel=1e-12)

    beta = 0.25
    expected_fac = math.sqrt(beta**2 + 1.0) - beta
    expected_dt = expected_fac * dt0
    expected_k = 1.0 / (expected_fac**2)

    noda.apply_rayleigh_damping(alpha=0.0, beta=beta)

    # Scaled stiffness
    assert noda.stifn[0] == pytest.approx(expected_k, rel=1e-12)

    # Time step after reduction
    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)
    assert dt == pytest.approx(expected_dt, rel=1e-12)
    assert dt < dt0


def test_rayleigh_damping_arbitrary_dt0():
    """Test stiffness damping reduction for general dt0."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 5000.0
    noda.free[0] = True

    dt0 = math.sqrt(2.0 * model.mass[0] / noda.stifn[0])
    alpha = 10.0
    beta = 0.001

    bb = beta / dt0 + 0.5 * alpha * dt0
    fac = math.sqrt(bb**2 + 1.0) - bb
    expected_dt = fac * dt0
    expected_k = 5000.0 / (fac**2)

    noda.apply_rayleigh_damping(alpha=alpha, beta=beta)
    assert noda.stifn[0] == pytest.approx(expected_k, rel=1e-12)

    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)
    assert dt == pytest.approx(expected_dt, rel=1e-12)


def test_mass_scaling_cst_holds_dt_min_with_damping():
    """Test that mass scaling CST holds dt >= dt_min even with damping active.

    When damping increases K to K / fac^2, CST must add mass dm so that:
        dt_sca * sqrt(2 * (M + dm) / K_scaled) >= dt_min
    """
    dt_min = 0.02
    dt_scale = 0.9
    # Initial: mass=2.0, stiffness=10000.0 -> dt0 = 0.02
    model, noda, _ = _make_noda(dt_noda="CST", dt_scale=dt_scale, dt_min=dt_min, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    initial_mass = model.mass[0]

    # Large damping that would drop dt significantly below dt_min
    beta = 0.05
    noda.apply_rayleigh_damping(alpha=20.0, beta=beta)

    # Without mass scaling, dt would drop:
    k_scaled = noda.stifn[0]
    assert k_scaled > 10000.0
    dt_undamped = math.sqrt(2.0 * initial_mass / k_scaled)
    assert dt_scale * dt_undamped < dt_min

    # Now run CST apply()
    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    dt = noda.apply(mass_eff, inv_mass, model.v, t=0.0)

    # Verify that CST held the time step at or above target
    assert dt_scale * dt >= dt_min - 1e-9
    # Mass was added
    assert model.mass[0] > initial_mass
    assert noda.mass_added > 0.0


def test_rayleigh_damping_node_groups():
    """Test that Rayleigh damping can be applied selectively to specific node groups."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, numnod=4, node_mass=2.0)
    noda.stifn[:] = 10000.0
    noda.free[:] = True

    # Node groups: group 1 has nodes [0, 1], group 2 has nodes [2, 3]
    node_groups = {1: [0, 1], 2: [2, 3]}

    # Apply damping only to group 1
    dampr = [{"group": 1, "alpha": 0.0, "beta": 0.01}]
    noda.apply_rayleigh_damping(dampr=dampr, node_groups=node_groups)

    # Group 1 nodes should have increased stiffness
    assert noda.stifn[0] > 10000.0
    assert noda.stifn[1] > 10000.0
    # Group 2 nodes should remain at original stiffness
    assert noda.stifn[2] == pytest.approx(10000.0, rel=1e-12)
    assert noda.stifn[3] == pytest.approx(10000.0, rel=1e-12)


def test_rayleigh_damping_time_window():
    """Test that damping is only applied within [tstart, tstop]."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    dampr = [{"alpha": 10.0, "beta": 0.005, "tstart": 0.1, "tstop": 0.5}]

    # At t=0.0 (before tstart), damping should NOT be applied
    noda.apply_rayleigh_damping(dampr=dampr, t=0.0)
    assert noda.stifn[0] == pytest.approx(10000.0, rel=1e-12)

    # At t=0.2 (within window), damping should be applied
    noda.apply_rayleigh_damping(dampr=dampr, t=0.2)
    assert noda.stifn[0] > 10000.0


def test_rayleigh_damping_fortran_array_input():
    """Test 2D numpy array DAMPR input format matching Fortran DAMPR(NRDAMP, NDAMP)."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    # Build a (22, 1) array matching Fortran DAMPR
    dampr = np.zeros((22, 1), dtype=np.float64)
    dampr[1, 0] = 1.0       # group 1
    dampr[2, 0] = 20.0      # DAMPAI (alpha)
    dampr[3, 0] = 0.002     # DAMPBI (beta)
    dampr[16, 0] = 0.0      # tstart
    dampr[17, 0] = 10.0     # tstop
    dampr[18, 0] = 0.0      # active flag
    dampr[20, 0] = 0.0      # ITYPE (0 = standard)

    node_groups = {1: [0]}
    noda.apply_rayleigh_damping(dampr=dampr, node_groups=node_groups, t=0.5)
    assert noda.stifn[0] > 10000.0


def test_rayleigh_damping_rotational_dofs():
    """Test stiffness scaling on rotational DOFs (stifr)."""
    inertia = np.full(4, 0.5, dtype=np.float64)
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=1.0, node_inertia=inertia)
    noda.stifn[0] = 5000.0
    noda.stifr[0] = 2000.0
    noda.free[0] = True

    noda.apply_rayleigh_damping(alpha=10.0, beta=0.005)
    assert noda.stifn[0] > 5000.0
    assert noda.stifr[0] > 2000.0


def test_nodal_time_step_step_method():
    """Test single-call step() method with integrated Rayleigh damping."""
    model, noda, _ = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff

    dt = noda.step(
        mass_eff,
        inv_mass,
        model.v,
        t=0.0,
        alpha=10.0,
        beta=0.001,
    )
    dt0 = math.sqrt(2.0 * model.mass[0] / 10000.0)
    assert dt < dt0


def test_standalone_apply_rayleigh_damping_nodal():
    """Test standalone apply_rayleigh_damping_nodal helper function."""
    mass = np.array([2.0, 3.0])
    stifn = np.array([10000.0, 15000.0])

    k_scaled, _, min_fac = apply_rayleigh_damping_nodal(stifn, mass, alpha=10.0, beta=0.002)
    assert min_fac < 1.0
    assert np.all(k_scaled > stifn)
