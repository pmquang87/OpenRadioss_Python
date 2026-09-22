"""
Unit tests for OpenRadioss /PROP/TYPE23 (/PROP/SPR_MAT: Spring with Material Laws).

Fortran references:
  - starter/source/properties/spring/hm_read_prop23.F
  - starter/source/elements/spring/rinit3.F (lines 440-507)
  - starter/source/elements/spring/rmass.F (lines 156-185)
  - engine/source/elements/spring/r23forc3.F
  - engine/source/elements/spring/r23law108.F & r23l108def3.F
  - engine/source/elements/spring/r23law113.F & r23l113def3.F
  - engine/source/elements/spring/r23law114.F & r23l114def3.F
  - engine/source/elements/spring/redef3.F90
  - engine/source/tools/seatbelts/redef_seatbelt.F90
  - engine/source/elements/spring/r4cum3.F
  - engine/source/elements/spring/r2cum3.F
  - engine/source/elements/spring/r2len3.F
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_mat
from pyradioss.materials.law108_yield_fit import Law108Params, SpringDOFParams
from pyradioss.materials.law113_yield_curve_fit import Law113Params, BeamSpringDOF
from pyradioss.materials.law114_seatbelt import Law114Seatbelt
from pyradioss.model.entities import Material, Property, PropType23
from pyradioss.model.model import ElementGroup, Model


class MockFunction:
    """Mock Radioss curve function."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)
        dx = np.diff(self.x)
        dx = np.where(np.abs(dx) < 1.0e-30, 1.0e-30, dx)
        self.slope = np.diff(self.y) / dx

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))

    def evaluate(self, val: float) -> float:
        return self.eval(val)


def test_type23_mass_and_inertia_imass1():
    """Verify TYPE23 mass and inertia initialization with Imass=1 (Area * L0 * rho).

    Fortran origin: rinit3.F lines 473-502, rmass.F lines 156-185.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Nodes at (0,0,0) and (2,0,0) -> L0 = 2.0 m
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = PropType23(
        id=1,
        imass=1,
        area_or_volume=0.05,  # Area = 0.05 m^2
        inertia=0.012,       # Inertia = 0.012 kg*m^2
        params={"imass": 1, "area": 0.05, "inertia": 0.012},
    )

    mat = Material(id=1, law=108, rho0=2000.0, title="Steel_Like")

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat, prop)], "model": model}
    )

    log = MessageLog()
    stride = 2
    massn = np.zeros(2 * stride)
    inertn = np.zeros(2 * stride)

    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]), massn=massn, inertn=inertn)

    # Expected: mass = Area * L0 * rho = 0.05 * 2.0 * 2000.0 = 200.0 kg
    assert group.state["mass"][0] == pytest.approx(200.0)
    assert group.state["inertia"][0] == pytest.approx(0.012)
    assert group.state["L0"][0] == pytest.approx(2.0)

    # Distributed half/half to node 1 and node 2 (rmass.F:156-185)
    assert massn[0] == pytest.approx(100.0)
    assert massn[1] == pytest.approx(100.0)
    assert inertn[0] == pytest.approx(0.006)
    assert inertn[1] == pytest.approx(0.006)


def test_type23_mass_and_inertia_imass2():
    """Verify TYPE23 mass initialization with Imass=2 (Volume * rho).

    Fortran origin: rinit3.F lines 500-502.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = PropType23(
        id=1,
        imass=2,
        area_or_volume=0.15,  # Volume = 0.15 m^3
        inertia=0.05,
        params={"imass": 2, "volume": 0.15, "inertia": 0.05},
    )

    mat = Material(id=1, law=108, rho0=1500.0)

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat, prop)], "model": model}
    )

    log = MessageLog()
    stride = 2
    massn = np.zeros(2 * stride)
    inertn = np.zeros(2 * stride)

    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]), massn=massn, inertn=inertn)

    # Expected: mass = Volume * rho = 0.15 * 1500.0 = 225.0 kg
    assert group.state["mass"][0] == pytest.approx(225.0)
    assert group.state["inertia"][0] == pytest.approx(0.05)
    assert massn[0] == pytest.approx(112.5)
    assert massn[1] == pytest.approx(112.5)


def test_type23_law114_seatbelt_loading_unloading():
    """Verify LAW114 seatbelt tension loading, unloading permanent offset, slack accommodation, and energy.

    Fortran origin: r23law114.F, r23l114def3.F, redef_seatbelt.F90.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Initial length L0 = 1.0 m
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Loading curve: strain -> Force [N]
    # e=0.0 -> F=0, e=0.1 -> F=1000 N, e=0.2 -> F=3000 N
    f_load = MockFunction(1, [0.0, 0.1, 0.2, 0.5], [0.0, 1000.0, 3000.0, 10000.0])
    # Unloading curve: steeper slope for unloading
    f_unl = MockFunction(2, [0.0, 0.05, 0.1], [0.0, 2000.0, 5000.0])
    model.functions = {1: f_load, 2: f_unl}

    mat114 = Law114Seatbelt(
        id=1,
        rho0=1000.0,
        params={
            "LMIN": 0.5,
            "FUN_L": 1,
            "FUN_UL": 2,
            "YOUNG": 0.0,      # Slack in compression (zero stiffness)
            "DAMP1": 50.0,     # Viscous damping
        }
    )

    prop = PropType23(
        id=1,
        imass=1,
        area_or_volume=0.01,
        params={"imass": 1, "area": 0.01},
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat114, prop)], "model": model}
    )

    log = MessageLog()
    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]))

    # Step 1: Stretch to L = 1.10 m (strain = 0.10, relative velocity = 0)
    x_step1 = np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], dtype=np.float64)
    v_step1 = np.zeros_like(x_step1)
    vr_step1 = np.zeros_like(x_step1)
    fint = np.zeros_like(x_step1)
    mint = np.zeros_like(x_step1)
    dt = 0.001

    dtc = spring_mat.forces_spring_mat_type23(group, x_step1, v_step1, vr_step1, dt, fint, mint, idx23=np.array([0]))

    # Loading at strain=0.10 should yield F = 1000 N
    assert group.state["force"][0] == pytest.approx(1000.0, rel=1e-3)
    # Action-reaction balance along x-axis
    assert fint[0, 0] == pytest.approx(1000.0, rel=1e-3)
    assert fint[1, 0] == pytest.approx(-1000.0, rel=1e-3)
    assert np.allclose(fint[:, 1:], 0.0)
    assert np.allclose(mint, 0.0)
    assert dtc[0] > 0.0

    # Step 2: Unload partially to L = 1.095 m (elastic unloading branch before slack)
    x_step2 = np.array([[0.0, 0.0, 0.0], [1.095, 0.0, 0.0]], dtype=np.float64)
    fint.fill(0.0)
    spring_mat.forces_spring_mat_type23(group, x_step2, v_step1, vr_step1, dt, fint, mint, idx23=np.array([0]))

    # During unloading, force drops below 1000 N
    f_unloaded = group.state["force"][0]
    assert 0.0 < f_unloaded < 1000.0
    assert fint[0, 0] == pytest.approx(f_unloaded, rel=1e-3)
    assert fint[1, 0] == pytest.approx(-f_unloaded, rel=1e-3)

    # Step 3: Slack state at L = 0.95 m (L < L0)
    x_step3 = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=np.float64)
    fint.fill(0.0)
    spring_mat.forces_spring_mat_type23(group, x_step3, v_step1, vr_step1, dt, fint, mint, idx23=np.array([0]))

    # When Young=0, compression/slack yields zero force (redef_seatbelt.F90)
    assert group.state["force"][0] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(fint, 0.0)


def test_type23_law108_6dof_hardening_and_moment_equilibrium():
    """Verify LAW108 6-DOF spring with isotropic hardening, rate effects, and angular momentum equilibrium.

    Fortran origin: r23law108.F, r23l108def3.F, r2cum3.F:125-140, redef3.F90.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Initial vector (0, 0, 0) to (1, 1, 0)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)

    # Curve for DOF 1 (Tx): strain/disp -> force
    f_yield = MockFunction(1, [0.0, 0.1, 0.2, 0.5], [0.0, 500.0, 800.0, 1500.0])
    model.functions = {1: f_yield}

    # Setup LAW108 6 DOFs
    dofs = [
        SpringDOFParams(stiff=10000.0, damp=10.0, hflag=1, fun_a=1, acoeft=1.0, bcoeft=0.1, dcoeft=1.0),
        SpringDOFParams(stiff=5000.0, damp=5.0, hflag=0),
        SpringDOFParams(stiff=5000.0, damp=5.0, hflag=0),
        SpringDOFParams(stiff=2000.0, damp=2.0, hflag=0),
        SpringDOFParams(stiff=2000.0, damp=2.0, hflag=0),
        SpringDOFParams(stiff=2000.0, damp=2.0, hflag=0),
    ]

    mat108 = Law108Params(id=1, rho0=7850.0, dofs=dofs, iequil=1)

    prop = PropType23(
        id=1,
        imass=2,
        area_or_volume=0.01,
        inertia=0.005,
        params={"imass": 2, "volume": 0.01, "inertia": 0.005, "iequil": 1},
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat108, prop)], "model": model}
    )

    log = MessageLog()
    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]))

    # Displace element: Tx stretched by 0.15 m, Ty sheared by 0.05 m
    x_curr = np.array([
        [0.0, 0.0, 0.0],
        [1.15, 1.05, 0.0],
    ], dtype=np.float64)
    v_curr = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    vr_curr = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],  # relative z-rotation rate = 1.0 rad/s
    ], dtype=np.float64)
    dt = 0.01

    fint = np.zeros_like(x_curr)
    mint = np.zeros_like(x_curr)

    dtc = spring_mat.forces_spring_mat_type23(group, x_curr, v_curr, vr_curr, dt, fint, mint, idx23=np.array([0]))

    # 1. Translational Action-Reaction: fint[n1] + fint[n2] == 0
    assert np.allclose(fint[0] + fint[1], 0.0, atol=1e-10)

    # 2. Angular Momentum Conservation (r2cum3.F:125-140):
    # Total moment about node 1: M1 + M2 + cross(x2 - x1, F2) == 0
    # Where F2 is the force on node 2: F2 = fint[1]
    dx = x_curr[1] - x_curr[0]
    total_moment = mint[0] + mint[1] + np.cross(dx, fint[1])
    assert np.allclose(total_moment, 0.0, atol=1e-10)

    # 3. Energy accumulation check
    assert group.state["eint"][0] > 0.0

    # 4. Critical time step
    assert dtc[0] > 0.0
    assert np.isfinite(dtc[0])


def test_type23_law113_corotational_beam_shear_moment_arm():
    """Verify LAW113 co-rotational beam axes and transverse shear moment arm equilibrium.

    Fortran origin: r23law113.F, r23l113def3.F, r4cum3.F lines 107-138.
    Transverse shear forces FY, FZ generate nodal moments:
      YMOM1 = MY - 0.5 * L * FZ,  ZMOM1 = MZ + 0.5 * L * FY
      YMOM2 = MY + 0.5 * L * FZ,  ZMOM2 = MZ - 0.5 * L * FY
    Ensuring identical moment equilibrium: M1 - M2 + cross(x2 - x1, -F_vec) == 0.
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Horizontal beam along x-axis from (0,0,0) to (2,0,0), L0 = 2.0 m
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Define 6 DOFs for LAW113
    dofs = [
        BeamSpringDOF(stiff=20000.0, damp=10.0, hflag=0),  # Axial
        BeamSpringDOF(stiff=15000.0, damp=8.0, hflag=0),   # Shear Y
        BeamSpringDOF(stiff=15000.0, damp=8.0, hflag=0),   # Shear Z
        BeamSpringDOF(stiff=5000.0, damp=5.0, hflag=0),    # Torsion
        BeamSpringDOF(stiff=8000.0, damp=5.0, hflag=0),    # Bending Y
        BeamSpringDOF(stiff=8000.0, damp=5.0, hflag=0),    # Bending Z
    ]

    mat113 = Law113Params(id=1, rho0=7800.0, dofs=dofs)

    prop = PropType23(
        id=1,
        imass=1,
        area_or_volume=0.02,
        inertia=0.01,
        params={"imass": 1, "area": 0.02, "inertia": 0.01},
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat113, prop)], "model": model}
    )

    log = MessageLog()
    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]))

    # Apply both axial elongation and transverse shear:
    # Node 1 at (0,0,0), Node 2 displaced to (2.10, 0.05, 0.08)
    x_curr = np.array([
        [0.0, 0.0, 0.0],
        [2.10, 0.05, 0.08],
    ], dtype=np.float64)
    v_curr = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 50.0, 80.0],
    ], dtype=np.float64)
    vr_curr = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    ], dtype=np.float64)
    dt = 0.001

    fint = np.zeros_like(x_curr)
    mint = np.zeros_like(x_curr)

    dtc = spring_mat.forces_spring_mat_type23(group, x_curr, v_curr, vr_curr, dt, fint, mint, idx23=np.array([0]))

    # 1. Translational equilibrium: fint[n1] + fint[n2] == 0
    assert np.allclose(fint[0] + fint[1], 0.0, atol=1e-10)

    # 2. Exact Moment Equilibrium for beam transverse shear (r4cum3.F):
    # Node 1 gets internal resisting moment mint[0]
    # Node 2 gets internal resisting moment mint[1]
    # Resisting force on node 2 is fint[1]
    # Moment about node 1: mint[0] + mint[1] + cross(x2 - x1, fint[1]) == 0
    dx = x_curr[1] - x_curr[0]
    net_moment = mint[0] + mint[1] + np.cross(dx, fint[1])
    assert np.allclose(net_moment, 0.0, atol=1e-10)

    # 3. Non-zero transverse shear forces produce non-zero moment couples
    assert np.linalg.norm(mint[0]) > 0.0
    assert np.linalg.norm(mint[1]) > 0.0
    assert group.state["eint"][0] > 0.0
    assert dtc[0] > 0.0


def test_type23_energy_conservation():
    """Verify internal energy matches mechanical work (trapezoidal integration) over a loading cycle.

    Fortran origin: redef3.F90 lines 1144-1145, redef_seatbelt.F90 lines 541-542.
    dE = 0.5 * (F_old + F) * d(delta)
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Linear spring in LAW114 with STIFF=5000 N/m, no damping
    mat = Law114Seatbelt(
        id=1,
        rho0=1000.0,
        params={"STIFF1": 5000.0, "DAMP1": 0.0, "YOUNG": 0.0}
    )

    prop = PropType23(
        id=1,
        imass=1,
        area_or_volume=0.01,
        params={"imass": 1, "area": 0.01},
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat, prop)], "model": model}
    )

    log = MessageLog()
    spring.init_group(group, model, log)
    spring_mat.init_spring_mat_type23(group, model, log, idx23=np.array([0]))

    # Step-wise extension from L=1.0 to L=1.1 in 10 steps
    n_steps = 10
    dL_total = 0.10
    work_accum = 0.0
    f_prev = 0.0

    for step in range(1, n_steps + 1):
        L_step = 1.0 + (step / n_steps) * dL_total
        x_step = np.array([[0.0, 0.0, 0.0], [L_step, 0.0, 0.0]], dtype=np.float64)
        fint = np.zeros_like(x_step)
        mint = np.zeros_like(x_step)

        spring_mat.forces_spring_mat_type23(group, x_step, None, None, 0.001, fint, mint, idx23=np.array([0]))

        f_curr = group.state["force"][0]
        step_dL = dL_total / n_steps
        work_accum += 0.5 * (f_prev + f_curr) * step_dL
        f_prev = f_curr

    # Theoretical elastic energy: 0.5 * K * delta^2 = 0.5 * 5000 * (0.1)^2 = 25.0 J
    expected_energy = 0.5 * 5000.0 * (dL_total ** 2)
    assert group.state["eint"][0] == pytest.approx(expected_energy, rel=1e-3)
    assert group.state["eint"][0] == pytest.approx(work_accum, rel=1e-4)


def test_type23_spring_forces_dispatch():
    """Verify that pyradioss.elements.spring.forces dispatches TYPE23 seamlessly."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    mat = Law114Seatbelt(
        id=1,
        rho0=1000.0,
        params={"STIFF1": 8000.0, "DAMP1": 0.0, "YOUNG": 0.0}
    )

    prop = Property(
        id=1,
        type=23,
        title="SPR_MAT_TEST",
        params={"imass": 1, "area": 0.01, "inertia": 0.001}
    )

    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), mat, prop)], "model": model}
    )

    log = MessageLog()
    spring.init_group(group, model, log)

    assert "idx23" in group.state
    assert len(group.state["idx23"]) == 1

    x = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    dtc = spring.forces(group, x, None, None, 0.001, fint, mint)

    # Delta = 0.05 m, K = 8000 N/m -> F = 400 N
    assert group.state["force"][0] == pytest.approx(400.0, rel=1e-3)
    assert fint[0, 0] == pytest.approx(400.0, rel=1e-3)
    assert fint[1, 0] == pytest.approx(-400.0, rel=1e-3)
    assert dtc[0] > 0.0
