"""Unit tests for Milestone M593: Integrated Fiber Beam Element (/BEAM + /PROP/TYPE18, /PROP/INT_BEAM).

Fortran alignment citations:
- ``starter/source/properties/beam/hm_read_prop18.F``
- ``starter/source/properties/beam/defbeam_sect_new.F90``
- ``engine/source/elements/beam/main_beam18.F``
- ``engine/source/elements/beam/mulaw_ib.F``
- ``engine/source/materials/mat/mat002/m2lawpi.F``
- ``engine/source/elements/beam/fail_beam18.F``

Tests verify:
1. Cross-section fiber generators (rectangular Gauss/Lobatto, circular, tubular, I-beam, discrete fibers).
2. Pure axial tension recovering exact N = E * A * eps_0.
3. Pure elastic bending recovering exact Euler-Bernoulli moments My = E * Iyy * kappa_y and Mz = E * Izz * kappa_z.
4. Progressive plastic yielding under increasing curvature with LAW2 reaching plastic moment Mp = Z * sigma_y.
5. 3D finite rigid-body translation and rotation producing zero strain increments and zero forces.
6. Dynamic explicit time-stepping simulation verifying total energy conservation (E_kin + E_int = const).
7. Starter keyword reading for /PROP/TYPE18 and /PROP/INT_BEAM (fixed & free format, parametric & discrete fibers).
8. Starter element-technology dispatch routing /BEAM elements with prop.type == 18 to model.beams_fiber.
9. Algorithmic tangent stiffness matrix symmetry and 6 rigid-body nullspace modes.
10. Defensive edge cases (dt <= 0, v=None, vr=None, off=0 element deletion).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import beam_fiber
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material, Part, Property, PropType18
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.initialization import build_element_groups, initialize_elements_and_mass


# ----------------------------------------------------------------------------
# Test Fixture Helpers
# ----------------------------------------------------------------------------

def _make_fiber_beam_model(
    coords: List[List[float]],
    conn: List[List[int]],
    isflag: int = 1,
    nitrs: int = 3,
    l_params: Optional[List[float]] = None,
    user_fibers: Optional[List[Tuple[float, float, float]]] = None,
    E: float = 210000.0,
    nu: float = 0.3,
    rho0: float = 7.85e-9,
    law: int = 1,
    jc_params: Optional[dict] = None,
    iref: int = 0,
    y0: float = 0.0,
    z0: float = 0.0,
) -> Tuple[Model, ElementGroup, MessageLog]:
    """Construct a Model and initialized ElementGroup for an integrated fiber beam."""
    if l_params is None:
        l_params = [10.0, 20.0, 0.0, 0.0, 0.0, 0.0]

    model = Model()
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()

    part = Part(id=1, prop_id=1, mat_id=1, title="FIBER_BEAM_PART")
    model.parts[1] = part
    model.parts_list = [part]

    # Generate section data to populate property
    y_pts, z_pts, a_pts, sec_props = beam_fiber.generate_fiber_section(
        isflag=isflag,
        nitrs=nitrs,
        l_params=l_params,
        nip=len(user_fibers) if user_fibers else 0,
        user_fibers=user_fibers,
        iref=iref,
        y0=y0,
        z0=z0,
    )

    p_params = {
        "isflag": isflag,
        "nitrs": nitrs,
        "l_params": l_params,
        "fibers": user_fibers or [],
        "iref": iref,
        "y0": y0,
        "z0": z0,
        "area": sec_props["area"],
        "iyy": sec_props["iyy"],
        "izz": sec_props["izz"],
        "ixx": sec_props["ixx"],
        "zy": sec_props["zy"],
        "zz": sec_props["zz"],
    }
    prop = Property(id=1, type=18, title="PROP_TYPE18", params=p_params)
    model.properties[1] = prop

    mat_params = {"E": E, "nu": nu}
    if law == 2 and jc_params is not None:
        mat_params.update(jc_params)
    mat = Material(id=1, law=law, rho0=rho0, params=mat_params)
    model.materials[1] = mat

    log = MessageLog()
    conn_arr = np.array(conn, dtype=int)
    n_elems = len(conn_arr)
    group = ElementGroup(
        ids=np.arange(1, n_elems + 1, dtype=int),
        conn=conn_arr,
        part=np.zeros(n_elems, dtype=int),
    )
    group.state = {
        "slices": [(slice(0, n_elems), mat, prop)],
    }
    if n_elems > 0:
        beam_fiber.init_group(group, model, log)
    return model, group, log


# ----------------------------------------------------------------------------
# 1. Cross-Section Generator Tests
# ----------------------------------------------------------------------------

def test_m593_fiber_section_generators_rectangular():
    """Verify rectangular section generation with Gauss (ISFLAG=1) and Lobatto (ISFLAG=3)."""
    # 1. Gauss-Legendre quadrature (ISFLAG=1)
    b, h = 10.0, 20.0
    y_pts, z_pts, a_pts, props = beam_fiber.generate_fiber_section(
        isflag=1, nitrs=3, l_params=[b, h, 0, 0, 0, 0]
    )
    assert len(y_pts) == 9
    assert np.isclose(props["area"], b * h)
    # Analytic second moment: Iyy = b * h^3 / 12, Izz = h * b^3 / 12
    analytic_iyy = b * (h**3) / 12.0
    analytic_izz = h * (b**3) / 12.0
    analytic_zy = 0.25 * b * (h**2)
    analytic_zz = 0.25 * h * (b**2)

    # With order 3 Gauss quadrature, polynomials up to degree 2*3-1 = 5 are integrated exactly!
    # z^2 is degree 2, so Iyy and Izz must be exact to machine precision:
    assert np.isclose(props["iyy"], analytic_iyy, rtol=1e-12)
    assert np.isclose(props["izz"], analytic_izz, rtol=1e-12)
    assert np.isclose(props["ixx"], analytic_iyy + analytic_izz, rtol=1e-12)
    # Z_y = int |z| dA: piecewise linear function with kink at 0, order 3 Gauss is within 15%:
    assert np.isclose(props["zy"], analytic_zy, rtol=0.15)
    assert np.isclose(props["zz"], analytic_zz, rtol=0.15)

    # 2. Gauss-Lobatto quadrature (ISFLAG=3)
    b, h = 12.0, 16.0
    y_lob, z_lob, a_lob, props_lob = beam_fiber.generate_fiber_section(
        isflag=3, nitrs=4, l_params=[b, h, 0, 0, 0, 0]
    )
    assert len(y_lob) == 16
    assert np.isclose(props_lob["area"], b * h)
    # Gauss-Lobatto of order 4 integrates polynomials up to degree 2*4-3 = 5 exactly.
    assert np.isclose(props_lob["iyy"], b * (h**3) / 12.0, rtol=1e-12)
    assert np.isclose(props_lob["izz"], h * (b**3) / 12.0, rtol=1e-12)


def test_m593_fiber_section_generators_circular_and_tubular():
    """Verify circular (ISFLAG=2, 17) and tubular (ISFLAG=18) cross-section generation."""
    # Solid circular section
    R = 8.0
    y_pts, z_pts, a_pts, props = beam_fiber.generate_fiber_section(
        isflag=2, nitrs=4, l_params=[R, 0, 0, 0, 0, 0]
    )
    analytic_area = math.pi * R**2
    analytic_ixx = 0.5 * math.pi * R**4
    analytic_iyy = 0.25 * math.pi * R**4
    # Ring discretization converges to exact circle within 2%
    assert np.isclose(props["area"], analytic_area, rtol=1e-4)
    assert np.isclose(props["iyy"], analytic_iyy, rtol=0.02)
    assert np.isclose(props["izz"], analytic_iyy, rtol=0.02)
    assert np.isclose(props["ixx"], analytic_ixx, rtol=0.02)

    # Tubular section (ISFLAG=18)
    Ro, Ri = 10.0, 6.0
    y_tub, z_tub, a_tub, props_tub = beam_fiber.generate_fiber_section(
        isflag=18, nitrs=3, l_params=[Ro, Ri, 0, 0, 0, 0]
    )
    tub_area = math.pi * (Ro**2 - Ri**2)
    tub_iyy = 0.25 * math.pi * (Ro**4 - Ri**4)
    assert np.isclose(props_tub["area"], tub_area, rtol=1e-4)
    assert np.isclose(props_tub["iyy"], tub_iyy, rtol=0.02)
    assert np.isclose(props_tub["izz"], tub_iyy, rtol=0.02)


def test_m593_fiber_section_generators_i_beam_and_discrete():
    """Verify I-beam (ISFLAG=10) and user-defined discrete fibers (ISFLAG=0)."""
    # I-beam
    bf, tf, h, tw = 10.0, 1.5, 20.0, 1.0
    y_ib, z_ib, a_ib, props_ib = beam_fiber.generate_fiber_section(
        isflag=10, nitrs=3, l_params=[bf, tf, h, tw, 0, 0]
    )
    expected_area = 2.0 * bf * tf + (h - 2.0 * tf) * tw
    assert np.isclose(props_ib["area"], expected_area)
    assert props_ib["iyy"] > 0
    assert props_ib["izz"] > 0

    # User-defined fibers (ISFLAG=0)
    user_fibers = [
        (-2.0, -3.0, 1.5),
        (2.0, -3.0, 1.5),
        (-2.0, 3.0, 1.5),
        (2.0, 3.0, 1.5),
    ]
    y_u, z_u, a_u, props_u = beam_fiber.generate_fiber_section(
        isflag=0, nip=4, user_fibers=user_fibers, iref=0
    )
    assert len(y_u) == 4
    assert np.isclose(props_u["area"], 6.0)
    # Barycenter should be (0, 0)
    assert np.isclose(props_u["y0"], 0.0)
    assert np.isclose(props_u["z0"], 0.0)
    # Iyy = sum(z^2 * A) = 4 * (9.0 * 1.5) = 54.0
    # Izz = sum(y^2 * A) = 4 * (4.0 * 1.5) = 24.0
    assert np.isclose(props_u["iyy"], 54.0)
    assert np.isclose(props_u["izz"], 24.0)


# ----------------------------------------------------------------------------
# 2. Pure Axial Tension Physics Test
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_pure_axial_tension():
    """Verify pure axial tension N = E * A * eps_0 to machine precision."""
    coords = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]
    b, h = 10.0, 20.0
    A = b * h
    E = 210000.0

    model, group, log = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[b, h, 0, 0, 0, 0],
        E=E,
        nu=0.3,
        law=1,
    )

    # Prescribe axial velocity at node 2: v2_x = 10.0, dt = 0.001
    # Delta L = v2_x * dt = 0.01. eps_0 = Delta L / L0 = 0.01 / 100 = 1e-4.
    dt = 0.001
    v = np.zeros((3, 3))
    v[1, 0] = 10.0
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)

    # Theoretical axial force: N = E * A * eps_0
    eps_0 = 0.01 / 100.0
    N_theo = E * A * eps_0
    assert np.isclose(N_theo, 210000.0 * 200.0 * 1e-4)

    st = group.state
    # Resultants in local coordinate system
    assert np.isclose(st["fres"][0, 0], N_theo, rtol=1e-12)
    assert np.isclose(st["fres"][0, 1], 0.0, atol=1e-10)
    assert np.isclose(st["fres"][0, 2], 0.0, atol=1e-10)
    assert np.isclose(st["mres"][0, 0], 0.0, atol=1e-10)
    assert np.isclose(st["mres"][0, 1], 0.0, atol=1e-10)
    assert np.isclose(st["mres"][0, 2], 0.0, atol=1e-10)

    # Check global equilibrium: -f1 - f2 = 0
    assert np.allclose(fint[0] + fint[1], 0.0, atol=1e-12)
    # Radioss nodal force sign: fint is accumulated with minus sign, so
    # fint[0, 0] = +N_theo, fint[1, 0] = -N_theo
    assert np.isclose(fint[0, 0], N_theo, rtol=1e-12)
    assert np.isclose(fint[1, 0], -N_theo, rtol=1e-12)

    # Verify internal energy: dE = 0.5 * N_theo * Delta L
    dE_expected = 0.5 * N_theo * 0.01
    assert np.isclose(st["eint"][0], dE_expected, rtol=1e-12)


# ----------------------------------------------------------------------------
# 3. Pure Elastic Bending Tests
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_pure_elastic_bending():
    """Verify pure bending recovering Euler-Bernoulli moments My = E*Iyy*kappa_y and Mz = E*Izz*kappa_z."""
    coords = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]
    b, h = 10.0, 20.0
    E = 200000.0

    # 1. Bending about local Z-axis (curvature kappa_z)
    model, group, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[b, h, 0, 0, 0, 0],
        E=E,
        nu=0.3,
        law=1,
    )
    Izz = group.state["slices"][0][2].params["izz"]

    # Apply pure angular rotation rate kz_dot = (t2_z - t1_z) / L
    dt = 0.002
    omega_z2 = 0.05
    vr = np.zeros((3, 3))
    vr[1, 2] = omega_z2
    v = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)

    L0 = 50.0
    dkz = omega_z2 * dt / L0
    # Fortran convention: deps_xx = -y * dkz, Mz = -sum(sig_xx * y * A) = +E * Izz * dkz
    Mz_theo = E * Izz * dkz

    assert np.isclose(group.state["mres"][0, 2], Mz_theo, rtol=1e-10)
    assert np.isclose(group.state["fres"][0, 0], 0.0, atol=1e-10)

    # 2. Bending about local Y-axis (curvature kappa_y)
    model_y, group_y, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[b, h, 0, 0, 0, 0],
        E=E,
        nu=0.3,
        law=1,
    )
    Iyy = group_y.state["slices"][0][2].params["iyy"]

    omega_y2 = 0.04
    vr_y = np.zeros((3, 3))
    vr_y[1, 1] = omega_y2
    fint_y = np.zeros((3, 3))
    mint_y = np.zeros((3, 3))

    beam_fiber.forces(group_y, model_y.x, v, vr_y, dt, fint_y, mint_y)

    dky = omega_y2 * dt / L0
    # Fortran convention: My = sum(sig_xx * z * A) = E * Iyy * dky
    My_theo = E * Iyy * dky

    assert np.isclose(group_y.state["mres"][0, 1], My_theo, rtol=1e-10)
    assert np.isclose(group_y.state["fres"][0, 0], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 4. Progressive Plastic Yielding Test (LAW2)
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_plastic_yielding_full_section():
    """Verify progressive cross-section plastic yielding under bending to Mp = Z * sigma_y."""
    coords = [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]
    b, h = 10.0, 20.0
    E = 200000.0
    sig_y = 300.0

    # Elastic-perfectly plastic material (LAW2 with B=0)
    jc_params = {"A": sig_y, "B": 0.0, "n": 1.0, "sig_max": sig_y}

    # Higher order quadrature (order 7) along height for fine plastic progression
    model, group, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=7,
        l_params=[b, h, 0, 0, 0, 0],
        E=E,
        nu=0.3,
        law=2,
        jc_params=jc_params,
    )

    sec_props = group.state["slices"][0][2].params
    Iyy = sec_props["iyy"]
    Zy = sec_props["zy"]

    # Elastic limit moment Me = W * sigma_y = (Iyy / (h/2)) * sigma_y
    Me = (Iyy / (0.5 * h)) * sig_y
    # Fully plastic moment Mp = Zy * sigma_y
    Mp = Zy * sig_y
    assert np.isclose(Mp, 0.25 * b * (h**2) * sig_y, rtol=0.04)

    L = 40.0
    dt = 0.001
    v = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    # Apply 100 incremental bending steps that drive the section well into plastic regime
    # Use symmetric rotation rates vr[0, 1] = -w, vr[1, 1] = +w to ensure pure bending (zero transverse shear)
    total_dky = 0.0
    moments_y = []
    eps_p_max = []

    # Curvature to initiate yield at extreme fiber: kappa_e = sig_y / (E * (h/2))
    kappa_e = sig_y / (E * (0.5 * h))
    # Target large curvature 15x beyond yield
    dky_step = (15.0 * kappa_e) / 100.0
    w = (dky_step * L) / (2.0 * dt)

    vr = np.zeros((3, 3))
    vr[0, 1] = -w
    vr[1, 1] = w

    for step in range(100):
        beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)
        my_curr = group.state["mres"][0, 1]
        moments_y.append(my_curr)
        eps_p_max.append(float(np.max(group.state["eps_p"][0])))
        total_dky += dky_step

    # Verify behavior:
    # 1. Moment must strictly increase monotonically (allow -1e-9 for machine roundoff at plateau)
    assert np.all(np.diff(moments_y) >= -1e-9)
    # 2. Plastic strains must accumulate in outer fibers
    assert eps_p_max[-1] > 0.005
    # 3. Final moment must asymptote near Mp and reach plastic moment capacity
    assert moments_y[-1] > Me
    assert np.isclose(moments_y[-1], Mp, rtol=1e-4)


# ----------------------------------------------------------------------------
# 5. Rigid-Body Invariance Test
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_rigid_body_invariance():
    """Verify that finite 3D rigid-body translation and rotation produce machine-zero internal forces."""
    # Arbitrary 3D orientation
    coords = [[12.5, -34.2, 56.1], [62.5, 15.8, -23.9], [15.0, -10.0, 80.0]]
    conn = [[0, 1, 2]]

    model, group, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[8.0, 14.0, 0, 0, 0, 0],
    )

    # 1. Pure rigid-body translation
    v_trans = np.array([25.0, -40.0, 65.0])
    v = np.zeros((3, 3))
    v[0] = v_trans
    v[1] = v_trans
    vr = np.zeros((3, 3))
    dt = 0.001
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)
    assert np.allclose(group.state["fres"], 0.0, atol=1e-12)
    assert np.allclose(group.state["mres"], 0.0, atol=1e-12)
    assert np.allclose(fint, 0.0, atol=1e-12)
    assert np.allclose(mint, 0.0, atol=1e-12)
    assert np.isclose(group.state["eint"][0], 0.0, atol=1e-12)

    # 2. Pure rigid-body rotation about origin with angular velocity omega
    omega = np.array([1.5, -2.2, 3.1])
    x0 = model.x[0]
    x1 = model.x[1]
    v[0] = np.cross(omega, x0)
    v[1] = np.cross(omega, x1)
    vr[0] = omega
    vr[1] = omega
    fint.fill(0.0)
    mint.fill(0.0)

    beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)
    assert np.allclose(group.state["fres"], 0.0, atol=1e-10)
    assert np.allclose(group.state["mres"], 0.0, atol=1e-10)
    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(mint, 0.0, atol=1e-10)
    assert np.isclose(group.state["eint"][0], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 6. Dynamic Explicit Energy Conservation Test (Leapfrog Verlet)
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_dynamic_energy_conservation():
    """Verify total energy conservation (E_kin + E_int = const) in dynamic explicit simulation."""
    coords = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]
    b, h = 10.0, 10.0

    model, group, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[b, h, 0, 0, 0, 0],
        E=200000.0,
        nu=0.3,
        rho0=7.85e-9,
        law=1,
    )

    st = group.state
    dt_crit = float(st["dt0"][0])
    dt = 0.05 * dt_crit

    # Initial condition: node 0 fixed, node 1 with initial velocity v0
    v0_x = 50.0
    m2 = float(st["mass"][0] / 2.0)
    e_kin0 = 0.5 * m2 * (v0_x**2)

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    v[1, 0] = v0_x

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    # Initial force calculation at t=0
    beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)
    a2_x = fint[1, 0] / m2

    # Leapfrog initialization: half-step velocity v^{1/2} = v^0 + 0.5 * a^0 * dt
    v_half = v0_x + 0.5 * a2_x * dt

    # Run 100 explicit central difference leapfrog cycles
    for cycle in range(100):
        # 1. Update position: x^{n+1} = x^n + v^{n+1/2} * dt
        model.x[1, 0] += v_half * dt

        # 2. Evaluate internal forces at x^{n+1}
        fint.fill(0.0)
        mint.fill(0.0)
        v[1, 0] = v_half
        beam_fiber.forces(group, model.x, v, vr, dt, fint, mint)

        # 3. Compute acceleration at t^{n+1}: a^{n+1} = fint / m
        a2_x = fint[1, 0] / m2

        # 4. Advance half-step velocity: v^{n+3/2} = v^{n+1/2} + a^{n+1} * dt
        v_next_half = v_half + a2_x * dt

        # Full-step velocity for kinetic energy at t^{n+1}: v^{n+1} = 0.5 * (v^{n+1/2} + v^{n+3/2})
        v_full = 0.5 * (v_half + v_next_half)
        v_half = v_next_half

    # Check energy at end of simulation
    e_kin_end = 0.5 * m2 * (v_full**2)
    e_int_end = float(st["eint"][0])
    e_tot_end = e_kin_end + e_int_end

    rel_error = abs(e_tot_end - e_kin0) / e_kin0
    # Symplectic leapfrog Verlet conserves total Hamiltonian energy with very high precision
    assert rel_error < 0.02


# ----------------------------------------------------------------------------
# 7. Starter Parsing and Formulation Dispatch Tests
# ----------------------------------------------------------------------------

def test_m593_starter_fixed_format_prop_type18_parsing(tmp_path: Path):
    """Verify fixed-format /PROP/TYPE18 parsing with parametric cross-section."""
    deck_text = (
        "/BEGIN\n"
        "TEST_STARTER_TYPE18\n"
        "/PROP/TYPE18/1\n"
        "Fiber Rectangular Beam\n"
        "#  ISFLAG    ISMSTR\n"
        "         1         0\n"
        "#       DM        DF\n"
        "       0.0       0.0\n"
        "#      NIP      IREF                  Y0                  Z0\n"
        "         1         0                 0.0                 0.0\n"
        "#    NITRS      IREF                  L1                  L2                  L3                  L4\n"
        "         3         0                10.0                20.0                 0.0                 0.0\n"
        "#                 L5                  L6\n"
        "                 0.0                 0.0\n"
        "/END\n"
    )
    f = tmp_path / "TEST_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    deck = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck, model, log)

    assert 1 in model.prop_int_beams
    p18 = model.prop_int_beams[1]
    assert isinstance(p18, PropType18)
    assert p18.isflag == 1
    assert p18.nitrs == 3
    assert np.isclose(p18.l1, 10.0)
    assert np.isclose(p18.l2, 20.0)
    assert np.isclose(p18.area, 200.0)
    assert np.isclose(p18.iyy, 10.0 * (20.0**3) / 12.0)
    assert np.isclose(p18.izz, 20.0 * (10.0**3) / 12.0)
    assert 1 in model.properties
    assert model.properties[1].type == 18


def test_m593_starter_free_format_discrete_fibers_parsing(tmp_path: Path):
    """Verify free-format /PROP/INT_BEAM parsing with discrete fibers (ISFLAG=0)."""
    deck_text = (
        "/BEGIN\n"
        "TEST_FREE_INT_BEAM\n"
        "/PROP/INT_BEAM/2\n"
        "Discrete Fiber Beam\n"
        "0, 0\n"
        "0.0, 0.0\n"
        "4, 0, 0.0, 0.0\n"
        "-2.0, -3.0, 1.5\n"
        " 2.0, -3.0, 1.5\n"
        "-2.0,  3.0, 1.5\n"
        " 2.0,  3.0, 1.5\n"
        "/END\n"
    )
    f = tmp_path / "TEST_FREE_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    deck = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck, model, log)

    assert 2 in model.prop_int_beams
    p18 = model.prop_int_beams[2]
    assert p18.isflag == 0
    assert p18.nip == 4
    assert len(p18.ips) == 4
    assert np.isclose(p18.area, 6.0)
    assert np.isclose(p18.iyy, 54.0)
    assert np.isclose(p18.izz, 24.0)


def test_m593_starter_beam_formulation_dispatch(tmp_path: Path):
    """Verify Starter dispatch divides /BEAM elements into standard and fiber groups."""
    deck_text = (
        "/BEGIN\n"
        "TEST_DISPATCH\n"
        "/NODE\n"
        "         1                 0.0                 0.0                 0.0\n"
        "         2                50.0                 0.0                 0.0\n"
        "         3                 0.0                10.0                 0.0\n"
        "         4               100.0                 0.0                 0.0\n"
        "/MAT/LAW1/1\n"
        "Steel\n"
        "           7.85e-9\n"
        "          210000.0                 0.3\n"
        "/PROP/BEAM/1\n"
        "Standard Beam Prop\n"
        "                 0                 0                 0\n"
        "             100.0            1000.0            1000.0            2000.0\n"
        "/PROP/TYPE18/2\n"
        "Integrated Fiber Beam Prop\n"
        "         1         0\n"
        "       0.0       0.0\n"
        "         1         0                 0.0                 0.0\n"
        "         3         0                10.0                10.0                 0.0                 0.0\n"
        "                 0.0                 0.0\n"
        "/PART/1\n"
        "Part Std Beam\n"
        "         1         1\n"
        "/PART/2\n"
        "Part Fiber Beam\n"
        "         2         1\n"
        "/BEAM/1\n"
        "         1         1         2         3\n"
        "/BEAM/2\n"
        "         2         2         4         3\n"
        "/END\n"
    )
    f = tmp_path / "TEST_DISP_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    deck = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck, model, log)
    build_element_groups(model, log)
    initialize_elements_and_mass(model, log)

    # Element 1 has prop 1 (type 3) -> model.beams
    assert model.beams is not None
    assert model.beams.n == 1
    assert model.beams.ids[0] == 1

    # Element 2 has prop 2 (type 18) -> model.beams_fiber
    assert model.beams_fiber is not None
    assert model.beams_fiber.n == 1
    assert model.beams_fiber.ids[0] == 2
    assert "fibers_y" in model.beams_fiber.state


# ----------------------------------------------------------------------------
# 8. Tangent Stiffness & Implicit Compatibility Tests
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_tangent_stiffness():
    """Verify 12x12 tangent stiffness matrix symmetry and 6 rigid-body nullspace modes."""
    coords = [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]

    model, group, _ = _make_fiber_beam_model(
        coords=coords,
        conn=conn,
        isflag=1,
        nitrs=3,
        l_params=[10.0, 15.0, 0, 0, 0, 0],
    )

    K = beam_fiber.tangent(group, model.x)
    assert K.shape == (1, 12, 12)
    K_elem = K[0]

    # Symmetry
    assert np.allclose(K_elem, K_elem.T, atol=1e-10)

    # Eigenvalue decomposition for rigid body modes
    eigvals = np.linalg.eigvalsh(K_elem)
    # The first 6 eigenvalues should be machine zeros (rigid body modes in 3D)
    rigid_modes = eigvals[:6]
    deform_modes = eigvals[6:]
    assert np.all(np.abs(rigid_modes) < 1e-7)
    assert np.all(deform_modes > 0.0)

    # Test static and implicit routines run without error
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_fiber.static_internal_forces(group, model.x, fint, mint)
    beam_fiber.implicit_internal_forces(group, model.x, fint, mint)


# ----------------------------------------------------------------------------
# 9. Defensive Edge Cases
# ----------------------------------------------------------------------------

def test_m593_beam_fiber_defensive_edge_cases():
    """Verify defensive handling of empty groups, dt <= 0, None vectors, and deactivated elements."""
    # Empty group
    empty_group = ElementGroup(ids=np.array([], dtype=int), conn=np.zeros((0, 3), dtype=int), part=np.array([], dtype=int))
    assert len(beam_fiber.init_group(empty_group, None, None)[0]) == 0
    assert len(beam_fiber.forces(empty_group, np.zeros((0, 3)), None, None, 0.001, np.zeros((0, 3)), np.zeros((0, 3)))) == 0
    assert beam_fiber.tangent(empty_group, np.zeros((0, 3))).shape == (0, 12, 12)

    # Group with deactivated element (off=0)
    coords = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_fiber_beam_model(coords=coords, conn=conn)
    group.state["off"][0] = 0.0  # deleted/failed

    dt = 0.001
    v = np.zeros((3, 3))
    v[1, 0] = 10.0
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    dt_step = beam_fiber.forces(group, model.x, v, None, dt, fint, mint)
    # Deactivated element returns EP30 and produces zero internal force
    assert dt_step[0] == EP30
    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)

    # dt <= 0 returns initial critical dt
    dt_zero = beam_fiber.forces(group, model.x, None, None, 0.0, fint, mint)
    assert len(dt_zero) == 1
