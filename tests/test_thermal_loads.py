"""Unit tests for thermal boundary conditions: /CONVEC and /RADIATION.

Tests physics, scaling laws, Fortran subroutine parity, and first-law thermal
energy conservation in lumped thermal systems.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.thermal_loads import (
    ConvecLoad,
    ConvecParams,
    RadiationLoad,
    RadiationParams,
    STEFAN_BOLTZMANN,
    ThermalLoadsManager,
    ThermalStepResult,
    compute_convec_flux,
    compute_radiation_flux,
    compute_segment_area,
    convec_subroutine,
    radiation_subroutine,
    simulate_lumped_cooling,
)


def test_convec_flux_linear():
    """Test linear convection cooling rate vs temperature difference."""
    h = 25.0  # W / (m^2 * K)
    t_inf = 300.0  # K

    # At equilibrium: zero flux
    assert compute_convec_flux(t_surf=300.0, t_inf=t_inf, h=h) == 0.0

    # Linear scaling with temperature difference
    delta_t1 = 20.0
    q1 = compute_convec_flux(t_surf=320.0, t_inf=t_inf, h=h)
    assert math.isclose(q1, h * delta_t1, rel_tol=1e-12)

    delta_t2 = 40.0
    q2 = compute_convec_flux(t_surf=340.0, t_inf=t_inf, h=h)
    assert math.isclose(q2, h * delta_t2, rel_tol=1e-12)
    assert math.isclose(q2, 2.0 * q1, rel_tol=1e-12)

    # Negative flux for heating (T_surf < T_inf)
    q_heat = compute_convec_flux(t_surf=250.0, t_inf=t_inf, h=h)
    assert q_heat < 0.0
    assert math.isclose(q_heat, -50.0 * h, rel_tol=1e-12)

    # Temperature-dependent h(T)
    cp = ConvecParams(h=lambda t: 10.0 + 0.1 * t, t_inf=300.0)
    assert math.isclose(cp.get_h(300.0), 40.0, rel_tol=1e-12)
    assert math.isclose(cp.get_h(400.0), 50.0, rel_tol=1e-12)


def test_radiation_flux_t4_scaling():
    """Test Stefan-Boltzmann T^4 radiation heat loss scaling."""
    eps = 0.85
    sigma = STEFAN_BOLTZMANN
    t_inf = 0.0  # Radiative emission to cold sink (0 K)

    # Zero flux at thermal equilibrium
    assert compute_radiation_flux(t_surf=300.0, t_inf=300.0, emissivity=eps, sigma=sigma) == 0.0

    # Pure T^4 scaling: double temperature quadruples T^2 and 16x T^4
    t1 = 300.0
    t2 = 600.0
    q1 = compute_radiation_flux(t_surf=t1, t_inf=t_inf, emissivity=eps, sigma=sigma)
    q2 = compute_radiation_flux(t_surf=t2, t_inf=t_inf, emissivity=eps, sigma=sigma)

    expected_q1 = eps * sigma * (300.0**4)
    expected_q2 = eps * sigma * (600.0**4)
    assert math.isclose(q1, expected_q1, rel_tol=1e-10)
    assert math.isclose(q2, expected_q2, rel_tol=1e-10)
    assert math.isclose(q2 / q1, 16.0, rel_tol=1e-10)

    # Radiative heating when T_surf < T_inf
    q_in = compute_radiation_flux(t_surf=300.0, t_inf=500.0, emissivity=eps, sigma=sigma)
    assert q_in < 0.0
    assert math.isclose(q_in, eps * sigma * (300.0**4 - 500.0**4), rel_tol=1e-10)

    # Emissivity clamping in [0, 1]
    rp_over = RadiationParams(emissivity=1.5)
    assert rp_over.emissivity == 1.0
    rp_under = RadiationParams(emissivity=-0.2)
    assert rp_under.emissivity == 0.0


def test_segment_geometry_area():
    """Test surface segment area computation for 4-node quads, 3-node triangles, and 2-node lines."""
    coords = {
        1: [0.0, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
        3: [2.0, 2.0, 0.0],
        4: [0.0, 2.0, 0.0],
        5: [0.0, 3.0, 0.0],
        6: [4.0, 0.0, 0.0],
    }

    # 4-node square of side 2: Area = 4.0
    a_quad = compute_segment_area([1, 2, 3, 4], coords)
    assert math.isclose(a_quad, 4.0, rel_tol=1e-12)

    # 3-node triangle with vertices (0,0), (4,0), (0,3): Area = 0.5 * 4 * 3 = 6.0
    a_tri = compute_segment_area([1, 6, 5], coords)
    assert math.isclose(a_tri, 6.0, rel_tol=1e-12)

    # 2-node line from (0,0,0) to (4,0,0): Length = 4.0
    a_line = compute_segment_area([1, 6], coords)
    assert math.isclose(a_line, 4.0, rel_tol=1e-12)

    # Tilted 3D quad: square of side 2 tilted 45 deg about x-axis
    coords_tilted = {
        1: [0.0, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
        3: [2.0, math.sqrt(2.0), math.sqrt(2.0)],
        4: [0.0, math.sqrt(2.0), math.sqrt(2.0)],
    }
    a_tilted = compute_segment_area([1, 2, 3, 4], coords_tilted)
    assert math.isclose(a_tilted, 4.0, rel_tol=1e-10)


def test_nodal_power_distribution():
    """Test nodal heat loss rate distribution Q_node = -q * Area / num_nodes."""
    coords = {
        1: [0.0, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
        3: [2.0, 2.0, 0.0],
        4: [0.0, 2.0, 0.0],
    }
    temps = {1: 350.0, 2: 350.0, 3: 350.0, 4: 350.0}

    # Quad segment: Area = 4.0 m^2, h = 10 W/(m^2 K), T_surf = 350 K, T_inf = 300 K
    # q = 10 * 50 = 500 W / m^2
    # P_loss = 500 * 4 = 2000 W
    # Q_node = -2000 / 4 = -500 W each
    cl = ConvecParams(
        id=1,
        h=10.0,
        t_inf=300.0,
        segments=[[1, 2, 3, 4]],
    )
    mgr = ThermalLoadsManager(convec_loads=[cl])
    res = mgr.compute_step(dt=0.1, time=0.0, temps=temps, coords=coords)

    assert math.isclose(res.power_conv_lost, 2000.0, rel_tol=1e-12)
    for n in (1, 2, 3, 4):
        assert math.isclose(res.q_conv_nodal[n], -500.0, rel_tol=1e-12)
        assert math.isclose(res.total_nodal_power[n], -500.0, rel_tol=1e-12)

    # Nodal power sums to -P_loss
    total_nodal = sum(res.total_nodal_power.values())
    assert math.isclose(total_nodal, -res.power_conv_lost, rel_tol=1e-12)


def test_thermal_loads_manager_time_window_and_energy():
    """Test start/stop time windowing and cumulative dissipated thermal energy."""
    cl = ConvecParams(
        id=1,
        h=10.0,
        t_inf=300.0,
        tstart=1.0,
        tstop=5.0,
        area=2.0,
    )
    rl = RadiationParams(
        id=2,
        emissivity=0.8,
        t_inf=300.0,
        tstart=2.0,
        tstop=6.0,
        area=2.0,
    )
    mgr = ThermalLoadsManager(convec_loads=[cl], radiation_loads=[rl])
    temps = {1: 400.0}

    # At t = 0.5: both inactive
    res0 = mgr.compute_step(dt=0.1, time=0.5, temps=temps)
    assert res0.power_conv_lost == 0.0
    assert res0.power_rad_lost == 0.0
    assert res0.cumul_energy_conv == 0.0
    assert res0.cumul_energy_rad == 0.0

    # At t = 1.5: only convection active
    # q_conv = 10 * (400 - 300) = 1000 W/m^2; P_loss = 1000 * 2 = 2000 W
    res1 = mgr.compute_step(dt=0.5, time=1.5, temps=temps)
    assert math.isclose(res1.power_conv_lost, 2000.0, rel_tol=1e-12)
    assert res1.power_rad_lost == 0.0
    assert math.isclose(res1.cumul_energy_conv, 1000.0, rel_tol=1e-12)

    # At t = 3.0: both active
    res2 = mgr.compute_step(dt=1.0, time=3.0, temps=temps)
    assert math.isclose(res2.power_conv_lost, 2000.0, rel_tol=1e-12)
    assert res2.power_rad_lost > 0.0
    assert math.isclose(res2.cumul_energy_conv, 3000.0, rel_tol=1e-12)


def test_convec_subroutine_fortran_parity():
    """Verify Fortran CONVEC subroutine emulation against OpenRadioss `convec.F`."""
    # 2 segments: 1 quad (nodes 1,2,3,4) and 1 tri (nodes 1,2,3)
    # IBCV: 6 rows, 2 columns
    ibcv = np.array([
        [1, 1],  # n1
        [2, 2],  # n2
        [3, 3],  # n3
        [4, 0],  # n4 (0 indicates triangle)
        [0, 0],  # ifunc
        [0, 0],  # isens
    ], dtype=np.int32)

    # FCONV: 6 rows, 2 columns
    fconv = np.array([
        [300.0, 300.0],  # FCY = T_inf
        [1.0, 1.0],      # FCX
        [20.0, 10.0],    # H
        [0.0, 0.0],      # TSTART
        [100.0, 100.0],  # TSTOP
        [1.0, 1.0],      # OFFG (active)
    ], dtype=np.float64)

    # Nodal coordinates: 4 nodes
    x = np.array([
        [0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    temp = np.array([350.0, 350.0, 350.0, 350.0], dtype=np.float64)
    fthe = np.zeros(4, dtype=np.float64)
    dt = 0.05

    # Segment 1 (quad): Area = 4.0, H = 20, TE = 350, T_INF = 300
    # FLUX1 = Area * H * (T_INF - TE) * dt = 4.0 * 20 * (-50) * 0.05 = -200 J
    # Added to nodes 1..4: -50 J each
    # Segment 2 (tri nodes 1,2,3): Area = 0.5 * 2 * 2 = 2.0, H = 10, TE = 350
    # FLUX2 = 2.0 * 10 * (-50) * 0.05 = -50 J
    # Added to nodes 1..3: -50/3 J each
    heat_conv = convec_subroutine(ibcv, fconv, x, temp, fthe, dt=dt)

    assert math.isclose(heat_conv, -250.0, rel_tol=1e-12)
    # Check node 4 receives only quad contribution: -50 J
    assert math.isclose(fthe[3], -50.0, rel_tol=1e-12)
    # Check node 1 receives quad + tri: -50 + (-50/3) = -66.6667 J
    assert math.isclose(fthe[0], -50.0 - 50.0 / 3.0, rel_tol=1e-12)


def test_radiation_subroutine_fortran_parity():
    """Verify Fortran RADIATION subroutine emulation against OpenRadioss `radiation.F`."""
    ibcr = np.array([
        [1],
        [2],
        [3],
        [4],
        [0],
        [0],
    ], dtype=np.int32)

    # Emissivity = 0.9, sigma = STEFAN_BOLTZMANN -> EMISIG = 0.9 * sigma
    emisig = 0.9 * STEFAN_BOLTZMANN
    fradia = np.array([
        [300.0],   # T_inf
        [1.0],
        [emisig],  # EMISIG
        [0.0],
        [100.0],
        [1.0],     # OFFG
    ], dtype=np.float64)

    x = np.array([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    temp = np.array([500.0, 500.0, 500.0, 500.0], dtype=np.float64)
    fthe = np.zeros(4, dtype=np.float64)
    dt = 0.1

    # Area = 1.0, TE = 500, T_INF = 300
    expected_flux = 1.0 * emisig * (300.0**4 - 500.0**4) * dt
    heat_rad = radiation_subroutine(ibcr, fradia, x, temp, fthe, dt=dt)

    assert math.isclose(heat_rad, expected_flux, rel_tol=1e-10)
    assert math.isclose(fthe[0], expected_flux / 4.0, rel_tol=1e-10)


def test_lumped_thermal_mass_convection_cooling_analytical():
    """Test heat energy conservation and analytical exponential cooling in lumped thermal mass.

    Governing ODE:
      M * cp * dT/dt = -h * A * (T - T_inf)
    Analytical solution:
      T(t) = T_inf + (T_init - T_inf) * exp( - h * A / (M * cp) * t )
    First law:
      Delta E_thermal + E_conv = 0
    """
    m_therm = 2.0  # kg
    cp = 500.0     # J / (kg * K) -> C_th = 1000 J / K
    h = 20.0       # W / (m^2 * K)
    area = 1.0     # m^2
    t_init = 400.0 # K
    t_inf = 300.0  # K
    t_end = 50.0   # s
    dt = 0.01      # s

    cl = ConvecParams(id=1, h=h, t_inf=t_inf, area=area)
    sim = simulate_lumped_cooling(
        m_therm=m_therm,
        cp=cp,
        t_init=t_init,
        convec=cl,
        t_end=t_end,
        dt=dt,
    )

    times = sim["times"]
    temps = sim["temps"]
    e_conv = sim["e_conv"]

    # Characteristic thermal time constant tau = (M * cp) / (h * A) = 1000 / 20 = 50 s
    tau = (m_therm * cp) / (h * area)
    t_analytical = t_inf + (t_init - t_inf) * np.exp(-times / tau)

    # Verify numerical solution matches analytical solution within < 0.2 K
    max_temp_diff = np.max(np.abs(temps - t_analytical))
    assert max_temp_diff < 0.2

    # Verify First Law of Thermodynamics:
    # Heat lost by mass = C_th * (T_init - T(t)) == E_conv(t)
    delta_e_mass = (m_therm * cp) * (t_init - temps)
    energy_conservation_err = np.max(np.abs(delta_e_mass - e_conv))
    # Relative energy error should be under 0.05%
    rel_err = energy_conservation_err / ((m_therm * cp) * (t_init - t_inf))
    assert rel_err < 5.0e-4


def test_lumped_thermal_mass_combined_convection_and_radiation():
    """Test simultaneous convection and radiation cooling with strict energy conservation."""
    m_therm = 1.5   # kg
    cp = 450.0      # J / (kg * K)
    t_init = 800.0  # K (hot body)
    t_inf = 300.0   # K
    area = 0.5      # m^2

    cl = ConvecParams(id=1, h=15.0, t_inf=t_inf, area=area)
    rl = RadiationParams(id=2, emissivity=0.85, t_inf=t_inf, area=area)

    sim = simulate_lumped_cooling(
        m_therm=m_therm,
        cp=cp,
        t_init=t_init,
        convec=cl,
        radiation=rl,
        t_end=30.0,
        dt=0.005,
    )

    temps = sim["temps"]
    e_conv = sim["e_conv"]
    e_rad = sim["e_rad"]
    c_th = m_therm * cp

    # Body must cool down monotonically
    assert temps[-1] < temps[0]
    assert np.all(np.diff(temps) <= 0.0)

    # Both convection and radiation must contribute positive dissipated energy
    assert e_conv[-1] > 0.0
    assert e_rad[-1] > 0.0

    # First Law of Thermodynamics: E_lost_by_mass == E_conv + E_rad
    delta_e_mass = c_th * (t_init - temps)
    total_dissipated = e_conv + e_rad
    max_energy_err = np.max(np.abs(delta_e_mass - total_dissipated))
    rel_energy_err = max_energy_err / (c_th * (t_init - t_inf))
    assert rel_energy_err < 1.0e-3


def test_engine_init_exports():
    """Verify clean top-level imports from pyradioss.engine."""
    import pyradioss.engine as pe

    assert hasattr(pe, "ConvecParams")
    assert hasattr(pe, "ConvecLoad")
    assert hasattr(pe, "RadiationParams")
    assert hasattr(pe, "RadiationLoad")
    assert hasattr(pe, "ThermalLoadsManager")
    assert hasattr(pe, "STEFAN_BOLTZMANN")
    assert hasattr(pe, "simulate_lumped_cooling")
