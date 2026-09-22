"""Unit tests for /RWALL/THERM thermal rigid walls with frictional contact heating.

Upstream Fortran references:
- hm_read_rwall_therm.F
- rgwal0.F (RGWALT)
"""

import math
import numpy as np
import pytest

from pyradioss.engine.rwall_thermal import (
    RwallThermal,
    RwallThermalParams,
)


def test_rwall_thermal_params_effusivity_and_partition():
    """Verify thermal effusivity and structural partition fraction calculation."""
    # e = sqrt(k * rho * cp)
    # k_w = 50, rho_w = 7800, cp_w = 500 -> e_w = sqrt(50 * 7800 * 500) = sqrt(1.95e8) = 13964.24
    # k_s = 50, rho_s = 7800, cp_s = 500 -> e_s = 13964.24
    # f_struct = e_s / (e_s + e_w) = 0.5
    params = RwallThermalParams(
        k_w=50.0, rho_w=7800.0, cp_w=500.0,
        k_struct=50.0, rho_struct=7800.0, cp_struct=500.0,
    )
    assert math.isclose(params.effusivity_wall, params.effusivity_struct, rel_tol=1e-5)
    assert math.isclose(params.partition_struct, 0.5, rel_tol=1e-5)

    # Asymmetric materials: copper structure on steel wall
    # copper: k=400, rho=8900, cp=385 -> e_cu = sqrt(400 * 8900 * 385) = 37024.3
    # steel:  e_st = 13964.24
    # f_struct = 37024.3 / (37024.3 + 13964.24) = 0.726
    params_cu = RwallThermalParams(
        k_struct=400.0, rho_struct=8900.0, cp_struct=385.0,
        k_w=50.0, rho_w=7800.0, cp_w=500.0,
    )
    e_cu = math.sqrt(400.0 * 8900.0 * 385.0)
    e_st = math.sqrt(50.0 * 7800.0 * 500.0)
    expected_f = e_cu / (e_cu + e_st)
    assert math.isclose(params_cu.partition_struct, expected_f, rel_tol=1e-5)


def test_rwall_thermal_conductance():
    """Verify thermal conductance from thermal resistance: h = 1 / R_th."""
    params = RwallThermalParams(thermal_resistance=0.002)
    assert math.isclose(params.conductance, 500.0, rel_tol=1e-5)


def test_frictional_heating_power_and_energy_conservation():
    """Verify Coulomb sliding frictional heating and First Law energy conservation."""
    params = RwallThermalParams(slide=2, fric=0.3, t_wall0=300.0, f_struct=0.6)
    wall = RwallThermal(params)

    f_normal = 1000.0  # N
    v_rel_tan = 10.0   # m/s
    dt = 0.01          # s
    node_mass = 2.0    # kg
    cp = 500.0         # J/(kg*K)

    # P_fric = mu * F_N * v_rel = 0.3 * 1000 * 10 = 3000 W
    # E_fric = 3000 * 0.01 = 30 J
    step = wall.apply_contact_step(
        node_id=1,
        f_normal=f_normal,
        v_rel_tan=v_rel_tan,
        node_mass=node_mass,
        dt=dt,
        cp_node=cp,
        node_temp_init=300.0,  # Same as wall -> Q_cond = 0
    )

    assert math.isclose(step["p_fric"], 3000.0, rel_tol=1e-5)
    assert math.isclose(step["e_fric"], 30.0, rel_tol=1e-5)
    assert math.isclose(step["q_node_fric"], 1800.0, rel_tol=1e-5)  # 0.6 * 3000
    assert math.isclose(step["q_wall_fric"], 1200.0, rel_tol=1e-5)  # 0.4 * 3000

    # Energy conservation check
    assert math.isclose(step["e_th_node"] + step["e_th_wall"], step["e_fric"], rel_tol=1e-9)
    assert math.isclose(wall.e_th_struct + wall.e_th_wall, wall.e_fric_total, rel_tol=1e-9)

    # Temperature rise: dT = Q_node * dt / (m * cp) = 1800 * 0.01 / (2 * 500) = 18 / 1000 = 0.018 K
    assert math.isclose(step["dt_node"], 0.018, rel_tol=1e-5)
    assert math.isclose(step["t_node"], 300.018, rel_tol=1e-5)


def test_interface_conductive_heat_transfer():
    """Verify conductive heat transfer between hot wall and cooler node."""
    # Hot wall at 500 K, cold node at 300 K, R_th = 0.01 -> h = 100 W/(m^2*K)
    # Area = 0.1 m^2 -> Q_cond = 100 * 0.1 * (500 - 300) = 2000 W
    params = RwallThermalParams(
        slide=0,  # Frictionless -> P_fric = 0
        t_wall0=500.0,
        thermal_resistance=0.01,
        area_contact=0.1,
    )
    wall = RwallThermal(params)

    step = wall.apply_contact_step(
        node_id=1,
        f_normal=100.0,
        v_rel_tan=0.0,
        node_mass=1.0,
        dt=0.05,
        cp_node=1000.0,
        node_temp_init=300.0,
    )

    assert math.isclose(step["p_fric"], 0.0, abs_tol=1e-9)
    assert math.isclose(step["q_cond"], 2000.0, rel_tol=1e-5)
    # dT = 2000 * 0.05 / (1.0 * 1000.0) = 0.1 K
    assert math.isclose(step["dt_node"], 0.1, rel_tol=1e-5)
    assert math.isclose(step["t_node"], 300.1, rel_tol=1e-5)


def test_vectorized_thermal_coupling():
    """Verify apply_thermal_coupling across multiple nodes simultaneously."""
    params = RwallThermalParams(slide=2, fric=0.2, t_wall0=350.0, f_struct=0.5)
    wall = RwallThermal(params)

    nodes = [10, 20, 30]
    fn = [500.0, 1000.0, 1500.0]
    vt = [2.0, 4.0, 6.0]
    masses = [1.0, 2.0, 3.0]
    dt = 0.001

    results = wall.apply_thermal_coupling(
        node_indices=nodes,
        f_normal_arr=fn,
        v_rel_tan_arr=vt,
        mass_arr=masses,
        dt=dt,
    )

    assert len(results) == 3
    # P_fric[0] = 0.2 * 500 * 2 = 200 W
    assert math.isclose(results[0]["p_fric"], 200.0, rel_tol=1e-5)
    # P_fric[1] = 0.2 * 1000 * 4 = 800 W
    assert math.isclose(results[1]["p_fric"], 800.0, rel_tol=1e-5)
    # P_fric[2] = 0.2 * 1500 * 6 = 1800 W
    assert math.isclose(results[2]["p_fric"], 1800.0, rel_tol=1e-5)
