"""Tests for extended MONVOL physics and airbag features:
- MONVOL/GAS isothermal expansion/compression (p*V = const)
- MONVOL/PRES prescribed pressure curve tracking
- /LEAK mass outflow rate and mass balance
- MONVOL/AIRBAG1 volume change work and 1st law of thermodynamics (dE = -p*dV)
- MONVOL/GAS isentropic compression (T*V^(gamma-1) = const)
- FVM injector jetting forces and reaction thrust projection
"""

import math
import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import (
    MonvolGas,
    MonvolPres,
    MonitoredVolume,
    LeakMat,
    FvmInjector,
    Surface,
    Material,
)
from pyradioss.engine.airbag import (
    update_airbag_thermodynamics,
    update_monvol_gas,
    update_monvol_pres,
    apply_leak_flow,
)
from pyradioss.engine.airbag_fvm import apply_injector_jetting_forces


def test_monvol_gas_isothermal():
    """Test 1: MONVOL/GAS at constant temperature satisfies p*V = const.

    Upstream Fortran reference: engine/source/airbag/volpvg.F (volpvga, lines 122-130).
    For ideal gas with constant T and constant m:
    p * V = m * R * T = constant.
    """
    model = Model()
    # Create MONVOL/GAS volume
    mv = MonvolGas(
        id=1,
        gamma=1.4,
        tini=300.0,
        rho_gas=1.2,
        pext=100000.0,
    )
    mv.isothermal = True
    mv.volume = 1.0  # Initial volume V0 = 1.0 m^3

    # Step 0: Initial state at t=0
    update_monvol_gas(mv, model, dt=0.001, current_time=0.0)

    p0 = mv.pressure
    v0 = mv.volume
    t0 = mv.temperature
    pv_const = p0 * v0

    assert p0 > 0.0
    assert t0 == pytest.approx(300.0, rel=1e-5)
    assert mv.mass == pytest.approx(1.2, rel=1e-5)

    # Step 1: Compress volume by 20% (V = 0.8 m^3)
    mv.volume = 0.8
    update_monvol_gas(mv, model, dt=0.001, current_time=0.001)

    assert mv.temperature == pytest.approx(300.0, rel=1e-5)
    assert mv.pressure * mv.volume == pytest.approx(pv_const, rel=1e-4)
    assert mv.pressure == pytest.approx(p0 * (1.0 / 0.8), rel=1e-4)

    # Step 2: Compress volume to half (V = 0.5 m^3)
    mv.volume = 0.5
    update_monvol_gas(mv, model, dt=0.001, current_time=0.002)

    assert mv.temperature == pytest.approx(300.0, rel=1e-5)
    assert mv.pressure * mv.volume == pytest.approx(pv_const, rel=1e-4)
    assert mv.pressure == pytest.approx(2.0 * p0, rel=1e-4)

    # Step 3: Expand volume to 1.5 m^3
    mv.volume = 1.5
    update_monvol_gas(mv, model, dt=0.001, current_time=0.003)

    assert mv.temperature == pytest.approx(300.0, rel=1e-5)
    assert mv.pressure * mv.volume == pytest.approx(pv_const, rel=1e-4)
    assert mv.pressure == pytest.approx(p0 / 1.5, rel=1e-4)


def test_monvol_pres_follows_curve():
    """Test 2: MONVOL/PRES follows prescribed pressure vs time curve.

    Upstream Fortran reference: engine/source/airbag/volpfv.F (volpfv, lines 31-112).
    """
    model = Model()
    # Define pressure vs time curve (t, P)
    t_samples = [0.0, 0.01, 0.02, 0.05, 0.10]
    p_samples = [100000.0, 150000.0, 220000.0, 180000.0, 120000.0]

    func = FunctTable(fct_id=1, x=np.array(t_samples), y=np.array(p_samples))
    model.functions[1] = func

    mv = MonvolPres(
        id=1,
        fct_id=1,
        fscale=1.0,
        p_ext=100000.0,
    )
    mv.volume = 0.5

    # Check exact matching at sampled times
    for t_val, expected_p in zip(t_samples, p_samples):
        update_monvol_pres(mv, model, dt=0.001, current_time=t_val)
        assert mv.pressure == pytest.approx(expected_p, rel=1e-6)

    # Check interpolation at midpoint t = 0.015
    # (between t=0.01 with P=150k and t=0.02 with P=220k -> P=185k)
    update_monvol_pres(mv, model, dt=0.001, current_time=0.015)
    assert mv.pressure == pytest.approx(185000.0, rel=1e-4)

    # Check scale factor fscale = 2.0
    mv.fscale = 2.0
    update_monvol_pres(mv, model, dt=0.001, current_time=0.02)
    assert mv.pressure == pytest.approx(2.0 * 220000.0, rel=1e-6)


def test_leak_mass_balance():
    """Test 3: /LEAK mass outflow rate and mass balance.

    Upstream Fortran reference: engine/source/airbag/volpvg.F (volpvgb, lines 277-457)
    and engine/source/airbag/fvvent0.F (lines 189-216).
    Checks that total mass decreases at correct rate dm = m_dot * dt.
    """
    model = Model()
    # Monitored volume with pressure higher than external pressure
    mv = MonvolGas(
        id=1,
        gamma=1.4,
        tini=300.0,
        rho_gas=2.0,
        pext=100000.0,
    )
    mv.volume = 1.0
    mv.mass = 2.0  # 2.0 kg
    mv.pressure = 200000.0  # 2 bar absolute vs 1 bar external
    mv.leak_area = 0.0005   # 5 cm^2 orifice area

    dt = 0.001  # 1 ms
    initial_mass = mv.mass

    # Step 1: Single leak flow step
    dm1 = apply_leak_flow(mv, model, dt=dt, current_time=0.0)

    assert dm1 > 0.0
    assert mv.mass == pytest.approx(initial_mass - dm1, rel=1e-8)
    assert mv.dm_leak == pytest.approx(dm1, rel=1e-8)
    assert mv.mass_flow_out == pytest.approx(dm1 / dt, rel=1e-8)

    # Verify theoretical isentropic outflow rate
    # P = 200 kPa, Pext = 100 kPa -> choked flow check:
    # Pcrit = 200e3 * (2/2.4)**(1.4/0.4) = 200e3 * 0.52828 = 105.65 kPa
    # Since Pext < Pcrit, flow is sonically choked with Pe = Pcrit!
    gamma = 1.4
    pcrit = 200000.0 * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    rho0 = 2.0  # density
    u_throat = math.sqrt(2.0 * gamma / (gamma - 1.0) * (200000.0 / rho0) * (1.0 - (pcrit / 200000.0) ** ((gamma - 1.0) / gamma)))
    rho_throat = rho0 * (pcrit / 200000.0) ** (1.0 / gamma)
    m_dot_theory = 0.0005 * rho_throat * u_throat

    assert mv.mass_flow_out == pytest.approx(m_dot_theory, rel=1e-4)

    # Step 2: Multi-step integration mass conservation
    total_dm = dm1
    current_mass = mv.mass
    for step in range(10):
        # Keep pressure constant to verify steady decrement
        mv.pressure = 200000.0
        dm_step = apply_leak_flow(mv, model, dt=dt, current_time=(step + 1) * dt)
        total_dm += dm_step
        current_mass -= dm_step
        assert mv.mass == pytest.approx(current_mass, rel=1e-8)

    assert initial_mass - mv.mass == pytest.approx(total_dm, rel=1e-8)


def test_airbag_volume_change_work():
    """Test 4: MONVOL volume change work satisfies 1st law of thermodynamics (dE = -p*dV).

    Upstream Fortran reference: engine/source/airbag/airbag1.F (lines 35-140)
    and engine/source/airbag/volpvg.F (volpvga lines 111-121).
    For closed adiabatic system without heat/mass transfer:
    dE = -p * dV = -dW  ==>  delta_E + W = 0.
    """
    model = Model()
    # Material for AIRBAG1
    mat = Material(id=1, law=19, title="AirbagGas")
    mat.r_spec = 287.05
    mat.params["CPA"] = 1004.0
    mat.params["CPB"] = 0.0
    mat.params["CPC"] = 0.0
    model.materials[1] = mat

    mv = MonitoredVolume(
        id=1,
        vol_type="AIRBAG1",
        matid=1,
        pext=100000.0,
        t_initial=300.0,
    )
    mv.volume = 1.0  # Initial volume V0 = 1.0 m^3

    # Step 0: Initialize
    update_airbag_thermodynamics(mv, model, dt=0.001, current_time=0.0)
    e_initial = mv.energy
    w_initial = getattr(mv, "work", 0.0)

    assert e_initial > 0.0
    assert w_initial == 0.0

    # Compress in 10 small steps from V=1.0 to V=0.9
    n_steps = 10
    v_start = 1.0
    v_end = 0.9
    for i in range(1, n_steps + 1):
        mv.volume = v_start + (v_end - v_start) * (i / n_steps)
        update_airbag_thermodynamics(mv, model, dt=0.001, current_time=i * 0.001)

    delta_e = mv.energy - e_initial
    total_w = mv.work

    # By 1st law of thermodynamics: delta_E + W = 0
    # The work is negative during compression (dV < 0), increasing internal energy delta_E > 0
    assert delta_e > 0.0
    assert total_w < 0.0
    assert abs(delta_e + total_w) / e_initial < 1e-3  # Exact within 0.1% for midpoint work


def test_monvol_gas_isentropic():
    """Test 5: MONVOL/GAS adiabatic isentropic compression satisfies T*V^(gamma-1) = const.

    Upstream Fortran reference: engine/source/airbag/volpvg.F (volpvga lines 131-139).
    For ideal gas with isentropic index gamma:
    T1 * V1^(gamma - 1) = T0 * V0^(gamma - 1)
    p1 * V1^gamma = p0 * V0^gamma
    """
    model = Model()
    gamma = 1.4
    t0 = 300.0
    v0 = 1.0

    mv = MonvolGas(
        id=1,
        gamma=gamma,
        tini=t0,
        rho_gas=1.2,
        pext=100000.0,
    )
    mv.isentropic = True
    mv.volume = v0

    # Initial state
    update_monvol_gas(mv, model, dt=0.001, current_time=0.0)
    p0 = mv.pressure
    tv_const = t0 * (v0 ** (gamma - 1.0))
    pv_gamma_const = p0 * (v0 ** gamma)

    assert mv.temperature == pytest.approx(t0, rel=1e-5)

    # Compress from V = 1.0 to V = 0.5 (compression ratio 2:1)
    mv.volume = 0.5
    update_monvol_gas(mv, model, dt=0.001, current_time=0.001)

    t1 = mv.temperature
    p1 = mv.pressure
    v1 = mv.volume

    # Theoretical values:
    # T1 = T0 * (V0/V1)^(gamma - 1) = 300 * (2)^0.4 = 300 * 1.3195079 = 395.852 K
    t_theory = t0 * (2.0 ** 0.4)
    # P1 = P0 * (V0/V1)^gamma = P0 * (2)^1.4 = P0 * 2.6390158
    p_theory = p0 * (2.0 ** 1.4)

    assert t1 == pytest.approx(t_theory, rel=1e-5)
    assert p1 == pytest.approx(p_theory, rel=1e-5)

    # Verify isentropic invariants
    assert t1 * (v1 ** (gamma - 1.0)) == pytest.approx(tv_const, rel=1e-5)
    assert p1 * (v1 ** gamma) == pytest.approx(pv_gamma_const, rel=1e-5)

    # Further compression to V = 0.25 (compression ratio 4:1)
    mv.volume = 0.25
    update_monvol_gas(mv, model, dt=0.001, current_time=0.002)

    t2 = mv.temperature
    p2 = mv.pressure
    v2 = mv.volume

    assert t2 * (v2 ** (gamma - 1.0)) == pytest.approx(tv_const, rel=1e-5)
    assert p2 * (v2 ** gamma) == pytest.approx(pv_gamma_const, rel=1e-5)


def test_injector_jetting_forces():
    """Test 6: FvmInjector jetting thrust projection onto FE mesh.

    Upstream Fortran reference: engine/source/airbag/fvbag1.F (lines 904-933)
    and engine/source/airbag/fvinjt6.F (lines 70-131).
    Total thrust magnitude F_thrust = m_dot * v_jet.
    Forces must satisfy Newton's 3rd law on nozzle boundary nodes.
    """
    model = Model()

    # Create a 4-node quad nozzle plate in z = 0 plane (1m x 1m, area = 1.0 m^2)
    # Nodes 0, 1, 2, 3
    x_nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    model.x = x_nodes

    # Surface 1 with one quad segment (normal points in +z: (0, 0, 1))
    surf = Surface(id=1, title="NozzleSurface")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=int)
    model.surfaces[1] = surf

    # Injector with mass flow = 2.5 kg/s and jet velocity = 120.0 m/s
    injector = FvmInjector(
        id=1,
        surf_id=1,
        mass_flow=2.5,
        scale_vel=120.0,
    )

    fext = np.zeros((4, 3), dtype=float)

    # Apply jetting reaction forces
    apply_injector_jetting_forces(injector, fext, model)

    # Theoretical thrust: F = m_dot * v_jet = 2.5 * 120.0 = 300.0 N in +z direction
    expected_thrust = 2.5 * 120.0  # 300 N
    total_fz = np.sum(fext[:, 2])

    assert total_fz == pytest.approx(expected_thrust, rel=1e-5)
    assert np.all(fext[:, :2] == pytest.approx(0.0))  # fx = fy = 0

    # Each of the 4 nodes receives equal share (75 N each)
    for n in range(4):
        assert fext[n, 2] == pytest.approx(expected_thrust / 4.0, rel=1e-5)
