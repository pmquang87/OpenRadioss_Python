"""Unit tests for /MONVOL/COMMU1 communicating airbag chambers.

Upstream OpenRadioss Fortran reference:
- starter/source/airbag/hm_read_monvol_type9.F (COMMU1 card reader, ITYPE=9)
- engine/source/airbag/airbaga1.F (lines 380-472, LEAKS AND COMMUNICATIONS)
- engine/source/airbag/airbag2.F (lines 435-565, AIRBAG COMMUNIQUANTS)
"""

import math
import numpy as np
import pytest

from pyradioss.engine.airbag_commu import (
    MonvolCommu1,
    CommuStepResult,
    critical_pressure_ratio,
    choked_flow_function,
    subsonic_flow_function,
    flow_function,
    compute_orifice_flow,
    step_airbag_commu,
    apply_airbag_communications,
)
from pyradioss.model.model import Model
from pyradioss.model.entities import MonitoredVolume


class MockChamber:
    """Mock monitored volume / airbag chamber holding thermodynamic state."""
    def __init__(
        self,
        id: int,
        volume: float = 1.0,
        pressure: float = 101325.0,
        temperature: float = 293.15,
        gamma: float = 1.4,
        r_spec: float = 287.05,
    ):
        self.id = id
        self.volume = float(volume)
        self.gamma = float(gamma)
        self.r_spec = float(r_spec)
        self.cv = self.r_spec / (self.gamma - 1.0)
        self.cp = self.gamma * self.cv
        self.temperature = float(temperature)
        self.pressure = float(pressure)
        # Mass from ideal gas law: P*V = m*R*T
        self.mass = (self.pressure * self.volume) / (self.r_spec * self.temperature)
        # Internal energy: E = m * cv * T
        self.energy = self.mass * self.cv * self.temperature


def test_critical_pressure_ratio_and_flow_functions():
    """Verify analytical critical pressure ratio and choked/subsonic flow functions."""
    gamma = 1.4
    # Analytical: r_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    # (2 / 2.4) ** (1.4 / 0.4) = (0.8333333) ** 3.5 ~= 0.52828178
    r_crit = critical_pressure_ratio(gamma)
    expected_rcrit = (2.0 / 2.4) ** 3.5
    assert math.isclose(r_crit, expected_rcrit, rel_tol=1e-7)
    assert 0.528 < r_crit < 0.529

    # Choked flow function value: (2 / 2.4) ** (2.4 / 0.8) = (5/6)**3 = 125/216 ~= 0.5787037
    psi_choked = choked_flow_function(gamma)
    expected_choked = (2.0 / 2.4) ** 3.0
    assert math.isclose(psi_choked, expected_choked, rel_tol=1e-7)

    # At r = r_crit, subsonic function must match choked value continuously
    psi_sub_at_crit = subsonic_flow_function(r_crit, gamma)
    assert math.isclose(psi_sub_at_crit, psi_choked, rel_tol=1e-6)

    # At r = 1.0 (equal pressure), flow function must be zero
    assert subsonic_flow_function(1.0, gamma) == 0.0


def test_equal_pressure_zero_flow():
    """Verify zero mass and energy flow between chambers at identical pressures."""
    ch1 = MockChamber(id=1, volume=0.5, pressure=200000.0, temperature=300.0)
    ch2 = MockChamber(id=2, volume=0.5, pressure=200000.0, temperature=300.0)

    commu = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.005,  # 50 cm^2
        cd=0.8,
    )

    res = step_airbag_commu(commu, ch1, ch2, dt=0.001, current_time=0.0)

    assert res.mass_flow_rate == 0.0
    assert res.mass_transferred == 0.0
    assert res.energy_transferred == 0.0
    assert res.regime == "zero"
    assert res.direction == "none"

    # Pressures and masses must remain unchanged
    assert math.isclose(ch1.pressure, 200000.0, rel_tol=1e-9)
    assert math.isclose(ch2.pressure, 200000.0, rel_tol=1e-9)
    assert math.isclose(ch1.mass, res.p1 * ch1.volume / (ch1.r_spec * ch1.temperature), rel_tol=1e-9)


def test_pressure_differential_flow_direction():
    """Verify flow direction from high-pressure to low-pressure chamber in bidirectional mode."""
    # Case A: Chamber 1 higher pressure -> flow from 1 to 2
    ch1 = MockChamber(id=1, volume=0.6, pressure=300000.0, temperature=320.0)
    ch2 = MockChamber(id=2, volume=0.4, pressure=100000.0, temperature=290.0)

    commu = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.002,
        cd=0.7,
        direction="bidirectional",
    )

    m1_init, m2_init = ch1.mass, ch2.mass
    res = step_airbag_commu(commu, ch1, ch2, dt=0.001, current_time=0.0)

    assert res.direction == "1to2"
    assert res.mass_flow_rate > 0.0
    assert res.mass_transferred > 0.0
    assert ch1.mass < m1_init
    assert ch2.mass > m2_init
    assert ch1.pressure < 300000.0
    assert ch2.pressure > 100000.0

    # Case B: Chamber 2 higher pressure -> flow from 2 to 1
    ch1 = MockChamber(id=1, volume=0.6, pressure=120000.0, temperature=290.0)
    ch2 = MockChamber(id=2, volume=0.4, pressure=250000.0, temperature=310.0)

    commu_rev = MonvolCommu1(
        id=2,
        chamber1_id=1,
        chamber2_id=2,
        area=0.002,
        cd=0.7,
        direction="bidirectional",
    )

    m1_init, m2_init = ch1.mass, ch2.mass
    res_rev = step_airbag_commu(commu_rev, ch1, ch2, dt=0.001, current_time=0.0)

    assert res_rev.direction == "2to1"
    assert res_rev.mass_flow_rate > 0.0
    assert ch2.mass < m2_init
    assert ch1.mass > m1_init
    assert ch2.pressure < 250000.0
    assert ch1.pressure > 120000.0


def test_check_valve_one_way_direction():
    """Verify check-valve / one-way behavior: permits forward flow and blocks reverse backflow."""
    commu_check = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.002,
        cd=0.8,
        direction="1to2",  # forward only
    )

    # Test reverse condition: P2 > P1
    ch1 = MockChamber(id=1, volume=0.5, pressure=100000.0, temperature=300.0)
    ch2 = MockChamber(id=2, volume=0.5, pressure=300000.0, temperature=300.0)

    res = step_airbag_commu(commu_check, ch1, ch2, dt=0.001, current_time=0.0)
    assert res.direction == "none"
    assert res.mass_flow_rate == 0.0
    assert res.mass_transferred == 0.0
    assert res.regime == "zero"

    # Test forward condition: P1 > P2
    ch1 = MockChamber(id=1, volume=0.5, pressure=300000.0, temperature=300.0)
    ch2 = MockChamber(id=2, volume=0.5, pressure=100000.0, temperature=300.0)

    res_fwd = step_airbag_commu(commu_check, ch1, ch2, dt=0.001, current_time=0.0)
    assert res_fwd.direction == "1to2"
    assert res_fwd.mass_flow_rate > 0.0


def test_choked_vs_subsonic_flow_regimes():
    """Verify flow regime identification: choked (P_down/P_up <= r_crit) vs subsonic (P_down/P_up > r_crit)."""
    r_crit = critical_pressure_ratio(1.4)  # ~0.5283

    # 1. Choked regime: P_down / P_up = 100 kPa / 400 kPa = 0.25 <= r_crit
    ch1_choked = MockChamber(id=1, volume=1.0, pressure=400000.0, temperature=300.0)
    ch2_choked = MockChamber(id=2, volume=1.0, pressure=100000.0, temperature=300.0)

    commu = MonvolCommu1(id=1, chamber1_id=1, chamber2_id=2, area=0.001, cd=1.0)
    res_choked = step_airbag_commu(commu, ch1_choked, ch2_choked, dt=0.001, current_time=0.0)

    assert res_choked.regime == "choked"
    # Analytical mass flow rate for choked flow:
    # mdot = Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2*(gamma-1)))
    expected_mdot_choked = 1.0 * 0.001 * 400000.0 * math.sqrt(1.4 / (287.05 * 300.0)) * choked_flow_function(1.4)
    assert math.isclose(res_choked.mass_flow_rate, expected_mdot_choked, rel_tol=1e-5)

    # 2. Subsonic regime: P_down / P_up = 320 kPa / 400 kPa = 0.80 > r_crit
    ch1_sub = MockChamber(id=1, volume=1.0, pressure=400000.0, temperature=300.0)
    ch2_sub = MockChamber(id=2, volume=1.0, pressure=320000.0, temperature=300.0)

    res_sub = step_airbag_commu(commu, ch1_sub, ch2_sub, dt=0.001, current_time=0.0)
    assert res_sub.regime == "subsonic"
    expected_psi_sub = subsonic_flow_function(0.8, 1.4)
    expected_mdot_sub = 1.0 * 0.001 * 400000.0 * math.sqrt(1.4 / (287.05 * 300.0)) * expected_psi_sub
    assert math.isclose(res_sub.mass_flow_rate, expected_mdot_sub, rel_tol=1e-5)

    # Choked flow rate must exceed subsonic flow rate at same upstream pressure
    assert res_choked.mass_flow_rate > res_sub.mass_flow_rate


def test_total_mass_and_total_energy_conservation():
    """Verify exact conservation of total mass and total internal energy across communicating chambers."""
    v1 = 0.6
    v2 = 0.4
    ch1 = MockChamber(id=1, volume=v1, pressure=450000.0, temperature=360.0)
    ch2 = MockChamber(id=2, volume=v2, pressure=120000.0, temperature=280.0)

    m_tot_initial = ch1.mass + ch2.mass
    e_tot_initial = ch1.energy + ch2.energy

    commu = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.0015,
        cd=0.75,
    )

    dt = 0.0002
    n_steps = 200

    for step in range(n_steps):
        t_curr = step * dt
        res = step_airbag_commu(commu, ch1, ch2, dt=dt, current_time=t_curr)

        m_tot = ch1.mass + ch2.mass
        e_tot = ch1.energy + ch2.energy

        # Conservation to machine floating-point precision
        assert math.isclose(m_tot, m_tot_initial, rel_tol=1e-12, abs_tol=1e-12), f"Mass drift at step {step}"
        assert math.isclose(e_tot, e_tot_initial, rel_tol=1e-12, abs_tol=1e-9), f"Energy drift at step {step}"

    # Verify monotonic convergence towards pressure equalization
    assert ch1.pressure < 450000.0
    assert ch2.pressure > 120000.0
    # Pressure gap must have decreased significantly
    assert (ch1.pressure - ch2.pressure) < (450000.0 - 120000.0)

    # Theoretical equilibrium pressure: P_eq = (gamma - 1) * E_tot / (V1 + V2)
    gamma = 1.4
    p_eq = (gamma - 1.0) * e_tot_initial / (v1 + v2)
    # Both pressures must bracket or approach p_eq
    assert ch1.pressure > p_eq
    assert ch2.pressure < p_eq


def test_opening_time_threshold():
    """Verify delayed communication activation by opening time threshold (Tcom / t_open)."""
    ch1 = MockChamber(id=1, volume=0.5, pressure=300000.0, temperature=300.0)
    ch2 = MockChamber(id=2, volume=0.5, pressure=100000.0, temperature=300.0)

    commu = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.001,
        cd=0.8,
        t_open=0.015,  # opens at t = 15 ms
    )

    assert commu.is_open is False

    # Step 1: t = 0.005 s (< 0.015 s) -> closed, zero flow
    res1 = step_airbag_commu(commu, ch1, ch2, dt=0.005, current_time=0.005)
    assert res1.is_open is False
    assert res1.regime == "closed"
    assert res1.mass_flow_rate == 0.0
    assert ch1.pressure == 300000.0

    # Step 2: t = 0.010 s (< 0.015 s) -> still closed
    res2 = step_airbag_commu(commu, ch1, ch2, dt=0.005, current_time=0.010)
    assert res2.is_open is False
    assert res2.mass_flow_rate == 0.0

    # Step 3: t = 0.015 s (>= 0.015 s) -> activates and flows
    res3 = step_airbag_commu(commu, ch1, ch2, dt=0.005, current_time=0.015)
    assert res3.is_open is True
    assert res3.mass_flow_rate > 0.0
    assert res3.mass_transferred > 0.0
    assert ch1.pressure < 300000.0


def test_opening_pressure_threshold_with_duration():
    """Verify burst membrane activation by pressure differential sustained for duration dtp_def."""
    ch1 = MockChamber(id=1, volume=0.5, pressure=250000.0, temperature=300.0)
    ch2 = MockChamber(id=2, volume=0.5, pressure=100000.0, temperature=300.0)

    # Threshold: dp_def = 100 kPa sustained for dtp_def = 0.004 s
    commu = MonvolCommu1(
        id=1,
        chamber1_id=1,
        chamber2_id=2,
        area=0.001,
        cd=0.8,
        dp_def=100000.0,
        dtp_def=0.004,
    )

    assert commu.is_open is False

    # dt = 0.002 s: dp = 150 kPa >= 100 kPa, but accumulated time = 0.002 s < 0.004 s
    res1 = step_airbag_commu(commu, ch1, ch2, dt=0.002, current_time=0.002)
    assert res1.is_open is False
    assert commu.dp_timer == 0.002

    # Second step: accumulated time reaches 0.004 s -> bursts and activates
    res2 = step_airbag_commu(commu, ch1, ch2, dt=0.002, current_time=0.004)
    assert res2.is_open is True
    assert res2.mass_flow_rate > 0.0
    assert ch1.pressure < 250000.0


def test_model_integration_apply_airbag_communications():
    """Verify integration with pyradioss Model and MonitoredVolume entities."""
    model = Model()

    # Create two MonitoredVolume chambers
    mv1 = MonitoredVolume(
        id=1,
        matid=1,
        pext=101325.0,
        t_initial=300.0,
        iequil=1,
    )
    mv1.volume = 0.8
    mv1.pressure = 350000.0
    mv1.temperature = 300.0
    mv1.r_spec = 287.05
    mv1.gamma = 1.4
    mv1.cv = 287.05 / 0.4
    mv1.mass = (350000.0 * 0.8) / (287.05 * 300.0)
    mv1.energy = mv1.mass * mv1.cv * mv1.temperature

    mv2 = MonitoredVolume(
        id=2,
        matid=1,
        pext=101325.0,
        t_initial=300.0,
        iequil=1,
    )
    mv2.volume = 0.5
    mv2.pressure = 101325.0
    mv2.temperature = 300.0
    mv2.r_spec = 287.05
    mv2.gamma = 1.4
    mv2.cv = 287.05 / 0.4
    mv2.mass = (101325.0 * 0.5) / (287.05 * 300.0)
    mv2.energy = mv2.mass * mv2.cv * mv2.temperature

    model.monitored_volumes[1] = mv1
    model.monitored_volumes[2] = mv2

    # Attach communication
    comm = MonvolCommu1(
        id=10,
        chamber1_id=1,
        chamber2_id=2,
        area=0.002,
        cd=0.75,
    )
    model.monvol_commus[10] = comm

    m_init_sum = mv1.mass + mv2.mass
    e_init_sum = mv1.energy + mv2.energy

    results = apply_airbag_communications(model, dt=0.001, current_time=0.001)

    assert len(results) == 1
    res = results[0]
    assert res.commu_id == 10
    assert res.direction == "1to2"
    assert res.mass_transferred > 0.0

    assert mv1.pressure < 350000.0
    assert mv2.pressure > 101325.0

    # Total conservation through model integration
    assert math.isclose(mv1.mass + mv2.mass, m_init_sum, rel_tol=1e-12)
    assert math.isclose(mv1.energy + mv2.energy, e_init_sum, rel_tol=1e-12)
