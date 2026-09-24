"""Unit tests for /SENSOR/HIC (Head Injury Criterion) and /SENSOR/NIC (Neck Injury Criterion).

Upstream OpenRadioss Fortran references:
- engine/source/tools/sensor/sensor_hic.F (SUBROUTINE SENSOR_HIC)
- engine/source/tools/sensor/sensor_nic.F (SUBROUTINE SENSOR_NIC)
- starter/source/tools/sensor/read_sensor_hic.F (SUBROUTINE READ_SENSOR_HIC)
- starter/source/tools/sensor/read_sensor_nic.F (SUBROUTINE READ_SENSOR_NIC)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.npcompat import trapezoid
from pyradioss.engine.biomech_sensors import (
    HicParams,
    HicSensor,
    NicParams,
    NicSensor,
    compute_bostrom_nic,
    compute_hic,
    compute_nij,
    sae_cfc_filter,
)
from pyradioss.engine.sensors import Sensors
from pyradioss.model.entities import Sensor
from pyradioss.model.model import Model


# ============================================================================
# 1. HIC Analytical Tests
# ============================================================================

def test_hic_constant_accel():
    """Verify that a constant acceleration a = 10g over 15 ms gives HIC = 0.015 * 10^2.5 ~= 4.7434."""
    g = 9.80665
    dt = 0.0001  # 0.1 ms
    t = np.arange(0.0, 0.015 + dt / 2.0, dt)
    accel = np.full_like(t, 10.0 * g)

    hic_val, t1, t2 = compute_hic(
        time=t,
        accel=accel,
        hic_period=0.015,
        gravity=g,
        time_unit=1.0,
    )
    opt_dt = t2 - t1

    expected_hic = 0.015 * (10.0 ** 2.5)  # 0.015 * 316.227766 = 4.7434165
    assert pytest.approx(hic_val, rel=1e-3) == expected_hic
    assert pytest.approx(opt_dt, rel=1e-2) == 0.015
    assert pytest.approx(t1, abs=1e-4) == 0.0
    assert pytest.approx(t2, abs=1e-4) == 0.015


def test_hic_half_sine_pulse():
    """Verify HIC for a half-sine acceleration pulse: a(t) = A0 * sin(pi * t / T0).

    For A0 = 50g, T0 = 10 ms (0.010 s):
      Average acceleration over full pulse = (2 * A0) / pi = 100 / pi ~= 31.831g.
      HIC over full pulse duration = T0 * (a_avg / g)^2.5 = 0.010 * (100 / pi)^2.5 ~= 57.164.
      However, the optimal sub-window for a half-sine pulse (Chou & Nyquist 1974)
      has width tau ~= 0.707 * T0 ~= 7.07 ms centered at T0/2, giving:
      HIC_max ~= 1.282 * HIC_full ~= 73.30.
    """
    g = 9.80665
    a0 = 50.0 * g
    t0 = 0.010  # 10 ms pulse
    dt = 0.00005  # 0.05 ms sampling
    t = np.arange(0.0, 0.020, dt)

    accel = np.zeros_like(t)
    pulse_mask = (t <= t0)
    accel[pulse_mask] = a0 * np.sin(np.pi * t[pulse_mask] / t0)

    # 1. Full-pulse duration fixed window check
    t_full = t[pulse_mask]
    a_full = accel[pulse_mask]
    avg_full = trapezoid(a_full, t_full) / (g * t0)
    hic_full = t0 * (avg_full ** 2.5)
    expected_full = t0 * (((2.0 * 50.0) / np.pi) ** 2.5)
    assert pytest.approx(hic_full, rel=0.005) == expected_full

    # 2. Optimal sliding window check
    hic_val, t1, t2 = compute_hic(
        time=t,
        accel=accel,
        hic_period=0.015,
        gravity=g,
    )
    opt_dt = t2 - t1

    # Optimal HIC for half-sine is ~73.30 with optimal duration ~0.707 * 10 ms = 7.07 ms
    assert pytest.approx(hic_val, rel=0.01) == 73.30
    assert pytest.approx(opt_dt, abs=0.001) == 0.00707
    assert pytest.approx(0.5 * (t1 + t2), abs=0.0005) == 0.005  # Centered at peak t = 5 ms


def test_hic_sliding_window_detection():
    """Verify that the sliding window optimizer isolates the peak interval in a 100 ms signal."""
    g = 9.80665
    dt = 0.0002
    t = np.arange(0.0, 0.100, dt)
    accel = np.full_like(t, 1.0 * g)  # 1g baseline

    # Add 40g pulse between 30 ms and 45 ms (duration 15 ms)
    pulse_mask = (t >= 0.030) & (t <= 0.045)
    accel[pulse_mask] = 40.0 * g

    hic_val, t1, t2 = compute_hic(
        time=t,
        accel=accel,
        hic_period=0.015,
        gravity=g,
    )
    opt_dt = t2 - t1

    expected_hic = 0.015 * (40.0 ** 2.5)  # 0.015 * 10119.2885 = 151.789
    assert pytest.approx(hic_val, rel=0.02) == expected_hic
    assert pytest.approx(t1, abs=0.002) == 0.030
    assert pytest.approx(t2, abs=0.002) == 0.045


# ============================================================================
# 2. Stateful HicSensor Tests
# ============================================================================

def test_hic_sensor_stateful_update_and_latching():
    """Verify HicSensor incremental state updating and latching behavior matching sensor_hic.F."""
    params = HicParams(
        id=1,
        hic_period=0.015,
        hic_crit=100.0,
        gravity=9.80665,
        npoint=150,
    )
    sensor = HicSensor(params)
    assert not sensor.status

    dt = 0.0001
    g = 9.80665

    # Period 1: Low acceleration (1g) up to 20 ms -> should not trigger (HIC << 100)
    for step in range(200):
        t = (step + 1) * dt
        fired = sensor.update(t, dt, np.array([0.0, 0.0, 1.0 * g]))
        assert not fired

    # Period 2: High acceleration (40g) from 20 ms to 35 ms -> HIC exceeds 100 -> triggers
    for step in range(200, 350):
        t = (step + 1) * dt
        sensor.update(t, dt, np.array([0.0, 0.0, 40.0 * g]))

    assert sensor.status is True
    assert sensor.fire_time is not None
    assert sensor.current_hic >= 100.0

    # Period 3: Return to zero acceleration -> sensor must remain latched
    latched_fire_time = sensor.fire_time
    for step in range(350, 450):
        t = (step + 1) * dt
        fired = sensor.update(t, dt, np.array([0.0, 0.0, 0.0]))
        assert fired is True
        assert sensor.fire_time == latched_fire_time


# ============================================================================
# 3. NIC & Nij Tests
# ============================================================================

def test_compute_nij_formulas():
    """Verify Nij neck injury criterion for tension-flexion and compression-extension."""
    fint_tens = 6806.0
    fint_comp = 6160.0
    mint_flex = 310.0
    mint_ext = 135.0

    # Tension + Flexion (Fz > 0, My > 0)
    nij_tf = compute_nij(
        fz=3403.0,
        my=155.0,
        fint_tens=fint_tens,
        fint_comp=fint_comp,
        mint_flex=mint_flex,
        mint_ext=mint_ext,
    )
    assert pytest.approx(nij_tf, rel=1e-5) == 1.0

    # Compression + Extension (Fz < 0, My < 0)
    nij_ce = compute_nij(
        fz=-3080.0,
        my=-67.5,
        fint_tens=fint_tens,
        fint_comp=fint_comp,
        mint_flex=mint_flex,
        mint_ext=mint_ext,
    )
    assert pytest.approx(nij_ce, rel=1e-5) == 1.0


def test_bostrom_nic_formula():
    """Verify Boström relative kinematics NIC: NIC = 0.2 * a_rel + v_rel^2."""
    a_rel = 15.0  # m/s^2
    v_rel = 2.0   # m/s
    nic = compute_bostrom_nic(a_rel, v_rel)
    assert pytest.approx(nic, rel=1e-6) == 0.2 * 15.0 + 2.0 ** 2  # 3.0 + 4.0 = 7.0


def test_nic_sensor_triggering():
    """Verify NicSensor threshold crossing and latching."""
    params = NicParams(
        id=2,
        nij_max=1.0,
        fint_tens=6806.0,
        fint_comp=6160.0,
        mint_flex=310.0,
        mint_ext=135.0,
    )
    sensor = NicSensor(params)
    assert not sensor.status

    # Subcritical load
    sensor.update(t=0.001, dt=0.001, fz=2000.0, my=50.0)
    assert not sensor.status
    assert sensor.current_nij < 1.0

    # Critical load: Fz = 4000, My = 200 -> Nij = 4000/6806 + 200/310 = 0.5877 + 0.6452 = 1.2329 > 1.0
    sensor.update(t=0.002, dt=0.001, fz=4000.0, my=200.0)
    assert sensor.status is True
    assert sensor.fire_time == 0.002

    # Return to zero load -> latched
    sensor.update(t=0.003, dt=0.001, fz=0.0, my=0.0)
    assert sensor.status is True


def test_sae_cfc_filter_frequency_response():
    """Verify SAE J211 Butterworth CFC filter removes high frequency noise."""
    dt = 0.0001  # 10 kHz sampling
    t = np.arange(0.0, 0.05, dt)

    # Low frequency signal (20 Hz) + High frequency noise (3000 Hz)
    sig_low = 10.0 * np.sin(2.0 * np.pi * 20.0 * t)
    sig_noise = 5.0 * np.sin(2.0 * np.pi * 3000.0 * t)
    sig = sig_low + sig_noise

    filtered = sae_cfc_filter(sig, dt, cfc=600.0)

    # The high frequency 3 kHz amplitude should be heavily attenuated
    # Peak-to-peak of filtered signal should be close to 20.0 (low frequency component)
    ptp_orig = np.ptp(sig)
    ptp_filt = np.ptp(filtered[len(filtered)//4: 3*len(filtered)//4])
    ptp_low = np.ptp(sig_low[len(sig_low)//4: 3*len(sig_low)//4])

    assert ptp_filt < ptp_orig
    assert pytest.approx(ptp_filt, rel=0.1) == ptp_low


# ============================================================================
# 4. Engine Sensors Integration Tests
# ============================================================================

def test_sensors_board_integration():
    """Verify Sensors integration with HIC and NIC sensors."""
    model = Model()
    model.nodes = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)}
    model._id2idx = {1: 0, 2: 1}
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros((2, 3), dtype=float)
    model.dt = 0.0001
    model.nodal_accel = np.zeros((2, 3), dtype=float)
    model.spring_fm = {101: (0.0, 0.0)}

    # Define sensors
    s_hic = Sensor(
        id=10,
        kind="HIC",
        node_id=1,
        hic_period=0.015,
        hic_val=100.0,
        gravity=9.80665,
        tdelay=0.0,
    )
    s_nic = Sensor(
        id=20,
        kind="NIC",
        spring_id=101,
        nij_max=1.0,
        fint_tens=6806.0,
        fint_comp=6160.0,
        mint_flex=310.0,
        mint_ext=135.0,
    )
    model.sensors = [s_hic, s_nic]

    log = MessageLog()
    sensors = Sensors(model, log)
    assert len(sensors) == 2
    assert not sensors.active(10)
    assert not sensors.active(20)

    # Step 1: Subcritical update
    sensors.update(0.001, log)
    assert not sensors.active(10)
    assert not sensors.active(20)

    # Step 2: Trigger NIC by setting spring axial force and moment
    model.spring_fm[101] = (5000.0, 200.0)
    sensors.update(0.002, log)
    assert sensors.active(20)
    assert pytest.approx(sensors.fire_time[20], rel=1e-5) == 0.002

    # Step 3: Trigger HIC by applying high acceleration (40g) over 150 steps
    g = 9.80665
    for step in range(3, 160):
        t = step * 0.0001
        model.nodal_accel[0] = np.array([0.0, 0.0, 40.0 * g])
        sensors.update(t, log)

    assert sensors.active(10)
    assert sensors.shifted_time(10, 0.020) is not None
