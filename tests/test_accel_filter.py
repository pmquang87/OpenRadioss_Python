"""
Unit tests for /ACCEL 4th-order Butterworth digital filter (SAE J211 / ISO 6487).

Upstream Fortran reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\tools\\accele\\accel1.F
    Subroutine ACCEL1
  - starter/source/tools/accele/lecacc.F

Tests:
1. DC gain is exactly 1.0 (constant acceleration passes through unmodified).
2. High-frequency attenuation (> f_c frequency attenuated by > 20 dB).
3. All 4 standard CFC classes (CFC60, CFC180, CFC600, CFC1000).
4. Skew coordinate frame transformation (a_local = R @ a_global).
5. Velocity integration: v_filtered += a_filtered * dt.
6. Direct Fortran subroutine ACCEL1 parity.
7. Frequency response magnitude at cutoff f_c (-3 dB / 1/sqrt(2)).
8. Pass-through when fc = 0 (FF == 0 in Fortran).
9. Nyquist frequency capping (f_eff <= 0.4 / dt).
10. Batch time series filtering equivalence (filter_series vs step-by-step).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.accel_filter import (
    AccelFilter,
    FilterCoefficients,
    accel1,
    compute_filter_coefficients,
    CFC_CUTOFF_FREQUENCIES,
    parse_cfc,
)


# ============================================================================
# 1. DC Gain Verification
# ============================================================================

def test_dc_gain_analytical_unity():
    """Verify that analytical transfer function H(z=1) = B(1) / A(1) == 1.0 exactly."""
    dt = 1.0e-5
    for cfc_name, fc in CFC_CUTOFF_FREQUENCIES.items():
        if isinstance(cfc_name, str) and not cfc_name.startswith("CFC"):
            continue
        coeffs = compute_filter_coefficients(fc, dt)

        # Stage 1 DC gain:
        num1 = coeffs.c0 + coeffs.c1 + coeffs.c2
        den1 = 1.0 - coeffs.c3 - coeffs.c4
        assert num1 / den1 == pytest.approx(1.0, abs=1.0e-10)

        # Stage 2 DC gain:
        num2 = coeffs.c5 + coeffs.c6 + coeffs.c7
        den2 = 1.0 - coeffs.c8 - coeffs.c9
        assert num2 / den2 == pytest.approx(1.0, abs=1.0e-10)

        # Cascaded 4th order DC gain via frequency response:
        flt = AccelFilter(fc=fc, dt=dt)
        h_dc = abs(flt.frequency_response([0.0], dt=dt)[0])
        assert h_dc == pytest.approx(1.0, abs=1.0e-10)

        # Polynomial transfer function sum:
        b, a = flt.transfer_function(dt)
        assert np.sum(b) / np.sum(a) == pytest.approx(1.0, rel=1.0e-5)


def test_dc_gain_steady_state_step():
    """Verify constant acceleration passes through unmodified in steady-state."""
    dt = 1.0e-5
    a_const = np.array([150.0, -85.0, 42.5])

    for cfc in (60, 180, 600, 1000):
        flt = AccelFilter(cfc=cfc, dt=dt)

        # Preloaded steady-state test:
        flt.reset(initial_a=a_const)
        a_out, _ = flt.update(a_const, dt=dt)
        np.testing.assert_allclose(a_out, a_const, rtol=1.0e-10, atol=1.0e-10)

        # Settling from zero test:
        flt_zero = AccelFilter(cfc=cfc, dt=dt)
        # Settle for 5000 cycles (50 ms)
        for _ in range(5000):
            a_out, _ = flt_zero.update(a_const, dt=dt)

        np.testing.assert_allclose(a_out, a_const, rtol=1.0e-5, atol=1.0e-5)


# ============================================================================
# 2. High-Frequency Attenuation (> f_c frequency attenuated by > 20 dB)
# ============================================================================

def test_high_frequency_attenuation_frequency_response():
    """Verify high-frequency attenuation > 20 dB for frequencies significantly above f_c."""
    dt = 1.0e-5  # f_nyq = 50 kHz
    flt = AccelFilter(cfc=60, dt=dt)  # fc = 100 Hz

    # Test at 5*fc (500 Hz) and 10*fc (1000 Hz)
    test_freqs = np.array([500.0, 1000.0, 2500.0])
    resp = flt.frequency_response(test_freqs)
    mag_db = 20.0 * np.log10(np.abs(resp))

    # At 5 * f_c (half decade above cutoff):
    # For a 4th-order filter, 20*log10(1 / sqrt(1 + (f/fc)^8)) = -10*log10(1 + 5^8) ~ -55 dB
    assert mag_db[0] < -20.0, f"Expected < -20 dB at 5*fc, got {mag_db[0]:.2f} dB"
    assert mag_db[1] < -40.0, f"Expected < -40 dB at 10*fc, got {mag_db[1]:.2f} dB"
    assert mag_db[2] < -60.0, f"Expected < -60 dB at 25*fc, got {mag_db[2]:.2f} dB"


def test_high_frequency_sine_wave_filtering():
    """Verify that a high-frequency sine wave is attenuated by > 20 dB in time domain."""
    dt = 1.0e-5
    t_end = 0.1  # 100 ms
    time = np.arange(0.0, t_end, dt)

    fc = 100.0  # CFC 60
    flt = AccelFilter(fc=fc, dt=dt)

    # 1000 Hz wave (10x cutoff frequency)
    f_high = 1000.0
    a_in = np.sin(2.0 * np.pi * f_high * time)

    a_filt, _ = flt.filter_series(time, a_in)

    # Ignore initial transient (first 30 ms), evaluate steady-state amplitude
    settled_idx = time > 0.03
    amp_in = 1.0
    amp_out = float(np.max(np.abs(a_filt[settled_idx])))

    attenuation_db = 20.0 * math.log10(amp_out / amp_in)
    assert attenuation_db < -20.0, f"Expected attenuation < -20 dB, got {attenuation_db:.2f} dB"


# ============================================================================
# 3. Standard CFC Classes (CFC 60, 180, 600, 1000)
# ============================================================================

@pytest.mark.parametrize(
    "cfc_input, expected_fc",
    [
        (60, 100.0),
        ("CFC60", 100.0),
        ("CFC_60", 100.0),
        (180, 300.0),
        ("CFC180", 300.0),
        ("CFC_180", 300.0),
        (600, 1000.0),
        ("CFC600", 1000.0),
        ("CFC_600", 1000.0),
        (1000, 1650.0),
        ("CFC1000", 1650.0),
        ("CFC_1000", 1650.0),
    ],
)
def test_cfc_classes_resolution_and_cutoff_frequency(cfc_input, expected_fc):
    """Verify that all standard CFC classes map to exact SAE J211 cutoff frequencies."""
    assert parse_cfc(cfc_input) == expected_fc

    flt = AccelFilter(cfc=cfc_input, dt=1.0e-6)
    assert flt.fc == expected_fc


@pytest.mark.parametrize("cfc, expected_fc", [(60, 100.0), (180, 300.0), (600, 1000.0), (1000, 1650.0)])
def test_cfc_classes_half_power_cutoff(cfc, expected_fc):
    """Verify that at f = f_c, the filter magnitude is -3.01 dB (1 / sqrt(2))."""
    dt = 1.0e-6  # small dt so prewarping distortion is negligible
    flt = AccelFilter(cfc=cfc, dt=dt)

    h_fc = flt.frequency_response([expected_fc], dt=dt)[0]
    mag_fc = abs(h_fc)
    expected_mag = 1.0 / math.sqrt(2.0)  # -3.0103 dB

    assert mag_fc == pytest.approx(expected_mag, rel=1.0e-3)
    mag_db = 20.0 * math.log10(mag_fc)
    assert mag_db == pytest.approx(-3.0103, abs=0.05)


# ============================================================================
# 4. Skew Coordinate Transformation
# ============================================================================

def test_skew_coordinate_transformation_90deg_yaw():
    """Verify rotation of acceleration into local skew frame: a_local = R @ a_global."""
    # 90-degree rotation around Z axis:
    # x_local = y_global, y_local = -x_global, z_local = z_global
    r_skew = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    flt = AccelFilter(cfc=1000, dt=1.0e-5, skew=r_skew)

    a_global = np.array([100.0, 200.0, 50.0])
    flt.reset(initial_a=a_global)

    a_local, _ = flt.update(a_global)

    expected_local = r_skew @ a_global  # [200.0, -100.0, 50.0]
    np.testing.assert_allclose(a_local, expected_local, rtol=1.0e-10)
    np.testing.assert_allclose(flt.a_filtered, expected_local, rtol=1.0e-10)
    np.testing.assert_allclose(flt.a_global_filtered, a_global, rtol=1.0e-10)


def test_skew_coordinate_transformation_arbitrary_rotation():
    """Verify arbitrary 3D orthogonal rotation matrix preserves vector norm."""
    # Euler rotation matrix (orthogonal: R @ R.T = I)
    theta = math.radians(30.0)
    phi = math.radians(45.0)
    psi = math.radians(60.0)

    # Direction cosine matrix
    r_z = np.array([
        [math.cos(psi), -math.sin(psi), 0.0],
        [math.sin(psi), math.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ])
    r_y = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ])
    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi), math.cos(phi)],
    ])
    r_skew = r_z @ r_y @ r_x

    flt = AccelFilter(cfc=600, dt=1.0e-5, skew=r_skew)
    a_glob = np.array([30.0, -70.0, 120.0])
    flt.reset(initial_a=a_glob)

    a_loc, _ = flt.update(a_glob)

    # Norm must be identically preserved
    norm_glob = np.linalg.norm(a_glob)
    norm_loc = np.linalg.norm(a_loc)
    assert norm_loc == pytest.approx(norm_glob, rel=1.0e-12)

    # Transformed vector check
    np.testing.assert_allclose(a_loc, r_skew @ a_glob, rtol=1.0e-12)


# ============================================================================
# 5. Integrated Velocity
# ============================================================================

def test_velocity_integration_constant_acceleration():
    """Verify velocity integration: v(t) = a * t under constant acceleration."""
    dt = 1.0e-4
    n_steps = 1000
    a_const = np.array([10.0, -20.0, 5.0])

    flt = AccelFilter(cfc=1000, dt=dt)
    flt.reset(initial_a=a_const)  # start at steady-state

    for step in range(1, n_steps + 1):
        a_filt, v_filt = flt.update(a_const, dt=dt)
        expected_v = a_const * (step * dt)
        np.testing.assert_allclose(v_filt, expected_v, rtol=1.0e-5, atol=1.0e-5)


# ============================================================================
# 6. Fortran SUBROUTINE ACCEL1 Parity
# ============================================================================

def test_fortran_accel1_subroutine_direct_parity():
    """Verify bit-for-bit parity between direct accel1 function and AccelFilter class."""
    dt = 2.5e-5
    fc = 300.0  # CFC 180
    r_skew = np.array([
        [0.8, -0.6, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 0.0, 1.0],
    ])

    flt = AccelFilter(fc=fc, dt=dt, skew=r_skew)

    # Parallel manual buffers for direct accel1 call
    a0 = np.zeros((3, 2), dtype=float)
    a1 = np.zeros((3, 2), dtype=float)
    a2 = np.zeros((3, 2), dtype=float)
    as_direct = np.zeros(3, dtype=float)
    vs_direct = np.zeros(3, dtype=float)

    rng = np.random.default_rng(42)
    for _ in range(50):
        a_in = rng.normal(scale=100.0, size=3)

        # 1. Update via class
        as_class, vs_class = flt.update(a_in, dt=dt)

        # 2. Update via direct Fortran subroutine
        accel1(
            a=a_in,
            ff=fc,
            a2=a2,
            a1=a1,
            a0=a0,
            as_out=as_direct,
            vs_out=vs_direct,
            skew=r_skew,
            dt=dt,
        )

        np.testing.assert_allclose(as_class, as_direct, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(vs_class, vs_direct, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(flt.a0, a0, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(flt.a1, a1, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(flt.a2, a2, rtol=1.0e-14, atol=1.0e-14)


# ============================================================================
# 7. Pass-Through When Cutoff is Zero (FF == 0)
# ============================================================================

def test_pass_through_when_fc_zero():
    """Verify that when fc = 0, raw acceleration is rotated and velocity integrated with no filter."""
    dt = 1.0e-4
    r_skew = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    flt = AccelFilter(fc=0.0, dt=dt, skew=r_skew)
    a_in = np.array([12.0, -34.0, 56.0])

    a_out, v_out = flt.update(a_in, dt=dt)

    # Immediately rotated without any delay or IIR filter dynamics
    expected_a = r_skew @ a_in
    np.testing.assert_allclose(a_out, expected_a)
    np.testing.assert_allclose(v_out, expected_a * dt)


# ============================================================================
# 8. Nyquist Frequency Capping (0.4 / dt)
# ============================================================================

def test_nyquist_frequency_capping():
    """Verify effective cutoff frequency is capped at ZEP4 / dt = 0.4 / dt."""
    dt = 1.0e-3  # 0.4 / dt = 400 Hz
    fc_requested = 1000.0  # exceeds 400 Hz

    coeffs = compute_filter_coefficients(fc_requested, dt)
    assert coeffs.fc_eff == pytest.approx(400.0)

    flt = AccelFilter(fc=fc_requested, dt=dt)
    c = flt.get_coefficients(dt)
    assert c.fc_eff == pytest.approx(400.0)


# ============================================================================
# 9. Time Series Batch Filtering Equivalence
# ============================================================================

def test_filter_series_batch_vs_step():
    """Verify filter_series batch processing yields identical results to step-by-step updates."""
    dt = 1.0e-4
    n_pts = 200
    time = np.linspace(0.0, (n_pts - 1) * dt, n_pts)

    # Complex signal with fundamental and high-frequency noise
    signal = 50.0 * np.sin(2.0 * np.pi * 30.0 * time) + 20.0 * np.cos(2.0 * np.pi * 800.0 * time)

    # 1. Batch filtering
    flt_batch = AccelFilter(cfc=60, dt=dt)
    a_batch, v_batch = flt_batch.filter_series(time, signal)

    # 2. Step-by-step filtering
    flt_step = AccelFilter(cfc=60, dt=dt)
    a_step = np.zeros(n_pts)
    v_step = np.zeros(n_pts)
    for i in range(n_pts):
        as_i, vs_i = flt_step.update(signal[i], dt=dt)
        a_step[i] = as_i[0]
        v_step[i] = vs_i[0]

    np.testing.assert_allclose(a_batch, a_step, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(v_batch, v_step, rtol=1.0e-12, atol=1.0e-12)
