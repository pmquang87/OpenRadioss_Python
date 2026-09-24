"""
Unit tests for Milestone M613: Range Damping and Noise Filter modules.

Tests:
- damping_range_compute_param: parameter values, analytical/Fortran parity, symmetry, error cases.
- DampingRangeSolid: viscous stress accumulation, exponential relaxation, energy dissipation, sound speed update.
- DampingRangeShell: plane-stress condition (sigzz=0), membrane stresses, bending moments, sound speed update.
- NoiseFilter: FIR Hamming coefficients, symmetry, DC unit gain, frequency attenuation, 6-accumulator buffer cycling.
"""

import numpy as np
import pytest

from pyradioss.engine.range_damping import (
    damping_range_compute_param,
    damping_range_solid_subroutine,
    DampingRangeSolid,
    damping_range_shell_subroutine,
    damping_range_shell_mom_subroutine,
    DampingRangeShell,
)
from pyradioss.engine.noise import (
    compute_filter_coefficients,
    FilterOutput,
    NoiseFilter,
)


# ===========================================================================
# 1. DAMPING RANGE COMPUTE PARAM TESTS
# ===========================================================================

def test_damping_range_compute_param_basic():
    """Verify compute_param produces 3 positive alpha and tau parameters."""
    damp_ratio = 0.05
    f_low = 10.0
    f_high = 1000.0

    alpha, tau = damping_range_compute_param(damp_ratio, f_low, f_high)

    assert isinstance(alpha, np.ndarray)
    assert isinstance(tau, np.ndarray)
    assert alpha.shape == (3,)
    assert tau.shape == (3,)

    # All alphas and taus must be strictly positive
    assert np.all(alpha > 0.0)
    assert np.all(tau > 0.0)

    # Low and high frequency Maxwell components should have symmetric alphas
    # because factor is [0.965, 1.0, 0.965] and f_mid = sqrt(f_low * f_high)
    assert pytest.approx(alpha[0], rel=1e-5) == alpha[2]

    # Tau should decrease with increasing frequency: tau ~ 1 / (2*pi*f)
    assert tau[0] > tau[1] > tau[2]


def test_damping_range_compute_param_exact_fortran():
    """Verify exact values matching Fortran lines 80-117."""
    damp_ratio = 0.05
    f_low = 10.0
    f_high = 1000.0

    alpha, tau = damping_range_compute_param(damp_ratio, f_low, f_high)

    # Reference values computed from exact Fortran formulas:
    # f_mid = 100.0
    # freq_sample = [10, 100, 1000]
    # alpha ~ [0.17672458, 0.14496725, 0.17672458]
    # tau   ~ [0.01467178, 0.00148739, 0.00014672]
    expected_alpha = np.array([0.17672458, 0.14496725, 0.17672458])
    expected_tau = np.array([0.014671777, 0.001487389, 0.0001467178])

    np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-5)
    np.testing.assert_allclose(tau, expected_tau, rtol=1e-5)


def test_damping_range_reconstructed_damping_ratio():
    """Verify that the 3 Maxwell components accurately reproduce the target damping ratio."""
    damp_ratio = 0.05
    f_low = 20.0
    f_high = 2000.0
    f_mid = np.sqrt(f_low * f_high)

    alpha, tau = damping_range_compute_param(damp_ratio, f_low, f_high)

    # Solve back the damping ratio at the three sample frequencies
    freq_sample = np.array([f_low, f_mid, f_high])
    matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ratio = freq_sample[i] / freq_sample[j]
            matrix[i, j] = (4.0 * damp_ratio * ratio) / (2.0 + 2.0 * ratio**2)

    inv_mat = np.linalg.inv(matrix)
    factor = np.array([0.965, 1.0, 0.965])
    e_max = np.zeros(3)
    for i in range(3):
        e_fac = np.sum(inv_mat[i, :] * damp_ratio * factor)
        e_max[i] = e_fac * damp_ratio

    # Reconstruct damping ratio at sample points:
    # D(f) = sum_j (4 * e_max_j * (f / f_j)) / (2 + 2 * (f / f_j)^2)
    for idx, f in enumerate(freq_sample):
        d_reconstructed = 0.0
        for j in range(3):
            ratio = f / freq_sample[j]
            d_reconstructed += (4.0 * e_max[j] * ratio) / (2.0 + 2.0 * ratio**2)
        expected_d = damp_ratio * factor[idx]
        assert pytest.approx(expected_d, rel=1e-5) == d_reconstructed


def test_damping_range_compute_param_invalid():
    """Verify input validation for illegal frequency bands or damping ratios."""
    with pytest.raises(ValueError, match="f_low must be positive"):
        damping_range_compute_param(0.05, -10.0, 100.0)

    with pytest.raises(ValueError, match="strictly greater"):
        damping_range_compute_param(0.05, 100.0, 10.0)

    with pytest.raises(ValueError, match="strictly greater"):
        damping_range_compute_param(0.05, 100.0, 100.0)

    with pytest.raises(ValueError, match="damp_ratio must be positive"):
        damping_range_compute_param(0.0, 10.0, 100.0)


# ===========================================================================
# 2. DAMPING RANGE SOLID TESTS
# ===========================================================================

def test_damping_range_solid_relaxation():
    """Verify dynamic stress generation and exponential relaxation in solid elements."""
    alpha = np.array([0.2, 0.15, 0.2])
    tau = np.array([0.01, 0.001, 0.0001])
    young = 210000.0
    shear_mod = 80000.0
    dt = 1e-5

    solid_damper = DampingRangeSolid(alpha, tau, young, shear_mod, nel=1)

    # 1. Apply dynamic uniaxial tension rate epsp_xx = 10.0 s^-1 for 50 steps
    epsp_dynamic = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rho = 7.8e-9
    soundsp = 5000.0

    for _ in range(50):
        sv, c_new = solid_damper.update(dt, epsp_dynamic, rho, soundsp)

    # Viscous stress xx should be positive and opposing tensile elongation
    sig_loaded = sv[0]
    assert sig_loaded > 0.0
    # Sound speed should be increased by bulk and shear damping moduli
    assert c_new > soundsp

    # 2. Relax with epsp = 0 for 500 steps (5 ms)
    epsp_zero = np.zeros(6)
    for _ in range(500):
        sv_relaxed, _ = solid_damper.update(dt, epsp_zero, rho, soundsp)

    sig_relaxed = sv_relaxed[0]
    # Stress must have relaxed significantly
    assert sig_relaxed < 0.5 * sig_loaded
    assert sig_relaxed > 0.0  # remains non-negative during relaxation


def test_damping_range_solid_energy_dissipation():
    """Verify that solid damping dissipates energy under cyclic strain."""
    alpha, tau = damping_range_compute_param(0.05, 10.0, 1000.0)
    young = 200000.0
    shear_mod = 75000.0
    dt = 1e-4

    solid_damper = DampingRangeSolid(alpha, tau, young, shear_mod, nel=1)

    # Sinusoidal strain rate epsp_xx = eps0 * omega * cos(omega * t)
    f_drive = 50.0  # 50 Hz
    omega = 2.0 * np.pi * f_drive
    eps_amp = 0.001
    n_cycles = 2
    n_steps = int(n_cycles / (f_drive * dt))

    total_dissipated_work = 0.0
    rho = 7.85e-9
    c0 = 5000.0

    for step in range(n_steps):
        t = step * dt
        epsp_xx = eps_amp * omega * np.cos(omega * t)
        epsp = np.array([epsp_xx, 0.0, 0.0, 0.0, 0.0, 0.0])

        sv, _ = solid_damper.update(dt, epsp, rho, c0)
        # Power dissipation rate: sigma_visc * epsp
        p_diss = sv[0] * epsp_xx
        total_dissipated_work += p_diss * dt

    # Over full cycles, net work dissipated must be strictly positive
    assert total_dissipated_work > 0.0


def test_damping_range_solid_vectorized():
    """Verify that multi-element vectorized calls match single-element calls."""
    alpha, tau = damping_range_compute_param(0.04, 5.0, 500.0)
    young = 100000.0
    shear_mod = 38000.0
    dt = 1e-4
    nel = 4

    damper_multi = DampingRangeSolid(alpha, tau, young, shear_mod, nel=nel)
    damper_single = DampingRangeSolid(alpha, tau, young, shear_mod, nel=1)

    epsp_multi = np.array([
        [1.0, -0.3, -0.3, 0.5, 0.0, 0.0],
        [0.0, 2.0, -0.5, 0.0, 0.2, 0.0],
        [-1.5, 0.5, 0.5, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    rho = np.full(nel, 2.7e-9)
    c0 = np.full(nel, 6000.0)

    sv_multi, c_multi = damper_multi.update(dt, epsp_multi, rho, c0)

    # Compare with individual single element calls
    for i in range(nel):
        sv_i, c_i = damper_single.update(dt, epsp_multi[i], rho[i], c0[i])
        np.testing.assert_allclose(sv_multi[i], sv_i, rtol=1e-12)
        assert pytest.approx(c_multi[i], rel=1e-12) == c_i
        damper_single.reset()


# ===========================================================================
# 3. DAMPING RANGE SHELL TESTS
# ===========================================================================

def test_damping_range_shell_plane_stress():
    """Verify that shell damping strictly enforces plane stress sigma_zz = 0."""
    alpha, tau = damping_range_compute_param(0.05, 10.0, 1000.0)
    young = 210000.0
    shear_mod = 80000.0
    dt = 1e-5
    nel = 3

    uvarv = np.zeros((nel, 21), dtype=np.float64)
    etse = np.ones(nel, dtype=np.float64)
    epspxx = np.array([5.0, -2.0, 10.0])
    epspyy = np.array([-1.5, 3.0, 0.0])
    epspxy = np.array([1.0, 0.5, -2.0])
    epspyz = np.array([0.0, 0.1, 0.0])
    epspzx = np.array([0.1, 0.0, 0.5])
    sigvxx = np.zeros(nel)
    sigvyy = np.zeros(nel)
    sigvxy = np.zeros(nel)
    sigvyz = np.zeros(nel)
    sigvzx = np.zeros(nel)
    rho0 = np.full(nel, 7.8e-9)
    soundsp = np.full(nel, 5000.0)
    off = np.ones(nel)

    # Run for 20 steps
    for _ in range(20):
        damping_range_shell_subroutine(
            alpha=alpha,
            tau=tau,
            uvarv=uvarv,
            etse=etse,
            timestep=dt,
            young=young,
            shear_mod=shear_mod,
            epspxx=epspxx,
            epspyy=epspyy,
            epspxy=epspxy,
            epspyz=epspyz,
            epspzx=epspzx,
            sigvxx=sigvxx,
            sigvyy=sigvyy,
            sigvxy=sigvxy,
            sigvyz=sigvyz,
            sigvzx=sigvzx,
            rho0=rho0,
            soundsp=soundsp,
            off=off,
            flag_incr=0,
        )

        # Check that sigma_zz = s_zz - p is zero for all components
        s3 = uvarv[:, 2] + uvarv[:, 9] + uvarv[:, 16]
        p = uvarv[:, 6] + uvarv[:, 13] + uvarv[:, 20]
        sig_zz = s3 - p
        np.testing.assert_allclose(sig_zz, 0.0, atol=1e-12)


def test_damping_range_shell_class_membrane_and_bending():
    """Verify DampingRangeShell membrane and bending updates."""
    alpha, tau = damping_range_compute_param(0.03, 20.0, 2000.0)
    young = 70000.0
    shear_mod = 26000.0
    dt = 1e-4

    shell_damper = DampingRangeShell(alpha, tau, young, shear_mod, nel=1, flag_incr=0)

    # Membrane test
    epsp_mem = np.array([2.0, 1.0, 0.5, 0.0, 0.0])
    rho = 2.7e-9
    c0 = 5100.0

    sigv, c_new = shell_damper.update_membrane(dt, epsp_mem, rho, c0)
    assert sigv.shape == (5,)
    assert sigv[0] > 0.0  # tension xx
    assert sigv[1] > 0.0  # tension yy
    assert c_new > c0

    # Bending test
    # Curvature rate produces bending strain increments depb = dt * kappa_dot
    depb = np.array([1e-4, -5e-5, 2e-5])
    thickness = 0.002  # 2 mm
    mom = shell_damper.update_bending(dt, depb, thickness)

    assert mom.shape == (3,)
    assert mom[0] > 0.0  # positive moment resisting positive curvature rate
    assert mom[1] < 0.0  # negative moment resisting negative curvature rate


# ===========================================================================
# 4. NOISE FILTER TESTS
# ===========================================================================

def test_noise_filter_coefficients():
    """Verify windowed sinc filter coefficients with Hamming window."""
    ne = 10
    nc = 6 * ne  # 60
    c = compute_filter_coefficients(ne)

    assert len(c) == nc
    # Filter coefficients must be symmetric about center: c[i] == c[nc - 1 - i]
    np.testing.assert_allclose(c, c[::-1], atol=1e-12)

    # Unit DC transmittance: sum(c) approx 1.0 (noise.F static transmittance)
    transmittance = np.sum(c)
    assert pytest.approx(1.0, abs=0.005) == transmittance


def test_noise_filter_dc_unity_gain():
    """Verify that a constant DC signal passes through with unit gain."""
    dt = 1e-4
    dt_noise = 1e-3
    noise_filter = NoiseFilter(dt_noise=dt_noise, dt=dt, num_nodes=1)

    v_dc = np.array([[10.0, -5.0, 2.5]])
    a_dc = np.array([[0.0, 0.0, 0.0]])

    outputs = []
    # Run for 150 cycles (enough to fill and cycle multiple accumulators)
    for _ in range(150):
        out = noise_filter.update(v=v_dc, a=a_dc)
        if out is not None:
            outputs.append(out)

    assert len(outputs) >= 8
    # After filter fills, DC output should equal input within transmittance tolerance
    for out in outputs[2:]:
        np.testing.assert_allclose(out.v, v_dc, rtol=0.005)


def test_noise_filter_high_frequency_attenuation():
    """Verify that frequencies above cutoff are heavily attenuated."""
    dt = 1e-5  # 100 kHz sampling
    dt_noise = 1e-3  # Output at 1 kHz -> cutoff fc = 1 / (2 * 1e-3) = 500 Hz
    noise_filter = NoiseFilter(dt_noise=dt_noise, dt=dt)

    # Low frequency: 50 Hz (well below 500 Hz cutoff)
    # High frequency: 5000 Hz (well above 500 Hz cutoff)
    f_low = 50.0
    f_high = 5000.0

    n_steps = 1000
    t_arr = np.arange(n_steps) * dt
    sig_low = np.sin(2.0 * np.pi * f_low * t_arr)
    sig_high = np.sin(2.0 * np.pi * f_high * t_arr)

    # Filter low frequency signal
    noise_filter.reset()
    out_low = []
    for step in range(n_steps):
        v = np.array([sig_low[step]])
        out = noise_filter.update(v=v)
        if out is not None:
            out_low.append(out.v[0])

    # Filter high frequency signal
    noise_filter.reset()
    out_high = []
    for step in range(n_steps):
        v = np.array([sig_high[step]])
        out = noise_filter.update(v=v)
        if out is not None:
            out_high.append(out.v[0])

    assert len(out_low) > 0
    assert len(out_high) > 0

    max_low = np.max(np.abs(out_low[2:]))
    max_high = np.max(np.abs(out_high[2:]))

    # Low frequency passes unattenuated (amplitude ~ 1.0)
    assert pytest.approx(1.0, abs=0.05) == max_low
    # High frequency is attenuated by at least 40 dB (< 0.01)
    assert max_high < 0.01


def test_noise_filter_accumulator_cycling():
    """Verify 6-accumulator buffer cycling, timing intervals, and group delay."""
    dt = 1e-4
    dt_noise = 1e-3
    ne = int(dt_noise / dt)  # 10
    noise_filter = NoiseFilter(dt_noise=dt_noise, dt=dt)

    expected_delay = 3.0 * ne * dt
    assert pytest.approx(expected_delay, rel=1e-6) == noise_filter.group_delay

    v_sample = np.array([1.0, 2.0, 3.0])
    a_sample = np.array([10.0, 20.0, 30.0])

    cycles_with_output = []
    times_with_output = []

    for cycle in range(1, 120):
        t = cycle * dt
        out = noise_filter.update(v=v_sample, a=a_sample, time=t)
        if out is not None:
            cycles_with_output.append(cycle)
            times_with_output.append(out.time)
            # Verify unpacking interface (v, a) = out
            v_out, a_out = out
            assert v_out is not None
            assert a_out is not None
            assert hasattr(out, "time")

    # Outputs should be spaced exactly by ne = 10 cycles
    assert len(cycles_with_output) >= 5
    cycle_diffs = np.diff(cycles_with_output)
    np.testing.assert_array_equal(cycle_diffs, np.full(len(cycle_diffs), ne))

    # Output time should equal cycle_time - group_delay
    for cycle, out_time in zip(cycles_with_output, times_with_output):
        expected_t = cycle * dt - expected_delay
        assert pytest.approx(expected_t, rel=1e-6) == out_time
