"""Biomechanical Injury Criteria Sensors: HIC (Head Injury Criterion) & NIC (Neck Injury Criterion).

Upstream OpenRadioss Fortran references:
- Head Injury Criterion Engine Sensor:
  `engine/source/tools/sensor/sensor_hic.F` (SUBROUTINE SENSOR_HIC, lines 32-211)
- Neck Injury Criterion Engine Sensor:
  `engine/source/tools/sensor/sensor_nic.F` (SUBROUTINE SENSOR_NIC, lines 32-200)
- Starter HIC Reader:
  `starter/source/tools/sensor/read_sensor_hic.F` (SUBROUTINE READ_SENSOR_HIC, lines 38-204)
- Starter NIC Reader:
  `starter/source/tools/sensor/read_sensor_nic.F` (SUBROUTINE READ_SENSOR_NIC, lines 36-206)

Physics & Formulation:
----------------------
1. Head Injury Criterion (HIC, NHTSA / FMVSS 208 / ECE R94):
   Evaluates head injury risk based on acceleration time history:
       HIC = max_{t1, t2} [ (t2 - t1) * ( (1 / (t2 - t1) * integral_{t1}^{t2} a(t)/g dt) ** 2.5 ) ]
   subject to duration constraint:
       t2 - t1 <= HIC_PERIOD (typically 15 ms = 0.015 s or 36 ms = 0.036 s).
   where:
   - a(t) = sqrt(ax^2 + ay^2 + az^2) is the resultant head acceleration magnitude.
   - g is the acceleration of gravity (default 9.80665 m/s^2 or deck unit).
   - a(t)/g is dimensionless acceleration in g's.
   - (t2 - t1) is duration in seconds.

   Sliding Window Algorithm (sensor_hic.F lines 125-165):
   The time span HIC_PERIOD is divided into NPOINT intervals of width delta_t = HIC_PERIOD / NPOINT.
   At each interval transition, average acceleration is stored in a sliding table of size NPOINT.
   The optimal sub-window [t1, t2] maximizing HIC is searched across all candidate intervals [i, j].
   The sensor triggers when HIC >= HIC_CRIT (e.g. 700 for HIC15 or 1000 for HIC36).

2. Neck Injury Criteria (NIC & Nij, ECE R12 / FMVSS 208):
   - Nij Formulation:
       Nij = Fz(t) / Fint + My(t) / Mint
     where:
     - Fz is the axial neck force (tension Fz > 0 or compression Fz < 0).
     - Fint is the critical intercept force (Fint_tens or Fint_comp).
     - My is the flexion/extension bending moment (flexion My > 0 or extension My < 0).
     - Mint is the critical intercept moment (Mint_flex or Mint_ext).
   - SAE J211 Butterworth CFC filter (sensor_nic.F lines 83-166) smooths force and moment signals.
   - Boström et al. Relative Kinematics NIC:
       NIC(t) = 0.2 * a_rel(t) + (v_rel(t))^2
     where a_rel = a_T1 - a_head and v_rel = v_T1 - v_head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM20


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class HicParams:
    """Parameters for /SENSOR/HIC head injury criterion.

    Fortran origin: ``read_sensor_hic.F`` lines 83-97.
    """
    id: int = 1
    title: str = ""
    node_id: int = 0             # Tracking node ID
    accel_id: int = 0            # Accelerometer ID
    dir: int = 1                 # 1=Resultant, 2=X, 3=Y, 4=Z
    hic_period: float = 0.015    # Maximum duration window (s), e.g. 0.015 (HIC15) or 0.036 (HIC36)
    hic_crit: float = 700.0      # Critical threshold triggering the sensor
    gravity: float = 9.80665     # Acceleration of gravity (m/s^2)
    tmin: float = 0.0            # Minimum duration before triggering
    tdelay: float = 0.0          # Delay time before activation
    npoint: int = 200            # Discretization points in sliding window
    time_unit: float = 1.0       # Time unit conversion factor to seconds


@dataclass
class NicParams:
    """Parameters for /SENSOR/NIC neck injury criterion.

    Fortran origin: ``read_sensor_nic.F`` lines 84-100.
    """
    id: int = 1
    title: str = ""
    spring_id: int = 0           # Neck spring element ID
    skew_id: int = 0             # Local coordinate system ID
    nij_max: float = 1.0         # Critical Nij injury threshold
    fint_tens: float = 6806.0    # Critical tension intercept force (N)
    fint_comp: float = 6160.0    # Critical compression intercept force (N)
    mint_flex: float = 310.0     # Critical flexion intercept moment (N*m)
    mint_ext: float = 135.0      # Critical extension intercept moment (N*m)
    tmin: float = 0.0            # Minimum time before triggering
    tdelay: float = 0.0          # Delay time
    cfc: float = 600.0           # SAE-J211 filter class (Hz, e.g. CFC600 or CFC1000)
    alpha: float = 1.0           # Filter scale parameter


# ============================================================================
# Analytical & Signal Processing Functions
# ============================================================================

def compute_hic(
    time: np.ndarray,
    accel: np.ndarray,
    hic_period: float = 0.015,
    gravity: float = 9.80665,
    time_unit: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute exact maximum Head Injury Criterion (HIC) over an acceleration signal.

    Solves:
        HIC = max_{t1, t2} [ (t2 - t1) * ( (1 / (t2 - t1) * integral_{t1}^{t2} a(t)/g dt) ** 2.5 ) ]
    subject to:
        0 < t2 - t1 <= hic_period.

    Parameters
    ----------
    time : np.ndarray
        Array of time points (monotonic increasing).
    accel : np.ndarray
        Array of acceleration magnitude values (same length as time).
    hic_period : float
        Maximum window duration (seconds, default 0.015 for HIC15).
    gravity : float
        Acceleration of gravity (default 9.80665 m/s^2).
    time_unit : float
        Factor to convert time to seconds (default 1.0).

    Returns
    -------
    hic_max : float
        The maximum HIC value.
    t1_opt : float
        Start time of the optimal window.
    t2_opt : float
        End time of the optimal window.
    """
    t_arr = np.asarray(time, dtype=float) * time_unit
    a_arr = np.asarray(accel, dtype=float)

    n = len(t_arr)
    if n < 2:
        return 0.0, 0.0, 0.0

    # Dimensionless acceleration in g's
    g_val = max(gravity, 1.0e-6)
    ag = a_arr / g_val

    # Cumulative trapezoidal integration: I[k] = integral_0^{t_k} ag(t) dt
    dt = np.diff(t_arr)
    ag_mid = 0.5 * (ag[:-1] + ag[1:])
    integral = np.zeros(n, dtype=float)
    integral[1:] = np.cumsum(ag_mid * dt)

    hic_max = 0.0
    t1_opt = float(t_arr[0])
    t2_opt = float(t_arr[0])

    # Search all pairs (i, j) with 0 < t_j - t_i <= hic_period
    for i in range(n - 1):
        ti = t_arr[i]
        # Find index range where ti < tj <= ti + hic_period
        t_max_win = ti + hic_period
        # Slice search
        j_candidates = np.where((t_arr[i + 1:] <= t_max_win + 1.0e-12))[0] + (i + 1)
        if len(j_candidates) == 0:
            continue

        dt_win = t_arr[j_candidates] - ti
        valid = dt_win > 1.0e-12
        if not np.any(valid):
            continue

        dt_valid = dt_win[valid]
        j_valid = j_candidates[valid]

        # Average acceleration over window
        d_int = integral[j_valid] - integral[i]
        a_avg = np.maximum(0.0, d_int / dt_valid)

        # HIC = delta_t * (a_avg ** 2.5)
        hic_candidates = dt_valid * (a_avg ** 2.5)

        idx_max = np.argmax(hic_candidates)
        if hic_candidates[idx_max] > hic_max:
            hic_max = float(hic_candidates[idx_max])
            t1_opt = float(ti)
            t2_opt = float(t_arr[j_valid[idx_max]])

    return hic_max, t1_opt, t2_opt


def compute_nij(
    fz: float | np.ndarray,
    my: float | np.ndarray,
    fint_tens: float = 6806.0,
    fint_comp: float = 6160.0,
    mint_flex: float = 310.0,
    mint_ext: float = 135.0,
) -> float | np.ndarray:
    """Compute Neck Injury Criterion (Nij).

    Fortran origin: ``sensor_nic.F`` lines 168-169.
        Nij = Fz / Fint + My / Mint
    where Fint and Mint depend on the sign of Fz (tension vs compression)
    and My (flexion vs extension):
    - Tension-Flexion (Ntf): Fz > 0, My > 0
    - Tension-Extension (Nte): Fz > 0, My < 0
    - Compression-Flexion (Ncf): Fz < 0, My > 0
    - Compression-Extension (Nce): Fz < 0, My < 0
    """
    is_scalar = np.isscalar(fz) and np.isscalar(my)
    f_arr = np.atleast_1d(np.asarray(fz, dtype=float))
    m_arr = np.atleast_1d(np.asarray(my, dtype=float))

    f_tens_safe = max(fint_tens, EM20)
    f_comp_safe = max(fint_comp, EM20)
    m_flex_safe = max(mint_flex, EM20)
    m_ext_safe = max(mint_ext, EM20)

    f_norm = np.where(f_arr >= 0.0, f_arr / f_tens_safe, np.abs(f_arr) / f_comp_safe)
    m_norm = np.where(m_arr >= 0.0, m_arr / m_flex_safe, np.abs(m_arr) / m_ext_safe)

    nij = f_norm + m_norm
    return float(nij[0]) if is_scalar else nij


def compute_bostrom_nic(
    a_rel: float | np.ndarray,
    v_rel: float | np.ndarray,
) -> float | np.ndarray:
    """Compute Boström Neck Injury Criterion:
        NIC(t) = 0.2 * a_rel(t) + (v_rel(t))^2
    where a_rel = a_T1 - a_head and v_rel = v_T1 - v_head.
    """
    is_scalar = np.isscalar(a_rel) and np.isscalar(v_rel)
    a = np.atleast_1d(np.asarray(a_rel, dtype=float))
    v = np.atleast_1d(np.asarray(v_rel, dtype=float))
    nic = 0.2 * a + (v ** 2)
    return float(nic[0]) if is_scalar else nic


def sae_cfc_filter(
    signal: np.ndarray,
    dt: float,
    cfc: float = 600.0,
) -> np.ndarray:
    """SAE J211 digital 4-pole Butterworth low-pass filter (forward-backward zero phase).

    Matches Fortran algorithm in ``engine/source/tools/sensor/sensor_nic.F`` lines 83-166.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n < 4 or dt <= 0.0:
        return sig.copy()

    # Filter constants (sensor_nic.F lines 84-93)
    wd = 2.0 * math.pi * cfc * 1.25  # Corner frequency
    beta = 0.5 * dt * wd
    wa = math.tan(beta)
    wa2 = wa ** 2
    sqr2 = math.sqrt(2.0)
    wa3 = 1.0 + sqr2 * wa + wa2

    a0 = wa2 / wa3
    a1 = 2.0 * a0
    a2 = a0
    b1 = -2.0 * (wa2 - 1.0) / wa3
    b2 = (-1.0 + sqr2 * wa - wa2) / wa3

    # Forward pass
    y_fwd = np.zeros(n, dtype=float)
    y_fwd[0] = sig[0]
    y_fwd[1] = sig[1]
    for i in range(2, n):
        y_fwd[i] = (
            a0 * sig[i]
            + a1 * sig[i - 1]
            + a2 * sig[i - 2]
            + b1 * y_fwd[i - 1]
            + b2 * y_fwd[i - 2]
        )

    # Backward pass
    y_out = np.zeros(n, dtype=float)
    y_out[-1] = y_fwd[-1]
    y_out[-2] = y_fwd[-2]
    for i in range(n - 3, -1, -1):
        y_out[i] = (
            a0 * y_fwd[i]
            + a1 * y_fwd[i + 1]
            + a2 * y_fwd[i + 2]
            + b1 * y_out[i + 1]
            + b2 * y_out[i + 2]
        )

    return y_out


# ============================================================================
# Engine State Trackers for HIC & NIC Sensors
# ============================================================================

class HicSensor:
    """Stateful HIC Sensor matching engine/source/tools/sensor/sensor_hic.F.

    Tracks sliding window intervals, integrates acceleration in g's,
    searches maximum HIC window, and manages triggering and latching.
    """

    def __init__(self, params: HicParams):
        self.params = params
        self.id = params.id
        self.npoint = max(params.npoint, 10)

        # Interval width
        self.time_interval = self.params.hic_period / float(self.npoint)

        # State matching sensor_hic.F lines 70-75
        self.time_prec = 0.0          # VAR(1): beginning of current interval
        self.acc_integral = 0.0       # VAR(2): running acceleration integral in current interval
        self.table_index = 0          # VAR(3): number of intervals stored
        self.hic_prec = 0.0           # VAR(4): previous peak HIC value
        self.value_table = np.zeros(self.npoint, dtype=float)  # VAR(5...): interval averages

        # Sensor status
        self.current_hic = 0.0
        self.opt_period = 0.0
        self.t1 = 0.0
        self.t2 = 0.0
        self.crit_time: Optional[float] = None
        self.fire_time: Optional[float] = None
        self.status = False           # Latched when True

    def update(
        self,
        t: float,
        dt: float,
        accel_vec: np.ndarray,
    ) -> bool:
        """Update HIC sensor at time t with current acceleration vector (ax, ay, az).

        Fortran origin: ``sensor_hic.F`` lines 95-194.

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Simulation time step increment.
        accel_vec : np.ndarray (3,)
            Nodal acceleration vector [ax, ay, az].

        Returns
        -------
        bool : True if sensor is active / fired.
        """
        if self.status:
            return True

        # Acceleration magnitude according to direction parameter (lines 105-115)
        if self.params.dir == 1:
            acc = float(np.linalg.norm(accel_vec))
        elif self.params.dir == 2:
            acc = float(accel_vec[0])
        elif self.params.dir == 3:
            acc = float(accel_vec[1])
        else:
            acc = float(accel_vec[2])

        # Acceleration in g's
        acc_g = acc / max(self.params.gravity, 1.0e-6)

        # Integration within current time interval
        test_time = self.time_prec + self.time_interval

        if t > test_time:
            delta_t = max(t - self.time_prec, 1.0e-12)
            # Average acceleration for finished interval
            current_value = (self.acc_integral + acc_g * dt) / delta_t

            if self.table_index == self.npoint:
                # Shift buffer left by 1 to make place for new interval
                self.value_table[:-1] = self.value_table[1:]
                self.value_table[-1] = current_value
            else:
                self.value_table[self.table_index] = current_value
                self.table_index += 1

            # Search interval sub-window maximizing HIC (sensor_hic.F lines 146-165)
            # HIC = period * ( (integral / period) ** 2.5 )
            hic_max = 0.0
            opt_period = 0.0
            t1_found = t
            t2_found = t

            cur_k = self.table_index
            for i in range(cur_k):
                # Integral from i to cur_k
                sub_table = self.value_table[i:cur_k]
                inc_count = cur_k - i
                period_tmp = delta_t * inc_count
                if period_tmp <= self.params.hic_period + 1.0e-12:
                    sub_int = np.sum(sub_table) * delta_t
                    mean_a = sub_int / period_tmp
                    hic_tmp = period_tmp * (mean_a ** 2.5) * self.params.time_unit
                    if hic_tmp > hic_max:
                        hic_max = hic_tmp
                        opt_period = period_tmp
                        t1_found = max(t - opt_period, 0.0)
                        t2_found = t

            self.current_hic = hic_max
            self.opt_period = opt_period
            self.t1 = t1_found
            self.t2 = t2_found

            self.time_prec = t
            self.acc_integral = 0.0

            # Check critical threshold
            if hic_max >= self.params.hic_crit:
                if self.crit_time is None:
                    self.crit_time = t
                if (t >= self.crit_time + self.params.tmin + self.params.tdelay):
                    self.status = True
                    self.fire_time = t
            else:
                if self.crit_time is not None and not self.status:
                    # Reset if criterion drops below threshold before tmin
                    self.crit_time = None
        else:
            # Accumulate running integral in current interval
            self.acc_integral += acc_g * dt

        return self.status


class NicSensor:
    """Stateful Neck Injury Criterion Sensor matching engine/source/tools/sensor/sensor_nic.F."""

    def __init__(self, params: NicParams):
        self.params = params
        self.id = params.id
        self.crit_time: Optional[float] = None
        self.fire_time: Optional[float] = None
        self.status = False
        self.current_nij = 0.0

    def update(
        self,
        t: float,
        dt: float,
        fz: float,
        my: float,
    ) -> bool:
        """Update NIC sensor with axial neck force Fz and bending moment My."""
        if self.status:
            return True

        self.current_nij = float(compute_nij(
            fz=fz,
            my=my,
            fint_tens=self.params.fint_tens,
            fint_comp=self.params.fint_comp,
            mint_flex=self.params.mint_flex,
            mint_ext=self.params.mint_ext,
        ))

        if self.current_nij >= self.params.nij_max:
            if self.crit_time is None:
                self.crit_time = t
            if (t >= self.crit_time + self.params.tmin + self.params.tdelay):
                self.status = True
                self.fire_time = t
        else:
            if self.crit_time is not None and not self.status:
                self.crit_time = None

        return self.status
