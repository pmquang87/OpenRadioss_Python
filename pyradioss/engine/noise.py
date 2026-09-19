"""
/NOISE — Digital FIR lowpass filter module for velocities and accelerations (M613).

Fortran origin:
  * Filter initialization and execution:
    ``engine/source/general_controls/computation/noise.F``
  * Node and element connectivity indexing:
    ``engine/source/general_controls/computation/initnoise.F``
  * Pressure averaging on noise nodes:
    ``engine/source/general_controls/computation/pnoise.F``

Theory:
OpenRadioss filters high-frequency structural noise in nodal velocities,
accelerations, and pressures in situ using a windowed-sinc FIR digital filter
with a Hamming window.

Given the simulation time step T_E = dt and user-requested output interval dt_noise:
  * N_E = max(1, int(dt_noise / T_E)) : decimation ratio (samples between outputs)
  * dt_noise_eff = N_E * T_E
  * T_C = 2 * dt_noise_eff : cutoff period (Nyquist period of output rate)
  * f_cutoff = 1 / T_C = 1 / (2 * N_E * T_E)
  * N_C = 6 * N_E : filter tap count
  * FAC = 1 / N_E
  * DEC = 3 * N_E - 0.5
  * T_0 = 3 * N_E * T_E : group delay (symmetric FIR filter delay)

Filter coefficients (for i = 1 .. N_C):
  * xi = (i - 1) - DEC
  * W = 0.54 + 0.46 * cos(2 * pi * xi / N_C)
  * If |pi * FAC * xi| < 0.1:
      C_i = FAC
    Else:
      C_i = W * sin(pi * xi * FAC) / (pi * xi)

Accumulator Bank:
To continuously filter the signal with delay T_0 and output decimated samples
every N_E cycles, 6 overlapping accumulators are maintained, staggered by
N_E cycles:
  J(K) = (1 - K) * N_E - 1   for K = 1 .. 6.
At each cycle, J(K) is incremented. When J(K) > 0, sample values are accumulated
weighted by C(J(K)). When J(K) reaches N_C, accumulator K is complete: it outputs
the filtered signal at time (t - T_0), resets J(K) to 0, and clears its buffer.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


def compute_filter_coefficients(ne: int) -> np.ndarray:
    """Compute FIR windowed sinc filter coefficients with Hamming window.

    Matches ``engine/source/general_controls/computation/noise.F`` (lines 113-134).

    Parameters
    ----------
    ne : int
        Number of samples between outputs (decimation factor). Must be >= 1.

    Returns
    -------
    np.ndarray
        Filter coefficients of length NC = 6 * NE.
    """
    ne = max(1, int(ne))
    nc = 6 * ne
    fac = 1.0 / float(ne)
    dec = float(3 * ne) - 0.5
    tol = 0.1
    pi2 = 2.0 * np.pi

    c = np.zeros(nc, dtype=np.float64)
    for i in range(1, nc + 1):
        xi = float(i - 1) - dec
        w = 0.54 + 0.46 * np.cos(pi2 * xi / float(nc))
        arg = np.abs(np.pi * fac * xi)
        if arg < tol:
            c[i - 1] = fac
        else:
            c[i - 1] = w * np.sin(np.pi * xi * fac) / (np.pi * xi)

    return c


class FilterOutput(tuple):
    """Filtered signal output container.

    Can be unpacked as a 2-tuple (v, a) or accessed via named attributes:
    out.v, out.a, out.time, out.p.
    """

    def __new__(
        cls,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        time: float = 0.0,
        p: Optional[np.ndarray] = None,
    ):
        return super().__new__(cls, (v, a))

    def __init__(
        self,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        time: float = 0.0,
        p: Optional[np.ndarray] = None,
    ):
        self.v = v
        self.a = a
        self.time = float(time)
        self.p = p

    def __repr__(self) -> str:
        return (
            f"FilterOutput(time={self.time:.6e}, "
            f"has_v={self.v is not None}, "
            f"has_a={self.a is not None})"
        )


class NoiseFilter:
    """In-situ digital FIR lowpass filter with Hamming window.

    Matches ``engine/source/general_controls/computation/noise.F`` and ``initnoise.F``.
    Maintains 6 overlapping accumulators spaced by N_E cycles to generate
    decimated lowpass filtered outputs at interval dt_noise.
    """

    def __init__(
        self,
        dt_noise: float,
        dt: float,
        num_nodes: Optional[int] = None,
        filter_velocity: bool = True,
        filter_acceleration: bool = True,
        filter_pressure: bool = False,
    ):
        """Initialize NoiseFilter.

        Parameters
        ----------
        dt_noise : float
            Requested output interval for filtered noise signal.
        dt : float
            Simulation time step T_E.
        num_nodes : int, optional
            Number of nodes to filter (optional, can be inferred on first call).
        filter_velocity : bool
            Whether to filter velocity signals.
        filter_acceleration : bool
            Whether to filter acceleration signals.
        filter_pressure : bool
            Whether to filter pressure signals.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if dt_noise <= 0.0:
            raise ValueError(f"dt_noise must be positive, got {dt_noise}")

        self.dt = float(dt)
        self.te = float(dt)
        self.requested_dt_noise = float(dt_noise)

        # NE : number of samples between outputs matching noise.F:93
        self.ne = max(1, int(self.requested_dt_noise / self.te))
        self.dt_noise = float(self.ne) * self.te
        self.tc = 2.0 * self.dt_noise
        self.nc = 6 * self.ne
        self.to = float(3 * self.ne) * self.te  # group delay

        self.c = compute_filter_coefficients(self.ne)
        self.transmittance = float(np.sum(self.c))

        self.filter_v = filter_velocity
        self.filter_a = filter_acceleration
        self.filter_p = filter_pressure

        self.num_nodes = num_nodes
        self.cycle = 0
        self.current_time = 0.0

        # Accumulator index array j[k] matching Fortran lines 143-145:
        # DO K=1,6: J(K) = (1-K)*NE - 1
        self.j = np.array([(1 - k) * self.ne - 1 for k in range(1, 7)], dtype=np.int32)

        # Buffers for the 6 accumulators (allocated dynamically on first call or pre-allocated)
        self.buf_v: Optional[np.ndarray] = None
        self.buf_a: Optional[np.ndarray] = None
        self.buf_p: Optional[np.ndarray] = None

        if num_nodes is not None and num_nodes > 0:
            if self.filter_v:
                self.buf_v = np.zeros((6, num_nodes, 3), dtype=np.float64)
            if self.filter_a:
                self.buf_a = np.zeros((6, num_nodes, 3), dtype=np.float64)

    @property
    def cutoff_frequency(self) -> float:
        """High frequency cutoff in Hz: 1 / T_C."""
        return 1.0 / self.tc

    @property
    def effective_sampling_frequency(self) -> float:
        """Effective sampling frequency in Hz: 1 / dt_noise."""
        return 1.0 / self.dt_noise

    @property
    def group_delay(self) -> float:
        """Filter group delay in seconds: T_0 = 3 * N_E * T_E."""
        return self.to

    def reset(self) -> None:
        """Reset filter accumulators and time."""
        self.cycle = 0
        self.current_time = 0.0
        self.j = np.array([(1 - k) * self.ne - 1 for k in range(1, 7)], dtype=np.int32)
        if self.buf_v is not None:
            self.buf_v.fill(0.0)
        if self.buf_a is not None:
            self.buf_a.fill(0.0)
        if self.buf_p is not None:
            self.buf_p.fill(0.0)

    def update(
        self,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        time: Optional[float] = None,
        p: Optional[np.ndarray] = None,
    ) -> Optional[FilterOutput]:
        """Process one time step and accumulate signal.

        Parameters
        ----------
        v : np.ndarray, optional
            Nodal velocities. Can be shape (N, 3), (N,), or scalar.
        a : np.ndarray, optional
            Nodal accelerations. Can be shape (N, 3), (N,), or scalar.
        dt : float, optional
            Current time step dt. If None, uses initial self.dt.
        time : float, optional
            Current simulation time. If None, tracked internally.
        p : np.ndarray, optional
            Element pressures.

        Returns
        -------
        FilterOutput or None
            If an accumulator completed at this step, returns FilterOutput with
            filtered signals at time t - T_0. Otherwise returns None.
        """
        step_dt = self.dt if dt is None else float(dt)
        if time is not None:
            self.current_time = float(time)
        else:
            self.current_time += step_dt
        self.cycle += 1

        # Lazy allocation of accumulator buffers to match input array shapes
        if v is not None and self.filter_v:
            v_arr = np.asarray(v, dtype=np.float64)
            if self.buf_v is None or self.buf_v.shape[1:] != v_arr.shape:
                self.buf_v = np.zeros((6, *v_arr.shape), dtype=np.float64)
        else:
            v_arr = None

        if a is not None and self.filter_a:
            a_arr = np.asarray(a, dtype=np.float64)
            if self.buf_a is None or self.buf_a.shape[1:] != a_arr.shape:
                self.buf_a = np.zeros((6, *a_arr.shape), dtype=np.float64)
        else:
            a_arr = None

        if p is not None and self.filter_p:
            p_arr = np.asarray(p, dtype=np.float64)
            if self.buf_p is None or self.buf_p.shape[1:] != p_arr.shape:
                self.buf_p = np.zeros((6, *p_arr.shape), dtype=np.float64)
        else:
            p_arr = None

        # Filter accumulation matching Fortran lines 178-206
        for k in range(6):
            self.j[k] += 1
            if self.j[k] > 0:
                cc = self.c[self.j[k] - 1]
                if v_arr is not None and self.buf_v is not None:
                    self.buf_v[k] += cc * v_arr
                if a_arr is not None and self.buf_a is not None:
                    self.buf_a[k] += cc * a_arr
                if p_arr is not None and self.buf_p is not None:
                    self.buf_p[k] += cc * p_arr

        # Check completed accumulators matching Fortran lines 217-233
        result: Optional[FilterOutput] = None
        for k in range(6):
            if self.j[k] == self.nc:
                self.j[k] = 0
                t_out = self.current_time - self.to

                out_v = self.buf_v[k].copy() if self.buf_v is not None else None
                out_a = self.buf_a[k].copy() if self.buf_a is not None else None
                out_p = self.buf_p[k].copy() if self.buf_p is not None else None

                if self.buf_v is not None:
                    self.buf_v[k].fill(0.0)
                if self.buf_a is not None:
                    self.buf_a[k].fill(0.0)
                if self.buf_p is not None:
                    self.buf_p[k].fill(0.0)

                result = FilterOutput(v=out_v, a=out_a, time=t_out, p=out_p)

        return result
