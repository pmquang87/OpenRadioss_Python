# Ported from OpenRadioss Fortran:
# Source: starter/source/loads/general/load_centri/hm_read_load_centri.F
# Function: HM_READ_LOAD_CENTRI (lines 42-318)
# Source: engine/source/loads/general/load_centri/cfield.F
# Subroutine: CFIELD_1 (lines 80-258)
# Source: engine/source/loads/general/load_centri/cfield_imp.F
# Subroutine: CFIELD_IMP (lines 36-183)
"""
/LOAD/CENTRI — Centrifugal Body Force and Rotational Acceleration Field.

Upstream OpenRadioss Fortran Reference:
----------------------------------------
- Starter card reader:
  ``starter/source/loads/general/load_centri/hm_read_load_centri.F`` (HM_READ_LOAD_CENTRI)
- Engine time-step evaluation:
  ``engine/source/loads/general/load_centri/cfield.F`` (CFIELD_1)
  ``engine/source/loads/general/load_centri/cfield_imp.F`` (CFIELD_IMP)

Physics Formulation (cfield.F:148-200):
----------------------------------------
Given a rotating reference frame or body with:
- Origin point X0 = (x0, y0, z0)
- Unit rotation axis vector u_axis = (u_x, u_y, u_z) (e.g. IDIR=4 for X, 5 for Y, 6 for Z, or arbitrary 3D axis)
- Instantaneous angular velocity omega(t) and angular acceleration alpha(t) = domega/dt:

For each node i with position x_i, velocity v_i, and mass m_i:
1. Relative position vector from origin:
     r_i = x_i - X0

2. Perpendicular distance vector from rotation axis (cfield.F lines 154-156):
     r_perp = r_i - (r_i . u_axis) * u_axis

3. Angular acceleration vector (cfield.F lines 157-159):
     dw = alpha * u_axis = (domega/dt) * u_axis

4. Relative centrifugal and Euler tangential accelerations (cfield.F lines 160-162):
     a_centri = (omega^2) * r_perp
     a_euler  = dw x r_perp
     a_rel    = a_centri + a_euler

5. Nodal body force:
     F_node = m_i * a_rel

6. External work accumulation (cfield.F lines 168-172):
     v_mid = v_i + 0.5 * dt * a_rel
     dW = sum_i m_i * (a_rel . v_mid) * dt
     WFEXT += dW
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class CentrifugalResult:
    """Evaluation result for centrifugal load."""

    forces: np.ndarray  # Shape (N, 3): nodal force vectors F_i = m_i * a_rel
    accelerations: np.ndarray  # Shape (N, 3): nodal relative accelerations a_rel
    distances: np.ndarray  # Shape (N, 3): perpendicular distance vectors r_perp
    omega: float  # Instantaneous rotational speed (rad/s)
    domega_dt: float  # Instantaneous angular acceleration (rad/s^2)
    work_increment: float  # External work increment dW = sum(F . v_mid) * dt
    active: bool  # True if sensor is triggered / load is active


@dataclass
class LoadCentri:
    """Parameters for /LOAD/CENTRI centrifugal load.

    Ported from:
      - ``starter/source/loads/general/load_centri/hm_read_load_centri.F``
      - ``engine/source/loads/general/load_centri/cfield.F``
    """

    id: int = 1
    title: str = ""
    grnod_id: int = 0  # Target node group (0 = all nodes)
    node_ids: Optional[List[int]] = None  # Optional explicit node list

    # Axis direction: 1/4 = X, 2/5 = Y, 3/6 = Z, or custom 3D vector
    dir_id: Optional[int] = None
    axis: Optional[Tuple[float, float, float]] = None
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # X0 = (x0, y0, z0)

    # Rotational speed
    omega: float = 0.0  # Constant speed (rad/s) if no curve
    funct_id: int = 0
    function: Optional[Callable[[float], float]] = None  # omega(t) curve
    function_derivative: Optional[Callable[[float], float]] = None  # domega/dt curve

    # Scale factors: scale_x applies to time, scale_y applies to angular velocity
    scale_x: float = 1.0  # Time scale factor (FCX / FAC(2, NL))
    scale_y: float = 1.0  # Velocity magnitude scale factor (FCY / FAC(1, NL))
    domega_dt: Optional[float] = None  # Constant angular acceleration (rad/s^2)

    # Frame & Sensor options
    frame_id: int = 0
    sensor_id: int = 0
    tstart: Optional[float] = None  # Sensor trigger time (delays load until t >= tstart)

    # Calculation flag: 2 = include angular acceleration domega/dt, 1 or 0 = pure centrifugal
    flag: int = 2

    def __post_init__(self) -> None:
        # Resolve axis from dir_id if axis is not explicitly provided
        if self.axis is None:
            if self.dir_id in (1, 4):
                self.axis = (1.0, 0.0, 0.0)
            elif self.dir_id in (2, 5):
                self.axis = (0.0, 1.0, 0.0)
            elif self.dir_id in (3, 6):
                self.axis = (0.0, 0.0, 1.0)
            else:
                self.axis = (0.0, 0.0, 1.0)
                self.dir_id = 6
        elif self.dir_id is None:
            self.dir_id = 6

        # Normalize rotation axis
        ax = np.array(self.axis, dtype=float)
        norm_ax = np.linalg.norm(ax)
        if norm_ax > 1e-14:
            self.axis = tuple(ax / norm_ax)  # type: ignore[assignment]
        else:
            self.axis = (0.0, 0.0, 1.0)

        if self.scale_x == 0.0:
            self.scale_x = 1.0
        if self.scale_y == 0.0:
            self.scale_y = 1.0

    @property
    def unit_axis(self) -> np.ndarray:
        return np.asarray(self.axis, dtype=float)

    @property
    def origin_point(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=float)

    def evaluate_kinematics(self, t: float) -> Tuple[float, float, bool]:
        """Evaluate (omega, domega_dt, is_active) at time t.

        Matches cfield.F lines 111-126:
        - Sensor check: if ISENS /= 0, ts = t - tstart; if ts < 0, inactive.
        - Curve evaluation: VROT = scale_y * FINTER(ts * scale_x)
        - Derivative: DWDT = scale_y * scale_x * dF/dx
        """
        # Sensor delay
        if self.tstart is not None:
            ts = t - self.tstart
            if ts < 0.0:
                return 0.0, 0.0, False
        else:
            ts = t

        # Scaled time
        t_scaled = ts * self.scale_x

        if self.function is not None:
            vrot = self.scale_y * float(self.function(t_scaled))
            if self.flag == 2:
                if self.function_derivative is not None:
                    dwdt = self.scale_y * self.scale_x * float(self.function_derivative(t_scaled))
                else:
                    # Finite difference approximation of derivative
                    dt_eps = 1e-6
                    vrot_plus = self.scale_y * float(self.function(t_scaled + dt_eps))
                    vrot_minus = self.scale_y * float(self.function(t_scaled - dt_eps))
                    dwdt = (vrot_plus - vrot_minus) / (2.0 * dt_eps) * self.scale_x
            else:
                dwdt = 0.0
        else:
            vrot = self.scale_y * self.omega
            dwdt = self.domega_dt if (self.flag == 2 and self.domega_dt is not None) else 0.0

        return vrot, dwdt, True


def compute_centrifugal_forces(
    nodes_x: np.ndarray,
    nodes_v: Optional[np.ndarray],
    nodes_m: Union[float, np.ndarray, Sequence[float]],
    load: LoadCentri,
    t: float = 0.0,
    dt: float = 1e-4,
) -> CentrifugalResult:
    """Compute nodal centrifugal forces, accelerations, and work increment matching cfield.F.

    Parameters:
    -----------
    nodes_x : np.ndarray
        Coordinates of nodes, shape (N, 3) or (3,).
    nodes_v : np.ndarray, optional
        Velocities of nodes, shape (N, 3). If None, assumed zero.
    nodes_m : float or np.ndarray
        Nodal masses, scalar or shape (N,).
    load : LoadCentri
        Centrifugal load definition.
    t : float
        Current simulation time.
    dt : float
        Current simulation time step.

    Returns:
    --------
    CentrifugalResult
        Contains forces, accelerations, distances, omega, domega_dt, and work increment.
    """
    pts = np.atleast_2d(nodes_x).copy()
    n_nodes = pts.shape[0]

    if nodes_v is None:
        vel = np.zeros_like(pts)
    else:
        vel = np.atleast_2d(nodes_v).copy()
        if vel.shape[0] != n_nodes:
            vel = np.zeros_like(pts)

    if np.isscalar(nodes_m):
        mass = np.full(n_nodes, float(nodes_m), dtype=float)
    else:
        mass = np.asarray(nodes_m, dtype=float)
        if mass.ndim == 0:
            mass = np.full(n_nodes, float(mass), dtype=float)
        elif len(mass) != n_nodes:
            mass = np.resize(mass, n_nodes)

    # Kinematics evaluation
    omega, dwdt, is_active = load.evaluate_kinematics(t)

    if not is_active:
        zero_3 = np.zeros_like(pts)
        if nodes_x.ndim == 1:
            return CentrifugalResult(
                forces=zero_3[0],
                accelerations=zero_3[0],
                distances=zero_3[0],
                omega=0.0,
                domega_dt=0.0,
                work_increment=0.0,
                active=False,
            )
        return CentrifugalResult(
            forces=zero_3,
            accelerations=zero_3,
            distances=zero_3,
            omega=0.0,
            domega_dt=0.0,
            work_increment=0.0,
            active=False,
        )

    # 1. Relative position vector: r = x - x0 (cfield.F lines 151-153)
    x0 = load.origin_point
    u_axis = load.unit_axis
    r = pts - x0[None, :]

    # 2. Perpendicular distance vector to rotation axis:
    # dist = r - (r . u_axis) * u_axis (cfield.F lines 154-156)
    proj = np.sum(r * u_axis[None, :], axis=1, keepdims=True)  # (N, 1)
    dist = r - proj * u_axis[None, :]  # (N, 3)

    # 3. Acceleration calculation (cfield.F lines 160-162)
    omega2 = omega * omega
    a_centri = dist * omega2  # Outward radial acceleration

    if load.flag == 2 and abs(dwdt) > 1e-15:
        # dw = dwdt * u_axis
        # a_euler = dw x dist
        dw = dwdt * u_axis
        a_euler = np.cross(dw[None, :], dist)  # Azimuthal tangential acceleration
        a_rel = a_centri + a_euler
    else:
        a_rel = a_centri

    # 4. Force on node: F_node = m_node * a_rel
    forces = mass[:, None] * a_rel

    # 5. External work calculation (cfield.F lines 168-172):
    # v_mid = v + 0.5 * dt * a_rel
    # dW = sum( m * (a_rel . v_mid) ) * dt
    v_mid = vel + 0.5 * dt * a_rel
    work_inc = float(np.sum(mass * np.sum(a_rel * v_mid, axis=1)) * dt)

    if nodes_x.ndim == 1:
        return CentrifugalResult(
            forces=forces[0],
            accelerations=a_rel[0],
            distances=dist[0],
            omega=omega,
            domega_dt=dwdt,
            work_increment=work_inc,
            active=True,
        )

    return CentrifugalResult(
        forces=forces,
        accelerations=a_rel,
        distances=dist,
        omega=omega,
        domega_dt=dwdt,
        work_increment=work_inc,
        active=True,
    )


def cfield_subroutine(
    x: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    u_axis: Sequence[float],
    x0: Sequence[float],
    omega: float,
    domega_dt: float = 0.0,
    dt: float = 1e-4,
    flag: int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Direct functional wrapper matching OpenRadioss SUBROUTINE CFIELD_1.

    Parameters:
    -----------
    x : np.ndarray (N, 3)
        Nodal positions.
    v : np.ndarray (N, 3)
        Nodal velocities.
    m : np.ndarray (N,)
        Nodal masses.
    u_axis : (3,)
        Unit rotation axis.
    x0 : (3,)
        Origin point on axis.
    omega : float
        Angular velocity (rad/s).
    domega_dt : float
        Angular acceleration (rad/s^2).
    dt : float
        Time step.
    flag : int
        Calculation flag (2 = include domega/dt, 1 = pure centrifugal).

    Returns:
    --------
    (forces, accelerations, work_increment)
    """
    load = LoadCentri(
        axis=tuple(u_axis),  # type: ignore[arg-type]
        origin=tuple(x0),  # type: ignore[arg-type]
        omega=omega,
        domega_dt=domega_dt,
        flag=flag,
    )
    res = compute_centrifugal_forces(x, v, m, load, t=0.0, dt=dt)
    return res.forces, res.accelerations, res.work_increment


class CentrifugalEngine:
    """Engine manager for multiple /LOAD/CENTRI centrifugal fields."""

    def __init__(self, loads: Optional[Sequence[LoadCentri]] = None):
        self.loads: List[LoadCentri] = list(loads) if loads else []
        self.wfext: float = 0.0  # Cumulative external work (WFEXT in cfield.F)

    def add_load(self, load: LoadCentri) -> None:
        self.loads.append(load)

    def step(
        self,
        nodes_x: np.ndarray,
        nodes_v: np.ndarray,
        nodes_m: np.ndarray,
        t: float,
        dt: float,
        f_ext: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Apply all active centrifugal loads to nodal external forces and accumulate work."""
        if f_ext is None:
            f_ext = np.zeros_like(nodes_x)

        step_work = 0.0
        for load in self.loads:
            res = compute_centrifugal_forces(nodes_x, nodes_v, nodes_m, load, t=t, dt=dt)
            if res.active:
                f_ext += res.forces
                step_work += res.work_increment

        self.wfext += step_work
        return f_ext, step_work


def build_load_centri(data: Union[Dict[str, Any], Any]) -> LoadCentri:
    """Build a LoadCentri instance from a dictionary, keyword record, or entity."""
    if isinstance(data, LoadCentri):
        return data

    d: Dict[str, Any] = {}
    if hasattr(data, "__dict__"):
        d = dict(data.__dict__)
    elif isinstance(data, dict):
        d = dict(data)

    cid = int(d.get("id", d.get("user_id", 1)))
    title = str(d.get("title", ""))
    grnod_id = int(d.get("grnod_id", d.get("grnd_id", d.get("entityid", 0))))
    node_ids = d.get("node_ids")

    rad_dir = d.get("rad_dir")
    dir_id_val = d.get("dir_id")
    dir_id = None
    if dir_id_val is not None:
        try:
            dir_id = int(dir_id_val)
        except (ValueError, TypeError):
            pass

    if dir_id is None and isinstance(rad_dir, str):
        s = rad_dir.strip().upper()
        if s in ("X", "1"):
            dir_id = 1
        elif s in ("Y", "2"):
            dir_id = 2
        elif s in ("Z", "3"):
            dir_id = 3
        elif s in ("XX", "4"):
            dir_id = 4
        elif s in ("YY", "5"):
            dir_id = 5
        elif s in ("ZZ", "6"):
            dir_id = 6
    elif dir_id is None and rad_dir is not None:
        try:
            dir_id = int(rad_dir)
        except (ValueError, TypeError):
            dir_id = 6
    elif dir_id is None:
        dir_id = 6

    axis = d.get("axis")
    origin = d.get("origin", (0.0, 0.0, 0.0))
    if "x0" in d or "y0" in d or "z0" in d:
        origin = (float(d.get("x0", 0.0)), float(d.get("y0", 0.0)), float(d.get("z0", 0.0)))

    omega = float(d.get("omega", d.get("magnitude", 0.0)))
    funct_id = int(d.get("funct_id", d.get("fct_id", d.get("curveid", 0))))
    function = d.get("function")
    function_deriv = d.get("function_derivative")

    scale_x = float(d.get("scale_x", d.get("xscale", 1.0)))
    scale_y = float(d.get("scale_y", d.get("magnitude", 1.0)))
    domega_dt = float(d.get("domega_dt")) if d.get("domega_dt") is not None else None

    frame_id = int(d.get("frame_id", d.get("inputsystem", 0)))
    sensor_id = int(d.get("sensor_id", d.get("rad_sensor_id", 0)))
    tstart = float(d.get("tstart")) if d.get("tstart") is not None else None
    flag = int(d.get("flag", d.get("rad_ivar_flag", 2)))

    return LoadCentri(
        id=cid,
        title=title,
        grnod_id=grnod_id,
        node_ids=node_ids,
        dir_id=dir_id,
        axis=axis,
        origin=origin,
        omega=omega,
        funct_id=funct_id,
        function=function,
        function_derivative=function_deriv,
        scale_x=scale_x,
        scale_y=scale_y,
        domega_dt=domega_dt,
        frame_id=frame_id,
        sensor_id=sensor_id,
        tstart=tstart,
        flag=flag,
    )
