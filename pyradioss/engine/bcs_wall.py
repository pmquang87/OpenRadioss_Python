"""
/BCS/WALL — Sliding Wall Boundary Condition Trigger Manager.

Fortran origin:
  - ``engine/source/boundary_conditions/bcs_wall_trigger.F90`` (BCS_WALL_TRIGGER)
  - ``starter/source/boundary_conditions/hm_read_bcs_wall.F90`` (card reading)
  - ``starter/source/boundary_conditions/init_bcs_wall.F90`` (initialization)

Physics:
  Sliding wall boundary condition (/BCS/WALL) activates or deactivates based on:
  1. Sensor status: if sensor_id > 0, sensor trigger time controls activation window.
  2. Time window: if no sensor (or sensor triggered), active during [tstart, tstop].
     tstart <= t <= tstop (or t < tstop).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from ..model.model import Model


class SlidingWallBcsEngine:
    """Sliding wall boundary condition trigger and status manager for /BCS/WALL."""

    def __init__(self, model: Any, log: Optional[Any] = None):
        self.model = model
        self.log = log
        self.is_enabled: Dict[int, bool] = {}

        # Cache walls from model.bcs_walls
        self.walls: Dict[int, Any] = getattr(model, "bcs_walls", {})

    def is_active(self, wall_id: int, t: float, sensors: Optional[Any] = None) -> bool:
        """Check if sliding wall boundary condition is currently active.

        Fortran origin: bcs_wall_trigger.F90 lines 86-138.
        """
        if wall_id not in self.walls:
            # Fallback search if wall_id not directly in keys
            found_wall = None
            for wid, w in self.walls.items():
                if wid == wall_id or getattr(w, "id", None) == wall_id:
                    found_wall = w
                    break
            if found_wall is None:
                return False
            wall = found_wall
        else:
            wall = self.walls[wall_id]

        sensor_id = int(getattr(wall, "sensor_id", 0) or getattr(wall, "sens_id", 0) or 0)
        tstart = float(getattr(wall, "tstart", 0.0) or 0.0)
        tstop = float(getattr(wall, "tstop", 0.0) or 0.0)
        if tstop <= 0.0:
            tstop = float("inf")

        # 1. Sensor dependency check
        if sensor_id > 0:
            if sensors is not None:
                # Check if sensor has fired
                if hasattr(sensors, "active") and not sensors.active(sensor_id):
                    self.is_enabled[wall_id] = False
                    return False
                if hasattr(sensors, "shifted_time"):
                    te = sensors.shifted_time(sensor_id, t)
                    if te is None:
                        self.is_enabled[wall_id] = False
                        return False

        # 2. Time window check [tstart, tstop]
        active = (tstart <= t <= tstop)
        self.is_enabled[wall_id] = active
        return active

    def check_triggers(self, t: float, sensors: Optional[Any] = None) -> Dict[int, bool]:
        """Evaluate and return active state for all registered /BCS/WALL conditions."""
        status = {}
        for wid in self.walls:
            status[wid] = self.is_active(wid, t, sensors)
        return status
