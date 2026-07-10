"""
/SENSOR — event sensors gating loads and contact interfaces (M6).

Fortran origin: ``engine/source/tools/sensor/`` (sensor_base.F and the
per-type routines) and the starter reader
``starter/source/tools/sensor/hm_read_sensor.F``. Two of the many
Radioss sensor types are ported:

* ``/SENSOR/TIME``  — fires at a fixed delay Tdelay;
* ``/SENSOR/DISP``  — fires when a node's displacement magnitude first
  exceeds Dmin (the crash-detection classic: the load/interface turns on
  when the structure has moved far enough).

Semantics (matching the original):

* a sensor LATCHES: once fired it stays active for the rest of the run
  (the fire time is part of the restart state — a chained run must not
  re-arm sensors);
* a load (/CLOAD, /PLOAD) gated by a sensor evaluates its function with
  the time origin SHIFTED to the fire time, f(t - t_fire) — the curve
  describes the load's own history from its activation, which is how
  airbag-style curves are written;
* a contact interface gated by a sensor is simply inactive before the
  fire time: no forces, no candidate tracking, and no time-step claim —
  when it fires, its NEAR-candidate stiffness accumulation (the M4
  interface-dt lesson) pulls dt down within a cycle, exactly like a
  fresh impact.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..model.model import Model


class Sensors:
    """Engine-side sensor status board."""

    def __init__(self, model: Model, log):
        self.model = model
        self.defs = {}                    # id -> (kind, params)
        self.fire_time: Dict[int, float] = {}   # id -> t_fire (latched)
        for sn in model.sensors:
            if sn.kind == "TIME":
                self.defs[sn.id] = ("TIME", sn.tdelay, -1)
                log.info(f"     /SENSOR/TIME/{sn.id}: FIRES AT T = "
                         f"{sn.tdelay:12.5E}")
            else:  # DISP
                try:
                    ni = model.node_index(sn.node_id)
                except KeyError:
                    log.error(f"/SENSOR/DISP/{sn.id}: unknown node "
                              f"{sn.node_id}", "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("DISP", sn.dmin, ni)
                log.info(f"     /SENSOR/DISP/{sn.id}: FIRES WHEN NODE "
                         f"{sn.node_id} MOVES {sn.dmin:g}")

    def __len__(self):
        return len(self.defs)

    # ------------------------------------------------------------------
    def update(self, t: float, log) -> None:
        """Poll every unfired sensor (cheap: a few scalar tests)."""
        for sid, (kind, val, ni) in self.defs.items():
            if sid in self.fire_time:
                continue
            if kind == "TIME":
                if t >= val:
                    self.fire_time[sid] = val   # fires exactly at Tdelay
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{t:12.5E}")
            else:
                d = self.model.x[ni] - self.model.x0[ni]
                if float(d @ d) >= val * val:
                    self.fire_time[sid] = t
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{t:12.5E} (DISPLACEMENT CRITERION)")

    # ------------------------------------------------------------------
    def active(self, sens_id: int) -> bool:
        """True when the option gated by ``sens_id`` may act (an option
        without a sensor — sens_id 0 — is always active; an unknown
        sensor id was flagged at init and reads as never-firing)."""
        if sens_id == 0:
            return True
        return sens_id in self.fire_time

    def shifted_time(self, sens_id: int, t: float) -> Optional[float]:
        """The load-evaluation time: t for ungated options, t - t_fire
        for gated ones, None while the sensor has not fired."""
        if sens_id == 0:
            return t
        tf = self.fire_time.get(sens_id)
        return None if tf is None else t - tf
