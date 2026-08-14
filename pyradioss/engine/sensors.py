"""
/SENSOR — event sensors gating loads and contact interfaces (M6, M84).

Fortran origin: ``engine/source/tools/sensor/`` (sensor_base.F and the
per-type routines) and the starter reader
``starter/source/tools/sensor/hm_read_sensor.F``. Supported types:

* ``/SENSOR/TIME`` — fires at a fixed delay Tdelay;
* ``/SENSOR/DISP`` — fires when a node's displacement magnitude first
  exceeds Dmin;
* ``/SENSOR/VEL``  — fires when a node's velocity magnitude first exceeds Vmax;
* ``/SENSOR/NOT``  — active when sens_id1 is inactive;
* ``/SENSOR/AND``  — active when both sens_id1 and sens_id2 are active;
* ``/SENSOR/OR``   — active when either sens_id1 or sens_id2 is active.

Semantics (matching the original):

* physical sensors (TIME, DISP, VEL) LATCH: once fired, they stay active
  for the rest of the run (the fire time is part of the restart state);
* logical sensors (NOT, AND, OR) update dynamically per cycle based on
  the state of their inputs and their Tdelay;
* a load (/CLOAD, /PLOAD) gated by a sensor evaluates its function with
  the time origin SHIFTED to the fire time, f(t - t_fire);
* a contact interface gated by a sensor is simply inactive before the
  fire time.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..model.model import Model


class Sensors:
    """Engine-side sensor status board."""

    def __init__(self, model: Model, log):
        self.model = model
        self.defs = {}                    # id -> (kind, params...)
        self.fire_time: Dict[int, float] = {}   # id -> t_fire (latched/active)
        self.status: Dict[int, bool] = {}       # id -> active status bool
        for sn in model.sensors:
            if sn.kind == "TIME":
                self.defs[sn.id] = ("TIME", sn.tdelay)
                log.info(f"     /SENSOR/TIME/{sn.id}: FIRES AT T = "
                         f"{sn.tdelay:12.5E}")
            elif sn.kind == "DISP":
                try:
                    ni = model.node_index(sn.node_id)
                except KeyError:
                    log.error(f"/SENSOR/DISP/{sn.id}: unknown node "
                              f"{sn.node_id}", "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("DISP", sn.dmin, ni, sn.tdelay)
                log.info(f"     /SENSOR/DISP/{sn.id}: FIRES WHEN NODE "
                         f"{sn.node_id} MOVES {sn.dmin:g}")
            elif sn.kind == "VEL":
                try:
                    ni = model.node_index(sn.node_id)
                except KeyError:
                    log.error(f"/SENSOR/VEL/{sn.id}: unknown node "
                              f"{sn.node_id}", "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("VEL", sn.vmax, ni, sn.tdelay, sn.fcut)
                log.info(f"     /SENSOR/VEL/{sn.id}: FIRES WHEN NODE "
                         f"{sn.node_id} VELOCITY >= {sn.vmax:g}")
            elif sn.kind == "NOT":
                self.defs[sn.id] = ("NOT", sn.sens_id1, sn.tdelay)
                log.info(f"     /SENSOR/NOT/{sn.id}: ACTIVE WHEN SENSOR "
                         f"{sn.sens_id1} IS INACTIVE")
            elif sn.kind == "AND":
                self.defs[sn.id] = ("AND", sn.sens_id1, sn.sens_id2, sn.tdelay)
                log.info(f"     /SENSOR/AND/{sn.id}: ACTIVE WHEN SENSORS "
                         f"{sn.sens_id1} AND {sn.sens_id2} ARE ACTIVE")
            elif sn.kind == "OR":
                self.defs[sn.id] = ("OR", sn.sens_id1, sn.sens_id2, sn.tdelay)
                log.info(f"     /SENSOR/OR/{sn.id}: ACTIVE WHEN SENSOR "
                         f"{sn.sens_id1} OR {sn.sens_id2} IS ACTIVE")

    def __len__(self):
        return len(self.defs)

    # ------------------------------------------------------------------
    def update(self, t: float, log) -> None:
        """Poll every physical sensor and update logical sensors."""
        # 1. Update physical sensors
        for sid, defn in self.defs.items():
            kind = defn[0]
            if kind in ("NOT", "AND", "OR"):
                continue
            if self.status.get(sid, False):
                continue
            if kind == "TIME":
                tdelay = defn[1]
                if t >= tdelay:
                    self.fire_time[sid] = tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{t:12.5E}")
            elif kind == "DISP":
                dmin, ni, tdelay = defn[1], defn[2], defn[3]
                if t >= tdelay:
                    d = self.model.x[ni] - self.model.x0[ni]
                    if float(d @ d) >= dmin * dmin:
                        self.fire_time[sid] = t
                        self.status[sid] = True
                        log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                                 f"{t:12.5E} (DISPLACEMENT CRITERION)")
            elif kind == "VEL":
                vmax, ni, tdelay, fcut = defn[1], defn[2], defn[3], defn[4]
                if t >= tdelay:
                    v = self.model.v[ni]
                    if float(v @ v) >= vmax * vmax:
                        self.fire_time[sid] = t
                        self.status[sid] = True
                        log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                                 f"{t:12.5E} (VELOCITY CRITERION)")

        # 2. Update logical sensors (iterate to handle cascaded dependencies)
        changed = True
        max_iters = len(self.defs) + 1
        it = 0
        while changed and it < max_iters:
            changed = False
            it += 1
            for sid, defn in self.defs.items():
                kind = defn[0]
                if kind not in ("NOT", "AND", "OR"):
                    continue
                old_active = self.status.get(sid, False)
                if kind == "NOT":
                    is1, tdelay = defn[1], defn[2]
                    new_active = (not self.active(is1)) and (t >= tdelay)
                elif kind == "AND":
                    is1, is2, tdelay = defn[1], defn[2], defn[3]
                    new_active = self.active(is1) and self.active(is2) and (t >= tdelay)
                elif kind == "OR":
                    is1, is2, tdelay = defn[1], defn[2], defn[3]
                    new_active = (self.active(is1) or self.active(is2)) and (t >= tdelay)
                else:
                    new_active = False

                if new_active != old_active:
                    changed = True
                    self.status[sid] = new_active
                    if new_active:
                        self.fire_time[sid] = t
                        log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                                 f"{t:12.5E} (LOGICAL {kind})")
                    else:
                        self.fire_time.pop(sid, None)
                        log.info(f" -- /SENSOR/{sid} DEACTIVATED AT TIME "
                                 f"{t:12.5E} (LOGICAL {kind})")

    # ------------------------------------------------------------------
    def active(self, sens_id: int) -> bool:
        """True when the option gated by ``sens_id`` may act (an option
        without a sensor — sens_id 0 — is always active; an unknown
        sensor id was flagged at init and reads as never-firing)."""
        if sens_id == 0:
            return True
        return self.status.get(sens_id, False)

    def shifted_time(self, sens_id: int, t: float) -> Optional[float]:
        """The load-evaluation time: t for ungated options, t - t_fire
        for gated ones, None while the sensor has not fired."""
        if sens_id == 0:
            return t
        if not self.active(sens_id):
            return None
        tf = self.fire_time.get(sens_id, t)
        return t - tf

