"""
/SENSOR — event sensors gating loads and contact interfaces (M6, M84).

Fortran origin: ``engine/source/tools/sensor/`` (sensor_base.F and the
per-type routines) and the starter reader
``starter/source/tools/sensor/hm_read_sensor.F``. Supported types:

* ``/SENSOR/TIME``   — fires at a fixed delay Tdelay;
* ``/SENSOR/DISP``   — fires when a node's displacement magnitude first
  exceeds Dmin;
* ``/SENSOR/VEL``    — fires when a node's velocity magnitude first exceeds Vmax;
* ``/SENSOR/DIST``   — fires when distance between two nodes is outside [dmin, dmax];
* ``/SENSOR/ACC``    — fires when a node's acceleration magnitude or component exceeds a_max;
* ``/SENSOR/ENERGY`` — fires when model energy (kinetic, internal, total) exceeds E_max;
* ``/SENSOR/RWALL``  — fires when normal or total force on a rigid wall exceeds F_max;
* ``/SENSOR/SECT``   — fires when cross-section resultant force exceeds F_max;
* ``/SENSOR/TEMP``   — fires when node temperature exceeds T_max;
* ``/SENSOR/NOT``    — active when sens_id1 is inactive;
* ``/SENSOR/AND``    — active when both sens_id1 and sens_id2 are active;
* ``/SENSOR/OR``     — active when either sens_id1 or sens_id2 is active.

Semantics (matching the original):

* physical sensors (TIME, DISP, VEL, DIST, ACC, ENERGY, RWALL, SECT, TEMP) LATCH:
  once fired, they stay active for the rest of the run (the fire time is part of the restart state);
* logical sensors (NOT, AND, OR) update dynamically per cycle based on
  the state of their inputs and their Tdelay;
* a load (/CLOAD, /PLOAD) gated by a sensor evaluates its function with
  the time origin SHIFTED to the fire time, f(t - t_fire);
* a contact interface gated by a sensor is simply inactive before the
  fire time.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any

import numpy as np

from ..model.model import Model


class Sensors:
    """Engine-side sensor status board."""

    def __init__(self, model: Model, log):
        self.model = model
        self.defs = {}                    # id -> (kind, params...)
        self.fire_time: Dict[int, float] = {}   # id -> t_fire (latched/active)
        self.crit_time: Dict[int, float] = {}   # id -> t_crit (threshold first crossed)
        self.status: Dict[int, bool] = {}       # id -> active status bool
        for sn in model.sensors:
            kind_upper = sn.kind.upper()
            if kind_upper == "TIME":
                self.defs[sn.id] = ("TIME", sn.tdelay)
                log.info(f"     /SENSOR/TIME/{sn.id}: FIRES AT T = "
                         f"{sn.tdelay:12.5E}")
            elif kind_upper == "DISP":
                try:
                    ni = model.node_index(sn.node_id)
                except KeyError:
                    log.error(f"/SENSOR/DISP/{sn.id}: unknown node "
                              f"{sn.node_id}", "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("DISP", sn.dmin, ni, sn.tdelay)
                log.info(f"     /SENSOR/DISP/{sn.id}: FIRES WHEN NODE "
                         f"{sn.node_id} MOVES {sn.dmin:g}")
            elif kind_upper == "VEL":
                try:
                    ni = model.node_index(sn.node_id)
                except KeyError:
                    log.error(f"/SENSOR/VEL/{sn.id}: unknown node "
                              f"{sn.node_id}", "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("VEL", sn.vmax, ni, sn.tdelay, sn.fcut)
                log.info(f"     /SENSOR/VEL/{sn.id}: FIRES WHEN NODE "
                         f"{sn.node_id} VELOCITY >= {sn.vmax:g}")
            elif kind_upper == "DIST":
                try:
                    nid1 = getattr(sn, "node_id1", 0) or getattr(sn, "node1", 0)
                    nid2 = getattr(sn, "node_id2", 0) or getattr(sn, "node2", 0)
                    n1 = model.node_index(nid1)
                    n2 = model.node_index(nid2)
                except (KeyError, ValueError):
                    log.error(f"/SENSOR/DIST/{sn.id}: unknown node(s) "
                              f"{getattr(sn, 'node_id1', None)}, {getattr(sn, 'node_id2', None)}",
                              "SENSOR CHECK")
                    continue
                self.defs[sn.id] = ("DIST", n1, n2, sn.dmin, sn.dmax, sn.tdelay)
                log.info(f"     /SENSOR/DIST/{sn.id}: FIRES WHEN DISTANCE BETWEEN "
                         f"NODES {nid1} AND {nid2} OUTSIDE [{sn.dmin:g}, {sn.dmax:g}]")
            elif kind_upper in ("ACC", "ACCE"):
                # Nodal / accelerometer sensor
                # Fortran: read_sensor_acc.F / sensor_acc.F
                nid = getattr(sn, "node_id", 0) or getattr(sn, "accel_id", 0)
                if not nid and getattr(sn, "acc_entries", None):
                    nid = sn.acc_entries[0][0]
                try:
                    ni = model.node_index(nid) if nid else 0
                except (KeyError, ValueError):
                    if hasattr(model, "_id2idx") and nid in model._id2idx:
                        ni = model._id2idx[nid]
                    else:
                        log.error(f"/SENSOR/ACC/{sn.id}: unknown node/accel {nid}", "SENSOR CHECK")
                        continue
                a_max = (
                    getattr(sn, "a_max", 0.0)
                    or getattr(sn, "acc_max", 0.0)
                    or getattr(sn, "tomin", 0.0)
                    or getattr(sn, "tomin1", 0.0)
                )
                if a_max == 0.0 and getattr(sn, "acc_entries", None):
                    a_max = sn.acc_entries[0][2]
                if a_max == 0.0:
                    a_max = getattr(sn, "fmax", 0.0)
                dir_str = getattr(sn, "dir", "")
                if not dir_str and getattr(sn, "acc_entries", None):
                    dir_str = sn.acc_entries[0][1]
                dir_str = (dir_str or "XYZ").upper()
                tmin = getattr(sn, "tmin", 0.0)
                if tmin == 0.0 and getattr(sn, "acc_entries", None) and len(sn.acc_entries[0]) > 3:
                    tmin = sn.acc_entries[0][3]
                self.defs[sn.id] = ("ACC", ni, a_max, dir_str, sn.tdelay, tmin)
                log.info(f"     /SENSOR/ACC/{sn.id}: FIRES WHEN ACCELERATION AT NODE "
                         f"{nid} ({dir_str}) >= {a_max:g}")
            elif kind_upper == "ENERGY":
                # Model energy sensor (kinetic, internal, total)
                # Fortran: read_sensor_energy.F / sensor_energy.F
                e_max = getattr(sn, "e_max", 0.0)
                energy_type = getattr(sn, "energy_type", "").upper()
                if e_max == 0.0:
                    if getattr(sn, "kemax", 1e30) < 1e29:
                        e_max = sn.kemax
                        if not energy_type:
                            energy_type = "KE"
                    elif getattr(sn, "iemax", 1e30) < 1e29:
                        e_max = sn.iemax
                        if not energy_type:
                            energy_type = "IE"
                    else:
                        e_max = getattr(sn, "emax", 0.0)
                if not energy_type:
                    energy_type = "TOT"
                e_min = getattr(sn, "e_min", None)
                if e_min is None:
                    if energy_type == "KE" and getattr(sn, "kemin", -1e30) > -1e29:
                        e_min = sn.kemin
                    elif energy_type == "IE" and getattr(sn, "iemin", -1e30) > -1e29:
                        e_min = sn.iemin
                part_id = getattr(sn, "part_id", 0)
                tmin = getattr(sn, "tmin", 0.0)
                self.defs[sn.id] = ("ENERGY", energy_type, e_max, e_min, part_id, sn.tdelay, tmin)
                log.info(f"     /SENSOR/ENERGY/{sn.id}: FIRES WHEN {energy_type} "
                         f"ENERGY >= {e_max:g}")
            elif kind_upper == "RWALL":
                # Rigid wall force sensor
                # Fortran: read_sensor_rwall.F / sensor_rwall.F
                rwall_id = getattr(sn, "rwall_id", 0)
                f_max = getattr(sn, "f_max", 0.0) or getattr(sn, "fmax", 0.0)
                f_min = getattr(sn, "f_min", 0.0) or getattr(sn, "fmin", 0.0)
                dir_str = (getattr(sn, "dir", "") or "TF").upper()
                tmin = getattr(sn, "tmin", 0.0)
                self.defs[sn.id] = ("RWALL", rwall_id, f_max, f_min, dir_str, sn.tdelay, tmin)
                log.info(f"     /SENSOR/RWALL/{sn.id}: FIRES WHEN RIGID WALL {rwall_id} "
                         f"FORCE ({dir_str}) >= {f_max:g}")
            elif kind_upper in ("SECT", "SECTION", "XSECTION"):
                # Cross-section force sensor
                # Fortran: read_sensor_sect.F / sensor_section.F
                sect_id = (
                    getattr(sn, "sect_id", 0)
                    or getattr(sn, "section_id", 0)
                    or getattr(sn, "crosssectionid", 0)
                )
                f_max = getattr(sn, "f_max", 0.0) or getattr(sn, "fmax", 0.0)
                f_min = getattr(sn, "f_min", 0.0) or getattr(sn, "fmin", 0.0)
                dir_str = (getattr(sn, "dir", "") or "TF").upper()
                tmin = getattr(sn, "tmin", 0.0)
                self.defs[sn.id] = ("SECT", sect_id, f_max, f_min, dir_str, sn.tdelay, tmin)
                log.info(f"     /SENSOR/SECT/{sn.id}: FIRES WHEN SECTION {sect_id} "
                         f"FORCE ({dir_str}) >= {f_max:g}")
            elif kind_upper in ("TEMP", "TEMPERATURE"):
                # Nodal temperature sensor
                # Fortran: read_sensor_temp.F / sensor_temp.F
                nid = getattr(sn, "node_id", 0)
                grnod_id = getattr(sn, "grnod_id", 0)
                t_max = getattr(sn, "t_max", 0.0)
                if t_max == 0.0 and getattr(sn, "tempmax", 1e30) < 1e29:
                    t_max = sn.tempmax
                if t_max == 0.0:
                    t_max = getattr(sn, "temp_max", 0.0)
                t_min = getattr(sn, "t_min", 0.0)
                if t_min == 0.0 and getattr(sn, "tempmin", 0.0) > 0.0:
                    t_min = sn.tempmin
                tmin = getattr(sn, "tmin", 0.0)
                ni = -1
                if nid:
                    try:
                        ni = model.node_index(nid)
                    except (KeyError, ValueError):
                        if hasattr(model, "_id2idx") and nid in model._id2idx:
                            ni = model._id2idx[nid]
                self.defs[sn.id] = ("TEMP", nid, ni, grnod_id, t_max, t_min, sn.tdelay, tmin)
                log.info(f"     /SENSOR/TEMP/{sn.id}: FIRES WHEN TEMPERATURE >= {t_max:g}")
            elif kind_upper == "HIC":
                from .biomech_sensors import HicParams, HicSensor
                nid = getattr(sn, "node_id", 0) or getattr(sn, "accel_id", 0)
                try:
                    ni = model.node_index(nid) if nid else 0
                except (KeyError, ValueError):
                    if hasattr(model, "_id2idx") and nid in model._id2idx:
                        ni = model._id2idx[nid]
                    else:
                        ni = 0
                sdir = getattr(sn, "dir", 1)
                if isinstance(sdir, str):
                    sdir_u = sdir.strip().upper()
                    dir_code = 2 if sdir_u in ("X", "1") else 3 if sdir_u in ("Y", "2") else 4 if sdir_u in ("Z", "3") else 1
                else:
                    dir_code = int(sdir) if sdir else 1
                hic_period = getattr(sn, "hic_period", 0.015) or 0.015
                hic_crit = getattr(sn, "hic_val", getattr(sn, "hic_crit", 700.0)) or 700.0
                grav = getattr(sn, "gravity", 9.80665) or 9.80665
                tmin = getattr(sn, "tmin", 0.0) or 0.0
                tdelay = getattr(sn, "tdelay", 0.0) or 0.0
                npoint = getattr(sn, "npoint", 200) or 200
                hp = HicParams(
                    id=sn.id,
                    title=getattr(sn, "title", f"HIC_{sn.id}"),
                    node_id=nid,
                    accel_id=getattr(sn, "accel_id", 0),
                    dir=dir_code,
                    hic_period=hic_period,
                    hic_crit=hic_crit,
                    gravity=grav,
                    tmin=tmin,
                    tdelay=tdelay,
                    npoint=npoint,
                )
                self.defs[sn.id] = ("HIC", HicSensor(hp), ni)
                log.info(f"     /SENSOR/HIC/{sn.id}: FIRES ON HIC >= {hic_crit:g} (NODE {nid})")
            elif kind_upper == "NIC":
                from .biomech_sensors import NicParams, NicSensor
                spring_id = getattr(sn, "spring_id", 0)
                nij_max = getattr(sn, "nij_max", 1.0) or 1.0
                fint_tens = getattr(sn, "fint_tens", 6806.0) or 6806.0
                fint_comp = getattr(sn, "fint_comp", 6160.0) or 6160.0
                mint_flex = getattr(sn, "mint_flex", 310.0) or 310.0
                mint_ext = getattr(sn, "mint_ext", 135.0) or 135.0
                tmin = getattr(sn, "tmin", 0.0) or 0.0
                tdelay = getattr(sn, "tdelay", 0.0) or 0.0
                cfc = getattr(sn, "cfc", 600.0) or 600.0
                alpha = getattr(sn, "alpha", 1.0) or 1.0
                np_params = NicParams(
                    id=sn.id,
                    title=getattr(sn, "title", f"NIC_{sn.id}"),
                    spring_id=spring_id,
                    nij_max=nij_max,
                    fint_tens=fint_tens,
                    fint_comp=fint_comp,
                    mint_flex=mint_flex,
                    mint_ext=mint_ext,
                    tmin=tmin,
                    tdelay=tdelay,
                    cfc=cfc,
                    alpha=alpha,
                )
                self.defs[sn.id] = ("NIC", NicSensor(np_params), spring_id)
                log.info(f"     /SENSOR/NIC/{sn.id}: FIRES ON NIJ >= {nij_max:g} (SPRING {spring_id})")
            elif kind_upper == "NOT":
                self.defs[sn.id] = ("NOT", sn.sens_id1, sn.tdelay)
                log.info(f"     /SENSOR/NOT/{sn.id}: ACTIVE WHEN SENSOR "
                         f"{sn.sens_id1} IS INACTIVE")
            elif kind_upper == "AND":
                self.defs[sn.id] = ("AND", sn.sens_id1, sn.sens_id2, sn.tdelay)
                log.info(f"     /SENSOR/AND/{sn.id}: ACTIVE WHEN SENSORS "
                         f"{sn.sens_id1} AND {sn.sens_id2} ARE ACTIVE")
            elif kind_upper == "OR":
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
                if sid not in self.crit_time:
                    d = self.model.x[ni] - self.model.x0[ni]
                    if float(d @ d) >= dmin * dmin:
                        self.crit_time[sid] = t
                if sid in self.crit_time and t >= self.crit_time[sid] + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (DISPLACEMENT CRITERION)")
            elif kind == "VEL":
                vmax, ni, tdelay, fcut = defn[1], defn[2], defn[3], defn[4]
                if sid not in self.crit_time:
                    v = self.model.v[ni]
                    if float(v @ v) >= vmax * vmax:
                        self.crit_time[sid] = t
                if sid in self.crit_time and t >= self.crit_time[sid] + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (VELOCITY CRITERION)")
            elif kind == "DIST":
                n1, n2, dmin, dmax, tdelay = defn[1], defn[2], defn[3], defn[4], defn[5]
                if sid not in self.crit_time:
                    diff = self.model.x[n1] - self.model.x[n2]
                    dist = float(np.linalg.norm(diff))
                    if dist < dmin or (dmax > 0.0 and dist > dmax):
                        self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (DISTANCE CRITERION)")
            elif kind == "ACC":
                # Nodal / accelerometer evaluation (sensor_acc.F lines 75-100)
                ni, a_max, dir_str, tdelay, tmin = defn[1], defn[2], defn[3], defn[4], defn[5]
                if sid not in self.crit_time:
                    a_arr = (
                        getattr(self.model, "a", None)
                        if getattr(self.model, "a", None) is not None
                        else getattr(self.model, "accel", None)
                    )
                    if a_arr is None:
                        a_arr = getattr(self.model, "accelerations", None)
                    if a_arr is not None and ni < len(a_arr):
                        a_vec = np.asarray(a_arr[ni], dtype=float)
                        if dir_str in ("X", "1"):
                            val = abs(float(a_vec[0]))
                        elif dir_str in ("Y", "2"):
                            val = abs(float(a_vec[1]))
                        elif dir_str in ("Z", "3"):
                            val = abs(float(a_vec[2]))
                        elif dir_str == "XY":
                            val = float(math.hypot(a_vec[0], a_vec[1]))
                        elif dir_str == "XZ":
                            val = float(math.hypot(a_vec[0], a_vec[2]))
                        elif dir_str == "YZ":
                            val = float(math.hypot(a_vec[1], a_vec[2]))
                        else:  # XYZ, TOTAL, MAG, or default
                            val = float(np.linalg.norm(a_vec))
                        if val >= a_max:
                            self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tmin + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tmin + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (ACCELERATION CRITERION)")
            elif kind == "ENERGY":
                # Model energy evaluation (sensor_energy.F lines 168-185)
                energy_type, e_max, e_min, part_id, tdelay, tmin = (
                    defn[1], defn[2], defn[3], defn[4], defn[5], defn[6]
                )
                if sid not in self.crit_time:
                    ke = ie = tot = None
                    energies_dict = getattr(self.model, "energies", None) or getattr(self.model, "energy", None)
                    if isinstance(energies_dict, dict):
                        ke = energies_dict.get("KE", energies_dict.get("KIN", None))
                        ie = energies_dict.get("IE", energies_dict.get("INT", None))
                        tot = energies_dict.get("TOT", energies_dict.get("TOTAL", None))

                    if ke is None and hasattr(self.model, "ke") and getattr(self.model, "ke") is not None:
                        ke = float(getattr(self.model, "ke"))
                    if ie is None and hasattr(self.model, "ie") and getattr(self.model, "ie") is not None:
                        ie = float(getattr(self.model, "ie"))
                    if tot is None and hasattr(self.model, "total_energy") and getattr(self.model, "total_energy") is not None:
                        tot = float(getattr(self.model, "total_energy"))

                    if ke is None and hasattr(self.model, "v") and hasattr(self.model, "mass"):
                        real = self.model.mass < 1e29
                        ke = float(0.5 * (self.model.mass[real, None] * self.model.v[real] ** 2).sum())
                    if ie is None and hasattr(self.model, "element_groups"):
                        ie_sum = 0.0
                        for _, grp in self.model.element_groups():
                            if hasattr(grp, "state") and "eint" in grp.state:
                                ie_sum += float(np.sum(grp.state["eint"]))
                        ie = ie_sum
                    if tot is None:
                        tot = (ke or 0.0) + (ie or 0.0)

                    if energy_type in ("KE", "KIN", "KINETIC"):
                        val = ke if ke is not None else 0.0
                    elif energy_type in ("IE", "INT", "INTERNAL"):
                        val = ie if ie is not None else 0.0
                    else:
                        val = tot if tot is not None else 0.0

                    cond = (val >= e_max) if e_max > 0.0 else False
                    if e_min is not None and val < e_min:
                        cond = True
                    if cond:
                        self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tmin + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tmin + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (ENERGY CRITERION)")
            elif kind == "RWALL":
                # Rigid wall force evaluation (sensor_rwall.F lines 80-125)
                rwall_id, f_max, f_min, dir_str, tdelay, tmin = (
                    defn[1], defn[2], defn[3], defn[4], defn[5], defn[6]
                )
                if sid not in self.crit_time:
                    forces = (
                        getattr(self.model, "rwall_forces", None)
                        or getattr(self.model, "rwall_reactions", None)
                        or getattr(self.model, "rwall_f", None)
                    )
                    f_val = None
                    if isinstance(forces, dict):
                        f_entry = forces.get(rwall_id, None)
                        if f_entry is None and len(forces) == 1 and rwall_id in (0, 1):
                            f_entry = next(iter(forces.values()))
                        if f_entry is not None:
                            if isinstance(f_entry, dict):
                                f_val = f_entry.get(dir_str, f_entry.get("TF", f_entry.get("FN", None)))
                            elif isinstance(f_entry, (np.ndarray, list, tuple)):
                                arr = np.asarray(f_entry, dtype=float)
                                if dir_str in ("X", "FX"):
                                    f_val = abs(arr[0])
                                elif dir_str in ("Y", "FY"):
                                    f_val = abs(arr[1])
                                elif dir_str in ("Z", "FZ"):
                                    f_val = abs(arr[2])
                                elif dir_str in ("FN", "NORMAL"):
                                    f_val = abs(arr[0])
                                elif dir_str in ("FT", "TANGENT"):
                                    f_val = float(np.linalg.norm(arr[1:])) if len(arr) > 1 else 0.0
                                else:
                                    f_val = float(np.linalg.norm(arr))
                            else:
                                f_val = abs(float(f_entry))
                    elif hasattr(self.model, "rwalls"):
                        for rw in self.model.rwalls:
                            if rw.id == rwall_id or (rwall_id == 0 and len(self.model.rwalls) == 1):
                                f_cand = getattr(rw, "force", getattr(rw, "f", getattr(rw, "fn", None)))
                                if f_cand is not None:
                                    f_val = abs(float(f_cand)) if np.isscalar(f_cand) else float(np.linalg.norm(f_cand))
                                break
                    if f_val is not None:
                        cond = (f_val >= f_max) if f_max > 0.0 else False
                        if f_min > 0.0 and f_val < f_min:
                            cond = True
                        if cond:
                            self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tmin + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tmin + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (RIGID WALL CRITERION)")
            elif kind == "SECT":
                # Cross-section force evaluation (sensor_section.F lines 85-215)
                sect_id, f_max, f_min, dir_str, tdelay, tmin = (
                    defn[1], defn[2], defn[3], defn[4], defn[5], defn[6]
                )
                if sid not in self.crit_time:
                    sforces = (
                        getattr(self.model, "section_forces", None)
                        or getattr(self.model, "sect_forces", None)
                        or getattr(self.model, "sections_forces", None)
                    )
                    f_val = None
                    if isinstance(sforces, dict):
                        s_entry = sforces.get(sect_id, None)
                        if s_entry is None and len(sforces) == 1 and sect_id in (0, 1):
                            s_entry = next(iter(sforces.values()))
                        if s_entry is not None:
                            if isinstance(s_entry, tuple) and len(s_entry) == 2:
                                F_vec = np.asarray(s_entry[0], dtype=float)
                                M_vec = np.asarray(s_entry[1], dtype=float)
                                if dir_str in ("FX", "X"):
                                    f_val = abs(float(F_vec[0]))
                                elif dir_str in ("FY", "Y"):
                                    f_val = abs(float(F_vec[1]))
                                elif dir_str in ("FZ", "Z"):
                                    f_val = abs(float(F_vec[2]))
                                elif dir_str == "FN":
                                    f_val = abs(float(F_vec[0]))
                                elif dir_str == "FT":
                                    f_val = float(np.linalg.norm(F_vec[1:])) if len(F_vec) > 1 else 0.0
                                elif dir_str in ("TM", "MOMENT", "M"):
                                    f_val = float(np.linalg.norm(M_vec))
                                elif dir_str == "MX":
                                    f_val = abs(float(M_vec[0]))
                                elif dir_str == "MY":
                                    f_val = abs(float(M_vec[1]))
                                elif dir_str == "MZ":
                                    f_val = abs(float(M_vec[2]))
                                else:  # TF or default resultant force
                                    f_val = float(np.linalg.norm(F_vec))
                            elif isinstance(s_entry, dict):
                                f_val = s_entry.get(dir_str, s_entry.get("TF", s_entry.get("FN", None)))
                            elif isinstance(s_entry, (np.ndarray, list)):
                                arr = np.asarray(s_entry, dtype=float)
                                if dir_str in ("FX", "X"):
                                    f_val = abs(float(arr[0]))
                                elif dir_str in ("FY", "Y"):
                                    f_val = abs(float(arr[1]))
                                elif dir_str in ("FZ", "Z"):
                                    f_val = abs(float(arr[2]))
                                else:
                                    f_val = float(np.linalg.norm(arr))
                            else:
                                f_val = abs(float(s_entry))
                    if f_val is not None:
                        cond = (f_val >= f_max) if f_max > 0.0 else False
                        if f_min > 0.0 and f_val < f_min:
                            cond = True
                        if cond:
                            self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tmin + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tmin + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (SECTION CRITERION)")
            elif kind == "TEMP":
                # Nodal temperature evaluation (sensor_temp.F lines 75-100)
                nid, ni, grnod_id, t_max, t_min, tdelay, tmin = (
                    defn[1], defn[2], defn[3], defn[4], defn[5], defn[6], defn[7]
                )
                if sid not in self.crit_time:
                    t_val = None
                    temps = (
                        getattr(self.model, "temperatures", None)
                        or getattr(self.model, "temperature", None)
                        or getattr(self.model, "temp", None)
                        or getattr(self.model, "t_node", None)
                    )
                    if temps is not None:
                        if isinstance(temps, dict):
                            if nid in temps:
                                t_val = float(temps[nid])
                            elif grnod_id and hasattr(self.model, "node_groups") and grnod_id in self.model.node_groups:
                                grp = self.model.node_groups[grnod_id]
                                node_list = getattr(grp, "nodes", getattr(grp, "node_ids", []))
                                grp_temps = [temps[n] for n in node_list if n in temps]
                                if grp_temps:
                                    t_val = max(grp_temps)
                        elif isinstance(temps, (np.ndarray, list)):
                            arr = np.asarray(temps, dtype=float)
                            if ni >= 0 and ni < len(arr):
                                t_val = float(arr[ni])
                            elif grnod_id and hasattr(self.model, "node_groups") and grnod_id in self.model.node_groups:
                                grp = self.model.node_groups[grnod_id]
                                idxs = getattr(grp, "node_idx", None)
                                if idxs is not None and len(idxs) > 0:
                                    t_val = float(np.max(arr[idxs]))
                                else:
                                    t_val = float(np.max(arr))
                            elif nid == 0:
                                t_val = float(np.max(arr))
                    if t_val is not None:
                        cond = (t_val >= t_max) if t_max > 0.0 else False
                        if t_min > 0.0 and t_val < t_min:
                            cond = True
                        if cond:
                            self.crit_time[sid] = t
                if sid in self.crit_time and (t + 1e-12) >= self.crit_time[sid] + tmin + tdelay:
                    self.fire_time[sid] = self.crit_time[sid] + tmin + tdelay
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (TEMPERATURE CRITERION)")
            elif kind == "HIC":
                hic_sensor, ni = defn[1], defn[2]
                dt = getattr(self.model, "dt", 0.0)
                if dt <= 0.0:
                    dt = max(t - getattr(self, "_t_prev", 0.0), 1e-6)
                node_accel = getattr(self.model, "nodal_accel", getattr(self.model, "a", None))
                if node_accel is not None and len(node_accel) > ni:
                    accel_vec = np.asarray(node_accel[ni], dtype=float)
                elif hasattr(self.model, "v") and self.model.v is not None and hasattr(self, "_v_prev") and len(self.model.v) > ni and len(self._v_prev) > ni:
                    accel_vec = (self.model.v[ni] - self._v_prev[ni]) / max(dt, 1e-12)
                else:
                    accel_vec = np.zeros(3, dtype=float)

                fired = hic_sensor.update(t, dt, accel_vec)
                if fired:
                    self.fire_time[sid] = hic_sensor.fire_time or t
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (HIC = {hic_sensor.current_hic:10.4E})")
            elif kind == "NIC":
                nic_sensor, spring_id = defn[1], defn[2]
                dt = getattr(self.model, "dt", 0.0)
                if dt <= 0.0:
                    dt = max(t - getattr(self, "_t_prev", 0.0), 1e-6)
                spring_fm = getattr(self.model, "spring_fm", {}) or {}
                fz, my = 0.0, 0.0
                if spring_id in spring_fm:
                    fm_entry = spring_fm[spring_id]
                    if isinstance(fm_entry, (tuple, list)) and len(fm_entry) >= 2:
                        fz, my = float(fm_entry[0]), float(fm_entry[1])
                    elif isinstance(fm_entry, (int, float)):
                        fz = float(fm_entry)
                fired = nic_sensor.update(t, dt, fz, my)
                if fired:
                    self.fire_time[sid] = nic_sensor.fire_time or t
                    self.status[sid] = True
                    log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                             f"{self.fire_time[sid]:12.5E} (NIJ = {nic_sensor.current_nij:10.4E})")



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
                    cond = not self.active(is1)
                elif kind == "AND":
                    is1, is2, tdelay = defn[1], defn[2], defn[3]
                    cond = self.active(is1) and self.active(is2)
                elif kind == "OR":
                    is1, is2, tdelay = defn[1], defn[2], defn[3]
                    cond = self.active(is1) or self.active(is2)
                else:
                    cond = False

                if cond:
                    if sid not in self.crit_time:
                        self.crit_time[sid] = t
                    new_active = t >= (self.crit_time[sid] + (tdelay or 0.0))
                else:
                    self.crit_time.pop(sid, None)
                    new_active = False

                if new_active != old_active:
                    changed = True
                    self.status[sid] = new_active
                    if new_active:
                        self.fire_time[sid] = self.crit_time[sid] + (tdelay or 0.0)
                        log.info(f" -- /SENSOR/{sid} ACTIVATED AT TIME "
                                 f"{t:12.5E} (LOGICAL {kind})")
                    else:
                        self.fire_time.pop(sid, None)
                        log.info(f" -- /SENSOR/{sid} DEACTIVATED AT TIME "
                                 f"{t:12.5E} (LOGICAL {kind})")

        self._t_prev = t
        if hasattr(self.model, "v") and self.model.v is not None:
            self._v_prev = self.model.v.copy()

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

