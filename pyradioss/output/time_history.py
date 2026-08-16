"""
Time-history output (the T01 file, /TFILE + /TH).

Fortran origin: ``engine/source/output/th/`` — the original writes a
binary T01 read by post-processors. The port writes CSV with a commented
header: column 1 is time, then the global variables, then one column per
requested /TH variable, named ``<kind><id>_<VAR>``.

Global columns (always present, matching the original's global T-file
variables): IE internal energy, KE kinetic energy (translational),
HE hourglass energy, CE contact energy, EN numerical dissipation of the
element dampers (M6, see the engine's step-6c ledger), DE /DAMP
dissipation (M6), EW external work, ERR energy error %, MASS (grows
under /DT/NODA/CST mass scaling), MOMX/Y/Z momentum components.

Per-node variables: DX DY DZ (displacement), VX VY VZ (velocity),
AX AY AZ (acceleration is not stored — approximated by force/mass).
Per-part variables: IE (internal energy of the part's elements),
KE (kinetic energy of the part's element masses, computed from their
nodes' velocities).
Per-section variables (M5, /TH/SECT): FX FY FZ MX MY MZ — the section
force/moment resultants computed by pyradioss/engine/sections.py.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..model.model import Model


class TimeHistory:
    def __init__(self, path: str, model: Model, log):
        self.path = path
        self.model = model
        self._fh = open(path, "w")
        self._cols: List[str] = [
            "TIME", "IE", "KE", "HE", "CE", "EN", "DE", "EW", "ERR%",
            "MASS", "MOMX", "MOMY", "MOMZ"]

        # resolve /TH requests once (Starter checked the ids)
        self._node_req = []   # (label, node_idx, var)
        self._part_req = []   # (label, part_id, var)
        self._sect_req = []   # (label, sect_id, var)
        self._other_req = []  # (label, id, var)
        for th in model.th_requests:
            for oid in th.ids:
                for var in th.variables:
                    if th.kind == "NODE":
                        self._node_req.append(
                            (f"N{oid}_{var}", model.node_index(oid), var))
                    elif th.kind == "SECT":
                        self._sect_req.append((f"S{oid}_{var}", oid, var))
                    elif th.kind == "PART":
                        self._part_req.append((f"P{oid}_{var}", oid, var))
                    else:
                        self._other_req.append((f"{th.kind[:2]}{oid}_{var}", oid, var))
        self._cols += [r[0] for r in self._node_req]
        self._cols += [r[0] for r in self._part_req]
        self._cols += [r[0] for r in self._sect_req]
        self._cols += [r[0] for r in self._other_req]
        # M39 output-path: the nodal DISPLACEMENT field (model.x - model.x0)
        # is only read by a /TH/NODE 'D*' request. Building the whole (N, 3)
        # difference every write when no such request exists is wasted work
        # — a 72k-node global-only T-file (the gasket cliff deck) allocated
        # and subtracted 1.7 MB per row for nothing. Resolve the need once.
        self._need_disp = any(var[0] == "D" for _, _, var in self._node_req)
        self._fh.write("# pyradioss time history (T01 equivalent)\n")
        self._fh.write(",".join(self._cols) + "\n")

    # ------------------------------------------------------------------
    def _part_value(self, pid: int, var: str) -> float:
        """IE/KE of one part, summed over its element groups."""
        model = self.model
        val = 0.0
        for _, group in model.element_groups():
            mask = group.state["part_ids"] == pid
            if not np.any(mask):
                continue
            if var == "IE":
                val += float(group.state["eint"][mask].sum())
            elif var == "KE":
                # kinetic energy of the element masses: 1/2 m_e <v^2>_nodes
                # (beams store a reduced 'mass_conn' — their 3rd node is
                # orientation only and carries no mass)
                conn = group.state.get("mass_conn", group.conn)
                ve = model.v[conn[mask]]
                v2 = np.einsum("nib,nib->n", ve, ve) / conn.shape[1]
                val += float(0.5 * (group.state["mass"][mask] * v2).sum())
        return val

    def write(self, t, energies, mass, momentum, sect_values=None) -> None:
        """``sect_values``: {sect_id: (F (3,), M (3,))} from
        SectionForces.compute — required only when /TH/SECT was asked."""
        model = self.model
        row = [t, energies["IE"], energies["KE"], energies["HE"],
               energies["CE"], energies["EN"], energies["DE"],
               energies["EW"], energies["ERR"], mass,
               momentum[0], momentum[1], momentum[2]]
        disp = (model.x - model.x0) if self._need_disp else None
        for _, idx, var in self._node_req:
            comp = {"X": 0, "Y": 1, "Z": 2}[var[-1]]
            if var[0] == "D":
                row.append(disp[idx, comp])
            elif var[0] == "V":
                row.append(model.v[idx, comp])
            else:
                row.append(0.0)
        for _, pid, var in self._part_req:
            row.append(self._part_value(pid, var))
        for _, sid, var in self._sect_req:
            comp = {"X": 0, "Y": 1, "Z": 2}[var[-1]]
            if sect_values is None or sid not in sect_values:
                row.append(0.0)
            else:
                F, M = sect_values[sid]
                row.append(F[comp] if var[0] == "F" else M[comp])
        for _ in self._other_req:
            row.append(0.0)
        self._fh.write(",".join(f"{x:.9E}" for x in row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
