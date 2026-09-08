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
                        self._other_req.append((f"{th.kind[:2]}{oid}_{var}", th.kind, oid, var))
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
                safe_conn = np.maximum(conn[mask], 0)
                ve = model.v[safe_conn]
                v2 = np.einsum("nib,nib->n", ve, ve) / conn.shape[1]
                ke_trans = 0.5 * (group.state["mass"][mask] * v2).sum()
                ke_rot = 0.0
                if getattr(model, "vr", None) is not None:
                    vre = model.vr[safe_conn]
                    if "dt_iner" in group.state:
                        # Shells / beams: dt_iner is the per-node rotational
                        # inertia share (cbilan.F: IN25 * VA2 * HALF).
                        # Sum |omega_i|^2 over nodes, multiply by dt_iner.
                        vr2_sum = np.einsum("nib,nib->n", vre, vre)
                        ke_rot = 0.5 * (group.state["dt_iner"][mask] * vr2_sum).sum()
                    elif "inertia" in group.state:
                        # Springs / mock groups: element-total inertia.
                        vr2_avg = np.einsum("nib,nib->n", vre, vre) / conn.shape[1]
                        ke_rot = 0.5 * (group.state["inertia"][mask] * vr2_avg).sum()
                val += float(ke_trans + ke_rot)
        return val

    def _other_value(self, kind: str, oid: int, var: str) -> float:
        model = self.model
        var_upper = var.upper()
        comp = {"X": 0, "Y": 1, "Z": 2}.get(var_upper[-1], 0)

        if kind == "RBODY":
            rbs = getattr(model, "rigid_bodies", None)
            rb = rbs.get(oid) if rbs is not None else None
            if rb is not None:
                if var_upper.startswith("D"):
                    return float(rb.x_cg[comp] - rb.x_cg0[comp]) if hasattr(rb, "x_cg0") else 0.0
                elif var_upper.startswith("V"):
                    return float(rb.v_cg[comp]) if hasattr(rb, "v_cg") else 0.0
                elif var_upper.startswith("F"):
                    return float(rb.f_res[comp]) if hasattr(rb, "f_res") else 0.0
                elif var_upper.startswith("M"):
                    return float(rb.m_res[comp]) if hasattr(rb, "m_res") else 0.0

        elif kind in ("SPRING", "SPRI"):
            g = getattr(model, "springs", None)
            if g is not None and oid in g.ids:
                row = np.where(g.ids == oid)[0]
                if len(row):
                    r = row[0]
                    st = g.state
                    if var_upper in ("F", "FORCE"):
                        if "fres" in st:
                            f = st["fres"][r]
                            return float(np.linalg.norm(f) if hasattr(f, "__len__") else f)
                        return float(st.get("force", np.zeros(len(g.ids)))[r])
                    elif var_upper in ("FX", "FY", "FZ"):
                        c = {"FX": 0, "FY": 1, "FZ": 2}[var_upper]
                        if "fres" in st and st["fres"].ndim > 1 and st["fres"].shape[1] > c:
                            return float(st["fres"][r, c])
                        if "force" in st:
                            n1, n2 = g.conn[r, 0], g.conn[r, 1]
                            dx = model.x[n2] - model.x[n1]
                            L = np.linalg.norm(dx)
                            if L > 1e-20:
                                return float(st["force"][r] * (dx[c] / L))
                        return 0.0
                    elif var_upper in ("D", "DISP"):
                        if "disp" in st:
                            return float(st["disp"][r])
                        if "L" in st and "L0" in st:
                            return float(st["L"][r] - st["L0"][r])
                        n1, n2 = g.conn[r, 0], g.conn[r, 1]
                        L = np.linalg.norm(model.x[n2] - model.x[n1])
                        L0 = np.linalg.norm(model.x0[n2] - model.x0[n1])
                        return float(L - L0)
                    elif var_upper in ("DX", "DY", "DZ"):
                        c = {"DX": 0, "DY": 1, "DZ": 2}[var_upper]
                        n1, n2 = g.conn[r, 0], g.conn[r, 1]
                        dx = (model.x[n2, c] - model.x0[n2, c]) - (model.x[n1, c] - model.x0[n1, c])
                        return float(dx)
                    elif var_upper in ("E", "IE", "ENERGY"):
                        return float(st["eint"][r]) if "eint" in st else 0.0

        elif kind in ("SHEL", "SHELL", "BRIC", "BRICK", "SH3N", "TETR"):
            for attr in ("shells", "shells_qbat", "shells_qeph", "bricks", "bricks_heph", "tetras", "sh3n", "sh3n_dkt18", "bric20s", "shel16s", "tetra10s", "quads"):
                g = getattr(model, attr, None)
                if g is not None and oid in g.ids:
                    row = np.where(g.ids == oid)[0]
                    if len(row):
                        r = row[0]
                        st = g.state
                        if var_upper in ("IE", "EINT", "ENERGY"):
                            return float(st["eint"][r]) if "eint" in st else 0.0
                        elif var_upper in ("EPSP", "PLASTIC_STRAIN"):
                            ep = st.get("epsp")
                            if ep is not None:
                                return float(ep[r].max() if ep[r].ndim > 0 else ep[r])
                        elif var_upper in ("VM", "VONM", "VON_MISES"):
                            from .anim_vtk import _von_mises
                            return float(_von_mises(attr, g)[r])
                        elif var_upper in ("SIGXX", "SIGYY", "SIGZZ", "SIGXY", "SIGYZ", "SIGZX"):
                            idx_map = {"SIGXX": 0, "SIGYY": 1, "SIGZZ": 2, "SIGXY": 3, "SIGYZ": 4, "SIGZX": 5}
                            sig = st.get("sig")
                            if sig is not None:
                                s_elem = sig[r]
                                if s_elem.ndim > 1:
                                    s_elem = s_elem.mean(axis=0)
                                return float(s_elem[idx_map[var_upper]])
                        elif var_upper in ("P", "PRESSURE"):
                            sig = st.get("sig")
                            if sig is not None:
                                s_elem = sig[r]
                                if s_elem.ndim > 1:
                                    s_elem = s_elem.mean(axis=0)
                                return float(-np.mean(s_elem[:3]))
        return 0.0

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
            elif var[0] == "A":
                if hasattr(model, "a") and model.a is not None:
                    row.append(model.a[idx, comp])
                else:
                    m = model.mass[idx]
                    fext_val = model.fext[idx, comp] if hasattr(model, "fext") and model.fext is not None else 0.0
                    fint_val = model.fint[idx, comp] if hasattr(model, "fint") and model.fint is not None else 0.0
                    row.append((fext_val - fint_val) / m if m > 0 else 0.0)
            elif var[0] == "F":
                row.append(model.fint[idx, comp] if hasattr(model, "fint") and model.fint is not None else 0.0)
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
        for _, kind, oid, var in self._other_req:
            row.append(self._other_value(kind, oid, var))
        self._fh.write(",".join(f"{x:.9E}" for x in row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
