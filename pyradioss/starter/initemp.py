"""Initial Temperature Field Generator (/INITEMP).

Ported from OpenRadioss Fortran:
- ``starter/source/initial_conditions/thermic/hm_read_initemp.F`` (SUBROUTINE HM_READ_INITEMP)
- ``starter/source/materials/therm/initemp_shell.F90`` (SUBROUTINE INITEMP_SHELL)
- ``starter/source/elements/solid/solide/sinit3.F`` (lines 272-283, Solid Element Temp Init)
- ``hm_cfg_files/config/CFG/radioss110/LOADS/initemp.cfg`` (/INITEMP Keyword Specifications)

Formulation & Supported Initial Temperature Fields
--------------------------------------------------
1. UNIFORM TEMPERATURE (distribution=0, fld_type=0):
   Uniform temperature T0 applied across a target node group (/GRNOD), part (/PART),
   element set, or all model nodes:
       T_i = T0

2. 3D LINEAR SPATIAL GRADIENT FIELD:
   Spatial temperature gradient vector G = (Gx, Gy, Gz) referenced to origin x0:
       T(x) = T0 + Gx * (x - x0) + Gy * (y - y0) + Gz * (z - z0)
            = T0 + G . (x - x0)

3. INDIVIDUAL NODAL TEMPERATURE TABLE (distribution=1, fld_type=1):
   Base temperature T0 applied to group/part, overridden by per-node table:
       T_node[nid] = T0_i

4. SHELL THROUGH-THICKNESS GRADIENT (INITEMP_SHELL):
   Through-thickness temperature distribution across shell integration points / layers:
   For normalized thickness coordinate xi in [-1, 1] (xi=-1 bottom, xi=0 mid, xi=+1 top):
       T(xi) = T_mid + xi * (T_top - T_bot) / 2
   If T_mid is explicitly prescribed and not equidistant:
       T(xi) = T_mid + xi * (T_mid - T_bot) for xi <= 0
       T(xi) = T_mid + xi * (T_top - T_mid) for xi > 0

5. ELEMENT INITIALIZATION (SINIT3 & INITEMP_SHELL):
   Solid and shell element mean temperatures are computed from nodal temperatures:
       T_elem = (1 / N_nod) * sum_{j=1}^{N_nod} T_node[j]
   - Hexa8: 1/8 sum of 8 vertex nodes (sinit3.F:276-280)
   - Tetra4: 1/4 sum of 4 vertex nodes
   - Shell: 1/4 (quad) or 1/3 (tria) sum of corner nodes (initemp_shell.F90:82-86)
   - Shell integration points: distributed through-thickness layers.

6. THERMAL STRAIN & THERMAL SOFTENING COUPLING:
   - Linear thermal expansion strain:
       Delta eps_th = alpha_th * (T - T_ref) * I
   - Johnson-Cook thermal softening (LAW02, LAW04):
       T* = clip((T - T_room) / (T_melt - T_room), 0.0, 1.0)
       C_T = 1.0 - (T*)^m
       sigma_y(T) = sigma_y0 * C_T
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.messages import MessageLog


def _get_shell_layer_coords(nip: int, zw: Optional[Any] = None) -> np.ndarray:
    """Compute normalized through-thickness layer coordinates xi_k in [-1, 1].

    Parameters
    ----------
    nip : int
        Number of integration points / layers through shell thickness.
    zw : optional
        Gauss point data structure if present.

    Returns
    -------
    np.ndarray
        Array of shape (nip,) with coordinates in [-1, 1].
    """
    if nip <= 1:
        return np.zeros(1, dtype=np.float64)
    if zw and len(zw) > 0 and hasattr(zw[0], "__len__") and len(zw[0][0]) == nip:
        # Gauss points relative to half-thickness scaled to [-1, 1]
        gp = np.asarray(zw[0][0], dtype=np.float64) * 2.0
        return gp
    # Standard Gauss-Legendre quadrature points in [-1, 1]
    gp, _ = np.polynomial.legendre.leggauss(nip)
    return np.asarray(gp, dtype=np.float64)


@dataclass
class InitempRecord:
    """Initial temperature definition record (/INITEMP).

    Fortran references:
      - ``hm_read_initemp.F`` lines 42-205 (HM_READ_INITEMP)
      - ``initemp_shell.F90`` lines 50-103 (INITEMP_SHELL)
      - ``sinit3.F`` lines 272-283 (SINIT3)
      - ``initemp.cfg`` (CFG card formats)

    Attributes
    ----------
    id : int
        Identifier of the /INITEMP card.
    title : str
        Description or name of the temperature field.
    t0 : float
        Base uniform temperature (default: 293.15 K).
    grnod_id : int, optional
        Target node group ID (/GRNOD).
    part_id : int, optional
        Target part ID (/PART).
    element_set_id : int, optional
        Target element set / subset ID.
    node_ids : Sequence[int], optional
        Explicit list of node IDs or indices.
    gradient : Sequence[float], optional
        3D spatial linear gradient vector G = (Gx, Gy, Gz) in [K/m].
    x0 : Sequence[float], optional
        Reference point x0 = (x0, y0, z0) for spatial gradient [m].
    t_top : float, optional
        Outer surface temperature for shells (xi = +1).
    t_mid : float, optional
        Mid-surface temperature for shells (xi = 0).
    t_bot : float, optional
        Inner surface temperature for shells (xi = -1).
    layer_temperatures : Sequence[float], optional
        Explicit array of layer temperatures through shell thickness.
    nodal_temps : Dict[int, float]
        Per-node temperature overrides {node_id: temperature}.
    fld_type : int
        0 = uniform/gradient on group or part, 1 = nodal table.
    additive : bool
        If True, adds temperature to existing field rather than overwriting.
    """

    id: int = 1
    title: str = ""
    t0: float = 293.15
    grnod_id: Optional[int] = None
    part_id: Optional[int] = None
    element_set_id: Optional[int] = None
    node_ids: Optional[Sequence[int]] = None
    gradient: Optional[Sequence[float]] = None
    x0: Optional[Sequence[float]] = None
    t_top: Optional[float] = None
    t_mid: Optional[float] = None
    t_bot: Optional[float] = None
    layer_temperatures: Optional[Sequence[float]] = None
    nodal_temps: Dict[int, float] = field(default_factory=dict)
    fld_type: int = 0
    additive: bool = False

    @property
    def node_temperatures(self) -> Dict[int, float]:
        """Alias for nodal_temps dictionary."""
        return self.nodal_temps

    @node_temperatures.setter
    def node_temperatures(self, val: Dict[int, float]) -> None:
        self.nodal_temps = dict(val)

    def has_spatial_gradient(self) -> bool:
        """Check whether a non-zero 3D spatial gradient is defined."""
        if self.gradient is None:
            return False
        g = np.asarray(self.gradient, dtype=np.float64)
        return bool(np.linalg.norm(g) > 1e-15)

    def has_shell_gradient(self) -> bool:
        """Check whether a shell through-thickness gradient is defined."""
        if self.layer_temperatures is not None and len(self.layer_temperatures) > 0:
            return True
        return (
            self.t_top is not None
            or self.t_bot is not None
            or self.t_mid is not None
        )

    def compute_nodal_temperature(self, coords: np.ndarray) -> np.ndarray:
        """Compute initial temperatures for given nodal coordinates.

        Formula:
            T(x) = T0 + G . (x - x0)

        Parameters
        ----------
        coords : np.ndarray
            Nodal coordinates of shape (N, 3) or (3,).

        Returns
        -------
        np.ndarray
            Array of temperatures of shape (N,).
        """
        pts = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        n_pts = len(pts)
        if n_pts == 0:
            return np.zeros(0, dtype=np.float64)

        t_arr = np.full(n_pts, float(self.t0), dtype=np.float64)

        if self.has_spatial_gradient():
            g = np.asarray(self.gradient, dtype=np.float64)[:3]
            ref_x0 = (
                np.asarray(self.x0, dtype=np.float64)[:3]
                if self.x0 is not None
                else np.zeros(3, dtype=np.float64)
            )
            dx = pts - ref_x0
            t_arr += np.dot(dx, g)

        return t_arr

    def compute_shell_layer_temperatures(
        self,
        nip: int,
        coords_xi: Optional[np.ndarray] = None,
        t_base: Optional[float] = None,
    ) -> np.ndarray:
        """Compute temperatures through shell thickness for nip integration points.

        Parameters
        ----------
        nip : int
            Number of integration points through thickness.
        coords_xi : np.ndarray, optional
            Normalized layer coordinates in [-1, 1]. Defaults to Gauss points.
        t_base : float, optional
            Reference base/mid temperature (e.g. from in-plane element mean).

        Returns
        -------
        np.ndarray
            Layer temperatures array of shape (nip,).
        """
        if nip <= 1:
            val = (
                self.t_mid
                if self.t_mid is not None
                else (t_base if t_base is not None else self.t0)
            )
            return np.array([val], dtype=np.float64)

        if self.layer_temperatures is not None and len(self.layer_temperatures) == nip:
            return np.asarray(self.layer_temperatures, dtype=np.float64)

        if coords_xi is None:
            xi = _get_shell_layer_coords(nip)
        else:
            xi = np.asarray(coords_xi, dtype=np.float64)

        # Resolve mid, top, bot
        if self.t_mid is not None:
            mid = float(self.t_mid)
        elif t_base is not None:
            mid = float(t_base)
        elif self.t_top is not None and self.t_bot is not None:
            mid = 0.5 * (float(self.t_top) + float(self.t_bot))
        else:
            mid = float(self.t0)

        top = float(self.t_top) if self.t_top is not None else mid
        bot = float(self.t_bot) if self.t_bot is not None else mid

        # Through-thickness interpolation:
        # Linear between bot (-1) and top (+1) if mid is midpoint
        if abs(mid - 0.5 * (top + bot)) < 1e-10:
            # T(xi) = 0.5*(top + bot) + 0.5*xi*(top - bot)
            t_layers = 0.5 * (top + bot) + 0.5 * xi * (top - bot)
        else:
            # Piecewise linear: bot -> mid for xi <= 0, mid -> top for xi > 0
            t_layers = np.where(
                xi <= 0.0,
                mid + xi * (mid - bot),
                mid + xi * (top - mid),
            )

        return np.asarray(t_layers, dtype=np.float64)


# Convenient alias
InitempParams = InitempRecord


def _resolve_target_nodes(record: Any, model: Any) -> np.ndarray:
    """Resolve 0-based node indices targeted by an InitempRecord or InitialTemperature.

    Parameters
    ----------
    record : InitempRecord or InitialTemperature
        Record defining targets.
    model : Model
        Finite element model.

    Returns
    -------
    np.ndarray
        Array of targeted 0-based node indices.
    """
    numnod = getattr(model, "numnod", 0)
    if numnod == 0 and hasattr(model, "x0"):
        numnod = len(model.x0)
    elif numnod == 0 and hasattr(model, "x"):
        numnod = len(model.x)
    elif numnod == 0 and hasattr(model, "node_ids"):
        numnod = len(model.node_ids)

    def _node_to_idx(nid: int) -> Optional[int]:
        if hasattr(model, "node_index") and callable(model.node_index):
            try:
                return int(model.node_index(nid))
            except (KeyError, IndexError, TypeError):
                pass
        if hasattr(model, "_id2idx") and isinstance(model._id2idx, dict):
            idx = model._id2idx.get(nid)
            if idx is not None:
                return int(idx)
        if hasattr(model, "node_ids") and model.node_ids is not None:
            matches = np.where(model.node_ids == nid)[0]
            if len(matches) > 0:
                return int(matches[0])
        if 0 <= nid < numnod:
            return int(nid)
        return None

    # 1. Explicit node IDs
    node_ids = getattr(record, "node_ids", None)
    if node_ids is not None and len(node_ids) > 0:
        indices = []
        for nid in node_ids:
            idx = _node_to_idx(int(nid))
            if idx is not None and 0 <= idx < numnod:
                indices.append(idx)
        return np.array(sorted(set(indices)), dtype=np.int64)

    # 2. Node Group (/GRNOD)
    grnod_id = getattr(record, "grnod_id", None)
    if grnod_id is not None and grnod_id != 0:
        grp = getattr(model, "node_groups", {}).get(grnod_id)
        if grp is not None:
            if getattr(grp, "node_idx", None) is not None:
                return np.asarray(grp.node_idx, dtype=np.int64)
            if getattr(grp, "node_ids", None) is not None:
                indices = []
                for nid in grp.node_ids:
                    idx = _node_to_idx(int(nid))
                    if idx is not None and 0 <= idx < numnod:
                        indices.append(idx)
                return np.array(sorted(set(indices)), dtype=np.int64)

    # 3. Part ID (/PART)
    part_id = getattr(record, "part_id", None)
    if part_id is not None:
        part_nodes = set()
        if hasattr(model, "element_groups") and callable(model.element_groups):
            for _, grp in model.element_groups():
                p_ids = grp.state.get("part_ids") if hasattr(grp, "state") else None
                if p_ids is not None:
                    mask = (p_ids == part_id)
                    if np.any(mask):
                        conn = grp.state.get("mass_conn", getattr(grp, "conn", None))
                        if conn is not None:
                            sub_conn = conn[mask]
                            valid = sub_conn[(sub_conn >= 0) & (sub_conn < numnod)]
                            part_nodes.update(valid.tolist())
        if part_nodes:
            return np.array(sorted(part_nodes), dtype=np.int64)

    # 4. Element Set ID
    elem_set_id = getattr(record, "element_set_id", getattr(record, "group_id", None))
    if elem_set_id is not None and hasattr(model, "egroups"):
        e_set = None
        for g_type in ("BRICK", "SHELL", "SOLID", "PART"):
            if g_type in model.egroups and elem_set_id in model.egroups[g_type]:
                e_set = model.egroups[g_type][elem_set_id]
                break
        if e_set is not None:
            members = set(getattr(e_set, "members", []))
            set_nodes = set()
            if hasattr(model, "element_groups") and callable(model.element_groups):
                for _, grp in model.element_groups():
                    ids = getattr(grp, "ids", None)
                    if ids is not None:
                        mask = np.isin(ids, list(members))
                        if np.any(mask):
                            conn = grp.state.get("mass_conn", getattr(grp, "conn", None))
                            if conn is not None:
                                sub_conn = conn[mask]
                                valid = sub_conn[(sub_conn >= 0) & (sub_conn < numnod)]
                                set_nodes.update(valid.tolist())
            if set_nodes:
                return np.array(sorted(set_nodes), dtype=np.int64)

    # 5. Individual Nodal Table only (fld_type == 1 without explicit group)
    nodal_temps = getattr(record, "nodal_temps", getattr(record, "node_temperatures", {}))
    if nodal_temps and (grnod_id is None or grnod_id == 0) and part_id is None:
        indices = []
        for nid in nodal_temps.keys():
            idx = _node_to_idx(int(nid))
            if idx is not None and 0 <= idx < numnod:
                indices.append(idx)
        if indices:
            return np.array(sorted(set(indices)), dtype=np.int64)

    # 6. Default: All nodes
    return np.arange(numnod, dtype=np.int64)


def apply_initemp(
    model: Any,
    initemp_list: Optional[Union[InitempRecord, Sequence[Any]]] = None,
    default_t0: float = 293.15,
    apply_to_elements: bool = True,
    log: Optional[MessageLog] = None,
) -> np.ndarray:
    """Initialize model nodal and element temperature fields from /INITEMP cards.

    Fortran origins:
      - ``starter/source/initial_conditions/thermic/hm_read_initemp.F``
      - ``starter/source/materials/therm/initemp_shell.F90``
      - ``starter/source/elements/solid/solide/sinit3.F``

    Parameters
    ----------
    model : Model
        Finite element model containing nodes, coordinates, and element groups.
    initemp_list : InitempRecord or sequence of records, optional
        List of temperature records to apply. If None, inspects ``model.initemp_records``,
        ``model.initemp``, or ``model.initemps``.
    default_t0 : float, optional
        Default base temperature if none specified (default: 293.15 K).
    apply_to_elements : bool, optional
        Whether to map nodal temperatures to element groups (group.state["temp"]).
    log : MessageLog, optional
        Logger instance for progress and diagnostic messages.

    Returns
    -------
    np.ndarray
        Array of nodal temperatures of shape (numnod,).
    """
    coords = getattr(model, "x0", None)
    if coords is None:
        coords = getattr(model, "x", None)

    numnod = getattr(model, "numnod", 0)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        if numnod == 0:
            numnod = len(coords)
    elif numnod > 0:
        coords = np.zeros((numnod, 3), dtype=np.float64)
    elif hasattr(model, "node_ids") and model.node_ids is not None:
        numnod = len(model.node_ids)
        coords = np.zeros((numnod, 3), dtype=np.float64)
    else:
        numnod = 0
        coords = np.zeros((0, 3), dtype=np.float64)

    # Initialize model nodal temperature array
    if hasattr(model, "temperature") and model.temperature is not None and len(model.temperature) == numnod:
        temperature = np.copy(model.temperature)
    elif hasattr(model, "temperatures") and model.temperatures is not None and len(model.temperatures) == numnod:
        temperature = np.copy(model.temperatures)
    else:
        temperature = np.full(numnod, default_t0, dtype=np.float64)

    # Helper node index lookup
    def _node_to_idx(nid: int) -> Optional[int]:
        if hasattr(model, "node_index") and callable(model.node_index):
            try:
                return int(model.node_index(nid))
            except (KeyError, IndexError, TypeError):
                pass
        if hasattr(model, "_id2idx") and isinstance(model._id2idx, dict):
            idx = model._id2idx.get(nid)
            if idx is not None:
                return int(idx)
        if hasattr(model, "node_ids") and model.node_ids is not None:
            matches = np.where(model.node_ids == nid)[0]
            if len(matches) > 0:
                return int(matches[0])
        if 0 <= nid < numnod:
            return int(nid)
        return None

    # Consolidate records
    records: List[Any] = []
    if initemp_list is not None:
        if isinstance(initemp_list, (InitempRecord, tuple, list)):
            if isinstance(initemp_list, InitempRecord):
                records = [initemp_list]
            else:
                records = list(initemp_list)
        else:
            records = [initemp_list]
    else:
        raw = getattr(model, "initemp_records", getattr(model, "initemp", []))
        if isinstance(raw, (list, tuple)):
            records = list(raw)
        elif isinstance(raw, dict):
            records = list(raw.values())
        if hasattr(model, "initemps") and isinstance(model.initemps, dict):
            for rec in model.initemps.values():
                if rec not in records:
                    records.append(rec)

    # Track shell records that define through-thickness gradients
    shell_gradient_records: List[InitempRecord] = []

    # Apply each record
    applied_count = 0
    for rec in records:
        # Convert InitialTemperature to InitempRecord if needed
        if not isinstance(rec, InitempRecord):
            it_id = getattr(rec, "id", 1)
            it_t0 = getattr(rec, "t0", default_t0)
            it_grnod = getattr(rec, "grnod_id", None)
            it_part = getattr(rec, "part_id", None)
            it_type = getattr(rec, "fld_type", 0)
            it_nodal = getattr(rec, "nodal_temps", getattr(rec, "node_temperatures", {}))
            it_title = getattr(rec, "title", "")
            it_grad = getattr(rec, "gradient", None)
            it_x0 = getattr(rec, "x0", None)
            it_top = getattr(rec, "t_top", None)
            it_mid = getattr(rec, "t_mid", None)
            it_bot = getattr(rec, "t_bot", None)
            it_layers = getattr(rec, "layer_temperatures", None)
            it_add = getattr(rec, "additive", False)

            initemp_rec = InitempRecord(
                id=it_id,
                title=it_title,
                t0=it_t0,
                grnod_id=it_grnod,
                part_id=it_part,
                fld_type=it_type,
                nodal_temps=it_nodal,
                gradient=it_grad,
                x0=it_x0,
                t_top=it_top,
                t_mid=it_mid,
                t_bot=it_bot,
                layer_temperatures=it_layers,
                additive=it_add,
            )
        else:
            initemp_rec = rec

        if initemp_rec.has_shell_gradient():
            shell_gradient_records.append(initemp_rec)

        target_nodes = _resolve_target_nodes(initemp_rec, model)
        if len(target_nodes) == 0:
            continue

        valid_mask = (target_nodes >= 0) & (target_nodes < numnod)
        valid_indices = target_nodes[valid_mask]
        if len(valid_indices) == 0:
            continue

        # Compute field values
        target_coords = coords[valid_indices]
        t_computed = initemp_rec.compute_nodal_temperature(target_coords)

        if initemp_rec.additive:
            temperature[valid_indices] += t_computed
        else:
            temperature[valid_indices] = t_computed

        # Override individual node values from table (fld_type=1)
        if initemp_rec.nodal_temps:
            for nid, t_val in initemp_rec.nodal_temps.items():
                idx = _node_to_idx(int(nid))
                if idx is not None and 0 <= idx < numnod:
                    if initemp_rec.additive:
                        temperature[idx] += float(t_val)
                    else:
                        temperature[idx] = float(t_val)

        applied_count += len(valid_indices)

    # Attach to model
    model.temperature = temperature
    model.temperatures = temperature

    # -------------------------------------------------------------------------
    # Map to Element Groups (sinit3.F, initemp_shell.F90)
    # -------------------------------------------------------------------------
    if apply_to_elements and hasattr(model, "element_groups") and callable(model.element_groups):
        for gname, group in model.element_groups():
            if not hasattr(group, "state") or not isinstance(group.state, dict):
                continue
            conn = group.state.get("mass_conn", getattr(group, "conn", None))
            if conn is None or len(conn) == 0:
                continue

            nel = len(conn)
            valid_mask = (conn >= 0) & (conn < numnod)
            safe_conn = np.where(valid_mask, conn, 0)
            node_t = temperature[safe_conn]
            sum_t = np.sum(np.where(valid_mask, node_t, 0.0), axis=1)
            counts = np.maximum(np.sum(valid_mask, axis=1), 1)
            tempel = sum_t / counts  # (nel,) mean element temperature

            is_shell = any(k in gname.upper() for k in ("SHELL", "SH3N", "QUAD", "TRIA", "TSHELL"))

            if is_shell:
                # Determine number of layers / integration points
                nip = 1
                sig = group.state.get("sig")
                if sig is not None and sig.ndim >= 2:
                    nip = sig.shape[1]
                elif "nlay" in group.state:
                    nip = int(group.state["nlay"])

                # Check if shell through-thickness gradient applies to this group
                p_ids = group.state.get("part_ids")
                matched_shell_rec: Optional[InitempRecord] = None
                for s_rec in shell_gradient_records:
                    if s_rec.part_id is not None and p_ids is not None:
                        if np.any(p_ids == s_rec.part_id):
                            matched_shell_rec = s_rec
                            break
                    elif s_rec.grnod_id is None and s_rec.part_id is None:
                        matched_shell_rec = s_rec
                        break

                if matched_shell_rec is not None and nip > 1:
                    xi = _get_shell_layer_coords(nip, group.state.get("zw"))
                    # Distribute through thickness
                    t_layers = np.zeros((nel, nip), dtype=np.float64)
                    for i in range(nel):
                        t_layers[i, :] = matched_shell_rec.compute_shell_layer_temperatures(
                            nip, coords_xi=xi, t_base=tempel[i]
                        )
                    group.state["temp"] = t_layers
                else:
                    if nip > 1:
                        group.state["temp"] = np.repeat(tempel[:, None], nip, axis=1)
                    else:
                        group.state["temp"] = tempel

                group.state["temp_mean"] = tempel

                # Also update mat_extra["temp"] if present
                if "mat_extra" in group.state and isinstance(group.state["mat_extra"], dict):
                    if "temp" in group.state["mat_extra"]:
                        cur_extra = group.state["mat_extra"]["temp"]
                        if isinstance(cur_extra, np.ndarray) and cur_extra.shape == group.state["temp"].shape:
                            cur_extra[:] = group.state["temp"]
                        else:
                            group.state["mat_extra"]["temp"] = group.state["temp"].copy()
            else:
                # Solid / Brick / Tetra / Beam / Truss
                group.state["temp"] = tempel
                if "mat_extra" in group.state and isinstance(group.state["mat_extra"], dict):
                    if "temp" in group.state["mat_extra"]:
                        cur_extra = group.state["mat_extra"]["temp"]
                        if isinstance(cur_extra, np.ndarray) and cur_extra.shape == tempel.shape:
                            cur_extra[:] = tempel
                        else:
                            group.state["mat_extra"]["temp"] = tempel.copy()

    if log is not None and applied_count > 0:
        log.info(f" /INITEMP: initialized temperatures for {applied_count} node mapping(s)")

    return temperature


# ----------------------------------------------------------------------------
# Thermal Strain and Softening Coupling
# ----------------------------------------------------------------------------

def compute_thermal_strain(
    temp: Union[float, np.ndarray],
    t_ref: float = 293.15,
    alpha: float = 0.0,
    elem_type: str = "SOLID",
) -> np.ndarray:
    """Compute thermal expansion strain tensor Delta eps_th = alpha * (T - T_ref) * I.

    Parameters
    ----------
    temp : float or np.ndarray
        Current temperature or temperature array.
    t_ref : float, optional
        Reference / stress-free temperature (default: 293.15 K).
    alpha : float, optional
        Thermal expansion coefficient [1/K].
    elem_type : str, optional
        "SOLID" (6-component Voigt) or "SHELL" / "PLANE_STRESS" (3 or 6 components).

    Returns
    -------
    np.ndarray
        Thermal strain tensor array.
    """
    t_arr = np.asarray(temp, dtype=np.float64)
    dt = t_arr - float(t_ref)
    eth = float(alpha) * dt

    is_solid = "SOLID" in elem_type.upper() or "BRICK" in elem_type.upper() or "TETRA" in elem_type.upper()

    if is_solid:
        if t_arr.ndim == 0:
            return np.array([eth, eth, eth, 0.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(t_arr.shape + (6,), dtype=np.float64)
        out[..., 0] = eth
        out[..., 1] = eth
        out[..., 2] = eth
        return out
    else:
        # Shell plane-stress in-plane strain: [exx, eyy, exy]
        if t_arr.ndim == 0:
            return np.array([eth, eth, 0.0], dtype=np.float64)
        out = np.zeros(t_arr.shape + (3,), dtype=np.float64)
        out[..., 0] = eth
        out[..., 1] = eth
        return out


def johnson_cook_thermal_softening(
    temp: Union[float, np.ndarray],
    t_room: float = 293.15,
    t_melt: float = 1793.15,
    m: float = 1.0,
) -> Union[float, np.ndarray]:
    """Evaluate Johnson-Cook thermal softening factor C_T = 1 - (T*)^m.

    Used in LAW02 (/MAT/LAW2) and LAW04 (/MAT/LAW4).

    Parameters
    ----------
    temp : float or np.ndarray
        Current absolute temperature.
    t_room : float, optional
        Room / reference temperature T_room.
    t_melt : float, optional
        Melting temperature T_melt.
    m : float, optional
        Thermal softening exponent m.

    Returns
    -------
    float or np.ndarray
        Softening multiplier C_T in [0, 1].
    """
    denom = max(float(t_melt) - float(t_room), 1e-20)
    t_arr = np.asarray(temp, dtype=np.float64)
    tstar = np.clip((t_arr - float(t_room)) / denom, 0.0, 1.0)
    ct = 1.0 - (tstar ** float(m))
    if t_arr.ndim == 0:
        return float(ct)
    return ct


# ----------------------------------------------------------------------------
# Builder Helper Functions
# ----------------------------------------------------------------------------

def build_uniform_initemp(
    t0: float,
    part_id: Optional[int] = None,
    grnod_id: Optional[int] = None,
    element_set_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    title: str = "",
    id: int = 1,
) -> InitempRecord:
    """Build a uniform initial temperature record.

    Parameters
    ----------
    t0 : float
        Uniform temperature value.
    part_id : int, optional
        Target part ID.
    grnod_id : int, optional
        Target node group ID.
    element_set_id : int, optional
        Target element set ID.
    node_ids : sequence of int, optional
        Explicit node IDs.
    title : str, optional
        Title string.
    id : int, optional
        Card ID.

    Returns
    -------
    InitempRecord
    """
    return InitempRecord(
        id=id,
        title=title or "UNIFORM_INITEMP",
        t0=float(t0),
        part_id=part_id,
        grnod_id=grnod_id,
        element_set_id=element_set_id,
        node_ids=node_ids,
        fld_type=0,
    )


def build_gradient_initemp(
    t0: float,
    gradient: Sequence[float],
    x0: Sequence[float] = (0.0, 0.0, 0.0),
    part_id: Optional[int] = None,
    grnod_id: Optional[int] = None,
    element_set_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    title: str = "",
    id: int = 1,
) -> InitempRecord:
    """Build a 3D linear spatial gradient initial temperature record.

    Formula:
        T(x) = T0 + Gx*(x - x0) + Gy*(y - y0) + Gz*(z - z0)

    Parameters
    ----------
    t0 : float
        Base temperature at reference point x0.
    gradient : sequence of float
        Gradient components (Gx, Gy, Gz) in [K/m].
    x0 : sequence of float, optional
        Reference origin coordinates (x0, y0, z0).
    part_id : int, optional
        Target part ID.
    grnod_id : int, optional
        Target node group ID.
    element_set_id : int, optional
        Target element set ID.
    node_ids : sequence of int, optional
        Explicit node IDs.
    title : str, optional
        Title string.
    id : int, optional
        Card ID.

    Returns
    -------
    InitempRecord
    """
    return InitempRecord(
        id=id,
        title=title or "GRADIENT_INITEMP",
        t0=float(t0),
        gradient=tuple(float(g) for g in gradient[:3]),
        x0=tuple(float(x) for x in x0[:3]),
        part_id=part_id,
        grnod_id=grnod_id,
        element_set_id=element_set_id,
        node_ids=node_ids,
        fld_type=0,
    )


def build_shell_gradient_initemp(
    t_mid: Optional[float] = None,
    t_top: Optional[float] = None,
    t_bot: Optional[float] = None,
    t0: float = 293.15,
    part_id: Optional[int] = None,
    grnod_id: Optional[int] = None,
    element_set_id: Optional[int] = None,
    title: str = "",
    id: int = 1,
) -> InitempRecord:
    """Build a shell through-thickness gradient initial temperature record.

    Parameters
    ----------
    t_mid : float, optional
        Mid-surface temperature (xi = 0).
    t_top : float, optional
        Top surface temperature (xi = +1).
    t_bot : float, optional
        Bottom surface temperature (xi = -1).
    t0 : float, optional
        Base temperature if t_mid is not given.
    part_id : int, optional
        Target part ID.
    grnod_id : int, optional
        Target node group ID.
    element_set_id : int, optional
        Target element set ID.
    title : str, optional
        Title string.
    id : int, optional
        Card ID.

    Returns
    -------
    InitempRecord
    """
    base = float(t_mid) if t_mid is not None else float(t0)
    return InitempRecord(
        id=id,
        title=title or "SHELL_GRADIENT_INITEMP",
        t0=base,
        t_mid=t_mid,
        t_top=t_top,
        t_bot=t_bot,
        part_id=part_id,
        grnod_id=grnod_id,
        element_set_id=element_set_id,
        fld_type=0,
    )


def build_nodal_table_initemp(
    t0: float,
    nodal_temps: Dict[int, float],
    grnod_id: Optional[int] = None,
    title: str = "",
    id: int = 1,
) -> InitempRecord:
    """Build an initial temperature record with individual nodal overrides.

    Parameters
    ----------
    t0 : float
        Base uniform temperature for the group.
    nodal_temps : dict
        Mapping {node_id: temperature_override}.
    grnod_id : int, optional
        Target node group ID.
    title : str, optional
        Title string.
    id : int, optional
        Card ID.

    Returns
    -------
    InitempRecord
    """
    return InitempRecord(
        id=id,
        title=title or "NODAL_TABLE_INITEMP",
        t0=float(t0),
        nodal_temps=dict(nodal_temps),
        grnod_id=grnod_id,
        fld_type=1,
    )


def parse_initemp_deck_cards(cards: Sequence[str]) -> List[InitempRecord]:
    """Parse raw text card lines defining /INITEMP blocks.

    Supported formats:
    - Line 1: /INITEMP/<id> or title
    - Line 2: T0  grnd_ID  fld_type
    - Lines 3+: T0_i  node_ID_i (when fld_type=1)
    - Extended keyword directives:
      GRADIENT Gx Gy Gz [x0 y0 z0]
      SHELL T_top T_mid T_bot
      PART part_id T0
    """
    records: List[InitempRecord] = []
    current_rec: Optional[InitempRecord] = None

    idx = 0
    while idx < len(cards):
        line = cards[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue

        if line.startswith("/INITEMP"):
            toks = line.split("/")
            rec_id = int(toks[2]) if len(toks) > 2 and toks[2].isdigit() else (len(records) + 1)
            # Read title if next line is not a number
            title = ""
            if idx < len(cards):
                next_line = cards[idx].strip()
                if next_line and not next_line.startswith("#") and not next_line.startswith("/"):
                    # check if next line starts with float
                    first_tok = next_line.split()[0]
                    try:
                        float(first_tok)
                    except ValueError:
                        title = next_line
                        idx += 1

            current_rec = InitempRecord(id=rec_id, title=title)
            records.append(current_rec)
            continue

        toks = line.split()
        if not toks:
            continue

        upper_0 = toks[0].upper()

        if upper_0 == "GRADIENT":
            if current_rec is None:
                current_rec = InitempRecord(id=len(records) + 1)
                records.append(current_rec)
            gx = float(toks[1]) if len(toks) > 1 else 0.0
            gy = float(toks[2]) if len(toks) > 2 else 0.0
            gz = float(toks[3]) if len(toks) > 3 else 0.0
            current_rec.gradient = (gx, gy, gz)
            if len(toks) >= 7:
                current_rec.x0 = (float(toks[4]), float(toks[5]), float(toks[6]))
            continue

        if upper_0 == "SHELL":
            if current_rec is None:
                current_rec = InitempRecord(id=len(records) + 1)
                records.append(current_rec)
            if len(toks) >= 4:
                current_rec.t_top = float(toks[1])
                current_rec.t_mid = float(toks[2])
                current_rec.t_bot = float(toks[3])
            elif len(toks) >= 3:
                current_rec.t_top = float(toks[1])
                current_rec.t_bot = float(toks[2])
            continue

        if upper_0 == "PART":
            if current_rec is None:
                current_rec = InitempRecord(id=len(records) + 1)
                records.append(current_rec)
            current_rec.part_id = int(toks[1])
            if len(toks) > 2:
                current_rec.t0 = float(toks[2])
            continue

        # Standard numeric cards:
        # Card format: T0  grnd_ID  [fld_type]
        # Or sub-table: T0_i  node_ID_i
        if current_rec is not None and current_rec.fld_type == 1 and len(toks) >= 2:
            try:
                t_val = float(toks[0])
                n_id = int(toks[1])
                current_rec.nodal_temps[n_id] = t_val
                continue
            except ValueError:
                pass

        try:
            t0 = float(toks[0])
            grnod_id = int(toks[1]) if len(toks) > 1 else 0
            fld_type = int(toks[2]) if len(toks) > 2 else 0
            if current_rec is None:
                current_rec = InitempRecord(id=len(records) + 1)
                records.append(current_rec)
            current_rec.t0 = t0
            current_rec.grnod_id = grnod_id if grnod_id != 0 else None
            current_rec.fld_type = fld_type
        except ValueError:
            continue

    return records
