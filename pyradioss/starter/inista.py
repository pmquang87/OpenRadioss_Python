"""
Initial Stress, Strain, and State Mapping (/INISTA, /INISTATE).

Fortran origins:
- ``starter/source/initial_conditions/inista/hm_read_inista.F`` (HM_READ_INISTA)
- ``starter/source/initial_conditions/inista/lec_inistate_yfile.F`` (LEC_INISTATE_YFILE)
- ``starter/source/initial_conditions/inista/yctrl.F`` (YCTRL)
- ``starter/source/elements/initia/lec_inistate.F`` (LEC_INISTATE)

Theory and Architecture
-----------------------
OpenRadioss supports pre-stressing and pre-straining of elements via initial state mapping:
1. State Variables:
   - Cauchy stress tensor: sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx
   - Equivalent plastic strain: epsp (pre-hardening materials start at elevated yield stress)
   - Bending / through-thickness distribution for shell elements:
     For shells with thickness t and layers z_k in [-t/2, t/2] (relative coordinate xi_k in [-1, 1]):
     sigma_k = sigma_membrane + xi_k * sigma_bending
   - Backstress tensor alpha for kinematic hardening (e.g. LAW46 / LAW131)
2. Initialization into model element state:
   - Populates group.state['sig'] and group.state['epsp']
   - Distributes through-thickness integration points for shells
   - Supports part-level (all elements in a part), group-level, or element-by-element mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.messages import MessageLog
from ..model.model import Model


@dataclass
class InistaRecord:
    """Initial stress, strain, and state mapping record.

    Fortran origins:
      - ``starter/source/initial_conditions/inista/hm_read_inista.F``
      - ``starter/source/initial_conditions/inista/lec_inistate_yfile.F``
      - ``starter/source/initial_conditions/inista/yctrl.F``

    Attributes
    ----------
    part_id : int, optional
        Target part ID. If specified, maps state to all elements belonging to this part.
    group_id : int, optional
        Target element group / subset ID.
    elem_id : int, optional
        Target individual element ID.
    elem_type : str, optional
        Element type hint: "SOLID", "BRICK", "SHELL", "QUAD", "TRIA", "SH3N".
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx : float
        Components of the initial Cauchy / membrane stress tensor.
    sigma_b_xx, sigma_b_yy, sigma_b_xy : float
        Components of initial bending stress tensor for shell elements.
    epsp : float
        Initial equivalent plastic strain (pre-hardening).
    layer : int, optional
        1-based integration point or layer index for shell elements. None applies across all layers.
    alpha : np.ndarray, optional
        Initial backstress tensor (6,) for kinematic hardening models (e.g. LAW46).
    uvar : list of float, optional
        User state / history variables.
    """

    part_id: Optional[int] = None
    group_id: Optional[int] = None
    elem_id: Optional[int] = None
    elem_type: Optional[str] = None
    sigma_xx: Optional[float] = None
    sigma_yy: Optional[float] = None
    sigma_zz: Optional[float] = None
    sigma_xy: Optional[float] = None
    sigma_yz: Optional[float] = None
    sigma_zx: Optional[float] = None
    sigma_b_xx: Optional[float] = None
    sigma_b_yy: Optional[float] = None
    sigma_b_xy: Optional[float] = None
    epsp: Optional[float] = None
    layer: Optional[int] = None
    alpha: Optional[np.ndarray] = None
    uvar: Optional[List[float]] = None

    @property
    def has_stress(self) -> bool:
        """True if any stress component was explicitly defined."""
        return any(
            v is not None
            for v in (
                self.sigma_xx, self.sigma_yy, self.sigma_zz,
                self.sigma_xy, self.sigma_yz, self.sigma_zx,
                self.sigma_b_xx, self.sigma_b_yy, self.sigma_b_xy,
            )
        )

    @property
    def sigma(self) -> np.ndarray:
        """Return the 6-component Cauchy / membrane stress array."""
        return np.array([
            self.sigma_xx if self.sigma_xx is not None else 0.0,
            self.sigma_yy if self.sigma_yy is not None else 0.0,
            self.sigma_zz if self.sigma_zz is not None else 0.0,
            self.sigma_xy if self.sigma_xy is not None else 0.0,
            self.sigma_yz if self.sigma_yz is not None else 0.0,
            self.sigma_zx if self.sigma_zx is not None else 0.0,
        ], dtype=np.float64)

    @sigma.setter
    def sigma(self, val: Union[Sequence[float], np.ndarray]) -> None:
        arr = np.asarray(val, dtype=np.float64)
        if len(arr) >= 1:
            self.sigma_xx = float(arr[0])
        if len(arr) >= 2:
            self.sigma_yy = float(arr[1])
        if len(arr) >= 3:
            self.sigma_zz = float(arr[2])
        if len(arr) >= 4:
            self.sigma_xy = float(arr[3])
        if len(arr) >= 5:
            self.sigma_yz = float(arr[4])
        if len(arr) >= 6:
            self.sigma_zx = float(arr[5])

    @property
    def sigma_b(self) -> np.ndarray:
        """Return the shell bending stress array."""
        return np.array([
            self.sigma_b_xx if self.sigma_b_xx is not None else 0.0,
            self.sigma_b_yy if self.sigma_b_yy is not None else 0.0,
            0.0,
            self.sigma_b_xy if self.sigma_b_xy is not None else 0.0,
            0.0,
            0.0,
        ], dtype=np.float64)

    @sigma_b.setter
    def sigma_b(self, val: Union[Sequence[float], np.ndarray]) -> None:
        arr = np.asarray(val, dtype=np.float64)
        if len(arr) >= 1:
            self.sigma_b_xx = float(arr[0])
        if len(arr) >= 2:
            self.sigma_b_yy = float(arr[1])
        if len(arr) >= 4:
            self.sigma_b_xy = float(arr[3])
        elif len(arr) == 3:
            self.sigma_b_xy = float(arr[2])


def _get_shell_layer_coords(nip: int, zw: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> np.ndarray:
    """Compute normalized through-thickness layer coordinates xi_k in [-1, 1]."""
    if nip <= 1:
        return np.zeros(1, dtype=np.float64)
    if zw and len(zw) > 0 and len(zw[0][0]) == nip:
        # Gauss points in [-1, 1] relative to half-thickness
        gp = zw[0][0] * 2.0
        return np.asarray(gp, dtype=np.float64)
    # Default Gauss-Legendre quadrature points in [-1, 1]
    gp, _ = np.polynomial.legendre.leggauss(nip)
    return np.asarray(gp, dtype=np.float64)


def apply_inista_record_to_group(
    record: InistaRecord,
    group_name: str,
    group: Any,
    indices: Sequence[int],
    log: Optional[MessageLog] = None,
) -> None:
    """Apply an InistaRecord to specified local element indices of an element group.

    Parameters
    ----------
    record : InistaRecord
        Initial state record to apply.
    group_name : str
        Element group name ("BRICK", "SHELL", "TETRA4", etc.).
    group : ElementGroup
        The target element group containing group.state.
    indices : sequence of int
        Local indices within the group to update.
    log : MessageLog, optional
        Logger instance.
    """
    if len(indices) == 0 or not hasattr(group, "state") or not isinstance(group.state, dict):
        return

    idx_arr = np.asarray(indices, dtype=np.int64)
    gname_upper = group_name.upper()
    sig_state = group.state.get("sig")
    epsp_state = group.state.get("epsp")
    if sig_state is None and epsp_state is None:
        return

    is_shell = (
        "SHELL" in gname_upper
        or "SH3N" in gname_upper
        or "SHEL" in gname_upper
        or (record.elem_type is not None and record.elem_type.upper() in ("SHELL", "QUAD", "TRIA", "SH3N"))
        or (sig_state is not None and sig_state.ndim == 3 and sig_state.shape[2] == 3)
    ) and (record.elem_type is None or record.elem_type.upper() not in ("SOLID", "BRICK", "TETRA"))

    # -------------------------------------------------------------------------
    # 1. Shell Elements
    # -------------------------------------------------------------------------
    if is_shell:
        s_mem = record.sigma
        s_bend = record.sigma_b
        has_bending = (
            abs(s_bend[0]) > 1e-15
            or abs(s_bend[1]) > 1e-15
            or abs(s_bend[3]) > 1e-15
        )

        if sig_state is not None and record.has_stress:
            if sig_state.ndim == 2:
                sig_view = sig_state[:, None, :]
            else:
                sig_view = sig_state

            nip = sig_view.shape[1]
            n_comp = sig_view.shape[2]
            xi = _get_shell_layer_coords(nip, group.state.get("zw"))

            for i in idx_arr:
                if record.layer is not None:
                    # Specific layer (1-based index)
                    k = record.layer - 1
                    if 0 <= k < nip:
                        if n_comp == 3:
                            sig_view[i, k, 0] = s_mem[0]
                            sig_view[i, k, 1] = s_mem[1]
                            sig_view[i, k, 2] = s_mem[3]
                        else:
                            sig_view[i, k, 0] = s_mem[0]
                            sig_view[i, k, 1] = s_mem[1]
                            sig_view[i, k, 2] = s_mem[2]
                            sig_view[i, k, 3] = s_mem[3]
                            sig_view[i, k, 4] = s_mem[4]
                            sig_view[i, k, 5] = s_mem[5]
                else:
                    # All layers: apply through-thickness distribution
                    for k in range(nip):
                        xi_k = float(xi[k]) if k < len(xi) else 0.0
                        if has_bending:
                            sxx_k = s_mem[0] + xi_k * s_bend[0]
                            syy_k = s_mem[1] + xi_k * s_bend[1]
                            sxy_k = s_mem[3] + xi_k * s_bend[3]
                        else:
                            sxx_k = s_mem[0]
                            syy_k = s_mem[1]
                            sxy_k = s_mem[3]

                        if n_comp == 3:
                            sig_view[i, k, 0] = sxx_k
                            sig_view[i, k, 1] = syy_k
                            sig_view[i, k, 2] = sxy_k
                        else:
                            sig_view[i, k, 0] = sxx_k
                            sig_view[i, k, 1] = syy_k
                            sig_view[i, k, 2] = s_mem[2]
                            sig_view[i, k, 3] = sxy_k
                            sig_view[i, k, 4] = s_mem[4]
                            sig_view[i, k, 5] = s_mem[5]

        if epsp_state is not None and record.epsp is not None:
            for i in idx_arr:
                if epsp_state.ndim == 1:
                    epsp_state[i] = record.epsp
                elif epsp_state.ndim == 2:
                    if record.layer is not None:
                        k = record.layer - 1
                        if 0 <= k < epsp_state.shape[1]:
                            epsp_state[i, k] = record.epsp
                    else:
                        epsp_state[i, :] = record.epsp

    # -------------------------------------------------------------------------
    # 2. Solid Elements
    # -------------------------------------------------------------------------
    else:
        s_vec = record.sigma  # (6,)
        if sig_state is not None and record.has_stress:
            for i in idx_arr:
                if sig_state.ndim == 2:
                    sig_state[i, :min(6, sig_state.shape[1])] = s_vec[:min(6, sig_state.shape[1])]
                elif sig_state.ndim == 3:
                    # Multi-point solid (e.g. 8 Gauss points)
                    if record.layer is not None:
                        k = record.layer - 1
                        if 0 <= k < sig_state.shape[1]:
                            sig_state[i, k, :min(6, sig_state.shape[2])] = s_vec[:min(6, sig_state.shape[2])]
                    else:
                        sig_state[i, :, :min(6, sig_state.shape[2])] = s_vec[:min(6, sig_state.shape[2])]

        if epsp_state is not None and record.epsp is not None:
            for i in idx_arr:
                if epsp_state.ndim == 1:
                    epsp_state[i] = record.epsp
                elif epsp_state.ndim == 2:
                    if record.layer is not None:
                        k = record.layer - 1
                        if 0 <= k < epsp_state.shape[1]:
                            epsp_state[i, k] = record.epsp
                    else:
                        epsp_state[i, :] = record.epsp

    # -------------------------------------------------------------------------
    # 3. Kinematic Backstress alpha
    # -------------------------------------------------------------------------
    if record.alpha is not None:
        if "alpha" not in group.state:
            group.state["alpha"] = np.zeros((group.n, 6), dtype=np.float64)
        alpha_st = group.state["alpha"]
        for i in idx_arr:
            if alpha_st.ndim == 2:
                alpha_st[i, :min(6, alpha_st.shape[1])] = record.alpha[:min(6, alpha_st.shape[1])]
            elif alpha_st.ndim == 3:
                if record.layer is not None:
                    k = record.layer - 1
                    if 0 <= k < alpha_st.shape[1]:
                        alpha_st[i, k, :min(6, alpha_st.shape[2])] = record.alpha[:min(6, alpha_st.shape[2])]
                else:
                    alpha_st[i, :, :min(6, alpha_st.shape[2])] = record.alpha[:min(6, alpha_st.shape[2])]


def apply_inista(
    model: Model,
    records: Optional[Sequence[InistaRecord]] = None,
    log: Optional[MessageLog] = None,
) -> None:
    """Map initial stresses, strains, and states to model element groups.

    Consolidates /INISTA records, /INIBRI records, and /INISHE records into
    group.state['sig'] and group.state['epsp'].

    Parameters
    ----------
    model : Model
        Target model with initialized element groups.
    records : sequence of InistaRecord, optional
        Direct records to apply. If None, uses model.inista_records and
        any parsed ini_bricks / ini_shells.
    log : MessageLog, optional
        Logger instance for reporting.
    """
    all_records: List[InistaRecord] = []
    if records is not None:
        all_records.extend(records)

    # Include any programmatic or parsed records attached to Model
    if hasattr(model, "inista_records") and model.inista_records:
        all_records.extend(model.inista_records)

    # Convert model.ini_bricks (/INIBRI) if present
    if hasattr(model, "ini_bricks") and model.ini_bricks:
        for eid, b_st in model.ini_bricks.items():
            s = b_st.sigma if hasattr(b_st, "sigma") and len(b_st.sigma) == 6 else np.zeros(6)
            ep = float(getattr(b_st, "epsp", 0.0))
            all_records.append(InistaRecord(
                elem_id=eid,
                sigma_xx=s[0], sigma_yy=s[1], sigma_zz=s[2],
                sigma_xy=s[3], sigma_yz=s[4], sigma_zx=s[5],
                epsp=ep,
            ))

    # Convert model.ini_shells (/INISHE) if present
    if hasattr(model, "ini_shells") and model.ini_shells:
        for eid, s_st in model.ini_shells.items():
            s = s_st.sigma if hasattr(s_st, "sigma") and len(s_st.sigma) >= 3 else np.zeros(6)
            sb = s_st.sigma_b if hasattr(s_st, "sigma_b") and len(s_st.sigma_b) >= 3 else np.zeros(6)
            ep = float(getattr(s_st, "epsp", 0.0))
            all_records.append(InistaRecord(
                elem_id=eid,
                sigma_xx=s[0], sigma_yy=s[1], sigma_zz=s[2] if len(s) > 2 else 0.0,
                sigma_xy=s[3] if len(s) > 3 else (s[2] if len(s) == 3 else 0.0),
                sigma_b_xx=sb[0], sigma_b_yy=sb[1],
                sigma_b_xy=sb[3] if len(sb) > 3 else (sb[2] if len(sb) == 3 else 0.0),
                epsp=ep,
            ))

    if not all_records:
        return

    # Build lookup maps for fast indexing:
    # 1. elem_id -> (group_name, group, local_idx)
    # 2. part_id -> list of (group_name, group, local_indices)
    elem_map: Dict[int, Tuple[str, Any, int]] = {}
    part_map: Dict[int, List[Tuple[str, Any, np.ndarray]]] = {}

    for gname, group in model.element_groups():
        if not hasattr(group, "ids") or group.ids is None or len(group.ids) == 0:
            continue
        for l_idx, eid in enumerate(group.ids):
            elem_map[int(eid)] = (gname, group, l_idx)

        p_ids = group.state.get("part_ids") if hasattr(group, "state") else None
        if p_ids is None and hasattr(group, "part") and group.part is not None:
            if hasattr(model, "parts_list") and len(model.parts_list) > 0:
                try:
                    p_ids = np.array([model.parts_list[int(idx)].id for idx in group.part], dtype=np.int64)
                except Exception:
                    p_ids = np.asarray(group.part, dtype=np.int64)
            else:
                p_ids = np.asarray(group.part, dtype=np.int64)

        if p_ids is not None:
            unique_parts = np.unique(p_ids)
            for pid in unique_parts:
                mask = np.where(p_ids == pid)[0]
                part_map.setdefault(int(pid), []).append((gname, group, mask))

    # Apply each record
    applied_count = 0
    for rec in all_records:
        # Case 1: Individual Element ID
        if rec.elem_id is not None:
            target = elem_map.get(rec.elem_id)
            if target is not None:
                gname, grp, l_idx = target
                apply_inista_record_to_group(rec, gname, grp, [l_idx], log=log)
                applied_count += 1
            elif log is not None:
                log.warning(f"/INISTA: element {rec.elem_id} not found in model", "INISTA")

        # Case 2: Entire Part ID
        elif rec.part_id is not None:
            targets = part_map.get(rec.part_id)
            if targets is not None:
                for gname, grp, indices in targets:
                    apply_inista_record_to_group(rec, gname, grp, indices, log=log)
                    applied_count += len(indices)
            elif log is not None:
                log.warning(f"/INISTA: part {rec.part_id} not found in model", "INISTA")

        # Case 3: Group ID (e.g. element set)
        elif rec.group_id is not None and hasattr(model, "egroups"):
            # Check element sets
            e_set = None
            for g_type in ("BRICK", "SHELL", "SOLID", "PART"):
                if g_type in model.egroups and rec.group_id in model.egroups[g_type]:
                    e_set = model.egroups[g_type][rec.group_id]
                    break
            if e_set is not None:
                members = getattr(e_set, "members", [])
                for eid in members:
                    target = elem_map.get(int(eid))
                    if target is not None:
                        gname, grp, l_idx = target
                        apply_inista_record_to_group(rec, gname, grp, [l_idx], log=log)
                        applied_count += 1

    if log is not None and applied_count > 0:
        log.info(f" /INISTA: mapped initial stress/strain to {applied_count} element state(s)")


def parse_inista_deck_cards(cards: Sequence[str]) -> List[InistaRecord]:
    """Parse raw text lines defining INISTA records (e.g. from a state table or card).

    Supported line formats:
      1. Free format:
         ELEM_ID  SIG_XX  SIG_YY  SIG_ZZ  SIG_XY  SIG_YZ  SIG_ZX  [EPSP]
      2. Part format:
         PART <part_id>  SIG_XX  SIG_YY  SIG_ZZ  SIG_XY  SIG_YZ  SIG_ZX  [EPSP]
      3. Shell layer format:
         ELEM_ID  LAYER <layer_num>  SIG_XX  SIG_YY  SIG_XY  [EPSP]
    """
    records: List[InistaRecord] = []
    for line in cards:
        line_s = line.strip()
        if not line_s or line_s.startswith("#") or line_s.startswith("/"):
            continue

        toks = line_s.split()
        if not toks:
            continue

        if toks[0].upper() == "PART":
            part_id = int(toks[1])
            floats = [float(x) for x in toks[2:]]
            sxx = floats[0] if len(floats) > 0 else 0.0
            syy = floats[1] if len(floats) > 1 else 0.0
            szz = floats[2] if len(floats) > 2 else 0.0
            sxy = floats[3] if len(floats) > 3 else 0.0
            syz = floats[4] if len(floats) > 4 else 0.0
            szx = floats[5] if len(floats) > 5 else 0.0
            epsp = floats[6] if len(floats) > 6 else 0.0
            records.append(InistaRecord(
                part_id=part_id,
                sigma_xx=sxx, sigma_yy=syy, sigma_zz=szz,
                sigma_xy=sxy, sigma_yz=syz, sigma_zx=szx,
                epsp=epsp,
            ))
        elif len(toks) >= 2 and toks[1].upper() == "LAYER":
            elem_id = int(toks[0])
            layer = int(toks[2])
            floats = [float(x) for x in toks[3:]]
            sxx = floats[0] if len(floats) > 0 else 0.0
            syy = floats[1] if len(floats) > 1 else 0.0
            sxy = floats[2] if len(floats) > 2 else 0.0
            epsp = floats[3] if len(floats) > 3 else 0.0
            records.append(InistaRecord(
                elem_id=elem_id, layer=layer,
                sigma_xx=sxx, sigma_yy=syy, sigma_xy=sxy,
                epsp=epsp,
            ))
        else:
            elem_id = int(toks[0])
            floats = [float(x) for x in toks[1:]]
            sxx = floats[0] if len(floats) > 0 else 0.0
            syy = floats[1] if len(floats) > 1 else 0.0
            szz = floats[2] if len(floats) > 2 else 0.0
            sxy = floats[3] if len(floats) > 3 else 0.0
            syz = floats[4] if len(floats) > 4 else 0.0
            szx = floats[5] if len(floats) > 5 else 0.0
            epsp = floats[6] if len(floats) > 6 else 0.0
            records.append(InistaRecord(
                elem_id=elem_id,
                sigma_xx=sxx, sigma_yy=syy, sigma_zz=szz,
                sigma_xy=sxy, sigma_yz=syz, sigma_zx=szx,
                epsp=epsp,
            ))

    return records
