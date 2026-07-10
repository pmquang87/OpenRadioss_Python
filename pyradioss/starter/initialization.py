"""
Starter finalization: user IDs → indices, element groups, node groups,
contact surfaces, lumped mass.

Fortran origin: the Starter phases after reading — ``USR2SYS`` id
conversion, ``s/c/t/r-init3`` element initialization, ``inimass``/
``initwg`` mass building, the group and surface builders under
``starter/source/model/sets``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..common.messages import MessageLog
from ..elements import KERNELS
from ..model.model import ElementGroup, Model

# element type name -> (attr on Model, nodes per element, required prop type)
_ETYPES = {
    "BRICK": ("bricks", 8, 14),
    "SHELL": ("shells", 4, 1),
    "TRUSS": ("trusses", 2, 2),
    "SPRING": ("springs", 2, 4),
}


# ----------------------------------------------------------------------------
# Elements: raw tuples -> ElementGroups with per-part slices
# ----------------------------------------------------------------------------

def build_element_groups(model: Model, log: MessageLog) -> None:
    """Convert the raw (id, part, nodes) tuples collected by the parsers
    into dense ElementGroups, **sorted by part** so that each part is a
    contiguous slice — the Python equivalent of the Fortran element
    *groups* (NGROUP blocks of same type/mat/prop), which lets the material
    law run vectorized on each slice."""
    for etype, (attr, nnode, req_prop) in _ETYPES.items():
        raw = model.raw_elems[etype]
        if not raw:
            continue
        raw.sort(key=lambda t: t[1])                 # sort by part id
        ids = np.array([t[0] for t in raw], dtype=np.int64)
        part_ids = np.array([t[1] for t in raw], dtype=np.int64)

        # user node ids -> indices (USR2SYS)
        conn = np.zeros((len(raw), nnode), dtype=np.int64)
        ok = True
        for k, (eid, pid, nodes) in enumerate(raw):
            try:
                conn[k] = model.node_indices(nodes)
            except KeyError as exc:
                log.error(f"/{etype} {eid}: unknown node id {exc}",
                          "ELEMENT CHECK")
                ok = False
        if not ok:
            continue

        # part index + per-part slices with resolved (mat, prop)
        part_idx = np.zeros(len(raw), dtype=np.int64)
        slices = []
        start = 0
        for pid in np.unique(part_ids):
            sel = np.where(part_ids == pid)[0]
            end = start + len(sel)                   # contiguous after sort
            part = model.parts.get(int(pid))
            if part is None:
                log.error(f"/{etype}: part {pid} not defined", "PART CHECK")
                start = end
                continue
            if part not in model.parts_list:
                model.parts_list.append(part)
            mat = model.materials.get(part.mat_id)
            prop = model.properties.get(part.prop_id)
            if mat is None:
                log.error(f"/PART/{pid}: material {part.mat_id} not defined",
                          "PART CHECK")
            if prop is None:
                log.error(f"/PART/{pid}: property {part.prop_id} not defined",
                          "PART CHECK")
            elif prop.type != req_prop:
                log.error(f"/PART/{pid}: /{etype} elements need /PROP/TYPE"
                          f"{req_prop}, got TYPE{prop.type}", "PART CHECK")
            if mat is not None and prop is not None:
                slices.append((slice(start, end), mat, prop))
            part_idx[sel] = model.parts_list.index(part) if part in \
                model.parts_list else 0
            start = end

        group = ElementGroup(ids=ids, conn=conn, part=part_idx)
        group.state["slices"] = slices
        group.state["part_ids"] = part_ids
        setattr(model, attr, group)


# ----------------------------------------------------------------------------
# Node groups and surfaces
# ----------------------------------------------------------------------------

def _nodes_of_parts(model: Model, part_ids: List[int]) -> np.ndarray:
    """All node indices used by elements of the given parts."""
    out: List[np.ndarray] = []
    for _, group in model.element_groups():
        mask = np.isin(group.state["part_ids"], part_ids)
        if np.any(mask):
            out.append(np.unique(group.conn[mask]))
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(out))


def resolve_node_groups(model: Model, log: MessageLog) -> None:
    """/GRNOD content (node/part/box lists) -> dense node index arrays."""
    for g in model.node_groups.values():
        idx: List[np.ndarray] = []
        if g.node_ids:
            try:
                idx.append(model.node_indices(g.node_ids))
            except KeyError as exc:
                log.error(f"/GRNOD/{g.id}: unknown node id {exc}",
                          "GROUP CHECK")
        if g.part_ids:
            idx.append(_nodes_of_parts(model, g.part_ids))
        for bid in g.box_ids:
            box = model.boxes.get(bid)
            if box is None:
                log.error(f"/GRNOD/{g.id}: unknown box {bid}", "GROUP CHECK")
                continue
            inside = np.all((model.x0 >= box.corner_min)
                            & (model.x0 <= box.corner_max), axis=1)
            idx.append(np.where(inside)[0])
        g.node_idx = (np.unique(np.concatenate(idx)) if idx
                      else np.zeros(0, dtype=np.int64))
        if g.node_idx.size == 0:
            log.warning(f"/GRNOD/{g.id} '{g.title}' is empty", "GROUP CHECK")


def _free_faces_of_bricks(model: Model, part_ids: List[int]) -> np.ndarray:
    """Outer (free) faces of the given solid parts: faces used by exactly
    one element. Fortran: the surface-from-part extraction of
    starter/source/model/sets/."""
    from ..elements.solid_hexa8 import _FACES
    g = model.bricks
    if g is None:
        return np.zeros((0, 4), dtype=np.int64)
    mask = np.isin(g.state["part_ids"], part_ids)
    conn = g.conn[mask]
    faces = conn[:, _FACES.reshape(-1)].reshape(-1, 4)      # (nelem*6, 4)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    return faces[counts[inverse] == 1]


def resolve_surfaces(model: Model, log: MessageLog) -> None:
    """/SURF content -> (nseg, 4) node-index arrays."""
    for s in model.surfaces.values():
        segs: List[np.ndarray] = []
        for row in s.seg_nodes:
            try:
                segs.append(model.node_indices(row)[None, :])
            except KeyError as exc:
                log.error(f"/SURF/{s.id}: unknown node id {exc}",
                          "SURFACE CHECK")
        if s.part_ids:
            # shell parts: every shell element is a segment
            if model.shells is not None:
                mask = np.isin(model.shells.state["part_ids"], s.part_ids)
                if np.any(mask):
                    segs.append(model.shells.conn[mask])
            # solid parts: free outer faces
            ff = _free_faces_of_bricks(model, s.part_ids)
            if len(ff):
                segs.append(ff)
        s.segments = (np.vstack(segs) if segs
                      else np.zeros((0, 4), dtype=np.int64))
        if s.segments.shape[0] == 0:
            log.warning(f"/SURF/{s.id} '{s.title}' has no segments",
                        "SURFACE CHECK")


# ----------------------------------------------------------------------------
# Element initialization + lumped mass (inimass)
# ----------------------------------------------------------------------------

def initialize_elements_and_mass(model: Model, log: MessageLog) -> None:
    """Run every kernel's init_group and assemble the lumped nodal mass
    (and rotational inertia, from shells). Also applies /INIVEL."""
    model.x = model.x0.copy()
    model.v = np.zeros((model.numnod, 3))
    model.vr = np.zeros((model.numnod, 3))
    model.mass = np.zeros(model.numnod)
    model.inertia = np.zeros(model.numnod)

    for name, group in model.element_groups():
        node_idx, mass_c, inertia_c = KERNELS[name].init_group(
            group, model, log)
        np.add.at(model.mass, node_idx, mass_c)
        if inertia_c is not None:
            np.add.at(model.inertia, node_idx, inertia_c)

    # /INIVEL
    for iv in model.inivel:
        g = model.node_groups.get(iv.grnod_id)
        if g is None or g.node_idx is None:
            log.error(f"/INIVEL/{iv.id}: unknown node group {iv.grnod_id}",
                      "INIVEL CHECK")
            continue
        model.v[g.node_idx] = iv.v

    # massless nodes: harmless if nothing ever loads them, fatal otherwise.
    # The Engine divides force by mass, so give unreferenced nodes a tiny
    # mass and warn (the original errors out for loaded massless nodes).
    massless = model.mass <= 0.0
    if np.any(massless):
        log.warning(f"{int(massless.sum())} node(s) carry no mass "
                    f"(not referenced by any element); they are frozen.",
                    "MASS INIT")
        model.mass[massless] = 1e30  # infinite mass = frozen node
