"""
Contact bookkeeping against element deletion (/FAIL, M3).

Fortran origin: the ``IDEL`` option of the interfaces —
``engine/source/interfaces/interf/intfop2.F`` (segment bookkeeping after
element deletion), ``engine/source/interfaces/int07/i7for3.F`` (dead-segment
skip: segments with parent ``GBUF%OFF <= 0`` are skipped in the force loop),
and ``engine/source/interfaces/interf/int_checksum.F`` (contact-surface
integrity checks).

Why this matters physically: when a /FAIL criterion deletes an element,
its former faces become *free surface* — crack faces. If the contact kept
its old segments, the two sides of a crack would keep pushing (or, worse,
a secondary node trapped inside the gap would keep pulling everything
around), which is exactly the artifact the notched-plate class of models
must not show. Symmetrically, a secondary node ALL of whose elements died
carries no structural meaning any more (it keeps flying with its mass,
but nothing is attached): it must stop being tracked so it cannot
transmit force to the surviving structure.

Both masks are cheap (one gather / one scatter per element group), so the
segment mask is refreshed every cycle and the node mask at every broad
phase — deletion between two broad phases cannot create a *new* candidate
pair anyway, it can only kill existing ones.
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model

#: provenance prefix of a segment whose parent element is a GHOST copy
#: (SPMD: a non-local element kept on the interface owner, see
#: pyradioss/spmd/domdec.py) — the element lives in ``model.spmd_ghost``
GHOST_PREFIX = "ghost:"


def _resolve_group(model: Model, gname: str):
    """Element group named by a segment provenance: a regular group
    attribute of the model, or ``"ghost:<group>"`` for the SPMD ghost-ring
    copy in ``model.spmd_ghost`` (serial models have no ghost ring)."""
    gname = str(gname)
    if gname.startswith(GHOST_PREFIX):
        ghosts = getattr(model, "spmd_ghost", None) or {}
        return ghosts.get(gname[len(GHOST_PREFIX):])
    return getattr(model, gname, None)


def _base_name(gname: str) -> str:
    """Group family of a provenance name ("ghost:shells" -> "shells")."""
    gname = str(gname)
    return gname[len(GHOST_PREFIX):] if gname.startswith(GHOST_PREFIX) else gname


def _all_groups(model: Model):
    """(name, group) over the model's element groups, then the SPMD ghost
    ring groups (same family names) — for the loops that must see every
    element touching a node.  Serial models: element_groups() only."""
    yield from model.element_groups()
    ghosts = getattr(model, "spmd_ghost", None) or {}
    for name, group in ghosts.items():
        if group is not None and group.n:
            yield name, group


def alive_segment_mask(model: Model, seg_gtype: np.ndarray,
                       seg_elem: np.ndarray) -> np.ndarray:
    """True for segments whose parent element is alive (off > 0).

    Segments without provenance (''), and segments of element groups that
    never delete (no 'off' array), stay True forever.
    """
    mask = np.ones(len(seg_gtype), dtype=bool)
    for gname in np.unique(seg_gtype):
        if gname == "":
            continue
        group = _resolve_group(model, gname)
        if group is None or getattr(group, "state", None) is None:
            continue
        off = group.state.get("off")
        if off is None:
            continue
        sel = (seg_gtype == gname) & (seg_elem >= 0) & (seg_elem < len(off))
        mask[sel] = off[seg_elem[sel]] > 0.0
    return mask


def any_deletable(model: Model, seg_gtype: np.ndarray,
                   sec_nodes: np.ndarray | None = None) -> bool:
    """False when no referenced group can ever delete an element — lets
    the per-cycle mask refresh be skipped entirely for plain models.

    When *sec_nodes* is given, also check all element groups that contain
    any of those nodes (the secondary side).  This catches the case where
    only secondary-side elements carry a /FAIL criterion.
    """
    # Main-surface check (original logic)
    for gname in np.unique(seg_gtype):
        if gname == "":
            continue
        group = _resolve_group(model, gname)
        if group is not None and group.state.get("off") is not None and group.state.get(
                "chk_fail", False):
            return True
    # Secondary-node check: scan all element groups for any that contain
    # a secondary node AND have failure capability.
    if sec_nodes is not None and len(sec_nodes) > 0 and hasattr(model, "element_groups"):
        for gname, group in _all_groups(model):
            if group.state.get("off") is None or not group.state.get(
                    "chk_fail", False):
                continue
            if np.isin(group.conn, sec_nodes).any():
                return True
    return False


def node_reference_counts(model: Model, alive_only: bool) -> np.ndarray:
    """How many elements reference each node (all groups). With
    ``alive_only`` the count is restricted to elements with off > 0."""
    cnt = np.zeros(model.numnod, dtype=np.int64)
    for _, group in _all_groups(model):
        conn = group.conn
        if alive_only:
            off = group.state.get("off")
            if off is not None:
                conn = conn[off > 0.0]
        flat = conn.reshape(-1)
        valid = (flat >= 0) & (flat < model.numnod)
        np.add.at(cnt, flat[valid], 1)
    return cnt


def tracked_node_mask(model: Model, ref_total: np.ndarray) -> np.ndarray:
    """True for nodes contact should keep tracking: nodes still belonging
    to at least one alive element — plus nodes that never belonged to any
    element (extra masses, orientation nodes: nothing to 'delete')."""
    alive = node_reference_counts(model, alive_only=True)
    return (ref_total == 0) | (alive > 0)
