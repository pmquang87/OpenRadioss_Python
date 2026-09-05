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
        off = getattr(model, gname).state.get("off")
        if off is None:
            continue
        sel = seg_gtype == gname
        mask[sel] = off[seg_elem[sel]] > 0.0
    return mask


def any_deletable(model: Model, seg_gtype: np.ndarray) -> bool:
    """False when no referenced group can ever delete an element — lets
    the per-cycle mask refresh be skipped entirely for plain models."""
    for gname in np.unique(seg_gtype):
        if gname == "":
            continue
        group = getattr(model, gname)
        if group.state.get("off") is not None and group.state.get(
                "chk_fail", False):
            return True
    return False


def node_reference_counts(model: Model, alive_only: bool) -> np.ndarray:
    """How many elements reference each node (all groups). With
    ``alive_only`` the count is restricted to elements with off > 0."""
    cnt = np.zeros(model.numnod, dtype=np.int64)
    for _, group in model.element_groups():
        conn = group.conn
        if alive_only:
            off = group.state.get("off")
            if off is not None:
                conn = conn[off > 0.0]
        np.add.at(cnt, conn.reshape(-1), 1)
    return cnt


def tracked_node_mask(model: Model, ref_total: np.ndarray) -> np.ndarray:
    """True for nodes contact should keep tracking: nodes still belonging
    to at least one alive element — plus nodes that never belonged to any
    element (extra masses, orientation nodes: nothing to 'delete')."""
    alive = node_reference_counts(model, alive_only=True)
    return (ref_total == 0) | (alive > 0)
