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
    "TETRA4": ("tetras", 4, 14),
    "SHELL": ("shells", 4, 1),
    "SH3N": ("sh3n", 3, 1),
    "TRUSS": ("trusses", 2, 2),
    "SPRING": ("springs", 2, 4),
    "BEAM": ("beams", 3, 3),
}


# ----------------------------------------------------------------------------
# Degenerated bricks: repeated nodes -> tetra conversion / clear rejection
# ----------------------------------------------------------------------------

def _convert_degenerated_bricks(model: Model, log: MessageLog) -> None:
    """Handle /BRICK cards with repeated node IDs (the classic Radioss way
    of writing lower-order solids in brick format).

    Fortran origin: the brick reader (hm_read_brick / sinit3) detects
    repeated nodes and switches the element to its degenerated formulation
    (tetra, penta...). This port converts the 4-distinct-node patterns
    (e.g. ``n1 n2 n3 n3 n5 n5 n5 n5``) to genuine /TETRA4 elements — the
    constant-strain tetra IS the right element for that geometry, whereas
    running it as a collapsed hexa leaves zero-volume sub-shapes in the
    hourglass base vectors. Distinct nodes are taken in order of first
    appearance, which maps every standard collapse pattern onto the
    positively-oriented tetra (checked again at element init).

    Pentas (6 distinct) and pyramids (5 distinct) are NOT silently
    degraded: the Starter stops with a clear message (roadmap item), which
    beats the M1 behaviour of failing later on a zero-volume Jacobian.
    """
    kept, moved = [], 0
    for (eid, pid, nodes) in model.raw_elems["BRICK"]:
        uniq = list(dict.fromkeys(nodes))         # distinct, order preserved
        if len(uniq) == 8:
            kept.append((eid, pid, nodes))
        elif len(uniq) == 4:
            model.raw_elems["TETRA4"].append((eid, pid, uniq))
            moved += 1
        else:
            log.error(
                f"/BRICK {eid}: degenerated brick with {len(uniq)} distinct "
                f"nodes (penta/pyramid) is not ported — use full hexas or "
                f"/TETRA4", "BRICK DEGEN")
    model.raw_elems["BRICK"] = kept
    if moved:
        log.info(f"     {moved} DEGENERATED /BRICK ELEMENT(S) CONVERTED "
                 f"TO /TETRA4")


# ----------------------------------------------------------------------------
# Elements: raw tuples -> ElementGroups with per-part slices
# ----------------------------------------------------------------------------

def build_element_groups(model: Model, log: MessageLog) -> None:
    """Convert the raw (id, part, nodes) tuples collected by the parsers
    into dense ElementGroups, **sorted by part** so that each part is a
    contiguous slice — the Python equivalent of the Fortran element
    *groups* (NGROUP blocks of same type/mat/prop), which lets the material
    law run vectorized on each slice."""
    _convert_degenerated_bricks(model, log)
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
# Material resolution: /FUNCT curve references, /FAIL attachment
# ----------------------------------------------------------------------------

def resolve_materials(model: Model, log: MessageLog) -> None:
    """Resolve everything a material references once the whole deck is
    read (deck order between /MAT, /FUNCT and /FAIL is free):

    * LAW36: pull the /FUNCT hardening curves into plain arrays in
      ``mat.params`` (curve_x/curve_y/curve_s + rates) so the Engine
      kernels never touch the function-table objects — the Fortran
      Starter does the same (curves are copied into the MLAW buffer);
    * /FAIL cards: attach each parsed FailureModel to its material.
    """
    for mat in model.materials.values():
        if mat.law != 36:
            continue
        cxs, cys, css = [], [], []
        ok = True
        for fid in mat.params["funct_ids"]:
            fct = model.functions.get(fid)
            if fct is None:
                log.error(f"/MAT/LAW36/{mat.id}: function {fid} not defined",
                          "MAT CHECK")
                ok = False
                continue
            if np.any(fct.x < 0.0):
                log.error(f"/MAT/LAW36/{mat.id}: curve {fid} has negative "
                          f"plastic-strain abscissae", "MAT CHECK")
                ok = False
            if fct.eval(0.0) <= 0.0:
                log.error(f"/MAT/LAW36/{mat.id}: curve {fid} gives a "
                          f"non-positive initial yield stress", "MAT CHECK")
                ok = False
            cxs.append(fct.x.copy())
            cys.append(fct.y.copy())
            css.append(fct.slope.copy())
        if ok:
            mat.params["curve_x"] = cxs
            mat.params["curve_y"] = cys
            mat.params["curve_s"] = css
            mat.params["rates"] = np.asarray(mat.params["rates"], dtype=float)

    for mat_id, fm, source in model.raw_fails:
        mat = model.materials.get(mat_id)
        if mat is None:
            log.error(f"/FAIL/{fm.type}/{mat_id}: material {mat_id} not "
                      f"defined", source)
            continue
        if mat.law == 1:
            log.warning(f"/FAIL/{fm.type}/{mat_id}: attached to elastic "
                        f"LAW1 — no plastic strain ever accumulates, the "
                        f"criterion will never trigger", source)
        if mat.fail is not None:
            log.warning(f"/FAIL/{fm.type}/{mat_id}: material already has a "
                        f"/FAIL card — replaced", source)
        mat.fail = fm

    # /EOS attachment (M6, same free-order pattern as /FAIL): the EOS
    # replaces the law's pressure for SOLID elements — only meaningful
    # for laws whose deviatoric response is pressure-independent
    for mat_id, es, source in model.raw_eos:
        mat = model.materials.get(mat_id)
        if mat is None:
            log.error(f"/EOS/{es.kind}/{mat_id}: material {mat_id} not "
                      f"defined", source)
            continue
        if mat.law not in (1, 2, 36):
            log.error(f"/EOS/{es.kind}/{mat_id}: an EOS can only attach "
                      f"to LAW1/LAW2/LAW36 (isotropic laws with a "
                      f"pressure-independent deviator), got LAW{mat.law}",
                      source)
            continue
        if mat.eos is not None:
            log.warning(f"/EOS/{es.kind}/{mat_id}: material already has "
                        f"an /EOS card — replaced", source)
        es.rho0 = mat.rho0
        mat.eos = es

    # /FAIL/JOHNSON D5 needs the material's adiabatic temperature (M6)
    for mat in model.materials.values():
        if (mat.fail is not None and mat.fail.type == "JOHNSON"
                and mat.fail.params.get("D5", 0.0) != 0.0
                and "mT" not in mat.params):
            log.warning(f"/FAIL/JOHNSON/{mat.id}: D5 given but the "
                        f"material has no thermal card (LAW2 card 6) — "
                        f"D5 term dropped", "FAIL CHECK")
            mat.fail.params["D5"] = 0.0


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


def _free_faces_of_bricks(model: Model, part_ids: List[int]):
    """Outer (free) faces of the given solid parts: faces used by exactly
    one element. Fortran: the surface-from-part extraction of
    starter/source/model/sets/. Returns (faces (n,4), parent element rows
    in model.bricks (n,)) — the provenance is what lets contact drop the
    faces of /FAIL-deleted elements (M3<->M4 interaction)."""
    from ..elements.solid_hexa8 import _FACES
    g = model.bricks
    if g is None:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    mask = np.isin(g.state["part_ids"], part_ids)
    erow = np.where(mask)[0]                                # rows in group
    conn = g.conn[mask]
    faces = conn[:, _FACES.reshape(-1)].reshape(-1, 4)      # (nelem*6, 4)
    owner = np.repeat(erow, 6)                              # face -> element
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    free = counts[inverse] == 1
    return faces[free], owner[free]


def _free_faces_of_tetras(model: Model, part_ids: List[int]):
    """Free triangular faces of /TETRA4 parts, as degenerate 4-node
    segments (3rd node repeated — Radioss triangle-segment convention).
    Returns (faces (n,4), parent element rows in model.tetras (n,))."""
    from ..elements.solid_tetra4 import _FACES
    g = model.tetras
    if g is None:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    mask = np.isin(g.state["part_ids"], part_ids)
    erow = np.where(mask)[0]
    conn = g.conn[mask]
    faces = conn[:, _FACES.reshape(-1)].reshape(-1, 3)      # (nelem*4, 3)
    owner = np.repeat(erow, 4)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    free = counts[inverse] == 1
    faces = faces[free]
    return np.column_stack([faces, faces[:, 2]]), owner[free]  # n4 = n3


def resolve_surfaces(model: Model, log: MessageLog) -> None:
    """/SURF content -> (nseg, 4) node-index arrays + per-segment
    provenance (parent element group/row, see Surface docstring)."""
    for s in model.surfaces.values():
        segs: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []   # parallel provenance pieces
        elems: List[np.ndarray] = []

        def _add(seg_arr, gtype, elem_rows):
            segs.append(seg_arr)
            gtypes.append(np.full(len(seg_arr), gtype, dtype="<U8"))
            elems.append(np.asarray(elem_rows, dtype=np.int64))

        for row in s.seg_nodes:
            try:
                # explicit /SURF/SEG segments have no parent element: they
                # are never dropped by element deletion (gtype '')
                _add(model.node_indices(row)[None, :], "", [-1])
            except KeyError as exc:
                log.error(f"/SURF/{s.id}: unknown node id {exc}",
                          "SURFACE CHECK")
        if s.part_ids:
            # shell parts: every shell element is a segment
            if model.shells is not None:
                mask = np.isin(model.shells.state["part_ids"], s.part_ids)
                if np.any(mask):
                    _add(model.shells.conn[mask], "shells", np.where(mask)[0])
            # 3-node shell parts: triangle segments (3rd node repeated)
            if model.sh3n is not None:
                mask = np.isin(model.sh3n.state["part_ids"], s.part_ids)
                if np.any(mask):
                    c3 = model.sh3n.conn[mask]
                    _add(np.column_stack([c3, c3[:, 2]]), "sh3n",
                         np.where(mask)[0])
            # solid parts: free outer faces (with their parent element)
            ff, fo = _free_faces_of_bricks(model, s.part_ids)
            if len(ff):
                _add(ff, "bricks", fo)
            ft, to = _free_faces_of_tetras(model, s.part_ids)
            if len(ft):
                _add(ft, "tetras", to)
        if segs:
            s.segments = np.vstack(segs)
            s.seg_gtype = np.concatenate(gtypes)
            s.seg_elem = np.concatenate(elems)
        else:
            s.segments = np.zeros((0, 4), dtype=np.int64)
            s.seg_gtype = np.zeros(0, dtype="<U8")
            s.seg_elem = np.zeros(0, dtype=np.int64)
        if s.segments.shape[0] == 0:
            log.warning(f"/SURF/{s.id} '{s.title}' has no segments",
                        "SURFACE CHECK")


# 4-node segment -> its 4 edges; a triangle segment (n4 = n3) yields the
# degenerate edge (n3, n3), filtered out below.
_SEG_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def resolve_lines(model: Model, log: MessageLog) -> None:
    """/LINE content -> (nseg, 2) edge node-index arrays + provenance.
    Must run AFTER resolve_surfaces (LINE/SURF reads resolved segments).

    Fortran: hm_read_lines.F builds IGRSLIN the same two ways (from a
    surface or from explicit segments)."""
    for ln in model.lines.values():
        edges: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []
        elems: List[np.ndarray] = []
        for sid in ln.surf_ids:
            surf = model.surfaces.get(sid)
            if surf is None or surf.segments is None:
                log.error(f"/LINE/{ln.id}: surface {sid} not defined",
                          "LINE CHECK")
                continue
            e = surf.segments[:, _SEG_EDGES.reshape(-1)].reshape(-1, 2)
            own_g = np.repeat(surf.seg_gtype, 4)
            own_e = np.repeat(surf.seg_elem, 4)
            keep = e[:, 0] != e[:, 1]        # drop degenerate triangle edge
            e, own_g, own_e = e[keep], own_g[keep], own_e[keep]
            # each interior edge appears twice (once per adjacent segment):
            # keep one copy — for contact both copies are identical springs
            _, first = np.unique(np.sort(e, axis=1), axis=0,
                                 return_index=True)
            edges.append(e[first])
            gtypes.append(own_g[first])
            elems.append(own_e[first])
        for row in ln.seg_nodes:
            try:
                edges.append(model.node_indices(row)[None, :])
                gtypes.append(np.array([""], dtype="<U8"))
                elems.append(np.array([-1], dtype=np.int64))
            except KeyError as exc:
                log.error(f"/LINE/{ln.id}: unknown node id {exc}",
                          "LINE CHECK")
        if edges:
            ln.segments = np.vstack(edges)
            ln.seg_gtype = np.concatenate(gtypes)
            ln.seg_elem = np.concatenate(elems)
        else:
            ln.segments = np.zeros((0, 2), dtype=np.int64)
            ln.seg_gtype = np.zeros(0, dtype="<U8")
            ln.seg_elem = np.zeros(0, dtype=np.int64)
        if ln.segments.shape[0] == 0:
            log.warning(f"/LINE/{ln.id} '{ln.title}' has no edges",
                        "LINE CHECK")


# ----------------------------------------------------------------------------
# Element initialization + lumped mass (inimass)
# ----------------------------------------------------------------------------

def initialize_elements_and_mass(model: Model, log: MessageLog) -> None:
    """Run every kernel's init_group and assemble the lumped nodal mass
    (and rotational inertia, from shells). Also applies /ADMAS and
    /INIVEL (TRA and AXIS)."""
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

    # /ADMAS (M5): non-structural mass, added BEFORE the massless-node
    # check so a standalone node + /ADMAS is a legitimate free point mass
    for am in model.admas:
        g = model.node_groups.get(am.grnod_id)
        if g is None or g.node_idx is None:
            log.error(f"/ADMAS/{am.id}: unknown node group {am.grnod_id}",
                      "ADMAS CHECK")
            continue
        model.mass[g.node_idx] += am.mass

    # /INIVEL
    for iv in model.inivel:
        g = model.node_groups.get(iv.grnod_id)
        if g is None or g.node_idx is None:
            log.error(f"/INIVEL/{iv.id}: unknown node group {iv.grnod_id}",
                      "INIVEL CHECK")
            continue
        if iv.kind == "AXIS":
            # rigid-rotation velocity field: v += omega * d x (x0 - P)
            r = model.x0[g.node_idx] - iv.origin
            model.v[g.node_idx] += iv.omega * np.cross(iv.axis, r)
        else:
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


# ----------------------------------------------------------------------------
# Rigid bodies (/RBODY, /RBE2): mass, COG, inertia tensor  (M5)
# ----------------------------------------------------------------------------

def initialize_rigid_bodies(model: Model, log: MessageLog) -> None:
    """Compute each rigid body's total mass, center of gravity and inertia
    tensor from its slave nodes, and resolve the master node.

    Fortran origin: ``starter/source/constraints/general/rbody/rbyini.F``
    (and ``inirby``), which assemble exactly these quantities into the
    RBY buffer. The theory is the discrete rigid body: the slaves are
    point masses m_s at positions x_s (plus the isotropic lumped nodal
    inertias I_s that shells contribute), so

        M  = sum m_s (+ added mass)
        xg = sum m_s x_s / sum m_s
        J  = sum m_s (|r|^2 E - r r^T) + sum I_s E + diag(Jadd),
             r = x_s - xg

    Notes baked in:

    * an /RBE2 master is a structural node: its own mass/position joins
      the sums; an /RBODY master usually carries no mass (frozen by the
      mass check) and is EXCLUDED from them — with Icog=1 (default) it is
      *relocated* to the computed COG, the Radioss ICoG behaviour;
    * added mass/inertia act AT the COG (they shift nothing);
    * element deletion (/FAIL) never changes nodal masses (the deleted
      element's mass stays, see the /FAIL plumbing) — so M, xg, J are
      computed ONCE here and stay exact for the whole run;
    * slaves that are massless standalone nodes are tolerated (they are
      carried kinematically) but contribute nothing to M/J — a body needs
      at least some real mass;
    * a near-singular inertia tensor (all slave mass on one line) is
      regularized with a small isotropic term and flagged: the rotation
      rate about the mass line is then meaningless but stays bounded.
    """
    seen = np.zeros(model.numnod, dtype=bool)
    for rb in model.rbodies:
        who = f"/{rb.kind}/{rb.id}"
        try:
            rb.master = model.node_index(rb.master_id)
        except KeyError:
            log.error(f"{who}: unknown master node {rb.master_id}",
                      "RBODY CHECK")
            continue
        g = model.node_groups.get(rb.grnod_id)
        if g is None or g.node_idx is None or g.node_idx.size == 0:
            log.error(f"{who}: slave node group {rb.grnod_id} is missing "
                      f"or empty", "RBODY CHECK")
            continue
        rb.slaves = g.node_idx[g.node_idx != rb.master]

        # one kinematic condition per node: overlapping bodies are an error
        body_nodes = np.concatenate([[rb.master], rb.slaves])
        if np.any(seen[body_nodes]):
            log.error(f"{who}: node(s) already belong to another rigid "
                      f"body", "RBODY CHECK")
            continue
        seen[body_nodes] = True

        # mass sums: slaves + the master if it is structural (real mass).
        # Frozen (1e30) masses are the mass-check placeholder for nodes no
        # element references — they carry NO physical mass.
        m = model.mass[rb.slaves].copy()
        m[m >= 1e29] = 0.0
        mm = model.mass[rb.master]
        m_master = mm if mm < 1e29 else 0.0
        msum = float(m.sum()) + m_master
        if msum + rb.added_mass <= 0.0:
            log.error(f"{who}: rigid body has no mass (give the slaves "
                      f"element mass or /ADMAS, or set the Mass field)",
                      "RBODY CHECK")
            continue
        if msum > 0.0:
            xg = (m[:, None] * model.x0[rb.slaves]).sum(axis=0)
            xg = (xg + m_master * model.x0[rb.master]) / msum
        else:
            xg = model.x0[rb.slaves].mean(axis=0)   # massless: geometric

        # inertia tensor about xg (point masses + isotropic nodal inertias)
        r = model.x0[rb.slaves] - xg
        r2 = np.einsum("nb,nb->n", r, r)
        J = (np.einsum("n,nb,nc->bc", m, r, r) * -1.0
             + np.eye(3) * float((m * r2).sum()))
        rm = model.x0[rb.master] - xg
        J += m_master * (np.eye(3) * float(rm @ rm) - np.outer(rm, rm))
        inert = model.inertia[rb.slaves]
        J += np.eye(3) * float(inert.sum())
        J += np.diag(rb.jadd if rb.jadd is not None else np.zeros(3))

        # regularize a singular tensor (collinear point masses): the spin
        # about the mass line has no physics — keep it bounded, warn once
        lam = np.linalg.eigvalsh(J)
        if lam[0] < 1e-8 * max(lam[2], 1e-30):
            J += np.eye(3) * max(1e-8 * lam[2], 1e-30)
            log.warning(f"{who}: (near-)singular inertia tensor — slave "
                        f"masses are collinear; the spin about that line "
                        f"is regularized", "RBODY CHECK")

        rb.mass_total = msum + rb.added_mass
        rb.xg = xg
        rb.J = J

        # ICoG = 1: relocate the master node to the COG (Radioss default)
        if rb.icog == 1 and rb.kind == "RBODY":
            model.x0[rb.master] = xg
            model.x[rb.master] = xg
        # the added mass physically rides the body: hang it on the master
        # so the Engine's nodal KE/momentum ledgers see it move
        if rb.added_mass > 0.0:
            if model.mass[rb.master] >= 1e29:      # was frozen-massless
                model.mass[rb.master] = rb.added_mass
            else:
                model.mass[rb.master] += rb.added_mass

        pj = np.linalg.eigvalsh(rb.J)
        log.info(f"     {who}: {len(rb.slaves)} SLAVE NODE(S), MASS = "
                 f"{rb.mass_total:12.5E}, COG = {xg[0]:12.5E} "
                 f"{xg[1]:12.5E} {xg[2]:12.5E}")
        log.info(f"       PRINCIPAL INERTIA . . . . : {pj[0]:12.5E} "
                 f"{pj[1]:12.5E} {pj[2]:12.5E}")
