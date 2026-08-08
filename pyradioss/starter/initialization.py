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
from ..input import prop_reader
from ..model.entities import Material
from ..model.model import ElementGroup, Model

# element type name -> (attr on Model, nodes per element, required prop type)
_ETYPES = {
    "BRICK": ("bricks", 8, 14),
    "QUAD": ("quads", 4, 14),
    "TETRA4": ("tetras", 4, 14),
    "SHELL": ("shells", 4, 1),
    "SH3N": ("sh3n", 3, 1),
    "TRUSS": ("trusses", 2, 2),
    "SPRING": ("springs", 2, 4),
    "BEAM": ("beams", 3, 3),
    "SHEL16": ("shel16s", 16, 20),
}

#: the "fictitious material law for spring elements" the reference assigns
#: to a /PART with mat_ID 0 on a spring property (hm_read_part.F): springs
#: read their mass/stiffness from the /PROP, never a material, so this is a
#: bare inert placeholder (never inactive — the Engine must not refuse it).
_SPRING_MAT_SINGLETON: Material = None       # type: ignore[assignment]


def _fictitious_spring_material() -> Material:
    global _SPRING_MAT_SINGLETON
    if _SPRING_MAT_SINGLETON is None:
        _SPRING_MAT_SINGLETON = Material(
            id=0, law=-1, rho0=0.0, title="fictitious spring material",
            params={"E": 0.0, "nu": 0.0})
    return _SPRING_MAT_SINGLETON


# ----------------------------------------------------------------------------
# Degenerated bricks: repeated nodes -> tetra conversion / clear rejection
# ----------------------------------------------------------------------------

def _convert_degenerated_bricks(model: Model, log: MessageLog) -> None:
    """Handle /BRICK cards with repeated node IDs (the classic Radioss way
    of writing lower-order solids in brick format).

    Fortran origin: the reference does NOT convert a degenerated brick into
    another element type — it runs it through the SAME 8-node hexa kernel
    with the repeated-node connectivity exactly as written, and merely
    COUNTS the collapsed nodes for a few local corrections.  That counter is
    ``engine/source/elements/solid/solide/degenes8.F`` (DEGENES8), whose
    whole body is "for each node, is it repeated elsewhere in IXS(1..8)?"
    reduced to ``IDEGE(I) = <repeats>/2``; its consumers are the solid
    drivers themselves (sforc3.F, s8eforc3.F, s8zforc3.F, szforc3.F ...),
    which use IDEGE only to pick the small-strain branch, to skip an
    optimisation, or to correct the characteristic length
    (``sdlen_dege.F``: "DELTAX correction for degenerated element").  The
    connectivity itself is never rewritten, and the Starter's element count
    is unchanged: RD-V-0700's HEXA_DEGE deck — 40 bricks, every one a
    6-distinct-node collapsed wedge — reports ``NUMELS: NUMBER OF 3D SOLID
    ELEMENTS . . . 40`` in the reference .out.

    So this port keeps a degenerated brick AS a brick (M39):

    * **8 distinct** — an ordinary hexa;
    * **5, 6 or 7 distinct** — a collapsed hexa: the connectivity is passed
      through UNCHANGED (repeats included) and the standard 8-node kernel
      runs it.  This is not an approximation of a penta/pyramid element —
      it *is* how the reference represents one.  The isoparametric map
      simply degenerates along the collapsed edge; the coincident nodes'
      shape functions sum, so the mass lumping ``mass/8`` per corner gives
      a repeated node its 2/8 share (exactly the reference's lumping) and
      the centroid Jacobian yields the wedge volume.  The classic pattern
      is the wedge ``n1 n1 n3 n4 n5 n5 n7 n8`` (bottom and top faces each
      collapsed along one edge), which is what HEXA_DEGE and RD_V_0240's
      Modele_HEXA_P14 are built from.  Before M39 these were REFUSED
      ("degenerated brick ... is not ported"), which cost the three
      official element-verification decks c12/c18/c27 their whole run —
      and, since every element in HEXA_DEGE is degenerate, also produced
      the follow-on "model has no elements";
    * **4 distinct** (e.g. ``n1 n2 n3 n3 n5 n5 n5 n5``) — promoted to a
      genuine /TETRA4.  Kept from the earlier port: for the FULLY collapsed
      pattern the constant-strain tetra is the better element (a collapsed
      hexa leaves zero-volume sub-shapes in the hourglass base vectors),
      and it is what the port's own /TETRA4 kernel is for.  Distinct nodes
      are taken in order of first appearance, which maps every standard
      collapse pattern onto the positively-oriented tetra (re-checked at
      element init);
    * **fewer than 4 distinct** — a brick collapsed past any 3-D shape has
      no volume to integrate; that stays a hard error.
    """
    kept, moved, degen = [], 0, 0
    for (eid, pid, nodes) in model.raw_elems["BRICK"]:
        uniq = list(dict.fromkeys(nodes))         # distinct, order preserved
        if len(uniq) == 8:
            kept.append((eid, pid, nodes))
        elif len(uniq) == 4:
            model.raw_elems["TETRA4"].append((eid, pid, uniq))
            moved += 1
        elif len(uniq) in (5, 6, 7):
            # collapsed hexa — connectivity AS WRITTEN, repeats included
            kept.append((eid, pid, nodes))
            degen += 1
        else:
            log.error(
                f"/BRICK {eid}: degenerated brick with only {len(uniq)} "
                f"distinct node(s) — collapsed past any 3-D shape, it has "
                f"no volume", "BRICK DEGEN")
    model.raw_elems["BRICK"] = kept
    if moved:
        log.info(f"     {moved} DEGENERATED /BRICK ELEMENT(S) CONVERTED "
                 f"TO /TETRA4")
    if degen:
        log.info(f"     {degen} DEGENERATED /BRICK ELEMENT(S) RUN AS "
                 f"COLLAPSED HEXA (penta/pyramid)")


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
                conn[k] = [model._id2idx[int(n)] if n != 0 else -1 for n in nodes]
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
            if prop is None:
                log.error(f"/PART/{pid}: property {part.prop_id} not defined",
                          "PART CHECK")
            elif not prop_reader.prop_type_ok(req_prop, prop):
                log.error(f"/PART/{pid}: /{etype} elements need /PROP/TYPE"
                          f"{req_prop}, got TYPE{prop.type}", "PART CHECK")
            # material resolution incl. the mat_ID = 0 rule (hm_read_part.F,
            # M38): a spring property (TYPE4/8/13...) may legally carry
            # mat_ID 0 — a fictitious material is assigned for the spring
            # elements; every material-required property (solids, shells,
            # trusses, beams) still needs a defined material.
            if mat is None:
                if part.mat_id == 0 and prop is not None \
                        and not prop_reader.material_required(prop.type):
                    mat = _fictitious_spring_material()
                else:
                    log.error(f"/PART/{pid}: material {part.mat_id} not "
                              f"defined", "PART CHECK")
            if mat is not None and prop is not None:
                slices.append((slice(start, end), mat, prop))
            part_idx[sel] = model.parts_list.index(part) if part in \
                model.parts_list else 0
            start = end

        group = ElementGroup(ids=ids, conn=conn, part=part_idx)
        group.state["slices"] = slices
        group.state["part_ids"] = part_ids
        setattr(model, attr, group)

    _dispatch_solid_formulations(model, log)
    _dispatch_shell_formulations(model, log)
    _dispatch_sh3n_formulations(model, log)


def _subset_element_group(src: ElementGroup, mask: np.ndarray) -> ElementGroup:
    """Row subset of an ElementGroup, slices rebuilt. Parts are
    contiguous after the part sort and each part routes WHOLLY to one
    formulation (the Ishell lives on the /PROP), so taking whole slices
    preserves both the ordering and the per-part contiguity."""
    g = ElementGroup(ids=src.ids[mask], conn=src.conn[mask],
                     part=src.part[mask])
    slices = []
    start = 0
    for sl, mat, prop in src.state["slices"]:
        cnt = int(mask[sl].sum())
        if cnt:
            slices.append((slice(start, start + cnt), mat, prop))
            start += cnt
    g.state["slices"] = slices
    g.state["part_ids"] = src.state["part_ids"][mask]
    return g


def _dispatch_solid_formulations(model: Model, log: MessageLog) -> None:
    """Solid element-technology dispatch (M64): split /BRICK parts whose
    /PROP/SOLID Isolid selects a dedicated formulation kernel out of the
    generic Belytschko-Tsay group, per elements.SOLID_ISOLID_GROUPS.
    Decks without such parts are left alone."""
    from ..elements import SOLID_ISOLID_GROUPS
    src = model.bricks
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        isolid = int(prop.params.get("isolid", 0) or 0)
        gname = SOLID_ISOLID_GROUPS.get(isolid)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        log.info(f"     {int(mask.sum())} /BRICK ELEMENT(S) ROUTED TO THE "
                 f"{gname.split('_', 1)[1].upper()} FORMULATION KERNEL "
                 f"(Isolid dispatch)")
    model.bricks = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_shell_formulations(model: Model, log: MessageLog) -> None:
    """Shell element-technology dispatch (M41): split /SHELL parts whose
    /PROP/SHELL Ishell selects a dedicated formulation kernel out of the
    generic Belytschko-Tsay group, per elements.SHELL_ISHELL_GROUPS
    (12 = QBAT -> model.shells_qbat). Decks without such parts are left
    byte-identically alone (the split never runs)."""
    from ..elements import SHELL_ISHELL_GROUPS
    src = model.shells
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        ishell = int(prop.params.get("ishell", 0) or 0)
        gname = SHELL_ISHELL_GROUPS.get(ishell)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        log.info(f"     {int(mask.sum())} /SHELL ELEMENT(S) ROUTED TO THE "
                 f"{gname.split('_', 1)[1].upper()} FORMULATION KERNEL "
                 f"(Ishell dispatch)")
    model.shells = _subset_element_group(src, keep) if keep.any() else None


def _dispatch_sh3n_formulations(model: Model, log: MessageLog) -> None:
    """SH3N element-technology dispatch: split /SH3N parts whose
    /PROP/SHELL Ish3n selects a dedicated formulation kernel out of the
    generic shell_tri3 group, per elements.SH3N_ISHELL_GROUPS.
    Decks without such parts are left alone."""
    from ..elements import SH3N_ISHELL_GROUPS
    src = model.sh3n
    if src is None or not src.n:
        return
    masks: Dict[str, np.ndarray] = {}
    for sl, mat, prop in src.state["slices"]:
        ish3n = int(prop.params.get("ish3n", 0) or 0)
        gname = SH3N_ISHELL_GROUPS.get(ish3n)
        if gname is not None:
            masks.setdefault(gname, np.zeros(src.n, dtype=bool))[sl] = True
    if not masks:
        return
    keep = np.ones(src.n, dtype=bool)
    for gname, mask in masks.items():
        keep &= ~mask
        setattr(model, gname, _subset_element_group(src, mask))
        log.info(f"     {int(mask.sum())} /SH3N ELEMENT(S) ROUTED TO THE "
                 f"{gname.split('_', 1)[1].upper()} FORMULATION KERNEL "
                 f"(Ish3n dispatch)")
    model.sh3n = _subset_element_group(src, keep) if keep.any() else None


# ----------------------------------------------------------------------------
# Material resolution: /FUNCT curve references, /FAIL attachment
# ----------------------------------------------------------------------------

def resolve_materials(model: Model, log: MessageLog) -> None:
    """Resolve everything a material references once the whole deck is
    read (deck order between /MAT, /FUNCT and /FAIL is free):

    * LAW36: pull the /FUNCT hardening curves into plain arrays in
      ``mat.params`` (curve_x/curve_y/curve_s + rates) so the Engine
      kernels never touch the function-table objects — the Fortran
      Starter does the same (curves are copied into the MLAW buffer).
      The per-curve Fscale_i (``params['yfac']``, M40) is baked into the
      copied ordinates AND slopes here: the reference applies YFAC at
      every engine evaluation (sigeps36.F ``Y1*YFAC(I,1)``,
      ``DYDX1*YFAC(I,1)`` — both value and derivative, before the
      strain-rate interpolation), which is algebraically identical to
      scaling the stored curve once;
    * /FAIL cards: attach each parsed FailureModel to its material.
    """
    for mat in model.materials.values():
        if mat.law != 36 or getattr(mat, "inactive", False):
            continue
        cxs, cys, css = [], [], []
        ok = True
        yfac = list(mat.params.get("yfac") or [])
        yfac += [1.0] * (len(mat.params["funct_ids"]) - len(yfac))
        for fid, yf in zip(mat.params["funct_ids"], yfac):
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
            if fct.eval(0.0) * yf <= 0.0:
                log.error(f"/MAT/LAW36/{mat.id}: curve {fid} gives a "
                          f"non-positive initial yield stress "
                          f"(Fscale={yf:g})", "MAT CHECK")
                ok = False
            cxs.append(fct.x.copy())
            cys.append(fct.y * yf)
            css.append(fct.slope * yf)
        if ok:
            mat.params["curve_x"] = cxs
            mat.params["curve_y"] = cys
            mat.params["curve_s"] = css
            mat.params["rates"] = np.asarray(mat.params["rates"], dtype=float)

    # LAW81 (M37 pack 2): resolve the four optional /FUNCT references
    # (K, G scale vs eps_p_vol; cohesion vs eps_p_dev; cap pressure Pb vs
    # eps_p_vol) into plain (x, y) arrays for the kernel, exactly like
    # LAW36 above.  Also surface the documented porosity cut.
    for mat in model.materials.values():
        if mat.law != 81 or getattr(mat, "inactive", False):
            continue
        if mat.params.pop("law81_porosity_ignored", False):
            log.warning(f"/MAT/LAW81/{mat.id}: pore-water block (Kw, P0r, "
                        f"sat0...) is not ported — porosity IGNORED "
                        f"(law81_druckerprager documented cut)",
                        "MAT CHECK")
        for fid, name in zip(mat.params.get("funct81_ids", []),
                             ("k", "g", "c", "pb")):
            if fid == 0:
                continue
            fct = model.functions.get(fid)
            if fct is None:
                log.error(f"/MAT/LAW81/{mat.id}: function {fid} not "
                          f"defined", "MAT CHECK")
                continue
            mat.params["curve81_" + name] = (fct.x.copy(), fct.y.copy())

    # M37 pack 1: LAW70 (loading/unloading tables + the law70_upd.F
    # derived constants), LAW35 (pressure curve) and LAW44 (tabulated
    # yield) resolve their /FUNCT references the same deck-order-free way
    for mat in model.materials.values():
        if getattr(mat, "inactive", False):
            continue
        if mat.law == 70:
            from ..materials import law70_tabfoam
            law70_tabfoam.resolve(mat, model, log)
        elif mat.law == 35:
            from ..materials import law35_kelvinmax
            law35_kelvinmax.resolve(mat, model, log)
        elif mat.law == 44:
            from ..materials import law44_cowper
            law44_cowper.resolve(mat, model, log)

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
        if getattr(mat, "inactive", False):
            # M37: an /EOS on a parsed-but-not-implemented law is kept as
            # part of the record (parse-clean) — the Engine refuses the
            # material anyway, so the EOS never acts
            es.rho0 = mat.rho0
            mat.eos = es
            continue
        if mat.law not in (1, 2, 36, 44, 999):
            log.error(f"/EOS/{es.kind}/{mat_id}: an EOS can only attach "
                      f"to LAW1/LAW2/LAW36/LAW44/GAS (isotropic laws "
                      f"with a pressure-independent deviator), got "
                      f"LAW{mat.law}", source)
            continue
        if mat.eos is not None:
            log.warning(f"/EOS/{es.kind}/{mat_id}: material already has "
                        f"an /EOS card — replaced", source)
        if mat.law == 999 and mat.rho0 == 0.0 \
                and es.params.get("rho0_card", 0.0) > 0.0:
            # /MAT/GAS has no density card of its own — the /EOS/IDEAL-GAS
            # RHO_0 field supplies it (M37 pack 1, materials/mat_gas.py)
            mat.rho0 = es.params["rho0_card"]
        es.rho0 = mat.rho0
        mat.eos = es

    # M37 pack 1: /MAT/GAS element-path resolution — build the IDEAL-GAS
    # EOS from programmatic P0/T0/RHO0 params when no /EOS card gave one
    # (materials/mat_gas.resolve_gas; harmless for injector-only gases)
    for mat in model.materials.values():
        if mat.law == 999 and not getattr(mat, "inactive", False):
            from ..materials import mat_gas
            mat_gas.resolve_gas(mat, log)

    # /ALE/MAT, /EULER/MAT, /HEAT/MAT parse-only notes (M37): attach to
    # the material's params (deck order is free, like /FAIL and /EOS) —
    # accepted and remembered, no physics acts on them
    for kind, mat_id, params, source in getattr(model, "raw_mat_notes", []):
        mat = model.materials.get(mat_id)
        if mat is None:
            log.warning(f"/{kind}/{mat_id}: material {mat_id} not defined "
                        f"— note ignored", source)
            continue
        mat.params[kind.replace("/", "_").lower() + "_note"] = params

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
# Node groups, element groups and surfaces
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


# element-group family key -> model attributes it spans (GRBRIC = ALL
# solids, like the Fortran IGRBRIC over the whole IXS; BEAM edges use
# the two END nodes only — the orientation node N3 is no geometry)
_EGROUP_FAMILIES = {
    "SHEL": ("shells", "shells_qbat", "shells_qeph"),
    "SH3N": ("sh3n", "sh3n_dkt18"),
    "BRIC": ("bricks", "tetras"),
    "QUAD": ("quads",),
    "TRUS": ("trusses",),
    "BEAM": ("beams",),
    "SPRI": ("springs",),
}


def _elem_row_lookup(model: Model, attr: str):
    """user element id -> row map for one element group (cached)."""
    g = getattr(model, attr, None)
    if g is None:
        return {}
    cache = g.state.get("_id2row")
    if cache is None:
        cache = {int(e): k for k, e in enumerate(g.ids)}
        g.state["_id2row"] = cache
    return cache


def _fixpoint(pending: dict, try_resolve, log: MessageLog, kind: str,
              on_cycle) -> None:
    """The iterative group-of-groups resolution of hm_grogro.F /
    hm_grogronod.F / hm_read_surfsurf.F: repeat passes resolving every
    item whose references are all resolved; when a full pass makes no
    progress the leftovers form reference CYCLES — reported as errors
    (the upstream 'ITER > N' MSGID 176/189 stop) and force-resolved
    empty so downstream consumers keep working."""
    guard = len(pending) + 1
    for _ in range(guard):
        progressed = False
        for gid in list(pending):
            if try_resolve(pending[gid]):
                del pending[gid]
                progressed = True
        if not pending:
            return
        if not progressed:
            break
    for gid, obj in pending.items():
        log.error(f"/{kind}/{gid}: circular {kind} group-of-groups "
                  f"reference (involves "
                  f"{sorted(pending)}) — group left empty", "GROUP CHECK")
        on_cycle(obj)


def resolve_entity_groups(model: Model, log: MessageLog) -> None:
    """ELEMENT groups (/GRSHEL, /GRSH3N, /GRBRIC, ..., /GRPART) ->
    resolved member rows (Fortran hm_lecgre.F + hm_grogro.F).

    Each group ends with ``members`` = [(model attr, row ndarray)]
    (family 'PART': ``part_ids_resolved``).  Group-of-group references
    are signed: negative ids REMOVE the referenced group's members,
    and removal wins over addition (the upstream BUFTMP = -1 rule);
    cycles are detected by the fixpoint guard."""
    for family, groups in model.egroups.items():
        attrs = _EGROUP_FAMILIES.get(family, ())
        if family != "PART" and not attrs and any(
                g.elem_ids or g.part_ids for g in groups.values()):
            log.warning(f"/GR{family}: element family not ported — "
                        f"groups resolve empty", "GROUP CHECK")

        def _base(g):
            """direct + part content as {attr: set(rows)} / part set."""
            if family == "PART":
                for pid in g.part_ids:
                    if pid not in model.parts:
                        log.error(f"/GRPART/{g.id}: unknown part {pid}",
                                  "GROUP CHECK")
                return set(p for p in g.part_ids if p in model.parts)
            rows = {a: set() for a in attrs}
            if not attrs:            # family without a ported element
                return rows
            missing = []
            for eid in g.elem_ids:
                for a in attrs:
                    r = _elem_row_lookup(model, a).get(eid)
                    if r is not None:
                        rows[a].add(r)
                        break
                else:
                    missing.append(eid)
            if missing:
                log.error(f"/GR{family}/{g.id}: unknown element id(s) "
                          f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                          "GROUP CHECK")
            if g.part_ids:
                for a in attrs:
                    eg = getattr(model, a, None)
                    if eg is None:
                        continue
                    mask = np.isin(eg.state["part_ids"], g.part_ids)
                    rows[a].update(np.where(mask)[0].tolist())
            return rows

        resolved: dict = {}
        base_cache: dict = {}

        def _try(g):
            if g.id not in base_cache:   # once — errors not duplicated
                base_cache[g.id] = _base(g)
            add = (set(base_cache[g.id]) if family == "PART"
                   else {a: set(v) for a, v in base_cache[g.id].items()})
            rem = set() if family == "PART" else {a: set() for a in attrs}
            for ref in g.group_ids:
                other = groups.get(abs(ref))
                if other is None:
                    log.warning(f"/GR{family}/{g.id}: unknown group "
                                f"{abs(ref)} — ignored", "GROUP CHECK")
                    continue
                if abs(ref) not in resolved:
                    return False                       # defer this pass
                if family == "PART":
                    (add if ref > 0 else rem).update(resolved[abs(ref)])
                else:
                    for a in attrs:
                        tgt = add if ref > 0 else rem
                        tgt[a].update(resolved[abs(ref)][a])
            if family == "PART":
                final = add - rem
                g.part_ids_resolved = sorted(final)
                resolved[g.id] = final
            else:
                final = {a: add[a] - rem[a] for a in attrs}
                g.members = [
                    (a, np.array(sorted(final[a]), dtype=np.int64))
                    for a in attrs if final[a]]
                resolved[g.id] = final
            return True

        def _on_cycle(g):
            if family == "PART":
                g.part_ids_resolved = []
            else:
                g.members = []

        _fixpoint(dict(groups), _try, log, f"GR{family}", _on_cycle)
        for g in groups.values():
            n = (len(g.part_ids_resolved) if family == "PART"
                 else sum(len(r) for _, r in (g.members or [])))
            if n == 0:
                log.warning(f"/GR{family}/{g.id} '{g.title}' is empty",
                            "GROUP CHECK")


def _nodes_in_box(model: Model, box, log: MessageLog,
                  who: str) -> np.ndarray:
    """Node indices inside one /BOX volume (rdbox.F membership tests,
    boundaries inclusive)."""
    x = model.x0

    def _pt(node_id, fallback):
        if node_id:
            try:
                return x[model.node_index(node_id)]
            except KeyError:
                log.error(f"{who}: /BOX/{box.id} references unknown "
                          f"node {node_id}", "GROUP CHECK")
                return None
        return fallback

    if box.kind == "RECTA":
        cmin, cmax = box.corner_min, box.corner_max
        if box.node1:
            p1 = _pt(box.node1, None)
            p2 = _pt(box.node2, None)
            if p1 is None or p2 is None:
                return np.zeros(0, dtype=np.int64)
            cmin, cmax = np.minimum(p1, p2), np.maximum(p1, p2)
        inside = np.all((x >= cmin) & (x <= cmax), axis=1)
        return np.where(inside)[0]
    if box.kind == "SPHER":
        c = _pt(box.node1, box.p1)
        if c is None:
            return np.zeros(0, dtype=np.int64)
        d2 = np.einsum("nb,nb->n", x - c, x - c)
        return np.where(d2 <= (0.5 * box.diameter) ** 2)[0]
    # CYLIN: finite cylinder p1 -> p2, radius D/2 (INSIDE_CYLINDER)
    p1 = _pt(box.node1, box.p1)
    p2 = _pt(box.node2, box.p2)
    if p1 is None or p2 is None:
        return np.zeros(0, dtype=np.int64)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        log.error(f"{who}: /BOX/CYLIN/{box.id} has a zero-length axis",
                  "GROUP CHECK")
        return np.zeros(0, dtype=np.int64)
    a = axis / length
    d = (x - p1) @ a
    radial = x - p1 - d[:, None] * a
    r2 = np.einsum("nb,nb->n", radial, radial)
    inside = (d >= 0.0) & (d <= length) & (r2 <= (0.5 * box.diameter) ** 2)
    return np.where(inside)[0]


def resolve_node_groups(model: Model, log: MessageLog) -> None:
    """/GRNOD content -> dense node index arrays.

    M37: runs AFTER resolve_entity_groups and resolve_surfaces —
    /GRNOD/SURF takes the nodes of resolved surface segments
    (hm_surfnod.F) and /GRNOD/GR<elem> the nodes of resolved element
    groups (hm_elngr.F).  /GRNOD/GRNOD (hm_grogronod.F) resolves by
    iterative fixpoint with cycle detection; NEGATIVE references REMOVE
    the referenced group's nodes and removal wins over addition,
    whatever the order (the upstream BUFTMP = -1 convention).  Groups
    are stored sorted by node index, like the upstream sorted groups.
    """
    def _base(g) -> np.ndarray:
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
            idx.append(_nodes_in_box(model, box, log, f"/GRNOD/{g.id}"))
        for first, last, incr in g.gene_ranges:
            uid = model.node_ids
            mask = (uid >= first) & (uid <= last)
            if incr > 1:
                mask &= (uid - first) % incr == 0
            idx.append(np.where(mask)[0])
        for sid in g.surf_ids:
            surf = model.surfaces.get(sid)
            if surf is None or surf.segments is None:
                log.error(f"/GRNOD/{g.id}: surface {sid} missing or "
                          f"unresolved", "GROUP CHECK")
                continue
            if surf.segments.size:
                idx.append(np.unique(surf.segments))
        for family, gid in g.egroup_refs:
            eg = model.egroups.get(family, {}).get(gid)
            if eg is None:
                log.error(f"/GRNOD/{g.id}: unknown /GR{family} group "
                          f"{gid}", "GROUP CHECK")
                continue
            for attr, rows in (eg.members or []):
                idx.append(np.unique(getattr(model, attr).conn[rows]))
        return (np.unique(np.concatenate(idx)) if idx
                else np.zeros(0, dtype=np.int64))

    resolved: dict = {}
    base_cache: dict = {}

    def _try(g) -> bool:
        if g.id not in base_cache:       # once — errors not duplicated
            base_cache[g.id] = _base(g)
        add = [base_cache[g.id]]
        rem = [np.zeros(0, dtype=np.int64)]
        for ref in g.grnod_ids:
            other = model.node_groups.get(abs(ref))
            if other is None:
                # upstream MSGID 174: unknown reference is a WARNING
                log.warning(f"/GRNOD/{g.id}: unknown node group "
                            f"{abs(ref)} — ignored", "GROUP CHECK")
                continue
            if abs(ref) not in resolved:
                return False                           # defer this pass
            (add if ref > 0 else rem).append(resolved[abs(ref)])
        final = np.setdiff1d(np.unique(np.concatenate(add)),
                             np.unique(np.concatenate(rem)))
        g.node_idx = final
        resolved[g.id] = final
        return True

    def _on_cycle(g):
        g.node_idx = np.zeros(0, dtype=np.int64)

    _fixpoint(dict(model.node_groups), _try, log, "GRNOD", _on_cycle)
    for g in model.node_groups.values():
        if g.node_idx is None or g.node_idx.size == 0:
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
    provenance (parent element group/row, see Surface docstring).

    M37 additions: /SURF/GRSHEL and /SURF/GRSH3N (segments from element
    groups — runs AFTER resolve_entity_groups) and /SURF/SURF
    (surface-of-surfaces, hm_read_surfsurf.F): the referenced surfaces'
    segments are concatenated by iterative fixpoint with cycle
    detection; a NEGATIVE reference includes the surface with its
    segment node order REVERSED (n4 n3 n2 n1 — the normal flips),
    provenance carried along either way."""
    def _base(s):
        segs: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []   # parallel provenance pieces
        elems: List[np.ndarray] = []

        def _add(seg_arr, gtype, elem_rows):
            segs.append(seg_arr)
            gtypes.append(np.full(len(seg_arr), gtype, dtype="<U16"))
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
            # QBAT shell parts (Ishell=12 split group, M41): same segments
            if model.shells_qbat is not None:
                mask = np.isin(model.shells_qbat.state["part_ids"],
                               s.part_ids)
                if np.any(mask):
                    _add(model.shells_qbat.conn[mask], "shells_qbat",
                         np.where(mask)[0])
            # QEPH shell parts (Ishell=24 split group, M41): same segments
            if model.shells_qeph is not None:
                mask = np.isin(model.shells_qeph.state["part_ids"],
                               s.part_ids)
                if np.any(mask):
                    _add(model.shells_qeph.conn[mask], "shells_qeph",
                         np.where(mask)[0])
            # DKT18 shell parts (Ish3n=2 split group)
            if model.sh3n_dkt18 is not None:
                mask = np.isin(model.sh3n_dkt18.state["part_ids"],
                               s.part_ids)
                if np.any(mask):
                    c3 = model.sh3n_dkt18.conn[mask]
                    _add(np.column_stack([c3, c3[:, 2]]), "sh3n_dkt18",
                         np.where(mask)[0])
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
        # /SURF/GRSHEL | /SURF/GRSH3N: every element of the group (M37)
        for family, gid in s.egroup_refs:
            eg = model.egroups.get(family, {}).get(gid)
            if eg is None:
                log.error(f"/SURF/{s.id}: unknown /GR{family} group {gid}",
                          "SURFACE CHECK")
                continue
            for attr, rows in (eg.members or []):
                conn = getattr(model, attr).conn[rows]
                if conn.shape[1] == 3:              # triangles: n4 = n3
                    conn = np.column_stack([conn, conn[:, 2]])
                _add(conn, attr, rows)
        if segs:
            return (np.vstack(segs), np.concatenate(gtypes),
                    np.concatenate(elems))
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))

    resolved: dict = {}
    base_cache: dict = {}

    def _finish(s, triple):
        s.segments, s.seg_gtype, s.seg_elem = triple
        resolved[s.id] = triple

    def _try(s) -> bool:
        if s.id not in base_cache:       # once — errors not duplicated
            base_cache[s.id] = _base(s)
        pieces = [base_cache[s.id]]
        for ref in s.surf_ids:
            other = model.surfaces.get(abs(ref))
            if other is None:
                log.warning(f"/SURF/{s.id}: unknown surface {abs(ref)} — "
                            f"ignored", "SURFACE CHECK")
                continue
            if abs(ref) not in resolved:
                return False                        # defer this pass
            seg, gt, el = resolved[abs(ref)]
            if ref < 0:
                # negative id: node order reversed — the normal flips
                # (hm_read_surfsurf.F NODES(L,4..1))
                seg = seg[:, ::-1]
            pieces.append((seg, gt, el))
        _finish(s, (np.vstack([p[0] for p in pieces]),
                    np.concatenate([p[1] for p in pieces]),
                    np.concatenate([p[2] for p in pieces])))
        return True

    def _on_cycle(s):
        _finish(s, (np.zeros((0, 4), dtype=np.int64),
                    np.zeros(0, dtype="<U8"),
                    np.zeros(0, dtype=np.int64)))

    _fixpoint(dict(model.surfaces), _try, log, "SURF", _on_cycle)
    for s in model.surfaces.values():
        if s.segments.shape[0] == 0:
            log.warning(f"/SURF/{s.id} '{s.title}' has no segments",
                        "SURFACE CHECK")


# 4-node segment -> its 4 edges; a triangle segment (n4 = n3) yields the
# degenerate edge (n3, n3), filtered out below.
_SEG_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def _surface_edges(model: Model, ln, surf_ids, log: MessageLog):
    """All edges of the listed surfaces' segments, with provenance —
    the raw union LINEDGE starts from (degenerate triangle edges
    dropped).  Returns (edges (n,2), gtype (n,), elem (n,))."""
    edges, gtypes, elems = [], [], []
    for sid in surf_ids:
        surf = model.surfaces.get(sid)
        if surf is None or surf.segments is None:
            log.error(f"/LINE/{ln.id}: surface {sid} not defined",
                      "LINE CHECK")
            continue
        e = surf.segments[:, _SEG_EDGES.reshape(-1)].reshape(-1, 2)
        own_g = np.repeat(surf.seg_gtype, 4)
        own_e = np.repeat(surf.seg_elem, 4)
        keep = e[:, 0] != e[:, 1]        # drop degenerate triangle edge
        edges.append(e[keep])
        gtypes.append(own_g[keep])
        elems.append(own_e[keep])
    if not edges:
        return (np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
    return (np.vstack(edges), np.concatenate(gtypes),
            np.concatenate(elems))


def resolve_lines(model: Model, log: MessageLog) -> None:
    """/LINE content -> (nseg, 2) edge node-index arrays + provenance.
    Must run AFTER resolve_surfaces (LINE/SURF|EDGE read resolved
    segments).

    Fortran: hm_read_lines.F builds IGRSLIN from a surface (all unique
    edges — /LINE/SURF), from a surface's BORDER (/LINE/EDGE, M37:
    ``linedge.F`` keeps only edges used by exactly ONE segment of the
    union of the listed surfaces, removing interior edges entirely),
    from other lines (/LINE/LINE, M37: concatenation by fixpoint with
    cycle detection), from 1-D element parts (/LINE/PART, M37) and from
    explicit node pairs (/LINE/SEG)."""
    def _base(ln):
        edges: List[np.ndarray] = []
        gtypes: List[np.ndarray] = []
        elems: List[np.ndarray] = []
        if ln.surf_ids:
            # every unique edge: interior edges appear once per adjacent
            # segment — keep one copy (both copies are identical springs)
            e, gt, el = _surface_edges(model, ln, ln.surf_ids, log)
            if len(e):
                _, first = np.unique(np.sort(e, axis=1), axis=0,
                                     return_index=True)
                edges.append(e[first])
                gtypes.append(gt[first])
                elems.append(el[first])
        if ln.edge_surf_ids:
            # /LINE/EDGE: BORDER edges only — count over the union of
            # the listed surfaces' segments, keep count == 1 (linedge.F
            # 'removal of internal segments except borders'); stored
            # (lo, hi) like the upstream sorted pairs
            e, gt, el = _surface_edges(model, ln, ln.edge_surf_ids, log)
            if len(e):
                key = np.sort(e, axis=1)
                _, first, counts = np.unique(
                    key, axis=0, return_index=True, return_counts=True)
                border = first[counts == 1]
                edges.append(key[border])
                gtypes.append(gt[border])
                elems.append(el[border])
        for pid in ln.part_ids:
            # /LINE/PART: every 1-D element of the part is an edge
            # (beams: end nodes N1-N2 — the orientation node carries no
            # geometry)
            if pid not in model.parts:
                log.error(f"/LINE/{ln.id}: unknown part {pid}",
                          "LINE CHECK")
                continue
            found = False
            for attr in ("trusses", "springs", "beams"):
                eg = getattr(model, attr, None)
                if eg is None:
                    continue
                mask = np.isin(eg.state["part_ids"], [pid])
                if np.any(mask):
                    edges.append(eg.conn[mask][:, :2])
                    gtypes.append(np.full(int(mask.sum()), attr,
                                          dtype="<U8"))
                    elems.append(np.where(mask)[0])
                    found = True
            if not found:
                log.warning(f"/LINE/{ln.id}: part {pid} has no 1-D "
                            f"(truss/beam/spring) elements", "LINE CHECK")
        for row in ln.seg_nodes:
            try:
                edges.append(model.node_indices(row)[None, :])
                gtypes.append(np.array([""], dtype="<U8"))
                elems.append(np.array([-1], dtype=np.int64))
            except KeyError as exc:
                log.error(f"/LINE/{ln.id}: unknown node id {exc}",
                          "LINE CHECK")
        if edges:
            return (np.vstack(edges), np.concatenate(gtypes),
                    np.concatenate(elems))
        return (np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))

    resolved: dict = {}
    base_cache: dict = {}

    def _finish(ln, triple):
        ln.segments, ln.seg_gtype, ln.seg_elem = triple
        resolved[ln.id] = triple

    def _try(ln) -> bool:
        if ln.id not in base_cache:      # once — errors not duplicated
            base_cache[ln.id] = _base(ln)
        pieces = [base_cache[ln.id]]
        for ref in ln.line_ids:
            other = model.lines.get(abs(ref))
            if other is None:
                log.warning(f"/LINE/{ln.id}: unknown line {abs(ref)} — "
                            f"ignored", "LINE CHECK")
                continue
            if abs(ref) not in resolved:
                return False                        # defer this pass
            seg, gt, el = resolved[abs(ref)]
            if ref < 0:
                seg = seg[:, ::-1]                  # reversed direction
            pieces.append((seg, gt, el))
        _finish(ln, (np.vstack([p[0] for p in pieces]),
                     np.concatenate([p[1] for p in pieces]),
                     np.concatenate([p[2] for p in pieces])))
        return True

    def _on_cycle(ln):
        _finish(ln, (np.zeros((0, 2), dtype=np.int64),
                     np.zeros(0, dtype="<U8"),
                     np.zeros(0, dtype=np.int64)))

    _fixpoint(dict(model.lines), _try, log, "LINE", _on_cycle)
    for ln in model.lines.values():
        if ln.segments.shape[0] == 0:
            log.warning(f"/LINE/{ln.id} '{ln.title}' has no edges",
                        "LINE CHECK")


# ----------------------------------------------------------------------------
# Reference systems (/SKEW, /FRAME) and the consumers that name one  (M39)
# ----------------------------------------------------------------------------

def resolve_skews(model: Model, log: MessageLog) -> None:
    """Build every /SKEW and /FRAME from the initial geometry, then bind
    each consumer's ``skew_ID``/``frame_ID`` to its row.

    Fortran origin: ``starter/source/tools/skew/hm_read_skw.F`` +
    ``hm_read_frm.F`` build the frames; each consumer's reader then walks
    ``ISKN(4,*)`` to turn the USER id into the array index it stores (e.g.
    hm_read_rbody.F 246-256, hm_read_prop08.F 141-149,
    hm_read_inivel.F 415-448).  An unknown id is a hard error there
    (ANCMSG 184/490) and here.

    Must run after the nodes exist (the MOV frames read x0) and before
    anything consumes a frame.
    """
    model.skews.resolve(model, log)

    def _bind(kind: str, user_id: int, who: str) -> int:
        """USER skew/frame id -> row; -1 (and an error) when unknown."""
        row = model.skews.index(kind, user_id)
        if row < 0:
            log.error(f"{who}: unknown {kind.lower()}_ID {user_id}",
                      f"{kind} CHECK")
        return row

    # ---- /BCS: the fixed DOFs are the skew's axes (bcs1.F) --------------
    for bc in model.bcs:
        bc.skew_row = _bind("SKEW", bc.skew_id, f"/BCS/{bc.id}") \
            if bc.skew_id else 0

    # ---- /ALE/BCS: the fixed DOFs are the skew's axes (alewdx_grid_bcs.F) --
    for bc in model.ale_bcs:
        bc.skew_row = _bind("SKEW", bc.skew_id, f"/ALE/BCS/{bc.id}") \
            if bc.skew_id else 0

    # ---- /IMPVEL + /IMPDISP: the imposed DOF is a skew axis (fixvel.F) --
    for im in list(model.impvel) + list(model.impdisp):
        im.skew_row = _bind("SKEW", im.skew_id,
                            f"/IMP*/{im.id}") if im.skew_id else 0

    # ---- /RBODY: Jxx/Jyy/Jzz are written in the skew (inirby.F CHBAS) ---
    for rb in model.rbodies:
        rb.skew_row = _bind("SKEW", rb.skew_id,
                            f"/RBODY/{rb.id}") if rb.skew_id else 0

    # ---- /PROP TYPE8: the spring's 6 DOFs are the skew's axes (r2def3) --
    for prop in model.properties.values():
        sid = int(prop.params.get("skew_id", 0) or 0)
        if not sid:
            continue
        ptype = int(getattr(prop, "type", 0) or 0)
        who = f"/PROP/{prop.id}"
        row = _bind("SKEW", sid, who)
        prop.params["skew_row"] = max(row, 0)
        if ptype == 13:
            # TYPE13 (SPR_BEAM) uses its skew as the INITIAL frame of a
            # co-rotational beam, not as a fixed one — the port's TYPE13
            # frame is the element frame (spring_general.py) and no corpus
            # deck puts a skew on a TYPE13, so this stays a loud defer.
            log.warning(f"{who}: TYPE13 (SPR_BEAM) skew_ID={sid} NOT "
                        f"PORTED — the port uses the element frame "
                        f"(N1->N2); the physics DEVIATES when the skew is "
                        f"not aligned with it", "PROP CHECK")

    # ---- /INIVEL/AXIS: the rotation axis + origin come from the /FRAME --
    for iv in model.inivel:
        if iv.kind != "AXIS" or not iv.frame_id:
            continue
        row = _bind("FRAME", iv.frame_id, f"/INIVEL/AXIS/{iv.id}")
        if row < 0:
            continue
        # hm_read_inivel.F 581-598: the axis is the frame's IDIR axis and
        # the rotation is about the FRAME ORIGIN; 437-439: the card's
        # Vxt/Vyt/Vzt are components IN the frame, rotated to global.
        iv.axis = model.skews.axes[row][iv.dir - 1].copy()
        iv.origin = model.skews.origins[row].copy()
        iv.v = model.skews.to_global(row, iv.v)
        sf = next((s for s in model.skews.entries
                   if s.kind == "FRAME" and s.id == iv.frame_id), None)
        if sf is not None and sf.imov != 0:
            # a MOVING frame is rebuilt every cycle by movfram.F; as an
            # /INIVEL it is only ever read ONCE, at t=0, so the initial
            # orientation IS the whole story and nothing is lost here.
            log.info(f"     /INIVEL/AXIS/{iv.id}: /FRAME/{sf.subtype}/"
                     f"{sf.id} is a MOVING frame; an initial condition "
                     f"reads it only at t=0, so its initial orientation "
                     f"is used (exact)")


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
            
        from pyradioss.engine.coloring import compute_element_colors
        if group.conn is not None and len(group.conn) > 0:
            c_idx, c_off = compute_element_colors(group.conn, model.numnod)
            group.state["color_indices"] = c_idx
            group.state["color_offsets"] = c_off

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
            # rigid-rotation velocity field: v += omega * d x (x0 - P),
            # plus the card's translational velocity Vt (the real AXIS
            # card carries Vxt/Vyt/Vzt next to VR — M37; zero for the
            # port's compact AXIS card, which has no Vt fields)
            r = model.x0[g.node_idx] - iv.origin
            model.v[g.node_idx] += iv.v + iv.omega * np.cross(iv.axis, r)
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
    # one kinematic condition per DOF: a node SLAVE of two bodies (or a
    # node serving as MASTER twice) is an error. A node that is the MASTER
    # of one body and a SLAVE of another is a rigid-body CHAIN (M14): the
    # original starter resolves such nestings into a PARENT_OF hierarchy
    # (rbody_part_modif.F90); this port carries the bodies as written —
    # the IMPLICIT solver resolves the chain by transform substitution
    # (implicit/constraints.py), the EXPLICIT engine refuses it loudly
    # (engine/rigid_body.build_rigid_bodies — its per-body 6-DOF
    # integrator has no nesting order).
    seen_slave = np.zeros(model.numnod, dtype=bool)
    seen_master = np.zeros(model.numnod, dtype=bool)
    # which body claimed each node as a slave first (for the message)
    slave_owner: dict = {}
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

        # ---- shared SECONDARY nodes (checkrby.F) --------------------------
        # A node claimed as a slave by two bodies is an over-constraint —
        # but ONLY when both claimants are ACTIVE.  The reference makes
        # exactly this distinction (checkrby.F, the "Report of secondary
        # double nodes" block): it walks the duplicated slave nodes,
        # collects the bodies sharing each one, and sets IACTI = 0 as soon
        # as ANY of them is inactive (NPBY(7,IRB) == 0, i.e. sens_ID != 0 —
        # hm_read_rbody.F: "IF(ISENS == 0) THEN NPBY(7,NRB)=1 ELSE
        # NPBY(7,NRB)=0").  Then
        #
        #     IF (IFOUND==0 .AND. IACTI> 0)  -> ERROR   (MSGID 3121)
        #     ELSEIF (IFOUND==0 .AND. IACTI==0) -> WARNING (MSGID 1026,
        #                          "sensor effect will be treated later")
        #
        # i.e. sharing is LEGAL between sensor-gated bodies: the sensors
        # choreograph which one owns the nodes at any instant.  That is a
        # real modelling technique — RD-E-1200's BIKERC defines overlapping
        # "all" / "all less wheels" / "rear wheel" bodies, every one gated
        # by a different /SENSOR, and the port's unconditional error
        # rejected the deck (M39 / M38-NEW-4).
        #
        # IFOUND is upstream's ifrbody_off.F90 probe for a /RBODY/OFF card
        # in the engine restart decks (a deactivated body cannot clash
        # either).  The port has no engine-deck /RBODY/OFF reader, so this
        # takes the IFOUND = 0 branch always — the STRICTER of the two, so
        # the cut can only ever over-report, never under-report.
        shared = rb.slaves[seen_slave[rb.slaves]] if len(rb.slaves) else \
            np.empty(0, dtype=np.int64)
        if len(shared) or seen_master[rb.master]:
            others = {}                       # (kind, id) -> body, deduped
            for n in shared:
                o = slave_owner.get(int(n))
                if o is not None:
                    others.setdefault((o.kind, o.id), o)
            others = [others[k] for k in sorted(others)]
            all_active = rb.sens_id == 0 and all(
                o.sens_id == 0 for o in others)
            if seen_master[rb.master] or all_active:
                log.error(f"{who}: node(s) already belong to another rigid "
                          f"body", "RBODY CHECK")
                continue
            names = ", ".join(f"/{o.kind}/{o.id}" for o in others)
            log.warning(f"{who}: shares slave node(s) with {names} — legal "
                        f"because the bodies are /SENSOR-gated (upstream "
                        f"resolves the overlap by activation, checkrby.F "
                        f"MSGID 1026).  NOTE the port does not gate rigid "
                        f"bodies by sensor (sens_ID is read but ignored), "
                        f"so the Engine would drive the shared nodes from "
                        f"every body at once", "RBODY CHECK")
        if seen_master[rb.slaves].any() or seen_slave[rb.master]:
            log.info(f"     {who}: rigid-body CHAIN (a master of one body "
                     f"is a slave of another) — supported by the IMPLICIT "
                     f"solver only (M14)")
        for n in rb.slaves:
            slave_owner.setdefault(int(n), rb)
        seen_slave[rb.slaves] = True
        seen_master[rb.master] = True

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
        # the card's added inertia. With a Skew_ID (M39) Jxx/Jyy/Jzz are
        # written in the SKEW's axes, so the diagonal tensor is rotated
        # into the global frame before it is added — inirby.F's
        # ``IF(NOSKEW/=0) CALL CHBAS(SKEW(1,NOSKEW), RBY(1,NRB))``, which
        # is M = A M A^T with A's columns the skew axes (chbas.F).  Done
        # ONCE here: the body carries its own rotation afterwards, so even
        # a /SKEW/MOV only contributes its initial orientation.
        jadd = np.diag(rb.jadd if rb.jadd is not None else np.zeros(3))
        row = getattr(rb, "skew_row", 0)
        if row:
            jadd = model.skews.rotate_tensor(row, jadd)
        J += jadd

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
