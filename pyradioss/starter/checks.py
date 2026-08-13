"""
Starter model checks (cross references, sanity).

Fortran origin: scattered through the Starter (every hm_read does its own
id checks; contchk/prelecdt do global ones). Centralized here: after
finalization, verify every reference between options resolves, so the
Engine can index blindly.
"""

from __future__ import annotations

from ..common.messages import MessageLog
from ..model.model import Model


# multi-material ALE/Euler laws (LAW51, LAW151/MULTIFLUID): their initial
# density is NOT a top-level RHO0 — it is carried by the submaterial
# references (mat_ID_ii) weighted by their volume fractions (Vfrac_ii), so
# the material card's own RHO0 field is legitimately blank/zero.  The
# reference Starter reads the submaterial densities to build the element
# mass and does NOT fatal-error the empty top-level density, so the port's
# null-RHO0 MAT CHECK must exempt the family (M38 / M37-BUG-2).
_MULTIMAT_ALE_LAWS = {51, 151}


# Laws whose RHO0 may legally be zero — the null-density MAT CHECK exempts
# them (M39 / M38-NEW-2).  Two distinct reasons:
#
# * LAW0 (/MAT/VOID) is MASSLESS BY DESIGN.  The upstream reader
#   ``starter/source/materials/mat/mat000/hm_read_mat00.F`` applies NO
#   positivity check to RHO0 and explicitly guards the only place the
#   density is divided by —  ``SDSP = SQRT(YOUNG/MAX(RHOR,EM20))`` — so a
#   void card with RHO0 = 0 is legal and produces a zero sound speed, not
#   an error.  The cfg agrees and is the sharpest evidence: every load-
#   bearing law's cfg CHECK block demands ``MAT_RHO > 0`` (e.g.
#   matl2_plas_johns.cfg) while ``MAT/matl_void0.cfg`` demands only
#   ``MAT_RHO >= 0``.  A void element contributes zero mass and zero
#   stress: dummy contact skins, airbag reference geometry, parts replaced
#   by a rigid body.  The port's fatal here was a false positive on the
#   RD-E-2700 Football decks (BAT_CIR / BAT_SQR), whose /MAT/VOID/12 skin
#   shells carry RHO0 = 0 verbatim;
# * the multimaterial ALE family carries its density on the submaterials
#   (see _MULTIMAT_ALE_LAWS above).
_NULL_RHO0_OK_LAWS = frozenset({0} | _MULTIMAT_ALE_LAWS)


# element family -> material laws its kernels implement (see the
# materials package dispatch; extending a kernel means extending this map).
#
# LAW0 (/MAT/VOID) is legal on EVERY family here, matching the upstream
# compatibility declaration in hm_read_mat00.F, which tags the void law
# SOLID_ISOTROPIC / SHELL_ISOTROPIC / SPRING_MATERIAL / BEAM_ALL / TRUSS /
# SPH — i.e. all of them (M39 / M38-NEW-2, closing the M38 prop-pack OPEN
# item: /PROP/VOID was already made universally family-compatible by
# ``prop_reader.prop_type_ok``, but the MATERIAL half still rejected LAW0
# on /BEAM and /TRUSS, so a void beam passed the property check and failed
# the material one).  The truss/beam kernels honour it: a void material's
# E = G = 0 makes every resultant identically zero (the void semantics)
# and their density divisions are guarded exactly as hm_read_mat00.F
# guards its own — see elements/truss.py and elements/beam_type3.py.
_ALLOWED_LAWS = {
    "bricks": {0, 1, 2, 24, 35, 36, 40, 42, 44, 62, 70, 81, 999},
    "tetras": {0, 1, 2, 24, 35, 36, 40, 42, 44, 62, 70, 81, 999},
    "shells": {0, 1, 2, 19, 27, 36, 44},
    # QBAT (Ishell=12, M41): the laws the layered kernel reuses from the
    # BT plumbing; no orthotropic (LAW19) shell_ortho wiring yet
    "shells_qbat": {0, 1, 2, 27, 36, 44},
    # QEPH (Ishell=24, M41): shares the BT layer plumbing INCLUDING the
    # shell_ortho fiber rotation (LAW19); the czfintn.F stabilization
    # runs isotropic moduli (czfintn_or orthotropic HM/HF deferred)
    "shells_qeph": {0, 1, 2, 19, 27, 36, 44},
    "sh3n": {0, 1, 2, 19, 27, 36, 44},
    "trusses": {0, 1, 2},
    "springs": None,          # springs ignore their material entirely
    "beams": {0, 1, 2},
}


def check_model(model: Model, log: MessageLog) -> None:
    if model.numnod == 0:
        log.error("model has no nodes", "MODEL CHECK")
    if not any(True for _ in model.element_groups()):
        log.error("model has no elements", "MODEL CHECK")

    # material law vs element family compatibility (fail in the Starter
    # with a clear message instead of a NotImplementedError mid-run)
    for name, group in model.element_groups():
        allowed = _ALLOWED_LAWS.get(name)
        if allowed is None:
            continue
        for sl, mat, prop in group.state["slices"]:
            if mat.law == 999 and mat.eos is None:
                # M37 pack 1: a /MAT/GAS on elements has no pressure or
                # stiffness of its own — the initial state must come
                # from an /EOS/IDEAL-GAS card (or the programmatic
                # P0/T0/RHO0 params); see materials/mat_gas.py.  Checked
                # before the density (a bare gas card has neither).
                log.error(f"/MAT/GAS/{mat.id} on {name} elements needs "
                          f"an /EOS/IDEAL-GAS card for its initial "
                          f"state (P0, gamma) — a bare gas card has no "
                          f"element pressure", "MAT CHECK")
            rho0 = getattr(mat, "rho0", 0.0) or 0.0
            if rho0 <= 0.0 and mat.law not in _NULL_RHO0_OK_LAWS:
                # M37: a used material MUST carry a positive initial
                # density — element masses cannot be initialized without
                # it (the reference Starter raises the same fatal check).
                # EXEMPT (see _NULL_RHO0_OK_LAWS): the multimaterial ALE
                # family (LAW51, LAW151/MULTIFLUID — M38 / M37-BUG-2),
                # whose density lives on the submaterial references +
                # volume fractions rather than the top-level RHO0; and
                # LAW0 / /MAT/VOID (M39 / M38-NEW-2), which is massless by
                # design — upstream applies no RHO0 check to it at all and
                # its cfg demands only MAT_RHO >= 0.  The ALE laws are
                # also 'inactive' in the port, so they fall through to the
                # physics-not-implemented warning below; VOID is active
                # and falls through to the family check.
                law_name = getattr(mat, "law_name", None) \
                    or f"LAW{mat.law}"
                log.error(f"/MAT/{law_name}/{mat.id} on {name} elements: "
                          f"zero or missing initial density "
                          f"(RHO0={rho0:g}) — element masses cannot be "
                          f"initialized", "MAT CHECK")
                continue
            if getattr(mat, "inactive", False):
                # M37: cfg-parsed law without ported physics — the
                # STARTER accepts it (params + density read, mass init
                # works); the ENGINE refuses to run the model (see
                # mat_reader.refuse_inactive_materials)
                law_name = getattr(mat, "law_name", f"LAW{mat.law}")
                log.warning(f"/MAT/{law_name}/{mat.id} on {name} "
                            f"elements: parsed, physics not implemented "
                            f"(M37) — the Engine will refuse to run "
                            f"this model", "MAT CHECK")
                continue
            if mat.law not in allowed:
                log.error(f"material LAW{mat.law} (/MAT {mat.id}) is not "
                          f"ported for {name} elements (supported: "
                          f"{sorted(allowed)})", "MAT CHECK")
            if mat.fail is not None and name in ("trusses", "springs",
                                                 "beams"):
                log.warning(f"/FAIL on /MAT {mat.id} is ignored for {name} "
                            f"(failure is ported for solids and shells)",
                            "MAT CHECK")

    # M38: element groups that reference a parsed-but-not-implemented
    # PROPERTY (InactiveProperty) — the Starter accepts them (params +
    # safe geometry read, mass init works); the Engine refuses to run the
    # model (see prop_reader.refuse_inactive_properties), mirroring the
    # inactive-material warning above.
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if getattr(prop, "inactive", False):
                pn = getattr(prop, "prop_name", None) or f"TYPE{prop.type}"
                log.warning(f"/PROP/{pn}/{prop.id} on {name} elements: "
                            f"parsed, physics not implemented (M38) — the "
                            f"Engine will refuse to run this model",
                            "PROP CHECK")

    def need_group(gid, who):
        if gid is not None and gid != 0 and gid not in model.node_groups:
            log.error(f"{who}: node group {gid} not defined", "CROSS REF")

    def need_funct(fid, who):
        if fid not in model.functions:
            log.error(f"{who}: function {fid} not defined", "CROSS REF")

    for bc in model.bcs:
        need_group(bc.grnod_id, f"/BCS/{bc.id}")
    for iv in model.inivel:
        need_group(iv.grnod_id, f"/INIVEL/{iv.id}")
    for gv in model.gravity:
        need_group(gv.grnod_id, f"/GRAV/{gv.id}")
        need_funct(gv.funct_id, f"/GRAV/{gv.id}")
    for cl in model.cloads:
        need_group(cl.grnod_id, f"/CLOAD/{cl.id}")
        need_funct(cl.funct_id, f"/CLOAD/{cl.id}")
    for imp in model.impvel:
        need_group(imp.grnod_id, f"/IMPVEL/{imp.id}")
        need_funct(imp.funct_id, f"/IMPVEL/{imp.id}")
    for imp in model.impdisp:
        need_group(imp.grnod_id, f"/IMPDISP/{imp.id}")
        need_funct(imp.funct_id, f"/IMPDISP/{imp.id}")
    for pl in model.ploads:
        need_funct(pl.funct_id, f"/PLOAD/{pl.id}")
        if pl.surf_id not in model.surfaces:
            log.error(f"/PLOAD/{pl.id}: surface {pl.surf_id} not defined",
                      "CROSS REF")
    for am in model.admas:
        need_group(am.grnod_id, f"/ADMAS/{am.id}")
    for rb in model.rbodies:
        need_group(rb.grnod_id, f"/{rb.kind}/{rb.id}")
        if rb.master_id not in model._id2idx:
            log.error(f"/{rb.kind}/{rb.id}: unknown master node "
                      f"{rb.master_id}", "CROSS REF")
    for r3 in model.rbe3:
        need_group(r3.grnod_id, f"/RBE3/{r3.id}")
        if r3.ref_id not in model._id2idx:
            log.error(f"/RBE3/{r3.id}: unknown reference node {r3.ref_id}",
                      "CROSS REF")
    for sc in model.sections:
        need_group(sc.grnod_id, f"/SECT/{sc.id}")
        if sc.node_id_ref and sc.node_id_ref not in model._id2idx:
            log.error(f"/SECT/{sc.id}: unknown reference node "
                      f"{sc.node_id_ref}", "CROSS REF")
    for rw in model.rwalls:
        need_group(rw.grnod_id, f"/RWALL/{rw.id}")
        if rw.grnod_id2:
            need_group(rw.grnod_id2, f"/RWALL/{rw.id}")
        if rw.node_id and rw.node_id not in model._id2idx:
            log.error(f"/RWALL/{rw.id}: unknown wall node {rw.node_id}",
                      "CROSS REF")
    for itf in model.interfaces:
        who = f"/INTER/TYPE{itf.type}/{itf.id}"
        if itf.type in (7, 2):
            # TYPE7 allows grnod_id = 0 (self-impact: secondary side
            # defaults to the main surface's own nodes); TYPE2 does not.
            if itf.type == 2 or itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            if itf.type == 2 and itf.grnod_id == 0:
                log.error(f"{who}: a tied interface needs a secondary "
                          f"node group", "CROSS REF")
            if itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined",
                          "CROSS REF")
        elif itf.type == 11:
            for lid in (itf.line_id1, itf.line_id2):
                if lid not in model.lines:
                    log.error(f"{who}: line {lid} not defined", "CROSS REF")
        elif itf.type == 24:
            if itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            for sid in (itf.surf_id1, itf.surf_id):
                if sid != 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
    for th in model.th_requests:
        if th.kind == "NODE":
            for nid in th.ids:
                if nid not in model._id2idx:
                    log.error(f"/TH/NODE/{th.id}: unknown node {nid}",
                              "CROSS REF")
        elif th.kind == "PART":
            for pid in th.ids:
                if pid not in model.parts:
                    log.error(f"/TH/PART/{th.id}: unknown part {pid}",
                              "CROSS REF")
        elif th.kind == "SECT":
            defined = {s.id for s in model.sections}
            for sid in th.ids:
                if sid not in defined:
                    log.error(f"/TH/SECT/{th.id}: unknown section {sid}",
                              "CROSS REF")
