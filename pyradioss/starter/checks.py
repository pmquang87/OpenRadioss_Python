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
    "bricks": {0, 1, 2, 3, 4, 24, 33, 35, 36, 40, 42, 44, 62, 70, 81, 83, 999},
    "tetras": {0, 1, 2, 3, 4, 24, 33, 35, 36, 40, 42, 44, 62, 70, 81, 999},
    "shells": {0, 1, 2, 3, 19, 27, 36, 44},
    # QBAT (Ishell=12, M41): the laws the layered kernel reuses from the
    # BT plumbing; no orthotropic (LAW19) shell_ortho wiring yet
    "shells_qbat": {0, 1, 2, 3, 27, 36, 44},
    # QEPH (Ishell=24, M41): shares the BT layer plumbing INCLUDING the
    # shell_ortho fiber rotation (LAW19); the czfintn.F stabilization
    # runs isotropic moduli (czfintn_or orthotropic HM/HF deferred)
    "shells_qeph": {0, 1, 2, 3, 19, 27, 36, 44},
    "sh3n": {0, 1, 2, 3, 19, 27, 36, 44},
    "trusses": {0, 1, 2},
    "springs": None,          # springs ignore their material entirely
    "beams": {0, 1, 2},
}


def check_model(model: Model, log: MessageLog) -> None:
    if model.numnod == 0:
        log.error("model has no nodes", "MODEL CHECK")
    if not any(True for _ in model.element_groups()):
        log.warning("model has no elements (deck may use only unported "
                    "element types)", "MODEL CHECK")

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
    for imp in model.impacc:
        need_group(imp.grnod_id, f"/IMPACC/{imp.id}")
        need_funct(imp.funct_id, f"/IMPACC/{imp.id}")
    for it in model.imptemp:
        need_group(it.grnod_id, f"/IMPTEMP/{it.id}")
        need_funct(it.funct_id, f"/IMPTEMP/{it.id}")
    for cl in model.centri_loads:
        need_group(cl.grnod_id, f"/LOAD/CENTRI/{cl.id}")
        need_funct(cl.funct_id, f"/LOAD/CENTRI/{cl.id}")
    for pl in model.ploads:
        need_funct(pl.funct_id, f"/PLOAD/{pl.id}")
        if pl.surf_id not in model.surfaces:
            log.error(f"/PLOAD/{pl.id}: surface {pl.surf_id} not defined",
                      "CROSS REF")
    for conv in model.convec_loads:
        need_funct(conv.funct_id, f"/CONVEC/{conv.id}")
        if conv.surf_id not in model.surfaces:
            log.error(f"/CONVEC/{conv.id}: surface {conv.surf_id} not defined",
                      "CROSS REF")
    for rad in model.radiation_loads:
        if rad.funct_id:
            need_funct(rad.funct_id, f"/RADIATION/{rad.id}")
        if rad.surf_id not in model.surfaces:
            log.error(f"/RADIATION/{rad.id}: surface {rad.surf_id} not defined",
                      "CROSS REF")
    for fl in model.impflux_loads:
        if fl.funct_id:
            need_funct(fl.funct_id, f"/IMPFLUX/{fl.id}")
        if fl.surf_id and fl.surf_id not in model.surfaces:
            log.error(f"/IMPFLUX/{fl.id}: surface {fl.surf_id} not defined",
                      "CROSS REF")
        if fl.grbric_id and ("bricks", fl.grbric_id) not in model.element_groups:
            log.error(f"/IMPFLUX/{fl.id}: brick group {fl.grbric_id} not defined",
                      "CROSS REF")
    for itemp in model.initemp:
        if itemp.grnod_id:
            need_group(itemp.grnod_id, f"/INITEMP/{itemp.id}")
    for iv in model.inivol:
        if iv.part_id and iv.part_id not in model.parts:
            log.error(f"/INIVOL/{iv.id}: part {iv.part_id} not defined",
                      "CROSS REF")
        for c in iv.containers:
            if c.surf_id not in model.surfaces and c.surf_id not in model.node_groups:
                log.error(f"/INIVOL/{iv.id}: container surface/group {c.surf_id} not defined",
                          "CROSS REF")
    solid_ids = set()
    for name in ("bricks", "tetra4", "tetra10"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            solid_ids.update(grp.ids.tolist())
    for elem_id in model.ini_bricks:
        if elem_id not in solid_ids:
            log.error(f"/INIBRI: solid element {elem_id} not defined",
                      "CROSS REF")

    shell_ids = set()
    for name in ("shells", "sh3n", "quads"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            shell_ids.update(grp.ids.tolist())
    for elem_id in model.ini_shells:
        if elem_id not in shell_ids:
            log.error(f"/INISHE: shell element {elem_id} not defined",
                      "CROSS REF")

    truss_ids = set(model.trusses.ids.tolist()) if model.trusses is not None and hasattr(model.trusses, "ids") and len(model.trusses.ids) > 0 else set()
    for elem_id in model.ini_trusses:
        if elem_id not in truss_ids:
            log.error(f"/INITRU: truss element {elem_id} not defined",
                      "CROSS REF")

    beam_ids = set(model.beams.ids.tolist()) if model.beams is not None and hasattr(model.beams, "ids") and len(model.beams.ids) > 0 else set()
    for elem_id in model.ini_beams:
        if elem_id not in beam_ids:
            log.error(f"/INIBEA: beam element {elem_id} not defined",
                      "CROSS REF")

    spring_ids = set(model.springs.ids.tolist()) if model.springs is not None and hasattr(model.springs, "ids") and len(model.springs.ids) > 0 else set()
    for elem_id in model.ini_springs:
        if elem_id not in spring_ids:
            log.error(f"/INISPR: spring element {elem_id} not defined",
                      "CROSS REF")

    for sens in model.sensors:
        who = f"/SENSOR/{sens.kind}/{sens.id}"
        if sens.kind in ("DISP", "VEL") and sens.node_id and sens.node_id not in model._id2idx:
            log.error(f"{who}: unknown node {sens.node_id}", "CROSS REF")
        elif sens.kind == "DIST":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "ENERGY" and sens.part_id and sens.part_id not in model.parts:
            log.error(f"{who}: unknown part {sens.part_id}", "CROSS REF")
        elif sens.kind == "TEMP" and sens.grnod_id:
            need_group(sens.grnod_id, who)
        elif sens.kind == "GAUGE":
            for gid, _, _ in sens.gauge_entries:
                if gid > 0 and gid not in model.gauge_points and gid not in model.gauges:
                    log.error(f"{who}: unknown gauge {gid}", "CROSS REF")
        elif sens.kind == "HIC":
            if sens.accel_id > 0 and sens.accel_id not in model.accelerometers:
                log.error(f"{who}: unknown accelerometer {sens.accel_id}", "CROSS REF")
        elif sens.kind == "WORK":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "RWALL":
            rw_ids = {rw.id for rw in model.rwalls}
            if sens.rwall_id > 0 and sens.rwall_id not in rw_ids:
                log.error(f"{who}: unknown rigid wall {sens.rwall_id}", "CROSS REF")
        elif sens.kind in ("XSECTION", "CROSSSECTION", "SECT"):
            sect_ids = {s.id for s in model.sections}
            if sens.sect_id > 0 and sens.sect_id not in sect_ids:
                log.error(f"{who}: unknown section {sens.sect_id}", "CROSS REF")
        elif sens.kind == "DIST_SURF":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")

    for am in model.admas:
        # mass_type 0/1: grnod_id is a node group; 2: surface; 3/4: part
        # group; 6/7: single part.  Only types 0/1 cross-ref node_groups.
        if am.mass_type in (0, 1):
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
        elif itf.type in (1, 3, 6, 12, 15, 20, 21, 23):
            for sid in (itf.surf_id, itf.surf_id1):
                if sid > 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
        elif itf.type in (5, 14):
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
        elif itf.type == 22:
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
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

    # Cyclic boundary conditions (M99)
    for cid, cb in getattr(model, "cyclic_bcs", {}).items():
        if cb.grnod1_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 1 {cb.grnod1_id} not defined", "CROSS REF")
        if cb.grnod2_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 2 {cb.grnod2_id} not defined", "CROSS REF")
        if cb.skew_id > 0 and cb.skew_id not in model.skews:
            log.error(f"/BCS/CYCLIC/{cid}: skew {cb.skew_id} not defined", "CROSS REF")

    # Solid part perturbations (M99)
    part_groups = model.egroups.get("PART", {})
    for pid, pt in getattr(model, "perturbations", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in part_groups and pt.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SOLID/{pid}: part group {pt.grpart_id} not defined", "CROSS REF")

    # Blast loads (M99)
    for bid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{bid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{bid}: detonation node {pb.node_id} not defined", "CROSS REF")

    # Plies & Laminates (M100)
    for ply_id, ply in getattr(model, "plies", {}).items():
        if ply.mat_id > 0 and ply.mat_id not in model.materials:
            log.error(f"/PLY/{ply_id}: material {ply.mat_id} not defined", "CROSS REF")
        if ply.skew_id > 0 and ply.skew_id not in model.skews:
            log.error(f"/PLY/{ply_id}: skew {ply.skew_id} not defined", "CROSS REF")

    for lam_id, lam in getattr(model, "laminates", {}).items():
        for lp in lam.plies:
            if lp.ply_id not in model.plies:
                log.error(f"/LAMINATE/{lam_id}: ply {lp.ply_id} not defined", "CROSS REF")
            if lp.mat_interply > 0 and lp.mat_interply not in model.materials:
                log.error(f"/LAMINATE/{lam_id}: interply material {lp.mat_interply} not defined", "CROSS REF")

    # Sub-interfaces (M100)
    inter_ids = {itf.id for itf in model.interfaces}
    for sub in getattr(model, "sub_interfaces", []):
        if sub.inter_id not in inter_ids:
            log.error(f"/INTER/SUB/{sub.id}: main interface {sub.inter_id} not defined", "CROSS REF")
        if sub.main_id1 > 0 and sub.main_id1 not in model.surfaces and sub.main_id1 not in model.lines:
            log.error(f"/INTER/SUB/{sub.id}: main entity 1 {sub.main_id1} not defined", "CROSS REF")
        if sub.main_id2 > 0 and sub.main_id2 not in model.surfaces:
            log.error(f"/INTER/SUB/{sub.id}: main entity 2 {sub.main_id2} not defined", "CROSS REF")

    # Guided cables (M149)
    for gcid, gc in getattr(model, "guided_cables", {}).items():
        if gc.grnod_id > 0 and gc.grnod_id not in model.node_groups:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: node group {gc.grnod_id} not defined", "CROSS REF")
        if gc.grpart_id > 0 and gc.grpart_id not in model.egroups.get("PART", {}) and gc.grpart_id not in model.parts:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: part group {gc.grpart_id} not defined", "CROSS REF")

    # Composite properties (M100)
    for prop_id, prop in model.properties.items():
        if prop.type in (10, 11, 16, 6) and hasattr(prop, "params"):
            sk = prop.params.get("skew_id", 0)
            if sk > 0 and sk not in model.skews:
                log.error(f"/PROP/TYPE{prop.type}/{prop_id}: skew {sk} not defined", "CROSS REF")
        if prop.type == 11 and hasattr(prop, "params"):
            for ly in prop.params.get("layers", []):
                mid = ly.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/TYPE11/{prop_id}: layer material {mid} not defined", "CROSS REF")

    # Shell and Failure Perturbations & SMS (M101)
    part_groups = model.egroups.get("PART", {})
    for pid, ps in getattr(model, "perturb_shells", {}).items():
        if ps.grpart_id > 0 and ps.grpart_id not in part_groups and ps.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SHELL/{pid}: part group/part {ps.grpart_id} not defined", "CROSS REF")

    fail_ids = {mat_id for mat_id, _, _ in getattr(model, "raw_fails", [])}
    for m in model.materials.values():
        if getattr(m, "failure", None) is not None:
            fail_ids.add(m.id)
    for pid, pf in getattr(model, "perturb_fails", {}).items():
        if pf.fail_id > 0 and pf.fail_id not in fail_ids:
            log.error(f"/PERTURB/FAIL/{pf.fail_type}/{pid}: failure criterion {pf.fail_id} not defined", "CROSS REF")

    if getattr(model, "sms_global", None) is not None:
        sms = model.sms_global
        if sms.grpart_id > 0 and sms.grpart_id not in part_groups and sms.grpart_id not in model.parts:
            log.error(f"/SMS: part group/part {sms.grpart_id} not defined", "CROSS REF")

    # Boundary conditions, Joints, Merge, Inicrack, Laser (M102)
    for bid, bcs in getattr(model, "bcs_nrf", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/NRF/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")

    sensor_ids = {s.id for s in model.sensors}
    for bid, bcs in getattr(model, "bcs_walls", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/WALL/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")
        if bcs.sensor_id > 0 and bcs.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bid}: sensor {bcs.sensor_id} not defined", "CROSS REF")

    for rid, rl in getattr(model, "rlinks", {}).items():
        if rl.grnod_id > 0 and rl.grnod_id not in model.node_groups:
            log.error(f"/RLINK/{rid}: node group {rl.grnod_id} not defined", "CROSS REF")
        if rl.skew_id > 0 and rl.skew_id not in model.skews:
            log.error(f"/RLINK/{rid}: skew {rl.skew_id} not defined", "CROSS REF")

    for cid, cj in getattr(model, "cyl_joints", {}).items():
        if cj.node_id1 > 0 and cj.node_id1 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id1} not defined", "CROSS REF")
        if cj.node_id2 > 0 and cj.node_id2 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id2} not defined", "CROSS REF")
        if cj.grnod_id > 0 and cj.grnod_id not in model.node_groups:
            log.error(f"/CYL_JOINT/{cid}: node group {cj.grnod_id} not defined", "CROSS REF")

    for gid, gj in getattr(model, "gjoints", {}).items():
        for nid in (gj.node_id0, gj.node_id1, gj.node_id2, gj.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/GJOINT/{gid}: node {nid} not defined", "CROSS REF")

    for mid, mn in getattr(model, "node_merges", {}).items():
        if mn.grnod_id > 0 and mn.grnod_id not in model.node_groups:
            log.error(f"/MERGE/NODE/{mid}: node group {mn.grnod_id} not defined", "CROSS REF")

    for iid, ic in getattr(model, "inicracks", {}).items():
        for seg in ic.segments:
            if seg.node_id1 > 0 and seg.node_id1 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id1} not defined", "CROSS REF")
            if seg.node_id2 > 0 and seg.node_id2 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id2} not defined", "CROSS REF")

    for lid, las in getattr(model, "laser_loads", {}).items():
        if las.curve_id > 0 and las.curve_id not in model.functions:
            log.error(f"/LASER/{lid}: function {las.curve_id} not defined", "CROSS REF")
        if las.fct_id_target > 0 and las.fct_id_target not in model.functions:
            log.error(f"/LASER/{lid}: function {las.fct_id_target} not defined", "CROSS REF")

    # Specialized loads, Preload, Damping & Controls (M103)
    for lid, pcyl in getattr(model, "pcyl_loads", {}).items():
        if pcyl.surf_id > 0 and pcyl.surf_id not in model.surfaces:
            log.error(f"/LOAD/PCYL/{lid}: surface {pcyl.surf_id} not defined", "CROSS REF")
        if pcyl.sens_id > 0 and pcyl.sens_id not in sensor_ids:
            log.error(f"/LOAD/PCYL/{lid}: sensor {pcyl.sens_id} not defined", "CROSS REF")
        if pcyl.frame_id > 0 and pcyl.frame_id not in model.skews:
            log.error(f"/LOAD/PCYL/{lid}: skew {pcyl.frame_id} not defined", "CROSS REF")
        if pcyl.table_id > 0 and pcyl.table_id not in model.tables:
            log.error(f"/LOAD/PCYL/{lid}: table {pcyl.table_id} not defined", "CROSS REF")

    for lid, pf in getattr(model, "pfluid_loads", {}).items():
        if pf.surf_id > 0 and pf.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lid}: surface {pf.surf_id} not defined", "CROSS REF")
        if pf.sens_id > 0 and pf.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lid}: sensor {pf.sens_id} not defined", "CROSS REF")
        if pf.fct_id_t > 0 and pf.fct_id_t not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_t} not defined", "CROSS REF")
        if pf.fct_id_pc > 0 and pf.fct_id_pc not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_pc} not defined", "CROSS REF")
        if pf.fct_id_vel > 0 and pf.fct_id_vel not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_vel} not defined", "CROSS REF")
        if pf.frame_id > 0 and pf.frame_id not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id} not defined", "CROSS REF")
        if pf.frame_id_vel > 0 and pf.frame_id_vel not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id_vel} not defined", "CROSS REF")

    for pid, pr in getattr(model, "preloads", {}).items():
        if pr.sect_id > 0 and pr.sect_id not in model.sections and pr.sect_id not in model.properties:
            log.error(f"/PRELOAD/{pid}: section {pr.sect_id} not defined", "CROSS REF")
        if pr.sens_id > 0 and pr.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/{pid}: sensor {pr.sens_id} not defined", "CROSS REF")
        if pr.fct_id > 0 and pr.fct_id not in model.functions:
            log.error(f"/PRELOAD/{pid}: function {pr.fct_id} not defined", "CROSS REF")

    for pid, pra in getattr(model, "preload_axials", {}).items():
        if pra.set_id > 0 and pra.set_id not in part_groups and pra.set_id not in model.parts:
            log.error(f"/PRELOAD/AXIAL/{pid}: part group/part {pra.set_id} not defined", "CROSS REF")
        if pra.sens_id > 0 and pra.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/AXIAL/{pid}: sensor {pra.sens_id} not defined", "CROSS REF")
        if pra.fun_id > 0 and pra.fun_id not in model.functions:
            log.error(f"/PRELOAD/AXIAL/{pid}: function {pra.fun_id} not defined", "CROSS REF")

    for did, di in getattr(model, "damp_inters", {}).items():
        if di.grnod_id > 0 and di.grnod_id not in model.node_groups:
            log.error(f"/DAMP/INTER/{did}: node group {di.grnod_id} not defined", "CROSS REF")
        if di.skew_id > 0 and di.skew_id not in model.skews:
            log.error(f"/DAMP/INTER/{did}: skew {di.skew_id} not defined", "CROSS REF")

    for did, dr in getattr(model, "damp_ranges", {}).items():
        if dr.grpart_id > 0 and dr.grpart_id not in part_groups and dr.grpart_id not in model.parts:
            log.error(f"/DAMP/RANGE/{did}: part group/part {dr.grpart_id} not defined", "CROSS REF")

    for cid, caa in getattr(model, "caa_controls", {}).items():
        if caa.surf_id > 0 and caa.surf_id not in model.surfaces:
            log.error(f"/CAA/{cid}: surface {caa.surf_id} not defined", "CROSS REF")
        if caa.grnod_id > 0 and caa.grnod_id not in model.node_groups:
            log.error(f"/CAA/{cid}: node group {caa.grnod_id} not defined", "CROSS REF")
        if caa.sens_id > 0 and caa.sens_id not in sensor_ids:
            log.error(f"/CAA/{cid}: sensor {caa.sens_id} not defined", "CROSS REF")

    # Gauges, Clusters, Ext Links, Flexible Bodies & Initial Fields (M104)
    for gid, g in getattr(model, "gauges", {}).items():
        if g.node_id > 0 and g.node_id not in model._id2idx:
            log.error(f"/GAUGE/{gid}: node {g.node_id} not defined", "CROSS REF")

    for cid, c in getattr(model, "clusters", {}).items():
        if c.skew_id > 0 and c.skew_id not in model.skews:
            log.error(f"/CLUSTER/{cid}: skew {c.skew_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "ext_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTLNK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    for fid, fx in getattr(model, "fxbodies", {}).items():
        if fx.node_id > 0 and fx.node_id not in model._id2idx:
            log.error(f"/FXBODY/{fid}: node {fx.node_id} not defined", "CROSS REF")

    grav_ids = {g.id for g in model.gravity}
    for iid, ig in getattr(model, "ini_gravs", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in part_groups and ig.grpart_id not in model.parts:
            log.error(f"/INIGRAV/{iid}: part group/part {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{iid}: surface {ig.surf_id} not defined", "CROSS REF")
        if ig.grav_id > 0 and ig.grav_id not in grav_ids:
            log.error(f"/INIGRAV/{iid}: gravity {ig.grav_id} not defined", "CROSS REF")

    for mid, m1 in getattr(model, "ini_map1ds", {}).items():
        for nid in (m1.node_id1, m1.node_id2):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP1D/{mid}: node {nid} not defined", "CROSS REF")

    for mid, m2 in getattr(model, "ini_map2ds", {}).items():
        for nid in (m2.node_id1, m2.node_id2, m2.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP2D/{mid}: node {nid} not defined", "CROSS REF")

    # Monitored Volumes, Airbag Leakage & ALE Controls (M105)
    for mid, mp in getattr(model, "monvol_pres", {}).items():
        if mp.surf_id > 0 and mp.surf_id not in model.surfaces:
            log.error(f"/MONVOL/PRES/{mid}: surface {mp.surf_id} not defined", "CROSS REF")
        if mp.fct_id > 0 and mp.fct_id not in model.functions:
            log.error(f"/MONVOL/PRES/{mid}: function {mp.fct_id} not defined", "CROSS REF")

    for mid, mg in getattr(model, "monvol_gases", {}).items():
        if mg.surf_id > 0 and mg.surf_id not in model.surfaces:
            log.error(f"/MONVOL/GAS/{mid}: surface {mg.surf_id} not defined", "CROSS REF")

    for mid, mc in getattr(model, "monvol_commus", {}).items():
        if mc.surf_id > 0 and mc.surf_id not in model.surfaces:
            log.error(f"/MONVOL/COMMU1/{mid}: surface {mc.surf_id} not defined", "CROSS REF")
        if mc.mat_id > 0 and mc.mat_id not in model.materials:
            log.error(f"/MONVOL/COMMU1/{mid}: material {mc.mat_id} not defined", "CROSS REF")

    for mid, ml in getattr(model, "monvol_lfluids", {}).items():
        if ml.surf_id > 0 and ml.surf_id not in model.surfaces:
            log.error(f"/MONVOL/LFLUID/{mid}: surface {ml.surf_id} not defined", "CROSS REF")
        for fid in (ml.fct_k, ml.fct_mtin, ml.fct_mtout, ml.fct_mpout, ml.fct_padd, ml.fct_pmax):
            if fid > 0 and fid not in model.functions:
                log.error(f"/MONVOL/LFLUID/{mid}: function {fid} not defined", "CROSS REF")

    for lid, lm in getattr(model, "leak_mats", {}).items():
        for fid in (lm.fct_id_e, lm.fct_id_lc, lm.fct_id_ac):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LEAK/{lid}: function {fid} not defined", "CROSS REF")

    for lid, al in getattr(model, "ale_links", {}).items():
        if al.grnod_id > 0 and al.grnod_id not in model.node_groups:
            log.error(f"/ALE/LINK/{lid}: node group {al.grnod_id} not defined", "CROSS REF")
        if al.fct_id > 0 and al.fct_id not in model.functions:
            log.error(f"/ALE/LINK/{lid}: function {al.fct_id} not defined", "CROSS REF")

    # Seatbelts Suite: /RETRACTOR & /SLIPRING (M106)
    sensor_ids = {s.id for s in model.sensors}
    for rid, ret in getattr(model, "retractors", {}).items():
        if ret.node_id > 0 and ret.node_id not in model._id2idx:
            log.error(f"/RETRACTOR/{rid}: node {ret.node_id} not defined", "CROSS REF")
        if ret.sens_id1 > 0 and ret.sens_id1 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id1} not defined", "CROSS REF")
        if ret.sens_id2 > 0 and ret.sens_id2 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id2} not defined", "CROSS REF")
        for fid in (ret.fct_id1, ret.fct_id2, ret.fct_id3):
            if fid > 0 and fid not in model.functions:
                log.error(f"/RETRACTOR/{rid}: function {fid} not defined", "CROSS REF")

    for sid, sr in getattr(model, "sliprings", {}).items():
        if sr.subtype == "SPRING":
            for nid in (sr.node_id, sr.node_id2):
                if nid > 0 and nid not in model._id2idx:
                    log.error(f"/SLIPRING/{sid}: node {nid} not defined", "CROSS REF")
        elif sr.subtype == "SHELL":
            if sr.node_id > 0 and sr.node_id not in model.node_groups:
                log.error(f"/SLIPRING/{sid}: node group {sr.node_id} not defined", "CROSS REF")
        if sr.sens_id > 0 and sr.sens_id not in sensor_ids:
            log.error(f"/SLIPRING/{sid}: sensor {sr.sens_id} not defined", "CROSS REF")
        for fid in (sr.fct_id1, sr.fct_id2, sr.fct_id3, sr.fct_id4):
            if fid > 0 and fid not in model.functions:
                log.error(f"/SLIPRING/{sid}: function {fid} not defined", "CROSS REF")

    # M107: Advanced Failure Criteria & SENSOR/NIC
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "SAHRAEI":
            for fid_key in ("fct_ratio", "fct_elsize"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/SAHRAEI on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "TAB2":
            for fid_key in ("epsf_id", "fct_exp"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/TAB2 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "GENE1":
            for fid_key in ("fct_idsm", "fct_idps"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/GENE1 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    spring_ids = set(model.springs.ids) if (getattr(model, "springs", None) is not None and model.springs.n > 0) else set()
    for sens in getattr(model, "sensors", []):
        if sens.kind == "NIC":
            if sens.spring_id > 0 and sens.spring_id not in spring_ids:
                log.error(f"/SENSOR/NIC/{sens.id}: spring {sens.spring_id} not defined", "CROSS REF")
            if sens.skew_id > 0 and sens.skew_id not in model.skews:
                log.error(f"/SENSOR/NIC/{sens.id}: skew {sens.skew_id} not defined", "CROSS REF")

    # M108: Classical Failure Models, Relative/Function Damping, FVM Airbags, Extended Contacts
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "ENERGY":
            fid = fm.params.get("fct_id", 0)
            if fid > 0 and fid not in model.functions:
                log.error(f"/FAIL/ENERGY on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    for d in getattr(model, "damps", []):
        if getattr(d, "kind", "GLOBAL") == "VREL":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/VREL/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.skew_id > 0 and d.skew_id not in model.skews:
                log.error(f"/DAMP/VREL/{d.id}: skew {d.skew_id} not defined", "CROSS REF")
        elif getattr(d, "kind", "GLOBAL") == "FUNCT":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/FUNCT/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.fct_id > 0 and d.fct_id not in model.functions:
                log.error(f"/DAMP/FUNCT/{d.id}: function {d.fct_id} not defined", "CROSS REF")

    for mid, fb in getattr(model, "monvol_fvmbags", {}).items():
        if fb.surf_id > 0 and fb.surf_id not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG1/{mid}: surface {fb.surf_id} not defined", "CROSS REF")
        if fb.mat_id > 0 and fb.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG1/{mid}: material {fb.mat_id} not defined", "CROSS REF")

    for inter in getattr(model, "interfaces", []):
        if inter.type == 19:
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/TYPE19/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE19/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
        elif inter.type == 21:
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
            if inter.surf_id1 > 0 and inter.surf_id1 not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id1} not defined", "CROSS REF")
        elif inter.type == 29:  # GUIDED_CABLE
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.grpart_id > 0 and inter.grpart_id not in model.egroups.get("PART", {}) and inter.grpart_id not in model.parts:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: part group {inter.grpart_id} not defined", "CROSS REF")

    # M109: Extended Multi-Physics Sensors & Properties
    for sens in getattr(model, "sensors", []):
        if sens.kind == "ENERGY":
            if sens.part_id > 0 and sens.part_id not in model.parts:
                log.error(f"/SENSOR/ENERGY/{sens.id}: part {sens.part_id} not defined", "CROSS REF")
            if sens.subset_id > 0 and sens.subset_id not in model.subsets:
                log.error(f"/SENSOR/ENERGY/{sens.id}: subset {sens.subset_id} not defined", "CROSS REF")
        elif sens.kind == "TEMP":
            if sens.grnod_id > 0 and sens.grnod_id not in model.node_groups:
                log.error(f"/SENSOR/TEMP/{sens.id}: node group {sens.grnod_id} not defined", "CROSS REF")

    for pid, prop in getattr(model, "properties", {}).items():
        if getattr(prop, "type", 0) in (21, 22):
            skew_id = prop.params.get("skew_id", 0)
            if skew_id > 0 and skew_id not in model.skews:
                log.error(f"/PROP/{pid}: skew {skew_id} not defined", "CROSS REF")
            for layer in prop.params.get("layers", []):
                mid = layer.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/{pid}: material {mid} not defined in layer", "CROSS REF")

    # M110: Detonation Wavefronts, Air Blast Loading & Dynamic Element Activation
    for det in getattr(model, "detonations", []):
        if det.mat_id > 0 and det.mat_id not in model.materials:
            log.error(f"/INIT/DET_{det.kind}/{det.id}: material {det.mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for act in getattr(model, "activations", []):
        if act.sens_id > 0 and act.sens_id not in sensor_ids:
            log.error(f"/ACTIV/{act.id}: sensor {act.sens_id} not defined", "CROSS REF")

    # M111: Extended Interfaces, Dual-Chamber Airbags & Autopositioning
    for mvid, mv in getattr(model, "monvol_fvmbag2s", {}).items():
        if mv.surf_id_ex > 0 and mv.surf_id_ex not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: external surface {mv.surf_id_ex} not defined", "CROSS REF")
        if mv.surf_id_in > 0 and mv.surf_id_in not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: internal surface {mv.surf_id_in} not defined", "CROSS REF")
        if mv.mat_id > 0 and mv.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: material {mv.mat_id} not defined", "CROSS REF")

    for ap in getattr(model, "autopositions", []):
        if ap.grnod_id > 0 and ap.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: node group {ap.grnod_id} not defined", "CROSS REF")
        if ap.surf_id > 0 and ap.surf_id not in model.surfaces:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: surface {ap.surf_id} not defined", "CROSS REF")
        if ap.skew_id > 0 and ap.skew_id not in model.skews:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: skew {ap.skew_id} not defined", "CROSS REF")

    # M112: Centrifugal & Pressure Loads, Advanced Initial Velocities, Final Geometry Imposed Fields, Thermal Rigid Walls, and SPH Boundary Suite
    for lcid, lc in getattr(model, "load_centris", {}).items():
        if lc.fct_id > 0 and lc.fct_id not in model.functions:
            log.error(f"/LOAD/CENTRI/{lcid}: function {lc.fct_id} not defined", "CROSS REF")
        if lc.sens_id > 0 and lc.sens_id not in sensor_ids:
            log.error(f"/LOAD/CENTRI/{lcid}: sensor {lc.sens_id} not defined", "CROSS REF")
        if lc.grnod_id > 0 and lc.grnod_id not in model.node_groups:
            log.error(f"/LOAD/CENTRI/{lcid}: node group {lc.grnod_id} not defined", "CROSS REF")
        if lc.frame_id > 0 and lc.frame_id not in model.skews:
            log.error(f"/LOAD/CENTRI/{lcid}: skew {lc.frame_id} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pfluids", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")
        for fid in (lp.fct_id_t, lp.fct_id_pc, lp.fct_id_vel):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LOAD/PFLUID/{lpid}: function {fid} not defined", "CROSS REF")
        for fid in (lp.frame_id, lp.frame_id_vel):
            if fid > 0 and fid not in model.skews:
                log.error(f"/LOAD/PFLUID/{lpid}: skew {fid} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pressures", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PRESSURE/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.fct_id > 0 and lp.fct_id not in model.functions:
            log.error(f"/LOAD/PRESSURE/{lpid}: function {lp.fct_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PRESSURE/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")

    for iaid, ia in getattr(model, "inivel_axes", {}).items():
        if ia.grnod_id > 0 and ia.grnod_id not in model.node_groups:
            log.error(f"/INIVEL/AXIS/{iaid}: node group {ia.grnod_id} not defined", "CROSS REF")
        if ia.frame_id > 0 and ia.frame_id not in model.skews:
            log.error(f"/INIVEL/AXIS/{iaid}: skew {ia.frame_id} not defined", "CROSS REF")
        if ia.sens_id > 0 and ia.sens_id not in sensor_ids:
            log.error(f"/INIVEL/AXIS/{iaid}: sensor {ia.sens_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivel_fvms", {}).items():
        if iv.skew_id > 0 and iv.skew_id not in model.skews:
            log.error(f"/INIVEL/FVM/{ivid}: skew {iv.skew_id} not defined", "CROSS REF")
        if iv.sens_id > 0 and iv.sens_id not in sensor_ids:
            log.error(f"/INIVEL/FVM/{ivid}: sensor {iv.sens_id} not defined", "CROSS REF")

    for inid, in_obj in getattr(model, "inivel_nodes", {}).items():
        for itm in in_obj.items:
            if itm.node_id > 0 and itm.node_id not in model._id2idx:
                log.error(f"/INIVEL/NODE/{inid}: node {itm.node_id} not defined", "CROSS REF")
            if itm.skew_id > 0 and itm.skew_id not in model.skews:
                log.error(f"/INIVEL/NODE/{inid}: skew {itm.skew_id} not defined", "CROSS REF")

    for idfid, idf in getattr(model, "impdisp_fgeos", {}).items():
        if idf.fct_id > 0 and idf.fct_id not in model.functions:
            log.error(f"/IMPDISP/FGEO/{idfid}: function {idf.fct_id} not defined", "CROSS REF")
        if idf.part_id > 0 and idf.part_id not in model.parts:
            log.error(f"/IMPDISP/FGEO/{idfid}: part {idf.part_id} not defined", "CROSS REF")
        if idf.sens_id > 0 and idf.sens_id not in sensor_ids:
            log.error(f"/IMPDISP/FGEO/{idfid}: sensor {idf.sens_id} not defined", "CROSS REF")
        for nd in idf.nodes:
            if nd["node_id"] > 0 and nd["node_id"] not in model._id2idx:
                log.error(f"/IMPDISP/FGEO/{idfid}: node {nd['node_id']} not defined", "CROSS REF")

    for ivfid, ivf in getattr(model, "impvel_fgeos", {}).items():
        if ivf.fct_id > 0 and ivf.fct_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_id} not defined", "CROSS REF")
        if ivf.fct_l_id > 0 and ivf.fct_l_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_l_id} not defined", "CROSS REF")
        if ivf.part_id > 0 and ivf.part_id not in model.parts:
            log.error(f"/IMPVEL/FGEO/{ivfid}: part {ivf.part_id} not defined", "CROSS REF")
        if ivf.sens_id > 0 and ivf.sens_id not in sensor_ids:
            log.error(f"/IMPVEL/FGEO/{ivfid}: sensor {ivf.sens_id} not defined", "CROSS REF")
        for n1, n2 in ivf.pairs:
            if n1 > 0 and n1 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n1} not defined", "CROSS REF")
            if n2 > 0 and n2 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n2} not defined", "CROSS REF")

    for rtid, rt in getattr(model, "rwall_therms", {}).items():
        if rt.node_id > 0 and rt.node_id not in model._id2idx:
            log.error(f"/RWALL/THERM/{rtid}: node {rt.node_id} not defined", "CROSS REF")
        if rt.grnod_id1 > 0 and rt.grnod_id1 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 1 {rt.grnod_id1} not defined", "CROSS REF")
        if rt.grnod_id2 > 0 and rt.grnod_id2 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 2 {rt.grnod_id2} not defined", "CROSS REF")
        if rt.fct_id > 0 and rt.fct_id not in model.functions:
            log.error(f"/RWALL/THERM/{rtid}: function {rt.fct_id} not defined", "CROSS REF")

    for sioid, sio in getattr(model, "sph_inouts", {}).items():
        if sio.surf_id > 0 and sio.surf_id not in model.surfaces:
            log.error(f"/SPH/INOUT/{sioid}: surface {sio.surf_id} not defined", "CROSS REF")
        if sio.part_id > 0 and sio.part_id not in model.parts:
            log.error(f"/SPH/INOUT/{sioid}: part {sio.part_id} not defined", "CROSS REF")
        if sio.fct_id > 0 and sio.fct_id not in model.functions:
            log.error(f"/SPH/INOUT/{sioid}: function {sio.fct_id} not defined", "CROSS REF")

    # M113: SPH Symmetry, Madymo Links/EXFEM, Random Noise, Accelerometers
    for sbid, sb in getattr(model, "sph_bcs", {}).items():
        if sb.frame_id > 0 and sb.frame_id not in model.skews:
            log.error(f"/SPHBCS/{sbid}: skew/frame {sb.frame_id} not defined", "CROSS REF")
        if sb.grnod_id > 0 and sb.grnod_id not in model.node_groups:
            log.error(f"/SPHBCS/{sbid}: node group {sb.grnod_id} not defined", "CROSS REF")

    for mlid, ml in getattr(model, "madymo_links", {}).items():
        if ml.node_id > 0 and ml.node_id not in model._id2idx:
            log.error(f"/MADYMO/LINK/{mlid}: node {ml.node_id} not defined", "CROSS REF")

    for meid, me in getattr(model, "madymo_exfems", {}).items():
        for pid in me.part_ids:
            if pid > 0 and pid not in model.parts:
                log.error(f"/MADYMO/EXFEM/{meid}: part {pid} not defined", "CROSS REF")

    for rn in getattr(model, "random_noises", []):
        if rn.grnod_id > 0 and rn.grnod_id not in model.node_groups:
            log.error(f"/RANDOM/GRNOD/{rn.grnod_id}: node group {rn.grnod_id} not defined", "CROSS REF")

    for aid, acc in getattr(model, "accelerometers", {}).items():
        if acc.node_id > 0 and acc.node_id not in model._id2idx:
            log.error(f"/ACCEL/{aid}: node {acc.node_id} not defined", "CROSS REF")
        if acc.skew_id > 0 and acc.skew_id not in model.skews:
            log.error(f"/ACCEL/{aid}: skew {acc.skew_id} not defined", "CROSS REF")

    # M114: Composite Failure, Propellant Combustion, Non-Uniform Added Mass, Extended Sections
    for mat_id, fc in getattr(model, "fail_composites", {}).items():
        if mat_id > 0 and mat_id not in model.materials:
            log.error(f"/FAIL/COMPOSITE/{mat_id}: material {mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "ebcs_propellants", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PROPELLANT/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.sens_id > 0 and pb.sens_id not in sensor_ids:
            log.error(f"/EBCS/PROPELLANT/{pbid}: sensor {pb.sens_id} not defined", "CROSS REF")
        for fid in (pb.f_func_id, pb.g_func_id, pb.h_func_id):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PROPELLANT/{pbid}: function {fid} not defined", "CROSS REF")

    for anid, an in getattr(model, "admas_non_uniforms", {}).items():
        for item in an.items:
            if an.kind == "NODE" and item.entity_id > 0 and item.entity_id not in model._id2idx:
                log.error(f"/ADMAS/NON_UNIFORM/{anid}: node {item.entity_id} not defined", "CROSS REF")
            elif an.kind == "PART" and item.entity_id > 0 and item.entity_id not in model.parts:
                log.error(f"/ADMAS/NON_UNIFORM_PART/{anid}: part {item.entity_id} not defined", "CROSS REF")

    egroups_shel = {**getattr(model, "egroups", {}).get("GRSHEL", {}), **getattr(model, "egroups", {}).get("SHEL", {})}
    egroups_bric = {**getattr(model, "egroups", {}).get("GRBRIC", {}), **getattr(model, "egroups", {}).get("BRIC", {})}

    for scid, sc in getattr(model, "sect_circles", {}).items():
        for nid in (sc.n1, sc.n2, sc.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/CIRCLE/{scid}: node {nid} not defined", "CROSS REF")
        if sc.grshel_id > 0 and sc.grshel_id not in egroups_shel:
            log.error(f"/SECT/CIRCLE/{scid}: shell group {sc.grshel_id} not defined", "CROSS REF")
        if sc.grbric_id > 0 and sc.grbric_id not in egroups_bric:
            log.error(f"/SECT/CIRCLE/{scid}: brick group {sc.grbric_id} not defined", "CROSS REF")

    for spid, sp in getattr(model, "sect_parals", {}).items():
        for nid in (sp.n1, sp.n2, sp.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/PARAL/{spid}: node {nid} not defined", "CROSS REF")
        if sp.grshel_id > 0 and sp.grshel_id not in egroups_shel:
            log.error(f"/SECT/PARAL/{spid}: shell group {sp.grshel_id} not defined", "CROSS REF")
        if sp.grbric_id > 0 and sp.grbric_id not in egroups_bric:
            log.error(f"/SECT/PARAL/{spid}: brick group {sp.grbric_id} not defined", "CROSS REF")

    for maid, ma in getattr(model, "monvol_areas", {}).items():
        if ma.surf_id_ext > 0 and ma.surf_id_ext not in model.surfaces:
            log.error(f"/MONVOL/AREA/{maid}: surface {ma.surf_id_ext} not defined", "CROSS REF")

    for sid, s in getattr(model, "surfaces", {}).items():
        for mid in getattr(s, "mat_ids", []):
            if mid > 0 and mid not in model.materials:
                log.error(f"/SURF/{sid}: material {mid} not defined", "CROSS REF")
        for pid in getattr(s, "prop_ids", []):
            if pid > 0 and pid not in model.properties:
                log.error(f"/SURF/{sid}: property {pid} not defined", "CROSS REF")
        for bid in getattr(s, "box_ids", []):
            if bid > 0 and bid not in model.boxes:
                log.error(f"/SURF/{sid}: box {bid} not defined", "CROSS REF")

    for pid, sr in getattr(model, "sph_reserves", {}).items():
        if sr.part_id > 0 and sr.part_id not in model.parts:
            log.error(f"/SPH/RESERVE/{pid}: part {sr.part_id} not defined", "CROSS REF")

    for item in getattr(model, "move_functs", []):
        fid = item[0] if isinstance(item, (tuple, list)) else getattr(item, "id", 0)
        if fid > 0 and fid not in model.functions:
            log.error(f"/MOVE_FUNCT/{fid}: function {fid} not defined", "CROSS REF")

    for eid, em in getattr(model, "eigen_modes", {}).items():
        if em.grnod_id > 0 and em.grnod_id not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_id} not defined", "CROSS REF")
        if em.grnod_bc > 0 and em.grnod_bc not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_bc} not defined", "CROSS REF")

    for mid, ff in getattr(model, "fail_fractals", {}).items():
        if ff.mat_id > 0 and ff.mat_id not in model.materials:
            log.error(f"/FAIL/FRACTAL/{mid}: material {ff.mat_id} not defined", "CROSS REF")

    for tid, tp in getattr(model, "transform_positions", {}).items():
        if tp.grnod_id > 0 and tp.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/{tid}: node group {tp.grnod_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "external_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTERN/LINK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    for fid, fm in getattr(model, "friction_models", {}).items():
        for p in fm.pairs:
            if p.part_id1 > 0 and p.part_id1 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id1} not defined", "CROSS REF")
            if p.part_id2 > 0 and p.part_id2 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id2} not defined", "CROSS REF")
            if p.grpart_id1 > 0 and p.grpart_id1 not in getattr(model, "part_groups", {}):
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id1} not defined", "CROSS REF")
            if p.grpart_id2 > 0 and p.grpart_id2 not in getattr(model, "part_groups", {}):
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id2} not defined", "CROSS REF")

    for bid, nb in getattr(model, "nbcs_blocks", {}).items():
        for n in nb.nodes:
            if n.node_id > 0 and n.node_id not in model._id2idx:
                log.error(f"/NBCS/{bid}: node {n.node_id} not defined", "CROSS REF")
            if n.skew_id > 0 and n.skew_id not in model.skews:
                log.error(f"/NBCS/{bid}: skew {n.skew_id} not defined", "CROSS REF")

    for nid in getattr(model, "refsta_nodes", {}):
        if nid > 0 and nid not in model._id2idx:
            log.error(f"/REFSTA: node {nid} not defined", "CROSS REF")

    # M150: EBCS, AMS, and Seatbelt Systems
    for pid, eb in getattr(model, "ebcs_pres", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PRES/{pid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_pres, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PRES/{pid}: function {fid} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_vel", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/VEL/{vid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_vx, eb.fct_vy, eb.fct_vz, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/VEL/{vid}: function {fid} not defined", "CROSS REF")

    for iid, eb in getattr(model, "ebcs_inlets", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/INLET/{iid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/INLET/{iid}: function {eb.funct_id} not defined", "CROSS REF")

    for fid, eb in getattr(model, "ebcs_fluxouts", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/FLUXOUT/{fid}: surface {eb.surf_id} not defined", "CROSS REF")

    for gid, eb in getattr(model, "ebcs_gradp0", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/GRADP0/{gid}: surface {eb.surf_id} not defined", "CROSS REF")

    for nid, eb in getattr(model, "ebcs_normv", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/NORMV/{nid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/NORMV/{nid}: function {eb.funct_id} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_valves", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/{eb.kind}/{vid}: surface {eb.surf_id} not defined", "CROSS REF")

    for mid, eb in getattr(model, "ebcs_monvols", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/MONVOL/{mid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.monvol_id > 0 and eb.monvol_id not in getattr(model, "monitored_volumes", {}) and eb.monvol_id not in getattr(model, "airbags", {}):
            log.error(f"/EBCS/MONVOL/{mid}: monvol {eb.monvol_id} not defined", "CROSS REF")

    if getattr(model, "ams_control", None) is not None:
        ams = model.ams_control
        if ams.grpart_id > 0 and ams.grpart_id not in part_groups and ams.grpart_id not in model.parts:
            log.error(f"/AMS: part group {ams.grpart_id} not defined", "CROSS REF")

    for sbid, sb in getattr(model, "seatbelt_systems", {}).items():
        for rid in sb.retractor_ids:
            if rid > 0 and rid not in getattr(model, "retractors", {}):
                log.error(f"/SEATBELT/{sbid}: retractor {rid} not defined", "CROSS REF")
        for sid in sb.slipring_ids:
            if sid > 0 and sid not in getattr(model, "sliprings", {}) and sid not in getattr(model, "slipring_shells", {}):
                log.error(f"/SEATBELT/{sbid}: slipring {sid} not defined", "CROSS REF")

    for bwid, bw in getattr(model, "bcs_walls", {}).items():
        if bw.grnod_id > 0 and bw.grnod_id not in model.node_groups and bw.grnod_id not in getattr(model, "node_sets", {}):
            log.error(f"/BCS/WALL/{bwid}: node group {bw.grnod_id} not defined", "CROSS REF")
        if bw.sensor_id > 0 and bw.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bwid}: sensor {bw.sensor_id} not defined", "CROSS REF")

    # M151: PBLAST, INIVOL, INIGRAV, INISTA, BEM, PERTURB
    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivols", {}).items():
        if iv.part_id > 0 and iv.part_id not in model.parts and iv.part_id not in part_groups:
            log.error(f"/INIVOL/{ivid}: part {iv.part_id} not defined", "CROSS REF")
        for c in iv.containers:
            if c.surf_id > 0 and c.surf_id not in model.surfaces:
                log.error(f"/INIVOL/{ivid}: container surface {c.surf_id} not defined", "CROSS REF")

    for igid, ig in getattr(model, "inigrav_loads", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in model.parts and ig.grpart_id not in part_groups:
            log.error(f"/INIGRAV/{igid}: part group {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{igid}: surface {ig.surf_id} not defined", "CROSS REF")

    for bemid, bem in getattr(model, "bem_controls", {}).items():
        if bem.surf_id > 0 and bem.surf_id not in model.surfaces:
            log.error(f"/BEM/{bem.subtype}/{bemid}: surface {bem.surf_id} not defined", "CROSS REF")
        if bem.grnod_aux_id > 0 and bem.grnod_aux_id not in model.node_groups and bem.grnod_aux_id not in getattr(model, "node_sets", {}):
            log.error(f"/BEM/{bem.subtype}/{bemid}: node group {bem.grnod_aux_id} not defined", "CROSS REF")

    for ptid, pt in getattr(model, "perturb_controls", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in model.parts and pt.grpart_id not in part_groups:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: part group {pt.grpart_id} not defined", "CROSS REF")
        if pt.fct_id > 0 and pt.fct_id not in model.functions:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: function {pt.fct_id} not defined", "CROSS REF")


