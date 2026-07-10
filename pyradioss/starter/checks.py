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


# element family -> material laws its kernels implement (see the
# materials package dispatch; extending a kernel means extending this map)
_ALLOWED_LAWS = {
    "bricks": {1, 2, 36, 42},
    "tetras": {1, 2, 36, 42},
    "shells": {1, 2, 27, 36},
    "sh3n": {1, 2, 27, 36},
    "trusses": {1, 2},
    "springs": None,          # springs ignore their material entirely
    "beams": {1, 2},
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
            if mat.law not in allowed:
                log.error(f"material LAW{mat.law} (/MAT {mat.id}) is not "
                          f"ported for {name} elements (supported: "
                          f"{sorted(allowed)})", "MAT CHECK")
            if mat.fail is not None and name in ("trusses", "springs",
                                                 "beams"):
                log.warning(f"/FAIL on /MAT {mat.id} is ignored for {name} "
                            f"(failure is ported for solids and shells)",
                            "MAT CHECK")

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
    for rw in model.rwalls:
        need_group(rw.grnod_id, f"/RWALL/{rw.id}")
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
