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


def check_model(model: Model, log: MessageLog) -> None:
    if model.numnod == 0:
        log.error("model has no nodes", "MODEL CHECK")
    if not any(True for _ in model.element_groups()):
        log.error("model has no elements", "MODEL CHECK")

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
        need_group(itf.grnod_id, f"/INTER/TYPE7/{itf.id}")
        if itf.surf_id not in model.surfaces:
            log.error(f"/INTER/TYPE7/{itf.id}: surface {itf.surf_id} "
                      f"not defined", "CROSS REF")
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
