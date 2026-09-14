"""
Animation states (the A-files, /ANIM) as legacy VTK.

Fortran origin: ``engine/source/output/anim/`` writes the proprietary
binary ANIM format (RunNameA001, A002, ...). The port writes **legacy VTK
unstructured grids** (RunNameA000.vtk, ...) which ParaView opens natively
as a time series — select the ``RunNameA..vtk`` file *group*.

Contents per state:
* field data: TIME (state time) and CYCLE (engine cycle) — the official
  anim_to_vtk converter's names; the title-line ``t=`` stays for parsers
  that predate the FIELD block
* points  = current node positions (deformed geometry)
* cells   = all elements (hexa/quad/line)
* point data: DISPLACEMENT, VELOCITY vectors (per /ANIM/VECT),
  NODE_ID user node ids (row-aligned with POINTS)
* cell data:  VONM von Mises stress, EPSP plastic strain (per /ANIM/ELEM),
  OFF element status (always written: 1 = alive, 0 = deleted by a /FAIL
  criterion or a material failure threshold — threshold/select on OFF in
  ParaView to hide the deleted elements),
  ELEMENT_ID / PART_ID user ids (aligned with the cell order)
  - solids: from the stress tensor; shells: worst layer;
    trusses/springs: |axial stress| (resp. 0)
* full result arrays (M42), named like the official anim_to_vtk output
  and appended behind the historical prefix; every array spans ALL cells
  with zero rows for foreign element families (converters read blind
  token streams):
  - ``TENSORS 3DELEM_Stress`` — solid Cauchy stress in the GLOBAL frame
    (the ported solids are the Isolid=1 Jaumann formulation: sig lives in
    the fixed global basis — a corotational solid kernel, if ever added,
    must rotate before emitting here).  The Voigt 6 [xx,yy,zz,xy,yz,zx]
    fills the 3x3 as ``[s0 s3 s4 / s3 s1 s5 / s4 s5 s2]``: yz at (0,2)
    and zx at (1,2), the official anim_to_vtk placement (not the
    textbook one) that the VTK->d3plot converter reads back by position
  - ``TENSORS 2DELEM_Stress_(lower)/(upper)`` — shell outer-fiber
    in-plane stress as ELEMENT-LOCAL plane-stress 3x3 (like the official
    tool; the corotational storage frame — rotated fiber->element for
    orthotropic slices), lower = layer 0 / upper = layer nip-1 per part
    slice, QBAT reduced by its 4-GP mean
  - ``SCALARS 3DELEM_Plastic_Strain / 2DELEM_Plastic_Strain_Lower/
    _Upper`` with the same layer selection
"""

from __future__ import annotations

import numpy as np

from ..elements import shell_ortho
from ..model.model import Model

# group name -> (VTK cell type id, node count written). Beams write only
# their two end nodes (the 3rd is the orientation node, not geometry).
_VTK_CELL = {
    "bricks": (12, 8),
    "bricks_heph": (12, 8),
    "bric20s": (25, 20),
    "shel16s": (12, 8),
    "tetras": (10, 4),
    "tetra10s": (24, 10),
    "shells": (9, 4),
    "shells_qbat": (9, 4),
    "shells_qeph": (9, 4),
    "sh3n": (5, 3),
    "sh3n_dkt18": (5, 3),
    "quads": (9, 4),
    "trusses": (3, 2),
    "springs": (3, 2),
    "beams": (3, 2),
}

_SOLID_FAMILIES = ("bricks", "bricks_heph", "tetras", "tetra10s", "bric20s", "shel16s", "quads")
_SHELL_FAMILIES = ("shells", "shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18")

# anim_to_vtk's symmetric-3x3 fill of the solid Voigt 6 [xx,yy,zz,xy,yz,
# zx], row-major: [s0 s3 s4 / s3 s1 s5 / s4 s5 s2].  yz lands at (0,2)
# and zx at (1,2) — NOT the textbook placement, but the official tool's,
# and the VTK->d3plot converter maps the slots back purely by position.
_VOIGT9 = [0, 3, 4, 3, 1, 5, 4, 5, 2]


def _plane_rows(s3: np.ndarray) -> np.ndarray:
    """(n, 3) element-local in-plane [sxx, syy, sxy] -> (n, 9) plane-stress
    3x3 rows ``[sxx sxy 0 / sxy syy 0 / 0 0 0]`` (the official shell
    TENSORS layout — transverse shear and szz stay zero)."""
    out = np.zeros((s3.shape[0], 9))
    out[:, 0] = s3[:, 0]
    out[:, 4] = s3[:, 1]
    out[:, 1] = out[:, 3] = s3[:, 2]
    return out


def _shell_layers(group_name: str, group):
    """Outer-fiber state of a shell group in the ELEMENT frame:
    ``(sig_lower (n,3), sig_upper (n,3), ep_lower (n,), ep_upper (n,))``.

    lower = layer 0 and upper = layer nip-1 of each part slice — the
    leggauss stations ascend from -t/2 (bottom) to +t/2 (top).  QBAT's
    GP-major ``(n, 4*nip_max, ...)`` state (layer il of GP ng at flat
    index ``ng*nip_max + il``) is reduced by the 4-GP mean, the GBUF%FOR
    convention.  Orthotropic slices (PROP types 9/16) store sig in the
    FIBER frame; it is rotated back to the element frame here exactly
    like the kernels rotate their resultants (rot_stress_m2e)."""
    st = group.state
    sig, ep = st["sig"], st.get("epsp")
    if sig.ndim == 2:
        sig = sig[:, None, :]
    if ep is not None and ep.ndim == 1:
        ep = ep[:, None]
    lo = np.zeros((group.n, 3))
    up = np.zeros((group.n, 3))
    eplo = np.zeros(group.n)
    epup = np.zeros(group.n)
    if group_name == "shells_qbat":
        nip_max = st["nip_max"]
        for isl, (sl, mat, prop) in enumerate(st["slices"]):
            nip = len(st["zw"][isl][0])
            kl = [ng * nip_max for ng in range(4)]
            ku = [ng * nip_max + nip - 1 for ng in range(4)]
            lo[sl] = sig[sl][:, kl, :].mean(axis=1)
            up[sl] = sig[sl][:, ku, :].mean(axis=1)
            if ep is not None:
                eplo[sl] = ep[sl][:, kl].mean(axis=1)
                epup[sl] = ep[sl][:, ku].mean(axis=1)
        return lo, up, eplo, epup
    ortho = st.get("ortho")
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        nip = len(st["zw"][isl][0])
        s_lo, s_up = sig[sl, 0, :], sig[sl, nip - 1, :]
        if ortho is not None and getattr(prop, "type", 0) in \
                shell_ortho.ORTHO_PROP_TYPES:
            s_lo = shell_ortho.rot_stress_m2e(s_lo, ortho[sl])
            s_up = shell_ortho.rot_stress_m2e(s_up, ortho[sl])
        lo[sl], up[sl] = s_lo, s_up
        if ep is not None:
            eplo[sl] = ep[sl, 0]
            epup[sl] = ep[sl, nip - 1]
    return lo, up, eplo, epup


def _write_block(fh, arr, fmt: str) -> None:
    """Byte-for-byte-identical, faster replacement for
    ``np.savetxt(fh, arr, fmt=fmt)`` at savetxt's default delimiter (' ')
    and newline ('\\n'); ``arr`` is 1-D or 2-D.

    np.savetxt formats every row with a Python ``format % tuple(row)`` over
    the numpy row (re-boxing each scalar) and issues ONE ``fh.write`` PER
    ROW — O(n) scalar boxing and O(n) write calls. This M39 output-path
    replacement unboxes the whole array once with ``.tolist()`` (a single C
    conversion), formats with the SAME ``%`` machinery on the same
    per-column spec (so the text is identical to the last bit — the anim
    fmt set "%.9E"/"%d"/"%.1f" is checked, incl. inf/nan/neg-zero, by
    tests/test_m7_backends), and writes each block in ONE call. Measured
    ~1.9x faster on the 72k-node gasket arrays; write_anim_state was
    ~2.6 s/state on the 65 439-brick cliff deck, essentially all of it in
    np.savetxt. Empty arrays write nothing (matches np.savetxt)."""
    a = np.asarray(arr)
    if a.shape[0] == 0:
        return
    if a.ndim == 1:
        fh.write("\n".join([fmt % v for v in a.tolist()]))
    else:
        rowfmt = " ".join([fmt] * a.shape[1])
        fh.write("\n".join([rowfmt % tuple(r) for r in a.tolist()]))
    fh.write("\n")


def _von_mises(group_name: str, group) -> np.ndarray:
    st = group.state
    if group_name in ("bricks", "bricks_heph", "tetras", "tetra10s", "bric20s", "shel16s", "quads"):
        s = st["sig"]
        if s.ndim == 3:
            s = s.mean(axis=1)
        return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2
                              + (s[:, 1] - s[:, 2]) ** 2
                              + (s[:, 2] - s[:, 0]) ** 2)
                       + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    if group_name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18"):
        # (n, nip, 3); QBAT stores (n, 4*nip, 3) GP-major — same reduction
        s = st["sig"]
        if s.ndim == 2:
            s = s[:, None, :]
        vm = np.sqrt(s[:, :, 0] ** 2 - s[:, :, 0] * s[:, :, 1]
                     + s[:, :, 1] ** 2 + 3.0 * s[:, :, 2] ** 2)
        return vm.max(axis=1)
    if group_name == "trusses":
        return np.abs(st["sig"])
    if group_name == "beams":
        # display value: |axial stress| = |N| / A (resultant over section)
        area = np.zeros(group.n)
        for sl, mat, prop in st["slices"]:
            area[sl] = prop.params["area"]
        return np.abs(st["fres"][:, 0]) / np.maximum(area, 1e-20)
    return np.zeros(group.n)


def _epsp(group_name: str, group) -> np.ndarray:
    st = group.state
    if "epsp" not in st:
        return np.zeros(group.n)
    e = st["epsp"]
    return e.max(axis=1) if e.ndim >= 2 else e


def _get_cell_info(name: str, g):
    ctype, nn = _VTK_CELL[name]
    if name == "tetra10s" and (g.conn[:, :nn] < 0).any():
        return (10, 4)  # degenerate/slaved tetra10 write 4 corner nodes
    if name == "bric20s" and (g.conn[:, :nn] < 0).any():
        return (12, 8)  # slaved bric20 write 8 corner nodes
    return (ctype, nn)


def write_anim_state(path: str, model: Model, t: float,
                     vect=("DIS", "VEL"), elem=("VONM", "EPSP"),
                     cycle: int = 0) -> None:
    n = model.numnod
    groups = list(model.element_groups())
    ncell = sum(g.n for _, g in groups)
    size = sum((1 + _get_cell_info(name, g)[1]) * g.n for name, g in groups)

    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"pyradioss state t={t:.9E}\n")
        fh.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        # state time and engine cycle as typed data, named exactly like the
        # official anim_to_vtk converter's FIELD block
        fh.write("FIELD FieldData 2\n")
        fh.write(f"TIME 1 1 double\n{t:.9E}\n")
        fh.write(f"CYCLE 1 1 int\n{cycle:d}\n")
        fh.write(f"POINTS {n} double\n")
        _write_block(fh, model.x, "%.9E")
        fh.write(f"CELLS {ncell} {size}\n")
        for name, g in groups:
            _, nn = _get_cell_info(name, g)
            block = np.hstack([np.full((g.n, 1), nn, dtype=np.int64),
                               g.conn[:, :nn]])
            _write_block(fh, block, "%d")
        fh.write(f"CELL_TYPES {ncell}\n")
        for name, g in groups:
            ctype, _ = _get_cell_info(name, g)
            _write_block(fh, np.full(g.n, ctype, dtype=np.int64),
                         "%d")

        fh.write(f"POINT_DATA {n}\n")
        if "DIS" in vect:
            fh.write("VECTORS DISPLACEMENT double\n")
            _write_block(fh, model.x - model.x0, "%.9E")
        if "VEL" in vect:
            fh.write("VECTORS VELOCITY double\n")
            _write_block(fh, model.v, "%.9E")
        # user node ids (ITAB), row-aligned with POINTS — appended after the
        # vectors so parsers reading the historical prefix keep working
        fh.write("SCALARS NODE_ID int 1\nLOOKUP_TABLE default\n")
        _write_block(fh, model.node_ids, "%d")

        if ncell:
            fh.write(f"CELL_DATA {ncell}\n")
            if "VONM" in elem:
                fh.write("SCALARS VONM double 1\nLOOKUP_TABLE default\n")
                for name, g in groups:
                    _write_block(fh, _von_mises(name, g), "%.9E")
            if "EPSP" in elem:
                fh.write("SCALARS EPSP double 1\nLOOKUP_TABLE default\n")
                for name, g in groups:
                    _write_block(fh, _epsp(name, g), "%.9E")
            # element status: deleted elements stay in the mesh (constant
            # topology keeps ParaView time series happy) but are flagged
            fh.write("SCALARS OFF double 1\nLOOKUP_TABLE default\n")
            for name, g in groups:
                off = g.state.get("off")
                if off is None:
                    off = np.ones(g.n)
                _write_block(fh, off, "%.1f")
            # user element / part ids in cell order (concatenated groups),
            # named like the official anim_to_vtk converter — appended last
            # so parsers reading the historical prefix keep working
            fh.write("SCALARS ELEMENT_ID int 1\nLOOKUP_TABLE default\n")
            for name, g in groups:
                _write_block(fh, g.ids, "%d")
            fh.write("SCALARS PART_ID int 1\nLOOKUP_TABLE default\n")
            for name, g in groups:
                _write_block(fh, g.state["part_ids"], "%d")
            # ---- official anim_to_vtk result arrays (M42), appended
            # behind the historical prefix.  Each array spans ALL cells:
            # rows of foreign families are zeros (downstream converters
            # consume blind token streams and rely on the counts).
            if any(name in _SHELL_FAMILIES for name, g in groups):
                layers = {name: _shell_layers(name, g)
                          for name, g in groups if name in _SHELL_FAMILIES}
                for label, comp in (("(lower)", 0), ("(upper)", 1)):
                    fh.write(f"TENSORS 2DELEM_Stress_{label} double\n")
                    for name, g in groups:
                        rows = _plane_rows(layers[name][comp]) \
                            if name in _SHELL_FAMILIES \
                            else np.zeros((g.n, 9))
                        _write_block(fh, rows, "%.9E")
                for label, comp in (("Lower", 2), ("Upper", 3)):
                    fh.write(f"SCALARS 2DELEM_Plastic_Strain_{label} "
                             f"double 1\nLOOKUP_TABLE default\n")
                    for name, g in groups:
                        arr = layers[name][comp] \
                            if name in _SHELL_FAMILIES else np.zeros(g.n)
                        _write_block(fh, arr, "%.9E")
            if any(name in _SOLID_FAMILIES for name, g in groups):
                fh.write("TENSORS 3DELEM_Stress double\n")
                for name, g in groups:
                    if name in _SOLID_FAMILIES:
                        s = g.state["sig"]
                        if s.ndim == 3:
                            s = s.mean(axis=1)
                        if s.shape[1] < 6:
                            pad = np.zeros((s.shape[0], 6 - s.shape[1]), dtype=s.dtype)
                            s = np.hstack([s, pad])
                        rows = s[:, _VOIGT9]
                    else:
                        rows = np.zeros((g.n, 9))
                    _write_block(fh, rows, "%.9E")
                fh.write("SCALARS 3DELEM_Plastic_Strain double 1\n"
                         "LOOKUP_TABLE default\n")
                for name, g in groups:
                    arr = g.state.get("epsp") if name in _SOLID_FAMILIES else None
                    if arr is not None and arr.ndim >= 2:
                        arr = arr.max(axis=1)
                    _write_block(fh, arr if arr is not None
                                 else np.zeros(g.n), "%.9E")
