"""
Animation states (the A-files, /ANIM) as legacy VTK.

Fortran origin: ``engine/source/output/anim/`` writes the proprietary
binary ANIM format (RunNameA001, A002, ...). The port writes **legacy VTK
unstructured grids** (RunNameA000.vtk, ...) which ParaView opens natively
as a time series — select the ``RunNameA..vtk`` file *group*.

Contents per state:
* points  = current node positions (deformed geometry)
* cells   = all elements (hexa/quad/line)
* point data: DISPLACEMENT, VELOCITY vectors (per /ANIM/VECT)
* cell data:  VONM von Mises stress, EPSP plastic strain (per /ANIM/ELEM),
  OFF element status (always written: 1 = alive, 0 = deleted by a /FAIL
  criterion or a material failure threshold — threshold/select on OFF in
  ParaView to hide the deleted elements)
  - solids: from the stress tensor; shells: worst layer;
    trusses/springs: |axial stress| (resp. 0)
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model

# group name -> (VTK cell type id, node count written). Beams write only
# their two end nodes (the 3rd is the orientation node, not geometry).
_VTK_CELL = {"bricks": (12, 8), "tetras": (10, 4), "shells": (9, 4),
             "sh3n": (5, 3), "trusses": (3, 2), "springs": (3, 2),
             "beams": (3, 2)}


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
    if group_name in ("bricks", "tetras"):
        s = st["sig"]
        return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2
                              + (s[:, 1] - s[:, 2]) ** 2
                              + (s[:, 2] - s[:, 0]) ** 2)
                       + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    if group_name in ("shells", "sh3n"):
        s = st["sig"]  # (n, nip, 3)
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
    return e.max(axis=1) if e.ndim == 2 else e


def write_anim_state(path: str, model: Model, t: float,
                     vect=("DIS", "VEL"), elem=("VONM", "EPSP")) -> None:
    n = model.numnod
    groups = list(model.element_groups())
    ncell = sum(g.n for _, g in groups)
    size = sum((1 + _VTK_CELL[name][1]) * g.n for name, g in groups)

    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"pyradioss state t={t:.9E}\n")
        fh.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {n} double\n")
        _write_block(fh, model.x, "%.9E")
        fh.write(f"CELLS {ncell} {size}\n")
        for name, g in groups:
            nn = _VTK_CELL[name][1]
            block = np.hstack([np.full((g.n, 1), nn, dtype=np.int64),
                               g.conn[:, :nn]])
            _write_block(fh, block, "%d")
        fh.write(f"CELL_TYPES {ncell}\n")
        for name, g in groups:
            _write_block(fh, np.full(g.n, _VTK_CELL[name][0], dtype=np.int64),
                         "%d")

        fh.write(f"POINT_DATA {n}\n")
        if "DIS" in vect:
            fh.write("VECTORS DISPLACEMENT double\n")
            _write_block(fh, model.x - model.x0, "%.9E")
        if "VEL" in vect:
            fh.write("VECTORS VELOCITY double\n")
            _write_block(fh, model.v, "%.9E")

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
