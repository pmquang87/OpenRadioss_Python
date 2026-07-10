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
* cell data:  VONM von Mises stress, EPSP plastic strain (per /ANIM/ELEM)
  - solids: from the stress tensor; shells: worst layer;
    trusses/springs: |axial stress| (resp. 0)
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model

_VTK_CELL = {"bricks": (12, 8), "shells": (9, 4), "trusses": (3, 2),
             "springs": (3, 2)}


def _von_mises(group_name: str, group) -> np.ndarray:
    st = group.state
    if group_name == "bricks":
        s = st["sig"]
        return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2
                              + (s[:, 1] - s[:, 2]) ** 2
                              + (s[:, 2] - s[:, 0]) ** 2)
                       + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    if group_name == "shells":
        s = st["sig"]  # (n, nip, 3)
        vm = np.sqrt(s[:, :, 0] ** 2 - s[:, :, 0] * s[:, :, 1]
                     + s[:, :, 1] ** 2 + 3.0 * s[:, :, 2] ** 2)
        return vm.max(axis=1)
    if group_name == "trusses":
        return np.abs(st["sig"])
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
        np.savetxt(fh, model.x, fmt="%.9E")
        fh.write(f"CELLS {ncell} {size}\n")
        for name, g in groups:
            nn = _VTK_CELL[name][1]
            block = np.hstack([np.full((g.n, 1), nn, dtype=np.int64), g.conn])
            np.savetxt(fh, block, fmt="%d")
        fh.write(f"CELL_TYPES {ncell}\n")
        for name, g in groups:
            np.savetxt(fh, np.full(g.n, _VTK_CELL[name][0], dtype=np.int64),
                       fmt="%d")

        fh.write(f"POINT_DATA {n}\n")
        if "DIS" in vect:
            fh.write("VECTORS DISPLACEMENT double\n")
            np.savetxt(fh, model.x - model.x0, fmt="%.9E")
        if "VEL" in vect:
            fh.write("VECTORS VELOCITY double\n")
            np.savetxt(fh, model.v, fmt="%.9E")

        if ncell:
            fh.write(f"CELL_DATA {ncell}\n")
            if "VONM" in elem:
                fh.write("SCALARS VONM double 1\nLOOKUP_TABLE default\n")
                for name, g in groups:
                    np.savetxt(fh, _von_mises(name, g), fmt="%.9E")
            if "EPSP" in elem:
                fh.write("SCALARS EPSP double 1\nLOOKUP_TABLE default\n")
                for name, g in groups:
                    np.savetxt(fh, _epsp(name, g), fmt="%.9E")
