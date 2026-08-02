"""Pin of the legacy-VTK animation-state schema (anim_vtk.write_anim_state).

Downstream consumers exist (ParaView time series, the VTK->d3plot
converter), but nothing asserted the file layout until now.  Pinned here:

* header/title lines and the ``t=`` title time (backward compatibility),
* the ``FIELD FieldData 2`` block with TIME/CYCLE (official anim_to_vtk
  naming), TIME identical to the title time, CYCLE wired from the engine,
* block order — the historical blocks keep their positions, the id arrays
  are appended after them,
* NODE_ID / ELEMENT_ID / PART_ID values against the deck's user numbering
  and against the model arrays they mirror (node_ids, group ids,
  state['part_ids'] concatenated in element_groups() order),
* every announced count is consumed exactly (the hand parser refuses
  short/overlong blocks), and meshio can read the file where installed.
"""

import contextlib
import glob
import io
import os
import re

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.output.anim_vtk import write_anim_state
from pyradioss.starter.starter import run_starter

# one brick (part 1, eid 501) + one shell on its top face (part 2, eid 302)
# with deliberately non-sequential user ids so identity mapping would fail
STARTER = """\
/BEGIN
vtk schema pin
/NODE
101 0 0 0
102 10 0 0
103 10 10 0
104 0 10 0
105 0 0 10
106 10 0 10
107 10 10 10
108 0 10 10
/BRICK/1
501 101 102 103 104 105 106 107 108
/SHELL/2
302 105 106 107 108
/PART/1
cube
1 1
/PART/2
lid
2 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SOLID/1
solid
1.1 0.05 0.1
/PROP/SHELL/2
lid
1 0 0 0
0.01 0.01 0.01 0 0
3 0 1.0
/GRNOD/PART/1
all
1 2
/INIVEL/TRA/1
push
0 0 -0.5 1
/END
"""

ENGINE = """\
/RUN/VTKS/1
0.02
/DT
0.9 0
/ANIM/DT
0 0.008
/PRINT/-1000
/STOP
15.0
"""


def _parse_vtk(path):
    """Strict hand parser for the port's legacy-ASCII dialect: every block
    must consume exactly the announced number of values, with no value
    spilling into the next block's lines."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    doc = {"header": lines[0], "title": lines[1], "fmt": lines[2],
           "dataset": lines[3], "order": [], "field": {}, "arrays": {}}
    i = 4
    npoints = ncur = 0

    def take(count, conv):
        nonlocal i
        vals = []
        while len(vals) < count:
            vals.extend(lines[i].split())
            i += 1
        assert len(vals) == count, \
            f"block ends mid-line at {path}:{i} ({len(vals)} != {count})"
        return np.array([conv(v) for v in vals])

    while i < len(lines):
        parts = lines[i].split()
        key = parts[0]
        i += 1
        doc["order"].append(" ".join(parts[:2]) if key in
                            ("SCALARS", "VECTORS", "TENSORS") else key)
        if key == "FIELD":
            for _ in range(int(parts[2])):
                name, ncomp, ntup, typ = lines[i].split()
                i += 1
                conv = float if typ == "double" else int
                doc["field"][name] = take(int(ncomp) * int(ntup), conv)
        elif key == "POINTS":
            npoints = int(parts[1])
            doc["arrays"]["POINTS"] = take(3 * npoints,
                                           float).reshape(npoints, 3)
        elif key == "CELLS":
            doc["arrays"]["CELLS"] = take(int(parts[2]), int)
        elif key == "CELL_TYPES":
            doc["arrays"]["CELL_TYPES"] = take(int(parts[1]), int)
        elif key == "POINT_DATA":
            ncur = int(parts[1])
            assert ncur == npoints
        elif key == "CELL_DATA":
            ncur = int(parts[1])
            assert ncur == len(doc["arrays"]["CELL_TYPES"])
        elif key == "VECTORS":
            doc["arrays"][parts[1]] = take(3 * ncur, float).reshape(ncur, 3)
        elif key == "SCALARS":
            assert lines[i] == "LOOKUP_TABLE default"
            i += 1
            conv = int if parts[2] == "int" else float
            doc["arrays"][parts[1]] = take(ncur, conv)
        elif key == "TENSORS":
            doc["arrays"][parts[1]] = take(9 * ncur,
                                           float).reshape(ncur, 3, 3)
        else:
            raise AssertionError(f"unknown block {key} at {path}:{i}")
    return doc


@pytest.fixture(scope="module")
def vtk_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("vtkschema")
    s, e = d / "VTKS_0000.rad", d / "VTKS_0001.rad"
    s.write_text(STARTER)
    e.write_text(ENGINE)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(s))
        model = run_engine(str(e))
    anims = sorted(glob.glob(str(d / "VTKSA*.vtk")))
    assert len(anims) >= 2, "engine produced too few animation states"
    return {"model": model, "anims": anims,
            "out": str(d / "VTKS_0001.out")}


def test_header_title_and_field_block(vtk_run):
    for path in vtk_run["anims"]:
        doc = _parse_vtk(path)
        assert doc["header"] == "# vtk DataFile Version 3.0"
        assert doc["fmt"] == "ASCII"
        assert doc["dataset"] == "DATASET UNSTRUCTURED_GRID"
        # title keeps the historical t= (parsers predating FIELD read it)
        m = re.fullmatch(r"pyradioss state t=(\S+)", doc["title"])
        assert m
        # FIELD is the first block and TIME repeats the title time exactly
        assert doc["order"][0] == "FIELD"
        assert doc["field"]["TIME"] == pytest.approx([float(m.group(1))],
                                                     abs=0.0)
        assert f"{doc['field']['TIME'][0]:.9E}" == m.group(1)
        assert doc["field"]["CYCLE"].dtype.kind == "i"


def test_cycle_wired_from_engine(vtk_run):
    cycles = [int(_parse_vtk(p)["field"]["CYCLE"][0])
              for p in vtk_run["anims"]]
    assert cycles == sorted(cycles)
    # the final state is written after the loop, at the run's cycle count
    text = open(vtk_run["out"]).read()
    total = int(re.search(r"CYCLES\s*\.[ .]*:\s*(\d+)", text).group(1))
    assert cycles[-1] == total > 0


def test_block_order_backward_compatible(vtk_run):
    """The historical blocks keep their relative order; the id arrays are
    appended after them (downstream parsers read a fixed prefix)."""
    doc = _parse_vtk(vtk_run["anims"][0])
    expected = ["FIELD", "POINTS", "CELLS", "CELL_TYPES", "POINT_DATA",
                "VECTORS DISPLACEMENT", "VECTORS VELOCITY",
                "SCALARS NODE_ID", "CELL_DATA", "SCALARS VONM",
                "SCALARS EPSP", "SCALARS OFF", "SCALARS ELEMENT_ID",
                "SCALARS PART_ID",
                # official anim_to_vtk result arrays (M42), behind the
                # historical prefix like the id arrays before them
                "TENSORS 2DELEM_Stress_(lower)",
                "TENSORS 2DELEM_Stress_(upper)",
                "SCALARS 2DELEM_Plastic_Strain_Lower",
                "SCALARS 2DELEM_Plastic_Strain_Upper",
                "TENSORS 3DELEM_Stress", "SCALARS 3DELEM_Plastic_Strain"]
    assert doc["order"] == expected


def test_id_arrays_carry_user_numbering(vtk_run):
    model = vtk_run["model"]
    for path in vtk_run["anims"]:
        doc = _parse_vtk(path)
        # points row i <-> node_ids[i]; the deck numbers nodes 101..108
        assert doc["arrays"]["NODE_ID"].tolist() == list(range(101, 109))
        assert np.array_equal(doc["arrays"]["NODE_ID"], model.node_ids)
        # cells in element_groups() order: the brick then the shell
        assert doc["arrays"]["CELL_TYPES"].tolist() == [12, 9]
        assert doc["arrays"]["ELEMENT_ID"].tolist() == [501, 302]
        assert doc["arrays"]["PART_ID"].tolist() == [1, 2]
        groups = list(model.element_groups())
        assert np.array_equal(
            doc["arrays"]["ELEMENT_ID"],
            np.concatenate([g.ids for _, g in groups]))
        assert np.array_equal(
            doc["arrays"]["PART_ID"],
            np.concatenate([g.state["part_ids"] for _, g in groups]))


def test_direct_call_cycle_default_and_explicit(vtk_run, tmp_path):
    model = vtk_run["model"]
    p0 = str(tmp_path / "default.vtk")
    write_anim_state(p0, model, 0.125)
    doc = _parse_vtk(p0)
    assert doc["field"]["CYCLE"].tolist() == [0]
    assert doc["field"]["TIME"].tolist() == [0.125]
    assert doc["title"] == "pyradioss state t=1.250000000E-01"
    p1 = str(tmp_path / "explicit.vtk")
    write_anim_state(p1, model, 0.25, cycle=42)
    assert _parse_vtk(p1)["field"]["CYCLE"].tolist() == [42]


def test_meshio_reads_the_file(vtk_run):
    meshio = pytest.importorskip("meshio")
    mesh = meshio.read(vtk_run["anims"][-1])
    assert len(mesh.points) == 8
    for name in ("DISPLACEMENT", "VELOCITY", "NODE_ID"):
        assert name in mesh.point_data
    for name in ("VONM", "EPSP", "OFF", "ELEMENT_ID", "PART_ID"):
        assert name in mesh.cell_data
