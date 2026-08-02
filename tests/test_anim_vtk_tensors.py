"""Value pins of the official anim_to_vtk tensor / per-layer cell arrays
(anim_vtk.write_anim_state) — the full stress state, not just VONM/EPSP.

Every assertion pins hand-set or engine-produced VALUES, because the
placement is unforgiving: any Voigt permutation, layer mix-up or frame
error changes the numbers, not just the names.

* ``TENSORS 3DELEM_Stress`` — solid GLOBAL-frame Cauchy stress (the port
  solids are the Isolid=1 Jaumann formulation: sig stays in the fixed
  global basis, solid_hexa8._post contracts it against global gradients).
  The Voigt state [xx, yy, zz, xy, yz, zx] lands in the 3x3 as
  ``[[s0,s3,s4],[s3,s1,s5],[s4,s5,s2]]`` — the official anim_to_vtk
  placement (yz at (0,2), zx at (1,2)), which the vtk2d3plot converter
  maps straight back to the d3plot slots.
* ``TENSORS 2DELEM_Stress_(lower)/(upper)`` — per-layer shell in-plane
  stress, ELEMENT-LOCAL plane stress ``[[sxx,sxy,0],[sxy,syy,0],[0,0,0]]``
  exactly like the official tool (transverse shear stays out).  lower =
  layer 0, upper = layer nip-1 of each part slice (leggauss stations
  ascend bottom -t/2 -> top +t/2); QBAT reduces its 4 in-plane Gauss
  points by the mean (the GBUF%FOR convention).  Orthotropic slices are
  rotated fiber -> element frame first (rot_stress_m2e).
* ``SCALARS 3DELEM_Plastic_Strain / 2DELEM_Plastic_Strain_Lower/_Upper``
  with the same layer selection.
* zero-fill: every cell-data array spans ALL cells; rows of foreign
  families are exact zeros (downstream parsers read blind token streams).
* the historical block prefix (VONM/EPSP/OFF/ids) is untouched — the
  ID-less port-dialect consumers keep reading their fixed positions.
* an engine run (bending plasticity) round-trips the final model state
  through the %.9E format bit-exactly.
"""

import contextlib
import glob
import io
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.output.anim_vtk import write_anim_state
from pyradioss.starter.starter import run_starter

from tests.test_anim_vtk_schema import _parse_vtk

# ---------------------------------------------------------------------------
# decks
# ---------------------------------------------------------------------------

# one brick (cell 0) + one nip=3 shell (cell 1): the state arrays are set by
# hand, so every emitted number is known in advance
STARTER_BS = """\
/BEGIN
tensor placement pin
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
/END
"""

# single QBAT (Ishell=12) shell, nip=3: pins the GP-major (n, 4*nip) layout
# reduction — layer il of GP ng sits at flat index ng*nip_max + il
STARTER_QBAT = """\
/BEGIN
qbat layer pin
/NODE
1 0 0 0
2 10 0 0
3 10 10 0
4 0 10 0
/SHELL/1
1 1 2 3 4
/PART/1
plate
1 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SHELL/1
plate
12 0 0 0
0.0 0.0 0.0 0 0
3 0 0.5
/END
"""

# solids-only / shells-only family gating decks
STARTER_SOLID_ONLY = """\
/BEGIN
solids only
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
/PART/1
cube
1 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SOLID/1
solid
1.1 0.05 0.1
/END
"""

STARTER_SHELL_ONLY = """\
/BEGIN
shells only
/NODE
1 0 0 0
2 10 0 0
3 10 10 0
4 0 10 0
/SHELL/1
1 1 2 3 4
/PART/1
plate
1 1
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
/PROP/SHELL/1
plate
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.5
/END
"""

# engine deck: LAW2 cantilever plate (2 shells, tip INIVEL -> root bending
# plasticity) + LAW2 brick sheared by its top-face INIVEL (solid epsp > 0,
# all six global stress components populated)
STARTER_RUN = """\
/BEGIN
tensor engine pin
/NODE
1 0 0 0
2 10 0 0
3 10 10 0
4 0 10 0
5 20 0 0
6 20 10 0
11 30 0 0
12 40 0 0
13 40 10 0
14 30 10 0
15 30 0 10
16 40 0 10
17 40 10 10
18 30 10 10
/SHELL/1
101 1 2 3 4
102 2 5 6 3
/BRICK/2
201 11 12 13 14 15 16 17 18
/PART/1
plate
1 1
/PART/2
cube
2 1
/MAT/LAW2/1
soft plastic
7.8e-6
210. 0.3
0.05 0.4 0.5
/PROP/SHELL/1
plate
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.5
/PROP/SOLID/2
cube
1.1 0.05 0.1
/GRNOD/NODE/1
clamp
1 4
/GRNOD/NODE/2
tip
5 6
/GRNOD/NODE/3
base
11 12 13 14
/GRNOD/NODE/4
top
15 16 17 18
/BCS/1
clamp
111 111 0 1
/BCS/2
base
111 111 0 3
/INIVEL/TRA/2
push tip
0 0 -20. 2
/FUNCT/1
step
0.0 1.0
100.0 1.0
/IMPVEL/3
top x drive
1 X 4 5.
/IMPVEL/4
top y drive
1 Y 4 3.
/IMPVEL/5
top z drive
1 Z 4 -4.
/END
"""

ENGINE_RUN = """\
/RUN/TPIN/1
0.05
/DT
0.9 0
/ANIM/DT
0 0.02
/PRINT/-1000
"""

TENSOR_BLOCKS = ("2DELEM_Stress_(lower)", "2DELEM_Stress_(upper)",
                 "3DELEM_Stress")
LAYER_SCALARS = ("2DELEM_Plastic_Strain_Lower", "2DELEM_Plastic_Strain_Upper",
                 "3DELEM_Plastic_Strain")


def _starter_model(tmp_path, text, name="PIN"):
    s = tmp_path / f"{name}_0000.rad"
    s.write_text(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(str(s))


def _write_parse(tmp_path, model, name="state.vtk"):
    p = str(tmp_path / name)
    write_anim_state(p, model, 0.5)
    return _parse_vtk(p)


def _fmt9(a):
    """The writer's %.9E view of an array — bit-exact comparisons without
    tolerance guessing (10 significant digits round-trip through double)."""
    return [f"{v:.9E}" for v in np.asarray(a, dtype=float).ravel().tolist()]


# ---------------------------------------------------------------------------
# placement pins (hand-set state)
# ---------------------------------------------------------------------------

def test_solid_tensor_placement_and_zero_fill(tmp_path):
    """Voigt [xx,yy,zz,xy,yz,zx] -> anim_to_vtk 3x3 rows
    [s0 s3 s4 / s3 s1 s5 / s4 s5 s2]; the shell cell's row is exact zeros."""
    model = _starter_model(tmp_path, STARTER_BS)
    model.bricks.state["sig"][0] = [11.0, 22.0, 33.0, 12.0, 23.0, 31.0]
    model.bricks.state["epsp"][0] = 0.25
    doc = _write_parse(tmp_path, model)
    t3 = doc["arrays"]["3DELEM_Stress"]
    assert t3.shape == (2, 3, 3)
    assert t3[0].tolist() == [[11.0, 12.0, 23.0],
                              [12.0, 22.0, 31.0],
                              [23.0, 31.0, 33.0]]
    assert t3[1].tolist() == [[0.0] * 3] * 3
    assert doc["arrays"]["3DELEM_Plastic_Strain"].tolist() == [0.25, 0.0]


def test_shell_layer_selection_and_zero_fill(tmp_path):
    """lower = layer 0, upper = layer nip-1 as [[sxx,sxy,0],[sxy,syy,0],
    [0,0,0]]; the mid layer appears nowhere; the brick's rows are zeros."""
    model = _starter_model(tmp_path, STARTER_BS)
    sh = model.shells.state
    sh["sig"][0, 0] = [1.0, 2.0, 3.0]
    sh["sig"][0, 1] = [4.0, 5.0, 6.0]
    sh["sig"][0, 2] = [7.0, 8.0, 9.0]
    sh["epsp"][0] = [0.101, 0.202, 0.303]
    doc = _write_parse(tmp_path, model)
    lo = doc["arrays"]["2DELEM_Stress_(lower)"]
    up = doc["arrays"]["2DELEM_Stress_(upper)"]
    assert lo[1].tolist() == [[1.0, 3.0, 0.0], [3.0, 2.0, 0.0], [0.0] * 3]
    assert up[1].tolist() == [[7.0, 9.0, 0.0], [9.0, 8.0, 0.0], [0.0] * 3]
    assert lo[0].tolist() == [[0.0] * 3] * 3        # brick cell
    assert up[0].tolist() == [[0.0] * 3] * 3
    assert doc["arrays"]["2DELEM_Plastic_Strain_Lower"].tolist() == [0.0,
                                                                     0.101]
    assert doc["arrays"]["2DELEM_Plastic_Strain_Upper"].tolist() == [0.0,
                                                                     0.303]
    # the mid layer (4/5/6, 0.202) must not leak into any emitted array
    emitted = np.concatenate(
        [doc["arrays"][k].ravel() for k in TENSOR_BLOCKS + LAYER_SCALARS])
    for leak in (4.0, 5.0, 6.0, 0.202):
        assert leak not in emitted


def test_qbat_gp_major_layer_mean(tmp_path):
    """QBAT (n, 4*nip) GP-major state: layer il of GP ng at ng*nip_max+il;
    the element value is the 4-GP mean of the requested layer."""
    model = _starter_model(tmp_path, STARTER_QBAT, name="QPIN")
    g = model.shells_qbat
    assert g is not None and g.n == 1
    st = g.state
    nip_max = st["nip_max"]
    assert nip_max == 3
    for ng in range(4):
        st["sig"][0, ng * nip_max + 0, 0] = float(ng + 1)       # bottom sxx
        st["sig"][0, ng * nip_max + 2, 1] = 10.0 * (ng + 1)     # top syy
        st["epsp"][0, ng * nip_max + 0] = 0.1 * (ng + 1)
        st["epsp"][0, ng * nip_max + 2] = 0.01 * (ng + 1)
    doc = _write_parse(tmp_path, model)
    lo = doc["arrays"]["2DELEM_Stress_(lower)"]
    up = doc["arrays"]["2DELEM_Stress_(upper)"]
    assert lo[0].tolist() == [[2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0] * 3]
    assert up[0].tolist() == [[0.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0] * 3]
    assert doc["arrays"]["2DELEM_Plastic_Strain_Lower"].tolist() == [0.25]
    assert doc["arrays"]["2DELEM_Plastic_Strain_Upper"].tolist() == [0.025]


def test_ortho_slice_rotated_fiber_to_element():
    """Orthotropic slices store sig in the FIBER frame; the writer must
    rotate it back to the element frame.  Fiber at 90 deg (c,s)=(0,1):
    [s11,s22,t12] -> [s22,s11,-t12] (rotov.F), hand-pinned."""
    from pyradioss.output.anim_vtk import _shell_layers
    sig = np.zeros((1, 2, 3))
    sig[0, 0] = [1.0, 2.0, 3.0]
    sig[0, 1] = [4.0, 5.0, 6.0]
    ep = np.array([[0.1, 0.2]])
    zst = np.array([-0.5, 0.5])
    group = SimpleNamespace(n=1, state=dict(
        slices=[(slice(0, 1), None, SimpleNamespace(type=9, params={}))],
        zw=[(zst, np.abs(zst))], sig=sig, epsp=ep,
        ortho=np.array([[0.0, 1.0]])))
    lo, up, eplo, epup = _shell_layers("shells", group)
    assert lo[0].tolist() == pytest.approx([2.0, 1.0, -3.0])
    assert up[0].tolist() == pytest.approx([5.0, 4.0, -6.0])
    assert eplo.tolist() == [0.1] and epup.tolist() == [0.2]
    # non-ortho prop type on the same state: raw fiber==element passthrough
    group.state["slices"] = [(slice(0, 1), None,
                              SimpleNamespace(type=1, params={}))]
    lo, up, _, _ = _shell_layers("shells", group)
    assert lo[0].tolist() == [1.0, 2.0, 3.0]
    assert up[0].tolist() == [4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# family gating + schema
# ---------------------------------------------------------------------------

def test_family_gating(tmp_path):
    """3DELEM blocks only when solids exist, 2DELEM only when shells do."""
    m_solid = _starter_model(tmp_path, STARTER_SOLID_ONLY, name="SOL")
    doc = _write_parse(tmp_path, m_solid, "solid.vtk")
    assert "3DELEM_Stress" in doc["arrays"]
    assert "3DELEM_Plastic_Strain" in doc["arrays"]
    for k in ("2DELEM_Stress_(lower)", "2DELEM_Stress_(upper)",
              "2DELEM_Plastic_Strain_Lower", "2DELEM_Plastic_Strain_Upper"):
        assert k not in doc["arrays"]
    m_shell = _starter_model(tmp_path, STARTER_SHELL_ONLY, name="SHE")
    doc = _write_parse(tmp_path, m_shell, "shell.vtk")
    assert "2DELEM_Stress_(lower)" in doc["arrays"]
    assert "2DELEM_Stress_(upper)" in doc["arrays"]
    assert "3DELEM_Stress" not in doc["arrays"]
    assert "3DELEM_Plastic_Strain" not in doc["arrays"]


def test_blocks_appended_after_historical_prefix(tmp_path):
    """The port-dialect prefix stays byte-positional; the official-dialect
    arrays are appended behind PART_ID in a pinned order."""
    model = _starter_model(tmp_path, STARTER_BS)
    p = str(tmp_path / "order.vtk")
    write_anim_state(p, model, 0.5)
    doc = _parse_vtk(p)
    assert doc["order"] == [
        "FIELD", "POINTS", "CELLS", "CELL_TYPES", "POINT_DATA",
        "VECTORS DISPLACEMENT", "VECTORS VELOCITY", "SCALARS NODE_ID",
        "CELL_DATA", "SCALARS VONM", "SCALARS EPSP", "SCALARS OFF",
        "SCALARS ELEMENT_ID", "SCALARS PART_ID",
        "TENSORS 2DELEM_Stress_(lower)", "TENSORS 2DELEM_Stress_(upper)",
        "SCALARS 2DELEM_Plastic_Strain_Lower",
        "SCALARS 2DELEM_Plastic_Strain_Upper",
        "TENSORS 3DELEM_Stress", "SCALARS 3DELEM_Plastic_Strain"]


# ---------------------------------------------------------------------------
# engine run: real data round-trips bit-exactly, layers genuinely differ
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("tensorrun")
    s, e = d / "TPIN_0000.rad", d / "TPIN_0001.rad"
    s.write_text(STARTER_RUN)
    e.write_text(ENGINE_RUN)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(s))
        model = run_engine(str(e))
    anims = sorted(glob.glob(str(d / "TPINA*.vtk")))
    assert len(anims) >= 2, "engine produced too few animation states"
    return {"model": model, "last": anims[-1], "anims": anims}


def test_engine_solid_tensor_roundtrip(engine_run):
    """The last anim state carries the final global-frame solid stress and
    plastic strain, %.9E-bit-exact against the model state; all six Voigt
    components of the sheared brick are alive and the file tensor's von
    Mises equals the VONM scalar."""
    model = engine_run["model"]
    doc = _parse_vtk(engine_run["last"])
    # cell order: bricks first (element_groups order), then the two shells
    assert doc["arrays"]["ELEMENT_ID"].tolist() == [201, 101, 102]
    s6 = model.bricks.state["sig"][0]
    # xx/yy/zz plus BOTH out-of-plane shears carry real values (the top
    # face is pushed in x, y and z); sxy alone is exempt (no load path)
    assert np.abs(s6[[0, 1, 2, 4, 5]]).min() > 1.0e-4, \
        "triaxial shear deck must load the yz/zx slots"
    t = doc["arrays"]["3DELEM_Stress"][0]
    assert np.array_equal(t, t.T), "emitted tensor must be symmetric"
    assert _fmt9([t[0, 0], t[1, 1], t[2, 2],
                  t[0, 1], t[0, 2], t[1, 2]]) == _fmt9(s6)
    ep = doc["arrays"]["3DELEM_Plastic_Strain"]
    assert model.bricks.state["epsp"][0] > 1.0e-4, "brick must yield"
    assert _fmt9(ep[:1]) == _fmt9(model.bricks.state["epsp"][:1])
    vm_t = np.sqrt(0.5 * ((t[0, 0] - t[1, 1]) ** 2 + (t[1, 1] - t[2, 2]) ** 2
                          + (t[2, 2] - t[0, 0]) ** 2)
                   + 3.0 * (t[0, 1] ** 2 + t[0, 2] ** 2 + t[1, 2] ** 2))
    assert vm_t == pytest.approx(doc["arrays"]["VONM"][0], rel=1.0e-8)


def test_engine_shell_layers_roundtrip(engine_run):
    """Root shell: real bending -> lower and upper stress genuinely differ,
    both %.9E-bit-exact against state layers 0 / nip-1, plane-stress rows
    zero, and per-layer plastic strain > 0 lands in the right slot."""
    model = engine_run["model"]
    doc = _parse_vtk(engine_run["last"])
    st = model.shells.state
    lo_f = doc["arrays"]["2DELEM_Stress_(lower)"][1]     # root shell, eid 101
    up_f = doc["arrays"]["2DELEM_Stress_(upper)"][1]
    for tf in (lo_f, up_f):
        assert tf[0, 2] == tf[1, 2] == tf[2, 2] == 0.0
        assert np.array_equal(tf, tf.T)
    lo_s, up_s = st["sig"][0, 0], st["sig"][0, 2]        # nip = 3
    assert _fmt9([lo_f[0, 0], lo_f[1, 1], lo_f[0, 1]]) == _fmt9(lo_s)
    assert _fmt9([up_f[0, 0], up_f[1, 1], up_f[0, 1]]) == _fmt9(up_s)
    # bending: the two fibers must carry genuinely different stress
    assert np.abs(lo_s - up_s).max() > 1.0e-3
    ep_lo = doc["arrays"]["2DELEM_Plastic_Strain_Lower"]
    ep_up = doc["arrays"]["2DELEM_Plastic_Strain_Upper"]
    assert st["epsp"][0, 0] > 1.0e-4 and st["epsp"][0, 2] > 1.0e-4, \
        "root shell must yield at both outer fibers"
    assert _fmt9(ep_lo[1:2]) == _fmt9(st["epsp"][0:1, 0])
    assert _fmt9(ep_up[1:2]) == _fmt9(st["epsp"][0:1, 2])
    # solid rows of the 2D blocks are exact zeros
    assert ep_lo[0] == ep_up[0] == 0.0
    assert doc["arrays"]["2DELEM_Stress_(lower)"][0].tolist() == \
        [[0.0] * 3] * 3


def test_engine_vonm_epsp_backward_compatible(engine_run):
    """The port-dialect scalars still carry the historical reductions on a
    deck with live plasticity (worst-layer VONM / max-layer EPSP)."""
    model = engine_run["model"]
    doc = _parse_vtk(engine_run["last"])
    st = model.shells.state
    s = st["sig"]
    vm = np.sqrt(s[:, :, 0] ** 2 - s[:, :, 0] * s[:, :, 1] + s[:, :, 1] ** 2
                 + 3.0 * s[:, :, 2] ** 2).max(axis=1)
    assert _fmt9(doc["arrays"]["VONM"][1:3]) == _fmt9(vm)
    assert _fmt9(doc["arrays"]["EPSP"][1:3]) == _fmt9(st["epsp"].max(axis=1))
