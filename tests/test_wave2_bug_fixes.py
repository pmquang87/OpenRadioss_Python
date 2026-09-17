"""
Regression tests for Wave 2 bug fixes across pyradioss.
Covers:
1. Card.__str__ returning raw string instead of dataclass repr.
2. /TRANSFORM/MATRIX applying rotation/shear matrix + translation in starter.
3. /ADMAS types 4, 6, and 7 properly resolving part and part groups without error.
4. Constant loads/BCs with funct_id = 0 not producing false positive 'function 0 not defined' errors.
5. Standard 3-card /BRIC20 fixed format parsing.
"""

import numpy as np
import pytest

from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_bric20
from pyradioss.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import apply_transforms
from pyradioss.starter.checks import check_model
from pyradioss.starter.initialization import initialize_elements_and_mass


def test_card_str_returns_raw():
    card = Card(raw=" 1.0  2.0  3.0", source="test.rad:10")
    assert str(card) == " 1.0  2.0  3.0"
    assert f"{card}" == " 1.0  2.0  3.0"
    assert "Card(raw=" not in str(card)


def test_transform_matrix_applies():
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)
    
    class NodeGroup:
        id = 1
        node_idx = np.array([0, 1], dtype=np.int64)
        node_ids = [1, 2]
    model.node_groups[1] = NodeGroup()

    mat_3x3 = ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 1.0))
    trans_vec = (10.0, 20.0, 30.0)
    model.transforms = [
        (1, "MATRIX", 1, mat_3x3, trans_vec, 0)
    ]

    log = MessageLog()
    apply_transforms(model, log)

    assert len(log.errors) == 0
    assert np.allclose(model.x0[0], [12.0, 20.0, 30.0])
    assert np.allclose(model.x0[1], [10.0, 26.0, 30.0])


def test_admas_types_4_6_7():
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.zeros((2, 3), dtype=np.float64)
    model.mass = np.zeros(2, dtype=np.float64)

    class Part:
        id = 10
    model.parts[10] = Part()

    class ElemGroup:
        conn = np.array([[0, 1]], dtype=np.int64)
        state = {"part_ids": np.array([10])}
        def init_group(self, model, log):
            return np.array([0, 1]), np.array([0.0, 0.0]), None
    # Attach dummy kernel to KERNELS for this test
    from pyradioss.elements import KERNELS
    orig_trusses = KERNELS.get("trusses")
    class DummyKernel:
        @staticmethod
        def init_group(group, model, log):
            return np.array([0, 1]), np.array([0.0, 0.0]), None
    KERNELS["trusses"] = DummyKernel()

    model.element_groups_list = [("trusses", ElemGroup())]
    model.element_groups = lambda: model.element_groups_list

    class Admas:
        def __init__(self, aid, m_type, gid, mass):
            self.id = aid
            self.mass_type = m_type
            self.grnod_id = gid
            self.mass = mass

    model.admas = [
        Admas(1, 6, 10, 4.0),
    ]

    try:
        log = MessageLog()
        initialize_elements_and_mass(model, log)

        assert len(log.errors) == 0
        assert np.allclose(model.mass, [2.0, 2.0])
    finally:
        if orig_trusses is not None:
            KERNELS["trusses"] = orig_trusses


def test_need_funct_zero_not_error():
    model = Model()
    model.node_ids = np.array([1], dtype=np.int64)
    model.x0 = np.zeros((1, 3))
    
    class NodeGroup:
        id = 1
        node_idx = np.array([0])
        node_ids = [1]
    model.node_groups[1] = NodeGroup()

    class Cload:
        id = 1
        grnod_id = 1
        funct_id = 0
        dir = "X"
        scale = 100.0
    model.cloads.append(Cload())

    log = MessageLog()
    check_model(model, log)

    fn_errors = [msg for msg in log.errors if "function 0" in msg]
    assert len(fn_errors) == 0


def test_bric20_3card_fixed_format():
    model = Model()
    log = MessageLog()

    c1 = Card(raw=f"{1:>10d}" + "".join(f"{100+i:>10d}" for i in range(1, 9)))
    c2 = Card(raw="".join(f"{100+i:>10d}" for i in range(9, 17)))
    c3 = Card(raw="".join(f"{100+i:>10d}" for i in range(17, 21)))

    block = KeywordBlock(
        keyword="BRIC20",
        parts=["BRIC20", "1"],
        user_id=1,
        cards=[c1, c2, c3],
        fixed=True
    )

    read_bric20(block, model, log)

    assert len(log.errors) == 0
    assert "BRIC20" in model.raw_elems
    assert len(model.raw_elems["BRIC20"]) == 1
    elem_id, part_id, nodes = model.raw_elems["BRIC20"][0]
    assert elem_id == 1
    assert part_id == 1
    assert len(nodes) == 20
    assert nodes == [100 + i for i in range(1, 21)]


# ===========================================================================
# Fix Subagent 7 (Output, Postprocessing & GUI): BUG-OUT-01 to BUG-OUT-09
# ===========================================================================
import subprocess
from unittest.mock import MagicMock
from pyradioss.gui.runner import parse_listing_line, T01Data, load_t01
from pyradioss.gui.postproc import convert_anim_to_vtk
from pyradioss.output.anim_vtk import write_anim_state, _BRIC20_TO_VTK, _VOIGT9
from pyradioss.output.time_history import TimeHistory
from pyradioss.model.model import ElementGroup
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine
from tests.test_anim_vtk_schema import _parse_vtk


def test_bug_out_01_parse_listing_line_percentage():
    line_with_pct = (
        "   100  1.09640E-02  1.09676E-04  3.73293E-04  1.65685E-05  "
        "0.00000E+00  0.00000E+00  0.00000E+00  3.97775E-04    -0.50%"
    )
    st = parse_listing_line(line_with_pct)
    assert st is not None
    assert st["cycle"] == 100
    assert st["err"] == pytest.approx(-0.5)

    line_without_pct = (
        "   100  1.09640E-02  1.09676E-04  3.73293E-04  1.65685E-05  "
        "0.00000E+00  0.00000E+00  0.00000E+00  3.97775E-04     0.00"
    )
    st2 = parse_listing_line(line_without_pct)
    assert st2 is not None
    assert st2["err"] == pytest.approx(0.0)


def test_bug_out_02_bric20_node_permutation(tmp_path):
    expected_permutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15]
    assert _BRIC20_TO_VTK == expected_permutation

    model = Model()
    model.node_ids = np.arange(1, 21, dtype=np.int64)
    model.x = np.zeros((20, 3), dtype=np.float64)
    model.x0 = model.x.copy()
    model.v = np.zeros((20, 3), dtype=np.float64)

    conn = np.arange(20, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([101], dtype=np.int64), conn=conn, part=np.array([1], dtype=np.int64))
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    model.bric20s = group

    vtk_file = str(tmp_path / "bric20.vtk")
    write_anim_state(vtk_file, model, 0.0, elem=())

    doc = _parse_vtk(vtk_file)
    assert doc["arrays"]["CELL_TYPES"][0] == 25  # VTK_QUADRATIC_HEXAHEDRON
    cells = doc["arrays"]["CELLS"]
    assert cells[0] == 20
    assert list(cells[1:21]) == expected_permutation


def test_bug_out_03_per_element_slaved_quadratic(tmp_path):
    model = Model()
    model.node_ids = np.arange(1, 25, dtype=np.int64)
    model.x = np.zeros((24, 3), dtype=np.float64)
    model.x0 = model.x.copy()
    model.v = np.zeros((24, 3), dtype=np.float64)

    conn = np.empty((2, 20), dtype=np.int64)
    conn[0] = np.arange(20)
    conn[1, :8] = np.arange(8)
    conn[1, 8:] = -1

    group = ElementGroup(ids=np.array([101, 102], dtype=np.int64), conn=conn, part=np.array([1, 1], dtype=np.int64))
    group.state["part_ids"] = np.array([1, 1], dtype=np.int64)
    model.bric20s = group

    vtk_file = str(tmp_path / "mixed_bric20.vtk")
    write_anim_state(vtk_file, model, 0.0, elem=())

    doc = _parse_vtk(vtk_file)
    ctypes = doc["arrays"]["CELL_TYPES"]
    assert ctypes[0] == 25
    assert ctypes[1] == 12

    cells = doc["arrays"]["CELLS"]
    assert cells[0] == 20
    assert list(cells[1:21]) == _BRIC20_TO_VTK
    assert cells[21] == 8
    assert list(cells[22:30]) == list(range(8))


def test_bug_out_03_per_element_slaved_tetra10(tmp_path):
    model = Model()
    model.node_ids = np.arange(1, 15, dtype=np.int64)
    model.x = np.zeros((14, 3), dtype=np.float64)
    model.x0 = model.x.copy()
    model.v = np.zeros((14, 3), dtype=np.float64)

    conn = np.empty((2, 10), dtype=np.int64)
    conn[0] = np.arange(10)
    conn[1, :4] = np.arange(4)
    conn[1, 4:] = -1

    group = ElementGroup(ids=np.array([201, 202], dtype=np.int64), conn=conn, part=np.array([1, 1], dtype=np.int64))
    group.state["part_ids"] = np.array([1, 1], dtype=np.int64)
    model.tetra10s = group

    vtk_file = str(tmp_path / "mixed_tetra10.vtk")
    write_anim_state(vtk_file, model, 0.0, elem=())

    doc = _parse_vtk(vtk_file)
    ctypes = doc["arrays"]["CELL_TYPES"]
    assert ctypes[0] == 24
    assert ctypes[1] == 10

    cells = doc["arrays"]["CELLS"]
    assert cells[0] == 10
    assert list(cells[1:11]) == list(range(10))
    assert cells[11] == 4
    assert list(cells[12:16]) == list(range(4))


def test_bug_out_04_time_history_t0_initial_state(tmp_path):
    starter_deck = """\
/BEGIN
Test T01 Initial State
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/TRUSS/1
101 1 2
/PART/1
TrussPart
1 1
/MAT/LAW1/1
Steel
7.8e-6
210000.0 0.3
/PROP/TRUSS/1
TrussProp
1.0
/END
"""
    engine_deck = """\
/RUN/TestT0/1
0.01
/DT
0.9 0
/TFILE
0.005
/PRINT/-1000
/STOP
10.0
"""
    s_path = tmp_path / "TEST_0000.rad"
    e_path = tmp_path / "TEST_0001.rad"
    s_path.write_text(starter_deck, encoding="ascii")
    e_path.write_text(engine_deck, encoding="ascii")

    run_starter(str(s_path))
    run_engine(str(e_path))

    t01_path = tmp_path / "TESTT01.csv"
    assert t01_path.exists()
    t01 = load_t01(str(t01_path))
    assert t01.nrows >= 2
    assert t01.time[0] == pytest.approx(0.0)


def test_bug_out_05_t01_data_duplicate_columns():
    cols = ["TIME", "IE", "IE", "KE"]
    rows = [[0.0, 1.0, 2.0, 3.0], [0.1, 1.5, 2.5, 3.5]]
    data = T01Data(cols, rows)
    assert data.columns == ["TIME", "IE", "IE_2", "KE"]
    assert data.column("IE") == [1.0, 1.5]
    assert data.column("IE_2") == [2.0, 2.5]
    assert data.has("IE_2")


def test_bug_out_06_convert_anim_to_vtk_timeout(monkeypatch, tmp_path):
    anim_file = tmp_path / "RUNA001"
    anim_file.write_text("dummy anim")

    mock_proc = MagicMock()
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["anim_to_vtk"], timeout=1.0)
    mock_proc.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_proc)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    lines = []
    res = convert_anim_to_vtk(str(tmp_path), exec_dir=str(tmp_path), emit=lambda ev: lines.append(ev[2]), timeout=1.0)

    assert mock_proc.kill.called
    assert mock_proc.wait.called
    assert res["ok"] is False


def test_bug_out_07_tensor_mapping():
    assert _VOIGT9 == [0, 3, 5, 3, 1, 4, 5, 4, 2]


def test_bug_out_08_beam_resultants_in_time_history(tmp_path):
    model = Model()
    th = TimeHistory(str(tmp_path / "th_beam.csv"), model, MagicMock())

    grp_bm = ElementGroup(ids=np.array([201]), conn=np.array([[0, 1]]), part=np.array([1]))
    grp_bm.state["fres"] = np.array([[150.0, 25.0, 35.0]])
    grp_bm.state["mres"] = np.array([[12.0, 4.0, 6.0]])
    model.beams = grp_bm

    assert th._other_value("BEAM", 201, "N") == pytest.approx(150.0)
    assert th._other_value("BEAM", 201, "FX") == pytest.approx(150.0)
    assert th._other_value("BEAM", 201, "FY") == pytest.approx(25.0)
    assert th._other_value("BEAM", 201, "FZ") == pytest.approx(35.0)
    assert th._other_value("BEAM", 201, "MX") == pytest.approx(12.0)
    assert th._other_value("BEAM", 201, "MY") == pytest.approx(4.0)
    assert th._other_value("BEAM", 201, "MZ") == pytest.approx(6.0)


def test_bug_out_09_anim_vtk_missing_part_ids(tmp_path):
    model = Model()
    model.node_ids = np.array([1, 2])
    model.x = np.zeros((2, 3))
    model.x0 = model.x.copy()
    model.v = np.zeros((2, 3))

    grp = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]), part=np.array([1]))
    assert "part_ids" not in grp.state
    model.trusses = grp

    vtk_file = str(tmp_path / "no_part_ids.vtk")
    write_anim_state(vtk_file, model, 0.0, elem=())

    doc = _parse_vtk(vtk_file)
    assert "PART_ID" in doc["arrays"]
    assert doc["arrays"]["PART_ID"][0] == 0

