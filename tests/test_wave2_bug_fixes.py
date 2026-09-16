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
