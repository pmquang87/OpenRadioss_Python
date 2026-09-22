"""Unit tests for /INISTA (Initial Stress, Strain, and State Mapping).

Fortran origins:
- ``starter/source/initial_conditions/inista/hm_read_inista.F``
- ``starter/source/initial_conditions/inista/lec_inistate_yfile.F``
- ``starter/source/initial_conditions/inista/yctrl.F``
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.materials import law02_johnson_cook
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.inista import (
    InistaRecord,
    apply_inista,
    apply_inista_record_to_group,
    parse_inista_deck_cards,
)
from pyradioss.starter.starter import run_starter


class TestInistaRecordDataClass:
    """Test InistaRecord properties, getters, and setters."""

    def test_record_defaults(self):
        rec = InistaRecord()
        assert rec.part_id is None
        assert rec.elem_id is None
        assert rec.sigma_xx is None
        assert rec.epsp is None
        assert not rec.has_stress
        assert np.allclose(rec.sigma, np.zeros(6))
        assert np.allclose(rec.sigma_b, np.zeros(6))

    def test_sigma_getter_setter(self):
        rec = InistaRecord()
        rec.sigma = [100.0, 50.0, -25.0, 12.5, 3.0, -4.0]
        assert rec.sigma_xx == 100.0
        assert rec.sigma_yy == 50.0
        assert rec.sigma_zz == -25.0
        assert rec.sigma_xy == 12.5
        assert rec.sigma_yz == 3.0
        assert rec.sigma_zx == -4.0
        assert np.allclose(rec.sigma, [100.0, 50.0, -25.0, 12.5, 3.0, -4.0])

    def test_sigma_b_getter_setter(self):
        rec = InistaRecord()
        # 3-element vector: [b_xx, b_yy, b_xy]
        rec.sigma_b = [30.0, 15.0, 5.0]
        assert rec.sigma_b_xx == 30.0
        assert rec.sigma_b_yy == 15.0
        assert rec.sigma_b_xy == 5.0
        assert np.allclose(rec.sigma_b[:4], [30.0, 15.0, 0.0, 5.0])

        # 4+ element vector: [b_xx, b_yy, b_zz, b_xy]
        rec.sigma_b = [40.0, 20.0, 0.0, 8.0]
        assert rec.sigma_b_xx == 40.0
        assert rec.sigma_b_yy == 20.0
        assert rec.sigma_b_xy == 8.0


class TestInistaSolidAndShellMapping:
    """Test mapping initial stress and strain to solid and shell element groups."""

    def test_solid_constant_stress_and_epsp(self):
        """Constant 3D Cauchy stress and plastic strain on solid brick elements."""
        n_elems = 3
        group = ElementGroup(
            ids=np.array([101, 102, 103], dtype=np.int64),
            conn=np.zeros((n_elems, 8), dtype=np.int64),
            part=np.zeros(n_elems, dtype=np.int64),
        )
        group.state = {
            "sig": np.zeros((n_elems, 6), dtype=np.float64),
            "epsp": np.zeros(n_elems, dtype=np.float64),
            "part_ids": np.array([2, 2, 2], dtype=np.int64),
        }

        rec = InistaRecord(
            elem_id=102,
            sigma_xx=120.0, sigma_yy=60.0, sigma_zz=-30.0,
            sigma_xy=15.0, sigma_yz=5.0, sigma_zx=-8.0,
            epsp=0.035,
        )
        apply_inista_record_to_group(rec, "bricks", group, [1])

        # Element 102 (local index 1) is updated
        assert np.allclose(group.state["sig"][1], [120.0, 60.0, -30.0, 15.0, 5.0, -8.0])
        assert abs(group.state["epsp"][1] - 0.035) < 1e-12

        # Elements 101 and 103 remain zero
        assert np.allclose(group.state["sig"][0], np.zeros(6))
        assert np.allclose(group.state["sig"][2], np.zeros(6))
        assert group.state["epsp"][0] == 0.0
        assert group.state["epsp"][2] == 0.0

    def test_shell_constant_stress_all_layers(self):
        """Constant in-plane Cauchy stress mapped across all shell integration points."""
        n_elems = 2
        nip = 3
        group = ElementGroup(
            ids=np.array([1, 2], dtype=np.int64),
            conn=np.zeros((n_elems, 4), dtype=np.int64),
            part=np.zeros(n_elems, dtype=np.int64),
        )
        group.state = {
            "sig": np.zeros((n_elems, nip, 3), dtype=np.float64),
            "epsp": np.zeros((n_elems, nip), dtype=np.float64),
            "part_ids": np.array([1, 1], dtype=np.int64),
        }

        rec = InistaRecord(
            elem_id=1,
            sigma_xx=150.0, sigma_yy=75.0, sigma_xy=25.0,
            epsp=0.02,
        )
        apply_inista_record_to_group(rec, "shells", group, [0])

        for k in range(nip):
            assert np.allclose(group.state["sig"][0, k], [150.0, 75.0, 25.0])
            assert abs(group.state["epsp"][0, k] - 0.02) < 1e-12

        # Element 2 untouched
        assert np.allclose(group.state["sig"][1], 0.0)
        assert np.allclose(group.state["epsp"][1], 0.0)

    def test_shell_through_thickness_bending(self):
        """Through-thickness bending distribution: sigma_k = sigma_mem + xi_k * sigma_bend."""
        n_elems = 1
        nip = 3
        # Gauss points in [-1, 1] for nip=3 are [-sqrt(3/5), 0, +sqrt(3/5)]
        xi_expected = np.sqrt(3.0 / 5.0)

        group = ElementGroup(
            ids=np.array([10], dtype=np.int64),
            conn=np.zeros((n_elems, 4), dtype=np.int64),
            part=np.zeros(n_elems, dtype=np.int64),
        )
        group.state = {
            "sig": np.zeros((n_elems, nip, 3), dtype=np.float64),
            "epsp": np.zeros((n_elems, nip), dtype=np.float64),
            "part_ids": np.array([1], dtype=np.int64),
        }

        s_mem = [100.0, 50.0, 20.0]
        s_bend = [30.0, 15.0, 5.0]

        rec = InistaRecord(
            elem_id=10,
            sigma_xx=s_mem[0], sigma_yy=s_mem[1], sigma_xy=s_mem[2],
            sigma_b_xx=s_bend[0], sigma_b_yy=s_bend[1], sigma_b_xy=s_bend[2],
            epsp=0.01,
        )
        apply_inista_record_to_group(rec, "shells", group, [0])

        # Bottom layer (xi ~ -0.7746)
        expected_bottom = np.array(s_mem) - xi_expected * np.array(s_bend)
        # Middle layer (xi = 0)
        expected_mid = np.array(s_mem)
        # Top layer (xi ~ +0.7746)
        expected_top = np.array(s_mem) + xi_expected * np.array(s_bend)

        assert np.allclose(group.state["sig"][0, 0], expected_bottom, atol=1e-10)
        assert np.allclose(group.state["sig"][0, 1], expected_mid, atol=1e-10)
        assert np.allclose(group.state["sig"][0, 2], expected_top, atol=1e-10)

    def test_shell_specific_layer_mapping(self):
        """Update a specific layer only using InistaRecord(layer=k)."""
        n_elems = 1
        nip = 3
        group = ElementGroup(
            ids=np.array([5], dtype=np.int64),
            conn=np.zeros((n_elems, 4), dtype=np.int64),
            part=np.zeros(n_elems, dtype=np.int64),
        )
        group.state = {
            "sig": np.zeros((n_elems, nip, 3), dtype=np.float64),
            "epsp": np.zeros((n_elems, nip), dtype=np.float64),
            "part_ids": np.array([1], dtype=np.int64),
        }

        # Modify layer 2 (middle layer, 1-indexed) only
        rec = InistaRecord(
            elem_id=5, layer=2,
            sigma_xx=88.0, sigma_yy=44.0, sigma_xy=11.0,
            epsp=0.07,
        )
        apply_inista_record_to_group(rec, "shells", group, [0])

        assert np.allclose(group.state["sig"][0, 0], 0.0)
        assert np.allclose(group.state["sig"][0, 1], [88.0, 44.0, 11.0])
        assert np.allclose(group.state["sig"][0, 2], 0.0)
        assert group.state["epsp"][0, 0] == 0.0
        assert abs(group.state["epsp"][0, 1] - 0.07) < 1e-12
        assert group.state["epsp"][0, 2] == 0.0


class TestInistaPartMapping:
    """Test mapping initial conditions by Part ID across multiple element groups."""

    def test_part_level_mapping(self):
        model = Model()
        model.parts = {
            1: Part(id=1, prop_id=1, mat_id=1, title="ShellPart"),
            2: Part(id=2, prop_id=2, mat_id=1, title="SolidPart"),
        }
        model.parts_list = [model.parts[1], model.parts[2]]

        # Shell group for Part 1 (2 elements)
        sh_grp = ElementGroup(
            ids=np.array([1, 2], dtype=np.int64),
            conn=np.zeros((2, 4), dtype=np.int64),
            part=np.array([0, 0], dtype=np.int64),
        )
        sh_grp.state = {
            "sig": np.zeros((2, 3, 3), dtype=np.float64),
            "epsp": np.zeros((2, 3), dtype=np.float64),
            "part_ids": np.array([1, 1], dtype=np.int64),
        }
        model.shells = sh_grp

        # Solid group for Part 2 (2 elements)
        br_grp = ElementGroup(
            ids=np.array([101, 102], dtype=np.int64),
            conn=np.zeros((2, 8), dtype=np.int64),
            part=np.array([1, 1], dtype=np.int64),
        )
        br_grp.state = {
            "sig": np.zeros((2, 6), dtype=np.float64),
            "epsp": np.zeros(2, dtype=np.float64),
            "part_ids": np.array([2, 2], dtype=np.int64),
        }
        model.bricks = br_grp

        # Apply /INISTA/PART records
        records = [
            InistaRecord(part_id=1, sigma_xx=100.0, sigma_yy=50.0, sigma_xy=10.0, epsp=0.01),
            InistaRecord(part_id=2, sigma_xx=200.0, sigma_yy=150.0, sigma_zz=50.0, epsp=0.05),
        ]
        apply_inista(model, records=records)

        # Verify shells got Part 1 state
        for elem_idx in (0, 1):
            for k in range(3):
                assert np.allclose(sh_grp.state["sig"][elem_idx, k], [100.0, 50.0, 10.0])
                assert abs(sh_grp.state["epsp"][elem_idx, k] - 0.01) < 1e-12

        # Verify bricks got Part 2 state
        for elem_idx in (0, 1):
            assert np.allclose(br_grp.state["sig"][elem_idx], [200.0, 150.0, 50.0, 0.0, 0.0, 0.0])
            assert abs(br_grp.state["epsp"][elem_idx] - 0.05) < 1e-12


class TestInistaPrehardeningPhysics:
    """Verify that non-zero initial plastic strain elevates the yield stress accordingly."""

    def test_epsp_elevates_yield_stress(self):
        """Elements with initial epsp resist yielding up to their pre-hardened threshold."""
        # Material: Johnson-Cook LAW2: sigma_y = A + B * eps_p^n
        # A = 200 MPa, B = 300 MPa, n = 0.5
        # epsp = 0.0  -> sigma_y = 200 MPa
        # epsp = 0.04 -> sigma_y = 200 + 300 * sqrt(0.04) = 200 + 300 * 0.2 = 260 MPa
        mat = Material(
            id=1, law=2, rho0=7.8e-9, title="MAT_JC",
            params={"E": 210000.0, "nu": 0.3, "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0}
        )

        # Baseline: Virgin element (epsp = 0.0)
        sig_virgin = np.zeros((1, 6), dtype=np.float64)
        epsp_virgin = np.array([0.0], dtype=np.float64)

        # Pre-hardened element (epsp = 0.04)
        sig_prehard = np.zeros((1, 6), dtype=np.float64)
        epsp_prehard = np.array([0.04], dtype=np.float64)

        # Apply a strain increment that produces trial stress of ~230 MPa
        # (above virgin yield of 200 MPa, but below pre-hardened yield of 260 MPa)
        # In uniaxial tension: deps_xx = sigma / E = 230 / 210000 ~ 1.0952e-3
        deps = np.array([[1.0952e-3, -0.3 * 1.0952e-3, -0.3 * 1.0952e-3, 0.0, 0.0, 0.0]])
        dt = 1e-6

        # Step 1: Virgin element should yield (accumulate epsp > 0)
        law02_johnson_cook.solid_update(mat, sig_virgin, deps.copy(), epsp_virgin, dt)
        assert epsp_virgin[0] > 0.0, "Virgin material should have yielded"

        # Step 2: Pre-hardened element should remain purely ELASTIC (epsp unchanged at 0.04)
        law02_johnson_cook.solid_update(mat, sig_prehard, deps.copy(), epsp_prehard, dt)
        assert abs(epsp_prehard[0] - 0.04) < 1e-12, "Pre-hardened material should not have yielded"

        # Step 3: Apply much larger strain (~350 MPa trial stress) -> both yield further
        deps_large = np.array([[2.5e-3, -0.3 * 2.5e-3, -0.3 * 2.5e-3, 0.0, 0.0, 0.0]])
        law02_johnson_cook.solid_update(mat, sig_prehard, deps_large, epsp_prehard, dt)
        assert epsp_prehard[0] > 0.04, "Pre-hardened material should yield once stress exceeds 260 MPa"


class TestInistaKinematicBackstress:
    """Test initial backstress tensor mapping for kinematic hardening."""

    def test_alpha_backstress_mapped(self):
        group = ElementGroup(
            ids=np.array([1], dtype=np.int64),
            conn=np.zeros((1, 8), dtype=np.int64),
            part=np.zeros(1, dtype=np.int64),
        )
        group.state = {
            "sig": np.zeros((1, 6), dtype=np.float64),
            "epsp": np.zeros(1, dtype=np.float64),
        }

        alpha_init = np.array([50.0, -25.0, -25.0, 10.0, 0.0, 0.0])
        rec = InistaRecord(elem_id=1, alpha=alpha_init)
        apply_inista_record_to_group(rec, "bricks", group, [0])

        assert "alpha" in group.state
        assert np.allclose(group.state["alpha"][0], alpha_init)


class TestInistaDeckParsingAndStarter:
    """Test /INISTA keyword deck parsing through starter."""

    _DECK_BOILERPLATE = """\
/BEGIN
TEST_INISTA_FULL_DECK
      2021         0
/MAT/LAW1/1
Steel
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PROP/TYPE14/2
Solid_Prop
         1         1         0         0         0         0         0
/PART/1
Part_Shell
         1         1
/PART/2
Part_Solid
         2         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
       201                 0.0                 0.0                 0.0
       202                10.0                 0.0                 0.0
       203                10.0                10.0                 0.0
       204                 0.0                10.0                 0.0
       205                 0.0                 0.0                10.0
       206                10.0                 0.0                10.0
       207                10.0                10.0                10.0
       208                 0.0                10.0                10.0
/SHELL/1
         1       101       102       103       104
/BRICK/2
         2       201       202       203       204       205       206       207       208
"""

    def test_inista_part_deck(self, tmp_path):
        """Run starter on a deck with /INISTA/PART cards."""
        deck = f"""\
{self._DECK_BOILERPLATE}
/INISTA/PART/1
150.0 75.0 0.0 25.0 0.0 0.0 0.02
20.0 10.0 5.0
/INISTA/PART/2
300.0 200.0 100.0 30.0 10.0 5.0 0.05
/END
"""
        p = tmp_path / "INISTA_TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        log = MessageLog()
        model = run_starter(str(p), log)
        assert len(log.errors) == 0

        # Verify shells received Part 1 initial state
        assert hasattr(model, "shells") and model.shells is not None
        assert np.all(model.shells.state["epsp"] == 0.02)
        # Verify bending was applied: bottom layer != top layer
        assert not np.allclose(model.shells.state["sig"][0, 0], model.shells.state["sig"][0, -1])

        # Verify brick received Part 2 initial state
        assert hasattr(model, "bricks") and model.bricks is not None
        assert np.allclose(model.bricks.state["sig"][0], [300.0, 200.0, 100.0, 30.0, 10.0, 5.0])
        assert abs(model.bricks.state["epsp"][0] - 0.05) < 1e-12

    def test_inista_stress_and_epsp_keywords(self, tmp_path):
        """Run starter on a deck with /INISTA/STRESS and /INISTA/EPSP."""
        deck = f"""\
{self._DECK_BOILERPLATE}
/INISTA/STRESS
2 180.0 90.0 -45.0 15.0 0.0 -10.0
/INISTA/EPSP
2 0.08
/END
"""
        p = tmp_path / "INISTA_ELEM_TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        log = MessageLog()
        model = run_starter(str(p), log)
        assert len(log.errors) == 0

        assert np.allclose(model.bricks.state["sig"][0], [180.0, 90.0, -45.0, 15.0, 0.0, -10.0])
        assert abs(model.bricks.state["epsp"][0] - 0.08) < 1e-12

    def test_parse_inista_deck_cards_helper(self):
        """Test parse_inista_deck_cards helper function on free text cards."""
        cards = [
            "# Comment line",
            "PART 10 100.0 50.0 0.0 10.0 0.0 0.0 0.01",
            "101 200.0 100.0 50.0 20.0 10.0 5.0 0.03",
            "202 LAYER 2 80.0 40.0 15.0 0.05",
        ]
        recs = parse_inista_deck_cards(cards)
        assert len(recs) == 3

        # Record 1: Part
        assert recs[0].part_id == 10
        assert recs[0].sigma_xx == 100.0
        assert recs[0].epsp == 0.01

        # Record 2: Element
        assert recs[1].elem_id == 101
        assert recs[1].sigma_xx == 200.0
        assert recs[1].epsp == 0.03

        # Record 3: Layer
        assert recs[2].elem_id == 202
        assert recs[2].layer == 2
        assert recs[2].sigma_xx == 80.0
        assert recs[2].epsp == 0.05
