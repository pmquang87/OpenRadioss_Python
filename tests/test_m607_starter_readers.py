"""
Tests for Milestone M607: Comprehensive Starter & Input Reader Completion.

Verifies:
1. /SPHIO and /SPH/INOUT, INLET, OUTLET parsing into model.sph_inouts.
2. /JOINT and /CYL_JO aliases mapping to cylindrical joints.
3. /LINK alias mapping to rigid links (model.rlinks).
4. /MADYMO/LINK and /MADYMO/EXFEM parsing.
5. /MERGE/RBODY, /PRELOAD/BOLT, /BCS/WALL parsing.
6. /DAMP/INTER, /DAMP/RANGE, /DAMP/FUNCT parsing.
7. /GAUGE/POINT, /SECT/CIRCLE, /SECT/PARAL, /SURF/SURF parsing.
8. Initial conditions with numeric IDs and title handling (/INIBRI, /INISHE, /INISH3, /INITRU, /INIBEA, /INISPR, /INIQUA, /INISPHCEL, /INIBRI/EREF, /INIVEL).
9. /FAIL/WINDSHIELD and /FAIL/WINDSHIELD_ALTER mapping to ALTER.
10. Silent bypass of /PRIVATE/METADATA/FATXML and engine control keywords.
11. StarterDeck / DeckWriter serializers and roundtrip conversion.
"""

from __future__ import annotations

import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_writer import StarterDeck


def _parse(deck_text: str) -> tuple[Model, MessageLog]:
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck_text, model=model, log=log)
    return model, log


class TestSphInOutReaders:
    """Tests for /SPHIO and /SPH/INOUT variants."""

    def test_sphio_free_format(self):
        deck = (
            "/BEGIN\n"
            "SPHIO TEST\n"
            "/SPHIO/1\n"
            "Inlet Boundary\n"
            "10 20 0\n"
            "1.0 100.0 500.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.sph_inouts
        s = model.sph_inouts[1]
        assert s.surf_id == 10
        assert s.part_id == 20
        assert s.rho_in == pytest.approx(1.0)
        assert s.p_in == pytest.approx(100.0)
        assert s.e_in == pytest.approx(500.0)

    def test_sph_inout_aliases(self):
        deck = (
            "/BEGIN\n"
            "SPH ALIASES TEST\n"
            "/SPH/INOUT/2\n"
            "IO 2\n"
            "11 21 0\n"
            "2.0 200.0 600.0\n"
            "/SPH/INLET/3\n"
            "Inlet 3\n"
            "12 22 0\n"
            "3.0 300.0 700.0\n"
            "/SPH/OUTLET/4\n"
            "Outlet 4\n"
            "13 23 0\n"
            "4.0 400.0 800.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 2 in model.sph_inouts
        assert 3 in model.sph_inouts
        assert 4 in model.sph_inouts
        assert model.sph_inouts[2].part_id == 21
        assert model.sph_inouts[3].part_id == 22
        assert model.sph_inouts[4].part_id == 23


class TestJointAndLinkReaders:
    """Tests for /JOINT, /CYL_JO, and /LINK aliases."""

    def test_cyl_joint_aliases(self):
        deck = (
            "/BEGIN\n"
            "JOINT TEST\n"
            "/JOINT/1\n"
            "Cylindrical Joint 1\n"
            "101 102 1 0 1e-4\n"
            "/CYL_JO/2\n"
            "Cylindrical Joint 2\n"
            "201 202 2 0 1e-4\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.cyl_joints
        assert 2 in model.cyl_joints
        assert model.cyl_joints[1].node1 == 101
        assert model.cyl_joints[1].node2 == 102
        assert model.cyl_joints[2].node1 == 201
        assert model.cyl_joints[2].node2 == 202

    def test_link_alias(self):
        deck = (
            "/BEGIN\n"
            "LINK TEST\n"
            "/LINK/5\n"
            "Rigid Link 5\n"
            "1 1 1 1 1 1 0 501 0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 5 in model.rlinks
        rl = model.rlinks[5]
        assert rl.grnod_id == 501
        assert rl.dofs == (1, 1, 1, 1, 1, 1)


class TestMadymoCouplings:
    """Tests for /MADYMO/LINK and /MADYMO/EXFEM."""

    def test_madymo_keywords(self):
        deck = (
            "/BEGIN\n"
            "MADYMO TEST\n"
            "/MADYMO/LINK/1\n"
            "Madymo Link 1\n"
            "10 20\n"
            "/MADYMO/EXFEM/2\n"
            "Madymo Exfem 2\n"
            "100 200 300\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.madymo_links
        assert 2 in model.madymo_exfems
        assert model.madymo_links[1].mdref == 10
        assert model.madymo_links[1].node_id == 20
        assert model.madymo_exfems[2].part_ids == [100, 200, 300]


class TestRigidBodyPreloadBcs:
    """Tests for /MERGE/RBODY, /PRELOAD/BOLT, and /BCS/WALL."""

    def test_merge_rbody_and_preload(self):
        deck = (
            "/BEGIN\n"
            "MERGE & PRELOAD TEST\n"
            "/MERGE/RBODY/1\n"
            "Merge Rbody 1\n"
            "10 20 30\n"
            "/PRELOAD/BOLT/2\n"
            "Bolt Preload 2\n"
            "100 0 0 50000.0\n"
            "/BCS/WALL/3\n"
            "Wall BCS 3\n"
            "101 201 1.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.merge_rbodies or 1 in model.rbody_merges
        assert 2 in model.preload_bolts
        assert 3 in model.bcs_walls
        bp = model.preload_bolts[2]
        assert bp.preload == pytest.approx(50000.0)


class TestDampingReaders:
    """Tests for /DAMP/INTER, /DAMP/RANGE, /DAMP/FUNCT."""

    def test_damping_keywords(self):
        deck = (
            "/BEGIN\n"
            "DAMPING TEST\n"
            "/DAMP/INTER/1\n"
            "Damp Inter 1\n"
            "10 0.05 100.0\n"
            "/DAMP/RANGE/2\n"
            "Damp Range 2\n"
            "10 0.01 0.05 10.0 100.0\n"
            "/DAMP/FUNCT/3\n"
            "Damp Funct 3\n"
            "10 5\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.damp_inters
        assert 2 in model.damp_ranges
        assert 3 in model.damp_functs


class TestGaugesSectionsSurfaces:
    """Tests for /GAUGE/POINT, /SECT/CIRCLE, /SECT/PARAL, /SURF/SURF."""

    def test_gauges_sections_surfaces(self):
        deck = (
            "/BEGIN\n"
            "GAUGES SECTIONS TEST\n"
            "/GAUGE/POINT/1\n"
            "Gauge 1\n"
            "10 1.0 2.0 3.0\n"
            "/SECT/CIRCLE/1\n"
            "Sect Circle 1\n"
            "10 1 2 5.0\n"
            "/SECT/PARAL/2\n"
            "Sect Paral 2\n"
            "11 1 2 3 4\n"
            "/SURF/SURF/1\n"
            "Surf Surf 1\n"
            "101 102 0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.gauge_points
        assert 1 in model.sect_circles
        assert 2 in model.sect_parals
        assert 1 in model.surf_surfs


class TestInitialConditionBlocks:
    """Tests initial condition cards with numeric IDs and title-optional handling."""

    def test_inibri_numeric_id_defaults_to_stress(self):
        deck = (
            "/BEGIN\n"
            "INIBRI NUMERIC ID TEST\n"
            "/INIBRI/1\n"
            "Brick Inistate 1\n"
            "101 100.0 200.0 300.0 10.0 20.0 30.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 101 in model.ini_bricks
        st = model.ini_bricks[101]
        assert st.sigma[0] == pytest.approx(100.0)
        assert st.sigma[1] == pytest.approx(200.0)
        assert st.sigma[2] == pytest.approx(300.0)

    def test_inishe_numeric_id_defaults_to_strs_f(self):
        deck = (
            "/BEGIN\n"
            "INISHE NUMERIC ID TEST\n"
            "/INISHE/1\n"
            "Shell Inistate 1\n"
            "201 1 1\n"
            "50.0 60.0 0.0 10.0 0.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 201 in model.ini_shells

    def test_inish3_numeric_id(self):
        deck = (
            "/BEGIN\n"
            "INISH3 NUMERIC ID TEST\n"
            "/INISH3/1\n"
            "Shell3 Inistate 1\n"
            "301 1 1\n"
            "70.0 80.0 0.0 15.0 0.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 301 in model.ini_shells
        assert "INISH3_STRS_F_1" in model.ini_state_tables

    def test_1d_inistate_numeric_ids_and_no_titles(self):
        deck = (
            "/BEGIN\n"
            "1D INISTATE NUMERIC ID TEST\n"
            "/INITRU/1\n"
            "401 2 0.0 1000.0 0.05 0.0\n"
            "/INIBEA/2\n"
            "501 100.0 200.0 300.0 10.0 20.0 30.0\n"
            "/INISPR/3\n"
            "601 4\n"
            "500.0 10.0 0.0 0.0 0.0\n"
            "0.0 0.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 401 in model.ini_trusses
        assert 501 in model.ini_beams
        assert 601 in model.ini_springs
        assert model.ini_trusses[401].force == pytest.approx(1000.0)
        assert model.ini_springs[601].force == pytest.approx(500.0)

    def test_iniqua_mapping(self):
        deck = (
            "/BEGIN\n"
            "INIQUA TEST\n"
            "/INIQUA/1\n"
            "Quad Inistate\n"
            "701 1 1\n"
            "120.0 130.0 0.0 25.0 0.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 701 in model.ini_shells
        assert "INIQUA_STRS_F_1" in model.ini_state_tables

    def test_inisphcel(self):
        deck = (
            "/BEGIN\n"
            "INISPHCEL TEST\n"
            "/INISPHCEL/1\n"
            "SPH Cell State 1\n"
            "101325.0 1000.0 2.5e5 10.0 0.0 0.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.ini_sphcels
        cel = model.ini_sphcels[1]
        assert cel.p == pytest.approx(101325.0)
        assert cel.rho == pytest.approx(1000.0)
        assert cel.e == pytest.approx(2.5e5)
        assert cel.vx == pytest.approx(10.0)

    def test_inivel_numeric_id_defaults_to_tra(self):
        deck = (
            "/BEGIN\n"
            "INIVEL NUMERIC ID TEST\n"
            "/INIVEL/1\n"
            "Initial Velocity 1\n"
            "10.0 0.0 0.0 10\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert len(model.inivel) == 1
        iv = model.inivel[0]
        assert iv.id == 1
        assert iv.v[0] == pytest.approx(10.0)
        assert iv.grnod_id == 10

    def test_inibri_eref(self):
        deck = (
            "/BEGIN\n"
            "INIBRI EREF TEST\n"
            "/INIBRI_EREF/1\n"
            "101 201\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert len(model.inibri_erefs) == 1
        assert model.inibri_erefs[0].sub_objects[0]["elem_id"] == 101
        assert model.inibri_erefs[0].sub_objects[0]["ref_elem_id"] == 201


class TestFailureModelAliases:
    """Tests /FAIL/WINDSHIELD, /FAIL/WINDSHIELD_ALTER mapping to ALTER."""

    def test_fail_windshield(self):
        deck = (
            "/BEGIN\n"
            "FAIL WINDSHIELD TEST\n"
            "/FAIL/WINDSHIELD/1\n"
            "Windshield Fail\n"
            "100.0 200.0 0.05 1.0\n"
            "/FAIL/WINDSHIELD_ALTER/2\n"
            "Windshield Alter Fail\n"
            "150.0 250.0 0.06 1.0\n"
            "/END\n"
        )
        model, log = _parse(deck)
        assert len(log.errors) == 0
        assert 1 in model.fail_alters
        assert 2 in model.fail_alters
        assert len(model.raw_fails) == 2
        assert model.raw_fails[0][1].type == "ALTER"
        assert model.raw_fails[1][1].type == "ALTER"


class TestBypassAndIgnoreKeywords:
    """Tests silent bypass of /PRIVATE/METADATA/FATXML and engine keywords."""

    def test_private_metadata_and_engine_keywords(self):
        deck = (
            "/BEGIN\n"
            "METADATA AND ENGINE BYPASS TEST\n"
            "/PRIVATE/METADATA/FATXML\n"
            "<fatxml><node id='1'/></fatxml>\n"
            "/IMPL/NONLIN\n"
            "/PROC\n"
            "/ABF\n"
            "/RUN/STOP/1\n"
            "/TFILE/0.01\n"
            "/RFILE/0.05\n"
            "/STOP/0.1\n"
            "/VERS/2022\n"
            "/END\n"
        )
        model, log = _parse(deck)
        # Should have 0 errors and 0 warnings about unported keywords
        assert len(log.errors) == 0
        unported_warnings = [w for w in log.warnings if "not ported" in w]
        assert len(unported_warnings) == 0


class TestDeckWriterSerializers:
    """Tests StarterDeck serializer methods and block conversion."""

    def test_deck_writer_serializers(self):
        deck = StarterDeck("SERIALIZER_TEST")
        deck.inibri_generic("STRESS", 1, "Brick State", ["101 100.0 200.0 300.0"])
        deck.inishe_generic("STRS_F", 1, "Shell State", ["201 50.0 60.0"])
        deck.initru_generic("FULL", 1, "Truss State", ["301 0.05 1000.0"])
        deck.inibea_generic("FULL", 1, "Beam State", ["401 100.0 200.0 300.0"])
        deck.inispr_generic("FULL", 1, "Spring State", ["501 500.0"])
        deck.inisphcel_generic(1, "SPH State", ["101325.0 1000.0 2.5e5"])
        deck.sphio_generic(1, "SPHIO", ["10 20 0"])
        deck.rlink_generic(1, "Link", ["1 1 1 1 1 1 0 501 0"])
        deck.cyl_joint_generic(1, "Joint", ["101 102 1 0 1e-4"])
        rendered = deck.render()

        assert "/INIBRI/STRESS/1" in rendered
        assert "/INISHE/STRS_F/1" in rendered
        assert "/INITRU/FULL/1" in rendered
        assert "/INIBEA/FULL/1" in rendered
        assert "/INISPR/FULL/1" in rendered
        assert "/INISPHCEL/1" in rendered
        assert "/SPHIO/1" in rendered
        assert "/LINK/1" in rendered
        assert "/JOINT/1" in rendered

    def test_deck_writer_card_block_roundtrip(self):
        deck_text = (
            "/BEGIN\n"
            "ROUNDTRIP TEST\n"
            "/INIBRI/STRESS/1\n"
            "Brick State\n"
            "101 100.0 200.0 300.0\n"
            "/INISHE/STRS_F/2\n"
            "Shell State\n"
            "201 50.0 60.0\n"
            "/INITRU/FULL/3\n"
            "Truss State\n"
            "301 0.05 1000.0\n"
            "/INIBEA/FULL/4\n"
            "Beam State\n"
            "401 100.0 200.0 300.0\n"
            "/INISPR/FULL/5\n"
            "Spring State\n"
            "501 500.0\n"
            "/INISPHCEL/6\n"
            "SPH State\n"
            "101325.0 1000.0 2.5e5\n"
            "/SPHIO/7\n"
            "SPHIO State\n"
            "10 20 0\n"
            "/LINK/8\n"
            "Rigid Link\n"
            "1 1 1 1 1 1 0 501 0\n"
            "/JOINT/9\n"
            "Cyl Joint\n"
            "101 102 1 0 1e-4\n"
            "/END\n"
        )
        from pyradioss.input.deck_reader import read_deck
        blocks = read_deck(deck_text)
        assert len(blocks) >= 9

        d = StarterDeck("ROUNDTRIP")
        for b in blocks:
            d.card_block(b)
        rendered = d.render()

        assert "/INIBRI/STRESS/1" in rendered
        assert "/INISHE/STRS_F/2" in rendered
        assert "/INITRU/FULL/3" in rendered
        assert "/INIBEA/FULL/4" in rendered
        assert "/INISPR/FULL/5" in rendered
        assert "/INISPHCEL/6" in rendered
        assert "/SPHIO/7" in rendered
        assert "/LINK/8" in rendered
        assert "/JOINT/9" in rendered

