"""
Tests for pyradioss/input/deck_writer.py (M36) — the fixed-format 2022
deck writer.

Three angles:

1. FIELD-LAYOUT tests: emitted cards are checked column-by-column against
   the authoritative hm_cfg_files CARD layouts (the format strings the
   real Fortran Starter parses with), for a representative set:
   /NODE (%10d%20lg%20lg%20lg), /CLOAD (fct/DIR/skew/sens/grnod +
   Ascale/Fscale columns), /INTER/TYPE7 (card A columns + blank cards),
   /RBODY (the Mass/grnd/ICoG dual-encoding columns).

2. ROUND-TRIP test: a port-dialect deck covering every emitter family is
   converted to the fixed format, then BOTH decks are parsed with the
   port's own reader chain — the resulting models must carry identical
   entity counts and identical key values.

3. Engine-deck conversion: cards pass through verbatim, /STOP is dropped
   (documented decision — the real Engine reader dies on the block).
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyradioss.common.messages import MessageLog                 # noqa: E402
from pyradioss.input import deck_writer as dw                    # noqa: E402
from pyradioss.input.deck_reader import read_deck                # noqa: E402
from pyradioss.input.starter_keywords import parse_starter_deck  # noqa: E402
from pyradioss.model.model import Model                          # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def parse_lines(path):
    """Parse a starter deck file into a Model with the port's own chain."""
    log = MessageLog()
    model = Model()
    parse_starter_deck(read_deck(str(path)), model, log)
    errors = [m for m in getattr(log, "messages", [])
              if getattr(m, "level", "") == "ERROR"]
    return model, log


def block_lines(text, header_prefix):
    """The lines of the block starting with header_prefix (incl. header)."""
    lines = text.splitlines()
    out, active = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("/"):
            active = s.startswith(header_prefix)
        if active:
            out.append(ln)
    return out


def data_cards(lines):
    """Block lines minus header and comments (blank cards KEPT)."""
    return [ln for ln in lines[1:] if not ln.lstrip().startswith("#")]


# ----------------------------------------------------------------------------
# 1. field layout vs the cfg card definitions
# ----------------------------------------------------------------------------

class TestFieldLayouts:
    def test_node_card_layout(self):
        """SETS/node.cfg: CARD('%10d%20lg%20lg%20lg', id, x, y, z)."""
        d = dw.StarterDeck("T")
        d.node([(42, 1.5, -2.0, 3.25e-3)])
        card = [ln for ln in block_lines(d.render(), "/NODE")
                if not ln.lstrip().startswith("#")][1]
        assert card[0:10] == "        42"          # %10d
        assert card[10:30].strip() == "1.5"        # %20lg, right-justified
        assert card[10:30] == f"{'1.5':>20}"
        assert card[30:50] == f"{'-2.0':>20}"
        assert card[50:70] == f"{'0.00325':>20}"
        assert float(card[50:70]) == 3.25e-3

    def test_cload_card_layout(self):
        """LOADS/cload.cfg: %10d %10s %10d(skew) %10d(sens) %10d(grnod)
        <10 blank> %20lg(Ascale_x) %20lg(Fscale_y).  skew/sens/Ascale
        blank, so the port token stream is [fct, DIR, grnod, Fscale]."""
        d = dw.StarterDeck("T")
        d.cload(1, "tip load", fct=7, direction="x", grnod=12, scale=-0.5)
        card = data_cards(block_lines(d.render(), "/CLOAD/1"))[1]
        assert card[0:10] == "         7"
        assert card[10:20] == "         X"          # DIR right-justified
        assert card[20:30].strip() == ""            # skew blank
        assert card[30:40].strip() == ""            # sens blank
        assert card[40:50] == "        12"          # grnod
        assert card[60:80].strip() == ""            # Ascale_x blank -> 1.0
        assert float(card[80:100]) == -0.5          # Fscale_y
        # port view: exactly four tokens, in the port's documented order
        assert card.split() == ["7", "X", "12", "-0.5"]

    def test_inter_type7_layout(self):
        """INTER/inter_type7.cfg (radioss2020): card A columns + the
        blank B/C/E/F cards the port must not see."""
        d = dw.StarterDeck("T")
        d.inter_type7(3, "contact", grnod=0, surf=1, istf=2, igap=1,
                      stfac=1.0, fric=0.25, gapmin=0.05, gapmax=0.5)
        cards = data_cards(block_lines(d.render(), "/INTER/TYPE7/3"))
        # cards: title, A, B(blank), C(blank), D, E(blank), F(blank)
        assert len(cards) == 7
        a = cards[1]
        assert a[0:10] == "         0"              # grnod (self-impact)
        assert a[10:20] == "         1"              # surf
        assert a[20:30] == "         2"              # Istf
        assert a[30:40].strip() == ""                # Ithe blank
        assert a[40:50] == "         1"              # Igap
        assert cards[2].strip() == "" and cards[3].strip() == ""
        dd = cards[4]
        assert float(dd[0:20]) == 1.0                # Stfac
        assert float(dd[20:40]) == 0.25              # Fric
        assert float(dd[40:60]) == 0.05              # GAPmin
        assert float(dd[60:80]) == 0.5   # port gap_max (real Tstart RESIDUE)
        assert cards[5].strip() == "" and cards[6].strip() == ""
        # port token view of the block: [0 1 2 1] then [stfac fric gap gapmax]
        toks = [c.split() for c in cards if c.strip()]
        assert toks[1] == ["0", "1", "2", "1"]
        assert toks[2] == ["1.0", "0.25", "0.05", "0.5"]

    def test_rbody_mass_dual_encoding(self):
        """RBODY/rbody.cfg (radioss2021): node sens skew Ispher Mass grnd
        Ikrem ICoG — with added mass the skew column carries the slave
        group id (identity /SKEW/FIX emitted) and grnd id == ICoG."""
        d = dw.StarterDeck("T")
        d.rbody(1, "impactor", master=9999, grnod=10, mass=2.0e-3, icog=1,
                grnod_members=("PART", [2]))
        text = d.render()
        card = data_cards(block_lines(text, "/RBODY/1"))[1]
        assert card[0:10] == "      9999"            # node_ID
        assert card[10:20].strip() == ""             # sens blank
        assert card[20:30] == "         1"           # Skew_ID = aux grp id
        assert card[30:40].strip() == ""             # Ispher blank
        assert float(card[40:60]) == 2.0e-3          # Mass (%20lg)
        assert card[60:70] == "         1"           # grnd_ID = ICoG
        assert card[80:90] == "         1"           # ICoG
        # port token view decodes as [master, grnod, mass, icog(=1)]
        assert card.split() == ["9999", "1", "0.002", "1", "1"]
        # the identity skew and the auxiliary group copy are in the deck
        assert block_lines(text, "/SKEW/FIX/1")
        assert block_lines(text, "/GRNOD/PART/1")

    def test_law42_blank_nu_card(self):
        """MAT/matl42_Ogden.cfg: rho / nu-card / mu / BLANK / alpha /
        BLANK — the nu card is blank (hm_read_mat42.F defaults 0.495)."""
        d = dw.StarterDeck("T")
        d.mat_law42(1, "rubber", 1.0e-6, [0.0008, -0.0002], [2.0, -2.0])
        cards = data_cards(block_lines(d.render(), "/MAT/LAW42/1"))
        assert len(cards) == 7           # title rho nu mu blank alpha blank
        assert cards[2].strip() == ""                # nu card blank
        assert float(cards[3][0:20]) == 0.0008       # mu_1
        assert float(cards[3][20:40]) == -0.0002     # mu_2
        assert cards[4].strip() == ""                # mu_6..10 blank
        assert float(cards[5][0:20]) == 2.0          # alpha_1
        assert cards[6].strip() == ""                # alpha_6..10 blank

    def test_law4_card_layout(self):
        """MAT/matl4_hyd_jcook.cfg: RHO / E nu / A B n eps_max sig_max /
        Pmin / C eps_dot_0 M Tmelt Tmax / RHOCP blank(40) T0."""
        d = dw.StarterDeck("T")
        d.mat_law4(1, "jcook", rho=7.85e-3, e=210000.0, nu=0.3,
                   a=250.0, b=400.0, n=0.4, eps_max=0.5, sig_max=800.0,
                   p_min=-500.0, c=0.05, eps_dot_0=1.0, m=1.0,
                   tmelt=1800.0, tmax=2000.0, rhocp=3.5e6, t0=300.0)
        cards = data_cards(block_lines(d.render(), "/MAT/LAW4/1"))
        assert len(cards) == 7
        assert cards[0] == "jcook"
        assert float(cards[1][0:20]) == 7.85e-3
        assert float(cards[2][0:20]) == 210000.0
        assert float(cards[2][20:40]) == 0.3
        assert float(cards[3][0:20]) == 250.0
        assert float(cards[3][20:40]) == 400.0
        assert float(cards[3][40:60]) == 0.4
        assert float(cards[3][60:80]) == 0.5
        assert float(cards[3][80:100]) == 800.0
        assert float(cards[4][0:20]) == -500.0
        assert float(cards[5][0:20]) == 0.05
        assert float(cards[5][20:40]) == 1.0
        assert float(cards[5][40:60]) == 1.0
        assert float(cards[5][60:80]) == 1800.0
        assert float(cards[5][80:100]) == 2000.0
        assert float(cards[6][0:20]) == 3.5e6
        assert cards[6][20:60] == " " * 40
        assert float(cards[6][60:80]) == 300.0

    def test_law10_card_layout(self):
        """MAT/matl10_law10.cfg (radioss2020): TITLE / RHO_I RHO_O / E Nu /
        A0 A1 A2 Amax / C0 C1 C2 C3 / Pmin Pext / B Mu_max."""
        d = dw.StarterDeck("T")
        d.mat_law10(
            mat_id=1,
            title="soil_sample",
            rho0=2.0e-9,
            rhor=2.0e-9,
            e=50000.0,
            nu=0.25,
            a0=15.0,
            a1=0.4,
            a2=0.005,
            amax=500.0,
            c0=10.0,
            c1=40000.0,
            c2=100.0,
            c3=50.0,
            pmin=-1e6,
            pext=101325.0,
            b=45000.0,
            mue_max=0.35,
        )
        cards = data_cards(block_lines(d.render(), "/MAT/LAW10/1"))
        assert len(cards) == 7
        assert cards[0] == "soil_sample"
        # Card 2: RHO_I, RHO_O
        assert float(cards[1][0:20]) == 2.0e-9
        assert float(cards[1][20:40]) == 2.0e-9
        # Card 3: E, Nu
        assert float(cards[2][0:20]) == 50000.0
        assert float(cards[2][20:40]) == 0.25
        # Card 4: A0, A1, A2, Amax
        assert float(cards[3][0:20]) == 15.0
        assert float(cards[3][20:40]) == 0.4
        assert float(cards[3][40:60]) == 0.005
        assert float(cards[3][60:80]) == 500.0
        # Card 5: C0, C1, C2, C3
        assert float(cards[4][0:20]) == 10.0
        assert float(cards[4][20:40]) == 40000.0
        assert float(cards[4][40:60]) == 100.0
        assert float(cards[4][60:80]) == 50.0
        # Card 6: Pmin, Pext
        assert float(cards[5][0:20]) == -1e6
        assert float(cards[5][20:40]) == 101325.0
        # Card 7: B, Mu_max
        assert float(cards[6][0:20]) == 45000.0
        assert float(cards[6][20:40]) == 0.35

    def test_law10_card_layouts_constants_and_aliases(self):
        from pyradioss.input.card_layouts import (
            MAT_LAW10_1, MAT_LAW10_2, MAT_LAW10_3, MAT_LAW10_4,
            MAT_LAW10_5, MAT_LAW10_6, MAT_LAW10_7,
            MAT_LAW10_CFG_1, MAT_LAW10_CFG_2, MAT_LAW10_CFG_3, MAT_LAW10_CFG_4,
            MAT_LAW10_CFG_5, MAT_LAW10_CFG_6, MAT_LAW10_CFG_7,
            CARD_LAYOUTS,
        )
        assert MAT_LAW10_1 == (100,)
        assert MAT_LAW10_2 == (20, 20)
        assert MAT_LAW10_3 == (20, 20)
        assert MAT_LAW10_4 == (20, 20, 20, 20)
        assert MAT_LAW10_5 == (20, 20, 20, 20)
        assert MAT_LAW10_6 == (20, 20)
        assert MAT_LAW10_7 == (20, 20)

        assert MAT_LAW10_CFG_1 == (100,)
        assert MAT_LAW10_CFG_2 == (20, 20)
        assert MAT_LAW10_CFG_3 == (20, 20)
        assert MAT_LAW10_CFG_4 == (20, 20, 20, 20)
        assert MAT_LAW10_CFG_5 == (20, 20, 20, 20)
        assert MAT_LAW10_CFG_6 == (20, 20)
        assert MAT_LAW10_CFG_7 == (20, 20)

        assert CARD_LAYOUTS["MAT_LAW10_1"] == [100]
        assert CARD_LAYOUTS["MAT_LAW10_2"] == [20, 20]
        assert CARD_LAYOUTS["MAT_LAW10_3"] == [20, 20]
        assert CARD_LAYOUTS["MAT_LAW10_4"] == [20, 20, 20, 20]
        assert CARD_LAYOUTS["MAT_LAW10_5"] == [20, 20, 20, 20]
        assert CARD_LAYOUTS["MAT_LAW10_6"] == [20, 20]
        assert CARD_LAYOUTS["MAT_LAW10_7"] == [20, 20]

        assert CARD_LAYOUTS["MAT_LAW10_CFG_1"] == [100]
        assert CARD_LAYOUTS["MAT_SOIL_CFG_1"] == [100]
        assert CARD_LAYOUTS["MAT_DPRAG1_1"] == [100]
        assert CARD_LAYOUTS["MAT_DPRAG1_CFG_1"] == [100]
        assert CARD_LAYOUTS["MAT_DPRAG_1"] == [20, 20]
        assert CARD_LAYOUTS["MAT_DPRAG_CFG_1"] == [20, 20]

    def test_law10_deck_writer_unit_id_and_aliases(self):
        d = dw.StarterDeck("T")
        d.mat_law10(mat_id=101, title="with_unit", rho0=1.5e-9, unit_id=3)
        cards = data_cards(block_lines(d.render(), "/MAT/LAW10/101/3"))
        assert len(cards) == 7
        assert cards[0] == "with_unit"
        assert float(cards[1][0:20]) == 1.5e-9

        # Test aliases mat_soil and mat_dprag1 for LAW10 on StarterDeck
        assert d.mat_soil == d.mat_law10
        assert d.mat_dprag1 == d.mat_law10

    def test_fmt_float_roundtrip(self):
        for v in (0.3, 7.8e-6, -9.81e-3, 1e30, 12345.6789012345,
                  5000000000000.0):
            assert float(dw.fmt_float(v)) == v
            assert len(dw.fmt_float(v)) == 20


# ----------------------------------------------------------------------------
# 2. round-trip: port-dialect deck -> fixed format -> same parsed model
# ----------------------------------------------------------------------------

PORT_DECK = """\
/BEGIN
round-trip coverage deck
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
9 0.0 0.0 2.0
10 1.0 0.0 2.0
11 1.0 1.0 2.0
12 0.0 1.0 2.0
99 0.5 0.5 5.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/SHELL/2
2 5 6 7 8
/PART/1
solid part
1 1
/PART/2
shell part
2 2
/MAT/LAW2/1
steel
7.8e-6
210.0 0.3
0.4 0.5 0.5
/MAT/LAW42/2
rubber
1.0e-6
0.0008 -0.0002
2.0 -2.0
0.495
/FAIL/BIQUAD/1
0.60 0.45 0.35 0.25 0.30
/PROP/SOLID/1
solid prop
1.1 0.05 0.1
/PROP/SHELL/2
shell prop
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.5
/FUNCT/1
ramp
0.0 0.0
0.02 1.0
100.0 1.0
/GRNOD/NODE/3
bottom
1 2 3 4
/GRNOD/NODE/2
top
9 10 11 12
/GRNOD/PART/10
solid nodes
1
/BCS/1
clamp
111 111 0 3
/INIVEL/TRA/1
push
0.0 0.0 -1.0 2
/GRAV/1
gravity
1 Z 0 -9.81e-3
/CLOAD/1
pull
1 X 2 0.5
/IMPVEL/1
drive (scale != 1 -> auxiliary scaled funct)
1 Z 2 -2.0
/SURF/PART/1
solid surface
1
/RBODY/1
rigid block with ballast
99 10 2.0e-3 1
/INTER/TYPE7/1
contact
0 1 2 1
1.0 0.1 0.0 0.5
/INTER/TYPE2/2
tied
1 1 1.5
/RWALL/PLANE/1
ground
0 0 0.0 0.0
0.0 0.0 0.0
0.0 0.0 1.0
/SECT/1
cut
2
/TH/NODE/1
watch
DX VX
9
/TH/PART/2
energies
IE KE
1
/END
"""


class TestRoundTrip:
    @pytest.fixture()
    def decks(self, tmp_path):
        old = tmp_path / "RT_old_0000.rad"
        old.write_text(PORT_DECK)
        new = tmp_path / "RT_new_0000.rad"
        dw.write_starter_from_port_lines(PORT_DECK.splitlines(), str(new),
                                         runname="RT")
        return old, new

    def test_same_model_counts(self, decks):
        old, new = decks
        mo, _ = parse_lines(old)
        mn, _ = parse_lines(new)
        assert len(mn.node_ids) == len(mo.node_ids) == 13
        for etype in ("BRICK", "SHELL"):
            assert len(mn.raw_elems[etype]) == len(mo.raw_elems[etype])
        assert set(mn.parts) == set(mo.parts)
        assert set(mn.materials) == set(mo.materials)
        assert set(mn.properties) == set(mo.properties)
        assert set(mn.surfaces) == set(mo.surfaces)
        assert len(mn.bcs) == len(mo.bcs)
        assert len(mn.inivel) == len(mo.inivel)
        assert len(mn.gravity) == len(mo.gravity)
        assert len(mn.cloads) == len(mo.cloads)
        assert len(mn.impvel) == len(mo.impvel)
        assert len(mn.rbodies) == len(mo.rbodies)
        assert len(mn.interfaces) == len(mo.interfaces)
        assert len(mn.rwalls) == len(mo.rwalls)
        assert len(mn.sections) == len(mo.sections)
        assert len(mn.th_requests) == len(mo.th_requests)
        assert len(mn.raw_fails) == len(mo.raw_fails)
        # the aux scaled function is the ONLY extra entity (+1 funct,
        # +1 node group for the /RBODY dual-encoding)
        assert len(mn.functions) == len(mo.functions) + 1
        assert len(mn.node_groups) == len(mo.node_groups) + 1

    def test_same_key_values(self, decks):
        old, new = decks
        mo, _ = parse_lines(old)
        mn, _ = parse_lines(new)
        # materials
        assert mn.materials[1].params["E"] == mo.materials[1].params["E"]
        assert mn.materials[1].params["nu"] == mo.materials[1].params["nu"]
        assert mn.materials[2].params["mu"] == mo.materials[2].params["mu"]
        assert mn.materials[2].params["nu"] == mo.materials[2].params["nu"]
        # shell thickness (the M35 field-order trap: Istrain vs Thick)
        assert mn.properties[2].params["thick"] == \
            mo.properties[2].params["thick"] == 0.5
        assert mn.properties[2].params["nip"] == 3
        # contact
        io, in_ = mo.interfaces[0], mn.interfaces[0]
        assert (in_.istf, in_.igap) == (io.istf, io.igap) == (2, 1)
        assert in_.stfac == io.stfac == 1.0
        assert in_.fric == io.fric == 0.1
        assert in_.gap_max == io.gap_max == 0.5
        t2o = [i for i in mo.interfaces if i.type == 2][0]
        t2n = [i for i in mn.interfaces if i.type == 2][0]
        assert t2n.dsearch == t2o.dsearch == 1.5
        # rigid wall geometry
        assert list(mn.rwalls[0].normal) == list(mo.rwalls[0].normal)
        assert list(mn.rwalls[0].point) == list(mo.rwalls[0].point)
        assert mn.rwalls[0].grnod_id == mo.rwalls[0].grnod_id
        # rigid body: mass + icog survive the dual-encoding; the slave
        # group id moves to the auxiliary copy but its MEMBERS are equal
        ro, rn = mo.rbodies[0], mn.rbodies[0]
        assert rn.master_id == ro.master_id == 99
        assert rn.added_mass == ro.added_mass == 2.0e-3
        assert rn.icog == ro.icog == 1
        go = mo.node_groups[ro.grnod_id]
        gn = mn.node_groups[rn.grnod_id]
        assert (gn.node_ids, gn.part_ids, gn.box_ids) == \
            (go.node_ids, go.part_ids, go.box_ids)
        # loads: CLOAD scale token order [fct DIR grnod scale]
        assert mn.cloads[0].scale == mo.cloads[0].scale == 0.5
        assert mn.cloads[0].grnod_id == mo.cloads[0].grnod_id == 2
        assert list(mn.gravity[0].direction) == list(mo.gravity[0].direction)
        assert mn.gravity[0].scale == mo.gravity[0].scale == -9.81e-3
        # IMPVEL scale folded into the auxiliary function: the effective
        # velocity history v(t) = scale*f(t) must be identical
        ivo, ivn = mo.impvel[0], mn.impvel[0]
        assert ivn.dof == ivo.dof
        assert ivn.grnod_id == ivo.grnod_id
        fo = mo.functions[ivo.funct_id]
        fn = mn.functions[ivn.funct_id]
        for xo, yo, xn, yn in zip(fo.x, fo.y, fn.x, fn.y):
            assert xn == xo
            assert yn * ivn.scale == yo * ivo.scale

    def test_no_parse_errors(self, decks):
        _, new = decks
        _, log = parse_lines(new)
        text = "\n".join(str(m) for m in getattr(log, "messages", []))
        assert "ERROR" not in text.upper() or not any(
            getattr(m, "level", "") == "ERROR"
            for m in getattr(log, "messages", []))


# ----------------------------------------------------------------------------
# 3. engine deck conversion
# ----------------------------------------------------------------------------

class TestEngineDeck:
    def test_stop_dropped_rest_verbatim(self, tmp_path):
        src = ["#RADIOSS ENGINE", "/RUN/T/1", "0.2", "/DT", "0.9  0.0",
               "/TFILE", "0.002", "/STOP", "15.0", "/PRINT/-100"]
        out = tmp_path / "T_0001.rad"
        dw.write_engine_from_port_lines(src, str(out))
        text = out.read_text().splitlines()
        body = [ln for ln in text if not ln.startswith("#")]
        assert body == ["/RUN/T/1", "0.2", "/DT", "0.9  0.0",
                        "/TFILE", "0.002", "/PRINT/-100"]
        # the dropped threshold is recorded in a comment
        assert any("/STOP 15.0" in ln for ln in text if ln.startswith("#"))

    def test_impl_cards_pass_through(self, tmp_path):
        src = ["/RUN/F/1", "1.0", "/IMPL", "/IMPL/FATIG/BASE",
               "0.0  250.0  3000  10  0  5",
               "5.0  5000000000000.0  0.03  0.0  0.0  500.0  20200"]
        out = tmp_path / "F_0001.rad"
        dw.write_engine_from_port_lines(src, str(out))
        body = [ln for ln in out.read_text().splitlines()
                if not ln.startswith("#")]
        assert body == src
