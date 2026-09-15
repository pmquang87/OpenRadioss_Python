"""Tests for Milestone M157: Advanced Contact Interfaces, Non-Linear Springs & Draping Suite.

Keywords tested:
- /PROP/TYPE26, /PROP/SPR_TAB, /PROP/P26_SPR_TAB (Tabular non-linear spring property)
- /PROP/TYPE27, /PROP/SPR_BDAMP, /PROP/P27_SPR_BDAMP (Bilateral damping spring property)
- /INTER/TYPE19 (Multi-segment/multi-surface interface)
- /INTER/TYPE25 (Advanced surface-to-surface penalty contact)
- /INTER/TYPE8 (Drawbead interface)
- /DRAPE, /TABLE/DRAPE (Composite fabric draping table)
"""
from __future__ import annotations
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model, log


def test_prop_type26_fixed(tmp_path: Path):
    # Card 0: Mass(20), blank(30), ISENSOR(10), ISFLAG(10), Ileng(10)
    # Card 1: NFUNC(10), NRATEN(10), SCALE(20), STIFF0(20), blank(20), ALPHA1(20)
    # Cards: FUN_LOAD(10), SCALE_LOAD(20), STRAINRATE_LOAD(20)
    # Cards: FUN_UNLOAD(10), SCALE_UNLOAD(20), STRAINRATE_UNLOAD(20)
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/PROP/TYPE26/101\n"
        "Nonlinear Tabular Spring\n"
        "                0.25                              10         1         2\n"
        "         2         1                 1.5                100.                                     0.1\n"
        "         5                 1.0                 0.0\n"
        "         6                 1.2                10.0\n"
        "         7                 0.8                 0.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    p = m.properties[101]
    assert p.type == 26
    assert p.title == "Nonlinear Tabular Spring"
    assert p.params["mass"] == pytest.approx(0.25)
    assert p.params["isensor"] == 10
    assert p.params["isflag"] == 1
    assert p.params["ileng"] == 2
    assert p.params["nfunc"] == 2
    assert p.params["nraten"] == 1
    assert p.params["scale"] == pytest.approx(1.5)
    assert p.params["stiff0"] == pytest.approx(100.0)
    assert p.params["alpha1"] == pytest.approx(0.1)
    assert len(p.params["load_curves"]) == 2
    assert p.params["load_curves"][0] == {"fun_load": 5, "scale_load": 1.0, "strainrate_load": 0.0}
    assert p.params["load_curves"][1] == {"fun_load": 6, "scale_load": 1.2, "strainrate_load": 10.0}
    assert len(p.params["unload_curves"]) == 1
    assert p.params["unload_curves"][0] == {"fun_unload": 7, "scale_unload": 0.8, "strainrate_unload": 0.0}


def test_prop_spr_tab_free(tmp_path: Path):
    deck = (
        "/PROP/SPR_TAB/202\n"
        "Free format Tabular Spring\n"
        "0.5 0 0 1\n"
        "1 0 2.0 50.0 0.05\n"
        "11 1.0 0.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    p = m.properties[202]
    assert p.type == 26
    assert p.params["mass"] == pytest.approx(0.5)
    assert p.params["ileng"] == 1
    assert p.params["nfunc"] == 1
    assert p.params["nraten"] == 0
    assert p.params["scale"] == pytest.approx(2.0)
    assert len(p.params["load_curves"]) == 1
    assert p.params["load_curves"][0]["fun_load"] == 11


def test_prop_type27_fixed(tmp_path: Path):
    # Card 0: MASS(20), blank(30), ISENSOR(10), ISFLAG(10), Ileng(10), Itens(10), Ifail(10)
    # Card 1: STIFF(20), DAMP(20), NEXP(20), MIN_RUP(20), MAX_RUP(20)
    # Card 2: GAP(20), blank(50), FSMOOTH(10), FCUT(20)
    # Card 3: FUN1(10), FUN2(10), ASCALE1(20), FSCALE1(20), ASCALE2(20), FSCALE2(20)
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/PROP/TYPE27/303\n"
        "Bilateral Damping Spring\n"
        "                 0.1                              20         1         0         1         2\n"
        "               500.0                10.0                 1.5               -0.05                0.08\n"
        "                0.01                                                  1                50.0\n"
        "        12        13                 1.1                 2.2                 1.3                 2.4\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    p = m.properties[303]
    assert p.type == 27
    assert p.title == "Bilateral Damping Spring"
    assert p.params["mass"] == pytest.approx(0.1)
    assert p.params["isensor"] == 20
    assert p.params["isflag"] == 1
    assert p.params["itens"] == 1
    assert p.params["ifail"] == 2
    assert p.params["stiff"] == pytest.approx(500.0)
    assert p.params["damp"] == pytest.approx(10.0)
    assert p.params["nexp"] == pytest.approx(1.5)
    assert p.params["min_rup"] == pytest.approx(-0.05)
    assert p.params["max_rup"] == pytest.approx(0.08)
    assert p.params["gap"] == pytest.approx(0.01)
    assert p.params["fsmooth"] == 1
    assert p.params["fcut"] == pytest.approx(50.0)
    assert p.params["fun1"] == 12
    assert p.params["fun2"] == 13
    assert p.params["ascale1"] == pytest.approx(1.1)
    assert p.params["fscale1"] == pytest.approx(2.2)
    assert p.params["ascale2"] == pytest.approx(1.3)
    assert p.params["fscale2"] == pytest.approx(2.4)


def test_prop_spr_bdamp_free(tmp_path: Path):
    deck = (
        "/PROP/SPR_BDAMP/404\n"
        "Free format BDAMP\n"
        "0.05 0 0 0 0 0\n"
        "250.0 5.0 1.0 0.0 0.0\n"
        "0.0 0 0.0\n"
        "21 22 1.0 1.0 1.0 1.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    p = m.properties[404]
    assert p.type == 27
    assert p.params["mass"] == pytest.approx(0.05)
    assert p.params["stiff"] == pytest.approx(250.0)
    assert p.params["fun1"] == 21
    assert p.params["fun2"] == 22


def test_inter_type19_fixed(tmp_path: Path):
    # Card 0: sec(10), main(10), istf(10), blank(10), igap(10), iedge(10), ibag(10), idel(10), icurv(10)
    # Card 1: GAPSCALE(20), GAPMAX(20)
    # Card 2: STMIN(20), STMAX(20)
    # Card 3: N1(10), N2(10)
    # Card 4: STFAC(20), FRIC(20), GAP(20), TSTART(20), TSTOP(20)
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/INTER/TYPE19/501\n"
        "Multi-Segment Interface 19\n"
        "         1         2         3                   1         0         0         1         0\n"
        "                 0.9                0.05\n"
        "                 1e3                 1e6\n"
        "         4         5\n"
        "                 1.2                 0.2               0.002                 0.0                10.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    assert len(m.interfaces) == 1
    inter = m.interfaces[0]
    assert inter.id == 501
    assert inter.type == 19
    assert inter.grnod_id == 1
    assert inter.surf_id == 2
    assert inter.istf == 3
    assert inter.igap == 1
    assert inter.gap_scale == pytest.approx(0.9)
    assert inter.gap_max == pytest.approx(0.05)
    assert inter.stmin == pytest.approx(1e3)
    assert inter.stmax == pytest.approx(1e6)
    assert inter.stfac == pytest.approx(1.2)
    assert inter.fric == pytest.approx(0.2)
    assert inter.gap == pytest.approx(0.002)
    assert inter.tstop == pytest.approx(10.0)


def test_inter_type19_free(tmp_path: Path):
    deck = (
        "/INTER/TYPE19/502\n"
        "Free TYPE19\n"
        "10 20 4 2 1 0 0 0\n"
        "0.8 0.02\n"
        "500.0 50000.0\n"
        "1 2\n"
        "1.0 0.15 0.001 0.0 5.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    inter = m.interfaces[0]
    assert inter.id == 502
    assert inter.type == 19
    assert inter.grnod_id == 10
    assert inter.surf_id == 20
    assert inter.istf == 4
    assert inter.igap == 2
    assert inter.fric == pytest.approx(0.15)


def test_inter_type25_full_fixed(tmp_path: Path):
    # Card 0: sec(10), main(10), istf(10), blank(10), igap(10), blank(20), idel(10)
    # Card 1: grnod(10), blank(30), prmesh(20), gap1(20), gap2(20)
    # Card 2: stmin(20), stmax(20), igap2(10)
    # Card 3: stfac(20), fric(20), blank(20), tstart(20), tstop(20)
    # Card 4: blank(7), deact_x(1), deact_y(1), deact_z(1), blank(20), inactiv(10), stiff_dc(20)
    # Card 5: ifric(10), ifiltr(10), xfreq(20), blank(10), isensor(10), blank(30), fric_id(10)
    # Card 6: c1(20), c2(20), c3(20), c4(20), c5(20)
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/INTER/TYPE25/601\n"
        "Advanced Surface Contact 25\n"
        "       100       200         5                   2                                       1\n"
        "        50                                                 0.5               0.003                0.01\n"
        "               100.0              1000.0         1\n"
        "                 1.5                0.25                                       0.0                20.0\n"
        "       100                    " + "         1" + "                15.0\n" +
        "         1         2                50.0                   5                             7\n"
        "                 0.1                 0.2                 0.3                 0.4                 0.5\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    assert len(m.interfaces) == 1
    inter = m.interfaces[0]
    assert inter.id == 601
    assert inter.type == 25
    assert inter.surf_id == 100
    assert inter.surf_id1 == 200
    assert inter.istf == 5
    assert inter.igap == 2
    assert inter.idel == 1
    assert inter.grnod_id == 50
    assert inter.gap == pytest.approx(0.003)
    assert inter.gap_max == pytest.approx(0.01)
    assert inter.stmin == pytest.approx(100.0)
    assert inter.stmax == pytest.approx(1000.0)
    assert inter.stfac == pytest.approx(1.5)
    assert inter.fric == pytest.approx(0.25)
    assert inter.tstop == pytest.approx(20.0)
    assert inter.inactiv == 1
    assert inter.stiff_dc == pytest.approx(15.0)
    assert inter.ifric == 1
    assert inter.ifiltr == 2
    assert inter.xfreq == pytest.approx(50.0)
    assert inter.isensor == 5
    assert inter.fric_id == 7
    assert inter.c1 == pytest.approx(0.1)
    assert inter.c5 == pytest.approx(0.5)


def test_inter_type8_fixed(tmp_path: Path):
    # Card 0: sec(10), main(10)
    # Card 1: blank(20), dbead_force(20), blank(20), tstart(20), tstop(20)
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/INTER/TYPE8/701\n"
        "Drawbead Interface\n"
        "        11        22\n"
        "                                   500.0                                     0.0                15.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    inter = m.interfaces[0]
    assert inter.id == 701
    assert inter.type == 8
    assert inter.grnod_id == 11
    assert inter.surf_id == 22
    assert inter.stfac == pytest.approx(500.0)
    assert inter.tstart == pytest.approx(0.0)
    assert inter.tstop == pytest.approx(15.0)


def test_drape_fixed_and_table(tmp_path: Path):
    deck = (
        "#RADIOSS STARTER\n"
        "/BEGIN\n"
        "TEST_M157\n"
        "                  0.                 0\n"
        "/DRAPE/801\n"
        "Composite Draping Table\n"
        "SHELL             10\n"
        "                0.95                45.0         3         2\n"
        "SHELL             20\n"
        "                0.90               -30.0         3         2\n"
        "/TABLE/DRAPE/802\n"
        "Table Draping Table 2\n"
        "GRSHEL            55\n"
        "                0.85                60.0         4         4\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    assert 801 in m.drapes
    d1 = m.drapes[801]
    assert d1.title == "Composite Draping Table"
    assert len(d1.slices) == 2
    assert d1.slices[0]["entity_type"] == "SHELL"
    assert d1.slices[0]["entity_id"] == 10
    assert d1.slices[0]["thinning"] == pytest.approx(0.95)
    assert d1.slices[0]["theta_slice"] == pytest.approx(45.0)
    assert d1.slices[0]["mat_id"] == 3
    assert d1.slices[0]["npt_slice"] == 2

    assert 802 in m.drapes
    d2 = m.drapes[802]
    assert len(d2.slices) == 1
    assert d2.slices[0]["entity_type"] == "GRSHEL"
    assert d2.slices[0]["entity_id"] == 55
    assert d2.slices[0]["theta_slice"] == pytest.approx(60.0)


def test_prop_aliases_m157(tmp_path: Path):
    deck = (
        "/PROP/P26_SPR_TAB/901\n"
        "Alias P26\n"
        "0.1 0 0 0\n"
        "0 0 1.0 10.0 0.0\n"
        "/PROP/P27_SPR_BDAMP/902\n"
        "Alias P27\n"
        "0.2 0 0 0 0 0\n"
        "100.0 1.0 1.0 0.0 0.0\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck)
    assert 901 in m.properties
    assert m.properties[901].type == 26
    assert 902 in m.properties
    assert m.properties[902].type == 27

