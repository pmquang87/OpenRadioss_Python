"""
Tests for /MAT/LAW106 and /MAT/JCOOK_ALM fixed/free format reading and deck writer round-trip (Milestone M575).
"""

import pytest
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_writer import StarterDeck, read_lines_to_blocks
from pyradioss.model.model import Model
from pyradioss.model.entities import MaterialLaw106
from pyradioss.common.messages import MessageLog


from pyradioss.input.deck_reader import _finalize_deck


def _parse_text(text: str) -> Model:
    model = Model()
    log = MessageLog()
    blocks = read_lines_to_blocks(text.splitlines())
    _finalize_deck(blocks)
    parse_starter_deck(blocks, model, log)
    return model


def test_read_mat_law106_fixed_format():
    """Verify reading /MAT/LAW106 in fixed 20-character field format."""
    c1 = f"{4.4e-6:>20g}{4.4e-6:>20g}"
    c2 = f"{110000.0:>20g}{0.34:>20g}{10:>10d}{20:>10d}{30:>10d}"
    c3 = f"{850.0:>20g}{400.0:>20g}{0.45:>20g}{0.15:>20g}{1200.0:>20g}"
    c4 = f"{0.0:>20g}{2:>10d}{5:>10d}{1e-5:>20g}{0.015:>20g}{1.0:>20g}"
    c5 = " " * 40 + f"{1.1:>20g}{1923.0:>20g}"
    c6 = f"{2.35e6:>20g}{0.9:>20g}{293.0:>20g}{293.0:>20g}"

    deck_text = f"""# OpenRadioss Starter input deck
/BEGIN
LAW106_FIXED_TEST
                2026                  10
/MAT/LAW106/1
Ti6Al4V Additive Manufacturing
#   RHO_INIT            RHO_REF
{c1}
#                  E                  NU   FCT_ID1   FCT_ID2   FCT_ID3
{c2}
#                  A                   B                   N             EPS_MAX           SIGMA_MAX
{c3}
#               FCUT        VP      NMAX                 TOL                 CJC               DEPS0
{c4}
#                                                          M               TMELT
{c5}
#             RHO_CP                 ETA                  T0                  TR
{c6}
/END
"""
    model = _parse_text(deck_text)
    assert 1 in model.mat_law106s
    m = model.mat_law106s[1]
    assert m.title == "Ti6Al4V Additive Manufacturing"
    assert m.rho0 == 4.4e-6
    assert m.rhor == 4.4e-6
    assert m.young == 110000.0
    assert m.nu == 0.34
    assert m.fct_id1 == 10
    assert m.fct_id2 == 20
    assert m.fct_id3 == 30
    assert m.sigy == 850.0
    assert m.beta == 400.0
    assert m.hard_n == 0.45
    assert m.ep_max == 0.15
    assert m.sig_max == 1200.0
    assert m.vp == 2
    assert m.nmax == 5
    assert m.tol == 1e-5
    assert m.cjc == 0.015
    assert m.deps0 == 1.0
    assert m.m == 1.1
    assert m.tmelt == 1923.0
    assert m.spheat == 2.35e6
    assert m.eta == 0.9
    assert m.t0 == 293.0
    assert m.tr == 293.0


def test_read_mat_jcook_alm_free_format():
    """Verify reading /MAT/JCOOK_ALM in free tokenized format."""
    deck_text = """/BEGIN
LAW106_FREE_TEST
20 10
/MAT/JCOOK_ALM/2
Inconel 718 ALM
7.8e-6 7.8e-6
205000.0 0.30 0 0 0
900.0 600.0 0.50 0.20 1500.0
1000.0 2 3 1e-6 0.02 1.0
1.0 1600.0
3.5e6 0.95 298.0 298.0
/END
"""
    model = _parse_text(deck_text)
    assert 2 in model.mat_jcook_alms
    m = model.mat_jcook_alms[2]
    assert m.title == "Inconel 718 ALM"
    assert m.rho0 == 7.8e-6
    assert m.young == 205000.0
    assert m.nu == 0.30
    assert m.sigy == 900.0
    assert m.beta == 600.0
    assert m.hard_n == 0.50
    assert m.ep_max == 0.20
    assert m.sig_max == 1500.0
    assert m.fcut == 1000.0
    assert m.cjc == 0.02
    assert m.deps0 == 1.0
    assert m.tmelt == 1600.0
    assert m.spheat == 3.5e6
    assert m.eta == 0.95


def test_deck_writer_roundtrip():
    """Verify StarterDeck.mat_law106 writes cards 1-6 and read_keyword_file recovers all parameters."""
    deck = StarterDeck("LAW106_ROUNDTRIP")
    deck.mat_law106(
        mat_id=5,
        title="Roundtrip_Material",
        rho0=7.85e-6,
        rhor=7.85e-6,
        young=210000.0,
        nu=0.3,
        fct_id1=101,
        fct_id2=102,
        fct_id3=103,
        sigy=450.0,
        beta=350.0,
        hard_n=0.4,
        ep_max=0.25,
        sig_max=950.0,
        fcut=500.0,
        vp=2,
        nmax=4,
        tol=1e-6,
        cjc=0.03,
        deps0=1.0,
        m=1.0,
        tmelt=1750.0,
        spheat=3.2e6,
        eta=0.85,
        t0=300.0,
        tr=300.0,
    )
    written_text = deck.write()
    assert "/MAT/LAW106/5" in written_text

    reloaded_model = _parse_text(written_text)
    assert 5 in reloaded_model.mat_law106s
    m2 = reloaded_model.mat_law106s[5]
    assert m2.id == 5
    assert m2.title == "Roundtrip_Material"
    assert pytest.approx(m2.rho0, rel=1e-5) == 7.85e-6
    assert pytest.approx(m2.young, rel=1e-5) == 210000.0
    assert pytest.approx(m2.nu, rel=1e-5) == 0.3
    assert m2.fct_id1 == 101
    assert m2.fct_id2 == 102
    assert m2.fct_id3 == 103
    assert pytest.approx(m2.sigy, rel=1e-5) == 450.0
    assert pytest.approx(m2.beta, rel=1e-5) == 350.0
    assert pytest.approx(m2.hard_n, rel=1e-5) == 0.4
    assert pytest.approx(m2.ep_max, rel=1e-5) == 0.25
    assert pytest.approx(m2.sig_max, rel=1e-5) == 950.0
    assert pytest.approx(m2.fcut, rel=1e-5) == 500.0
    assert m2.vp == 2
    assert m2.nmax == 4
    assert pytest.approx(m2.tol, rel=1e-5) == 1e-6
    assert pytest.approx(m2.cjc, rel=1e-5) == 0.03
    assert pytest.approx(m2.deps0, rel=1e-5) == 1.0
    assert pytest.approx(m2.tmelt, rel=1e-5) == 1750.0
    assert pytest.approx(m2.spheat, rel=1e-5) == 3.2e6
    assert pytest.approx(m2.eta, rel=1e-5) == 0.85
    assert pytest.approx(m2.t0, rel=1e-5) == 300.0
    assert pytest.approx(m2.tr, rel=1e-5) == 300.0
