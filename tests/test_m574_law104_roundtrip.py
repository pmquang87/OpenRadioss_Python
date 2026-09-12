"""Roundtrip deck writing and re-parsing tests for /MAT/LAW104 (M574).

Verifies:
- StarterDeck.mat_law104 writes standard cards matching OpenRadioss CFG format.
- StarterDeck.mat_drucker, mat_johns_voce_drucker, mat_plas_druck write proper headers.
- Re-parsing written decks produces matching MaterialLaw104 entity parameters.
- Roundtrip with custom title, unit IDs, reference density, and full physics card stack.
- Multi-element deck roundtrip with both 3D solids and 2D shells.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MaterialLaw104
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_materials


def _roundtrip(tmp_path: Path, deck: StarterDeck) -> Model:
    text = deck.render()
    p = tmp_path / "ROUNDTRIP_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_roundtrip_mat_law104_kwargs(tmp_path: Path):
    """Write /MAT/LAW104 using kwargs and re-read."""
    d = StarterDeck("M574_KWARGS")
    d.mat_law104(
        mid=104,
        title="Alloy 718 Drucker Plasticity",
        rho=8.19e-9,
        refer_rho=8.19e-9,
        young=205000.0,
        nu=0.3,
        ires=1,
        sigma0_yld=450.0,
        h=1200.0,
        q_voce=350.0,
        b_voce=12.5,
        c_dr=1.2,
        c_jc=0.015,
        eps0=1.0,
        fcut=1000.0,
        tss=0.0015,
        tref=293.15,
        tini=293.15,
        eta=0.9,
        cp=435.0,
        eps_iso=10.0,
        eps_ad=1000.0,
    )

    model = _roundtrip(tmp_path, d)
    assert 104 in model.mat_law104s
    mat = model.mat_law104s[104]

    assert mat.id == 104
    assert "Alloy 718" in mat.title
    assert math.isclose(mat.rho, 8.19e-9, rel_tol=1.0e-6)
    assert math.isclose(mat.refer_rho, 8.19e-9, rel_tol=1.0e-6)
    assert math.isclose(mat.young, 205000.0, rel_tol=1.0e-6)
    assert math.isclose(mat.nu, 0.3, rel_tol=1.0e-6)
    assert mat.ires == 1
    assert math.isclose(mat.sigma0_yld, 450.0, rel_tol=1.0e-6)
    assert math.isclose(mat.h, 1200.0, rel_tol=1.0e-6)
    assert math.isclose(mat.q_voce, 350.0, rel_tol=1.0e-6)
    assert math.isclose(mat.b_voce, 12.5, rel_tol=1.0e-6)
    assert math.isclose(mat.c_dr, 1.2, rel_tol=1.0e-6)
    assert math.isclose(mat.c_jc, 0.015, rel_tol=1.0e-6)
    assert math.isclose(mat.eps0, 1.0, rel_tol=1.0e-6)
    assert math.isclose(mat.fcut, 1000.0, rel_tol=1.0e-6)
    assert math.isclose(mat.tss, 0.0015, rel_tol=1.0e-6)
    assert math.isclose(mat.tref, 293.15, rel_tol=1.0e-6)
    assert math.isclose(mat.tini, 293.15, rel_tol=1.0e-6)
    assert math.isclose(mat.eta, 0.9, rel_tol=1.0e-6)
    assert math.isclose(mat.cp, 435.0, rel_tol=1.0e-6)
    assert math.isclose(mat.eps_iso, 10.0, rel_tol=1.0e-6)
    assert math.isclose(mat.eps_ad, 1000.0, rel_tol=1.0e-6)


def test_roundtrip_mat_law104_entity(tmp_path: Path):
    """Write /MAT/LAW104 from an existing MaterialLaw104 entity object and re-read."""
    orig = MaterialLaw104(
        id=204,
        title="High Strength Steel Drucker",
        rho0=7.85e-9,
        refer_rho=7.85e-9,
        young=210000.0,
        nu=0.28,
        ires=2,
        sigma_r=600.0,
        h=800.0,
        qv=200.0,
        bv=8.0,
        cdr=-1.5,
        cjc=0.02,
        epsp0=0.001,
        fcut=500.0,
        tss=0.002,
        tref=300.0,
        tini=300.0,
        eta=0.85,
        cp=460.0,
        eps_iso=5.0,
        eps_ad=500.0,
    )

    d = StarterDeck("M574_ENTITY")
    d.mat_law104(mat_obj=orig)

    model = _roundtrip(tmp_path, d)
    assert 204 in model.mat_law104s
    mat = model.mat_law104s[204]

    assert mat.id == 204
    assert "High Strength Steel" in mat.title
    assert math.isclose(mat.rho, 7.85e-9, rel_tol=1.0e-6)
    assert math.isclose(mat.young, 210000.0, rel_tol=1.0e-6)
    assert math.isclose(mat.nu, 0.28, rel_tol=1.0e-6)
    assert mat.ires == 2
    assert math.isclose(mat.sigma_r, 600.0, rel_tol=1.0e-6)
    assert math.isclose(mat.h, 800.0, rel_tol=1.0e-6)
    assert math.isclose(mat.qv, 200.0, rel_tol=1.0e-6)
    assert math.isclose(mat.bv, 8.0, rel_tol=1.0e-6)
    assert math.isclose(mat.cdr, -1.5, rel_tol=1.0e-6)
    assert math.isclose(mat.cjc, 0.02, rel_tol=1.0e-6)
    assert math.isclose(mat.epsp0, 0.001, rel_tol=1.0e-6)


def test_roundtrip_mat_aliases(tmp_path: Path):
    """Write using synonyms (/MAT/DRUCKER, /MAT/JOHNS_VOCE_DRUCKER, /MAT/PLAS_DRUCK) and re-read."""
    d = StarterDeck("M574_ALIASES")
    d.mat_drucker(
        mid=301,
        title="Drucker Synonym",
        rho=7.8e-9,
        young=200000.0,
        nu=0.3,
        sigma0_yld=350.0,
        c_dr=0.8,
    )
    d.mat_johns_voce_drucker(
        mid=302,
        title="Johns Voce Drucker Synonym",
        rho=7.8e-9,
        young=200000.0,
        nu=0.3,
        sigma0_yld=400.0,
        c_dr=-0.5,
        q_voce=150.0,
        b_voce=10.0,
    )
    d.mat_plas_druck(
        mid=303,
        title="Plas Druck Synonym",
        rho=7.8e-9,
        young=200000.0,
        nu=0.3,
        sigma0_yld=500.0,
        c_dr=2.0,
    )

    model = _roundtrip(tmp_path, d)
    assert 301 in model.mat_law104s
    assert 302 in model.mat_law104s
    assert 303 in model.mat_law104s

    assert math.isclose(model.mat_law104s[301].cdr, 0.8, rel_tol=1.0e-6)
    assert math.isclose(model.mat_law104s[302].cdr, -0.5, rel_tol=1.0e-6)
    assert math.isclose(model.mat_law104s[302].qv, 150.0, rel_tol=1.0e-6)
    assert math.isclose(model.mat_law104s[303].cdr, 2.0, rel_tol=1.0e-6)


def test_roundtrip_deck_solids_and_shells(tmp_path: Path):
    """Write a deck containing LAW104 with both solid and shell elements, check starter diagnostics."""
    d = StarterDeck("M574_ELEMENTS")
    d.mat_law104(
        mid=1,
        title="Drucker Mat",
        rho=7.85e-9,
        young=210000.0,
        nu=0.3,
        sigma0_yld=350.0,
        cdr=1.0,
    )
    d.prop_solid(pid=1, title="Solid Prop")
    d.prop_shell(pid=2, title="Shell Prop", thick=1.5)
    d.part(1, "Solid Part", 1, 1)
    d.part(2, "Shell Part", 2, 1)

    # Add 8 nodes
    deck_nodes = [
        (1, 0.0, 0.0, 0.0),
        (2, 10.0, 0.0, 0.0),
        (3, 10.0, 10.0, 0.0),
        (4, 0.0, 10.0, 0.0),
        (5, 0.0, 0.0, 10.0),
        (6, 10.0, 0.0, 10.0),
        (7, 10.0, 10.0, 10.0),
        (8, 0.0, 10.0, 10.0),
    ]
    d.node(deck_nodes)

    # 1 Solid Hexa8 (part 1)
    d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
    # 1 Shell Quad4 (part 2)
    d.shell(2, [(2, 1, 2, 3, 4)])

    model = _roundtrip(tmp_path, d)
    assert 1 in model.mat_law104s
    assert len(model.raw_elems["BRICK"]) == 1
    assert len(model.raw_elems["SHELL"]) == 1

    # Verify diagnostic checks pass without any errors
    log = MessageLog()
    check_materials(model, log)
    assert not log.errors, f"Diagnostic check errors: {log.errors}"
