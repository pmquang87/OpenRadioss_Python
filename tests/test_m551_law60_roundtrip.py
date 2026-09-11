"""
Milestone M551: /MAT/LAW60 (/MAT/PLAS_T3, /MAT/FABRIC)
Exhaustive Roundtrip, Serialization, Starter Diagnostics & Negative Testing Suite.

Fortran origins:
  - starter/source/materials/mat/mat060/hm_read_mat60.F
  - engine/source/materials/mat/mat060/sigeps60.F
  - engine/source/materials/mat/mat060/sigeps60c.F
  - config/CFG/radioss2023/MAT/matl60_PLAS_T3.cfg

Audits:
  1. Fixed-Format and Free-Format Deck Roundtrip:
     - StarterDeck.mat_law60, mat_plas_t3, mat_fabric in both fixed and free format.
     - 5-function and 10-function configurations.
     - Parameter sweeps: id, rho, ref_rho, e, nu, eps_p_max, eps_t1, eps_t2,
       nfunc, fsmooth, mat_hard, fcut, xr_fun, ifunce, mat_fscale, einf, ce, funcs, fscales, rates.
     - Object invocation with MatLaw60 dataclass.
     - Comma-delimited and space-delimited free format.
     - Two-way cross-dialect roundtrip (free -> parse -> fixed -> parse).
     - 100% parameter fidelity verification across both Model.mat_law60s and Model.materials.

  2. Negative Validation Tests:
     - rho0 <= 0 rejected.
     - E <= 0 rejected.
     - nu < 0 or nu >= 0.5 rejected.
     - eps_t1 >= eps_t2 (when active) rejected.
     - Non-monotonic strain rates (RATE[i] >= RATE[i+1]) rejected.
     - Referenced curve IDs not found in deck /FUNCT rejected.
     - Incompatible elements: trusses, beams, springs rejected by starter checks.
     - Compatible elements accepted: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads.
     - _ALLOWED_LAWS dictionary membership.

  3. Restart File (.rst) Serialization:
     - MatLaw60 dataclass pickling / unpickling fidelity.
     - Law60Params physics object serialization / deserialization.
     - extra_shapes dictionary verification (uvar and off60).
     - Material state with uvar history variables and off60 flags preserved across write_restart / read_restart.
     - Complete engine simulation restart continuation with LAW60 (solid and shell).
     - Chained vs unchained reference execution parity.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import pickle
from typing import Any, Dict, List
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.engine.engine import run_engine
from pyradioss.input.card_layouts import LAYOUTS
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    extra_shapes,
    sound_speed,
    solid_update,
    shell_update,
)
from pyradioss.model.entities import (
    MatFabric,
    MatLaw60,
    MatPlasT3,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law60,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_DECK") -> tuple[Model, MessageLog]:
    """Write deck text to temporary file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law60_fidelity(
    m: MatLaw60,
    expected: dict[str, Any],
    mat: Material | None = None,
) -> None:
    """Verify 100% parameter fidelity against expected values dictionary."""
    mid = expected["id"]
    assert m.id == mid
    assert m.rho == pytest.approx(expected["rho"])
    assert m.rho0 == pytest.approx(expected.get("ref_rho") or expected["rho"])
    if "ref_rho" in expected:
        assert m.ref_rho == pytest.approx(expected["ref_rho"])
        assert m.refer_rho == pytest.approx(expected["ref_rho"])
    assert m.e == pytest.approx(expected["e"])
    assert m.E == pytest.approx(expected["e"])
    assert m.nu == pytest.approx(expected["nu"])
    assert m.eps_p_max == pytest.approx(expected["eps_p_max"])
    assert m.eps_t1 == pytest.approx(expected["eps_t1"])
    assert m.eps_t2 == pytest.approx(expected["eps_t2"])
    assert m.nfunc == expected["nfunc"]
    assert m.fsmooth == expected["fsmooth"]
    assert m.mat_hard == pytest.approx(expected["mat_hard"])
    assert m.chard == pytest.approx(expected["mat_hard"])
    assert m.fcut == pytest.approx(expected["fcut"])
    assert m.xr_fun == expected["xr_fun"]
    assert m.ifunce == expected["ifunce"]
    assert m.mat_fscale == pytest.approx(expected["mat_fscale"])
    assert m.fpscale == pytest.approx(expected["mat_fscale"])
    assert m.einf == pytest.approx(expected["einf"])
    assert m.ce == pytest.approx(expected["ce"])
    assert m.funcs == expected["funcs"]
    assert m.fun_ids == expected["funcs"]
    assert m.fscales == pytest.approx(expected["fscales"])
    assert m.rates == pytest.approx(expected["rates"])
    assert m.eps_rates == pytest.approx(expected["rates"])
    if "title" in expected:
        assert m.title == expected["title"]

    if mat is not None:
        assert mat.id == mid
        assert mat.law == 60
        assert mat.rho0 == pytest.approx(expected.get("ref_rho") or expected["rho"])
        params = mat.params
        assert params["E"] == pytest.approx(expected["e"])
        assert params["nu"] == pytest.approx(expected["nu"])
        assert params["eps_p_max"] == pytest.approx(expected["eps_p_max"])
        assert params["eps_t1"] == pytest.approx(expected["eps_t1"])
        assert params["eps_t2"] == pytest.approx(expected["eps_t2"])
        assert params["nfunc"] == expected["nfunc"]
        assert params["fsmooth"] == expected["fsmooth"]
        assert params["mat_hard"] == pytest.approx(expected["mat_hard"])
        assert params["fcut"] == pytest.approx(expected["fcut"])
        assert params["xr_fun"] == expected["xr_fun"]
        assert params["ifunce"] == expected["ifunce"]
        assert params["mat_fscale"] == pytest.approx(expected["mat_fscale"])
        assert params["einf"] == pytest.approx(expected["einf"])
        assert params["ce"] == pytest.approx(expected["ce"])
        assert params["funcs"] == expected["funcs"]
        assert params["fscales"] == pytest.approx(expected["fscales"])
        assert params["rates"] == pytest.approx(expected["rates"])


# ============================================================================
# 1. Fixed-Format and Free-Format Deck Roundtrip
# ============================================================================

class TestLaw60FixedFormatRoundtrip:
    """Audit fixed-format roundtrip for /MAT/LAW60, /MAT/PLAS_T3, /MAT/FABRIC."""

    def test_law60_fixed_5_functions_roundtrip(self, tmp_path: Path):
        """Construct 5-function configuration via StarterDeck.mat_law60, render fixed, parse back."""
        deck = StarterDeck("FIXED_60_5")
        for fid in (101, 102, 103, 104, 105):
            deck.funct(fid, f"Yield_{fid}", [(0.0, 200.0 + fid), (0.1, 300.0 + fid)])

        exp = {
            "id": 1,
            "title": "Fabric Mat Fixed 5",
            "rho": 7.85e-9,
            "ref_rho": 7.85e-9,
            "e": 210000.0,
            "nu": 0.30,
            "eps_p_max": 0.40,
            "eps_t1": 0.15,
            "eps_t2": 0.25,
            "nfunc": 5,
            "fsmooth": 1,
            "mat_hard": 0.35,
            "fcut": 950.0,
            "xr_fun": 2,
            "ifunce": 1,
            "mat_fscale": 1.15,
            "einf": 55000.0,
            "ce": 12.5,
            "funcs": [101, 102, 103, 104, 105],
            "fscales": [1.0, 1.05, 1.10, 1.15, 1.20],
            "rates": [0.0, 10.0, 50.0, 100.0, 500.0],
        }

        deck.mat_law60(
            mid=exp["id"],
            title=exp["title"],
            rho=exp["rho"],
            ref_rho=exp["ref_rho"],
            e=exp["e"],
            nu=exp["nu"],
            eps_p_max=exp["eps_p_max"],
            eps_t1=exp["eps_t1"],
            eps_t2=exp["eps_t2"],
            nfunc=exp["nfunc"],
            fsmooth=exp["fsmooth"],
            mat_hard=exp["mat_hard"],
            fcut=exp["fcut"],
            xr_fun=exp["xr_fun"],
            ifunce=exp["ifunce"],
            mat_fscale=exp["mat_fscale"],
            einf=exp["einf"],
            ce=exp["ce"],
            funcs=exp["funcs"],
            fscales=exp["fscales"],
            rates=exp["rates"],
            fixed_format=True,
        )

        rad_file = tmp_path / "law60_fixed_5_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        assert 1 in model.mat_law60s
        assert 1 in model.materials
        _assert_law60_fidelity(model.mat_law60s[1], exp, model.materials[1])

    def test_law60_fixed_10_functions_roundtrip(self, tmp_path: Path):
        """Construct 10-function configuration via StarterDeck.mat_law60, render fixed, parse back."""
        deck = StarterDeck("FIXED_60_10")
        funcs = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
        fscales = [1.0, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.25]
        rates = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

        for fid in funcs:
            deck.funct(fid, f"Yield_{fid}", [(0.0, 250.0), (0.2, 450.0)])

        exp = {
            "id": 2,
            "title": "High-Rate Plasticity 10 Curves",
            "rho": 1.25e-6,
            "ref_rho": 1.25e-6,
            "e": 70000.0,
            "nu": 0.33,
            "eps_p_max": 0.60,
            "eps_t1": 0.20,
            "eps_t2": 0.35,
            "nfunc": 10,
            "fsmooth": 0,
            "mat_hard": 0.10,
            "fcut": 600.0,
            "xr_fun": 0,
            "ifunce": 0,
            "mat_fscale": 1.0,
            "einf": 0.0,
            "ce": 0.0,
            "funcs": funcs,
            "fscales": fscales,
            "rates": rates,
        }

        deck.mat_law60(
            mid=exp["id"],
            title=exp["title"],
            rho=exp["rho"],
            ref_rho=exp["ref_rho"],
            e=exp["e"],
            nu=exp["nu"],
            eps_p_max=exp["eps_p_max"],
            eps_t1=exp["eps_t1"],
            eps_t2=exp["eps_t2"],
            nfunc=exp["nfunc"],
            fsmooth=exp["fsmooth"],
            mat_hard=exp["mat_hard"],
            fcut=exp["fcut"],
            xr_fun=exp["xr_fun"],
            ifunce=exp["ifunce"],
            mat_fscale=exp["mat_fscale"],
            einf=exp["einf"],
            ce=exp["ce"],
            funcs=exp["funcs"],
            fscales=exp["fscales"],
            rates=exp["rates"],
            fixed_format=True,
        )

        rad_file = tmp_path / "law60_fixed_10_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        assert 2 in model.mat_law60s
        assert 2 in model.materials
        _assert_law60_fidelity(model.mat_law60s[2], exp, model.materials[2])

    def test_plas_t3_and_fabric_fixed_synonyms_roundtrip(self, tmp_path: Path):
        """Construct models using StarterDeck.mat_plas_t3 and mat_fabric in fixed format."""
        deck = StarterDeck("SYNONYMS_FIXED")
        deck.funct(11, "F11", [(0.0, 100.0), (0.1, 200.0)])
        deck.funct(12, "F12", [(0.0, 120.0), (0.1, 220.0)])

        exp_plas = {
            "id": 11,
            "title": "Plas T3 Fixed",
            "rho": 2.7e-9,
            "ref_rho": 2.7e-9,
            "e": 69000.0,
            "nu": 0.33,
            "eps_p_max": 0.5,
            "eps_t1": 0.2,
            "eps_t2": 0.3,
            "nfunc": 5,
            "fsmooth": 1,
            "mat_hard": 0.2,
            "fcut": 450.0,
            "xr_fun": 1,
            "ifunce": 0,
            "mat_fscale": 1.0,
            "einf": 0.0,
            "ce": 0.0,
            "funcs": [11, 12],
            "fscales": [1.0, 1.1],
            "rates": [0.0, 25.0],
        }
        deck.mat_plas_t3(
            mid=exp_plas["id"],
            title=exp_plas["title"],
            rho=exp_plas["rho"],
            ref_rho=exp_plas["ref_rho"],
            e=exp_plas["e"],
            nu=exp_plas["nu"],
            eps_p_max=exp_plas["eps_p_max"],
            eps_t1=exp_plas["eps_t1"],
            eps_t2=exp_plas["eps_t2"],
            nfunc=exp_plas["nfunc"],
            fsmooth=exp_plas["fsmooth"],
            mat_hard=exp_plas["mat_hard"],
            fcut=exp_plas["fcut"],
            xr_fun=exp_plas["xr_fun"],
            ifunce=exp_plas["ifunce"],
            mat_fscale=exp_plas["mat_fscale"],
            einf=exp_plas["einf"],
            ce=exp_plas["ce"],
            funcs=exp_plas["funcs"],
            fscales=exp_plas["fscales"],
            rates=exp_plas["rates"],
            fixed_format=True,
        )

        deck.funct(21, "F21", [(0.0, 50.0), (0.1, 150.0)])
        exp_fab = {
            "id": 12,
            "title": "Fabric Fixed",
            "rho": 1.4e-9,
            "ref_rho": 1.4e-9,
            "e": 15000.0,
            "nu": 0.38,
            "eps_p_max": 0.7,
            "eps_t1": 0.3,
            "eps_t2": 0.45,
            "nfunc": 5,
            "fsmooth": 0,
            "mat_hard": 0.05,
            "fcut": 200.0,
            "xr_fun": 0,
            "ifunce": 1,
            "mat_fscale": 1.05,
            "einf": 8000.0,
            "ce": 5.0,
            "funcs": [21],
            "fscales": [1.0],
            "rates": [0.0],
        }
        deck.mat_fabric(
            mid=exp_fab["id"],
            title=exp_fab["title"],
            rho=exp_fab["rho"],
            ref_rho=exp_fab["ref_rho"],
            e=exp_fab["e"],
            nu=exp_fab["nu"],
            eps_p_max=exp_fab["eps_p_max"],
            eps_t1=exp_fab["eps_t1"],
            eps_t2=exp_fab["eps_t2"],
            nfunc=exp_fab["nfunc"],
            fsmooth=exp_fab["fsmooth"],
            mat_hard=exp_fab["mat_hard"],
            fcut=exp_fab["fcut"],
            xr_fun=exp_fab["xr_fun"],
            ifunce=exp_fab["ifunce"],
            mat_fscale=exp_fab["mat_fscale"],
            einf=exp_fab["einf"],
            ce=exp_fab["ce"],
            funcs=exp_fab["funcs"],
            fscales=exp_fab["fscales"],
            rates=exp_fab["rates"],
            fixed_format=True,
        )

        rad_file = tmp_path / "synonyms_fixed_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        _assert_law60_fidelity(model.mat_law60s[11], exp_plas, model.materials[11])
        _assert_law60_fidelity(model.mat_law60s[12], exp_fab, model.materials[12])

    @pytest.mark.parametrize(
        "nfunc,fsmooth,xr_fun,ifunce,mat_hard,einf,ce",
        [
            (5, 0, 0, 0, 0.0, 0.0, 0.0),
            (5, 1, 1, 1, 0.25, 15000.0, 4.5),
            (10, 0, 2, 0, 0.50, 0.0, 0.0),
            (10, 1, 0, 1, 0.08, 40000.0, 12.0),
        ],
    )
    def test_law60_fixed_parameter_sweeps(
        self,
        tmp_path: Path,
        nfunc: int,
        fsmooth: int,
        xr_fun: int,
        ifunce: int,
        mat_hard: float,
        einf: float,
        ce: float,
    ):
        """Verify roundtrip fidelity across diverse parameter sets in fixed format."""
        deck = StarterDeck(f"SWEEP_FIXED_{nfunc}_{fsmooth}")
        n_curves = nfunc
        funcs = [500 + i for i in range(n_curves)]
        fscales = [1.0 + 0.05 * i for i in range(n_curves)]
        rates = [float(i * 20.0) for i in range(n_curves)]

        exp = {
            "id": 99,
            "title": f"Sweep Mat nfunc={nfunc}",
            "rho": 8.0e-9,
            "ref_rho": 8.0e-9,
            "e": 195000.0,
            "nu": 0.28,
            "eps_p_max": 0.45,
            "eps_t1": 0.12,
            "eps_t2": 0.22,
            "nfunc": nfunc,
            "fsmooth": fsmooth,
            "mat_hard": mat_hard,
            "fcut": 850.0,
            "xr_fun": xr_fun,
            "ifunce": ifunce,
            "mat_fscale": 1.10,
            "einf": einf,
            "ce": ce,
            "funcs": funcs,
            "fscales": fscales,
            "rates": rates,
        }

        deck.mat_law60(
            mid=exp["id"],
            title=exp["title"],
            rho=exp["rho"],
            ref_rho=exp["ref_rho"],
            e=exp["e"],
            nu=exp["nu"],
            eps_p_max=exp["eps_p_max"],
            eps_t1=exp["eps_t1"],
            eps_t2=exp["eps_t2"],
            nfunc=exp["nfunc"],
            fsmooth=exp["fsmooth"],
            mat_hard=exp["mat_hard"],
            fcut=exp["fcut"],
            xr_fun=exp["xr_fun"],
            ifunce=exp["ifunce"],
            mat_fscale=exp["mat_fscale"],
            einf=exp["einf"],
            ce=exp["ce"],
            funcs=exp["funcs"],
            fscales=exp["fscales"],
            rates=exp["rates"],
            fixed_format=True,
        )

        rad_file = tmp_path / f"sweep_{nfunc}_{fsmooth}_{xr_fun}_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        _assert_law60_fidelity(model.mat_law60s[99], exp, model.materials[99])

    def test_law60_fixed_object_invocation(self, tmp_path: Path):
        """Verify StarterDeck.mat_law60 accepting MatLaw60 dataclass directly."""
        m_obj = MatLaw60(
            id=77,
            rho=1.5e-6,
            ref_rho=1.5e-6,
            e=2200.0,
            nu=0.34,
            eps_p_max=0.55,
            eps_t1=0.25,
            eps_t2=0.45,
            nfunc=5,
            fsmooth=1,
            mat_hard=0.18,
            fcut=120.0,
            xr_fun=2,
            ifunce=1,
            mat_fscale=1.2,
            einf=600.0,
            ce=14.0,
            funcs=[301, 302],
            fscales=[1.0, 1.1],
            rates=[0.0, 50.0],
            title="Dataclass Object Test",
        )

        deck = StarterDeck("OBJ_TEST")
        deck.mat_law60(m_obj, fixed_format=True)
        rad_file = tmp_path / "obj_test_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        m_res = model.mat_law60s[77]
        assert m_res.e == pytest.approx(2200.0)
        assert m_res.funcs == [301, 302]
        assert m_res.title == "Dataclass Object Test"


# ============================================================================
# 2. Free-Format Roundtrip & Two-Way Cross-Dialect Audit
# ============================================================================

class TestLaw60FreeFormatRoundtrip:
    """Audit free-format roundtrip, comma/space delimiters, and cross-dialect parsing."""

    def test_law60_free_5_and_10_functions_roundtrip(self, tmp_path: Path):
        """Construct 5-function and 10-function models in free-format and parse back."""
        deck = StarterDeck("FREE_60")

        exp5 = {
            "id": 1,
            "title": "Free Format 5 Curves",
            "rho": 7.8e-9,
            "ref_rho": 7.8e-9,
            "e": 200000.0,
            "nu": 0.29,
            "eps_p_max": 0.4,
            "eps_t1": 0.15,
            "eps_t2": 0.25,
            "nfunc": 5,
            "fsmooth": 1,
            "mat_hard": 0.2,
            "fcut": 800.0,
            "xr_fun": 1,
            "ifunce": 1,
            "mat_fscale": 1.1,
            "einf": 30000.0,
            "ce": 8.0,
            "funcs": [1, 2, 3],
            "fscales": [1.0, 1.05, 1.10],
            "rates": [0.0, 10.0, 100.0],
        }
        deck.mat_law60(
            mid=exp5["id"],
            title=exp5["title"],
            rho=exp5["rho"],
            ref_rho=exp5["ref_rho"],
            e=exp5["e"],
            nu=exp5["nu"],
            eps_p_max=exp5["eps_p_max"],
            eps_t1=exp5["eps_t1"],
            eps_t2=exp5["eps_t2"],
            nfunc=exp5["nfunc"],
            fsmooth=exp5["fsmooth"],
            mat_hard=exp5["mat_hard"],
            fcut=exp5["fcut"],
            xr_fun=exp5["xr_fun"],
            ifunce=exp5["ifunce"],
            mat_fscale=exp5["mat_fscale"],
            einf=exp5["einf"],
            ce=exp5["ce"],
            funcs=exp5["funcs"],
            fscales=exp5["fscales"],
            rates=exp5["rates"],
            fixed_format=False,
        )

        exp10 = {
            "id": 2,
            "title": "Free Format 10 Curves",
            "rho": 1.5e-6,
            "ref_rho": 1.5e-6,
            "e": 2500.0,
            "nu": 0.35,
            "eps_p_max": 0.5,
            "eps_t1": 0.2,
            "eps_t2": 0.3,
            "nfunc": 10,
            "fsmooth": 0,
            "mat_hard": 0.1,
            "fcut": 100.0,
            "xr_fun": 0,
            "ifunce": 0,
            "mat_fscale": 1.0,
            "einf": 0.0,
            "ce": 0.0,
            "funcs": [11, 12, 13, 14, 15, 16, 17, 18],
            "fscales": [1.0] * 8,
            "rates": [0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
        }
        deck.mat_plas_t3(
            mid=exp10["id"],
            title=exp10["title"],
            rho=exp10["rho"],
            ref_rho=exp10["ref_rho"],
            e=exp10["e"],
            nu=exp10["nu"],
            eps_p_max=exp10["eps_p_max"],
            eps_t1=exp10["eps_t1"],
            eps_t2=exp10["eps_t2"],
            nfunc=exp10["nfunc"],
            fsmooth=exp10["fsmooth"],
            mat_hard=exp10["mat_hard"],
            fcut=exp10["fcut"],
            xr_fun=exp10["xr_fun"],
            ifunce=exp10["ifunce"],
            mat_fscale=exp10["mat_fscale"],
            einf=exp10["einf"],
            ce=exp10["ce"],
            funcs=exp10["funcs"],
            fscales=exp10["fscales"],
            rates=exp10["rates"],
            fixed_format=False,
        )

        rad_file = tmp_path / "law60_free_0000.rad"
        deck.write(str(rad_file))

        blocks = read_deck(str(rad_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        _assert_law60_fidelity(model.mat_law60s[1], exp5, model.materials[1])
        _assert_law60_fidelity(model.mat_law60s[2], exp10, model.materials[2])

    def test_law60_free_comma_delimited(self, tmp_path: Path):
        """Parse raw comma-separated free-format card input."""
        deck_text = """# RADIOSS STARTER
/BEGIN
TEST_COMMA
/MAT/LAW60/5
Comma Separated Law60
1.3e-6, 1.3e-6
2200.0, 0.32, 0.45, 0.18, 0.28
5, 1, 0.15, 150.0
2, 1.25, 1, 500.0, 10.0
51, 52
1.0, 1.15
0.0, 50.0
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, "TEST_COMMA")
        assert len(log.errors) == 0
        assert 5 in model.mat_law60s
        m = model.mat_law60s[5]
        assert m.id == 5
        assert m.rho == pytest.approx(1.3e-6)
        assert m.e == pytest.approx(2200.0)
        assert m.nu == pytest.approx(0.32)
        assert m.eps_p_max == pytest.approx(0.45)
        assert m.eps_t1 == pytest.approx(0.18)
        assert m.eps_t2 == pytest.approx(0.28)
        assert m.nfunc == 5
        assert m.fsmooth == 1
        assert m.mat_hard == pytest.approx(0.15)
        assert m.fcut == pytest.approx(150.0)
        assert m.xr_fun == 2
        assert m.mat_fscale == pytest.approx(1.25)
        assert m.ifunce == 1
        assert m.einf == pytest.approx(500.0)
        assert m.ce == pytest.approx(10.0)
        assert m.funcs == [51, 52]
        assert m.fscales == [pytest.approx(1.0), pytest.approx(1.15)]
        assert m.rates == [pytest.approx(0.0), pytest.approx(50.0)]

    def test_law60_two_way_cross_dialect_roundtrip(self, tmp_path: Path):
        """Parse free-format deck, write back as fixed-format deck, parse again, and compare parameters."""
        raw_free = """# RADIOSS STARTER
/BEGIN
CROSS_DIALECT
/MAT/FABRIC/10
Cross Dialect Fabric
1.6e-6 1.6e-6
1800.0 0.36 0.50 0.20 0.35
5 1 0.25 120.0
1 1.05 0 0.0 0.0
81 82 83
1.0 1.1 1.2
0.0 10.0 100.0
/END
"""
        # Step 1: Parse free format
        model1, log1 = _parse_deck_str(tmp_path, raw_free, "CROSS_1")
        assert len(log1.errors) == 0
        m1 = model1.mat_law60s[10]

        # Step 2: Render as fixed format via StarterDeck
        deck_fixed = StarterDeck("CROSS_FIXED")
        deck_fixed.mat_law60(m1, fixed_format=True)
        fixed_path = tmp_path / "cross_fixed_0000.rad"
        deck_fixed.write(str(fixed_path))

        # Step 3: Parse fixed format back
        blocks = read_deck(str(fixed_path))
        model2 = Model()
        log2 = MessageLog()
        parse_starter_deck(blocks, model2, log2)
        assert len(log2.errors) == 0

        # Step 4: Compare parameters between model1 and model2
        m2 = model2.mat_law60s[10]
        assert m2.id == m1.id
        assert m2.rho == pytest.approx(m1.rho)
        assert m2.ref_rho == pytest.approx(m1.ref_rho)
        assert m2.e == pytest.approx(m1.e)
        assert m2.nu == pytest.approx(m1.nu)
        assert m2.eps_p_max == pytest.approx(m1.eps_p_max)
        assert m2.eps_t1 == pytest.approx(m1.eps_t1)
        assert m2.eps_t2 == pytest.approx(m1.eps_t2)
        assert m2.nfunc == m1.nfunc
        assert m2.fsmooth == m1.fsmooth
        assert m2.mat_hard == pytest.approx(m1.mat_hard)
        assert m2.fcut == pytest.approx(m1.fcut)
        assert m2.xr_fun == m1.xr_fun
        assert m2.ifunce == m1.ifunce
        assert m2.mat_fscale == pytest.approx(m1.mat_fscale)
        assert m2.einf == pytest.approx(m1.einf)
        assert m2.ce == pytest.approx(m1.ce)
        assert m2.funcs == m1.funcs
        assert m2.fscales == pytest.approx(m1.fscales)
        assert m2.rates == pytest.approx(m1.rates)


# ============================================================================
# 3. Negative Validation Tests
# ============================================================================

class TestLaw60NegativeValidation:
    """Verify log.errors catches all invalid parameter boundaries."""

    @pytest.mark.parametrize("bad_rho", [0.0, -1.0, -1e-6])
    def test_negative_density_rejected(self, bad_rho: float):
        """Initial density rho0 <= 0 must be rejected with an error in log."""
        m = Material(id=1, law=60, rho0=bad_rho, params={"E": 200000.0, "nu": 0.3})
        log = MessageLog()
        check_mat_law60(m, log)
        assert log.has_errors
        assert any("density" in e.lower() or "rho" in e.lower() for e in log.errors)

    @pytest.mark.parametrize("bad_e", [0.0, -100.0, -210000.0])
    def test_negative_youngs_modulus_rejected(self, bad_e: float):
        """Young's modulus E <= 0 must be rejected with an error in log."""
        m = Material(id=2, law=60, rho0=7.85e-9, params={"E": bad_e, "nu": 0.3})
        log = MessageLog()
        check_mat_law60(m, log)
        assert log.has_errors
        assert any("young" in e.lower() or "e" in e.lower() for e in log.errors)

    @pytest.mark.parametrize("bad_nu", [-0.1, -0.01, 0.5, 0.52, 0.9])
    def test_invalid_poisson_ratio_rejected(self, bad_nu: float):
        """Poisson's ratio outside [0.0, 0.5) must be rejected with an error in log."""
        m = Material(id=3, law=60, rho0=7.85e-9, params={"E": 200000.0, "nu": bad_nu})
        log = MessageLog()
        check_mat_law60(m, log)
        assert log.has_errors
        assert any("poisson" in e.lower() or "nu" in e.lower() for e in log.errors)

    @pytest.mark.parametrize("valid_nu", [0.0, 0.2, 0.33, 0.49, 0.4999])
    def test_valid_poisson_ratio_accepted(self, valid_nu: float):
        """Valid Poisson's ratio in [0.0, 0.5) must pass without error."""
        m = Material(id=4, law=60, rho0=7.85e-9, params={"E": 200000.0, "nu": valid_nu})
        log = MessageLog()
        check_mat_law60(m, log)
        assert not log.has_errors

    @pytest.mark.parametrize(
        "eps_t1,eps_t2",
        [
            (0.3, 0.2),      # inverted
            (0.25, 0.25),    # equal
            (1.0, 0.5),      # inverted large
        ],
    )
    def test_tensile_failure_strains_rejected(self, eps_t1: float, eps_t2: float):
        """Tensile failure strains with eps_t1 >= eps_t2 must be rejected."""
        m = Material(
            id=5,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "eps_t1": eps_t1, "eps_t2": eps_t2},
        )
        log = MessageLog()
        check_mat_law60(m, log)
        assert log.has_errors
        assert any("eps_t1" in e.lower() and "eps_t2" in e.lower() for e in log.errors)

    def test_tensile_failure_strains_default_accepted(self):
        """Default or valid tensile failure strains must pass without error."""
        # 1. Defaults (1e30, 2e30)
        m_default = Material(
            id=6,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "eps_t1": 1.0e30, "eps_t2": 2.0e30},
        )
        log = MessageLog()
        check_mat_law60(m_default, log)
        assert not log.has_errors

        # 2. Valid active strains (0.15 < 0.25)
        m_active = Material(
            id=7,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "eps_t1": 0.15, "eps_t2": 0.25},
        )
        log = MessageLog()
        check_mat_law60(m_active, log)
        assert not log.has_errors

    @pytest.mark.parametrize(
        "bad_rates",
        [
            [10.0, 5.0],                  # strictly decreasing
            [0.0, 10.0, 10.0],            # repeated rate
            [0.0, 50.0, 20.0, 100.0],     # non-monotonic order
            [100.0, 80.0, 60.0],          # reverse
        ],
    )
    def test_non_monotonic_strain_rates_rejected(self, bad_rates: list[float]):
        """Non-monotonic strain rates (RATE[i] >= RATE[i+1]) must be rejected."""
        m = Material(
            id=8,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "rates": bad_rates},
        )
        log = MessageLog()
        check_mat_law60(m, log)
        assert log.has_errors
        assert any("strain rates" in e.lower() or "strictly increasing" in e.lower() for e in log.errors)

    def test_strictly_increasing_strain_rates_accepted(self):
        """Strictly increasing strain rates must pass without error."""
        m = Material(
            id=9,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "rates": [0.0, 1.0, 10.0, 50.0, 100.0, 500.0]},
        )
        log = MessageLog()
        check_mat_law60(m, log)
        assert not log.has_errors

    def test_missing_curve_id_rejected(self):
        """Referenced curve ID not in model.functions or model.tables must be flagged."""
        m = Material(
            id=10,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "funcs": [101, 999]},
        )
        model = Model()
        model.functions[101] = FunctTable(101, [0.0, 0.1], [200.0, 300.0])
        log = MessageLog()
        check_mat_law60(m, log=log, model=model)
        assert log.has_errors
        assert any("999" in e for e in log.errors)

    def test_all_existing_curves_accepted(self):
        """All referenced curve IDs present in model.functions must pass."""
        m = Material(
            id=11,
            law=60,
            rho0=7.85e-9,
            params={"E": 200000.0, "nu": 0.3, "funcs": [101, 102]},
        )
        model = Model()
        model.functions[101] = FunctTable(101, [0.0, 0.1], [200.0, 300.0])
        model.functions[102] = FunctTable(102, [0.0, 0.1], [250.0, 350.0])
        log = MessageLog()
        check_mat_law60(m, log=log, model=model)
        assert not log.has_errors

    @pytest.mark.parametrize("rejected_fam", ["trusses", "beams", "springs"])
    def test_incompatible_elements_rejected_by_check_model(self, rejected_fam: str):
        """Trusses, beams, and springs must be rejected by check_model for LAW60."""
        mat = MatLaw60(id=1, rho=7.85e-9, e=200000.0, nu=0.3)
        mat.law = 60
        mat.law_name = "LAW60"

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        model.materials[1] = mat
        model.element_groups = lambda f=rejected_fam: [(f, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert log.has_errors
        assert any(
            "/MAT/LAW60/1 (/MAT/PLAS_T3) is not supported for" in str(e) and rejected_fam in str(e)
            for e in log.errors
        ), f"Failed to reject incompatible family {rejected_fam}: {log.errors}"

    @pytest.mark.parametrize("rejected_fam", ["trusses", "beams", "springs"])
    def test_incompatible_elements_rejected_by_check_mat_law60(self, rejected_fam: str):
        """check_mat_law60 directly flags incompatible element families attached in model."""
        mat = MatLaw60(id=1, rho=7.85e-9, e=200000.0, nu=0.3)

        class DummyElement:
            def __init__(self):
                self.mat_id = 1

        class DummyGroup:
            def values(self):
                return [DummyElement()]

        class DummyModel:
            def element_groups(self):
                return [(rejected_fam, DummyGroup())]

        log = MessageLog()
        check_mat_law60(mat, log=log, model=DummyModel())
        assert log.has_errors
        assert any("not supported for" in str(e) and rejected_fam in str(e) for e in log.errors)

    def test_compatible_elements_accepted(self):
        """Solid and shell families must be accepted for LAW60."""
        for fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            assert 60 in _ALLOWED_LAWS[fam]
            assert "60" in _ALLOWED_LAWS[fam]
            assert "LAW60" in _ALLOWED_LAWS[fam]
            assert "PLAS_T3" in _ALLOWED_LAWS[fam]
            assert "FABRIC" in _ALLOWED_LAWS[fam]
            assert "MAT_PLAS_T3" in _ALLOWED_LAWS[fam]
            assert "MAT_FABRIC" in _ALLOWED_LAWS[fam]

        for rejected in ("trusses", "beams"):
            assert 60 not in _ALLOWED_LAWS[rejected]
            assert "LAW60" not in _ALLOWED_LAWS[rejected]


# ============================================================================
# 4. Restart File (.rst) Serialization Audit
# ============================================================================

class TestLaw60RestartSerialization:
    """Audit serialization and engine continuation from .rst restart files."""

    def test_mat_law60_dataclass_pickling(self):
        """MatLaw60 dataclass survives direct serialization and restoration."""
        orig = MatLaw60(
            id=60,
            rho=1.5e-6,
            ref_rho=1.5e-6,
            e=2200.0,
            nu=0.33,
            eps_p_max=0.55,
            eps_t1=0.22,
            eps_t2=0.38,
            nfunc=5,
            fsmooth=1,
            mat_hard=0.2,
            fcut=120.0,
            xr_fun=2,
            ifunce=1,
            mat_fscale=1.15,
            einf=600.0,
            ce=12.0,
            funcs=[101, 102],
            fscales=[1.0, 1.1],
            rates=[0.0, 50.0],
            title="Pickle Serialization Test",
        )

        data = pickle.dumps(orig)
        restored = pickle.loads(data)

        assert restored.id == orig.id
        assert restored.rho == pytest.approx(orig.rho)
        assert restored.ref_rho == pytest.approx(orig.ref_rho)
        assert restored.rho0 == pytest.approx(orig.rho0)
        assert restored.e == pytest.approx(orig.e)
        assert restored.E == pytest.approx(orig.E)
        assert restored.nu == pytest.approx(orig.nu)
        assert restored.eps_p_max == pytest.approx(orig.eps_p_max)
        assert restored.eps_t1 == pytest.approx(orig.eps_t1)
        assert restored.eps_t2 == pytest.approx(orig.eps_t2)
        assert restored.nfunc == orig.nfunc
        assert restored.fsmooth == orig.fsmooth
        assert restored.mat_hard == pytest.approx(orig.mat_hard)
        assert restored.chard == pytest.approx(orig.chard)
        assert restored.fcut == pytest.approx(orig.fcut)
        assert restored.xr_fun == orig.xr_fun
        assert restored.ifunce == orig.ifunce
        assert restored.mat_fscale == pytest.approx(orig.mat_fscale)
        assert restored.fpscale == pytest.approx(orig.fpscale)
        assert restored.einf == pytest.approx(orig.einf)
        assert restored.ce == pytest.approx(orig.ce)
        assert restored.funcs == orig.funcs
        assert restored.fun_ids == orig.fun_ids
        assert restored.fscales == pytest.approx(orig.fscales)
        assert restored.rates == pytest.approx(orig.rates)
        assert restored.eps_rates == pytest.approx(orig.eps_rates)
        assert restored.title == orig.title

    def test_law60_params_physics_pickling(self):
        """Law60Params constructed via build_law60 survives serialization with exact wave speeds."""
        m = MatLaw60(
            id=1,
            rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            nfunc=2,
            funcs=[10, 20],
            fscales=[1.0, 1.1],
            rates=[0.0, 50.0],
        )
        functs = {
            10: FunctTable(10, [0.0, 0.1, 0.3], [250.0, 350.0, 450.0]),
            20: FunctTable(20, [0.0, 0.1, 0.3], [280.0, 380.0, 480.0]),
        }
        params = build_law60(m, functs=functs)

        data = pickle.dumps(params)
        restored: Law60Params = pickle.loads(data)

        assert restored.id == params.id
        assert restored.rho0 == pytest.approx(params.rho0)
        assert restored.E == pytest.approx(params.E)
        assert restored.nu == pytest.approx(params.nu)
        assert restored.nfunc == params.nfunc
        assert restored.sound_speed_solid(params.rho0) == pytest.approx(params.sound_speed_solid(params.rho0))
        assert restored.sound_speed_shell(params.rho0) == pytest.approx(params.sound_speed_shell(params.rho0))

    def test_extra_shapes_solid_and_shell(self):
        """extra_shapes must allocate uvar (5 + nfunc,) and off60 flags for solids and shells."""
        m_sol = Material(id=1, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.3, "nfunc": 5})
        s_sol = extra_shapes(m_sol, nip=None)
        assert s_sol["uvar"] == (10,)
        assert s_sol["off60"] == ()

        m_sh = Material(id=2, law=60, rho0=7.8e-9, params={"E": 200000.0, "nu": 0.3, "nfunc": 10})
        s_sh = extra_shapes(m_sh, nip=4)
        assert s_sh["uvar"] == (4, 15)
        assert s_sh["off60"] == (4,)

    def test_uvar_and_off60_material_state_restart_preservation(self, tmp_path: Path):
        """Verify element persistent uvar history and off60 flags survive write_restart and read_restart."""
        deck = StarterDeck("RST_UVAR_OFF")
        deck.funct(1, "Curve1", [(0.0, 200.0), (0.2, 400.0)])
        deck.mat_law60(mid=1, rho=7.85e-9, e=210000.0, nu=0.3, nfunc=5, funcs=[1])
        deck.prop_solid(pid=1, title="SolidProp", isolid=1)
        deck.prop_shell(pid=2, title="ShellProp", ishell=1, thick=1.0)
        deck.part(pid=1, title="SolidPart", prop_id=1, mat_id=1)
        deck.part(pid=2, title="ShellPart", prop_id=2, mat_id=1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
        ]
        deck.node(nodes)
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
        deck.shell(2, [[2, 1, 2, 3, 4]])

        rad_file = tmp_path / "rst_uvar_off_0000.rad"
        deck.write(str(rad_file))

        model = run_starter(str(rad_file))
        assert model is not None

        groups = dict(model.element_groups())
        assert "bricks" in groups
        assert "shells" in groups

        # Synthesize realistic uvar and off60 states in element groups
        bg = groups["bricks"]
        if "mat_extra" not in bg.state:
            bg.state["mat_extra"] = {}
        uvar_solid = np.array([[0.045, 320.0, 10.0, 1.0, 293.15, 1.0, 0.0, 0.0, 0.0, 0.0]])
        off60_solid = np.array([1.0])
        bg.state["mat_extra"]["uvar"] = uvar_solid
        bg.state["mat_extra"]["off60"] = off60_solid

        shg = groups["shells"]
        if "mat_extra" not in shg.state:
            shg.state["mat_extra"] = {}
        uvar_shell = np.zeros((1, 1, 10))
        uvar_shell[0, 0, :5] = [0.082, 350.0, 25.0, 1.0, 310.0]
        off60_shell = np.array([[0.0]])  # deleted integration point
        shg.state["mat_extra"]["uvar"] = uvar_shell
        shg.state["mat_extra"]["off60"] = off60_shell

        # Save to restart file
        rst_path = tmp_path / "rst_uvar_off_0001.rst"
        engine_dict = {
            "cycle": 150,
            "t": 1.5e-6,
            "dt": 1.0e-8,
            "energies": {"internal": 12.34, "kinetic": 5.67},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Restore from restart file
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 150
        assert rest_engine["t"] == pytest.approx(1.5e-6)
        assert rest_engine["energies"]["internal"] == pytest.approx(12.34)

        rest_groups = dict(rest_model.element_groups())
        rest_bg = rest_groups["bricks"]
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["uvar"],
            uvar_solid,
            err_msg="Solid uvar history variables did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["off60"],
            off60_solid,
            err_msg="Solid off60 flags did not survive restart restoration",
        )

        rest_shg = rest_groups["shells"]
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["uvar"],
            uvar_shell,
            err_msg="Shell uvar history variables did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["off60"],
            off60_shell,
            err_msg="Shell off60 flags did not survive restart restoration",
        )

    def test_engine_restart_continuation_solid_hexa8(self, tmp_path: Path):
        """Test complete engine simulation restart continuation for Hexa8 with LAW60."""
        name = "eng_rst_hexa"
        deck = StarterDeck(name)
        deck.funct(1, "YieldCurve", [(0.0, 250.0), (0.1, 350.0), (0.3, 450.0)])
        deck.mat_law60(mid=1, title="SteelLAW60", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
        deck.prop_solid(pid=1, title="PropHexa", isolid=1)
        deck.part(pid=1, title="PartHexa", prop_id=1, mat_id=1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 2.0, 0.0, 0.0), (3, 2.0, 2.0, 0.0), (4, 0.0, 2.0, 0.0),
            (5, 0.0, 0.0, 2.0), (6, 2.0, 0.0, 2.0), (7, 2.0, 2.0, 2.0), (8, 0.0, 2.0, 2.0),
        ]
        deck.node(nodes)
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
        deck.grnod_node(1, "TopNodes", [5, 6, 7, 8])
        deck.inivel_tra(1, "TensionKick", [0.0, 0.0, 25.0], 1)

        rad_file = tmp_path / f"{name}_0000.rad"
        deck.write(str(rad_file))

        # Starter
        m_start = run_starter(str(rad_file))
        assert m_start is not None
        assert (tmp_path / f"{name}_0000.rst").exists()

        # Engine Leg 1: run to 1.5e-7 s
        eng1_path = tmp_path / f"{name}_0001.rad"
        eng1_path.write_text(
            f"/RUN/{name}/1\n1.5e-7\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg1 = run_engine(str(eng1_path))
        assert m_leg1 is not None
        rst1_path = tmp_path / f"{name}_0001.rst"
        assert rst1_path.exists()

        # Verify state in rst1
        m_rst1, eng_rst1 = read_restart(str(rst1_path))
        assert eng_rst1 is not None
        assert eng_rst1["cycle"] == m_leg1.engine_state.cycle
        bg1 = dict(m_rst1.element_groups())["bricks"]
        assert "mat_extra" in bg1.state
        assert "uvar" in bg1.state["mat_extra"]
        assert "off60" in bg1.state["mat_extra"]

        # Engine Leg 2: resume from _0001.rst and run to 3.0e-7 s
        eng2_path = tmp_path / f"{name}_0002.rad"
        eng2_path.write_text(
            f"/RUN/{name}/2\n3.0e-7\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg2 = run_engine(str(eng2_path))
        assert m_leg2 is not None
        assert m_leg2.engine_state.cycle > m_leg1.engine_state.cycle
        assert (tmp_path / f"{name}_0002.rst").exists()

        # Verify continuation preserved history variables and active state
        bg2 = dict(m_leg2.element_groups())["bricks"]
        assert bg2.state["mat_extra"]["off60"][0] == 1.0

    def test_engine_restart_continuation_shell_bt4(self, tmp_path: Path):
        """Test complete engine simulation restart continuation for BT4 shell with LAW60."""
        name = "eng_rst_shell"
        deck = StarterDeck(name)
        deck.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 300.0), (0.3, 400.0)])
        deck.mat_fabric(mid=1, title="FabricShell", rho=1.5e-6, e=2500.0, nu=0.35, funcs=[1])
        deck.prop_shell(pid=1, title="PropShell", ishell=1, thick=0.5)
        deck.part(pid=1, title="PartShell", prop_id=1, mat_id=1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ]
        deck.node(nodes)
        deck.shell(1, [[1, 1, 2, 3, 4]])
        deck.grnod_node(1, "PullNodes", [2, 3])
        deck.inivel_tra(1, "InplaneKick", [5.0, 0.0, 0.0], 1)

        rad_file = tmp_path / f"{name}_0000.rad"
        deck.write(str(rad_file))

        # Starter
        m_start = run_starter(str(rad_file))
        assert m_start is not None

        # Leg 1: run to 2.0e-5 s
        eng1_path = tmp_path / f"{name}_0001.rad"
        eng1_path.write_text(
            f"/RUN/{name}/1\n2.0e-5\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg1 = run_engine(str(eng1_path))
        assert m_leg1 is not None

        # Leg 2: resume and run to 4.0e-5 s
        eng2_path = tmp_path / f"{name}_0002.rad"
        eng2_path.write_text(
            f"/RUN/{name}/2\n4.0e-5\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg2 = run_engine(str(eng2_path))
        assert m_leg2 is not None
        assert m_leg2.engine_state.cycle > m_leg1.engine_state.cycle
        assert (tmp_path / f"{name}_0002.rst").exists()

        shg2 = dict(m_leg2.element_groups())["shells"]
        assert "mat_extra" in shg2.state
        assert "uvar" in shg2.state["mat_extra"]
        assert "off60" in shg2.state["mat_extra"]

    def test_engine_restart_unchained_parity(self, tmp_path: Path):
        """Verify that restart chaining matches unchained reference execution."""
        def make_deck(prefix: str) -> Path:
            d = StarterDeck(prefix)
            d.funct(1, "YieldCurve", [(0.0, 220.0), (0.1, 320.0), (0.3, 420.0)])
            d.mat_law60(mid=1, title="SteelParity", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
            d.prop_solid(pid=1, title="PropSolid", isolid=1)
            d.part(pid=1, title="PartSolid", prop_id=1, mat_id=1)
            nodes = [
                (1, 0.0, 0.0, 0.0), (2, 2.0, 0.0, 0.0), (3, 2.0, 2.0, 0.0), (4, 0.0, 2.0, 0.0),
                (5, 0.0, 0.0, 2.0), (6, 2.0, 0.0, 2.0), (7, 2.0, 2.0, 2.0), (8, 0.0, 2.0, 2.0),
            ]
            d.node(nodes)
            d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
            d.grnod_node(1, "KickNodes", [5, 6, 7, 8])
            d.inivel_tra(1, "Kick", [0.0, 0.0, 50.0], 1)
            f = tmp_path / f"{prefix}_0000.rad"
            d.write(str(f))
            return f

        # Probe to determine exact stable time step
        probe_rad = make_deck("probe")
        run_starter(str(probe_rad))
        probe_eng = tmp_path / "probe_0001.rad"
        probe_eng.write_text("/RUN/probe/1\n1.0e-9\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        run_engine(str(probe_eng))
        _, eng_dict = read_restart(str(tmp_path / "probe_0001.rst"))
        dt0 = eng_dict["dt"]

        # Run legs aligned on exact cycle boundaries
        t1 = 2 * dt0
        t2 = 4 * dt0

        # 1. Unchained execution: run from 0 to t2 directly
        unc_rad = make_deck("unc_run")
        run_starter(str(unc_rad))
        unc_eng = tmp_path / "unc_run_0001.rad"
        unc_eng.write_text(f"/RUN/unc_run/1\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_unc = run_engine(str(unc_eng))

        # 2. Chained execution: Leg 1 (0 to t1), Leg 2 (t1 to t2)
        chn_rad = make_deck("chn_run")
        run_starter(str(chn_rad))
        chn_eng1 = tmp_path / "chn_run_0001.rad"
        chn_eng1.write_text(f"/RUN/chn_run/1\n{t1!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn1 = run_engine(str(chn_eng1))

        chn_eng2 = tmp_path / "chn_run_0002.rad"
        chn_eng2.write_text(f"/RUN/chn_run/2\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn2 = run_engine(str(chn_eng2))

        # 3. Compare unchained vs chained kinematics and energies
        assert m_chn2.engine_state.cycle == m_unc.engine_state.cycle
        np.testing.assert_allclose(
            m_chn2.x,
            m_unc.x,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Nodal positions mismatch between chained and unchained runs",
        )
        np.testing.assert_allclose(
            m_chn2.v,
            m_unc.v,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Nodal velocities mismatch between chained and unchained runs",
        )
