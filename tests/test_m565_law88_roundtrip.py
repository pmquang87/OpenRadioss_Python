"""
Milestone M565: /MAT/LAW88 (/MAT/TABULATED_HYPERELASTIC, /MAT/HYPER_ELAS, /MAT/TAB_HYP)
Exhaustive Roundtrip, Serialization, Boundary Cases & Official Benchmark Suite.

Fortran origins:
  - starter/source/materials/mat/mat088/hm_read_mat88.F90 (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat088/sigeps88.F90 (3D solid constitutive update kernel)
  - engine/source/materials/mat/mat088/sigeps88c.F90 (2D shell plane-stress constitutive update kernel)
  - hm_cfg_files/config/CFG/radioss2017/MAT/mat_law88.cfg
  - hm_cfg_files/config/CFG/radioss2026/MAT/mat_law88.cfg
  - Official deck: tests/data/rd_decks/rd_e/RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW88/LAW88.txt
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import read_mat_law88
from pyradioss.materials.law88_tab_hyp import (
    Law88Params,
    MatparamLaw88,
    build_law88,
    sound_speed_solid,
    sound_speed_shell,
    resolve,
)
from pyradioss.model.entities import (
    MaterialLaw88,
    MatLaw88,
    MatTabulatedHyperelastic,
    MatHyperElas,
    MatTabHyp,
    Material,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law88


def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW88") -> Tuple[Model, MessageLog]:
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n/END\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law88(block, model, log)
    return model, log


def _assert_law88_exact(
    mat: MaterialLaw88,
    *,
    mid: int,
    title: str,
    rho0: float,
    nu: float,
    bulk: float,
    fcut: float = 0.0,
    fsmooth: int = 0,
    nl: int = 1,
    ifunc_unload: int = 0,
    fscale_unload: float = 1.0,
    hys: float = 0.0,
    shape: float = 1.0,
    tension: int = 0,
    rtype: int = 0,
    func_load_list: Sequence[int] = (1,),
    fscale_load_list: Sequence[float] = (1.0,),
    rate_load_list: Sequence[float] = (0.0,),
    lamfit_list: Sequence[float] = (0.0,),
    sgl: float = 0.0,
    sw: float = 0.0,
    st: float = 0.0,
    g: float = 0.0,
    sigf: float = 0.0,
    kfail: float = 0.0,
    gam1: float = 0.0,
    gam2: float = 0.0,
    eh: float = 0.0,
    failip: int = 0,
    beta: float = 0.0,
    tol: float = 1e-12,
) -> None:
    assert mat.id == mid
    assert mat.title == title
    assert math.isclose(mat.rho0, rho0, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.nu, nu, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.bulk, bulk, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.fcut, fcut, rel_tol=tol, abs_tol=tol)
    assert mat.fsmooth == fsmooth
    assert mat.nl == nl
    assert mat.ifunc_unload == ifunc_unload
    assert math.isclose(mat.fscale_unload, fscale_unload, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.hys, hys, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.shape, shape, rel_tol=tol, abs_tol=tol)
    assert mat.tension == tension
    assert mat.rtype == rtype
    assert list(mat.func_load_list) == list(func_load_list)
    assert len(mat.fscale_load_list) == len(fscale_load_list)
    for a, b in zip(mat.fscale_load_list, fscale_load_list):
        assert math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    assert len(mat.rate_load_list) == len(rate_load_list)
    for a, b in zip(mat.rate_load_list, rate_load_list):
        assert math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    assert len(mat.lamfit_list) == len(lamfit_list)
    for a, b in zip(mat.lamfit_list, lamfit_list):
        assert math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.sgl, sgl, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.sw, sw, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.st, st, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.g, g, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.sigf, sigf, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.kfail, kfail, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.gam1, gam1, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.gam2, gam2, rel_tol=tol, abs_tol=tol)
    assert math.isclose(mat.eh, eh, rel_tol=tol, abs_tol=tol)
    assert mat.failip == failip
    assert math.isclose(mat.beta, beta, rel_tol=tol, abs_tol=tol)


class TestLaw88FixedFormatRoundtrip:
    @pytest.mark.parametrize("kw_alias,method_name", [
        ("LAW88", "mat_law88"),
        ("TABULATED_HYPERELASTIC", "mat_tabulated_hyperelastic"),
        ("HYPER_ELAS", "mat_hyper_elas"),
        ("TAB_HYP", "mat_tab_hyp"),
    ])
    def test_fixed_format_all_synonyms(self, tmp_path: Path, kw_alias: str, method_name: str):
        deck = StarterDeck("M565_FIXED")
        writer_fn = getattr(deck, method_name)

        writer_fn(
            mid=10,
            title=f"Rubber_{kw_alias}",
            rho0=1.1e-3,
            nu=0.495,
            bulk=1500.0,
            fcut=25.0,
            fsmooth=1,
            nl=2,
            ifunc_unload=100,
            fscale_unload=1.0,
            hys=0.5,
            shape=2.0,
            tension=1,
            rtype=0,
            func_load_list=[101, 102],
            fscale_load_list=[1.0, 1.2],
            rate_load_list=[0.0, 100.0],
            lamfit_list=[1e-3, 1e-3],
            sgl=50.0,
            sw=10.0,
            st=2.0,
            g=15.0,
            sigf=200.0,
            kfail=0.9,
            gam1=0.02,
            gam2=0.03,
            eh=0.1,
            failip=1,
            fixed_format=True,
        )

        deck_str = deck.write()
        assert f"/MAT/{kw_alias}/10" in deck_str
        assert f"Rubber_{kw_alias}" in deck_str

        model, log = _parse_deck_str(tmp_path, deck_str, name=f"fixed_{kw_alias}")
        assert 10 in model.mat_law88s
        mat = model.mat_law88s[10]

        assert math.isclose(mat.fscale_load_list[0], 0.05, rel_tol=1e-12)
        assert math.isclose(mat.fscale_load_list[1], 0.06, rel_tol=1e-12)
        assert math.isclose(mat.fscale_load_card[0], 1.0, rel_tol=1e-12)
        assert math.isclose(mat.fscale_load_card[1], 1.2, rel_tol=1e-12)

        deck2 = StarterDeck("M565_FIXED_RT")
        writer_fn2 = getattr(deck2, method_name)
        writer_fn2(mat, fixed_format=True)
        deck_str2 = deck2.write()

        model2, _ = _parse_deck_str(tmp_path, deck_str2, name=f"fixed_rt2_{kw_alias}")
        mat2 = model2.mat_law88s[10]
        assert math.isclose(mat2.fscale_load_list[0], 0.05, rel_tol=1e-12)
        assert math.isclose(mat2.fscale_load_list[1], 0.06, rel_tol=1e-12)
        assert math.isclose(mat2.bulk, 1500.0, rel_tol=1e-12)
        assert math.isclose(mat2.kfail, 0.9, rel_tol=1e-12)


class TestLaw88FreeFormatRoundtrip:
    @pytest.mark.parametrize("kw_alias,method_name", [
        ("LAW88", "mat_law88"),
        ("TABULATED_HYPERELASTIC", "mat_tabulated_hyperelastic"),
        ("HYPER_ELAS", "mat_hyper_elas"),
        ("TAB_HYP", "mat_tab_hyp"),
    ])
    def test_free_format_all_synonyms(self, tmp_path: Path, kw_alias: str, method_name: str):
        deck = StarterDeck("M565_FREE")
        writer_fn = getattr(deck, method_name)

        writer_fn(
            mid=20,
            title=f"FreeRubber_{kw_alias}",
            rho0=9.5e-4,
            nu=0.48,
            bulk=1200.0,
            fcut=10.0,
            fsmooth=0,
            nl=1,
            ifunc_unload=0,
            fscale_unload=1.0,
            hys=0.7,
            shape=1.5,
            tension=0,
            rtype=1,
            func_load_list=[201],
            fscale_load_list=[1.5],
            rate_load_list=[0.0],
            lamfit_list=[0.0],
            sgl=25.0,
            sw=0.0,
            st=0.0,
            g=8.0,
            sigf=150.0,
            kfail=0.5,
            gam1=0.01,
            gam2=0.02,
            eh=0.05,
            failip=2,
            free=True,
        )

        deck_str = deck.write()
        assert f"/MAT/{kw_alias}/20" in deck_str

        model, log = _parse_deck_str(tmp_path, deck_str, name=f"free_{kw_alias}")
        assert 20 in model.mat_law88s
        mat = model.mat_law88s[20]

        _assert_law88_exact(
            mat,
            mid=20,
            title=f"FreeRubber_{kw_alias}",
            rho0=9.5e-4,
            nu=0.48,
            bulk=1200.0,
            fcut=10.0,
            fsmooth=0,
            nl=1,
            ifunc_unload=0,
            fscale_unload=1.0,
            hys=0.7,
            shape=1.5,
            tension=0,
            rtype=1,
            func_load_list=[201],
            fscale_load_list=[1.5],
            rate_load_list=[0.0],
            lamfit_list=[0.0],
            sgl=25.0,
            sw=0.0,
            st=0.0,
            g=8.0,
            sigf=150.0,
            kfail=0.5,
            gam1=0.01,
            gam2=0.02,
            eh=0.05,
            failip=2,
            beta=0.0,
            tol=1e-12,
        )


class TestLaw88SixCardsExactRoundtrip:
    def test_all_six_cards_roundtrip_precision(self, tmp_path: Path):
        m_in = MaterialLaw88(
            id=33,
            title="HighPrecisionOgden",
            rho0=1.23456789e-3,
            ref_rho=1.23456789e-3,
            nu=0.49123456,
            bulk=9876.54321,
            fcut=123.456,
            fsmooth=2,
            nl=3,
            ifunc_unload=400,
            fscale_unload=0.888888,
            hys=0.654321,
            shape=1.414213,
            tension=1,
            rtype=2,
            func_load_list=[401, 402, 403],
            fscale_load_list=[1.0, 1.25, 1.5],
            fscale_load_card=[1.0, 1.25, 1.5],
            rate_load_list=[0.0, 10.0, 100.0],
            lamfit_list=[1e-4, 2e-4, 3e-4],
            sgl=100.0,
            sw=0.0,
            st=0.0,
            g=45.6789,
            sigf=333.222,
            kfail=0.75,
            gam1=0.1234,
            gam2=0.5678,
            eh=0.05,
            failip=3,
            beta=0.0,
        )

        deck = StarterDeck("TEST_6CARDS")
        deck.mat_law88(m_in, fixed_format=True)
        deck_str = deck.write()

        model, log = _parse_deck_str(tmp_path, deck_str, name="six_cards_prec")
        m_out = model.mat_law88s[33]

        _assert_law88_exact(
            m_out,
            mid=33,
            title="HighPrecisionOgden",
            rho0=1.23456789e-3,
            nu=0.49123456,
            bulk=9876.54321,
            fcut=123.456,
            fsmooth=2,
            nl=3,
            ifunc_unload=400,
            fscale_unload=0.888888,
            hys=0.654321,
            shape=1.414213,
            tension=1,
            rtype=2,
            func_load_list=[401, 402, 403],
            fscale_load_list=[1.0, 1.25, 1.5],
            rate_load_list=[0.0, 10.0, 100.0],
            lamfit_list=[1e-4, 2e-4, 3e-4],
            sgl=100.0,
            sw=0.0,
            st=0.0,
            g=45.6789,
            sigf=333.222,
            kfail=0.75,
            gam1=0.1234,
            gam2=0.5678,
            eh=0.05,
            failip=3,
            beta=0.0,
            tol=1e-8,
        )

    def test_pickle_and_dict_serialization(self):
        m = MaterialLaw88(
            id=44,
            title="PickleRubber",
            rho0=1.0e-3,
            nu=0.49,
            bulk=2000.0,
            nl=1,
            func_load_list=[501],
            fscale_load_list=[1.0],
            rate_load_list=[0.0],
            g=25.0,
            kfail=0.8,
            failip=1,
        )

        pickled = pickle.dumps(m)
        m_unpickled = pickle.loads(pickled)
        assert m_unpickled.id == 44
        assert m_unpickled.title == "PickleRubber"
        assert math.isclose(m_unpickled.bulk, 2000.0)
        assert m_unpickled.func_load_list == [501]

        mp = MatparamLaw88.from_dict_or_obj(m)
        assert math.isclose(mp.bulk, 2000.0)
        assert math.isclose(mp.shear, 25.0)
        assert mp.iparam[2] == 1
        assert mp.iparam[4] == 1
        assert math.isclose(mp.uparam[4], 0.8)


class TestOfficialBenchmarkDeckLAW88:
    BENCHMARK_PATH = Path(
        "tests/data/rd_decks/rd_e/RD-E-5600_Hyperelastic_material/"
        "56_HyperElastic_Material/Ogden_model/LAW88/LAW88.txt"
    )

    def test_official_benchmark_law88_deck_parse(self):
        assert self.BENCHMARK_PATH.exists(), f"Benchmark deck missing at {self.BENCHMARK_PATH}"

        blocks = read_deck(str(self.BENCHMARK_PATH))
        assert len(blocks) > 0, "Failed to parse any blocks from LAW88.txt"

        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law88(block, model, log)

        assert 1 in model.mat_law88s, "Material 1 not found in model.mat_law88s"
        assert 1 in model.materials, "Material 1 not found in model.materials"

        m88 = model.mat_law88s[1]
        mat = model.materials[1]

        assert m88.id == 1
        assert m88.title.lower() == "rubber"
        assert math.isclose(m88.rho0, 1e-9, rel_tol=1e-12)
        assert math.isclose(m88.nu, 0.4997, rel_tol=1e-12)
        assert math.isclose(m88.bulk, 913.0824653215, rel_tol=1e-12)
        assert m88.nl == 1
        assert m88.ifunc_unload == 0
        assert math.isclose(m88.fscale_unload, 1.0, rel_tol=1e-12)
        assert m88.shape == 1.0
        assert m88.func_load_list == [3]
        assert m88.fscale_load_list == [1.0]
        assert m88.rate_load_list == [0.0]

        log_check = MessageLog()
        check_mat_law88(mat=m88, log=log_check)
        assert not log_check.has_errors, f"check_mat_law88 generated errors: {log_check.errors}"

    def test_official_benchmark_law88_starter_engine_moduli(self):
        blocks = read_deck(str(self.BENCHMARK_PATH))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law88(block, model, log)

        m88 = model.mat_law88s[1]
        mp = MatparamLaw88.from_dict_or_obj(m88)

        c_solid = sound_speed_solid(mp)
        c_shell = sound_speed_shell(mp)
        assert c_solid > 0.0, f"Solid sound speed must be positive: {c_solid}"
        assert c_shell > 0.0, f"Shell sound speed must be positive: {c_shell}"
        assert math.isfinite(c_solid)
        assert math.isfinite(c_shell)

    def test_official_benchmark_curve_resolution(self):
        blocks = read_deck(str(self.BENCHMARK_PATH))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law88(block, model, log)

        class MockFunct:
            def __init__(self, fid: int, x: list[float], y: list[float]):
                self.id = fid
                self.x = np.array(x, dtype=np.float64)
                self.y = np.array(y, dtype=np.float64)

        model.functions[3] = MockFunct(3, [0.0, 0.2, 0.5, 1.0, 1.5], [0.0, 0.5, 1.2, 2.8, 5.0])

        mat = model.materials[1]
        resolve(mat, model, log)

        assert "table" in mat.params
        tables = mat.params["table"]
        assert len(tables) >= 1
        tbl = tables[0]
        assert tbl.ndim == 1
        assert np.all(tbl.x1 >= 1.0)
        assert len(tbl.x1) == 300
        assert len(tbl.y1d) == 300


class TestLaw88BoundaryChecksAndRejections:
    """Audit starter checks, validation limits, and element rejections."""

    def test_rejection_of_non_positive_density(self):
        mat = MaterialLaw88(id=1, rho0=0.0, bulk=100.0, nl=1, func_load_list=[1])
        log = MessageLog()
        check_mat_law88(mat=mat, log=log)
        errors = [m for m in log.errors if "DENSITY" in m.upper()]
        assert len(errors) > 0

    def test_rejection_of_zero_nl(self):
        mat = MaterialLaw88(id=2, rho0=1e-3, bulk=100.0, nl=0)
        log = MessageLog()
        check_mat_law88(mat=mat, log=log)
        errors = [m for m in log.errors if "NL = 0" in m.upper() or "NO LOADING" in m.upper()]
        assert len(errors) > 0

    def test_rejection_of_descending_strain_rates(self):
        mat = MaterialLaw88(
            id=3,
            rho0=1e-3,
            bulk=100.0,
            nl=2,
            func_load_list=[1, 2],
            rate_load_list=[100.0, 10.0],  # Descending!
        )
        log = MessageLog()
        check_mat_law88(mat=mat, log=log)
        errors = [m for m in log.errors if "ASCENDING" in m.upper() or "RATE" in m.upper()]
        assert len(errors) > 0

    def test_rejection_of_1d_elements(self):
        mat = MaterialLaw88(id=4, rho0=1e-3, bulk=100.0, nl=1, func_load_list=[1])
        model = Model()
        model.mat_law88s[4] = mat
        model.materials[4] = Material(id=4, law=88, rho0=1e-3, title="LAW88")
        model.parts[1] = Part(id=1, mat_id=4, prop_id=1, title="BeamPart")
        model.properties[1] = Property(id=1, type="BEAM", title="BeamProp")
        model.raw_elems["BEAM"] = [(1, 1, [1, 2, 3])]

        log = MessageLog()
        check_mat_law88(model=model, mat_id=4, log=log)
        errors = [m for m in log.errors if "1D" in m.upper() or "BEAM" in m.upper()]
        assert len(errors) > 0
