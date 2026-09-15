"""Roundtrip tests for /MAT/LAW110 (/MAT/VEGTER, /MAT/PLAS_VEGTER) deck writer and parser (M579).

Verifies:
  - Emission via StarterDeck.mat_law110, StarterDeck.mat_vegter, StarterDeck.mat_plas_vegter.
  - Roundtrip accuracy for:
      * Icrit = 1: Standard Vegter with reference points (fun, r, fps1, fps2, fsh).
      * Icrit = 2: Vegter with Alps hinge point parameter.
      * Icrit = 3: Vegter with Rm, Ag, r engineering inputs and hardening curves.
      * Icrit = 4: Vegter-Lite with weight factors (wps, wsh).
  - Roundtrip emission from an existing MaterialLaw110 dataclass instance.
"""

from pathlib import Path
import pytest

from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import read_starter_deck
from pyradioss.model.entities import MaterialLaw110


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "deck_0000.rad"
    deck_path.write_text(text, encoding="utf-8")
    raw_deck = read_deck(deck_path)
    model = Model()
    log = MessageLog()
    read_starter_deck(raw_deck, model, log)
    return model, log


class TestLaw110Roundtrip:
    """Test deck writer emission and reader roundtrip."""

    def test_roundtrip_icrit1_standard(self, tmp_path: Path):
        """Test Icrit=1 standard Vegter roundtrip."""
        d1 = StarterDeck("TEST_ICRIT1")
        angles = [
            [1.0, 1.8, 1.05, 0.02, 0.52],
            [0.98, 1.45, 1.02, 0.01, 0.50],
            [1.02, 2.10, 1.08, 0.03, 0.54],
        ]
        d1.mat_law110(
            mid=1101,
            title="Vegter_Icrit1",
            rho=7.85e-6,
            refer_rho=7.85e-6,
            young=210000.0,
            nu=0.3,
            ires=2,
            icrit=1,
            tab_yld=0,
            xscale=1.0,
            yscale=1.0,
            fbi=1.0,
            rhobi=1.0,
            sigma_r=250.0,
            dsigm=150.0,
            beta=12.0,
            omega=0.5,
            hard_n=0.22,
            eps0=0.001,
            sigs=500.0,
            dg0=1.0e5,
            deps0=1.0,
            m=0.1,
            tini=293.0,
            chard=0.0,
            fcut=1.0e20,
            vp=2,
            ismooth=1,
            tab_temp=0,
            angles_data=angles,
        )
        out_path1 = tmp_path / "icrit1_0000.rad"
        d1.write(str(out_path1))

        # Read back
        raw = read_deck(out_path1)
        model = Model()
        log = MessageLog()
        read_starter_deck(raw, model, log)
        assert not log.errors, f"Errors parsing written deck: {log.errors}"
        assert 1101 in model.mat_law110s

        m = model.mat_law110s[1101]
        assert m.rho0 == pytest.approx(7.85e-6)
        assert m.refer_rho == pytest.approx(7.85e-6)
        assert m.young == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.3)
        assert m.icrit == 1
        assert m.ires == 2
        assert m.sigma_r == pytest.approx(250.0)
        assert m.dsigm == pytest.approx(150.0)
        assert m.beta == pytest.approx(12.0)
        assert m.omega == pytest.approx(0.5)
        assert m.hard_n == pytest.approx(0.22)
        assert m.eps0 == pytest.approx(0.001)
        assert m.sigs == pytest.approx(500.0)
        assert m.dg0 == pytest.approx(1.0e5)
        assert m.deps0 == pytest.approx(1.0)
        assert m.m == pytest.approx(0.1)
        assert m.tini == pytest.approx(293.0)
        assert len(m.angles_data) == 3
        for i in range(3):
            for j in range(5):
                assert m.angles_data[i][j] == pytest.approx(angles[i][j])

    def test_roundtrip_icrit2_alps(self, tmp_path: Path):
        """Test Icrit=2 alps formulation roundtrip."""
        d = StarterDeck("TEST_ICRIT2")
        angles = [
            [1.0, 1.7, 1.04, 0.48, 0.51],
            [1.01, 1.5, 1.02, 0.50, 0.49],
            [0.99, 2.0, 1.06, 0.52, 0.53],
        ]
        d.mat_vegter(
            mid=1102,
            title="Vegter_Alps",
            rho=7.8e-6,
            young=205000.0,
            nu=0.31,
            ires=1,
            icrit=2,
            sigma_r=220.0,
            dsigm=180.0,
            beta=8.0,
            hard_n=0.25,
            angles_data=angles,
        )
        out_path = tmp_path / "icrit2_0000.rad"
        d.write(str(out_path))

        raw = read_deck(out_path)
        model = Model()
        log = MessageLog()
        read_starter_deck(raw, model, log)
        assert not log.errors
        assert 1102 in model.mat_law110s

        m = model.mat_law110s[1102]
        assert m.icrit == 2
        assert m.ires == 1
        assert m.young == pytest.approx(205000.0)
        assert m.nu == pytest.approx(0.31)
        assert len(m.angles_data) == 3
        for i in range(3):
            for j in range(5):
                assert m.angles_data[i][j] == pytest.approx(angles[i][j])

    def test_roundtrip_icrit3_engineering(self, tmp_path: Path):
        """Test Icrit=3 engineering Rm/Ag/r formulation roundtrip."""
        d = StarterDeck("TEST_ICRIT3")
        d.mat_plas_vegter(
            mid=1103,
            title="Vegter_RmAg",
            rho=7.8e-6,
            young=210000.0,
            nu=0.3,
            icrit=3,
            sigma_r=260.0,
            dsigm=140.0,
            beta=15.0,
            hard_n=0.20,
            rm_0=350.0,
            rm_45=330.0,
            rm_90=360.0,
            ag_0=18.0,
            ag_45=22.0,
            ag_90=16.0,
            r_0=1.8,
            r_45=1.4,
            r_90=2.2,
        )
        out_path = tmp_path / "icrit3_0000.rad"
        d.write(str(out_path))

        raw = read_deck(out_path)
        model = Model()
        log = MessageLog()
        read_starter_deck(raw, model, log)
        assert not log.errors
        assert 1103 in model.mat_law110s

        m = model.mat_law110s[1103]
        assert m.icrit == 3
        assert m.rm_0 == pytest.approx(350.0)
        assert m.rm_45 == pytest.approx(330.0)
        assert m.rm_90 == pytest.approx(360.0)
        assert m.ag_0 == pytest.approx(18.0)
        assert m.ag_45 == pytest.approx(22.0)
        assert m.ag_90 == pytest.approx(16.0)
        assert m.r_0 == pytest.approx(1.8)
        assert m.r_45 == pytest.approx(1.4)
        assert m.r_90 == pytest.approx(2.2)

    def test_roundtrip_icrit4_lite(self, tmp_path: Path):
        """Test Icrit=4 Vegter-Lite formulation roundtrip."""
        d = StarterDeck("TEST_ICRIT4")
        angles = [
            [1.0, 1.9, 0.55, 0.52],
            [0.97, 1.4, 0.50, 0.49],
            [1.03, 2.2, 0.56, 0.53],
        ]
        d.mat_law110(
            mid=1104,
            title="Vegter_Lite",
            rho=7.8e-6,
            young=210000.0,
            nu=0.3,
            icrit=4,
            sigma_r=280.0,
            dsigm=120.0,
            beta=10.0,
            hard_n=0.24,
            fbi=1.04,
            rhobi=0.96,
            angles_data=angles,
        )
        out_path = tmp_path / "icrit4_0000.rad"
        d.write(str(out_path))

        raw = read_deck(out_path)
        model = Model()
        log = MessageLog()
        read_starter_deck(raw, model, log)
        assert not log.errors
        assert 1104 in model.mat_law110s

        m = model.mat_law110s[1104]
        assert m.icrit == 4
        assert m.fbi == pytest.approx(1.04)
        assert m.rhobi == pytest.approx(0.96)
        assert len(m.angles_data) == 3
        for i in range(3):
            for j in range(4):
                assert m.angles_data[i][j] == pytest.approx(angles[i][j])

    def test_roundtrip_from_dataclass_instance(self, tmp_path: Path):
        """Test passing an existing MaterialLaw110 dataclass directly to StarterDeck."""
        mat_in = MaterialLaw110(
            mid=1105,
            title="DataclassInstance",
            rho0=7.8e-6,
            young=200000.0,
            nu=0.29,
            icrit=1,
            sigma_r=300.0,
            dsigm=100.0,
            beta=14.0,
            hard_n=0.19,
            angles_data=[[1.0, 1.6, 1.0, 0.0, 0.5]],
        )
        d = StarterDeck("DATACLASS_DECK")
        d.mat_law110(mat_in)
        out_path = tmp_path / "dataclass_0000.rad"
        d.write(str(out_path))

        raw = read_deck(out_path)
        model = Model()
        log = MessageLog()
        read_starter_deck(raw, model, log)
        assert not log.errors
        assert 1105 in model.mat_law110s
        m = model.mat_law110s[1105]
        assert m.id == 1105
        assert m.young == pytest.approx(200000.0)
        assert m.nu == pytest.approx(0.29)
        assert m.sigma_r == pytest.approx(300.0)
        assert len(m.angles_data) == 1
