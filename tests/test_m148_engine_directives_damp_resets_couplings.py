"""Tests for Milestone M148: Engine Directives Suite II.

Tests engine keywords:
- /DAMP, /DAMP/DT
- /MASS/RESET
- /SENSOR/RESET, /SENS/RESET
- /VIPER, /VIPER/ON
- /MADYMO/ON, /MADYMO/ON2, /MADYMO/MPP
- /RAD2R/ON, /RAD2RAD/ON
- /FVMBAG/REMESH, /FVMBAG/MODIF
- /PERF/SORT1, /PERF/SORT2, /PERF/SORT3
- /DT1TET10, /DTTSH
- /REPORT, /REPORT/DT
- /NEGVOL/STOP, /NEGVOL/DEL
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_engine(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_engine_damp_and_dt(tmp_path):
    deck = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/DAMP/DT
0.001
/DAMP/1
101
0.05 0.02
0.0 5.0
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0
    assert ec.damp_dt == pytest.approx(0.001)
    assert ec.damp_grpart == 101
    assert ec.damp_alpha == pytest.approx(0.05)
    assert ec.damp_beta == pytest.approx(0.02)
    assert ec.damp_tstart == pytest.approx(0.0)
    assert ec.damp_tstop == pytest.approx(5.0)


def test_engine_mass_and_sensor_resets(tmp_path):
    deck = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/MASS/RESET
/SENSOR/RESET
10 20 30
40 50
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0
    assert ec.mass_reset is True
    assert ec.sensor_reset == [10, 20, 30, 40, 50]

    # Test empty /SENS/RESET resets all (0)
    deck2 = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/SENS/RESET
/END
"""
    ec2, log2 = _parse_engine(tmp_path, deck2)
    assert len(log2.errors) == 0
    assert ec2.sensor_reset == [0]


def test_engine_couplings(tmp_path):
    deck = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/VIPER/ON
/MADYMO/MPP
/RAD2R/ON
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0
    assert ec.viper_active is True
    assert ec.madymo_mode == "MPP"
    assert ec.rad2r_active is True


def test_engine_airbag_and_perf_sorting(tmp_path):
    deck = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/FVMBAG/REMESH
/FVMBAG/MODIF
/PERF/SORT2
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0
    assert ec.fvbag_remesh is True
    assert ec.fvbag_modif is True
    assert ec.perf_sort == 2

    deck_sort3 = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/PERF/SORT3
/END
"""
    ec3, log3 = _parse_engine(tmp_path, deck_sort3)
    assert len(log3.errors) == 0
    assert ec3.perf_sort == 0


def test_engine_element_dt_and_reports_negvol(tmp_path):
    deck = """# OpenRadioss Engine Deck
/RUN/TEST/1
10.0
/DT1TET10
3
/DTTSH
/REPORT
500
/REPORT/DT
0.05
/NEGVOL/DEL
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0
    assert ec.dt1tet10 == 3
    assert ec.dttsh is True
    assert ec.report_freq == 500
    assert ec.report_dt == pytest.approx(0.05)
    assert ec.negvol_action == "DEL"
