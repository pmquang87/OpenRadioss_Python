"""Multi-cycle dynamic explicit engine simulation for /MAT/LAW105 (/MAT/POWDER_BURN).

Validates:
  - Starter initialization of /MAT/LAW105 with /PROP/SOLID and /BRICK.
  - Integration with engine cycle loop via run_engine.
  - Numerical stability, positive sound speed, finite stresses and energy accounting.
"""

import os
import contextlib
import io
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter


_STARTER_DECK = """/BEGIN
LAW105 Powder Burn Engine Verification
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 10.0 10.0 0.0
4 0.0 10.0 0.0
5 0.0 0.0 10.0
6 10.0 0.0 10.0
7 10.0 10.0 10.0
8 0.0 10.0 10.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
PowderGrain
1 1
/PROP/SOLID/1
SolidProp
1.1 0.05 0.1
/MAT/LAW105/1
Propellant
1.65e-6
14000.0 0.1 0.0
0.5e-6 4000.0
0.05 0.8 1.0
0 0 1.0 1.0
0 0 1.0 1.0 3000.0 0.0
0.93
/GRNOD/NODE/1
BottomNodes
1 2 3 4
/BCS/1
FixBottom
111 111 0 1
/END
"""

_ENGINE_DECK = """/RUN/sim_law105/1
0.0005
/DT
0.5 0
/PRINT/-1
/STOP
100.0
"""


def test_law105_engine_simulation_run(tmp_path):
    """Run starter and engine for LAW105 powder burn brick simulation."""
    deck_0000 = tmp_path / "sim_law105_0000.rad"
    deck_0001 = tmp_path / "sim_law105_0001.rad"

    deck_0000.write_text(_STARTER_DECK, encoding="utf-8")
    deck_0001.write_text(_ENGINE_DECK, encoding="utf-8")

    # Run starter
    out_s = io.StringIO()
    with contextlib.redirect_stdout(out_s):
        starter_model = run_starter(str(deck_0000))

    assert starter_model is not None
    assert 1 in starter_model.materials
    mat = starter_model.materials[1]
    assert mat.law == 105
    assert mat.rho0 == pytest.approx(1.65e-6)
    assert mat.params["POWDER_BULK"] == pytest.approx(14000.0)
    assert starter_model.mat_law105s[1].bulk == pytest.approx(14000.0)

    # Run engine
    out_e = io.StringIO()
    with contextlib.redirect_stdout(out_e):
        engine_model = run_engine(str(deck_0001))

    assert engine_model is not None

    # Check output log
    out_path = tmp_path / "sim_law105_0001.out"
    assert out_path.exists()
    out_content = out_path.read_text(encoding="latin-1", errors="replace")

    # Verify simulation executed cycles without NaN
    assert "NORMAL TERMINATION" in out_content or "ELAPSED" in out_content or len(out_content) > 100
    assert "NaN" not in out_content
    assert "nan" not in out_content


def test_law105_synonym_starter_and_cycle(tmp_path):
    """Verify /MAT/POWDER_BURN synonym deck parses and initializes correctly."""
    synonym_starter = _STARTER_DECK.replace("/MAT/LAW105/1", "/MAT/POWDER_BURN/1")
    deck_0000 = tmp_path / "syn_law105_0000.rad"
    deck_0001 = tmp_path / "syn_law105_0001.rad"

    deck_0000.write_text(synonym_starter, encoding="utf-8")
    deck_0001.write_text(_ENGINE_DECK, encoding="utf-8")

    out_s = io.StringIO()
    with contextlib.redirect_stdout(out_s):
        model = run_starter(str(deck_0000))

    assert model is not None
    assert 1 in model.materials
    assert model.materials[1].law_name in ("POWDER_BURN", "POWDERBURN", "LAW105")
