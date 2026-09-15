"""Tests for Milestone M117: Modal Extraction (/EIG), Shell Frame Formulation (/SHFRA/V4),
Integration Thickness Flag (/INTTHICK/V5), State Stress File (/STATE/STR_FILE, /STR_FILE),
Memory Allocation Request (/MEMORY), Contact Default Type 25 (/DEF_INTER/TYPE25),
and Pressure Load Alias (/PLOAD).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.starter import StarterError


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M117
      2022         0
/MAT/LAW1/1
LinearElastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell_1
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/10
NodeGroup10
         1         2
/GRNOD/NODE/20
NodeGroup20
         3         4
/SURF/PART/1
SurfPart1
         1
/FUNCT/1
PressCurve
                 0.0                 0.0
                 1.0                 1.0
"""


def _run(tmp_path: Path, deck: str, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    try:
        model = run_starter(str(p), log)
    except StarterError:
        model = None
    return model, log


def test_m117_eig_fixed_and_free(tmp_path):
    deck = _BOILERPLATE + """\
/EIG/1
ModalExtraction1
        10        20   111 000         0
        20         1                50.0                 0.1
         5        10       100         1                1e-6
/EIG/2
ModalExtraction2
        10        20   111000         1
        15         0                40.0                 0.0
         4         8        50         0                1e-5
eigen_extra_modes.dat
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.eigen_modes
    e1 = model.eigen_modes[1]
    assert e1.id == 1
    assert e1.title == "ModalExtraction1"
    assert e1.grnod_id == 10
    assert e1.grnod_bc == 20
    assert e1.nmod == 20
    assert e1.cutfreq == 50.0
    assert e1.freqmin == 0.1
    assert e1.nbloc == 5
    assert e1.tol == 1e-6

    assert 2 in model.eigen_modes
    e2 = model.eigen_modes[2]
    assert e2.id == 2
    assert e2.title == "ModalExtraction2"
    assert e2.ifile == 1
    assert e2.filename == "eigen_extra_modes.dat"


def test_m117_shfra_and_intthick(tmp_path):
    deck = _BOILERPLATE + """\
/SHFRA/V4
/INTTHICK/V5
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.shfra_v4 is True
    assert model.intthick_v5 is True


def test_m117_state_str_file(tmp_path):
    deck = _BOILERPLATE + """\
/STATE/STR_FILE
         1
stress_output.str.gz
/STR_FILE
         0
stress_plain.str
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.stress_files) == 2
    assert model.stress_files[0].izip == 1
    assert model.stress_files[0].filename == "stress_output.str.gz"
    assert model.stress_files[1].izip == 0
    assert model.stress_files[1].filename == "stress_plain.str"


def test_m117_memory(tmp_path):
    deck = _BOILERPLATE + """\
/MEMORY
      5000                   0.75
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.memory_requests) == 1
    assert model.memory_requests[0].nmots == 5000
    assert model.memory_requests[0].rate == 0.75


def test_m117_def_inter_type25(tmp_path):
    deck = _BOILERPLATE + """\
/DEF_INTER/TYPE25
         2         1         1      1000      1000         1      1000      1000
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "TYPE25" in model.def_inter
    d25 = model.def_inter["TYPE25"]
    assert d25["istf"] == 2
    assert d25["igap"] == 1


def test_m117_pload(tmp_path):
    deck = _BOILERPLATE + """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PLOAD/1
PressureOnSurface1
         1         1         0                                       1.0                 2.5
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.ploads) == 1
    pl = model.ploads[0]
    assert pl.id == 1
    assert pl.surf_id == 1
    assert pl.funct_id == 1
    assert pl.scale == 2.5


def test_m117_cross_reference_errors(tmp_path):
    deck = _BOILERPLATE + """\
/EIG/99
FaultyEigen
       999       888   111 000         0
        10         0                10.0                 0.0
         5         5        50         0                1e-6
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/EIG/99: node group 999 not defined" in err_text
    assert "/EIG/99: node group 888 not defined" in err_text
