"""
Milestone M86 — /FAIL/FLD (Forming Limit Diagram Failure Model).

Tests:
1. Parser & layout for /FAIL/FLD (fixed and free format)
2. Material resolution and /FUNCT binding
3. In-plane principal strain calculation and limit curve evaluation (true & engineering strain)
4. Shell layer damage accumulation and element deletion under different Ifail_sh modes
5. End-to-end explicit simulation with /FAIL/FLD on a tension shell model running through Engine cycles
"""

import io
import contextlib
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.engine.engine import run_engine
from pyradioss.failure import fld
from pyradioss.model.entities import FailureModel
from pyradioss.starter.starter import run_starter


_MINIMAL_ELEM_DECK = """/PROP/TYPE1/1
Shell_Prop
1 1 1 0 0 0 0
1.0 1.0 1.0 0 0 0 0 0
1.0 0.833333
/PART/1
Plate
1 1
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 10.0 10.0 0.0
4 0.0 10.0 0.0
/SHELL/1
1 1 2 3 4
"""


def test_fld_parsing_fixed_and_free(tmp_path):
    # Free format: /FAIL/FLD/mat_id
    deck_free = f"""/BEGIN
TEST_FREE
/FUNCT/100
FLD Curve
-0.2 0.4
0.0  0.2
0.2  0.35
/MAT/PLAS_JOHNS/1
Steel
7.8e-9
210000.0 0.3
200.0 400.0 0.5 0.0 0.0
/FAIL/FLD/1
100 1 0 0 0.0 0.0 0 0
{_MINIMAL_ELEM_DECK}
/END
"""
    p1 = tmp_path / "TEST_FREE_0000.rad"
    p1.write_text(deck_free)
    log1 = MessageLog()
    m1 = run_starter(str(p1), log1)
    assert len(log1.errors) == 0
    assert 1 in m1.materials
    mat1 = m1.materials[1]
    assert mat1.fail is not None
    assert mat1.fail.type == "FLD"
    assert mat1.fail.params["fct_id"] == 100
    assert mat1.fail.ifail_sh == 1
    assert "function" in mat1.fail.params
    assert mat1.fail.params["function"] is not None
    assert len(mat1.fail.params["function"].x) == 3

    # Fixed format: /FAIL/FLD/mat_id
    # Format: "%10d%10d          %10d%20lg%20lg%10d%10d", fct_ID, Ifail_sh, fct_IDadv, Rani, Dadv, Istrain, Ixfem
    deck_fixed = f"""# RADIOSS STARTER
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_FIXED
      2021         0
/FUNCT/200
FLD Curve Fixed
                -0.1                 0.3
                 0.0                0.15
                 0.1                0.25
/MAT/PLAS_JOHNS/1
Alu
              2.7e-9
             70000.0                0.33
               150.0               250.0                 0.4                 0.0                 0.0
/FAIL/FLD/1
       200         2                   0                 1.5                 0.1         1         0
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Plate
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                10.0                 0.0                 0.0
         3                10.0                10.0                 0.0
         4                 0.0                10.0                 0.0
/SHELL/1
         1         1         2         3         4
/END
"""
    p2 = tmp_path / "TEST_FIXED_0000.rad"
    p2.write_text(deck_fixed)
    log2 = MessageLog()
    m2 = run_starter(str(p2), log2)
    assert len(log2.errors) == 0
    assert 1 in m2.materials
    mat2 = m2.materials[1]
    assert mat2.fail is not None
    assert mat2.fail.type == "FLD"
    assert mat2.fail.params["fct_id"] == 200
    assert mat2.fail.ifail_sh == 2
    assert mat2.fail.params["istrain"] == 1
    assert mat2.fail.params["rani"] == pytest.approx(1.5)
    assert mat2.fail.params["dadv"] == pytest.approx(0.1)
    assert mat2.fail.params["function"] is not None


def test_fld_principal_strains_and_curve_eval():
    # Define an FLD curve: (emin, emaj)
    curve = FunctTable(fct_id=1, title="FLD", x=np.array([-0.2, 0.0, 0.2]), y=np.array([0.4, 0.2, 0.35]))
    fm_true = FailureModel(type="FLD", ifail_sh=1, params={"fct_id": 1, "function": curve, "istrain": 0})

    # Case 1: Uniaxial tension along x: eps = [0.2, -0.06, 0.0]
    # S1 = 0.07, S2 = 0.13, Q = 0.13 => emaj = 0.2, emin = -0.06
    # Curve at emin=-0.06: linear interp between (-0.2, 0.4) and (0.0, 0.2):
    # slope = (0.2 - 0.4) / (0.0 - -0.2) = -1.0 => EM = 0.2 - (-1.0)*0.06 = 0.26
    # Damage = emaj / EM = 0.2 / 0.26 = 0.76923
    eps = np.array([[0.2, -0.06, 0.0]])
    dama = np.zeros(1)
    sig = np.zeros((1, 3))
    broken = fld.shell_step(fm_true, sig, 0.0, eps[0], dt=1e-6, dama=dama, eps_tot=eps)
    assert not broken[0]
    assert dama[0] == pytest.approx(0.2 / 0.26, rel=1e-3)

    # Now increase strain past limit: eps_xx = 0.3 => emaj = 0.3 => Damage = 0.3 / 0.26 = 1.1538 >= 1.0 -> broken
    eps_over = np.array([[0.3, -0.06, 0.0]])
    broken = fld.shell_step(fm_true, sig, 0.0, eps_over[0], dt=1e-6, dama=dama, eps_tot=eps_over)
    assert broken[0]
    assert dama[0] > 1.0


def test_fld_engineering_strain_conversion():
    # Test istrain=1 (engineering strain input)
    # If curve has engineering strains: (-0.1, 0.2) and (0.0, 0.1)
    curve = FunctTable(fct_id=2, title="Eng FLD", x=np.array([-0.1, 0.0, 0.1]), y=np.array([0.2, 0.1, 0.15]))
    fm_eng = FailureModel(type="FLD", ifail_sh=1, params={"fct_id": 2, "function": curve, "istrain": 1})

    eps = np.array([[0.15, -0.05, 0.0]])
    dama = np.zeros(1)
    sig = np.zeros((1, 3))
    broken = fld.shell_step(fm_eng, sig, 0.0, eps[0], dt=1e-6, dama=dama, eps_tot=eps)
    assert dama[0] > 0.0


def test_fld_ifail_sh_modes():
    curve = FunctTable(fct_id=1, title="FLD", x=np.array([-0.2, 0.0, 0.2]), y=np.array([0.4, 0.2, 0.35]))

    # Ifail_sh = 4: no deletion even if damage >= 1.0
    fm_no_del = FailureModel(type="FLD", ifail_sh=4, params={"fct_id": 1, "function": curve, "istrain": 0, "ifail_sh": 4})
    eps = np.array([[0.5, 0.0, 0.0]])  # emaj = 0.5 >> 0.2
    dama = np.zeros(1)
    sig = np.zeros((1, 3))
    broken = fld.shell_step(fm_no_del, sig, 0.0, eps[0], dt=1e-6, dama=dama, eps_tot=eps)
    assert not broken[0]  # Ifail_sh=4 never reports broken
    assert dama[0] >= 1.0


def test_fld_end_to_end_shell_simulation(tmp_path):
    # End-to-end explicit deck with 1 shell element in tension with /FAIL/FLD
    starter_deck = f"""/BEGIN
TENSION_SHELL_FLD
/FUNCT/1
FLD_Curve
-0.5 0.5
0.0  0.05
0.5  0.5
/MAT/PLAS_JOHNS/1
Alu
2.7e-6
70000.0 0.33
100.0 200.0 0.5 0.0 0.0
/FAIL/FLD/1
1 1 0 0 0.0 0.0 0 0
{_MINIMAL_ELEM_DECK}
/GRNOD/NODE/1
Left_Nodes
1 4
/GRNOD/NODE/2
Right_Nodes
2 3
/BCS/1
Fix_Left
111 111 0 1
/INIVEL/TRA/1
Init_Vel
100.0 0.0 0.0 2
/END
"""
    engine_deck = """/RUN/TENSION_SHELL_FLD/1
0.1
/TFILE
0.01
/DT/NODA/CST
0.9 1e-6
/END
"""
    s = tmp_path / "TENSION_0000.rad"
    e = tmp_path / "TENSION_0001.rad"
    s.write_text(starter_deck)
    e.write_text(engine_deck)

    log = MessageLog()
    m_starter = run_starter(str(s), log)
    assert len(log.errors) == 0
    assert m_starter.materials[1].fail is not None
    assert m_starter.materials[1].fail.type == "FLD"

    with contextlib.redirect_stdout(io.StringIO()):
        m_engine = run_engine(str(e))
    assert m_engine is not None
    # Shell group state has dama allocated
    assert "dama" in m_engine.shells.state
    assert m_engine.shells.state["dama"].shape[0] == 1
