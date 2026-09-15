"""M96 – Initial Element State Suite: /INIBRI and /INISHE (stress, strain, thickness)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import InitialBrickState, InitialShellState
from pyradioss.starter.starter import run_starter


# ──────────────────── reusable boilerplate ────────────────────────────
_ELEM = """\
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PROP/TYPE14/2
Solid_Prop
         1         1         0         0         0         0         0
/PART/1
Part_Shell
         1         1
/PART/2
Part_Solid
         2         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
       201                 0.0                 0.0                 0.0
       202                10.0                 0.0                 0.0
       203                10.0                10.0                 0.0
       204                 0.0                10.0                 0.0
       205                 0.0                 0.0                10.0
       206                10.0                 0.0                10.0
       207                10.0                10.0                10.0
       208                 0.0                10.0                10.0
/SHELL/1
         1       101       102       103       104
/BRICK/2
         2       201       202       203       204       205       206       207       208
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /INIBRI tests (Solid element initial state)
# ══════════════════════════════════════════════════════════════════════

class TestInibri:
    """/INIBRI initial stress, plastic strain, and density."""

    def test_inibri_stress_fixed(self, tmp_path):
        """Parse fixed-format /INIBRI/STRESS for solid elements."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIBRI_STRESS_FIXED
      2021         0
{_ELEM}
/INIBRI/STRESS
# bric_IDst              SIGMA_x             SIGMA_y             SIGMA_z
         2               100.0                50.0               -25.0
#              SIGMA_xy            SIGMA_yz            SIGMA_xz
               12.5                 0.0                -5.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.ini_bricks
        st = model.ini_bricks[2]
        assert isinstance(st, InitialBrickState)
        assert st.elem_id == 2
        assert np.allclose(st.sigma, [100.0, 50.0, -25.0, 12.5, 0.0, -5.5])

    def test_inibri_stress_free(self, tmp_path):
        """Parse free-format /INIBRI/STRESS."""
        deck = f"""\
/BEGIN
TEST_INIBRI_STRESS_FREE
{_ELEM}
/INIBRI/STRESS
2 200.0 150.0 50.0 20.0 10.0 5.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.ini_bricks
        st = model.ini_bricks[2]
        assert np.allclose(st.sigma, [200.0, 150.0, 50.0, 20.0, 10.0, 5.0])

    def test_inibri_epsp_and_dens(self, tmp_path):
        """Parse /INIBRI/EPSP and /INIBRI/DENS."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIBRI_SCALARS
      2021         0
{_ELEM}
/INIBRI/EPSP
         2               0.045
/INIBRI/DENS
         2             7.85e-9
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.ini_bricks
        st = model.ini_bricks[2]
        assert abs(st.epsp - 0.045) < 1e-12
        assert abs(st.rho - 7.85e-9) < 1e-15


# ══════════════════════════════════════════════════════════════════════
#  /INISHE tests (Shell element initial state)
# ══════════════════════════════════════════════════════════════════════

class TestInishe:
    """/INISHE initial stress, plastic strain, and thickness."""

    def test_inishe_epsp_and_thick(self, tmp_path):
        """Parse /INISHE/EPSP and /INISHE/THICK."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INISHE_SCALARS
      2021         0
{_ELEM}
/INISHE/EPSP
         1                0.12
/INISHE/THICK
         1                 1.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.ini_shells
        sh = model.ini_shells[1]
        assert isinstance(sh, InitialShellState)
        assert sh.elem_id == 1
        assert abs(sh.epsp - 0.12) < 1e-12
        assert abs(sh.thick - 1.5) < 1e-12

    def test_inishe_strs_f_fixed(self, tmp_path):
        """Parse fixed-format /INISHE/STRS_F membrane & bending stress."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INISHE_STRS_FIXED
      2021         0
{_ELEM}
/INISHE/STRS_F
# shell_ID nb_integr       npg               Thick
         1         0         1                 1.2
#                   Em                  Eb                  H1                  H2                  H3
                10.0                 5.0                 0.1                 0.2                 0.0
#              sigma_1             sigma_2            sigma_12            sigma_23            sigma_31
               120.0                80.0                15.0                 0.0                 0.0
#                eps_p            sigma_b1            sigma_b2           sigma_b12
                0.05                40.0                20.0                 5.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.ini_shells
        sh = model.ini_shells[1]
        assert abs(sh.thick - 1.2) < 1e-12
        assert abs(sh.em - 10.0) < 1e-12
        assert abs(sh.eb - 5.0) < 1e-12
        assert np.allclose(sh.h_energy, [0.1, 0.2, 0.0])
        assert np.allclose([sh.sigma[0], sh.sigma[1], sh.sigma[3]], [120.0, 80.0, 15.0])
        assert abs(sh.epsp - 0.05) < 1e-12
        assert np.allclose([sh.sigma_b[0], sh.sigma_b[1], sh.sigma_b[3]], [40.0, 20.0, 5.0])

    def test_inish3_support(self, tmp_path):
        """Parse /INISH3 initial state for triangle shells."""
        deck = f"""\
/BEGIN
TEST_INISH3_FREE
{_ELEM}
/SH3N/1
10 101 102 103
/INISH3/EPSP
10 0.08
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.ini_shells
        assert abs(model.ini_shells[10].epsp - 0.08) < 1e-12
