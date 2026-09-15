"""Tests for Milestone M203:
- Kinematic Gear/Rack/Diff Constraints: /LAGMUL/GEAR (/GEAR), /LAGMUL/RACK (/RACK), /LAGMUL/DIFF (/DIFF)
- Guided Cable Interface: /INTER/TYPE26 (/INTER/GUIDED_CABLE, /TYPE26)
- Python Scripted Sensor: /SENSOR/PYTHON
- Failure Criteria Suite: /FAIL/JOHNSON, /FAIL/BIQUAD, /FAIL/FLD, /FAIL/CONNECT, /FAIL/FRACTAL_DMG, /FAIL/ORTHENERG
- Adaptive Meshing Set: /ADMESH/SET
- SPH Numerical Gauge: /GAUGE/SPH
- Checksum Calculation Directives: /CHECKSUM/START, /CHECKSUM/END
- 6-Point Spatial Alignment: /TRANSFORM/POS (/POS, /POSITION)
- Gas Injector Properties: /PROP/INJECT1, /PROP/INJECT2, /INJECT1, /INJECT2
- Box & Model Entity Aliases
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailJohnson, FailBiquad, FailFld, FailConnect, FailFractalDmg, FailOrthenerg,
    AdmeshSet, GaugeSph, ChecksumDirective, TransformPosition,
    GearConstraint, RackConstraint, DiffConstraint, GuidedCable, SensorPython,
    Box, BoxRect, BoxCyl, BoxSphere
)


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m203_failure_johnson_and_biquad(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Johnson and Biquad Failure Test
1 1
/FAIL/JOHNSON/101
                 0.1                 0.2                 0.3                 0.4                 0.5
                 1.0                   1                   1                 0.0                 0.0                   0                   0
/FAIL/BIQUAD/102
                 0.1                 0.2                 0.3                 0.4                 0.5
                 0.0                   0                   2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.fails_johnson
    fj = model.fails_johnson[101]
    assert fj.mat_id == 101
    assert pytest.approx(fj.d1) == 0.1
    assert pytest.approx(fj.d2) == 0.2
    assert pytest.approx(fj.d3) == 0.3
    assert pytest.approx(fj.d4) == 0.4
    assert pytest.approx(fj.d5) == 0.5
    assert pytest.approx(fj.eps_dot_0) == 1.0
    assert fj.ifail_sh == 1

    assert 102 in model.fails_biquad
    fb = model.fails_biquad[102]
    assert fb.mat_id == 102
    assert pytest.approx(fb.c1) == 0.1
    assert pytest.approx(fb.c2) == 0.2
    assert pytest.approx(fb.c3) == 0.3
    assert pytest.approx(fb.c4) == 0.4
    assert pytest.approx(fb.c5) == 0.5
    assert fb.s_flag == 2


def test_m203_failure_fld_and_connect(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
FLD and Connect Failure Test
1 1
/FAIL/FLD/201
        10         1         2         0                 0.0                 0.0         2         0
                 0.8                 0.5
                 0.2                 1.5
        999
/FAIL/CONNECT/202
                 0.1                 1.5                 1.0                   1                   1                   1                   0
                 0.2                 1.2                 1.0                   2
                 0.0                 0.0                 0.0                 0.0                 0.0
                 0.0                 0.0                 1.0
        888
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.fails_fld
    fld = model.fails_fld[201]
    assert fld.mat_id == 201
    assert fld.fct_id == 10
    assert fld.i_marg == 2
    assert pytest.approx(fld.factor_marginal) == 0.8
    assert pytest.approx(fld.factor_loosemetal) == 0.5
    assert fld.istrain == 2
    assert pytest.approx(fld.fcut) == 0.2
    assert pytest.approx(fld.alpha) == 1.5
    assert fld.fail_id == 999

    assert 202 in model.fails_connect
    fc = model.fails_connect[202]
    assert fc.mat_id == 202
    assert pytest.approx(fc.epsilon_maxn) == 0.1
    assert pytest.approx(fc.exponent_n) == 1.5
    assert pytest.approx(fc.alpha_n) == 1.0
    assert fc.r_fct_id_n == 1
    assert pytest.approx(fc.epsilon_maxt) == 0.2
    assert pytest.approx(fc.exponent_t) == 1.2
    assert pytest.approx(fc.alpha_t) == 1.0
    assert fc.r_fct_id_t == 2
    assert pytest.approx(fc.area_scale) == 1.0
    assert fc.fail_id == 888


def test_m203_failure_fractal_and_orthenerg(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Fractal and Orthenerg Failure Test
1 1
/FAIL/FRACTAL_DMG/301
         1         2         3         4
                 0.5                 0.8        1234          10           1
         777
/FAIL/ORTHENERG/302
                 0.1                                       1         1
                 100                 1.0                   0                 120                 1.2                   0
                 200                 2.0                   0                 220                 2.2                   0
                 300                 3.0                   0                 320                 3.2                   0
                  40                 0.4                   0                  45                 0.5                   0
                  50                 0.5                   0                  55                 0.6                   0
                  60                 0.6                   0                  65                 0.7                   0
         666
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.fails_fractal_dmg
    ffd = model.fails_fractal_dmg[301]
    assert ffd.mat_id == 301
    assert ffd.grsh4n_1 == 1
    assert ffd.grsh3n_1 == 2
    assert ffd.grsh4n_2 == 3
    assert ffd.grsh3n_2 == 4
    assert pytest.approx(ffd.damage) == 0.5
    assert pytest.approx(ffd.probability) == 0.8
    assert ffd.seed == 1234
    assert ffd.num_walk == 10
    assert ffd.printout == 1
    assert ffd.fail_id == 777

    assert 302 in model.fails_orthenerg
    fo = model.fails_orthenerg[302]
    assert fo.mat_id == 302
    assert pytest.approx(fo.pthickfail) == 0.1
    assert fo.nmod == 1
    assert fo.failip == 1
    assert pytest.approx(fo.sigma_11t) == 100.0
    assert pytest.approx(fo.g_11t) == 1.0
    assert pytest.approx(fo.sigma_11c) == 120.0
    assert pytest.approx(fo.g_11c) == 1.2
    assert pytest.approx(fo.sigma_22t) == 200.0
    assert pytest.approx(fo.g_22t) == 2.0
    assert pytest.approx(fo.sigma_33t) == 300.0
    assert pytest.approx(fo.sigma_12t) == 40.0
    assert pytest.approx(fo.sigma_23t) == 50.0
    assert pytest.approx(fo.sigma_31t) == 60.0
    assert fo.fail_id == 666


def test_m203_admesh_set_and_gauge_sph(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Admesh Set and Gauge SPH Test
1 1
/ADMESH/SET/401
Admesh Set 1
                45.0         2                0.05
         101         3                 0.1
/GAUGE/SPH/501
SPH Gauge 1
         100                10.0                 200                0.01
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 401 in model.admesh_sets
    adm = model.admesh_sets[401]
    assert adm.id == 401
    assert pytest.approx(adm.angle_criteria) == 45.0
    assert adm.inilev == 2
    assert pytest.approx(adm.thkerr) == 0.05
    assert adm.grnd_id == 101
    assert adm.level == 3
    assert pytest.approx(adm.tdelay) == 0.1

    assert 501 in model.gauge_sphs
    gs = model.gauge_sphs[501]
    assert gs.id == 501
    assert gs.node_id == 100
    assert pytest.approx(gs.fcut) == 10.0
    assert gs.shell_id == 200
    assert pytest.approx(gs.dist) == 0.01


def test_m203_checksums_and_transform_pos(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Checksum and Transform POS Test
1 1
/CHECKSUM/START/1
         100         200
/CHECKSUM/END/2
         300         400
/TRANSFORM/POS/601
Transform Position 1
         101         1         2         3         4         5         6                             0
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
                 0.0                 0.0                 1.0
                 1.0                 1.0                 0.0
                 1.0                 1.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(model.checksums) == 2
    cs1 = model.checksums[0]
    assert cs1.id == 1
    assert cs1.action == "START"
    assert cs1.val1 == 100
    assert cs1.val2 == 200

    cs2 = model.checksums[1]
    assert cs2.id == 2
    assert cs2.action == "END"
    assert cs2.val1 == 300
    assert cs2.val2 == 400

    assert 601 in model.transform_positions
    tp = model.transform_positions[601]
    assert tp.id == 601
    assert tp.grnod_id == 101
    assert tp.node_ids == (1, 2, 3, 4, 5, 6)
    assert tp.points[1] == (1.0, 0.0, 0.0)
    assert tp.points[2] == (0.0, 1.0, 0.0)
    assert model.transforms_pos is model.transform_positions


def test_m203_kinematic_constraints_and_guided_cable(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Lagmul Gear Rack Diff and Type26 Test
1 1
/LAGMUL/GEAR/701
Gear Constraint 1
           1         2                 2.5         1         1         0         0
/RACK/702
Rack Constraint 1
           3         4                10.0         1         2         0         0
/DIFF/703
Diff Constraint 1
           5         6         7                 0.5
/INTER/TYPE26/801
Guided Cable Interface 1
           1         2         3                 0.1                 0.2
/SENSOR/PYTHON/901
Python Sensor 1
                 0.5
def check_sensor(time, state):
    return time > 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 701 in model.gears
    g = model.gears[701]
    assert g.node1 == 1
    assert g.node2 == 2
    assert pytest.approx(g.ratio) == 2.5

    assert 702 in model.racks
    r = model.racks[702]
    assert r.node1 == 3
    assert r.node2 == 4
    assert pytest.approx(r.pitch_radius) == 10.0

    assert 703 in model.diffs
    d = model.diffs[703]
    assert d.node0 == 5
    assert d.node1 == 6
    assert d.node2 == 7
    assert pytest.approx(d.ratio) == 0.5

    assert 801 in model.guided_cables
    gc = model.guided_cables[801]
    assert gc.id == 801
    assert gc.grnod_main == 1
    assert gc.grnod_sub == 2
    assert pytest.approx(gc.frad) == 0.1
    assert pytest.approx(gc.fric) == 0.2

    assert 901 in model.sensors_python
    sp = model.sensors_python[901]
    assert sp.id == 901
    assert pytest.approx(sp.t_delay) == 0.5
    assert "def check_sensor" in sp.code


def test_m203_box_and_model_aliases():
    m = Model()
    assert m.boxes_cyl is m.boxes
    assert m.boxes_rect is m.boxes
    assert m.boxes_sphere is m.boxes
    assert m.pressure_loads is m.load_pressures
    assert m.fails_johnson is m.fail_johnsons
    assert m.fails_biquad is m.fail_biquads
    assert m.fails_fld is m.fail_flds
    assert m.fails_connect is m.fail_connects
    assert m.fails_fractal_dmg is m.fail_fractal_dmgs
    assert m.fails_orthenerg is m.fail_orthenergs
    assert m.transforms_pos is m.transform_positions
    assert m.pos_transforms is m.transform_positions

    assert BoxRect is Box
    assert BoxCyl is Box
    assert BoxSphere is Box
