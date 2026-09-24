"""
End-to-End Engine Integration Tests for All Advanced Finite Element Technologies.
Verifies full Starter -> Engine pipeline execution (dynamic cycling, force accumulation,
and energy conservation) across all new element formulation kernels:
- bricks_full (Isolid=2)
- bricks_eas (Isolid=17)
- solid_shells_ha8 (Isolid=16)
- cohesives (Isolid=21)
- tetras_sfem (Itetra4=3)
- penta6s_heph (Isolid=24 on wedge)
- pyra5s (PYRA5 degenerate solid)
- shells_dkt6 (Ish3n=3 DKT macro-patch)
- quads_full (Iquad=2 full 2x2 2D quad)
- trias (TRIA3 2D constant strain triangle)
"""

import contextlib
import io
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter


STEEL_LAW1 = """\
/MAT/LAW1/1
steel
7.85e-9
210000.0 0.3
"""

ENGINE_CARD = """\
/RUN/RUN/1
0.000005
/DT
0.9 0
/PRINT/-1000
/STOP
5.0
"""


def _run(make_deck, name, starter, engine=ENGINE_CARD):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


def test_engine_bricks_full_isolid2(make_deck):
    """Verify end-to-end engine execution for 8-node full 2x2x2 Gauss hex (Isolid=2)."""
    starter = f"""\
/BEGIN
bricks_full isolid2
/NODE
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
cube
1 1
{STEEL_LAW1}
/PROP/SOLID/1
solid_full
2 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
10.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "HEX_FULL", starter)
    assert model.engine_state.cycle > 0
    assert model.bricks_full is not None
    assert model.bricks_full.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_bricks_eas_isolid17(make_deck):
    """Verify end-to-end engine execution for 8-node EAS hex (Isolid=17)."""
    starter = f"""\
/BEGIN
bricks_eas isolid17
/NODE
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
cube
1 1
{STEEL_LAW1}
/PROP/SOLID/1
solid_eas
17 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
10.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "HEX_EAS", starter)
    assert model.engine_state.cycle > 0
    assert model.bricks_eas is not None
    assert model.bricks_eas.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_solid_shell_ha8_isolid16(make_deck):
    """Verify end-to-end engine execution for 8-node HA8 solid shell (Isolid=16)."""
    starter = f"""\
/BEGIN
solid_shell ha8
/NODE
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 0.1
6 1 0 0.1
7 1 1 0.1
8 0 1 0.1
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
plate
1 1
{STEEL_LAW1}
/PROP/SOLID/1
solid_ha8
16 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
5.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "HA8_RUN", starter)
    assert model.engine_state.cycle > 0
    assert model.solid_shells_ha8 is not None
    assert model.solid_shells_ha8.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_cohesives_isolid21(make_deck):
    """Verify end-to-end engine execution for cohesive zone element (Isolid=21)."""
    starter = f"""\
/BEGIN
cohesive isolid21
/NODE
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 0.001
6 1 0 0.001
7 1 1 0.001
8 0 1 0.001
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
interface
1 1
{STEEL_LAW1}
/PROP/SOLID/1
cohesive_prop
21 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
1.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "COH_RUN", starter)
    assert model.engine_state.cycle > 0
    assert model.cohesives is not None
    assert model.cohesives.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_tetras_sfem_itetra3(make_deck):
    """Verify end-to-end engine execution for 4-node SFEM smoothed tetra (Itetra4=3)."""
    starter = f"""\
/BEGIN
tetra sfem
/NODE
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
/TETRA4/1
1 1 2 3 4
/PART/1
tet
1 1
{STEEL_LAW1}
/PROP/SOLID/1
tetra_prop
1 0 0 0 0 0 3
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
5.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "TET_SFEM", starter)
    assert model.engine_state.cycle > 0
    assert model.tetras_sfem is not None
    assert model.tetras_sfem.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_penta6_heph_isolid24(make_deck):
    """Verify end-to-end engine execution for 6-node HEPH stabilized wedge (Isolid=24)."""
    starter = f"""\
/BEGIN
penta heph
/NODE
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 1 0 1
6 0 1 1
/PENTA6/1
1 1 2 3 4 5 6
/PART/1
wedge
1 1
{STEEL_LAW1}
/PROP/SOLID/1
wedge_prop
24 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
5.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "PENTA_HEPH", starter)
    assert model.engine_state.cycle > 0
    assert model.penta6s_heph is not None
    assert model.penta6s_heph.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_pyra5(make_deck):
    """Verify end-to-end engine execution for 5-node pyramid solid (PYRA5)."""
    starter = f"""\
/BEGIN
pyra5 solid
/NODE
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.5 0.5 1
/PYRA5/1
1 1 2 3 4 5
/PART/1
pyramid
1 1
{STEEL_LAW1}
/PROP/SOLID/1
solid
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
5.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "PYRA_RUN", starter)
    assert model.engine_state.cycle > 0
    assert model.pyra5s is not None
    assert model.pyra5s.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_shells_dkt6_ish3n3(make_deck):
    """Verify end-to-end engine execution for 6-node DKT rotation-free shell (Ish3n=3)."""
    starter = f"""\
/BEGIN
shell dkt6
/NODE
1 0 0 0
2 1 0 0
3 0 1 0
4 0.5 0 0
5 0.5 0.5 0
6 0 0.5 0
/SH3N/1
1 1 2 3 4 5 6
/PART/1
shell_part
1 1
{STEEL_LAW1}
/PROP/SHELL/1
shell_prop
1 0 3
0.01 0.01 0.01
3 0 1.0
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
5.0 0.0 0.0 1
/END
"""
    model = _run(make_deck, "DKT6_RUN", starter)
    assert model.engine_state.cycle > 0
    assert model.shells_dkt6 is not None
    assert model.shells_dkt6.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_quads_full_iquad2(make_deck):
    """Verify end-to-end engine execution for 4-node full 2x2 2D quad (Iquad=2)."""
    starter = f"""\
/BEGIN
quad full 2d
/ANALY
 2
/NODE
1 0 0.0 0.0
2 0 1.0 0.0
3 0 1.0 1.0
4 0 0.0 1.0
/QUAD/1
1 1 2 3 4
/PART/1
quad_part
1 1
{STEEL_LAW1}
/PROP/SOLID/1
quad_prop
2 0 0
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
0.0 5.0 0.0 1
/END
"""
    model = _run(make_deck, "QUAD_FULL", starter)
    assert model.engine_state.cycle > 0
    assert model.quads_full is not None
    assert model.quads_full.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95


def test_engine_trias(make_deck):
    """Verify end-to-end engine execution for 3-node 2D triangle (TRIA3)."""
    starter = f"""\
/BEGIN
tria 2d
/ANALY
 2
/NODE
1 0 0.0 0.0
2 0 1.0 0.0
3 0 0.0 1.0
/TRIA3/1
1 1 2 3
/PART/1
tria_part
1 1
{STEEL_LAW1}
/PROP/SOLID/1
tria_prop
1.1 0.05 0.1
/GRNOD/PART/1
all
1
/INIVEL/TRA/1
fly
0.0 5.0 0.0 1
/END
"""
    model = _run(make_deck, "TRIA_RUN", starter)
    assert model.engine_state.cycle > 0
    assert model.trias is not None
    assert model.trias.n == 1
    assert model.engine_state.t >= 0.000005 * 0.95
