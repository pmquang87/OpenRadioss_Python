"""Tests for Milestone M118: Fractal Damage Failure (/FAIL/FRACTAL_DMG, /FAIL/FRACTAL),
Spatial Positioning Transform (/TRANSFORM/POS, /TRANSFORM/POSITION),
ALE Grid Velocity Controls (/ALE/GRID/FLOW-TRACKING, /ALE/GRID/LAGRANGE, /ALE/ZERO),
Architecture Specifications (/ARCH), Keyword Documentation Tag (/ALTDOCTAG),
External Links (/EXTERN/LINK, /EXTLINK), Subdomains (/SUBDOMAIN),
and LS-DYNA Include (/INCLUDE_LS-DYNA, /INCLUDE_DYNA).
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
Test_Deck_M118
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
/GRSHEL/1
ShellGroup1
         1
/GRSH3N/2
TriaGroup2
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


def test_m118_fail_fractal_fixed_and_free(tmp_path):
    deck = _BOILERPLATE + """\
/FAIL/FRACTAL_DMG/1
         1         2         1         2
                0.25                0.75        42        10         1
       100
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.fail_fractals
    ff1 = model.fail_fractals[1]
    assert ff1.mat_id == 1
    assert ff1.grsh4n_1 == 1
    assert ff1.grsh3n_1 == 2
    assert ff1.damage == 0.25
    assert ff1.probability == 0.75
    assert ff1.seed == 42
    assert ff1.num_walk == 10
    assert ff1.printout == 1
    assert ff1.fail_id == 100


def test_m118_transform_position(tmp_path):
    deck = _BOILERPLATE + """\
/TRANSFORM/POSITION/1
TransformPos1
        10         1         2         3         4         0         0                   0
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 1.0                 1.0                 0.0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.transform_positions
    tp = model.transform_positions[1]
    assert tp.id == 1
    assert tp.title == "TransformPos1"
    assert tp.grnod_id == 10
    assert tp.node_ids[:4] == (1, 2, 3, 4)
    assert len(tp.points) >= 3
    assert tp.points[0] == (0.0, 0.0, 0.0)
    assert tp.points[1] == (1.0, 0.0, 0.0)


def test_m118_ale_grid_controls(tmp_path):
    deck = _BOILERPLATE + """\
/ALE/GRID/FLOW-TRACKING
         1                 1.5
         1                 0.8
/ALE/GRID/LAGRANGE
/ALE/ZERO
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.ale_grid_flow_tracking is not None
    assert model.ale_grid_flow_tracking["is_def"] == 1
    assert model.ale_grid_flow_tracking["scale_def"] == 1.5
    assert model.ale_grid_flow_tracking["is_rot"] == 1
    assert model.ale_grid_flow_tracking["scale_rot"] == 0.8
    assert model.ale_grid_lagrange is True
    assert model.ale_zero is True


def test_m118_arch_and_altdoctag(tmp_path):
    deck = _BOILERPLATE + """\
/ARCH
         0         1         2         3         4         5         6         7
/ALTDOCTAG
Keyword Reference OpenRadioss M118
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.arch_specs) == 1
    assert model.arch_specs[0].mach == (0, 1, 2, 3, 4, 5, 6, 7)
    assert len(model.altdoctags) == 1
    assert "Keyword Reference OpenRadioss M118" in model.altdoctags[0]


def test_m118_external_link_and_subdomain(tmp_path):
    deck = _BOILERPLATE + """\
/EXTERN/LINK/1
ExternalLink1
        10
/SUBDOMAIN/10
SubdomainTen
         1
/INCLUDE_LS-DYNA
included_model.k
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.external_links
    assert model.external_links[1].grnod_id == 10
    assert 10 in model.subdomains
    assert model.subdomains[10].part_ids == [1]


def test_m118_cross_reference_errors(tmp_path):
    deck = _BOILERPLATE + """\
/FAIL/FRACTAL_DMG/99
         1         2         1         2
                0.25                0.75        42        10         1
/TRANSFORM/POS/99
FaultyTransform
       999         1         2         3         4         0         0                   0
/EXTERN/LINK/99
FaultyExtLink
       888
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/FAIL/FRACTAL/99: material 99 not defined" in err_text
    assert "/TRANSFORM/99: node group 999 not defined" in err_text
    assert "/EXTERN/LINK/99: node group 888 not defined" in err_text


def test_m118_free_format(tmp_path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Free_Format_M118
/MAT/LAW1/1
LinearElastic
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell_1
1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/GRNOD/NODE/10
NodeGroup10
1 2
/FAIL/FRACTAL/1
1 0 0 0
0.5 0.9 1234 50 0
99
/TRANSFORM/POS/2
FreePos2
10 1 2 3 4 0 0 0
0.0 0.0 0.0
2.0 0.0 0.0
/EXTLINK/2
ExtLinkFree
10
/SUBDOMAIN/20
SubdomainFree
1 -2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.fail_fractals
    assert model.fail_fractals[1].damage == 0.5
    assert model.fail_fractals[1].seed == 1234
    assert 2 in model.transform_positions
    assert model.transform_positions[2].grnod_id == 10
    assert 2 in model.external_links
    assert model.external_links[2].grnod_id == 10
    assert 20 in model.subdomains
    assert 1 in model.subdomains[20].part_ids
    assert 2 in model.subdomains[20].neg_part_ids

