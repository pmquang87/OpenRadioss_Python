"""M37 GROUP/SET + deck-infrastructure tests.

The subtype machinery this file covers is the single biggest structural
unlock of the official-deck corpus (48 % of the cases reference at least
one of these keywords): /GRNOD/SURF, recursive /GRNOD/GRNOD (fixpoint +
cycle detection + the negative-id REMOVE convention), /GRNOD/BOX with
the three real /BOX geometries, element groups (/GRSHEL, /GRSH3N,
/GRBRIC, /GRPART...), /SURF/SURF, /SURF/GRSHEL|GRSH3N, /SURF/PART/EXT,
/LINE/EDGE|LINE|PART, /FUNCT_SMOOTH and the /UNIT + /BEGIN work-unit
system.

Real snippets are byte-faithful extracts from the official corpus
(TWISBEAM RD-E-0100, S_BEAM RD-E-0300, DBEND_44 RD-E-2500,
Cantilever_beam RD-V-0020, main_TEST4 RD-E-2601) so the tests fail
exactly when the parser/resolver regresses on real decks.  The /UNIT
conversion numbers are validated against the REAL Windows Fortran
starter run on main_TEST4 (listing: INITIAL DENSITY 2.8000E-09 from a
0.0028 g/mm/ms card in an Mg/mm/s model; YOUNG MODULUS unchanged at
71000).
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import SmoothFunctTable
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.units import apply_unit_conversions
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              resolve_entity_groups,
                                              resolve_lines,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)


def _parse(deck_text, tmp_path):
    f = tmp_path / "GRP_0000.rad"
    f.write_text(deck_text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    return model, log


def _resolve(deck_text, tmp_path):
    """Parse + the full Starter resolve chain in the M37 order:
    units -> materials -> elements -> element groups -> surfaces ->
    lines -> node groups."""
    model, log = _parse(deck_text, tmp_path)
    apply_unit_conversions(model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_entity_groups(model, log)
    resolve_surfaces(model, log)
    resolve_lines(model, log)
    resolve_node_groups(model, log)
    return model, log


def _nodes(model, g):
    """Resolved node group -> sorted user node ids."""
    return sorted(int(model.node_ids[i]) for i in g.node_idx)


# ============================================================================
# Hand-built legacy (free-format) plate: 6 nodes, 2 quad shells
#
#   4---5---6        shells 11 (1 2 5 4) and 12 (2 3 6 5), part 1:
#   | 11| 12|        the interior edge 2-5 is shared, all other edges
#   1---2---3        are borders.
# ============================================================================

PLATE = """/BEGIN
plate
/NODE
1 0 0 0
2 1 0 0
3 2 0 0
4 0 1 0
5 1 1 0
6 2 1 0
/SHELL/1
11 1 2 5 4
12 2 3 6 5
/PART/1
plate
1 1
/MAT/LAW1/1
steel
7.8e-9
210000 0.3
/PROP/SHELL/1
shell
1.0 3 0.01
"""


# ============================================================================
# 1. /GRNOD subtypes
# ============================================================================

def test_grnod_surf_nodes_of_surface(tmp_path):
    """/GRNOD/SURF = the nodes of the surface's segments (hm_surfnod.F);
    the group is defined BEFORE the surface — two-pass resolution."""
    deck = PLATE + (
        "/GRNOD/SURF/10\nnodes of surf\n5\n"
        "/SURF/PART/5\nplate skin\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[10]) == [1, 2, 3, 4, 5, 6]


def test_grnod_grnod_recursive_and_negative(tmp_path):
    """Group-of-groups: unions resolve recursively; a NEGATIVE id
    REMOVES the referenced group's nodes, and removal wins over
    addition whatever the reference order (hm_grogronod.F BUFTMP=-1)."""
    deck = PLATE + (
        "/GRNOD/NODE/1\nbase\n1 2 3\n"
        "/GRNOD/NODE/2\nremove me\n2\n"
        "/GRNOD/GRNOD/3\nunion minus\n1 -2\n"
        "/GRNOD/GRNOD/4\nremoval first\n-2 1\n"
        "/GRNOD/GRNOD/5\nnested\n3\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[3]) == [1, 3]
    # removal-wins: same result with the negative reference first
    assert _nodes(model, model.node_groups[4]) == [1, 3]
    # second-level nesting sees the child's FINAL (removed) content
    assert _nodes(model, model.node_groups[5]) == [1, 3]


def test_grnod_grnod_cycle_detected(tmp_path):
    """A reference cycle must be caught (upstream MSGID 176 'ITER >
    NGRNOD' stop), reported as an error, and the groups left empty."""
    deck = PLATE + (
        "/GRNOD/GRNOD/1\na\n2\n"
        "/GRNOD/GRNOD/2\nb\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    errs = "\n".join(log.errors)
    assert "circular" in errs and "GRNOD" in errs, log.errors
    assert model.node_groups[1].node_idx.size == 0
    assert model.node_groups[2].node_idx.size == 0


def test_grnod_grnod_unknown_reference_warns(tmp_path):
    """An unknown referenced group is a WARNING (upstream MSGID 174),
    not an error — the rest of the group still resolves."""
    deck = PLATE + (
        "/GRNOD/NODE/1\nbase\n4\n"
        "/GRNOD/GRNOD/2\nrefs\n1 999\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert any("999" in w for w in log.warnings), log.warnings
    assert _nodes(model, model.node_groups[2]) == [4]


def test_grnod_gene_ranges(tmp_path):
    """/GRNOD/GENE first/last pairs select by USER id; GEN_INCR adds the
    increment (MOD(id - first, incr) == 0, hm_lecgrn.F)."""
    deck = PLATE + (
        "/GRNOD/GENE/7\nrange\n2 5\n"
        "/GRNOD/GEN_INCR/8\nstepped\n1 6 5\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[7]) == [2, 3, 4, 5]
    assert _nodes(model, model.node_groups[8]) == [1, 6]


def test_grnod_grshel_nodes_of_element_group(tmp_path):
    """/GRNOD/GRSHEL = the nodes of the /GRSHEL element group."""
    deck = PLATE + (
        "/GRSHEL/SHEL/3\nleft shell\n11\n"
        "/GRNOD/GRSHEL/9\nits nodes\n3\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[9]) == [1, 2, 4, 5]


# ============================================================================
# 2. Element groups
# ============================================================================

def test_grshel_direct_and_part(tmp_path):
    deck = PLATE + (
        "/GRSHEL/SHEL/1\ndirect\n12\n"
        "/GRSHEL/PART/2\nby part\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    (attr, rows), = model.egroups["SHEL"][1].members
    assert attr == "shells"
    assert model.shells.ids[rows].tolist() == [12]
    (attr2, rows2), = model.egroups["SHEL"][2].members
    assert sorted(model.shells.ids[rows2].tolist()) == [11, 12]


def test_grshel_group_of_groups_negative_and_cycle(tmp_path):
    """/GRSHEL/GRSHEL follows the same signed fixpoint as /GRNOD/GRNOD
    (hm_grogro.F); cycles are detected."""
    deck = PLATE + (
        "/GRSHEL/SHEL/1\nall\n11 12\n"
        "/GRSHEL/SHEL/2\nminus\n12\n"
        "/GRSHEL/GRSHEL/3\ndiff\n1 -2\n"
        "/GRSHEL/GRSHEL/4\ncycle a\n5\n"
        "/GRSHEL/GRSHEL/5\ncycle b\n4\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    (attr, rows), = model.egroups["SHEL"][3].members
    assert model.shells.ids[rows].tolist() == [11]
    errs = "\n".join(log.errors)
    assert "circular" in errs, log.errors
    assert model.egroups["SHEL"][4].members == []


def test_grbric_part_and_grnod_grbric(tmp_path):
    """/GRBRIC/PART takes all SOLID elements of the parts (bricks and
    tetras — IGRBRIC spans the whole IXS); /GRBRIC/BRIC takes ids."""
    deck = (
        "/BEGIN\ncube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n21 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n"
        "/MAT/LAW1/1\nsteel\n7.8e-9\n210000 0.3\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRBRIC/PART/55\nsolids\n1\n"
        "/GRBRIC/BRIC/56\ndirect\n21\n"
        "/GRNOD/GRBRIC/57\nnodes\n55\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    (attr, rows), = model.egroups["BRIC"][55].members
    assert attr == "bricks" and model.bricks.ids[rows].tolist() == [21]
    (attr2, rows2), = model.egroups["BRIC"][56].members
    assert model.bricks.ids[rows2].tolist() == [21]
    assert _nodes(model, model.node_groups[57]) == list(range(1, 9))


def test_grpart_part_list(tmp_path):
    deck = PLATE + ("/GRPART/PART/44\nparts\n1\n/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.egroups["PART"][44].part_ids_resolved == [1]


# ============================================================================
# 3. Surfaces
# ============================================================================

def test_surf_grshel_segments(tmp_path):
    """/SURF/GRSHEL: every element of the group becomes a segment, with
    element provenance (surftage)."""
    deck = PLATE + (
        "/GRSHEL/SHEL/3\ngrp\n11\n"
        "/SURF/GRSHEL/30\nskin\n3\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    s = model.surfaces[30]
    assert s.segments.shape == (1, 4)
    assert sorted(model.node_ids[s.segments[0]].tolist()) == [1, 2, 4, 5]
    assert s.seg_gtype[0] == "shells"


def test_surf_surf_concatenation_and_reversal(tmp_path):
    """/SURF/SURF concatenates the referenced surfaces' segments; a
    NEGATIVE reference reverses the node order (the normal flips —
    hm_read_surfsurf.F NODES(L,4..1)); resolution is order-free."""
    deck = PLATE + (
        "/SURF/SURF/40\ncombined\n41 -42\n"
        "/SURF/PART/41\nskin\n1\n"
        "/SURF/SEG/42\nexplicit\n1 2 5 4\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    s40 = model.surfaces[40]
    s41 = model.surfaces[41]
    s42 = model.surfaces[42]
    assert s40.segments.shape[0] == s41.segments.shape[0] + 1
    # the negative reference contributes s42's segment REVERSED
    assert s40.segments[-1].tolist() == s42.segments[0][::-1].tolist()
    # provenance survives the concatenation
    assert set(s40.seg_gtype) == {"shells", ""}


def test_surf_surf_cycle_detected(tmp_path):
    deck = PLATE + (
        "/SURF/SURF/1\na\n2\n"
        "/SURF/SURF/2\nb\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    errs = "\n".join(log.errors)
    assert "circular" in errs and "SURF" in errs, log.errors
    assert model.surfaces[1].segments.shape[0] == 0


# ============================================================================
# 4. Lines
# ============================================================================

def test_line_edge_border_edges_only(tmp_path):
    """/LINE/EDGE keeps only the BORDER edges (used by exactly one
    segment): the plate has 7 unique edges, 6 of them borders — the
    shared interior edge 2-5 is removed ENTIRELY (linedge.F), where
    /LINE/SURF keeps one copy of every unique edge."""
    deck = PLATE + (
        "/SURF/PART/1\nskin\n1\n"
        "/LINE/EDGE/1\nborder\n1\n"
        "/LINE/SURF/2\nall edges\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    border = model.lines[1].segments
    allsurf = model.lines[2].segments
    assert allsurf.shape[0] == 7
    assert border.shape[0] == 6
    i2, i5 = model.node_index(2), model.node_index(5)
    pairs = {tuple(sorted(e)) for e in border.tolist()}
    assert tuple(sorted((i2, i5))) not in pairs
    # /LINE/SURF DOES carry the interior edge
    pairs_all = {tuple(sorted(e)) for e in allsurf.tolist()}
    assert tuple(sorted((i2, i5))) in pairs_all


def test_line_line_concatenation(tmp_path):
    deck = PLATE + (
        "/LINE/SEG/1\nedge a\n1 2\n"
        "/LINE/SEG/2\nedge b\n2 3\n"
        "/LINE/LINE/3\nboth\n1 2\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.lines[3].segments.shape[0] == 2


def test_line_line_cycle_detected(tmp_path):
    deck = PLATE + (
        "/LINE/LINE/1\na\n2\n"
        "/LINE/LINE/2\nb\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert any("circular" in e for e in log.errors), log.errors


def test_line_part_1d_elements(tmp_path):
    """/LINE/PART: every 1-D element (here trusses) of the part becomes
    an edge with provenance."""
    deck = (
        "/BEGIN\nrods\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 2 0 0\n"
        "/TRUSS/2\n31 1 2\n32 2 3\n"
        "/PART/2\nrods\n2 1\n"
        "/MAT/LAW1/1\nsteel\n7.8e-9\n210000 0.3\n"
        "/PROP/TRUSS/2\nrod\n1.0\n"
        "/LINE/PART/6\nrod line\n2\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    ln = model.lines[6]
    assert ln.segments.shape[0] == 2
    assert set(ln.seg_gtype) == {"trusses"}


# ============================================================================
# 5. Boxes (RECTA node corners, CYLIN, SPHER)
# ============================================================================

def test_box_recta_corner_nodes(tmp_path):
    """RECTA corners can be NODES (recta.cfg N1/N2) — resolved against
    the mesh positions."""
    deck = PLATE + (
        "/BOX/RECTA/1\nnode box\n"
        "1 5 0\n"                    # N1=node 1, N2=node 5 -> [0,1]x[0,1]
        "0 0 0\n0 0 0\n"
        "/GRNOD/BOX/2\ninside\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[2]) == [1, 2, 4, 5]


def test_box_cylin_membership(tmp_path):
    """CYLIN: finite cylinder P1->P2, radius D/2, caps inclusive
    (rdbox.F INSIDE_CYLINDER). Axis along x through y=z=0: nodes 1,2,3
    (y=0) are within radius 0.5; nodes 4,5,6 (y=1) are not."""
    deck = PLATE + (
        "/BOX/CYLIN/1\ncyl\n"
        "0 0 1.0\n"                  # Base_N=0 Dir_N=0 Diameter=1.0
        "-0.5 0 0\n2.5 0 0\n"
        "/GRNOD/BOX/2\nin cyl\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[2]) == [1, 2, 3]


def test_box_spher_membership(tmp_path):
    """SPHER: |x - c| <= D/2 — center at node 2 (1,0,0), D=2.1 catches
    nodes 1,2,3 (distance 1) and node 5 (distance 1) but not 4/6
    (distance sqrt(2))."""
    deck = PLATE + (
        "/BOX/SPHER/1\nball\n"
        "0 2.1\n"                    # N1=0 -> center from Xp card
        "1.0 0 0\n"
        "/GRNOD/BOX/2\nin ball\n1\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[2]) == [1, 2, 3, 5]


# ============================================================================
# 6. REAL corpus snippets (byte-faithful option blocks + fixed mesh)
# ============================================================================

FIXED_HEADER = (
    "/BEGIN\n"
    "GRPTEST                                                            \n"
    "      2022         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n"
)


def _fi(*ids):
    """A fixed %10d id card."""
    return "".join(f"{i:>10d}" for i in ids)


def _fnode(nid, x, y, z):
    return f"{nid:>10d}{x:>20}{y:>20}{z:>20}"


#: fixed-dialect version of the 2-shell plate (parts 1; shells 11, 12)
FIXED_PLATE = "\n".join([
    "/NODE",
    _fnode(1, 0.0, 0.0, 0.0), _fnode(2, 1.0, 0.0, 0.0),
    _fnode(3, 2.0, 0.0, 0.0), _fnode(4, 0.0, 1.0, 0.0),
    _fnode(5, 1.0, 1.0, 0.0), _fnode(6, 2.0, 1.0, 0.0),
    "/SHELL/1",
    _fi(11, 1, 2, 5, 4), _fi(12, 2, 3, 6, 5),
    "/PART/1",
    "plate",
    _fi(1, 1),
    "/MAT/LAW1/1",
    "steel",
    f"{7.8e-9:>20}",
    f"{210000.0:>20}{0.3:>20}",
    "/PROP/SHELL/1",
    "shell",
    "",                              # flags card (blank = defaults)
    "",                              # hourglass card
    _fi(3, 1) + f"{1.0:>20}",        # N Istrain Thick
]) + "\n"


# byte-faithful from TWISBEAM_0000.rad lines 469-479 (RD-E-0100),
# adapted only in the part list (the corpus card lists parts 1 2 3; the
# fixture mesh has part 1 — unknown parts contribute nothing upstream
# and here alike)
TWISBEAM_CHAIN = """\
/GRNOD/GRNOD/12
New GRNOD 12
        13
#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|
/GRNOD/GRNOD/13
New GRNOD 12 _parts_subgroup_
        14
#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|
/GRNOD/PART/14
New GRNOD 12 _parts_subsubgroup_
         1         2         3
"""


def test_real_twisbeam_grnod_grnod_chain(tmp_path):
    """RD-E-0100 TWISBEAM: /GRNOD/GRNOD/12 -> 13 -> /GRNOD/PART/14 — a
    two-level recursive chain (1216 references in the corpus)."""
    deck = FIXED_HEADER + FIXED_PLATE + TWISBEAM_CHAIN + "/END\n"
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    expected = [1, 2, 3, 4, 5, 6]
    assert _nodes(model, model.node_groups[14]) == expected
    assert _nodes(model, model.node_groups[13]) == expected
    assert _nodes(model, model.node_groups[12]) == expected


# byte-faithful from S_BEAM_0000.rad (RD-E-0300): the /GRNOD/SURF ->
# /SURF/PART pair every /INTER secondary side of that family uses
S_BEAM_GRNOD_SURF = """\
/GRNOD/SURF/20
INTER_group_20_of_SURF
        19
/SURF/PART/19
INTER_group_19_of_PART
         1
"""


def test_real_sbeam_grnod_surf(tmp_path):
    deck = FIXED_HEADER + FIXED_PLATE + S_BEAM_GRNOD_SURF + "/END\n"
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert _nodes(model, model.node_groups[20]) == [1, 2, 3, 4, 5, 6]


# byte-faithful from DBEND_44_0000.rad (RD-E-2500 spring-back — the
# 'ROLLIN'-style /GRNOD/BOX sets): real /BOX/RECTA layout with the
# N1/N2/ISKEW card and the ITYPE trailing column
DBEND_BOX = """\
/BOX/RECTA/1
BOX
#       N1        N2     ISKEW                                                                 ITYPE
         0         0         0                                                                     0
#                XP1                 YP1                 ZP1
                -.01                -.01                -.01
#                XP2                 YP2                 ZP2
                 .01               17.49                 .01
/GRNOD/BOX/1
NUM3HS1D00_bcs1_100_011
         1
"""


def test_real_dbend_box_recta(tmp_path):
    """The DBEND box spans x,z in [-0.01, 0.01], y in [-0.01, 17.49]:
    of the fixture plate only nodes 1 (0,0,0) and 4 (0,1,0) fall in."""
    deck = FIXED_HEADER + FIXED_PLATE + DBEND_BOX + "/END\n"
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    box = model.boxes[1]
    assert box.kind == "RECTA"
    assert box.corner_min == pytest.approx([-0.01, -0.01, -0.01])
    assert box.corner_max == pytest.approx([0.01, 17.49, 0.01])
    assert _nodes(model, model.node_groups[1]) == [1, 4]


# byte-faithful from BIKERC_1506_1104_0000.rad (RD-E-1200): the
# /SURF/GRSHEL -> /GRSHEL/SHEL pattern (element ids adapted to the
# fixture mesh: the corpus block lists shells 138202+)
BIKE_SURF_GRSHEL = """\
/SURF/GRSHEL/32
New SURF 18 _parts_subsubgroup_ _4 nodes shells_subgroup_
         3
/GRSHEL/SHEL/3
New SURF 18 _parts_subsubgroup_ _4 nodes shells_subsubgroup_
        11        12
"""


def test_real_bike_surf_grshel(tmp_path):
    deck = FIXED_HEADER + FIXED_PLATE + BIKE_SURF_GRSHEL + "/END\n"
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    s = model.surfaces[32]
    assert s.segments.shape[0] == 2
    assert set(s.seg_gtype) == {"shells"}


# byte-faithful from BOXBEAM_0000.rad (RD-E-1701): /LINE/EDGE of a
# /SURF/PART — the 190-case pattern
BOXBEAM_LINE_EDGE = """\
/LINE/EDGE/1
edges of boxbeam
         1
/SURF/PART/1
master surface
         1
"""


def test_real_boxbeam_line_edge(tmp_path):
    deck = FIXED_HEADER + FIXED_PLATE + BOXBEAM_LINE_EDGE + "/END\n"
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    ln = model.lines[1]
    assert ln.segments.shape[0] == 6      # borders only (7 unique edges)
    i2, i5 = model.node_index(2), model.node_index(5)
    pairs = {tuple(sorted(e)) for e in ln.segments.tolist()}
    assert tuple(sorted((i2, i5))) not in pairs


# ============================================================================
# 7. /FUNCT_SMOOTH
# ============================================================================

# byte-faithful from Cantilever_beam_0000.rad (RD-V-0020)
CANTILEVER_SMOOTH = """\
/FUNCT_SMOOTH/1
Smooth Load Function
#            Ascalex             Fscaley             Ashiftx             Fshifty
                   0                   0                   0                   0
#                  X                   Y
                   0                   0
                   1                   1
"""


def test_real_funct_smooth_parses_and_evaluates(tmp_path):
    """The RD-V-0020 smooth ramp: quintic smoothstep between (0,0) and
    (1,1), CLAMPED outside (finter_smooth.F)."""
    deck = FIXED_HEADER + CANTILEVER_SMOOTH + "/END\n"
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    f = model.functions[1]
    assert isinstance(f, SmoothFunctTable)
    # midpoint of the smoothstep: s^3(10 - 15s + 6s^2) at s = 0.5 -> 0.5
    assert f.eval(0.5) == pytest.approx(0.5)
    # quarter point: 0.25^3 * (10 - 3.75 + 0.375) = 0.103515625
    assert f.eval(0.25) == pytest.approx(0.103515625)
    # C2 ends: zero slope -> values hug the ends near the data points
    assert f.eval(0.01) == pytest.approx(0.0, abs=1e-4)
    # CLAMPED outside the definition interval (the linear /FUNCT
    # extrapolates instead)
    assert f.eval(2.0) == pytest.approx(1.0)
    assert f.eval(-1.0) == pytest.approx(0.0)


def test_funct_smooth_scale_and_shift(tmp_path):
    """Ascalex/Fscaley/Ashiftx/Fshifty transform the points AT READ
    time (hm_read_funct.F ISMOOTH=1: X*fac + shift before storage)."""
    deck = FIXED_HEADER + (
        "/FUNCT_SMOOTH/2\n"
        "scaled\n"
        f"{2.0:>20}{3.0:>20}{10.0:>20}{1.0:>20}\n"
        f"{0.0:>20}{0.0:>20}\n"
        f"{1.0:>20}{1.0:>20}\n"
        "/END\n")
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    f = model.functions[2]
    assert f.x.tolist() == [10.0, 12.0]      # x*2 + 10
    assert f.y.tolist() == [1.0, 4.0]        # y*3 + 1


def test_funct_smooth_blank_scale_card(tmp_path):
    """A BLANK scale card (the corpus rdv_0530 style) means all
    defaults — scale 1, shift 0."""
    deck = FIXED_HEADER + (
        "/FUNCT_SMOOTH/3\n"
        "smooth function\n"
        "\n"
        f"{0.0:>20}{0.0:>20}\n"
        f"{0.15:>20}{0.3447:>20}\n"
        f"{1.0:>20}{0.3447:>20}\n"
        "/END\n")
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    f = model.functions[3]
    assert f.x.tolist() == pytest.approx([0.0, 0.15, 1.0])
    assert f.eval(0.5) == pytest.approx(0.3447)   # flat tail segment


# ============================================================================
# 8. /UNIT + /BEGIN work units
# ============================================================================

# byte-faithful from main_TEST4_0000.rad (RD-E-2601): a /MAT written in
# a LOCAL g/mm/ms system inside an Mg/mm/s model.  Ground truth from
# the REAL Windows starter's listing on this very deck:
#     INITIAL DENSITY . . . = 2.8000000000000E-09
#     YOUNG MODULUS . . . . =  71000.00000000
MAIN_TEST4 = """\
/BEGIN
main_TEST4
      2019         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/PLAS_JOHNS/1/1
Alu
#              RHO_I
               .0028                   0
#                  E                  Nu     Iflag
               71000                 .33         0
#                  a                   b                   n           EPS_p_max            SIG_max0
                 290               562.3                 .63               .1559                 425
#                  c           EPS_DOT_0       ICC   Fsmooth               F_cut               Chard
                   0                   0         0         0                   0                   0
#                  m              T_melt              rhoC_p                 T_r
                   0                   0                   0                   0
/UNIT/1
unit for mat
#              MUNIT               LUNIT               TUNIT
                   g                  mm                  ms
/UNIT/2
unit for PROP
#              MUNIT               LUNIT               TUNIT
                  Mg                  mm                   s
/END
"""


def test_real_main_test4_unit_conversion(tmp_path):
    """The /UNIT machinery reproduces the REAL starter numerically:
    density .0028 g/mm3 -> 2.8e-9 Mg/mm3 (factor 1e-6); the stress
    fields' g/mm/ms -> Mg/mm/s ratio is exactly 1 (E stays 71000, the
    yield constants stay as written) — validated against the Fortran
    binary's listing."""
    model, log = _resolve(MAIN_TEST4, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.rho0 == pytest.approx(2.8e-9)
    assert mat.params["E"] == pytest.approx(71000.0)
    assert mat.params["A"] == pytest.approx(290.0)
    assert mat.params["B"] == pytest.approx(562.3)
    # /UNIT blocks themselves parsed (g = 1e-3 kg; Mg = 1e3 kg)
    assert model.units[1] == pytest.approx((1e-3, 1e-3, 1e-3))
    assert model.units[2] == pytest.approx((1e3, 1e-3, 1.0))
    assert model.unit_work == pytest.approx((1e3, 1e-3, 1.0))


def test_unit_conversion_nontrivial_stress_ratio(tmp_path):
    """A /MAT in SI (kg/m/s) inside an Mg/mm/s model: rho * 1e-12,
    E * 1e-6 (2.1e11 Pa -> 2.1e5 MPa) — the full dimension formula."""
    deck = (
        "/BEGIN\n"
        "UCONV                                                              \n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/UNIT/5\n"
        "SI\n"
        "                  kg                   m                   s\n"
        "/MAT/LAW1/1/5\n"
        "steel SI\n"
        f"{7800.0:>20}\n"
        f"{2.1e11:>20}{0.3:>20}\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.rho0 == pytest.approx(7.8e-9)
    assert mat.params["E"] == pytest.approx(2.1e5)
    assert mat.params["nu"] == pytest.approx(0.3)   # dimensionless


def test_unit_conversion_prop_shell_thickness(tmp_path):
    """/PROP quantities convert too: a thickness written in meters
    lands in millimeters (length ratio 1e3)."""
    deck = (
        "/BEGIN\n"
        "PCONV                                                              \n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/UNIT/5\n"
        "SI\n"
        "                  kg                   m                   s\n"
        "/PROP/SHELL/1/5\n"
        "si shell\n"
        "\n"
        "\n"
        f"{3:>10d}{1:>10d}{0.002:>20}\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.properties[1].params["thick"] == pytest.approx(2.0)
    assert model.properties[1].params["nip"] == 3     # dimensionless


def test_unit_numeric_factor_fields(tmp_path):
    """/BEGIN unit fields may be plain SI factors (TWISBEAM writes
    '453.35 25.4 1' — a lbf-inch system)."""
    deck = (
        "/BEGIN\n"
        "TWISBEAM                                                           \n"
        "       140         0\n"
        "              453.35                25.4                   1\n"
        "              453.35                25.4                   1\n"
        "/END\n")
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.unit_work == pytest.approx((453.35, 25.4, 1.0))


def test_unit_identity_is_silent(tmp_path):
    """A local /UNIT equal to the work system converts by 1.0 exactly —
    no warnings, values untouched (the main_TEST4 /PROP case)."""
    deck = (
        "/BEGIN\n"
        "IDU                                                                \n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/UNIT/2\n"
        "work units\n"
        "                  Mg                  mm                   s\n"
        "/PROP/SHELL/1/2\n"
        "shell\n"
        "\n"
        "\n"
        f"{3:>10d}{1:>10d}{1.5:>20}\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert not log.warnings, log.warnings
    assert model.properties[1].params["thick"] == pytest.approx(1.5)


def test_unit_unknown_id_errors(tmp_path):
    deck = (
        "/BEGIN\n"
        "BADU                                                               \n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/MAT/LAW1/1/99\n"
        "steel\n"
        f"{7.8e-9:>20}\n"
        f"{210000.0:>20}{0.3:>20}\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert any("UNIT/99" in e for e in log.errors), log.errors


def test_unit_on_unconverted_keyword_warns_loudly(tmp_path):
    """A non-MAT/PROP block referencing a DIFFERENT local unit system
    is not converted — but no longer silently ignored: loud warning."""
    deck = (
        "/BEGIN\n"
        "WARNU                                                              \n"
        "      2022         0\n"
        "                  Mg                  mm                   s\n"
        "                  Mg                  mm                   s\n"
        "/UNIT/5\n"
        "SI\n"
        "                  kg                   m                   s\n"
        "/FUNCT/7/5\n"
        "curve in SI\n"
        f"{0.0:>20}{0.0:>20}\n"
        f"{1.0:>20}{1.0:>20}\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert not log.errors, log.errors
    assert any("FUNCT" in w and "NOT unit-converted" in w
               for w in log.warnings), log.warnings


def test_unit_without_begin_units_errors(tmp_path):
    """Local /UNIT references without a /BEGIN work system cannot be
    converted — an error, not silence (legacy port decks carry no
    unit cards)."""
    deck = (
        "/BEGIN\nnounits\n"
        "/UNIT/5\nSI\n"
        "                  kg                   m                   s\n"
        "/MAT/LAW1/1/5\nsteel\n7.8e-9\n210000 0.3\n"
        "/END\n")
    model, log = _resolve(deck, tmp_path)
    assert any("work unit system" in e for e in log.errors), log.errors
