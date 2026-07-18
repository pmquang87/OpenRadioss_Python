"""M39 — the RD-V-0700 SOLID imposed-velocity energy-injection bug.

Root cause (a /PROP/SOLID READER column bug, not an element/kinematics bug)
--------------------------------------------------------------------------
M38 fixed the RD-V-0700 SHELL/TRIA cycle-1 anomaly (the endpoint-vs-midstep
constraint-work booking in ``apply_kinematic``; see
test_m38_ledger_hardening).  The SAME family's BRICK and TETRA decks kept
INJECTING energy and were flagged ``NUMERICAL ENERGY INJECTION ... RUN
UNSTABLE`` within a handful of cycles.

The cause was in the FIXED-FORMAT /PROP/SOLID reader (added in M38).  The
real 2022 layout is

    card 0:  Isolid Ismstr Iale Icpre Itetra10 Inpts Itetra4 Iframe Dn
    card 1:  qa qb h Lambda Mu

The reader tried to pick the ``qa qb h`` card by *skipping cards whose
tokens are all integers*.  But the flag card (card 0) ends in ``Dn = 0.0``,
a FLOAT token, so ``all(tok.isdigit())`` is False and the flag card slipped
through as the ``qa qb h`` card.  The port then read

    qa = Isolid (= 18 or 24)      (should be 1.1)
    h  = Itetra4 / Icpre (= -1)   (should be 0.1)

A NEGATIVE hourglass viscosity turns the Flanagan-Belytschko viscous
hourglass damper into an AMPLIFIER: the modal hourglass velocity grows
~exponentially from round-off (measured ~6x/cycle) until it dominates,
HOURGLASS ENERGY runs NEGATIVE, and the numerical-injection guard fires
(bricks c13/c19/c20 at -38..-76 %).  The tetra has no hourglass but
inherited qa = 18 (a ~16x bulk viscosity), destabilising it too.

The Fortran reference runs every deck to NORMAL with HOURGLASS ENERGY == 0
for the whole run (Isolid=18/24 are FULLY INTEGRATED — 8 integration
points, no hourglass), confirming the physics is stable.

The fix (starter_keywords.read_prop, ptype==14) reads qa/qb/h by COLUMN-CUT
of data card 1 for fixed-format decks, so h = 0.1 (a proper viscous damper)
and qa = 1.1.  This file is the anti-regression canary for exactly that.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter

# A real fixed-2022 /BEGIN header makes the blocks parse in the FIXED
# dialect (block.fixed True) — the path that carried the bug.
FIXED_BEGIN = (
    "/BEGIN\nm39-solid-impvel\n      2022         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n")


def _col(v, w):
    return f"{v:>{w}}"


def _prop_solid_fixed(isolid, icpre, itetra4, qa, qb, h):
    """The exact real-layout /PROP/SOLID card pair that broke the reader:
    an integer flag card ending in the Dn=0.0 FLOAT, then the qa/qb/h card
    (%20lg fields)."""
    flags = (_col(isolid, 10) + _col(0, 10) + _col("", 10) + _col(icpre, 10)
             + _col(0, 10) + _col(0, 10) + _col(itetra4, 10) + _col("", 10)
             + _col("0.0", 20))
    visc = _col(qa, 20) + _col(qb, 20) + _col(h, 20)
    dtm = _col("0.0", 20) + _col("0.0", 20)
    return ("/PROP/SOLID/100001\nFoam\n"
            + flags + "\n" + visc + "\n" + dtm + "\n")


def _parse_prop(tmp_path, body):
    deck = tmp_path / "p_0000.rad"
    deck.write_text(FIXED_BEGIN + body + "/END\n")
    log, model = MessageLog(), Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == "PROP":
            read_prop(b, model, log)
    return model, log


# ---------------------------------------------------------------------------
# Part A — the reader unit test (the direct anti-regression canary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("isolid", [18, 24])
def test_prop_solid_fixed_reads_qa_qb_h_not_the_flag_card(tmp_path, isolid):
    """The real fixed-layout /PROP/SOLID card must yield qa=1.1, qb=0.05,
    h=0.1 — NOT qa=Isolid (18/24) and h=Itetra4 (-1).  A negative h is the
    energy-injecting bug this whole file exists to prevent."""
    model, log = _parse_prop(
        tmp_path, _prop_solid_fixed(isolid, -1, -1, 1.1, 0.05, 0.1))
    p = model.properties[100001]
    assert p.type == 14
    assert p.params["qa"] == pytest.approx(1.1), p.params
    assert p.params["qb"] == pytest.approx(0.05), p.params
    assert p.params["h"] == pytest.approx(0.1), p.params
    # the decisive guard: a POSITIVE hourglass viscosity (a damper, not an
    # amplifier) and a physical (not Isolid-sized) bulk viscosity
    assert p.params["h"] > 0.0
    assert p.params["qa"] < 5.0


def test_prop_solid_free_short_form_unchanged(tmp_path):
    """The port's historical free-format short form 'qa qb h' still reads
    directly (the else branch is untouched)."""
    deck = tmp_path / "f_0000.rad"
    deck.write_text("/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n")
    log, model = MessageLog(), Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == "PROP":
            read_prop(b, model, log)
    p = model.properties[1]
    assert p.params["qa"] == pytest.approx(1.1)
    assert p.params["qb"] == pytest.approx(0.05)
    assert p.params["h"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Part B — end-to-end energy canary: a single brick / tetra under an
# impulsive /IMPVEL start must stay energy-balanced (was UNSTABLE).
# ---------------------------------------------------------------------------

_LAW1 = "/MAT/LAW1/1\nsteel\n7.8e-6\n210. 0.3\n"
# const-1.0 velocity curve: the driven face jumps 0 -> 1 at cycle 1 (the
# impulsive start that detonated the solids with the misread negative h)
_FUNCT = "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"


def _brick_starter(hg="0.1"):
    # unit cube; bottom face (1-4) fully fixed, top face (5-8) driven in +z.
    # The fixed-bottom / free-top-lateral asymmetry seeds the hourglass
    # modes, so a negative h detonates and a correct h stays quiet.
    return (
        "/BEGIN\nmini brick impvel\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\nbrick\n1 1\n" + _LAW1 +
        f"/PROP/SOLID/1\nsolid\n1.1 0.05 {hg}\n"
        "/GRNOD/NODE/1\nbottom\n1 2 3 4\n"
        "/GRNOD/NODE/2\ntop\n5 6 7 8\n"
        "/BCS/1\nfix bottom\n111 111 0 1\n"
        "/IMPVEL/1\npush z\n1 Z 2 1.0\n" + _FUNCT + "/END\n")


def _tetra_starter():
    return (
        "/BEGIN\nmini tetra impvel\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
        "/TETRA4/1\n1 1 2 3 4\n"
        "/PART/1\ntet\n1 1\n" + _LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nbase\n1 2 3\n"
        "/GRNOD/NODE/2\napex\n4\n"
        "/BCS/1\nfix base\n111 111 0 1\n"
        "/IMPVEL/1\npush z\n1 Z 2 1.0\n" + _FUNCT + "/END\n")


_ENGINE = "/RUN/{n}/1\n0.02\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n1000.0\n"


def _run(make_deck, name, starter, engine):
    """Run starter+engine; return (model, [ERROR% per printed cycle],
    [HE per printed cycle])."""
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    errs, he = [], []
    out_path = os.path.splitext(e)[0] + ".out"
    for line in open(out_path):
        tok = line.split()
        if len(tok) == 10 and tok[0].isdigit():
            try:
                errs.append(float(tok[9]))
                he.append(float(tok[5]))          # ENERGY-HE column
            except ValueError:
                pass
    return model, errs, he


def test_brick_impvel_energy_balanced(make_deck):
    """A single brick under an impulsive /IMPVEL start runs to the end with
    ~0 energy error and NO numerical-injection abort (was UNSTABLE within a
    handful of cycles: HE negative, ERRN < -30 %)."""
    model, errs, he = _run(make_deck, "MBRK", _brick_starter(),
                           _ENGINE.format(n="MBRK"))
    assert errs, "no ledger lines parsed"
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.t == pytest.approx(0.02, rel=1e-6)
    # every printed cycle balanced (the canary: no energy injection)
    assert max(abs(x) for x in errs) < 0.5, errs
    # hourglass energy must stay >= ~0 (a NEGATIVE HE is the amplifier
    # signature of the misread h=-1); allow round-off slack
    assert min(he) > -1e-6 * max(1.0, max(abs(h) for h in he)), min(he)


def test_tetra_impvel_energy_balanced(make_deck):
    """A single tetra (no hourglass, but it inherited the misread qa=18) is
    likewise energy-balanced to the end under the impulsive /IMPVEL."""
    model, errs, he = _run(make_deck, "MTET", _tetra_starter(),
                           _ENGINE.format(n="MTET"))
    assert errs, "no ledger lines parsed"
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.t == pytest.approx(0.02, rel=1e-6)
    assert max(abs(x) for x in errs) < 0.5, errs


def test_negative_hourglass_would_detonate(make_deck):
    """Teeth for the canary: the SAME brick with a NEGATIVE hourglass
    coefficient (h = -1, exactly what the reader bug produced) DOES blow up
    — the viscous damper becomes an amplifier.  Proves the balanced result
    above is a real stability property, not a silent no-op."""
    model, errs, he = _run(make_deck, "MNEG", _brick_starter(hg="-1.0"),
                           _ENGINE.format(n="MNEG"))
    # it must either trip the injection/energy guard OR drive HE strongly
    # negative — the exact failure signature of the RD-V-0700 solids
    detonated = bool(model.engine_state.stop_reason) or (he and min(he) < -1e-3)
    assert detonated, (model.engine_state.stop_reason, min(he) if he else None)


# ---------------------------------------------------------------------------
# Part C — the M38 shell fix is preserved (shells were already correct)
# ---------------------------------------------------------------------------

def test_shell_impvel_still_balanced(make_deck):
    """A shell under the same impulsive /IMPVEL start stays balanced — the
    M38 midstep constraint-work fix (freed the shells/trias) is untouched by
    the M39 solid-reader fix."""
    starter = (
        "/BEGIN\nmini shell impvel\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nsh\n1 1\n" + _LAW1 +
        "/PROP/SHELL/1\nsh\n0 0 0 0\n0.01 0.01 0.01\n5 0 1.0\n"
        "/GRNOD/NODE/1\nedge0\n1 4\n"
        "/GRNOD/NODE/2\nedge1\n2 3\n"
        "/BCS/1\nfix edge0\n111 111 0 1\n"
        "/IMPVEL/1\npull x\n1 X 2 1.0\n" + _FUNCT + "/END\n")
    model, errs, he = _run(make_deck, "MSHL", starter,
                           _ENGINE.format(n="MSHL"))
    assert errs, "no ledger lines parsed"
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert max(abs(x) for x in errs) < 0.5, errs
