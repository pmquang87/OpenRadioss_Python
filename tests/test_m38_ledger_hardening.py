"""M38 — energy-ledger + hardening forensics (regression tests).

Part A — the RD-V-0700 imposed-velocity CYCLE-1 energy anomaly.
----------------------------------------------------------------
The nine RD-V-0700 SAMP-family decks (Johnson-Cook failure verification:
shells / trias / hexas under /IMPVEL) drive single elements with velocity
curves that are NON-ZERO at t=0 — e.g. FUNCT/1 = exp(t/10), which is 1.0
at t=0, so the driven node jumps 0 -> v_imp in the first cycle. The engine
booked the constraint work as ``J . v_imp`` (the endpoint velocity), which
is TWICE the leapfrog-consistent midstep work ``J . (v_old + v_imp)/2`` at
such an impulsive start. The reference energy is then the over-booked
external work while KE = 1/2 m v_imp^2 sits at exactly half of it — the
**-50.0% cycle-1 energy error** every one of the nine decks reported in the
M37 parity run (``parity_m37.json``: nine NO-CHANNELS cases terminating
"ENERGY ERROR -50.0%"). The original books the same midstep identity in
``engine/source/constraints/general/impvel/fixvel.F`` (the work term
``DW = 1/4 MS (A DT12 + 2 V)(A - AOLD)`` expands to ``1/2 J (v_old+v_imp)``).

The fix (engine/kinematics.py ``apply_kinematic`` + the v_old passed from
engine.py) books the constraint work at the midstep, so an impulsive start
balances to round-off instead of doubling the external work.

Part B / C findings are documented in VALIDATION and the M38 report; the
LAW2 Iflag=1 yield-input conversion is exercised in
``test_law2_iflag1_*`` below.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import (_law2_iflag1_to_abn,
                                              parse_starter_deck)
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

# A truss driven at a CONSTANT imposed velocity of 1.0 from t=0: the driven
# end (node 2) jumps 0 -> 1 in cycle 1 (the impulsive start), the anchor
# (node 1) is fully fixed. FUNCT/1 = 1.0 everywhere.
IMPULSIVE_IMPVEL_STARTER = (
    "/BEGIN\nm38 impulsive impvel probe\n"
    "/NODE\n1 0 0 0\n2 1 0 0\n"
    "/TRUSS/1\n1 1 2\n"
    "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
    "/PROP/TRUSS/1\nbar\n1.0\n"
    "/GRNOD/NODE/1\ndriven\n2\n"
    "/GRNOD/NODE/2\nanchor\n1\n"
    "/BCS/1\nfix anchor\n111 111 0 2\n"
    "/IMPVEL/1\npull x const\n1 X 1 1.0\n"
    "/FUNCT/1\nconst one\n0.0 1.0\n10.0 1.0\n"
    "/END\n"
)


def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _run_listing_errors(make_deck, name, starter, engine):
    """Run starter+engine and return (model, [ERROR% of every printed
    cycle]) parsed from the RunName_0001.out listing."""
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    out_path = os.path.splitext(e)[0] + ".out"
    errs = []
    for line in open(out_path):
        tok = line.split()
        # a ledger line: cycle t dt IE KE HE CE EN EW ERROR%  (10 fields)
        if len(tok) == 10 and tok[0].isdigit():
            try:
                errs.append(float(tok[9]))
            except ValueError:
                pass
    return model, errs


# ---------------------------------------------------------------------------
# Part A — unit level: the midstep constraint-work identity
# ---------------------------------------------------------------------------

def test_impvel_impulsive_start_books_half_the_endpoint_work(make_deck):
    """An impulsive /IMPVEL start (v_old = 0, v_imp = 1) must book the
    leapfrog MIDSTEP work  J . (v_old+v_imp)/2 = 1/2 m v_imp^2 , NOT the
    endpoint work  J . v_imp = m v_imp^2 .  Booking the endpoint (the old
    behaviour) doubles the external work and produces exactly -50% cycle-1
    energy error on the RD-V-0700 decks."""
    model = _starter_only(make_deck, "IMPW", IMPULSIVE_IMPVEL_STARTER)
    loads = LoadsAndConstraints(model, MessageLog())
    i2 = model.node_index(2)
    m = float(model.mass[i2])

    v = np.zeros_like(model.x)          # free velocities (all at rest)
    vr = np.zeros_like(model.x)
    v_old = np.zeros_like(model.x)      # start-of-cycle velocities (rest)
    w = loads.apply_kinematic(0.5, v, vr, model.mass, model.x, 1e-4, v_old)

    assert v[i2, 0] == pytest.approx(1.0)          # velocity enforced
    # the impulsive-start work is 1/2 m v_imp^2, i.e. exactly the KE created
    assert w == pytest.approx(0.5 * m * 1.0 ** 2, rel=1e-12)
    assert w != pytest.approx(m * 1.0 ** 2)        # NOT the endpoint work


def test_impvel_steady_state_booking_unchanged(make_deck):
    """When the node already MOVES at v_imp at the start of the cycle
    (v_old == v_imp) the midstep average equals v_imp, so the booking
    reduces to J . v_imp — the fix is a no-op in steady motion and only
    corrects the transient. Here a force has drifted the free velocity to
    v_imp + d; the constraint impulse is -m d and the work is -m d v_imp."""
    model = _starter_only(make_deck, "IMPS", IMPULSIVE_IMPVEL_STARTER)
    loads = LoadsAndConstraints(model, MessageLog())
    i2 = model.node_index(2)
    m = float(model.mass[i2])

    d = 0.01
    v = np.zeros_like(model.x)
    v[i2, 0] = 1.0 + d                  # free vel drifted above v_imp
    v_old = np.zeros_like(model.x)
    v_old[i2, 0] = 1.0                  # was AT v_imp at cycle start (steady)
    w = loads.apply_kinematic(0.5, v, np.zeros_like(v), model.mass,
                              model.x, 1e-4, v_old)
    # midstep = (1.0 + 1.0)/2 = v_imp -> w = J*v_imp = m*(-d)*1.0
    assert w == pytest.approx(m * (-d) * 1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Part A — full run: cycle-1 energy balance on the impulsive-start deck
# ---------------------------------------------------------------------------

def test_impulsive_impvel_cycle1_energy_balance(make_deck):
    """End-to-end reproduction of the RD-V-0700 anomaly on a minimal
    imposed-velocity deck: the driven end jumps 0 -> 1 at cycle 1. The
    cycle-1 energy error must be ~0 (was -50.0% before the midstep fix),
    and the balance must stay tight for the whole run."""
    engine = "/RUN/MINI/1\n0.002\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n1000.0\n"
    model, errs = _run_listing_errors(make_deck, "MINI",
                                      IMPULSIVE_IMPVEL_STARTER, engine)
    assert errs, "no ledger lines parsed"
    # the decisive assertion: cycle 1 (the impulsive start) is balanced
    assert abs(errs[0]) < 0.05, f"cycle-1 error {errs[0]}% (was -50.0%)"
    # and every subsequent printed cycle stays balanced
    assert max(abs(x) for x in errs) < 0.5
    # sanity: the run reached the requested end time (did not abort early)
    assert model.engine_state.t == pytest.approx(0.002, rel=1e-6)
    assert not model.engine_state.stop_reason


def test_impulsive_impvel_external_work_equals_energy(make_deck):
    """At the impulsive start the external work booked must equal the
    kinetic energy created (1/2 m v_imp^2), not twice it. Checks the
    cycle-1 numbers directly from the exposed engine state."""
    # run a single cycle: t_end just past one step
    engine = "/RUN/ONE1/1\n1.0e-4\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n1000.0\n"
    s, e = make_deck("ONE1", IMPULSIVE_IMPVEL_STARTER, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    st = model.engine_state
    assert st.cycle == 1
    i2 = model.node_index(2)
    ke = 0.5 * float(model.mass[i2]) * float(model.v[i2, 0]) ** 2
    # external work == kinetic energy created (balance closed)
    assert st.wext == pytest.approx(ke, rel=1e-9)


# ---------------------------------------------------------------------------
# Part C — /MAT/LAW2 Iflag=1 (SIG_Y/UTS/EUTS) -> a/b/n conversion
# ---------------------------------------------------------------------------

def _parse(deck_text, tmp_path):
    f = tmp_path / "L2_0000.rad"
    f.write_text(deck_text)
    model, log = Model(), MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
    return model, log


def test_law2_iflag1_conversion_matches_fortran():
    """The Iflag=1 (SIG_Y/UTS/EUTS) -> a/b/n conversion must reproduce the
    Fortran starter's values on the RD-HWX-T-1000 aluminium card
    (SIG_Y=0.09026, UTS=0.175, EUTS=0.24, E=60.4). The reference a/b/n are
    the values the Fortran starter prints ('YIELD COEFFICIENT A/B/N')."""
    log = MessageLog()

    class _Blk:
        user_id = 1
        source = "test"

    a, b, n = _law2_iflag1_to_abn(0.09026, 0.175, 0.24, 60.4, _Blk(), log)
    assert a == pytest.approx(0.09026)
    assert b == pytest.approx(0.2232020270107, rel=1e-9)   # Fortran printout
    assert n == pytest.approx(0.3683065281433, rel=1e-9)   # Fortran printout
    assert not log.warnings
    # the fit satisfies BOTH constraints it is derived from: the flow curve
    # passes through the true-UTS point AND the Considere necking condition
    rm, ag = 0.175 * 1.24, np.log(1.24)
    assert a + b * ag ** n == pytest.approx(rm, rel=1e-9)      # flow curve
    assert b * n * ag ** (n - 1.0) == pytest.approx(rm, rel=1e-9)  # necking


def test_law2_iflag1_fixed_format_deck(tmp_path):
    """The T1000 LAW2 Iflag=1 card in the REAL fixed-format dialect
    (byte-faithful columns, /BEGIN Invers 2024 -> fixed reader) must parse
    to a LAW2 material with the converted a/b/n — no error (M37 blocker
    c33: 'Iflag=1 ... not ported')."""
    deck = (
        "/BEGIN\n"
        "iflag1 tensile\n"
        "      2024         0\n"
        "                  kg                  mm                  ms\n"
        "                  kg                  mm                  ms\n"
        "/MAT/PLAS_JOHNS/1\n"
        "Aluminium\n"
        "2.70000000000000E-06\n"
        "                60.4                0.33         1\n"
        "             0.09026               0.175                0.24"
        "                0.75\n"
        "/END\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.law == 2
    assert mat.params["A"] == pytest.approx(0.09026)
    assert mat.params["B"] == pytest.approx(0.2232020270107, rel=1e-9)
    assert mat.params["n"] == pytest.approx(0.3683065281433, rel=1e-9)
    assert mat.params["eps_p_max"] == pytest.approx(0.75)     # from the card
    assert mat.params["sig_max"] == pytest.approx(1e30)       # blank -> none


def test_law2_iflag0_still_reads_abn_directly(tmp_path):
    """Iflag=0 (or absent) must still read a/b/n literally — the conversion
    only triggers on Iflag=1."""
    deck = ("/BEGIN\nt\n/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n"
            "300 800 0.5 0 0\n/END\n")
    model, log = _parse(deck, tmp_path)
    assert not log.errors
    p = model.materials[1].params
    assert (p["A"], p["B"], p["n"]) == pytest.approx((300.0, 800.0, 0.5))


def test_law2_iflag1_n_gt_one_caps_to_linear():
    """When the fit gives an exponent n>1 the Fortran caps it to 1 (linear
    hardening) with a refit + warning (hm_read_mat02_jc MSGID 277)."""
    log = MessageLog()

    class _Blk:
        user_id = 9
        source = "t"

    # SIG_Y close to UTS with a large EUTS pushes n above 1
    a, b, n = _law2_iflag1_to_abn(0.97, 1.0, 0.3, 210.0, _Blk(), log)
    assert n == pytest.approx(1.0)                    # capped
    assert any("n>1" in str(w) for w in log.warnings)
    # linear refit is positive (real hardening slope)
    assert b > 0.0


def test_law2_chard_kinematic_warns(tmp_path):
    """A non-zero Chard (kinematic hardening) must be surfaced as a warning
    — the port's radial return is purely isotropic (no back-stress)."""
    deck = ("/BEGIN\nt\n/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n"
            "300 800 0.5 0 0\n0.1 1.0 0 0 0 0.5\n/END\n")
    model, log = _parse(deck, tmp_path)
    assert not log.errors
    assert any("Chard" in str(w) and "kinematic" in str(w)
               for w in log.warnings)


def test_law2_chard_zero_is_silent(tmp_path):
    """Chard=0 (pure isotropic, e.g. the c26 Hardening deck) draws no
    kinematic-hardening warning."""
    deck = ("/BEGIN\nt\n/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n"
            "300 800 0.5 0 0\n0.1 1.0 0 0 0 0\n/END\n")
    model, log = _parse(deck, tmp_path)
    assert not any("Chard" in str(w) for w in log.warnings)
