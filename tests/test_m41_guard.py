"""M41 â€” the RD-E-1000 Sf_0.1 energy-guard startup regression (task_29ec1751).

The finding (VALIDATION.md M40 Â§3.5)
-----------------------------------
The three RD-E-1000 Bending Sf_0.1 variants (c40/c42/c44 = BT_type1/3/4)
REGRESSED from M39's ~256 k-cycle runs to **cycle-100 startup aborts** on the
port's âˆ’15 % energy-error guard, against **~1e-8 absolute energies** (IE
3.75e-08 at cycle 100).  These decks load very slowly (/DT/NODA scale 0.1), so
for the opening cycles the total reference energy is numerical dust and a fixed
leapfrog half-step lag between the booked external work and the realized
IE+KE reads as a huge percentage (ERR âˆ’84 % at cycle 1, âˆ’17 % at cycle 100),
with no instability present.  The Fortran engine runs every one to NORMAL.

The upstream conditioning (engine/source/output/ecrit.F)
--------------------------------------------------------
The original computes its ENERGY ERROR only when the reference/expected total
energy clears an ABSOLUTE floor, and otherwise reports zero::

    IF(ABS(ENTOT1B) > EM20) THEN   ERR = ENTOTB/ENTOT1B - ONE
    ELSE                           ERR = ZERO          (the check is inert)

(the printed percentage is additionally clamped to Â±99.9 %, X99), and the stop
thresholds DEMXK/DEMXS default to EP20/EP30 when the deck sets no /STOP limit
(freform.F 782/803) â€” i.e. upstream does not abort on energy error by default
at all.  The port keeps a live 15 % default limit as a young-port divergence
safety net, so M41 mirrors ONLY the denominator conditioning: both percentage
guards are held inert until the reference energy ``REF`` clears an absolute
floor (``engine._ENERGY_START_FLOOR``).  The 15 %/30 % limits are UNCHANGED â€”
the fix conditions the reference, it does not weaken the limit.

What these tests pin
--------------------
* the fix is an ABSOLUTE energy floor, and the %-limit stays 15 % (not
  relaxed);
* a near-zero-energy startup deck runs cleanly past the cycle-100 check (the
  c40/c42/c44 behaviour), and WITHOUT the floor the same deck aborts there â€”
  so the conditioning is load-bearing, not a silent no-op;
* the guard RE-ARMS once the reference energy is real â€” a genuine imbalance
  above the floor still trips it (the fix conditions, it does not disable);
* the M39-era ``h = -1`` hourglass 'teeth' negative control still detonates
  under the default 15 % limit â€” a real instability is still caught.

The near-zero-energy reproducer
-------------------------------
A single flat shell whose four nodes are driven in RIGID DRILLING rotation
(Dir=ZZ, about the shell normal) by a slowly-ramped /IMPVEL.  The drive books
external work into the nodes' ROTATIONAL kinetic energy, which the balance's
KE term (translational Â½Â·mÂ·vÂ² only â€” exactly like the Fortran ENCIN, whose
rotational ENROT is a separate channel) does not count, so the run carries a
large NEGATIVE %-error (ERR â†’ âˆ’100 %) at near-zero ABSOLUTE energy: the
RD-E-1000 Sf_0.1 startup pathology in miniature.  The rigid rotation strains
nothing, so IE == KE == 0 and the reference energy REF equals the booked
external work exactly, which lets the tests reason about REF straight from the
listing's EXT-WORK column.
"""

import contextlib
import io
import os

import pytest

import pyradioss.engine.engine as eng
from pyradioss.engine.engine import run_engine
from pyradioss.model.model import EngineControls
from pyradioss.starter.starter import run_starter

# The production floor, captured at import so it survives monkeypatching of the
# live module attribute (the guard reads eng._ENERGY_START_FLOOR at runtime).
PROD_FLOOR = eng._ENERGY_START_FLOOR

STEEL_LAW1 = "/MAT/LAW1/1\nsteel elastic\n7.8e-6\n210. 0.3\n"

# Ramp scale for the drilling drive: sized so the booked external work clears
# the balance's internal 1e-12 REF floor (so ERR is a true −100 %, not a
# rounded 0) within the first cycles while still sitting far below the
# _ENERGY_START_FLOOR dust threshold — REF ≈ 1.9e-7 at cycle 100, crossing
# 1e-6 only near cycle 900.
_DRILL_FSCALE = 300.0


def _drill_shell_starter(fscale=_DRILL_FSCALE):
    """A single shell rigidly spun about its normal with fixed ZZ rotation:
    external work booked by /IMPVEL is zeroed by /BCS → uncounted rotational KE
    → large %-error at dust energy (simulating the startup leapfrog lag)."""
    return (
        "/BEGIN\nm41 drill shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nsh\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nsh\n0 0 0 0\n0.01 0.01 0.01\n5 0 1.0\n"
        "/GRNOD/NODE/1\nall4\n1 2 3 4\n"
        f"/IMPVEL/1\ndrill spin\n5 ZZ 0 {fscale}\n"
        "/BCS/1\nfix drill\n000 001 0 1\n"
        # linear velocity ramp 0 -> 1 over t=500 (slow loading), then flat
        "/FUNCT/5\nslow ramp\n0.0 0.0\n500.0 1.0\n500000.0 1.0\n"
        "/END\n")


def _teeth_brick_starter(hg):
    """The M39 negative control: a unit brick under an impulsive /IMPVEL whose
    NEGATIVE hourglass coefficient (h = âˆ’1) turns the Flanaganâ€“Belytschko
    viscous damper into an amplifier â€” a genuinely unstable run whose energy
    runs away super-linearly (see test_m39_solid_impvel)."""
    return (
        "/BEGIN\nm41 teeth\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\nbrick\n1 1\n" + STEEL_LAW1 +
        f"/PROP/SOLID/1\nsolid\n1.1 0.05 {hg}\n"
        "/GRNOD/NODE/1\nbottom\n1 2 3 4\n/GRNOD/NODE/2\ntop\n5 6 7 8\n"
        "/BCS/1\nfix bottom\n111 111 0 1\n"
        "/IMPVEL/1\npush z\n1 Z 2 1.0\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        "/END\n")


def _run(make_deck, name, starter, engine):
    """Run starter+engine; return (model, [ledger-row dicts from the .out])."""
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    rows = []
    out = os.path.splitext(e)[0] + ".out"
    for line in open(out):
        tok = line.split()
        if len(tok) == 10 and tok[0].isdigit():
            try:
                rows.append({"cyc": int(tok[0]), "t": float(tok[1]),
                             "IE": float(tok[3]), "KE": float(tok[4]),
                             "EW": float(tok[8]), "ERR": float(tok[9])})
            except ValueError:
                pass
    return model, rows


def _ref_of(row):
    """The balance reference REF the guards normalize by, reconstructed from a
    listing row â€” the max of the energy magnitudes (see engine._energies).
    For the rigid-drill deck IE == KE == 0, so REF == |EW| exactly."""
    return max(abs(row["EW"]), row["KE"], row["IE"], 1e-12)


# ---------------------------------------------------------------------------
# The fix is an absolute floor, and the limit is NOT weakened
# ---------------------------------------------------------------------------

def test_floor_is_an_absolute_energy_not_a_weakened_percentage():
    """The conditioning is an ABSOLUTE energy floor on the reference (mirroring
    ecrit.F's ABS(ENTOT1B) > EM20 gate), NOT a relaxed %-limit.  Pin both: the
    floor is a small positive absolute energy, and the default divergence limit
    is UNCHANGED at 15 % (task_29ec1751: the limit must not be weakened)."""
    assert isinstance(PROD_FLOOR, float)
    assert 0.0 < PROD_FLOOR < 1.0e-3          # a dust-band absolute floor
    assert EngineControls().energy_error_stop == 15.0   # limit unchanged


# ---------------------------------------------------------------------------
# The near-zero-energy startup deck now passes (the c40/c42/c44 fix)
# ---------------------------------------------------------------------------

def test_near_zero_energy_startup_runs_past_cycle_100(make_deck):
    """THE FIX: a slowly-loaded deck whose startup energy is numerical dust
    (REF < the floor, |ERR| â‰ˆ 100 % on that dust) now runs cleanly past the
    cycle-100 guard check that aborted RD-E-1000 Sf_0.1 c40/c42/c44.  Truncated
    (t_end = 0.1) to stay within the dust band."""
    model, rows = _run(make_deck, "MDRILL", _drill_shell_starter(),
                       "/RUN/MDRILL/1\n0.1\n/DT\n0.9 0\n/PRINT/-100\n")
    assert rows, "no ledger lines parsed"
    # ran clean to the (truncated) end â€” no guard abort during startup
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.t == pytest.approx(0.1, rel=1e-6)
    # got past the cycle-100 check where the regression aborted
    assert model.engine_state.cycle > 100
    # every printed cycle is genuinely near-zero energy (dust) yet carries a
    # large %-error: exactly the ill-posed regime the floor conditions out
    for r in rows:
        assert _ref_of(r) < PROD_FLOOR, r
    assert min(r["ERR"] for r in rows) < -50.0     # the huge false %-error


def test_without_the_floor_the_same_deck_aborts_at_cycle_100(make_deck,
                                                             monkeypatch):
    """Teeth for the fix: with the floor disabled (floor = 0, the pre-M41
    behaviour) the SAME startup deck aborts at the first guard check (cycle
    100) on ~1e-9 absolute energy â€” proving the deck genuinely exercises the
    regression and the conditioning is load-bearing, not a silent no-op."""
    monkeypatch.setattr(eng, "_ENERGY_START_FLOOR", 0.0)
    model, rows = _run(make_deck, "MDRILL0", _drill_shell_starter(),
                       "/RUN/MDRILL0/1\n0.1\n/DT\n0.9 0\n/PRINT/-100\n")
    assert "ENERGY ERROR" in model.engine_state.stop_reason
    assert model.engine_state.cycle == 100          # the first guard check
    # the abort fired on dust â€” the near-zero denominator the fix conditions out
    assert _ref_of(rows[-1]) < PROD_FLOOR, rows[-1]


# ---------------------------------------------------------------------------
# The guard is CONDITIONED, not disabled â€” it re-arms above the floor
# ---------------------------------------------------------------------------

def test_guard_rearms_once_reference_energy_clears_the_floor(make_deck):
    """The fix CONDITIONS the guard, it does not DISABLE it: run the same
    persistent-imbalance deck long enough that the reference energy grows past
    the floor, and the energy-error guard fires again â€” a real (non-dust)
    imbalance is still caught.  (Mirrors the real c40/c41: once IE reaches ~1e5
    the port's injection guard correctly trips on the excited hourglass mode.)
    The abort must land past the startup dust band, never at cycle 100."""
    model, rows = _run(make_deck, "MREARM", _drill_shell_starter(),
                       "/RUN/MREARM/1\n1.0\n/DT\n0.9 0\n/PRINT/-100\n")
    assert "ENERGY ERROR" in model.engine_state.stop_reason
    # it did NOT abort in the dust band â€” it survived startup and only tripped
    # once the reference energy was real (the guard only fires when REF > floor)
    assert model.engine_state.cycle > 100
    assert _ref_of(rows[-1]) >= PROD_FLOOR, rows[-1]   # the abort is on REAL energy


# ---------------------------------------------------------------------------
# The genuine-instability negative control still detonates
# ---------------------------------------------------------------------------

def test_teeth_negative_control_still_detonates(make_deck):
    """The M39-era negative control at the guard level: the h = âˆ’1 hourglass
    'teeth' brick (a genuine instability) must STILL trip the divergence guard
    under the default 15 % limit â€” the floor conditioning must not blind the
    guard to real energy injection.  (REF crosses the floor at cycle 1; the
    numerical-injection guard fires by cycle 3 at ERRN â‰ˆ âˆ’50 %.)"""
    model, _ = _run(make_deck, "MTEETH", _teeth_brick_starter("-1.0"),
                    "/RUN/MTEETH/1\n0.02\n/DT\n0.9 0\n/PRINT/-1\n")
    stop = model.engine_state.stop_reason
    assert stop, "teeth negative control did not detonate"
    # an ENERGY guard caught it (not merely a dt collapse) â€” the guard is live
    # above the floor
    assert "ENERGY" in stop, stop
    assert model.engine_state.t < 0.02             # stopped early, before t_end


def test_teeth_positive_control_stays_balanced(make_deck):
    """Symmetric control: the SAME brick with the CORRECT positive hourglass
    coefficient (h = +0.1) runs to the end with no guard abort â€” the floor
    conditioning does not silence a healthy run either."""
    model, rows = _run(make_deck, "MPOS", _teeth_brick_starter("0.1"),
                       "/RUN/MPOS/1\n0.02\n/DT\n0.9 0\n/PRINT/-100\n")
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.t == pytest.approx(0.02, rel=1e-6)
    assert max(abs(r["ERR"]) for r in rows) < 0.5      # balanced throughout





# ---------------------------------------------------------------------------
# M67: NAN/INF divergence backstop tests KE, IE, and HE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nan_field", ["KE", "IE", "HE"])
def test_nan_inf_divergence_backstop_catches_all_energy_channels(make_deck, monkeypatch, nan_field):
    """The unconditional NAN/INF backstop must catch NaN in any of the primary
    energy channels (KE, IE, HE), not just KE."""
    
    orig_energies = eng._energies
    
    def mocked_energies(model, state):
        e = orig_energies(model, state)
        if state.cycle >= 1:
            e[nan_field] = float("nan")
        return e
        
    monkeypatch.setattr(eng, "_energies", mocked_energies)
    
    # Run the same stable starter deck as test_teeth_positive_control_stays_balanced
    model, rows = _run(make_deck, f"MNAN_{nan_field}", _teeth_brick_starter("0.1"),
                       f"/RUN/MNAN_{nan_field}/1\n0.02\n/DT\n0.9 0\n/PRINT/-100\n")
    
    stop = model.engine_state.stop_reason
    assert stop and "NAN/INF DETECTED" in stop, f"Did not catch NaN in {nan_field}"
    assert model.engine_state.cycle <= 2


#
# ---------------------------------------------------------------------------
# M67: NAN/INF divergence backstop tests KE, IE, and HE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nan_field", ["KE", "IE", "HE"])
def test_nan_inf_divergence_backstop_catches_all_energy_channels(make_deck, monkeypatch, nan_field):
    """The unconditional NAN/INF backstop must catch NaN in any of the primary
    energy channels (KE, IE, HE), not just KE."""
    
    orig_energies = eng._energies
    
    def mocked_energies(model, state):
        e = orig_energies(model, state)
        if state.cycle >= 1:
            e[nan_field] = float("nan")
        return e
        
    monkeypatch.setattr(eng, "_energies", mocked_energies)
    
    # Run the same stable starter deck, but print every cycle so the guard is
    # evaluated immediately when the NaN is injected at cycle 1.
    model, rows = _run(make_deck, f"MNAN_{nan_field}", _teeth_brick_starter("0.1"),
                       f"/RUN/MNAN_{nan_field}/1\n0.02\n/DT\n0.9 0\n/PRINT/-1\n")
    
    stop = model.engine_state.stop_reason
    assert stop and "NAN/INF DETECTED" in stop, f"Did not catch NaN in {nan_field}"
    assert model.engine_state.cycle <= 2
