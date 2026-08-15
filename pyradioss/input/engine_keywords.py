"""
Engine file parsers: ``*_0001.rad`` → EngineControls.

Fortran origin: the Engine's own reader (``engine/source/input/lectur.F``
and friends, storing into ``dt_mod``, ``output_mod``...). The engine file
is small: it only *controls* the run (final time, time-step options, output
frequencies) — the model itself comes from the Starter restart file.

Supported cards (layouts documented per parser below)::

    /RUN/RunName/run#          card: T_stop  (run# >= 2 resumes the
                               previous run's restart — M6 chaining)
    /VERS/...                  ignored (input version)
    /TFILE                     card: dT_history
    /ANIM/DT                   card: T_start  dT_anim
    /ANIM/VECT/<VEL|DIS|ACC>   request nodal vector in animation files
    /ANIM/ELEM/<VONM|EPSP>     request element scalar in animation files
    /DT                        card: Scale  [dT_min]
    /DT/NODA                   card: Scale  [dT_min]  — nodal time step
    /DT/NODA/CST               card: Scale  dT_min    — mass scaling (M6)
    /STATE/DT                  card: T_start  dT — restart snapshots (M6)
    /PRINT/-n                  listing line every n cycles
    /STOP                      card: E_error_max_%   (energy error stop)
"""

from __future__ import annotations

from typing import List

from ..common.messages import MessageLog
from ..model.model import EngineControls
from .deck_reader import KeywordBlock


def _parse_multi_input_table(block, ec, base_line, log, card):
    """Read the M28 MULTI-INPUT table off a /IMPL/PSD/MULTI or /IMPL/FATIG/MINPUT
    card, starting at card line ``base_line``:

        line base_line     : ninput  cohmodel  gamma  phase  [decay  speed
                              [gamma1  phase1  [decay1  speed1  gfunct0  gfunct1]]]
        lines base_line+1..: cload_funct  psd_funct  [x  y  z]   (one per input)

    Columns 6-7 (gamma1, phase1) are the M29 SCALAR-coherence END pair; columns 8-11
    (decay1, speed1, gfunct0, gfunct1) are the M30 FREQUENCY-DEPENDENT-evolutionary
    schedule (a time-varying exponential decay / speed, or a per-pair measured
    coherence-shape /FUNCT start -> end).

    ``cohmodel`` = 0 constant coherence gamma_ab = gamma (phase theta_ab = phase
    degrees on every off-diagonal pair), 1 exponential/decay coherence (needs the
    per-input positions x y z, decay coefficient and reference speed). Each input
    row names the /CLOAD /FUNCT id that identifies its spatial load pattern and the
    auto-PSD /FUNCT id giving its G_a(f). Fills ``ec.impl_mi_*``. Returns True on a
    usable table (>= 1 input)."""
    cards = block.cards
    if len(cards) <= base_line:
        log.warning(f"{card}: no MULTI-INPUT table found (expected a header "
                    "'ninput cohmodel gamma phase' line then one row per input: "
                    "cload_funct psd_funct [x y z])", block.source)
        return False
    hdr = cards[base_line].floats()
    ninput = int(hdr[0]) if len(hdr) > 0 and hdr[0] > 0 else 0
    ec.impl_mi_cohmodel = int(hdr[1]) if len(hdr) > 1 and hdr[1] >= 0 else 0
    ec.impl_mi_gamma = float(hdr[2]) if len(hdr) > 2 else 0.0
    ec.impl_mi_phase = float(hdr[3]) if len(hdr) > 3 else 0.0
    ec.impl_mi_decay = float(hdr[4]) if len(hdr) > 4 else 0.0
    ec.impl_mi_speed = float(hdr[5]) if len(hdr) > 5 and hdr[5] > 0 else 1.0
    # M29 EVOLUTIONARY-COHERENCE schedule (optional): the END coherence gamma_ab(t_1)
    # and phase theta_ab(t_1) the coherence DRIFTS to across the /EVOL windows
    # (interpolated from the start pair gamma / phase above). Columns 7-8 of the
    # header: gamma1 [phase1]. A negative (or absent) gamma1 means NO coherence
    # drift (the M28 stationary special case) — the driver then holds gamma_ab fixed.
    ec.impl_mi_gamma1 = float(hdr[6]) if len(hdr) > 6 else -1.0
    ec.impl_mi_phase1 = float(hdr[7]) if len(hdr) > 7 else ec.impl_mi_phase
    # M30 FREQUENCY-DEPENDENT-EVOLUTIONARY-COHERENCE schedule (optional): the
    # coherence is a FULL gamma_ab(f) frequency shape that ALSO drifts window to
    # window. Two schedules share the header's trailing columns:
    #   * the EXPONENTIAL / convection model (cohmodel = 1) with a TIME-VARYING decay
    #     coefficient decay1 (col 8) and reference speed speed1 (col 9) the field
    #     DRIFTS to (a decorrelation frequency that moves through the mission); a
    #     negative (or absent) decay1 / speed1 means NO drift of that quantity;
    #   * a per-pair MEASURED coherence SHAPE gamma(f) as a /FUNCT id: gfunct0 (col
    #     10) the START shape, gfunct1 (col 11) the END shape (defaults to gfunct0 —
    #     a stationary frequency-dependent coherence, the M28 special case).
    ec.impl_mi_decay1 = float(hdr[8]) if len(hdr) > 8 else -1.0
    ec.impl_mi_speed1 = float(hdr[9]) if len(hdr) > 9 else -1.0
    ec.impl_mi_gfunct0 = int(hdr[10]) if len(hdr) > 10 and hdr[10] > 0 else 0
    ec.impl_mi_gfunct1 = (int(hdr[11]) if len(hdr) > 11 and hdr[11] > 0
                          else ec.impl_mi_gfunct0)
    inputs = []
    for r in range(ninput):
        li = base_line + 1 + r
        if li >= len(cards):
            break
        v = cards[li].floats()
        if len(v) < 2:
            continue
        cf = int(v[0]); pf = int(v[1])
        x = float(v[2]) if len(v) > 2 else 0.0
        y = float(v[3]) if len(v) > 3 else 0.0
        z = float(v[4]) if len(v) > 4 else 0.0
        inputs.append((cf, pf, x, y, z))
    ec.impl_mi_inputs = tuple(inputs)
    if len(inputs) < 1:
        log.warning(f"{card}: MULTI-INPUT table has no usable input rows "
                    "(each row: cload_funct psd_funct [x y z])", block.source)
        return False
    return True


def parse_engine_deck(blocks: List[KeywordBlock],
                      log: MessageLog) -> EngineControls:
    ec = EngineControls()
    for block in blocks:
        key = block.key0
        try:
            if key == "RUN":
                # /RUN/RunName/run_number — the run name defines output file
                # names; card 1 = final time T_stop.
                if len(block.parts) >= 2:
                    ec.run_name = block.parts[1]
                if block.cards:
                    ec.t_end = block.cards[0].floats()[0]
                else:
                    log.error("/RUN: missing T_stop card", block.source)
            elif key == "VERS":
                pass  # input version — irrelevant to the port
            elif key == "TFILE":
                if block.cards:
                    ec.th_dt = block.cards[0].floats()[0]
            elif key == "ANIM":
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "DT":
                    vals = block.cards[0].floats() if block.cards else [0.0]
                    # card: T_start dT  (a single value is taken as dT)
                    ec.anim_dt = vals[1] if len(vals) > 1 else vals[0]
                elif sub == "VECT" and len(block.parts) > 2:
                    v = block.parts[2].upper()
                    if v not in ec.anim_vect:
                        ec.anim_vect.append(v)
                elif sub == "ELEM" and len(block.parts) > 2:
                    v = block.parts[2].upper()
                    if v not in ec.anim_elem:
                        ec.anim_elem.append(v)
                else:
                    log.warning(f"/ANIM/{sub} not ported", block.source)
            elif key == "DT":
                # /DT: card = Scale [dT_min]. The scale multiplies the
                # critical time step (default 0.9); if dt falls below
                # dT_min the Engine stops.
                #
                # /DT/NODA (M6): the step is bounded by the NODAL time
                # step dt_i = sqrt(2 M_i / K_i) instead of the worst
                # element (see engine/mass_scaling.py).
                # /DT/NODA/CST (M6): additionally, mass is ADDED to the
                # critical nodes so the step never falls below dT_min
                # (mass scaling — the added mass and its momentum/energy
                # effect are tracked and reported).
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "NODA":
                    ec.dt_noda = ("CST" if len(block.parts) > 2 and
                                  block.parts[2].upper() == "CST"
                                  else "NODA")
                elif sub == "AMS":
                    ec.dt_ams = True
                    if len(block.parts) > 2:
                        try:
                            ec.dt_ams_igrp = int(block.parts[2])
                        except ValueError:
                            pass
                if block.cards:
                    vals = block.cards[0].floats()
                    if vals:
                        # a zero/blank scale means the DEFAULT 0.9 (the
                        # reference reads dTsca=0 as 'use default' — the
                        # RD-V-0220 oracle deck writes '0.0 1e-7'), and a
                        # zero scale would divide the /DT/NODA/CST mass
                        # target by zero (M37)
                        ec.dt_scale = vals[0] if vals[0] > 0.0 else 0.9
                    if len(vals) > 1:
                        ec.dt_min = vals[1]
                        
                    # Advanced Mass Scaling parameters
                    if ec.dt_ams:
                        if len(block.cards) > 1:
                            v1 = block.cards[1].floats()
                            if v1:
                                ec.dt_ams_tol = v1[0]
                        if len(block.cards) > 2:
                            v2 = block.cards[2].floats()
                            if v2:
                                ec.dt_ams_itmax = int(v2[0])
                if ec.dt_noda == "CST" and ec.dt_min <= 0.0:
                    log.warning("/DT/NODA/CST without a positive dT_min "
                                "adds no mass", block.source)
            elif key == "STATE":
                # /STATE/DT (M6): card = T_start dT — periodic full
                # restart snapshots (each refreshes RunName_{nn}.rst; the
                # port's equivalent of the original's /STATE state files
                # + /RFILE restart cadence). Independently of /STATE, the
                # Engine ALWAYS writes the restart at termination — that
                # is the RunName_{nn+1}.rad chaining contract.
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub != "DT":
                    log.warning(f"/STATE/{sub} not ported (DT supported)",
                                block.source)
                elif block.cards:
                    vals = block.cards[0].floats()
                    ec.state_tstart = vals[0] if vals else 0.0
                    ec.state_dt = vals[1] if len(vals) > 1 else \
                        (vals[0] if vals else 0.0)
            elif key == "IMPL":
                # /IMPL (M8): switch the run into implicit-static mode. The
                # port mirrors the OpenRadioss /IMPL cards minimally through
                # a small set of sub-cards; the run's final "time" (/RUN
                # T_stop) is reinterpreted as the FINAL LOAD FACTOR.
                #
                #   /IMPL                 (bare) implicit on, defaults
                #   /IMPL/DTINI  card: dt_incr          load-factor increment
                #   /IMPL/NEWTON card: tol  max_iter    Newton controls
                #   /IMPL/LSOLVER/<superlu|cholmod|mumps>   direct solver
                #   /IMPL/NONLIN[/N]      (M9) NONLINEAR GEOMETRY: updated-
                #                         Lagrangian frame + geometric
                #                         stiffness K_geo — large
                #                         displacement, the original
                #                         /IMPL/NONLIN's default behaviour.
                #                         /IMPL/NONLIN/SMDISP keeps the
                #                         small-displacement (M8) path,
                #                         mirroring the original's SMDISP.
                #                         A numeric /N (the original's
                #                         nonlinear-solver strategy pick) is
                #                         accepted and ignored — the port
                #                         always runs full Newton.
                #   /IMPL/ARCL   card: dl  max_inc  it_des
                #                         (M9) Crisfield arc-length
                #                         continuation for limit points /
                #                         snap-through (implies NONLIN).
                #                         All three optional: initial radius
                #                         (default: from the first predictor
                #                         at the DTINI increment), increment
                #                         cap, target iterations/increment.
                #   /IMPL/DYNA/1 card: alpha
                #                         (M10) implicit DYNAMICS, HHT-alpha
                #                         time integration. The card value is
                #                         the HHT alpha ITSELF (the original
                #                         reads HHT_A in freimpl.F — NOT a
                #                         spectral radius): 0 = the
                #                         trapezoidal rule, -1/3 <= alpha < 0
                #                         adds high-frequency dissipation
                #                         (rho_inf = (1+a)/(1-a)). Newmark
                #                         gamma/beta follow as 1/2 - a and
                #                         (1-a)^2/4 (imp_dyna.F DYNA_INI).
                #   /IMPL/DYNA/2 card: gamma  beta
                #                         (M10) implicit DYNAMICS, plain
                #                         Newmark with gamma/beta given
                #                         directly IN THAT ORDER (the
                #                         original's NM_A -> DY_G = gamma,
                #                         NM_B -> DY_B = beta). Defaults
                #                         0.5 / 0.25 = the unconditionally
                #                         stable trapezoidal rule.
                #   /IMPL/DYNA            (bare) = /IMPL/DYNA/2 defaults
                #                         (trapezoidal; a convenience the
                #                         original does not spell — its
                #                         reader requires the /1 or /2).
                #   /IMPL/DYNA/DAMP card: a  b
                #                         (M11) RAYLEIGH damping in the
                #                         implicit system: C = a*M + b*K
                #                         (freimpl.F reads DAMPA_IMP then
                #                         DAMPB_IMP; the card IMPLIES
                #                         dynamics — IF (IDYNA==0) IDYNA=1
                #                         in the source). Damping force,
                #                         effective-tangent term and the
                #                         dissipation ledger live in
                #                         implicit/dynamics.py.
                #   /IMPL/DT/STOP card: dt_min  dt_max
                #   /IMPL/DT/1    card: it_w  sc_up  it_dn  sc_dn
                #                         (M11) AUTOMATIC step control
                #                         (imp_dt.F IMP_DTN, IDTC = 1): on
                #                         non-convergence the step is CUT by
                #                         sc_dn and retried (never below
                #                         dt_min); after a step converging
                #                         in <= it_w iterations it GROWS by
                #                         sc_up back toward dt_max (default:
                #                         the /IMPL/DTINI value). The
                #                         control is ON by default with the
                #                         port defaults (statics.py) — the
                #                         cards only tune it, mirroring the
                #                         original where the cut happens
                #                         regardless of /IMPL/DT. it_dn is
                #                         accepted and unused (the original
                #                         reads NL_DTN for its IDTC = 2/3
                #                         arc-length variants — deferred).
                #   /IMPL/BUCKL/1|2 card: Emin  Emax  Nmode ...
                #                         (M11) linearized BUCKLING
                #                         extraction (imp_buck.F) after the
                #                         static prestress increments: the
                #                         (K_mat + mu K_geo) phi = 0
                #                         eigensolve of implicit/buckling.py
                #                         reported in the listing and on the
                #                         result object. Nmode = NBUCK; the
                #                         Emin/Emax search range and the
                #                         ARPACK controls (MSGL, MAXSET,
                #                         SHIFT) are accepted and unused
                #                         (dense eigh at this port's model
                #                         sizes). A bare /IMPL/BUCKL errors
                #                         like the original ("OBSOLETE, USE
                #                         /IMPL/BUCKL/1 OR /2").
                #
                # With /IMPL/DYNA the /RUN "time" is PHYSICAL TIME again and
                # /IMPL/DTINI is the physical time step (statics reinterprets
                # them as load factor / increment).
                #
                # Unknown sub-cards warn and are skipped, exactly like the
                # rest of the reader.
                ec.implicit = True
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub in ("", "STATIC", "STAT", "LINEAR", "L"):
                    if block.cards:  # a lone /IMPL card may carry dt_incr
                        vals = block.cards[0].floats()
                        if vals:
                            ec.impl_dt = vals[0]
                elif sub in ("NONLIN", "NLGEOM", "NL"):
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    # /IMPL/NONLIN/SMDISP = the original's explicit small-
                    # displacement restriction: exactly the M8 linear path
                    ec.impl_nlgeom = sub2 != "SMDISP"
                    if block.cards:  # may carry dt_incr like the bare card
                        vals = block.cards[0].floats()
                        if vals:
                            ec.impl_dt = vals[0]
                elif sub in ("ARCL", "ARC", "RIKS"):
                    ec.impl_arc = True
                    ec.impl_nlgeom = True
                    if block.cards:
                        vals = block.cards[0].floats()
                        if vals:
                            ec.impl_arc_dl = vals[0]
                        if len(vals) > 1 and vals[1] > 0:
                            ec.impl_arc_maxinc = int(vals[1])
                        if len(vals) > 2 and vals[2] > 0:
                            ec.impl_arc_itdes = int(vals[2])
                elif sub == "DTINI":
                    if block.cards:
                        vals = block.cards[0].floats()
                        if vals:
                            ec.impl_dt = vals[0]
                elif sub == "DT":
                    # /IMPL/DT/STOP (dt_min dt_max) and /IMPL/DT/1 (the
                    # IDTC = 1 iteration-count control card: it_w sc_up
                    # it_dn sc_dn) — M11, imp_dt.F. The control itself is
                    # always on; these tune it. IDTC 2/3 deferred.
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    vals = block.cards[0].floats() if block.cards else []
                    if sub2 == "STOP":
                        if vals:
                            ec.impl_dt_min = vals[0]
                        if len(vals) > 1:
                            ec.impl_dt_max = vals[1]
                    elif sub2 == "1":
                        if vals and vals[0] > 0:
                            ec.impl_dt_itw = int(vals[0])
                        if len(vals) > 1 and vals[1] > 1.0:
                            ec.impl_dt_scaleup = vals[1]
                        # vals[2] = it_dn: read by the original for its
                        # IDTC 2/3 variants — accepted, unused here
                        if len(vals) > 3 and 0.0 < vals[3] < 1.0:
                            ec.impl_dt_scaledn = vals[3]
                    elif sub2 == "FIXP":
                        fixp_vals = []
                        for card in block.cards:
                            fixp_vals.extend(card.floats())
                        if len(fixp_vals) > 100:
                            log.warning(f"/IMPL/DT/FIXP/{block.user_id} maximum "
                                        f"100 fix points permitted", block.source)
                            fixp_vals = fixp_vals[:100]
                        # Fortran reads into DTIMPF and then calls ORDER_DTF
                        ec.impl_dt_fixp = sorted(fixp_vals)
                    else:
                        log.warning(f"/IMPL/DT/{sub2} not ported — ignored "
                                    f"(supports STOP, 1, FIXP; the IDTC 2/3 "
                                    f"arc-length step controls are "
                                    f"deferred)", block.source)
                elif sub == "BUCKL":
                    # /IMPL/BUCKL/n (M11, imp_buck.F): n = 1 after a linear
                    # run, 2 after a nonlinear one — the port runs the same
                    # eigensolve on the converged prestressed state either
                    # way. A bare /IMPL/BUCKL is OBSOLETE in the original
                    # and errors the same way here.
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    if sub2 not in ("1", "2"):
                        log.error(
                            "/IMPL/BUCKL is obsolete — use /IMPL/BUCKL/1 "
                            "or /IMPL/BUCKL/2 (mirroring the original "
                            "reader's check)", block.source)
                    else:
                        ec.impl_buckl = int(sub2)
                        vals = block.cards[0].floats() if block.cards else []
                        # card: EMIN_B EMAX_B NBUCK MSGL MAXSET SHIFT — only
                        # NBUCK drives the dense eigensolve
                        if len(vals) > 2 and vals[2] > 0:
                            ec.impl_buckl_nmode = int(vals[2])
                elif sub in ("EIGV", "EIG"):
                    # /IMPL/EIGV (M16 — a PORT card; freimpl.F has no modal
                    # branch, so this drives the consistent-mass eigensolver
                    # of implicit/modal.py the way /IMPL/BUCKL drives
                    # buckling.py). Optional /STRS suffix requests the
                    # PRESTRESSED spectrum K = K_mat + K_geo (implies the
                    # static prestress increments have run). Card: Nmode
                    # (number of natural frequencies) — the search-range /
                    # Lanczos controls of the commercial card are accepted
                    # and unused (dense eigh, see modal.py).
                    ec.impl_eigv = True
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    if sub2 in ("STRS", "STRESS", "PRESTRESS"):
                        ec.impl_eigv_prestress = True
                        ec.impl_nlgeom = True   # prestress needs the K_geo path
                    vals = block.cards[0].floats() if block.cards else []
                    if vals and vals[0] > 0:
                        ec.impl_eigv_nmode = int(vals[0])
                elif sub in ("CEIGV", "CEIG", "CMPLX", "COMPLEX"):
                    # /IMPL/CEIGV (M18 — a PORT card; freimpl.F has no
                    # complex/damped eigensolver). Drives the QEP / state-space
                    # complex-mode analysis of implicit/complex_modal.py the
                    # way /IMPL/EIGV drives the real eigensolver. Sub-keywords:
                    #   /IMPL/CEIGV        card: Nmode  (complex modes)
                    #   /IMPL/CEIGV/STRS   prestressed spectrum K = K_mat+K_geo
                    #   /IMPL/CEIGV/TRAN   card: t_end dt [nmode]  (complex-mode
                    #                      superposition TRANSIENT)
                    #   /IMPL/CEIGV/FRF    card: fmin fmax nf [nmode]  (damped
                    #                      complex FRF sweep)
                    # The damping C is the /IMPL/DYNA/DAMP Rayleigh a,b PLUS the
                    # deck's discrete dashpots (/PROP/SPRING c, /DAMP).
                    ec.impl_ceigv = True
                    ec.implicit = True
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    vals = block.cards[0].floats() if block.cards else []
                    if sub2 in ("STRS", "STRESS", "PRESTRESS"):
                        ec.impl_ceigv_prestress = True
                        ec.impl_nlgeom = True
                        if vals and vals[0] > 0:
                            ec.impl_ceigv_nmode = int(vals[0])
                    elif sub2 in ("TRAN", "TRANSIENT", "DYNA"):
                        ec.impl_ceigv_tran = True
                        if vals:
                            ec.impl_ceigv_tend = vals[0]
                        if len(vals) > 1 and vals[1] > 0:
                            ec.impl_ceigv_dt = vals[1]
                        if len(vals) > 2 and vals[2] > 0:
                            ec.impl_ceigv_nmode = int(vals[2])
                    elif sub2 in ("FRF", "FREQ", "HARMONIC"):
                        ec.impl_ceigv_frf = True
                        if len(vals) > 0:
                            ec.impl_ceigv_fmin = vals[0]
                        if len(vals) > 1:
                            ec.impl_ceigv_fmax = vals[1]
                        if len(vals) > 2 and vals[2] > 0:
                            ec.impl_ceigv_nf = int(vals[2])
                        if len(vals) > 3 and vals[3] > 0:
                            ec.impl_ceigv_nmode = int(vals[3])
                    else:
                        if vals and vals[0] > 0:
                            ec.impl_ceigv_nmode = int(vals[0])
                elif sub in ("MODAL", "MSUP"):
                    # /IMPL/MODAL/... (M17 — PORT cards; freimpl.F has no
                    # mode-superposition path). Drives the modal-transient /
                    # modal-damping library of implicit/modal_response.py the
                    # way /IMPL/EIGV drives the eigensolver. Sub-keywords:
                    #   /IMPL/MODAL/DYNA  card: t_end  dt  [nmode]
                    #                     (mode-superposition TRANSIENT; add
                    #                     /MACC as a 4th field or the /STRS
                    #                     suffix for prestressed modes)
                    #   /IMPL/MODAL/DAMP  card: zeta   (uniform modal damping;
                    #                     Rayleigh a,b comes from /IMPL/DYNA/
                    #                     DAMP, mapped consistently — see
                    #                     modal_response.rayleigh_ratios)
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    if sub2 in ("DYNA", "DYNAMIC", "DYN", ""):
                        ec.impl_modal_dyna = True
                        ec.implicit = True   # the static prestress driver hosts
                        vals = block.cards[0].floats() if block.cards else []
                        if vals:
                            ec.impl_modal_tend = vals[0]
                        if len(vals) > 1 and vals[1] > 0:
                            ec.impl_modal_dt = vals[1]
                        if len(vals) > 2 and vals[2] > 0:
                            ec.impl_modal_nmode = int(vals[2])
                        # /MACC (mode-acceleration) may ride as a 4th part
                        rest = [p.upper() for p in block.parts[3:]]
                        if "MACC" in rest or (len(vals) > 3 and vals[3] != 0):
                            ec.impl_modal_macc = True
                        if "STRS" in rest or "STRESS" in rest:
                            ec.impl_modal_prestress = True
                            ec.impl_nlgeom = True
                    elif sub2 == "MACC":
                        ec.impl_modal_macc = True
                    elif sub2 == "DAMP":
                        vals = block.cards[0].floats() if block.cards else []
                        if vals:
                            ec.impl_modal_zeta = vals[0]
                        if ec.impl_modal_zeta < 0.0:
                            log.warning(
                                "/IMPL/MODAL/DAMP: negative modal damping "
                                "ratio INJECTS energy", block.source)
                    else:
                        log.warning(f"/IMPL/MODAL/{sub2} not ported — ignored "
                                    f"(supports DYNA, DAMP, MACC)",
                                    block.source)
                elif sub in ("FREQ", "FRF", "HARMONIC"):
                    # /IMPL/FREQ (M17 — PORT card): harmonic / steady-state
                    # frequency response over a swept band. Card:
                    #   fmin  fmax  nf  [zeta]  [nmode]
                    # (a shaker-driven FRF; the excitation is the deck's
                    # /CLOAD pattern taken as the harmonic force amplitude).
                    ec.impl_freq = True
                    ec.implicit = True
                    vals = block.cards[0].floats() if block.cards else []
                    if len(vals) > 0:
                        ec.impl_freq_fmin = vals[0]
                    if len(vals) > 1:
                        ec.impl_freq_fmax = vals[1]
                    if len(vals) > 2 and vals[2] > 0:
                        ec.impl_freq_nf = int(vals[2])
                    if len(vals) > 3 and vals[3] > 0:
                        ec.impl_freq_zeta = vals[3]
                    if len(vals) > 4 and vals[4] > 0:
                        ec.impl_modal_nmode = int(vals[4])
                    if ec.impl_freq_fmax <= ec.impl_freq_fmin:
                        log.warning(
                            "/IMPL/FREQ: fmax <= fmin — set a positive sweep "
                            "band (fmin fmax nf)", block.source)
                elif sub in ("PSD", "RANDOM", "SPECTRAL"):
                    # /IMPL/PSD (M19 — a PORT card; freimpl.F has no
                    # random-vibration path). Stationary random / spectral
                    # (PSD) response S_uu = |H|^2 S_ff through the M17 real-mode
                    # FRF (or the M18 complex FRF), reporting the RMS, spectral
                    # moments and crossing/peak rates. Sub-keywords:
                    #   /IMPL/PSD        card: fmin fmax nf funct [nmode]
                    #                    (force PSD; the /CLOAD pattern is the
                    #                    spatial force pattern, funct = S_ff(f))
                    #   /IMPL/PSD/BASE   card: fmin fmax nf funct [dir] [nmode]
                    #                    (rigid-base ACCELERATION PSD, dir 0..5)
                    #   /IMPL/PSD/CPLX   card: fmin fmax nf funct [nmode]
                    #                    (force PSD via the M18 COMPLEX FRF —
                    #                    non-classical damping)
                    ec.impl_psd = True
                    ec.implicit = True
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    vals = block.cards[0].floats() if block.cards else []
                    if len(vals) > 0:
                        ec.impl_psd_fmin = vals[0]
                    if len(vals) > 1:
                        ec.impl_psd_fmax = vals[1]
                    if len(vals) > 2 and vals[2] > 0:
                        ec.impl_psd_nf = int(vals[2])
                    if len(vals) > 3 and vals[3] > 0:
                        ec.impl_psd_funct = int(vals[3])
                    if sub2 in ("BASE", "SUPPORT", "ACCEL"):
                        ec.impl_psd_base = True
                        # dir then nmode after the funct id
                        if len(vals) > 4 and vals[4] >= 0:
                            ec.impl_psd_dir = int(vals[4])
                        if len(vals) > 5 and vals[5] > 0:
                            ec.impl_psd_nmode = int(vals[5])
                    elif sub2 in ("CPLX", "COMPLEX", "CEIGV"):
                        ec.impl_psd_cplx = True
                        if len(vals) > 4 and vals[4] > 0:
                            ec.impl_psd_nmode = int(vals[4])
                    elif sub2 in ("STRS", "STRESS", "PRESTRESS"):
                        ec.impl_psd_prestress = True
                        ec.impl_nlgeom = True
                        if len(vals) > 4 and vals[4] > 0:
                            ec.impl_psd_nmode = int(vals[4])
                    elif sub2 in ("MULTI", "MINPUT", "MIMO", "COHERENT"):
                        # M28: MULTI-INPUT / partially-coherent random response.
                        # The single-input funct (line 0) is the reference PSD for
                        # the side-by-side listing; the MULTI-INPUT table follows
                        #   line 1     : ninput cohmodel gamma phase [decay speed]
                        #   lines 2..  : cload_funct psd_funct [x y z]  (per input)
                        # S_uu = H S_ff H^H is recovered ALONGSIDE the single-input
                        # answer. A PORT sub-flag (freimpl.F has no multi-input /
                        # coherence path). See implicit/multi_input_response.py.
                        ec.impl_psd_multi = True
                        if len(vals) > 4 and vals[4] > 0:
                            ec.impl_psd_nmode = int(vals[4])
                        _parse_multi_input_table(block, ec, 1, log,
                                                 "/IMPL/PSD/MULTI")
                    else:
                        if len(vals) > 4 and vals[4] > 0:
                            ec.impl_psd_nmode = int(vals[4])
                    if ec.impl_psd_funct <= 0:
                        log.warning(
                            "/IMPL/PSD: no input-PSD /FUNCT id on the card "
                            "(fmin fmax nf funct) — the run will error at the "
                            "analysis", block.source)
                elif sub in ("RSPEC", "RESPSPEC", "SPECTRUM"):
                    # /IMPL/RSPEC (M19 — a PORT card; freimpl.F has no
                    # response-spectrum path). Design-response-spectrum modal
                    # combination: per-mode peak r_i = Gamma_i Sa/omega^2
                    # combined by SRSS and CQC (Der Kiureghian 1981). Card:
                    #   funct  dir  [zeta]  [nmode]
                    # (funct = the design spectrum Sa(f); dir the 0..5 rigid
                    # excitation direction). Both SRSS and CQC are reported.
                    # /IMPL/RSPEC/STRS extracts the modes on the prestressed K.
                    ec.impl_rspec = True
                    ec.implicit = True
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    vals = block.cards[0].floats() if block.cards else []
                    if len(vals) > 0 and vals[0] > 0:
                        ec.impl_rspec_funct = int(vals[0])
                    if len(vals) > 1 and vals[1] >= 0:
                        ec.impl_rspec_dir = int(vals[1])
                    if len(vals) > 2 and vals[2] > 0:
                        ec.impl_rspec_zeta = vals[2]
                    if len(vals) > 3 and vals[3] > 0:
                        ec.impl_rspec_nmode = int(vals[3])
                    if sub2 in ("STRS", "STRESS", "PRESTRESS"):
                        ec.impl_rspec_prestress = True
                        ec.impl_nlgeom = True
                    if ec.impl_rspec_funct <= 0:
                        log.warning(
                            "/IMPL/RSPEC: no design-spectrum /FUNCT id on the "
                            "card (funct dir zeta nmode) — the run will error "
                            "at the analysis", block.source)
                elif sub in ("FATIG", "FATIGUE", "SPECTRALFATIGUE"):
                    # /IMPL/FATIG (M20 — a PORT card; freimpl.F has no
                    # spectral-fatigue solver). Random-vibration (spectral)
                    # fatigue: recover a STRESS PSD from the M19 stress modes,
                    # evaluate the narrow-band (Bendat) / Dirlik /
                    # Wirsching-Light / Tovo-Benasciutti damage of an S-N curve
                    # N = C*S^-m under a Miner sum. Two card lines:
                    #   line 1 (PSD sweep): fmin fmax nf funct [nmode]
                    #   line 2 (S-N + opts): m C [zeta] [mean ult] [mcdur seed]
                    # Sub-keywords:
                    #   /IMPL/FATIG        force PSD (the /CLOAD pattern)
                    #   /IMPL/FATIG/BASE   rigid-base ACCELERATION PSD; the
                    #                      base direction is appended to line 1
                    #                      (fmin fmax nf funct dir [nmode])
                    #   /IMPL/FATIG/STRS   modes on the prestressed K
                    #   /IMPL/FATIG/MULT   (M21) MULTIAXIAL / critical-plane —
                    #                      the FULL 6-Voigt stress-tensor
                    #                      cross-PSD reduced to an equivalent
                    #                      (von Mises / normal / shear critical
                    #                      plane) scalar PSD; composes with BASE
                    #                      / STRS (any order, e.g.
                    #                      /IMPL/FATIG/MULT/BASE)
                    #   /IMPL/FATIG/MULT/NPROP (M22) NON-PROPORTIONAL — the
                    #                      critical-plane TIME-DOMAIN path count
                    #                      of the rotating shear path (MCC/MRH
                    #                      shear amplitude + Findley /
                    #                      Fatemi-Socie with the per-plane max
                    #                      normal stress + the F_np factor);
                    #                      implies MULT, reuses the M21 seeded
                    #                      synthesiser (needs mcdur/seed on line
                    #                      2). Line 2 optionally appends k sigy:
                    #                      m C zeta mean ult mcdur seed [k sigy]
                    #   /IMPL/FATIG/MULT/NPROP/SPEC (M23) SPECTRAL non-
                    #                      proportional — the frequency-domain
                    #                      F_np + critical-plane damage estimated
                    #                      DIRECTLY from the cross-PSD moment
                    #                      matrices (NO synthesised history),
                    #                      reported ALONGSIDE the M21 spectral and
                    #                      M22 time-domain answers (all three side
                    #                      by side). Implies NPROP (hence MULT).
                    #   /IMPL/FATIG/NGAUSS (M24) NON-GAUSSIAN / KURTOSIS —
                    #                      correct the Gaussian spectral damage
                    #                      (scalar OR the MULT equivalent scalar)
                    #                      for a target kurtosis / skewness by the
                    #                      Winterstein Hermite model + the
                    #                      lambda_ng closed-form factor, with a
                    #                      non-Gaussian Monte-Carlo cross-check,
                    #                      reported ALONGSIDE the M20 Gaussian
                    #                      numbers. Composes with MULT / NPROP /
                    #                      SPEC (orthogonal). The kurtosis (and
                    #                      optional skewness) go on a THIRD card
                    #                      line: kurt [skew].
                    ec.impl_fatig = True
                    ec.implicit = True
                    # scan ALL sub-keywords (the modifiers compose in any order)
                    subs = {p.upper() for p in block.parts[2:]}
                    is_base = bool(subs & {"BASE", "SUPPORT", "ACCEL"})
                    is_strs = bool(subs & {"STRS", "STRESS", "PRESTRESS"})
                    is_mult = bool(subs & {"MULT", "MULTI", "MULTIAXIAL",
                                           "CRITPLANE"})
                    # M22: NON-PROPORTIONAL path counting; implies MULT
                    is_nprop = bool(subs & {"NPROP", "NONPROP",
                                            "NONPROPORTIONAL", "PATH"})
                    # M23: SPECTRAL non-proportional; implies NPROP (hence MULT)
                    is_spec = bool(subs & {"SPEC", "SPECTRAL", "FREQ",
                                           "FREQUENCY"})
                    # M24: NON-GAUSSIAN / KURTOSIS correction; orthogonal — it
                    # composes with (does NOT imply) MULT / NPROP / SPEC
                    is_ngauss = bool(subs & {"NGAUSS", "NONGAUSS",
                                             "NONGAUSSIAN", "KURTOSIS", "KURT"})
                    # M25: NON-STATIONARY / EVOLUTIONARY-PSD correction; also
                    # orthogonal — composes with (does NOT imply) MULT / NPROP /
                    # SPEC / NGAUSS. The RMS scale-vs-time modulation /FUNCT (and
                    # optional nseg) go on a DEDICATED card line AFTER the M24
                    # kurtosis line: modfunct [nseg].
                    is_nstat = bool(subs & {"NSTAT", "NONSTAT", "NONSTATIONARY",
                                            "MISSION"})
                    # M26: FULLY EVOLUTIONARY / NON-SEPARABLE-PSD correction; also
                    # orthogonal — composes with (does NOT imply) MULT / NPROP /
                    # SPEC / NGAUSS / NSTAT. The spectral SHAPE drifts with time (a
                    # swept centre frequency / broadening bandwidth), each window
                    # carrying its OWN full PSD (a spectrogram) — the general
                    # non-separable extension of the M25 separable |A(t)|^2 S(w)
                    # model. Its drifting-shape schedule (fc0 fc1 bw0 bw1 nwin) goes
                    # on a DEDICATED card line AFTER the M25 modulation line; the
                    # RMS level schedule / mission span are shared with /NSTAT's
                    # modulation /FUNCT (so /EVOL naturally composes with /NSTAT).
                    is_evol = bool(subs & {"EVOL", "EVOLUTIONARY",
                                           "NONSEPARABLE", "SPECTROGRAM",
                                           "CHIRP"})
                    # M27: FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR correction;
                    # the M21/M23 critical-plane reductions evaluated PER WINDOW of
                    # the full 6x6 stress-TENSOR cross-PSD, the critical plane /
                    # F_np RE-SEARCHED from the window's OWN tensor (so the plane
                    # may ROTATE / F_np may DRIFT window to window), Miner-summed —
                    # the joint-tensor lift of the M26 fixed-reduction scalar
                    # spectrogram. /JOINT (or /TENSOR) IMPLIES MULT + EVOL and
                    # reuses the /EVOL drifting-shape schedule line; it composes
                    # with /NPROP / /SPEC / /NGAUSS / /NSTAT.
                    is_joint = bool(subs & {"JOINT", "TENSOR", "JOINTTENSOR"})
                    # M28: MULTI-INPUT / partially-coherent random fatigue; the
                    # response / stress-tensor cross-PSD driven by SEVERAL
                    # simultaneous random inputs with a full Hermitian input
                    # cross-spectral matrix S_ff (auto-PSDs on the diagonal,
                    # coherence gamma_ab + phase off-diagonal). Composes with (does
                    # NOT imply) MULT / NPROP / SPEC / NGAUSS / NSTAT / EVOL / JOINT
                    # — the multi-input S_sigmasigma flows into all of them
                    # UNCHANGED. Its input-pattern TABLE (ninput cohmodel gamma
                    # phase, then one 'cload_funct psd_funct [x y z]' row per input)
                    # goes on DEDICATED card lines AFTER any M24-M26 lines. The
                    # multi-input answers are reported ALONGSIDE the single-input
                    # numbers (a "multi_input" sub-entry). A PORT sub-flag
                    # (freimpl.F has no multi-input / coherence path). See
                    # implicit/multi_input_response.py + multi_input_fatigue.py.
                    is_minput = bool(subs & {"MINPUT", "MULTIINPUT", "MIMO",
                                             "COHERENT"})
                    # M30: FREQUENCY-DEPENDENT + TIME-VARYING (evolutionary) INPUT
                    # COHERENCE — the coherence gamma_ab(f, t) varies with BOTH
                    # frequency AND time (a measured gamma_ab(f) shape drifting window
                    # to window, and/or the M28 exponential/convection field with a
                    # time-varying decay / speed). IMPLIES MINPUT (hence MULT) + EVOL
                    # and reuses the /EVOL drifting-shape schedule line; composes with
                    # /JOINT / /NSTAT. Its frequency-shape schedule lives on the M28
                    # multi-input header's trailing columns (decay1 speed1 gfunct0
                    # gfunct1). A PORT sub-flag (freimpl.F has no
                    # frequency-dependent-time-varying-coherence path). See
                    # implicit/freq_evolutionary_multi_input.py.
                    is_fcoh = bool(subs & {"FCOH", "FREQCOH", "FREQCOHERENCE",
                                           "FDRIFT", "FREQEVOL"})
                    # M31: CONTINUOUS WIGNER-VILLE / LOEVE INSTANTANEOUS time-frequency
                    # spectrum — replace the M26-M30 short-time WINDOWED spectrogram
                    # with a CONTINUOUS instantaneous spectrum S_WV(omega, t) evaluated
                    # at a fine instant grid, reduced AT EACH INSTANT (the critical
                    # plane / F_np drifting CONTINUOUSLY) and Miner-INTEGRATED. IMPLIES
                    # EVOL (it needs the drifting-shape schedule) and composes with
                    # /JOINT / /MINPUT / /FCOH (whichever tensor / multi-input path is
                    # active becomes the continuous-instantaneous one) + /NSTAT. Its
                    # grid-refinement factor `refine` and Cohen-class smoothing width
                    # `smooth` live on the trailing columns of the /EVOL drifting-shape
                    # line (fc0 fc1 bw0 bw1 nwin [refine smooth]). The windowed
                    # spectrogram is EXACTLY the refine = 1 / smooth = 0 limit
                    # (delegated byte-identically). A PORT sub-flag (freimpl.F has no
                    # time-frequency / Wigner-Ville / spectral solver). See
                    # implicit/wigner_ville_fatigue.py.
                    is_wville = bool(subs & {"WVILLE", "WV", "WIGNER",
                                             "WIGNERVILLE", "INST", "INSTANT",
                                             "TFR", "CONTINUOUS"})
                    # M34: the EXACT translation-process CORRELATION-DISTORTION INVERSION
                    # (Grigoriu / Nataf / NORTA correlation matching) — solve the
                    # underlying-Gaussian correlation rho^U so the component-wise Hermite
                    # transform of the joint 6x6 tensor reproduces the TARGET covariance
                    # EXACTLY (not merely M33's leading order), the preservation error
                    # driven to ~0. Composes with (does NOT imply anything on its own) the
                    # M33 /JOINT + /NGAUSS + /WVILLE joint path; a NEW covariance-exact
                    # sub-entry reported alongside the M33 leading-order joint. A PORT
                    # sub-flag. See implicit/joint_nongaussian_fatigue.py.
                    is_exact = bool(subs & {"EXACT", "NORTA", "NATAF",
                                            "GRIGORIU", "COVEXACT"})
                    # M36: NON-GAUSSIAN COPULA / NON-TRANSLATION JOINT DISTRIBUTION
                    # Replaces the Gaussian copula with a t-copula.
                    is_copula = bool(subs & {"COPULA", "TCOPULA"})
                    if is_wville:
                        is_evol = True                # continuous spectrum needs the
                        #                               drifting-shape / evol schedule
                    if is_fcoh:
                        is_minput = True              # frequency-dep coherence needs
                        is_evol = True                # the multi-input + evol paths
                    if is_minput:
                        is_mult = True                # multi-input needs the tensor
                    if is_joint:
                        is_evol = True
                        is_mult = True
                    if is_spec:
                        is_nprop = True
                    if is_nprop:
                        is_mult = True
                    v0 = block.cards[0].floats() if block.cards else []
                    v1 = (block.cards[1].floats()
                          if len(block.cards) > 1 else [])
                    # M24: line 3 carries the target kurtosis (and optional
                    # skewness): kurt [skew]
                    v2 = (block.cards[2].floats()
                          if len(block.cards) > 2 else [])
                    if len(v0) > 0:
                        ec.impl_fatig_fmin = v0[0]
                    if len(v0) > 1:
                        ec.impl_fatig_fmax = v0[1]
                    if len(v0) > 2 and v0[2] > 0:
                        ec.impl_fatig_nf = int(v0[2])
                    if len(v0) > 3 and v0[3] > 0:
                        ec.impl_fatig_funct = int(v0[3])
                    if is_mult:
                        ec.impl_fatig_mult = True
                    if is_nprop:
                        ec.impl_fatig_nprop = True
                    if is_spec:
                        ec.impl_fatig_spec = True
                    if is_ngauss:
                        ec.impl_fatig_ngauss = True
                    if is_nstat:
                        ec.impl_fatig_nstat = True
                    if is_evol:
                        ec.impl_fatig_evol = True
                    if is_joint:
                        ec.impl_fatig_joint = True
                    if is_minput:
                        ec.impl_fatig_minput = True
                    if is_fcoh:
                        ec.impl_mi_fcoh = True
                    if is_wville:
                        ec.impl_fatig_wville = True
                    if is_exact:
                        ec.impl_fatig_exact = True
                    if is_base:
                        ec.impl_fatig_base = True
                        if len(v0) > 4 and v0[4] >= 0:
                            ec.impl_fatig_dir = int(v0[4])
                        if len(v0) > 5 and v0[5] > 0:
                            ec.impl_fatig_nmode = int(v0[5])
                    else:
                        if is_strs:
                            ec.impl_fatig_prestress = True
                            ec.impl_nlgeom = True
                        if len(v0) > 4 and v0[4] > 0:
                            ec.impl_fatig_nmode = int(v0[4])
                    # line 2: the S-N curve + options
                    if len(v1) > 0 and v1[0] > 0:
                        ec.impl_fatig_snm = v1[0]
                    if len(v1) > 1 and v1[1] > 0:
                        ec.impl_fatig_snc = v1[1]
                    if len(v1) > 2 and v1[2] > 0:
                        ec.impl_fatig_zeta = v1[2]
                    if len(v1) > 3:
                        ec.impl_fatig_mean = v1[3]
                    if len(v1) > 4 and v1[4] > 0:
                        ec.impl_fatig_ult = v1[4]
                    if len(v1) > 5 and v1[5] > 0:
                        ec.impl_fatig_mcdur = v1[5]
                    if len(v1) > 6 and v1[6] > 0:
                        ec.impl_fatig_seed = int(v1[6])
                    # M22 NON-PROPORTIONAL extras (optional, appended to line 2):
                    #   ... seed [k sigy] — the Findley/Fatemi-Socie normal
                    # sensitivity k and the yield stress sigma_y
                    if len(v1) > 7 and v1[7] > 0:
                        ec.impl_fatig_k = v1[7]
                    if len(v1) > 8 and v1[8] > 0:
                        ec.impl_fatig_sigy = v1[8]
                    # M24 line 3: target kurtosis [skewness] for the non-Gaussian
                    # correction (kurt = 3, skew = 0 is Gaussian, a no-op)
                    if len(v2) > 0 and v2[0] > 0:
                        ec.impl_fatig_kurt = v2[0]
                    if len(v2) > 1:
                        ec.impl_fatig_skew = v2[1]
                    # M32: compose /NGAUSS with /WVILLE for a TIME-VARYING kurtosis
                    # gamma_4(t) drifting ALONG the continuous Wigner-Ville spectrum.
                    # The M24 kurtosis line optionally appends a kurtosis-vs-time
                    # /FUNCT id (col 3, sampled continuously onto the fine instant
                    # grid) and/or an END kurtosis (col 4) for a linear sweep:
                    #   kurt [skew [kfunct [kurt1]]]. When both are given the /FUNCT
                    # wins (a measured / arbitrary schedule). Only meaningful with
                    # /WVILLE (the continuous spectrum); with a stationary /NGAUSS it
                    # is ignored (the constant kurt applies). See implicit/
                    # nongaussian_wigner_ville_fatigue.py.
                    if len(v2) > 2 and v2[2] > 0:
                        ec.impl_fatig_kfunct = int(v2[2])
                    if len(v2) > 3 and v2[3] > 0:
                        ec.impl_fatig_kurt1 = v2[3]
                    # M33 JOINT-TENSOR NON-GAUSSIAN: when /JOINT composes with /NGAUSS,
                    # up to 6 PER-COMPONENT target kurtoses (the Voigt stress components
                    # xx yy zz xy yz zx) on the TRAILING columns 4..9 of the M24
                    # kurtosis line impose the kurtosis JOINTLY on the TENSOR components
                    # (a VECTOR Winterstein-Hermite / translation transform preserving
                    # the marginal variances / covariance), so the resolved critical
                    # plane INHERITS the INDUCED kurtosis of the joint tensor statistics
                    # — the M33 joint-tensor distribution (vs the M24/M32 kurtosis on
                    # the already-resolved scalar). A NON-EMPTY per-component list (>= 1
                    # value on col 4+) triggers the M33 joint path; fewer than 6 values
                    # pad with the scalar kurt (col 0). All == 3 is Gaussian (a no-op).
                    # Only meaningful with /JOINT + /WVILLE; see implicit/
                    # joint_nongaussian_fatigue.py.
                    if is_joint and is_ngauss and len(v2) > 4:
                        kfill = float(ec.impl_fatig_kurt)
                        jk = [float(v2[i]) for i in range(4, min(10, len(v2)))]
                        jk += [kfill] * (6 - len(jk))     # pad to 6 with the scalar kurt
                        ec.impl_fatig_joint_kurt = tuple(jk[:6])
                    if is_ngauss and ec.impl_fatig_kurt <= 0.0:
                        log.warning(
                            "/IMPL/FATIG/NGAUSS needs a target kurtosis on line "
                            "3 (kurt [skew]); kurt = 3 is Gaussian (a no-op)",
                            block.source)
                    # M25 NON-STATIONARY: the RMS scale-vs-time modulation /FUNCT
                    # (and optional block count nseg) live on a DEDICATED card
                    # line AFTER the M24 kurtosis line — so its index is 2 (no
                    # NGAUSS) or 3 (NGAUSS also present): modfunct [nseg]
                    if is_nstat:
                        nline = 3 if is_ngauss else 2
                        vN = (block.cards[nline].floats()
                              if len(block.cards) > nline else [])
                        if len(vN) > 0 and vN[0] > 0:
                            ec.impl_fatig_modfunct = int(vN[0])
                        if len(vN) > 1 and vN[1] > 0:
                            ec.impl_fatig_nstat_nseg = int(vN[1])
                        if ec.impl_fatig_modfunct <= 0:
                            log.warning(
                                "/IMPL/FATIG/NSTAT needs an RMS scale-vs-time "
                                "modulation /FUNCT id on the line after the "
                                "sweep/S-N (and kurtosis, if NGAUSS) lines: "
                                "modfunct [nseg]", block.source)
                    # M26 EVOLUTIONARY: the drifting-shape schedule
                    # (fc0 fc1 bw0 bw1 nwin) lives on a DEDICATED card line AFTER
                    # the M25 modulation line — so its index is 2 + (1 if NGAUSS)
                    # + (1 if NSTAT). The RMS level schedule / mission span are
                    # taken from the shared modulation /FUNCT (impl_fatig_modfunct),
                    # so /EVOL composes with /NSTAT (and can reuse a modfunct even
                    # when NSTAT is absent).
                    if is_evol:
                        eline = 2 + (1 if is_ngauss else 0) + (1 if is_nstat
                                                               else 0)
                        vE = (block.cards[eline].floats()
                              if len(block.cards) > eline else [])
                        if len(vE) > 0:
                            ec.impl_fatig_evol_fc0 = vE[0]
                        if len(vE) > 1:
                            ec.impl_fatig_evol_fc1 = vE[1]
                        else:
                            ec.impl_fatig_evol_fc1 = ec.impl_fatig_evol_fc0
                        if len(vE) > 2:
                            ec.impl_fatig_evol_bw0 = vE[2]
                        if len(vE) > 3:
                            ec.impl_fatig_evol_bw1 = vE[3]
                        else:
                            ec.impl_fatig_evol_bw1 = ec.impl_fatig_evol_bw0
                        if len(vE) > 4 and vE[4] > 0:
                            ec.impl_fatig_evol_nwin = int(vE[4])
                        if ec.impl_fatig_evol_nwin <= 0:
                            log.warning(
                                "/IMPL/FATIG/EVOL needs a positive window count "
                                "nwin on the drifting-shape line (fc0 fc1 bw0 bw1 "
                                "nwin) after the sweep/S-N (and kurtosis/modulation"
                                ", if NGAUSS/NSTAT) lines", block.source)
                        # M31 WVILLE: the CONTINUOUS-spectrum grid-refinement factor
                        # (fine instants per window) and Cohen-class cross-term
                        # smoothing width are the trailing columns 6 / 7 of the SAME
                        # drifting-shape line: fc0 fc1 bw0 bw1 nwin [refine smooth].
                        if len(vE) > 5 and vE[5] > 0:
                            ec.impl_fatig_wv_refine = int(vE[5])
                        if len(vE) > 6 and vE[6] >= 0:
                            ec.impl_fatig_wv_smooth = float(vE[6])
                            
                    # M36 COPULA: the copula type and params live on a DEDICATED
                    # card line AFTER the M26 drifting-shape line (and before MINPUT)
                    if is_copula:
                        cline = 2 + (1 if is_ngauss else 0) + (1 if is_nstat else 0) + (1 if is_evol else 0)
                        vC = (block.cards[cline].floats() if len(block.cards) > cline else [])
                        ec.impl_fatig_copula = "t"
                        if len(vC) > 0 and vC[0] > 0.0:
                            ec.impl_fatig_copula_params = vC[0]
                        else:
                            ec.impl_fatig_copula_params = 4.0
                    # M28 MULTI-INPUT: the input-pattern TABLE lives on DEDICATED
                    # card lines AFTER any M24 kurtosis / M25 modulation / M26
                    # drifting-shape lines — so its base index is 2 + (1 if NGAUSS)
                    # + (1 if NSTAT) + (1 if EVOL). Header 'ninput cohmodel gamma
                    # phase [decay speed]' then one 'cload_funct psd_funct [x y z]'
                    # row per input. The single-input funct (line 1) is the
                    # reference PSD for the side-by-side listing.
                    if is_minput:
                        miline = (2 + (1 if is_ngauss else 0)
                                  + (1 if is_nstat else 0)
                                  + (1 if is_evol else 0)
                                  + (1 if is_copula else 0))
                        _parse_multi_input_table(block, ec, miline, log,
                                                 "/IMPL/FATIG/MINPUT")
                    if is_nprop and ec.impl_fatig_mcdur <= 0.0:
                        log.warning(
                            "/IMPL/FATIG/MULT/NPROP needs a Monte-Carlo record "
                            "length (mcdur on line 2: m C zeta mean ult mcdur "
                            "seed) — the path count runs on the synthesised "
                            "history; without it there is no time domain to "
                            "count", block.source)
                    if ec.impl_fatig_funct <= 0:
                        log.warning(
                            "/IMPL/FATIG: no input-PSD /FUNCT id on line 1 "
                            "(fmin fmax nf funct) — the run will error at the "
                            "analysis", block.source)
                    if ec.impl_fatig_snm <= 0.0 or ec.impl_fatig_snc <= 0.0:
                        log.warning(
                            "/IMPL/FATIG: no valid S-N curve on line 2 (m C) — "
                            "a positive slope m and coefficient C are required "
                            "(N = C*S^-m)", block.source)
                elif sub in ("NEWTON", "SOLVINFO"):
                    if block.cards:
                        vals = block.cards[0].floats()
                        if vals:
                            ec.impl_tol = vals[0]
                        if len(vals) > 1:
                            ec.impl_max_iter = int(vals[1])
                elif sub in ("LSOLVER", "SOLVER"):
                    ec.impl_linsolve = (block.parts[2].lower()
                                        if len(block.parts) > 2 else "")
                elif sub in ("DYNA", "DYNAMIC", "DYN"):
                    # /IMPL/DYNA (M10): implicit DYNAMICS — Newmark / HHT.
                    # Sub-sub-keyword mirrors the original's IDYNA read
                    # (freimpl.F): /1 = HHT (card: alpha), /2 = Newmark
                    # (card: gamma beta), /DAMP = Rayleigh damping (NOT
                    # ported — deferred, see PORTING_GUIDE M10).
                    sub2 = (block.parts[2].upper()
                            if len(block.parts) > 2 else "")
                    if sub2 == "DAMP":
                        # M11: Rayleigh damping C = a*M + b*K. freimpl.F
                        # reads DAMPA_IMP, DAMPB_IMP from one card and the
                        # option IMPLIES dynamics (IF (IDYNA==0) IDYNA=1 —
                        # HHT with alpha unset, i.e. trapezoidal).
                        ec.impl_dyna_damp = True
                        if ec.impl_dyna == 0:
                            ec.impl_dyna = 1
                        vals = block.cards[0].floats() if block.cards else []
                        if vals:
                            ec.impl_dyna_dampa = vals[0]
                        if len(vals) > 1:
                            ec.impl_dyna_dampb = vals[1]
                        if ec.impl_dyna_dampa < 0.0 or \
                                ec.impl_dyna_dampb < 0.0:
                            log.warning(
                                "/IMPL/DYNA/DAMP: negative Rayleigh "
                                "coefficient — this INJECTS energy",
                                block.source)
                    elif sub2 in ("", "1", "2"):
                        ec.impl_dyna = int(sub2) if sub2 else 2
                        vals = block.cards[0].floats() if block.cards else []
                        if ec.impl_dyna == 1:
                            # HHT: the card value IS alpha (freimpl.F reads
                            # HHT_A; unset -> -1e-20, i.e. trapezoidal)
                            ec.impl_dyna_alpha = vals[0] if vals else 0.0
                            if not -1.0 / 3.0 - 1e-12 <= ec.impl_dyna_alpha \
                                    <= 0.0:
                                log.warning(
                                    f"/IMPL/DYNA/1: alpha = "
                                    f"{ec.impl_dyna_alpha:g} outside the "
                                    f"HHT range [-1/3, 0] — second-order "
                                    f"accuracy / unconditional stability "
                                    f"not guaranteed", block.source)
                        else:
                            # Newmark: gamma then beta (DY_G = NM_A,
                            # DY_B = NM_B in imp_dyna.F)
                            if vals:
                                ec.impl_dyna_gamma = vals[0]
                            if len(vals) > 1:
                                ec.impl_dyna_beta = vals[1]
                            g, b = ec.impl_dyna_gamma, ec.impl_dyna_beta
                            # unconditional stability iff 2*beta >= gamma
                            # >= 1/2 (Hughes, The FEM, table 9.1.1)
                            if g < 0.5 or 2.0 * b < g:
                                log.warning(
                                    f"/IMPL/DYNA/2: gamma = {g:g}, beta = "
                                    f"{b:g} is NOT unconditionally stable "
                                    f"(needs 2*beta >= gamma >= 1/2)",
                                    block.source)
                    else:
                        log.warning(f"/IMPL/DYNA/{sub2} not ported — "
                                    f"ignored (supports 1, 2)", block.source)
                else:
                    log.warning(f"/IMPL/{sub} not ported — ignored (supports "
                                f"DTINI, NEWTON, LSOLVER, NONLIN, ARCL, "
                                f"DYNA, EIGV, CEIGV, BUCKL, MODAL, FREQ, "
                                f"PSD, RSPEC)",
                                block.source)
            elif key == "PRINT":
                # /PRINT/-100 → one listing line every 100 cycles (the minus
                # sign is the Radioss convention for 'every n cycles').
                if len(block.parts) > 1:
                    ec.print_cycles = abs(int(block.parts[1]))
                elif block.cards:
                    # Alternately, /PRINT \n N_print (M82)
                    v = block.cards[0].floats()
                    if v and v[0] != 0.0:
                        ec.print_cycles = abs(int(v[0]))
            elif key == "STOP":
                # /STOP card: Emax Mmax Nmax NTH NANIM. Emax = 0.0 (blank or
                # explicit 0, as official decks write '0 0 0 1 1') means "no
                # user limit" in the real engine — NOT a 0% tolerance. Keep
                # the 15% default then; only a positive Emax overrides it
                # (M37: this zero mis-read aborted 14 official decks at
                # their first energy check).
                if block.cards:
                    emax = block.cards[0].floats()[0]
                    if emax > 0.0:
                        ec.energy_error_stop = emax
            elif key == "DEBUG":
                # /DEBUG or /DEBUG/<suboption> (M120): fredebug.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                val = 1
                if len(block.parts) > 2:
                    try:
                        val = int(block.parts[2])
                    except ValueError:
                        val = 1
                elif sub.isdigit():
                    val = int(sub)
                    sub = "CORE"
                if not sub:
                    sub = "CORE"
                ec.debug_flags[sub] = val
                if sub == "ACC" and block.cards:
                    v = block.cards[0].floats()
                    if v:
                        ec.debug_acc_start = v[0]
                    if len(v) > 1:
                        ec.debug_acc_freq = int(v[1])
            elif key == "BCS":
                # /BCS/ON or /BCS/OFF (M120): frebcs.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else "ON"
                active = (sub == "ON")
                for c in block.cards:
                    if c.is_blank:
                        continue
                    for tok in c.tokens():
                        try:
                            bc_id = int(float(tok))
                            if bc_id > 0:
                                ec.bcs_active[bc_id] = active
                        except ValueError:
                            pass
            elif key == "RBODY":
                # /RBODY/ON or /RBODY/OFF (M120): frerbo.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else "ON"
                active = (sub == "ON")
                for c in block.cards:
                    if c.is_blank:
                        continue
                    for tok in c.tokens():
                        try:
                            rb_id = int(float(tok))
                            if rb_id > 0:
                                ec.rbody_active[rb_id] = active
                        except ValueError:
                            pass
            elif key == "ALE":
                # /ALE/ON or /ALE/OFF (M120): fraleonoff.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else "ON"
                active = (sub == "ON")
                for c in block.cards:
                    if c.is_blank:
                        continue
                    for tok in c.tokens():
                        try:
                            part_id = int(float(tok))
                            if part_id > 0:
                                ec.ale_active[part_id] = active
                        except ValueError:
                            pass
            elif key == "NOIS":
                # /NOIS or /NOIS/DT (M120): frenois.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "DT":
                    if block.cards:
                        v = block.cards[0].floats()
                        if len(v) > 1:
                            ec.noise_tstart = v[0]
                            ec.noise_dt = v[1]
                        elif v:
                            ec.noise_dt = v[0]
                elif sub:
                    ec.noise_flags[sub] = True
                else:
                    if block.cards:
                        v = block.cards[0].floats()
                        if len(v) > 1:
                            ec.noise_tstart = v[0]
                            ec.noise_dt = v[1]
                        elif v:
                            ec.noise_dt = v[0]
            elif key == "H3D":
                # /H3D/DT, /H3D/NODA, /H3D/ELEM, /H3D/SHELL, /H3D/SOLID (M120): redkey1_h3d.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "DT":
                    if block.cards:
                        v = block.cards[0].floats()
                        if len(v) > 1:
                            ec.h3d_dt = v[1]
                        elif v:
                            ec.h3d_dt = v[0]
                elif sub:
                    channel = "/".join(block.parts[1:]).upper()
                    if channel not in ec.h3d_requests:
                        ec.h3d_requests.append(channel)
            elif key == "FLOW":
                # /FLOW/DT or /FLOW (M120): freflw.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "DT" or not sub:
                    if block.cards:
                        v = block.cards[0].floats()
                        if len(v) > 1:
                            ec.flow_dt = v[1]
                        elif v:
                            ec.flow_dt = v[0]
            elif key == "UPWIND":
                # /UPWIND (M120): freupwind.F
                ec.upwind_active = True
                if block.cards:
                    v = block.cards[0].floats()
                    if len(v) > 0 and v[0] > 0.0:
                        ec.upwind_mom = v[0]
                    if len(v) > 1 and v[1] > 0.0:
                        ec.upwind_mass_eng = v[1]
                    if len(v) > 2 and v[2] > 0.0:
                        ec.upwind_wet_surf = v[2]
            elif key == "EIG":
                # /EIG/OFF (M120): freeig.F
                sub = block.parts[1].upper() if len(block.parts) > 1 else ""
                if sub == "OFF":
                    for c in block.cards:
                        if c.is_blank:
                            continue
                        for tok in c.tokens():
                            try:
                                m_id = int(float(tok))
                                if m_id > 0:
                                    ec.eig_off.append(m_id)
                            except ValueError:
                                pass
            else:
                log.warning(f"engine keyword /{'/'.join(block.parts)} "
                            f"not ported — ignored", block.source)
        except (ValueError, IndexError) as exc:
            log.error(f"while reading /{'/'.join(block.parts)}: {exc}",
                      block.source)
    return ec
