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
                elif sub:
                    log.warning(f"/DT/{sub} not ported — treated as /DT",
                                block.source)
                if block.cards:
                    vals = block.cards[0].floats()
                    if vals:
                        ec.dt_scale = vals[0]
                    if len(vals) > 1:
                        ec.dt_min = vals[1]
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
                    else:
                        log.warning(f"/IMPL/DT/{sub2} not ported — ignored "
                                    f"(supports STOP, 1; the IDTC 2/3 "
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
                                f"DYNA, EIGV, BUCKL, MODAL, FREQ)",
                                block.source)
            elif key == "PRINT":
                # /PRINT/-100 → one listing line every 100 cycles (the minus
                # sign is the Radioss convention for 'every n cycles').
                if len(block.parts) > 1:
                    ec.print_cycles = abs(int(block.parts[1]))
            elif key == "STOP":
                if block.cards:
                    ec.energy_error_stop = block.cards[0].floats()[0]
            else:
                log.warning(f"engine keyword /{'/'.join(block.parts)} "
                            f"not ported — ignored", block.source)
        except (ValueError, IndexError) as exc:
            log.error(f"while reading /{'/'.join(block.parts)}: {exc}",
                      block.source)
    return ec
