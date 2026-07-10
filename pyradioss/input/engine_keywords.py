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
