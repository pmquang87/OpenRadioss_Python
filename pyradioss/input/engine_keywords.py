"""
Engine file parsers: ``*_0001.rad`` → EngineControls.

Fortran origin: the Engine's own reader (``engine/source/input/lectur.F``
and friends, storing into ``dt_mod``, ``output_mod``...). The engine file
is small: it only *controls* the run (final time, time-step options, output
frequencies) — the model itself comes from the Starter restart file.

Supported cards (layouts documented per parser below)::

    /RUN/RunName/run#          card: T_stop
    /VERS/...                  ignored (input version)
    /TFILE                     card: dT_history
    /ANIM/DT                   card: T_start  dT_anim
    /ANIM/VECT/<VEL|DIS|ACC>   request nodal vector in animation files
    /ANIM/ELEM/<VONM|EPSP>     request element scalar in animation files
    /DT                        card: Scale  [dT_min]
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
                # dT_min the Engine stops (the original can also switch to
                # mass scaling here — /DT/NODA/CST — not ported).
                if block.cards:
                    vals = block.cards[0].floats()
                    if vals:
                        ec.dt_scale = vals[0]
                    if len(vals) > 1:
                        ec.dt_min = vals[1]
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
