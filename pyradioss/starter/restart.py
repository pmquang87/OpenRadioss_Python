"""
Restart files: the Starter->Engine handshake AND the Engine->Engine
chaining contract (M6).

Fortran origin: ``starter/source/restart/ddsplit/wrrest.F`` writes the
binary restart (historically ``*.rst`` / ``RunName_0000.r00``) holding
every initialized array; ``engine/source/output/restart/rdresb.F`` reads
it back — and ``engine/source/output/restart/wrrestp.F`` writes the
ENGINE restart at the end of a run (and at /STATE snapshots), which the
NEXT engine deck (``RunName_0002.rad``) resumes from. The *format* is
private to the code pair, so the port is free to choose: a Python pickle
of the whole :class:`Model` object plus a small header for versioning —
same contract, zero hand-written serialization code to maintain.

The chaining contract (M6)
--------------------------
``RunName_NNNN.rad`` reads ``RunName_{NNNN-1}.rst`` and writes
``RunName_NNNN.rst`` when it finishes. A Starter restart carries the
model only (``engine`` is None — the run starts at t = 0); an Engine
restart additionally carries the accumulated solver state the model
arrays cannot express:

* the clock and ledgers (t, cycle count, external work, contact energy,
  the error-reference peak and initial energy, the numerical/damping
  dissipation counters, the mass-scaling totals),
* the NEXT time step (dt is computed at the END of each cycle for the
  next one — a resumed run must take exactly the step the unchained run
  would have taken),
* output bookkeeping (TH/ANIM next-write times and the animation file
  counter, so a chained run continues the numbering),
* per-rigid-body dynamic state (orientation R, angular momentum L,
  reference velocity/position — the body frame cannot be reconstructed
  from nodal positions),
* latched /SENSOR fire times (a chained run must not re-arm sensors).

Everything else is deliberately RECONSTRUCTED from the model arrays at
engine start (tied-contact projections and release states, contact
candidate lists, constraint fix masks...): the acceptance test of the
contract is that a chained run reproduces the unchained one exactly —
same cycle count, energies to round-off (asserted by the M6 tests).
"""

from __future__ import annotations

import pickle
from typing import Optional, Tuple

from .. import __version__
from ..model.model import Model

_MAGIC = "pyradioss-restart"


def write_restart(model: Model, path: str,
                  engine: Optional[dict] = None) -> None:
    """Write a restart: Starter flavour (``engine=None``) or Engine
    flavour (``engine`` = the accumulated-state dict, see module doc)."""
    with open(path, "wb") as fh:
        pickle.dump({"magic": _MAGIC, "version": __version__,
                     "model": model, "engine": engine}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)


def read_restart(path: str) -> Tuple[Model, Optional[dict]]:
    """Read a restart; returns (model, engine_state) with engine_state
    None for a Starter restart (fresh run from t = 0)."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not (isinstance(data, dict) and data.get("magic") == _MAGIC):
        raise ValueError(f"{path} is not a pyradioss restart file")
    return data["model"], data.get("engine")
