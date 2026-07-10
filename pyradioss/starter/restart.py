"""
Restart file: the Starter->Engine handshake.

Fortran origin: ``starter/source/restart/ddsplit/wrrest.F`` writes the
binary restart (historically ``*.rst`` / ``RunName_0000.r00``) holding
every initialized array; ``engine/source/output/restart/rdresb.F`` reads
it back. The *format* is private to the code pair, so the port is free to
choose: we use a Python pickle of the whole :class:`Model` object plus a
small header for versioning. Same contract, zero hand-written
serialization code to maintain.
"""

from __future__ import annotations

import pickle

from .. import __version__
from ..model.model import Model

_MAGIC = "pyradioss-restart"


def write_restart(model: Model, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump({"magic": _MAGIC, "version": __version__,
                     "model": model}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def read_restart(path: str) -> Model:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not (isinstance(data, dict) and data.get("magic") == _MAGIC):
        raise ValueError(f"{path} is not a pyradioss restart file")
    return data["model"]
