"""
pyradioss.engine — the Engine program (explicit solver).

Fortran origin: the ``engine/`` half of the OpenRadioss tree; the central
file is ``engine/source/engine/resol.F`` (the time-integration driver),
ported as :mod:`pyradioss.engine.engine`.
"""

from .engine import run_engine  # noqa: F401
