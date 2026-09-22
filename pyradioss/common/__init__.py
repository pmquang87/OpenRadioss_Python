"""
pyradioss.common — shared low-level services.

Fortran origin: ``common_source/`` in the OpenRadioss tree, which holds code
shared between Starter and Engine (constants, math tools, the message system,
function-table interpolation, ...).
"""

from . import constants, fastmath, graph, messages, npcompat, polygon_clip, splines, tables  # noqa: F401
