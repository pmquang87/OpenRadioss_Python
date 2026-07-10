"""
pyradioss.output — result files.

Fortran origin: ``engine/source/output/`` — the listing writer (ecrit.F),
the time-history writer (thbuf/wrtdes, binary T01 format) and the ANIM
writer (sortie_main.F + anim/*, proprietary A-file format).

Port format decisions (documented deviations):
* T01  -> plain CSV  (openable anywhere; column meanings in the header)
* ANIM -> legacy VTK (openable directly in ParaView, the post-processor
          the OpenRadioss HOWTO itself recommends)
* .out listings keep the original's spirit (cycle table, energy balance).
"""

from .time_history import TimeHistory  # noqa: F401
from .anim_vtk import write_anim_state  # noqa: F401
