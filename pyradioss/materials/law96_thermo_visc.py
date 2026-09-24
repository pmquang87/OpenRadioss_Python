"""LAW96 — Thermo-Elasto-Viscoplastic Polymer Formulation (/MAT/LAW96, /MAT/THERMO_VISC_POLY).

Upstream OpenRadioss Fortran reference:
- Constitutive Update:
  `engine/source/materials/mat/mat096/sigeps96.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971/MAT/mat_096.cfg`
"""

from .law96_thermovp import (
    Law96Params,
    build_law96,
    resolve,
    needs_defgrad,
    extra_shapes,
    sound_speed,
    calc_yield as _calc_yield,
    _solid_update_single,
    solid_update,
    _shell_update_single,
    shell_update,
    solid_tangent,
    consistent_solid_tangent,
    shell_tangent,
    consistent_shell_tangent,
    tangent,
)

__all__ = [
    "Law96Params",
    "build_law96",
    "resolve",
    "needs_defgrad",
    "extra_shapes",
    "sound_speed",
    "_calc_yield",
    "_solid_update_single",
    "solid_update",
    "_shell_update_single",
    "shell_update",
    "solid_tangent",
    "consistent_solid_tangent",
    "shell_tangent",
    "consistent_shell_tangent",
    "tangent",
]
