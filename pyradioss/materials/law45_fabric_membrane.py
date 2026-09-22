# Ported from OpenRadioss Fortran:
# Source: engine/source/materials/mat/mat045/sigeps45.F
# Function: SIGEPS45 (lines 31-338)
# Source: engine/source/materials/mat/mat045/sigeps45c.F
# Function: SIGEPS45C (lines 30-397)
"""LAW45 — Fabric / membrane orthotropic material model with rate-dependent Zhao plasticity.

Re-exports all definitions from law45_orth_fabric for consistent naming conventions.
"""

from pyradioss.materials.law01_elastic import solid_update as _elastic_solid_update
from pyradioss.materials.law45_orth_fabric import (
    Law45Params,
    resolve,
    build_law45,
    extra_shapes,
    needs_defgrad,
    sound_speed,
    shell_update,
    solid_update,
    shell_tangent,
    consistent_shell_tangent,
    shell_membrane_tangent,
    tangent_law45_shell,
    solid_tangent,
    consistent_solid_tangent,
    tangent_law45_solid,
    tangent,
)

__all__ = [
    "Law45Params",
    "resolve",
    "build_law45",
    "extra_shapes",
    "needs_defgrad",
    "sound_speed",
    "shell_update",
    "solid_update",
    "shell_tangent",
    "consistent_shell_tangent",
    "shell_membrane_tangent",
    "tangent_law45_shell",
    "solid_tangent",
    "consistent_solid_tangent",
    "tangent_law45_solid",
    "tangent",
]
