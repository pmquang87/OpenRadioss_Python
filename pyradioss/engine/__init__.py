"""
pyradioss.engine — the Engine program (explicit solver).

Fortran origin: the ``engine/`` half of the OpenRadioss tree; the central
file is ``engine/source/engine/resol.F`` (the time-integration driver),
ported as :mod:`pyradioss.engine.engine`.
"""

from .engine import run_engine  # noqa: F401
from .element_erosion import compute_sdlenmax, check_solid_geometric_erosion, SdLenMaxArray  # noqa: F401
from .range_damping import (  # noqa: F401
    damping_range_compute_param,
    DampingRangeSolid,
    DampingRangeShell,
    damping_range_solid_subroutine,
    damping_range_shell_subroutine,
    damping_range_shell_mom_subroutine,
)
from .noise import compute_filter_coefficients, FilterOutput, NoiseFilter  # noqa: F401
from .fsi_coupling import (  # noqa: F401
    FSIInterface,
    fsi_compute_slave_normals,
    fsi_pressure_to_force,
    fsi_velocity_compatibility,
    fsi_step,
)

