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
from .bolt_preload import (  # noqa: F401
    BoltPreloadEngine,
    BoltPreloadParams,
    PreloadMethod,
    PreloadPhase,
    build_bolt_preloads,
)

from .thermal_loads import (  # noqa: F401
    ConvecLoad,
    ConvecParams,
    RadiationLoad,
    RadiationParams,
    STEFAN_BOLTZMANN,
    ThermalLoadsManager,
    ThermalStepResult,
    compute_convec_flux,
    compute_radiation_flux,
    compute_segment_area,
    convec_subroutine,
    radiation_subroutine,
    simulate_lumped_cooling,
)
try:
    from .accel_filter import (  # noqa: F401
        AccelFilter,
        FilterCoefficients,
        accel1,
        compute_filter_coefficients,
        CFC_CUTOFF_FREQUENCIES,
        parse_cfc,
    )
except ImportError:
    pass

from .pfluid import (  # noqa: F401
    PfluidEngine,
    PfluidLoad,
    PfluidLoadParams,
    PfluidSegmentResult,
    PfluidStepResult,
    compute_segment_normal_and_area,
    pfluid_subroutine,
)


from .sph_boundary import (  # noqa: F401
    SphBoundaryManager,
    SphInflow,
    SphInflowParams,
    SphOutflow,
    SphOutflowParams,
    build_sph_inflow,
    build_sph_outflow,
)
from .load_pcyl import (  # noqa: F401
    PcylLoadEngine,
    PcylLoadParams,
    PcylSegment,
    build_pcyl_loads,
)

try:
    from .inigrav import (  # noqa: F401
        IniGravParams,
        SoilLayer,
        GeostaticStressResult,
        compute_geostatic_stress,
        compute_depth_along_gravity,
        apply_inigrav,
        check_geostatic_equilibrium,
        k0_from_phi,
        k0_from_nu,
        build_inigrav,
    )
except ImportError:
    pass

from .load_centri import (  # noqa: F401
    LoadCentri,
    CentrifugalResult,
    compute_centrifugal_forces,
    cfield_subroutine,
    CentrifugalEngine,
    build_load_centri,
)



