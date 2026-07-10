"""
Numerical constants used throughout the solver.

Fortran origin: ``common_source/comm/constant.inc`` (and the ``ZERO``,
``EM20``, ``EP30`` … parameters sprinkled through every Fortran file).
The original defines named constants like

    my_real, parameter :: ZERO = 0.0, EM20 = 1.0E-20, EP30 = 1.0E30

mostly to guarantee identical literals everywhere and to make the "guard"
values greppable. We keep the same *names* here so that a reader coming from
the Fortran immediately recognizes e.g. ``EM20`` as the tiny value used to
avoid division by zero.
"""

# Tiny value used to protect divisions (Fortran: EM20). Any length, area or
# velocity magnitude smaller than this is treated as zero.
EM20 = 1.0e-20

# Huge value used to initialize minima (Fortran: EP30), e.g. the search for
# the smallest element time step starts from EP30.
EP30 = 1.0e30

# Default time-step *scale factor* applied to the critical (Courant) time
# step. The pure central-difference stability limit dt <= 2/omega_max is an
# upper bound computed from linearized element stiffness; non-linearity,
# contact and hourglass forces erode it, so Radioss multiplies by a safety
# factor. Fortran default: 0.9 (see engine /DT card documentation and
# engine/source/time_step/).
DEFAULT_DT_SCALE = 0.9

# Default bulk-viscosity coefficients for solid elements (quadratic qa and
# linear qb terms) — same defaults as the Radioss /PROP/SOLID card.
# They damp shock oscillations by adding a viscous pressure
#   q = rho * l * (qa^2 * l * tr(D)^2  -  qb * c * tr(D))   when tr(D) < 0.
DEFAULT_QA = 1.1
DEFAULT_QB = 0.05

# Default hourglass resistance coefficient for one-point-integrated elements
# (Radioss /PROP/SOLID "h" and /PROP/SHELL "hm/hf/hr" default 0.01–0.1 range;
# we use 0.1, the common practical value, LS-DYNA's QM default is 0.10 too).
DEFAULT_HOURGLASS = 0.10

# Shear correction factor for Mindlin–Reissner shells (5/6): the transverse
# shear stress distribution through the thickness is parabolic, not constant,
# so the constant-shear-strain kinematics overestimates shear stiffness by
# exactly 6/5 for a homogeneous section.
SHEAR_FACTOR = 5.0 / 6.0
