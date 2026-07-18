"""
NumPy version-compatibility shims.

Why this module exists
----------------------
``pyproject.toml`` declares ``dependencies = ["numpy>=1.22"]``, so pyradioss
promises to run on the NumPy 1.x series. That promise is load-bearing: the
post-processing stack (``vortex_radioss`` -> ``lasso-python``) hard-pins
``numpy>=1.23.3,<2.0.0``, so any environment that can read a Radioss animation
file through lasso is *necessarily* a NumPy 1.x environment. pyradioss must
therefore work on both majors rather than assume the 2.x API.

NumPy 2.0 (NEP 52) renamed a handful of long-standing functions. Code written
against the new spelling raises ``AttributeError`` at call time on 1.x — a
runtime failure, not an import-time one, so it hides until the affected code
path is actually exercised. The aliases below resolve the spelling *once, at
import*, against whichever NumPy is installed, and the rest of the package
imports the name from here instead of reaching for ``np.<name>`` directly.

Each shim must be a *rename only*. Anything that changed semantics across the
major boundary belongs in a real adapter with its own tests, not here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trapezoid"]


# ``np.trapz`` -> ``np.trapezoid`` (NEP 52). This is a pure rename: both
# spellings are the same composite trapezoidal integrator with the identical
# signature ``(y, x=None, dx=1.0, axis=-1)`` and identical results, so
# selecting between them cannot perturb a fatigue moment or a PSD integral.
#
# ``np.trapezoid`` is tried first so that on NumPy >= 2.0 the legacy name is
# never touched (it is removed there, and on the 1.x releases that carry the
# forward-port it is deprecated).
try:
    trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2.0
    trapezoid = np.trapz
