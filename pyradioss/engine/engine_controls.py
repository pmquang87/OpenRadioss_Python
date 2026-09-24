"""
Engine controls module (M614 Component 5 & 6).

Exposes EngineControls for engine card configuration, parsing, and serialization.
Originates from OpenRadioss Engine input files (*_0001.rad) and freimpl.F directives.
"""

from __future__ import annotations

from ..model.model import EngineControls

__all__ = ["EngineControls"]
