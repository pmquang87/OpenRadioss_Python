"""
Guards the NumPy-1.x/2.x portability contract (``pyradioss.common.npcompat``).

``pyproject.toml`` declares ``numpy>=1.22``, and the post-processing stack
(``vortex_radioss`` -> ``lasso-python``) hard-pins ``numpy<2.0.0``, so a NumPy
1.x environment is a *supported and routinely-installed* configuration, not a
legacy corner. NumPy-2.0-only spellings therefore cannot be used bare.

Such a spelling fails at call time rather than import time, so it survives
collection and only detonates when the owning code path runs — which is how the
bare ``trapezoid`` spelling reached seven modules unnoticed. The source scan
below is the cheap import-independent net that catches the next one.

(This file is itself scanned, so it must not name a guarded spelling in the
``np.`` form outside the ``NUMPY2_ONLY`` pattern.)
"""

import pathlib
import re

import numpy as np
import pytest

from pyradioss.common.npcompat import trapezoid

REPO = pathlib.Path(__file__).resolve().parents[1]

# Names introduced by NumPy 2.0 (NEP 52 renames + the array-API surface) that
# do not exist on the declared 1.22 floor. Bare use raises AttributeError at
# call time on 1.x; each needs an alias in pyradioss.common.npcompat instead.
# ``np.astype``/``np.concat``/``np.pow`` are anchored on the opening paren so
# the ubiquitous ``.astype(`` *method* does not false-positive.
NUMPY2_ONLY = re.compile(
    r"\bnp\.(?:"
    r"trapezoid|vecdot|matrix_transpose|permute_dims|isdtype|bitwise_count"
    r"|unique_values|unique_counts|unique_all|unique_inverse"
    r"|strings\.|long\b|ulong\b"
    r"|astype\(|concat\(|pow\("
    r")"
)

# npcompat.py is where the 2.x spelling is legitimately named.
EXEMPT = {REPO / "pyradioss" / "common" / "npcompat.py"}


def _sources():
    for base in ("pyradioss", "tests"):
        for path in (REPO / base).rglob("*.py"):
            if path not in EXEMPT:
                yield path


def test_no_bare_numpy2_only_spellings():
    """No module reaches for a NumPy-2.0-only name outside the compat shim."""
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = NUMPY2_ONLY.search(line)
            if match:
                rel = path.relative_to(REPO).as_posix()
                offenders.append(f"{rel}:{lineno}: {match.group(0)}")

    assert not offenders, (
        "NumPy-2.0-only spelling(s) used bare - these raise AttributeError on "
        "the supported NumPy 1.x floor. Add an alias to "
        "pyradioss/common/npcompat.py and import it from there:\n  "
        + "\n  ".join(offenders)
    )


def test_trapezoid_is_the_composite_trapezoidal_rule():
    """The shim resolves to the real integrator, not a stub or a stale name."""
    # Non-uniform grid, so a mistaken dx=1 default would show up.
    x = np.array([0.0, 0.5, 2.0, 2.5])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    expected = 0.5 * ((1.0 + 3.0) * 0.5 + (3.0 + 2.0) * 1.5 + (2.0 + 4.0) * 0.5)
    assert trapezoid(y, x) == pytest.approx(expected, rel=0, abs=0)

    # Exact on a linear integrand: int_0^1 2x dx = 1.
    s = np.linspace(0.0, 1.0, 101)
    assert trapezoid(2.0 * s, s) == pytest.approx(1.0, rel=1e-12)


def _reload_npcompat():
    import importlib

    from pyradioss.common import npcompat

    return importlib.reload(npcompat)


def test_prefers_the_numpy2_spelling_when_present(monkeypatch):
    """On NumPy >= 2.0 the shim must bind the new name, never the legacy one.

    That branch is unreachable on a 1.x machine, so force it: inject the 2.x
    spelling and re-resolve. Without this the 2.x path ships untested from any
    NumPy 1.x developer or CI environment.
    """
    sentinel = object()
    monkeypatch.setattr(np, "trapezoid", sentinel, raising=False)
    try:
        assert _reload_npcompat().trapezoid is sentinel
    finally:
        monkeypatch.undo()
        _reload_npcompat()


@pytest.mark.skipif(not hasattr(np, "trapz"),
                    reason="legacy spelling already removed (NumPy >= 2.0)")
def test_falls_back_to_the_legacy_spelling_when_absent(monkeypatch):
    """On NumPy < 2.0 the shim must fall back rather than raise at import."""
    monkeypatch.delattr(np, "trapezoid", raising=False)
    try:
        assert _reload_npcompat().trapezoid is np.trapz
    finally:
        monkeypatch.undo()
        _reload_npcompat()


def test_trapezoid_keeps_the_axis_and_dx_signature():
    """Call sites pass ``x`` positionally and ``axis=`` by keyword."""
    y = np.arange(12.0).reshape(4, 3)
    x = np.array([0.0, 1.0, 3.0, 6.0])
    got = trapezoid(y, x, axis=0)
    assert got.shape == (3,)

    # dx= path (no x) must also survive the rename.
    assert trapezoid(np.array([0.0, 1.0, 2.0]), dx=2.0) == pytest.approx(4.0)
