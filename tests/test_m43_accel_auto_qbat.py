import pytest
import os
from pyradioss.accel import auto_select_backend, _state, _AUTO_MIN_ELEMENTS

class MockGroup:
    def __init__(self, n):
        self.n = n

class MockModel:
    def __init__(self, groups):
        self.groups = groups
    
    def element_groups(self):
        for k, v in self.groups.items():
            yield k, v

@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "auto")
    _state.update(name=None, mod=None, forced=False)
    yield
    _state.update(name=None, mod=None, forced=False)

def test_auto_select_backend_qbat_exclusion():
    # Model with >= _AUTO_MIN_ELEMENTS elements but contains QBAT
    model_qbat = MockModel({"shells_qbat": MockGroup(_AUTO_MIN_ELEMENTS)})
    
    # Should fall back to numpy despite being large enough
    backend = auto_select_backend(model_qbat, log=None, explicit=True)
    assert backend == "numpy", f"Expected numpy for QBAT deck, got {backend}"
    assert _state["name"] == "numpy"
    assert not _state["forced"]

def test_auto_select_backend_qeph_exclusion():
    # Model with >= _AUTO_MIN_ELEMENTS elements but contains QEPH
    model_qeph = MockModel({"shells_qeph": MockGroup(_AUTO_MIN_ELEMENTS)})
    
    # Should fall back to numpy despite being large enough
    backend = auto_select_backend(model_qeph, log=None, explicit=True)
    assert backend == "numpy", f"Expected numpy for QEPH deck, got {backend}"
    assert _state["name"] == "numpy"
    assert not _state["forced"]

def test_auto_select_backend_normal():
    # Model with >= _AUTO_MIN_ELEMENTS elements, no QBAT/QEPH
    model_normal = MockModel({"shells": MockGroup(_AUTO_MIN_ELEMENTS)})
    
    # Should select numba (if installed) or numpy. Assuming numba is installed in this env.
    try:
        import numba # noqa
        expected = "numba"
    except ImportError:
        expected = "numpy"

    backend = auto_select_backend(model_normal, log=None, explicit=True)
    assert backend == expected, f"Expected {expected} for normal deck, got {backend}"
