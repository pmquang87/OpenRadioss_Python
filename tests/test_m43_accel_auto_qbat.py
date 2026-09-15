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
