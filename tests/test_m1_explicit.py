"""Tests for M1 explicit basics and starter initialization."""
import numpy as np
import pytest
from pyradioss.model.model import Model
from pyradioss.model.entities import InitialVelocity
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_skews
from pyradioss.model.skew import SkewFrame


def test_inivel_double_rotation_guard():
    model = Model()
    log = MessageLog()
    iv = InitialVelocity(id=1, grnod_id=1, v=np.array([10.0, 0.0, 0.0]), kind="TRANS", iskew=1)
    model.inivel.append(iv)

    model.skews.add(SkewFrame(id=1, kind="SKEW", subtype="FIX",
                              origin_card=np.zeros(3),
                              yaxis=np.array([0.0, 1.0, 0.0]),
                              zaxis=np.array([0.0, 0.0, 1.0])))
    model.skews.resolve(model, log)
    resolve_skews(model, log)
    v_first = iv.v.copy()

    # Second resolve (e.g. after /TRANSFORM)
    resolve_skews(model, log)
    assert np.allclose(iv.v, v_first)
    assert getattr(iv, "_v_is_global", False) is True


def test_inivel_axis_double_rotation_guard():
    model = Model()
    log = MessageLog()
    iv = InitialVelocity(id=1, grnod_id=1, v=np.array([0.0, 5.0, 0.0]), kind="AXIS", frame_id=1, dir=3)
    model.inivel.append(iv)

    model.skews.add(SkewFrame(id=1, kind="FRAME", subtype="FIX",
                              origin_card=np.zeros(3),
                              yaxis=np.array([0.0, 1.0, 0.0]),
                              zaxis=np.array([0.0, 0.0, 1.0])))
    model.skews.resolve(model, log)
    resolve_skews(model, log)
    v_first = iv.v.copy()

    # Second resolve (e.g. after /TRANSFORM)
    resolve_skews(model, log)
    assert np.allclose(iv.v, v_first)
    assert getattr(iv, "_v_is_global", False) is True
