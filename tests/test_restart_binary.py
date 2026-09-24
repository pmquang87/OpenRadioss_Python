"""
Tests for Fortran-compatible binary restart I/O (Work Stream 16).

Verifies:
- FortranBinaryFile read/write round-trip for integers, floats, arrays, strings
- Correct 4-byte record markers (Fortran unformatted sequential format)
- write_restart_binary -> read_restart_binary round-trip preserves model and engine state
- Auto-detection in read_restart: pickle detected as pickle, binary as binary
- Chaining works with binary restart (write run 1, resume in run 2)
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from pyradioss.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter import run_starter
from pyradioss.starter.restart import (
    FortranBinaryFile,
    is_binary_restart,
    read_restart,
    read_restart_binary,
    write_restart,
    write_restart_binary,
)


class TestFortranBinaryFile:
    """Test low-level Fortran unformatted sequential record reader/writer."""

    def test_record_markers_layout(self, tmp_path: Path) -> None:
        """Verify 4-byte record marker layout: [4-byte len] [data] [4-byte len]."""
        file_path = tmp_path / "test_marker.bin"
        val = 12345

        with FortranBinaryFile(file_path, "wb") as fb:
            fb.write_int(val)

        # Inspect raw bytes on disk
        raw = file_path.read_bytes()
        assert len(raw) == 12  # 4 header + 4 payload + 4 footer

        hdr_len = struct.unpack("<i", raw[:4])[0]
        data_val = struct.unpack("<i", raw[4:8])[0]
        ftr_len = struct.unpack("<i", raw[8:12])[0]

        assert hdr_len == 4
        assert data_val == 12345
        assert ftr_len == 4

    def test_record_marker_mismatch_raises(self, tmp_path: Path) -> None:
        """Corrupted footer marker must raise ValueError."""
        file_path = tmp_path / "corrupt_marker.bin"
        # Write valid header 4, data 4 bytes, but corrupt footer with 8
        bad_data = struct.pack("<i", 4) + b"\x01\x02\x03\x04" + struct.pack("<i", 8)
        file_path.write_bytes(bad_data)

        with FortranBinaryFile(file_path, "rb") as fb:
            with pytest.raises(ValueError, match="marker mismatch"):
                fb.read_record()

    def test_truncated_record_raises(self, tmp_path: Path) -> None:
        """Truncated payload must raise ValueError."""
        file_path = tmp_path / "trunc.bin"
        # Header says 10 bytes, but only 4 bytes provided
        bad_data = struct.pack("<i", 10) + b"\x01\x02\x03\x04"
        file_path.write_bytes(bad_data)

        with FortranBinaryFile(file_path, "rb") as fb:
            with pytest.raises(ValueError, match="Truncated"):
                fb.read_record()

    def test_scalars_roundtrip(self, tmp_path: Path) -> None:
        """Verify round-trip of integers, floats, and strings."""
        file_path = tmp_path / "scalars.bin"

        ints = [0, 1, -1, 42, -999999, 2143942393]
        floats = [0.0, -0.0, 1.0, -3.141592653589793, 1.2345e-8, 1.99e30]
        strings = ["HELLO", "OPENRADIOSS", "SHORT", "A" * 100]

        with FortranBinaryFile(file_path, "wb") as fb:
            for ival in ints:
                fb.write_int(ival)
            for fval in floats:
                fb.write_float(fval)
            for sval in strings:
                fb.write_string(sval, length=len(sval) + 4)

        with FortranBinaryFile(file_path, "rb") as fb:
            for ival in ints:
                read_i = fb.read_int()
                assert read_i == ival
            for fval in floats:
                read_f = fb.read_float()
                assert read_f == fval
            for sval in strings:
                read_s = fb.read_string(length=len(sval) + 4)
                assert read_s.strip() == sval

    def test_arrays_roundtrip(self, tmp_path: Path) -> None:
        """Verify round-trip for 1D, 2D, and 3D numpy arrays across dtypes."""
        file_path = tmp_path / "arrays.bin"

        arr_i32 = np.array([10, -20, 30, -40, 50], dtype=np.int32)
        arr_i64 = np.arange(100, dtype=np.int64)
        arr_f64_2d = np.linspace(0.0, 10.0, 24).reshape((8, 3))
        arr_f32_3d = np.arange(60, dtype=np.float32).reshape((3, 4, 5))
        arr_empty = np.zeros((0, 3), dtype=np.float64)

        with FortranBinaryFile(file_path, "wb") as fb:
            fb.write_array(arr_i32)
            fb.write_array(arr_i64)
            fb.write_array(arr_f64_2d)
            fb.write_array(arr_f32_3d)
            fb.write_array(arr_empty)

        with FortranBinaryFile(file_path, "rb") as fb:
            r_i32 = fb.read_array(np.int32, count=len(arr_i32))
            r_i64 = fb.read_array(np.int64, count=100)
            r_f64 = fb.read_array(np.float64, count=(8, 3))
            r_f32 = fb.read_array(np.float32, count=(3, 4, 5))
            r_empty = fb.read_array(np.float64, count=(0, 3))

        np.testing.assert_array_equal(r_i32, arr_i32)
        np.testing.assert_array_equal(r_i64, arr_i64)
        np.testing.assert_array_equal(r_f64, arr_f64_2d)
        np.testing.assert_array_equal(r_f32, arr_f32_3d)
        assert r_empty.shape == (0, 3)

    def test_in_memory_bytesio(self) -> None:
        """FortranBinaryFile works with in-memory BytesIO streams."""
        buf = io.BytesIO()
        fb_write = FortranBinaryFile(buf, "wb")
        fb_write.write_string("TEST_IN_MEMORY")
        fb_write.write_int(999)

        buf.seek(0)
        fb_read = FortranBinaryFile(buf, "rb")
        assert fb_read.read_string() == "TEST_IN_MEMORY"
        assert fb_read.read_int() == 999


class TestRestartBinaryRoundtrip:
    """Test full write_restart_binary and read_restart_binary serialization."""

    def _build_test_model(self) -> Model:
        model = Model()
        model.title = "TEST_BINARY_RESTART_MODEL"

        # 8 nodes
        n = 8
        model.node_ids = np.arange(1, n + 1, dtype=np.int64)
        model.x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        model.x0 = model.x.copy()
        model.v = np.ones((n, 3), dtype=np.float64) * 0.5
        model.vr = np.zeros((n, 3), dtype=np.float64)
        model.a = np.zeros((n, 3), dtype=np.float64)
        model.mass = np.ones(n, dtype=np.float64) * 2.5
        model.mass0 = model.mass.copy()
        model.inertia = np.zeros(n, dtype=np.float64)
        model._id2idx = {int(nid): i for i, nid in enumerate(model.node_ids)}

        # Brick element group
        brick_ids = np.array([101], dtype=np.int64)
        brick_conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        brick_part = np.array([1], dtype=np.int64)
        brick_state = {
            "sig": np.array([[100.0, 50.0, 20.0, 5.0, 2.0, 1.0]], dtype=np.float64),
            "eps": np.array([[0.01, 0.005, 0.002, 0.0, 0.0, 0.0]], dtype=np.float64),
            "epsp": np.array([0.005], dtype=np.float64),
            "eint": np.array([15.2], dtype=np.float64),
            "dmg": np.array([0.15], dtype=np.float64),
            "off": np.array([1.0], dtype=np.float64),
            "mat_extra": {
                "uvar": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
                "off60": np.array([1.0], dtype=np.float64),
            },
        }
        model.bricks = ElementGroup(
            ids=brick_ids, conn=brick_conn, part=brick_part, state=brick_state
        )

        # Shell element group
        shell_ids = np.array([201], dtype=np.int64)
        shell_conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
        shell_part = np.array([2], dtype=np.int64)
        shell_state = {
            "sig": np.zeros((1, 3, 3), dtype=np.float64),
            "eint": np.array([2.5], dtype=np.float64),
            "epsp": np.array([0.0], dtype=np.float64),
        }
        model.shells = ElementGroup(
            ids=shell_ids, conn=shell_conn, part=shell_part, state=shell_state
        )

        # Entity metadata
        model.materials[1] = Material(id=1, law=1, rho0=7.8e-6, params={"E": 210000.0, "nu": 0.3})
        model.properties[1] = Property(id=1, type=14, title="SolidProp")
        model.parts[1] = Part(id=1, prop_id=1, mat_id=1)
        model.parts_list = [model.parts[1]]

        return model

    def test_starter_restart_roundtrip(self, tmp_path: Path) -> None:
        """Starter restart (engine=None) preserves model state and returns engine_state=None."""
        model = self._build_test_model()
        rst_path = tmp_path / "starter_test.rst"

        write_restart_binary(model, str(rst_path), engine=None)
        assert rst_path.exists()

        m_read, eng_read = read_restart_binary(str(rst_path))
        assert eng_read is None
        assert m_read.title == model.title
        assert m_read.numnod == model.numnod

        # Node arrays
        np.testing.assert_array_equal(m_read.node_ids, model.node_ids)
        np.testing.assert_allclose(m_read.x, model.x, rtol=1e-15)
        np.testing.assert_allclose(m_read.x0, model.x0, rtol=1e-15)
        np.testing.assert_allclose(m_read.v, model.v, rtol=1e-15)
        np.testing.assert_allclose(m_read.mass, model.mass, rtol=1e-15)

        # Element groups
        groups = dict(m_read.element_groups())
        assert "bricks" in groups
        assert "shells" in groups

        bg = groups["bricks"]
        np.testing.assert_array_equal(bg.ids, model.bricks.ids)
        np.testing.assert_array_equal(bg.conn, model.bricks.conn)
        np.testing.assert_array_equal(bg.part, model.bricks.part)
        np.testing.assert_allclose(bg.state["sig"], model.bricks.state["sig"])
        np.testing.assert_allclose(bg.state["epsp"], model.bricks.state["epsp"])
        np.testing.assert_allclose(bg.state["eint"], model.bricks.state["eint"])

        # Entities
        assert 1 in m_read.materials
        assert m_read.materials[1].law == 1
        assert 1 in m_read.parts

    def test_engine_restart_roundtrip(self, tmp_path: Path) -> None:
        """Engine restart preserves both model state and full engine snapshot."""
        model = self._build_test_model()
        rst_path = tmp_path / "engine_test.rst"

        eng_snapshot = {
            "t": 0.025,
            "cycle": 1500,
            "dt": 1.5e-5,
            "dt_prev": 1.48e-5,
            "wext": 125.4,
            "econt": 8.3,
            "epeak": 150.0,
            "ndel": 3,
            "e_num": -0.02,
            "e_madd": 0.05,
            "e_damp": 0.12,
            "e0": 1000.0,
            "next_th": 0.03,
            "next_anim": 0.04,
            "anim_no": 7,
            "next_state": 0.05,
            "sensors": {1: 0.015, 2: 0.0},
            "sensors_status": {1: True, 2: False},
            "rbodies": {
                1: {
                    "R": np.eye(3),
                    "L": np.array([0.1, 0.2, 0.3]),
                    "v_ref": np.array([1.0, 0.0, 0.0]),
                    "w": np.array([0.0, 0.0, 0.5]),
                    "x_ref": np.array([0.5, 0.5, 0.5]),
                    "xg": np.array([0.5, 0.5, 0.5]),
                }
            },
            "noda": {
                "mass_added": 0.08,
                "iner_added": 0.01,
                "e_madd": 0.02,
                "mom_added": np.array([0.1, 0.2, 0.3]),
                "mass0": 20.0,
                "reported": True,
            },
            "dyn_relax": {"ke_prev": 3.45},
            "custom_metadata": "EXTRA_KEY_PRESERVED",
        }

        write_restart_binary(model, str(rst_path), engine=eng_snapshot)
        m_read, eng_read = read_restart_binary(str(rst_path))

        assert eng_read is not None
        assert eng_read["cycle"] == 1500
        assert eng_read["t"] == pytest.approx(0.025)
        assert eng_read["dt"] == pytest.approx(1.5e-5)
        assert eng_read["dt_prev"] == pytest.approx(1.48e-5)
        assert eng_read["wext"] == pytest.approx(125.4)
        assert eng_read["econt"] == pytest.approx(8.3)
        assert eng_read["ndel"] == 3
        assert eng_read["anim_no"] == 7

        # Rigid bodies
        assert 1 in eng_read["rbodies"]
        np.testing.assert_allclose(eng_read["rbodies"][1]["R"], np.eye(3))
        np.testing.assert_allclose(eng_read["rbodies"][1]["L"], [0.1, 0.2, 0.3])

        # Sensors
        assert eng_read["sensors"][1] == pytest.approx(0.015)
        assert eng_read["sensors_status"][1] is True
        assert eng_read["sensors_status"][2] is False

        # Noda
        assert eng_read["noda"] is not None
        assert eng_read["noda"]["mass_added"] == pytest.approx(0.08)
        assert eng_read["noda"]["reported"] is True
        np.testing.assert_allclose(eng_read["noda"]["mom_added"], [0.1, 0.2, 0.3])

        # Extra keys preserved
        assert eng_read["custom_metadata"] == "EXTRA_KEY_PRESERVED"


class TestFormatSelectionAndAutoDetection:
    """Test format='pickle' vs format='binary' and auto-detection in read_restart."""

    def test_auto_detection_pickle_and_binary(self, tmp_path: Path) -> None:
        """read_restart auto-detects pickle files as pickle, binary files as binary."""
        model = Model()
        model.title = "AUTODETECT_TEST"
        model.node_ids = np.array([1, 2], dtype=np.int64)
        model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        model.mass = np.array([1.0, 1.0], dtype=np.float64)
        model._id2idx = {1: 0, 2: 1}

        path_pickle = str(tmp_path / "model_pickle.rst")
        path_binary = str(tmp_path / "model_binary.rst")

        # Write both formats
        write_restart(model, path_pickle, format="pickle")
        write_restart(model, path_binary, format="binary")

        # Check raw magic detection
        assert not is_binary_restart(open(path_pickle, "rb").read(32))
        assert is_binary_restart(open(path_binary, "rb").read(32))

        # read_restart must load both transparently
        m_p, eng_p = read_restart(path_pickle)
        m_b, eng_b = read_restart(path_binary)

        assert eng_p is None
        assert eng_b is None
        assert m_p.title == m_b.title == "AUTODETECT_TEST"
        np.testing.assert_array_equal(m_p.node_ids, m_b.node_ids)
        np.testing.assert_allclose(m_p.x, m_b.x)
        np.testing.assert_allclose(m_p.mass, m_b.mass)

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        """Passing an unrecognized format name to write_restart raises ValueError."""
        model = Model()
        with pytest.raises(ValueError, match="Unknown restart format"):
            write_restart(model, str(tmp_path / "bad.rst"), format="yaml")

    def test_corrupted_file_raises_value_error(self, tmp_path: Path) -> None:
        """Corrupted/non-restart file raises ValueError."""
        bad_path = tmp_path / "corrupt.rst"
        bad_path.write_bytes(b"NOT_A_RESTART_FILE_AT_ALL_1234567890")
        with pytest.raises(ValueError, match="not a pyradioss restart file"):
            read_restart(str(bad_path))


class TestEngineChainingWithBinaryRestart:
    """Test engine simulation chaining (Run 1 -> Run 2) resuming from binary restart."""

    def test_engine_chaining_binary(self, tmp_path: Path) -> None:
        """Run starter, convert/write _0000.rst to binary, run Leg 1, convert/write

        _0001.rst to binary, run Leg 2. Leg 2 resumes from binary restart seamlessly.
        """
        name = "chain_bin"

        # 1. Build starter input deck
        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0),
            (6, 1.0, 0.0, 1.0),
            (7, 1.0, 1.0, 1.0),
            (8, 0.0, 1.0, 1.0),
        ]
        deck = StarterDeck(name)
        deck.node(nodes)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "PartHexa", 1, 1)
        deck.mat_law1(1, "Steel", 7.8e-6, 210000.0, 0.3)
        deck.prop_solid(1, "PropHexa")
        deck.grnod_node(1, "AllNodes", [1, 2, 3, 4, 5, 6, 7, 8])
        deck.inivel_tra(1, "Kick", [10.0, 0.0, 0.0], 1)

        starter_file = tmp_path / f"{name}_0000.rad"
        starter_file.write_text(deck.render(), encoding="utf-8")

        # Run Starter to generate initial model
        m_start = run_starter(str(starter_file))
        assert m_start is not None
        rst0_path = tmp_path / f"{name}_0000.rst"
        assert rst0_path.exists()

        # Re-save _0000.rst in Fortran binary format
        write_restart(m_start, str(rst0_path), engine=None, format="binary")
        assert is_binary_restart(open(rst0_path, "rb").read(32))

        # 2. Engine Leg 1: run to 1.0e-5 s (resumes from binary _0000.rst)
        eng1_path = tmp_path / f"{name}_0001.rad"
        eng1_path.write_text(
            f"/RUN/{name}/1\n1.0e-5\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg1 = run_engine(str(eng1_path))
        assert m_leg1 is not None
        assert m_leg1.engine_state.cycle > 0
        rst1_path = tmp_path / f"{name}_0001.rst"
        assert rst1_path.exists()

        # Read _0001.rst and re-save in Fortran binary format
        m_rst1, eng_rst1 = read_restart(str(rst1_path))
        assert eng_rst1 is not None
        assert eng_rst1["cycle"] == m_leg1.engine_state.cycle
        write_restart(m_rst1, str(rst1_path), engine=eng_rst1, format="binary")
        assert is_binary_restart(open(rst1_path, "rb").read(32))

        # 3. Engine Leg 2: resumes from binary _0001.rst and runs to 2.0e-5 s
        eng2_path = tmp_path / f"{name}_0002.rad"
        eng2_path.write_text(
            f"/RUN/{name}/2\n2.0e-5\n/DT\n0.5 0.0\n/PRINT/-1\n",
            encoding="utf-8",
        )
        m_leg2 = run_engine(str(eng2_path))
        assert m_leg2 is not None
        assert m_leg2.engine_state.cycle > m_leg1.engine_state.cycle
        assert m_leg2.engine_state.t > m_leg1.engine_state.t

        rst2_path = tmp_path / f"{name}_0002.rst"
        assert rst2_path.exists()
