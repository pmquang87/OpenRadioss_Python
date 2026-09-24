"""
Tests of the SPMD communicator layer (pyradioss/spmd/comm.py) — the port
of ``engine/source/mpi/spmd_mod.F90`` and ``generic/glob_min.F``.

* SerialComm: every collective is the identity (the serial build, where
  every SPMD_* routine is an empty #else branch);
* ThreadComm group of 3: allreduce SUM/MAX/MIN, allgather, bcast, gather,
  barrier, a ring of isend/irecv, and the glob_min packet semantics of
  glob_min.F (slot 0 MIN, slots 1-2 follow the winner, slots 3/4/6 SUM,
  slot 5 MIN, slots 7-9 MAX);
* Mpi4pyComm under a real ``mpirun -np 2`` (skipped without mpirun or
  mpi4py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

from pyradioss.spmd.comm import (SERIAL, SPMD_MAX, SPMD_MIN, SPMD_PROD,
                                 SPMD_SUM, SerialComm, ThreadComm,
                                 mpi_world_size)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_group(n, fn, timeout=60.0):
    """Run ``fn(comm)`` on every member of a ThreadComm group of ``n``;
    return the per-rank results, re-raising the first exception."""
    comms = ThreadComm.group(n)
    out = [None] * n
    err = [None] * n

    def work(r):
        try:
            out[r] = fn(comms[r])
        except BaseException as exc:  # noqa: BLE001
            err[r] = exc
            try:
                comms[r].abort(2)
            except SystemExit:
                pass

    ths = [threading.Thread(target=work, args=(r,), daemon=True)
           for r in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout)
        assert not t.is_alive(), "thread group deadlocked"
    for e in err:
        if e is not None:
            raise e
    return out


# ---------------------------------------------------------------- serial
def test_serial_comm_identities():
    c = SerialComm()
    assert c.rank == 0 and c.size == 1
    assert not c.active and c.is_root
    a = np.array([1.0, -2.0, 3.0])
    for op in (SPMD_SUM, SPMD_MAX, SPMD_MIN, SPMD_PROD):
        r = c.allreduce(a, op)
        np.testing.assert_array_equal(r, a)
        assert r is not a               # a NEW array (inputs immutable)
    assert c.allreduce_scalar(2.5) == 2.5
    assert c.allgather({"k": 1}) == [{"k": 1}]
    assert c.bcast("x") == "x"
    assert c.gather(7) == [7]
    assert c.barrier() is None
    pk = np.arange(10.0)
    np.testing.assert_array_equal(c.glob_min(pk), pk)
    with pytest.raises(RuntimeError):
        c.isend(a, 0, 1)
    with pytest.raises(RuntimeError):
        c.irecv(a, 0, 1)
    assert SERIAL.size == 1


def test_mpi_world_size_is_one_outside_mpirun(monkeypatch):
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS",
                "PMIX_RANK", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        monkeypatch.delenv(var, raising=False)
    assert mpi_world_size() == 1


# ---------------------------------------------------------------- threads
def test_thread_group_allreduce_ops():
    def fn(c):
        a = np.array([c.rank + 1.0, 10.0 - c.rank])
        return (c.allreduce(a, SPMD_SUM), c.allreduce(a, SPMD_MAX),
                c.allreduce(a, SPMD_MIN), c.allreduce_scalar(c.rank, SPMD_SUM))
    res = run_group(3, fn)
    for s, mx, mn, sc in res:
        np.testing.assert_array_equal(s, [6.0, 27.0])
        np.testing.assert_array_equal(mx, [3.0, 10.0])
        np.testing.assert_array_equal(mn, [1.0, 8.0])
        assert sc == 3.0
    assert all(r[0].tobytes() == res[0][0].tobytes() for r in res)


def test_thread_group_allgather_bcast_gather_barrier():
    def fn(c):
        ag = c.allgather(("r", c.rank))
        bc = c.bcast(f"from{c.rank}", root=2)
        ga = c.gather(c.rank * 10, root=1)
        c.barrier()
        return ag, bc, ga, c.active, c.is_root
    res = run_group(3, fn)
    for r, (ag, bc, ga, active, root) in enumerate(res):
        assert ag == [("r", 0), ("r", 1), ("r", 2)]
        assert bc == "from2"
        assert ga == ([0, 10, 20] if r == 1 else None)
        assert active and root == (r == 0)


def test_thread_group_ring_isend_irecv():
    def fn(c):
        n = c.size
        right, left = (c.rank + 1) % n, (c.rank - 1) % n
        buf = np.empty((4, 2))
        req_r = c.irecv(buf, left, 120)      # post the receive first
        sbuf = np.full((4, 2), float(c.rank))
        req_s = c.isend(sbuf, right, 120)
        sbuf[:] = -1.0                       # the queue owns a copy
        got = req_r.wait()
        req_s.wait()
        return got.copy()
    res = run_group(3, fn)
    for r, got in enumerate(res):
        np.testing.assert_array_equal(got, np.full((4, 2), (r - 1) % 3))


def test_thread_group_message_shape_mismatch_raises():
    def fn(c):
        if c.rank == 0:
            c.isend(np.zeros(3), 1, 5)
            return None
        buf = np.empty(4)
        with pytest.raises(RuntimeError):
            c.irecv(buf, 0, 5).wait()
        return "ok"
    assert run_group(2, fn)[1] == "ok"


def test_glob_min_packet_semantics():
    """glob_min.F slot rules on 3 domains."""
    packets = [
        [5.0e-6, 3, 101, 1, 2, 7.0, 1, 0, 0, 0],
        [2.0e-6, 4, 202, 1, 0, 3.0, 0, 1, 0, 1],
        [9.0e-6, 5, 303, 1, 5, 9.0, 2, 0, 1, 0],
    ]

    def fn(c):
        return c.glob_min(packets[c.rank])
    res = run_group(3, fn)
    for out in res:
        assert out[0] == 2.0e-6              # DT2 MIN
        assert out[1] == 4 and out[2] == 202  # ITYPTS/NELTS of the winner
        assert out[3] == 3.0                  # IEXICODT SUM
        assert out[4] == 7.0                  # IMSCH SUM
        assert out[5] == 3.0                  # TSTOP MIN
        assert out[6] == 3.0                  # IWIOUT SUM
        assert out[7] == 1.0 and out[8] == 1.0 and out[9] == 1.0  # MAX
        assert out.tobytes() == res[0].tobytes()   # identical everywhere


def test_glob_min_tie_breaking():
    """An exact tie on slot 0: comm.py applies the glob_min.F pair rule in
    increasing rank order and the Fortran compares AFTER the min
    (``IF(RIN(1) == RINOUT(1))``), so the LATER-visited packet — the
    HIGHEST rank of the tie — supplies the critical element (documented in
    comm._glob_min_pair)."""
    packets = [
        [1.0e-6, 1, 11, 0, 0, 0, 0, 0, 0, 0],
        [1.0e-6, 2, 22, 0, 0, 0, 0, 0, 0, 0],
        [4.0e-6, 3, 33, 0, 0, 0, 0, 0, 0, 0],
    ]

    def fn(c):
        return c.glob_min(packets[c.rank])
    for out in run_group(3, fn):
        assert out[0] == 1.0e-6
        assert (out[1], out[2]) == (2, 22)


def test_glob_min_rejects_bad_packet():
    with pytest.raises(ValueError):
        SERIAL.glob_min(np.zeros(9))


# ---------------------------------------------------------------- mpirun
_MPI_SNIPPET = r"""
import sys
sys.path.insert(0, {repo!r})
import numpy as np
from pyradioss.spmd.comm import Mpi4pyComm, mpi_world_size
c = Mpi4pyComm()
assert mpi_world_size() == 2 and c.size == 2 and c.active
r = c.allreduce(np.array([c.rank + 1.0, 5.0]))
pk = np.zeros(10); pk[0] = 1.0 + c.rank; pk[2] = 10 + c.rank; pk[3] = 1.0
g = c.glob_min(pk)
buf = np.empty(3)
rq = c.irecv(buf, 1 - c.rank, 120)
c.isend(np.full(3, float(c.rank)), 1 - c.rank, 120).wait()
rq.wait()
with open({out!r} + "_%d.txt" % c.rank, "w") as fh:
    fh.write(" ".join(str(x) for x in (c.rank, r[0], r[1], g[0], g[2], g[3],
                                        buf[0])))
"""


def _have_mpi():
    if shutil.which("mpirun") is None:
        return False
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _have_mpi(), reason="mpirun or mpi4py not available")
def test_mpi4py_comm_under_mpirun(tmp_path):
    out = str(tmp_path / "res")
    env = dict(os.environ)
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    cmd = ["mpirun", "-np", "2", sys.executable, "-c",
           _MPI_SNIPPET.format(repo=REPO, out=out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                       env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [open(f"{out}_{r}.txt").read().split() for r in range(2)]
    for rank, row in enumerate(rows):
        vals = [float(x) for x in row]
        assert int(vals[0]) == rank
        assert vals[1] == 3.0 and vals[2] == 10.0      # allreduce SUM
        assert vals[3] == 1.0 and vals[4] == 10.0      # glob_min winner
        assert vals[5] == 2.0                           # SUM slot
        assert vals[6] == float(1 - rank)               # point-to-point
