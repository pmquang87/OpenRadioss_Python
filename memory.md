# Memory & Development State

**Current State**: Working on M43 (Exclude QBAT/QEPH from numba auto rule).
- Created implementation plan for M43.
- Approved by user.
- Fixed the logic in `pyradioss/accel/__init__.py`.
- Wrote and passed tests for M43.
- Fast tier tests for M43 are running.
- Profiling timing check on `rigid_impactor` (QBAT deck) is running to get numpy vs numba timings.

**Important Info**:
- The project aims to implement features one milestone at a time (currently M42 to M46).
- M42 is complete.
- M43 excludes QBAT/QEPH elements from the auto `numba` rule because they lack JIT kernels, meaning numba falls back to pure Python loops for them which are slower than pure NumPy.
