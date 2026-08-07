import re

with open("docs/STATE.md", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the "What is implemented (M1 -> M41)" header to M50
text = text.replace("## What is implemented (M1 → M41)", "## What is implemented (M1 → M50)")

# Add the new rows to the table
table_additions = """| M42 | SH3N rotational inertia fix (c3inmas.F alignment) |
| M43–M45 | Auto-backend fallback rules; QBAT and QEPH numba JIT kernels |
| M46 | Removed unused condensed length logic from shell_bt4 |
| M47 | DKT18 shell element ported with Numba acceleration |
| M48–M49 | INTER/TYPE24 parsing and engine logic (forces, broad/narrow phase, dt_int stability) |
| M50 | BT-family cdefo3 branches: c43 node-1-relative velocity form, c45 Z2 warp correction |"""

text = text.replace("| M49 | INTER/TYPE24 stability (dt_int pull-down before impact, penalty damping verification) |", table_additions)

with open("docs/STATE.md", "w", encoding="utf-8") as f:
    f.write(text)

with open(r"C:\Users\pmqua\.gemini\antigravity\brain\ae4b6e32-96b3-40b2-8749-a7e52e5aa382\walkthrough.md", "a", encoding="utf-8") as f:
    f.write("""
## M50: BT Shells cdefo3 Parity Branches (Completed)

- **Implementation**: Ported the missing `cdefo3.F` branches for `IHBE=2/3` and `IHBE=4` formulations into `shell_bt4.py`.
- **Details**: The previous `rot2_mask` logic implicitly gated all velocity-warping corrections to `IHBE <= 1`. We replaced this with an exact `ihbe_mask` generated during geometry preprocessing, allowing the `forces()` loop to conditionally apply the `Z2` node-1-relative velocity and warping vector modifications specific to `IHBE=2,3` and `IHBE=4`.
- **Validation**:
  - The `ihbe_mask` correctness is verified in `test_m41_bt_rotation.py`.
  - Numba JIT kernels gracefully accept these corrections because the modifications act directly on the NumPy vectors *before* they are passed to the `shell_post` routines, maintaining mathematical purity without needing to complicate the kernel interface.
  - The fast-tier test suite passes 100% cleanly with no performance regression.
""")
