# DKT18 Unrolling Task
You are tasked with unrolling specific functions from the OpenRadioss Fortran DKT18 formulation into Python scalar code.
The Fortran source files are located at: `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\sh3n\coquedk\`.
You must write explicit scalar loops over elements (`n`). DO NOT use `np.einsum`, `np.dot`, `np.cross`, or array-returning slicing inside the loops.
You must use basic math operations (+, -, *, /, np.sqrt, abs).

Example:
```python
import numpy as np

def unrolled_func(xe, ve):
    n = len(xe)
    out = np.empty(n)
    for e in range(n):
        rx = xe[e, 1, 0] - xe[e, 0, 0]
        # ...
```
Save your result in `C:\Users\pmqua\.gemini\antigravity\brain\ae4b6e32-96b3-40b2-8749-a7e52e5aa382\scratch\dkt18_unrolled_{TASK_NAME}.py`.

Tasks:
1. Geometry (`cdkcoor3.F`): `CDKCOOR3` - corotational frame and local coordinates.
2. Derivatives (`cdkderi3.F`): `CDKDERI3` - derivatives and gradient operators.
3. Rates (`cdkdefo3.F`): `CDKDEFO3`, `CDKCURV3` - membrane and bending rates.
4. Assembly (`cdkfint3.F`): `CDKFINT3`, `CDKFCUM3` - internal forces and resultants.
