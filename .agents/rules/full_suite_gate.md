
After completing every milestone (all sub-steps committed and green on fast tier),
run the **full** test suite as the final gate — not just the fast tier:

```
.venv\Scripts\python.exe -m pytest -q
```

This includes the 16 `slow`-marked tests (~25–35 min total) and catches
regressions that the fast tier (`-m "not slow"`) misses.
Only after the full suite is green should the milestone be declared done.
