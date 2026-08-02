---
description: Implement one milestone of docs/STATE.md §Roadmap, test-first, with review gate
---

1. Read AGENTS.md and docs/STATE.md in full (they are short; the big
   PORTING_GUIDE.md / VALIDATION.md are grep-only reference).
2. Identify Milestone $1 in docs/STATE.md §Roadmap. Do not touch any other
   milestone. Read the Fortran files the milestone entry cites (under
   C:\OpenRadioss\source, READ-ONLY) before planning.
3. Produce an Implementation Plan artifact for Milestone $1 only: files to
   create/modify, the failing tests to write first, the Fortran citations,
   and any intended deviation from the milestone entry. STOP for review.
4. After approval: write the failing tests in tests\test_m$1_<slug>.py, show
   them red: `.venv\Scripts\python.exe -m pytest -q tests\test_m$1_<slug>.py`
5. Implement minimally until that file is green. Commit at every green
   sub-step (`git add -A`, then `git commit`) — interruptions are normal.
6. Run the fast tier: `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
   — everything green. If the milestone realigned any pre-existing test,
   list each one with the numeric before/after and why it is a tightening.
7. If the milestone entry demands validation evidence (physics change), run
   the parity command given in its acceptance line (see the
   validation-compare skill) and quote the numbers.
8. Tick the milestone checkbox in docs/STATE.md, add a one-line result note.
9. `git add -A` then `git commit -m "M$1: <summary>"`. Verify with
   `git log --oneline -1` and `git status`.
10. Produce a Walkthrough artifact and stop. The next milestone gets a fresh
    conversation.
