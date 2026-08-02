# Antigravity kickoff prompts

Copy-paste these into Antigravity (2.0 agent manager or the IDE — same
harness, same files; workspace pointed at
`C:\Users\pmqua\PycharmProjects\OpenRadioss_Python`, main folder, no
worktree). One milestone = one NEW conversation. Model advice: **Claude Opus
4.6 (thinking)** for physics/numerics milestones (M42, M44, M45, parity
work), **Gemini 3.1 Pro (High)** fine for tooling/reader milestones (M43,
M46); mixing models spreads quota.

## Conversation 0 (optional, recommended): assumption check

```
/grill-me Read AGENTS.md and @docs/STATE.md. You will implement Milestone 42
soon. Before any code: ask me every clarifying question you have about the
sh3n inertia formula, the upstream c3inmas.F reference (C:\OpenRadioss\source,
READ-ONLY), the phase-sensitive tests you expect to shift, and the parity
harness. Do not write code in this conversation.
```

## Per-milestone prompt (N = 42, 43, ...)

```
Read AGENTS.md and @docs/STATE.md. Implement Milestone N ONLY, exactly as
specified in STATE.md's Roadmap entry (goal, files, acceptance).

Process, in order:
1. Read the upstream Fortran the milestone cites (under C:\OpenRadioss\source,
   READ-ONLY) - fetch and quote the actual lines, do not recall them.
2. Produce an Implementation Plan artifact for Milestone N and STOP for my
   review. List: files you will create/modify, the failing tests you will
   write first, the Fortran citations, and any point where you intend to
   deviate from STATE.md (deviations need my approval).
3. After approval: write the failing tests, show them failing with
   .venv\Scripts\python.exe -m pytest -q tests\test_mN_<slug>.py
4. Implement until that file is green. Commit at every green sub-step with
   git add -A and git commit (quota pauses are normal; commits survive).
5. Run the fast tier: .venv\Scripts\python.exe -m pytest -q -m "not slow"
   - everything must be green. If you had to realign any pre-existing test,
   list each one with numeric before/after and why it is a tightening.
6. If the milestone's acceptance line demands validation evidence, run the
   parity command from .agents/skills/validation-compare/SKILL.md and show
   me the numbers (rel_rms per channel, class).
7. Tick the Milestone N checkbox in docs/STATE.md, add a one-line result
   note, commit everything: git commit -m "MN: <summary>". Verify with
   git log --oneline -1 and git status.
8. Produce a Walkthrough artifact and stop. Do NOT start Milestone N+1.

Hard rules (also in AGENTS.md): only .venv\Scripts\python.exe - never bare
python; no pip install; one flat terminal command per call; C:\OpenRadioss,
C:\OpenRadioss_old and all of E:\ are READ-ONLY; never launch pyradioss-gui
or any watch process; never weaken a test.
```

Alternative if the `/milestone` workflow command is picked up from
`.agents/workflows/milestone.md`, simply run:

```
/milestone 42
```

## Review checklist for YOU at each milestone gate

- Implementation Plan artifact matches the STATE.md roadmap entry (no silent
  scope creep)?
- Fortran citations quoted from the actual files (not recalled)?
- Tests were shown RED before implementation?
- Fast tier green — and shown, not asserted?
- Any modified pre-existing test justified numerically as a tightening?
- Physics change ⇒ parity numbers quoted in the walkthrough?
- STATE.md checkbox ticked + committed (git log shows the commit)?
