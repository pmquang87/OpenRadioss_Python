# Subagent Parallelization Preference

Always maximize parallelism by using as many **Gemini Flash 3.8 High**
(`Model: "flash"`) subagents as possible for independent tasks:

- Research tasks (code review, Fortran verification, dispatch audits) → `research` type, `Model: "flash"`
- Write tasks (test runs, commits, file edits) → `worker` type (define if needed), `Model: "flash"`
- Launch all independent tasks simultaneously in a single `invoke_subagent` call
- Don't serialize work that can be parallelized
