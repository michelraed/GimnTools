# Codex instructions — DOI/SV-PSF task

You are working in GimnTools (Python, Numba, Tkinter), on branch `codex/doi-svpsf-20261118`.
**Read `docs/DOI_SVPSF_TASK.md` first; it is the authoritative scoped task.**
The backup branch `backup/main-pre-doi-svpsf-20261118` must never be modified.
Implement and test incrementally on this task branch. Preserve legacy results and report real test results and scientific limitations; synthetic tests alone do not validate the scanner.

**Release instruction from owner:** After the full implementation passes required checks, integrate the finished code into `main` (push/merge), **without AI instruction/task artifacts in `main`**. Explicitly exclude `AGENTS.md`, `docs/DOI_SVPSF_TASK.md`, any new AI prompt/plan/transcript files and agent scratchpads from the final `main` tree and PR diff. Preserve legitimate user-facing scientific documentation, examples and tests. Prefer a clean integration PR and merge it when checks pass and permissions allow; do not force-push. If tests fail, a fundamental scientific blocker remains, or GitHub permissions/policies prevent merge, keep `main` unchanged, report the blocker and leave the PR for review. Before completion, verify `main` includes implemented code/UI but not the excluded AI files.
