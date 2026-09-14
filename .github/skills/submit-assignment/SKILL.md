---
name: submit-assignment
description: Guarded submission for Assignment 6; use when the user says submit assignment 6.
---

# Submit assignment 6

1. Do not edit any files during submission.
2. Run `git status --short` and inspect staged and unstaged diffs.
3. Only `AGENTS.md` and `assignment6.py` may be changed, untracked, or staged. Stop and report any other path, including the provided skill, tests, or data. Do not hide unexpected changes.
4. Run `python -m unittest -v` and `git diff --check`. Stop on any failure; never bypass tests.
5. Stage exactly `git add -- AGENTS.md assignment6.py`; never use `git add .`.
6. Inspect `git diff --cached --check` and `git diff --cached --name-only`; stop if checks fail or staged paths are not limited to the two deliverables. If nothing is staged, do not create an empty commit; verify the existing commit instead.
7. Commit with `git commit -m "Complete assignment 6"`, then run `git push origin HEAD`. Stop on any failure.
8. Run `git status --short` and `git log -1 --oneline`, verify the GitHub Actions run for the pushed commit, and report the commit and result. If Actions is pending or unavailable, say so; do not claim success.
