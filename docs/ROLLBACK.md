# Rollback

How to safely undo a change if a phase turns out wrong, per the
"one request = one logical change" discipline in `.cursorrules` --
small, isolated commits make rollback cheap.

## General procedure

1. Identify the last commit where `pytest -q` fully passed
   (check `docs/TEST_CHECKLIST.md` for what "passed" meant at that point).
2. `git log --oneline` to find that commit hash.
3. `git diff <good_commit> HEAD -- src/ tests/` to see exactly what
   changed since, and confirm the change is actually the cause.
4. Prefer `git revert <bad_commit>` (keeps history, undoes the change)
   over `git reset --hard` (destructive) unless the bad commit was
   never pushed/shared.
5. Re-run `pytest -q` after reverting to confirm the regression is gone.
6. Record what happened in `docs/Decisions.md` if the rollback reflects
   a reversed architectural decision, not just a bug fix.

## Phase-specific notes

- **Storage (`storage/memory.py`) changes:** since deletion is
  tombstone-based (Decision D-003), a bad `delete`/`insert` change
  is unlikely to corrupt data irreversibly within a process -- but
  once Phase 11 (persistence) exists, always roll back the on-disk
  format version alongside the code, never just the code.
- **IVF changes (Phases 5-7, once implemented):** always re-run the
  full correctness suite from `docs/Constraints.md`
  ("Required for every new index") against brute force before trusting
  a rollback fixed things -- a partial revert can leave clustering and
  inverted-list code out of sync.
- **Benchmark/config changes (`configs/default.yaml`):** since
  `.cursorrules` requires every benchmark to state its configuration,
  keep old config values in `docs/Decisions.md` when you change them,
  so past benchmark numbers stay interpretable after a rollback.

## What NOT to roll back silently

Never revert a correctness fix just to make a benchmark number look
better (`.cursorrules`: "Never improve latency by silently reducing
correctness"). If a rollback would do that, stop and document why the
regression is being accepted instead.