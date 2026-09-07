# Decisions

Architectural decision log. Per `.cursorrules`, every meaningful
decision records: context, options, decision, reason, tradeoffs,
impact, and AI model/version when known.

---

## D-001: Zero-vector cosine similarity is defined as 0.0

- **Context:** Cosine similarity is undefined for a zero vector
  (division by a zero norm). Silently producing `NaN` would let
  invalid states leak into top-k selection.
- **Options considered:**
  1. Raise an exception on any zero vector.
  2. Return `NaN` and let callers handle it.
  3. Define similarity with a zero vector as `0.0`.
- **Decision:** Option 3.
- **Reason:** Keeps the function total (always returns a comparable
  float), matches the intuition that "no direction" means "no
  similarity", and avoids `NaN` propagating into sorting/top-k logic.
- **Tradeoffs:** A zero vector is treated as equally (dis)similar to
  everything, which may not be desired in every downstream use --
  callers who need different behavior should filter zero vectors
  explicitly before searching.
- **Impact:** `core/distance.py` (`cosine_similarity`,
  `batch_cosine_similarity`), covered by
  `tests/unit/test_distance.py::test_zero_vector_is_defined_as_zero_similarity`.
- **AI model/version:** Claude (Sonnet 5), repository scaffolding session.

---

## D-002: Deterministic tie-breaking by ascending id

- **Context:** `architecture.md` requires deterministic tie-breaking
  when scores are equal, but doesn't fix the exact rule.
- **Options considered:**
  1. Leave tie order undefined (whatever the sort algorithm returns).
  2. Break ties by insertion order.
  3. Break ties by ascending id.
- **Decision:** Option 3.
- **Reason:** Ids are already the stable identity of a record;
  sorting by id is simple, testable, and doesn't require tracking
  insertion order separately.
- **Tradeoffs:** None significant at this scale; revisit if ids are
  ever non-comparable (e.g. UUIDs) in a future phase.
- **Impact:** `exact/brute_force.py::search`.
- **AI model/version:** Claude (Sonnet 5), repository scaffolding session.

---

## D-003: Deletion strategy is tombstoning, not immediate removal

- **Context:** `architecture.md` section 7 specifies marking vectors
  deleted while retaining physical position, deferring compaction.
- **Decision:** `MemoryStorage.delete` sets `active=False` and keeps
  the record; `as_matrix()` / `active_records()` filter by `active`.
- **Reason:** Simple and safe; avoids rebuilding IVF clusters on every
  delete (relevant once Phase 6/7 land).
- **Tradeoffs:** Memory is not reclaimed until compaction (Phase 12).
- **Impact:** `storage/memory.py`.
- **AI model/version:** Claude (Sonnet 5), repository scaffolding session.

---

<!-- Add new decisions above this line, most recent last, following the same template. -->