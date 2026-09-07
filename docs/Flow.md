# Flow

Concrete, current data-flow diagrams (as implemented today, not
aspirational -- see `architecture.md` for the full target design).

## Insert (exact index, implemented)

```text
VectorIndex.insert(id, vector)
        |
        v
BruteForceIndex.insert(id, vector)
        |
        v
MemoryStorage.insert(id, vector)
        |
        +-- validate_vector(vector, expected_dim)
        +-- fix self._dim on first insert
        +-- reject duplicate active id
        |
        v
VectorRecord(id, vector, active=True) stored
```

## Search (exact index, implemented)

```text
VectorIndex.search(query, k)
        |
        v
BruteForceIndex.search(query, k)
        |
        +-- reject k <= 0
        +-- MemoryStorage.as_matrix() -> (ids, matrix) over active records
        +-- batch_cosine_similarity(query, matrix)
        |     +-- dimension check
        |     +-- zero-vector rows/query score 0.0
        +-- sort by (-score, id)  [deterministic tie-break]
        +-- take first k
        |
        v
List[SearchResult(id, score)]
```

## Delete (exact index, implemented)

```text
VectorIndex.delete(id)
        |
        v
BruteForceIndex.delete(id)
        |
        v
MemoryStorage.delete(id)
        |
        +-- reject unknown/already-inactive id
        +-- mark record.active = False   (tombstone, see Decisions D-003)
```

## Planned: IVF-Flat search (Phase 7, not yet implemented)

```text
query
  |
  v
find nearest centroids (clustering.py, Phase 5)
  |
  v
select n_probe clusters
  |
  v
collect candidate ids from inverted_lists.py (Phase 6)
  |
  v
exact cosine similarity on candidates only
  |
  v
top-k (same SearchResult contract as brute force)
```

## Planned: Benchmark (Phase 8, not yet implemented)

```text
configs/default.yaml
       |
       v
scripts/generate_data.py  --> data/processed/ (gitignored)
       |
       +--> build BruteForceIndex --> ground truth top-10
       +--> build IVFFlatIndex(n_probe=p) for each p in sweep
                |
                v
       evaluation/recall.py + evaluation/latency.py
                |
                v
       recall-vs-latency table / curve
```