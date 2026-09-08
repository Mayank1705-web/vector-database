# Vector DB Frontend

This frontend is a thin browser client for the project's existing FastAPI vector database.

It does **not** reimplement embeddings, k-means, inverted lists, or vector search in JavaScript.

## Backend

From the repository root:

```powershell
uvicorn vector_db.api.server:app --reload
```

The frontend expects:

```text
http://127.0.0.1:8000
```

## Frontend

From the repository root:

```powershell
python -m http.server 5500 -d frontend
```

Open:

```text
http://127.0.0.1:5500
```

## API operations

The UI calls the existing endpoints:

- `GET /health`
- `POST /insert`
- `POST /search`
- `DELETE /delete/{id}`

Search and insert vectors must be 128-dimensional because the configured benchmark/index dataset uses `dim: 128`.

## Important architecture note

The browser only handles presentation and HTTP requests.

The actual vector operations remain in the Python project:

```text
Browser
  -> FastAPI
  -> VectorIndex
  -> IVF-Flat
  -> NumPy cosine similarity
```

The repository's semantic demo is separate and uses `all-MiniLM-L6-v2` to create real 384-dimensional text embeddings. The current API contract is vector-based, so this frontend intentionally accepts vectors rather than pretending to perform text embedding locally.
