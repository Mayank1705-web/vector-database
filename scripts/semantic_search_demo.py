"""Interactive semantic-search demo using a real sentence embedding model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer

from vector_db.approximate.ivf.index import IVFIndex


MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 4
N_PROBE = 2
TOP_K = 3


SAMPLE_STATEMENTS = [
    "I enjoy programming and building software applications.",
    "I like developing software and writing code.",
    "Python is my favorite programming language.",
    "I am learning data structures and algorithms.",
    "I practice coding problems on programming platforms.",
    "I enjoy working with databases and storing information.",
    "Vector databases are useful for similarity search.",
    "I am interested in machine learning and artificial intelligence.",
    "I enjoy training machine learning models.",
    "I like playing football with my friends.",
    "I enjoy going to the gym and exercising.",
    "I love traveling and exploring new places.",
]


def load_embedding_model() -> SentenceTransformer:
    """Load the pretrained sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def text_to_vector(
    text: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """Convert text into a normalized real sentence embedding."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("text must not be empty.")

    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vector = np.asarray(vector, dtype=np.float64)

    if vector.ndim != 1:
        raise ValueError("Embedding model must return a one-dimensional vector.")

    if vector.size == 0:
        raise ValueError("Embedding model returned an empty vector.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Embedding contains non-finite values.")

    return vector


def build_demo_index(
    model: SentenceTransformer,
) -> tuple[IVFIndex, dict[int, str], int]:
    """Build an IVF-Flat index from the sample statement collection."""
    vectors = np.vstack(
        [
            text_to_vector(statement, model)
            for statement in SAMPLE_STATEMENTS
        ]
    )

    dimension = vectors.shape[1]

    index = IVFIndex(
        n_clusters=N_CLUSTERS,
        n_probe=N_PROBE,
        kmeans_iterations=10,
        seed=42,
    )

    index.fit(vectors)

    statements = {
        statement_id: statement
        for statement_id, statement in enumerate(SAMPLE_STATEMENTS)
    }

    return index, statements, dimension


def print_results(
    query: str,
    results: list,
    statements: dict[int, str],
) -> None:
    """Print search results in a readable format."""
    print()
    print("Query")
    print("-----")
    print(query)

    print()
    print("Best matches")
    print("------------")

    if not results:
        print("No matching statements found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {statements[result.id]}")
        print(f"   Similarity: {result.score:.4f}")


def main() -> None:
    """Run the interactive semantic-search demo."""
    print("Loading embedding model...")

    try:
        model = load_embedding_model()
    except Exception as exc:
        print(f"Failed to load embedding model: {exc}")
        return

    index, statements, dimension = build_demo_index(model)

    print()
    print("Semantic Search Demo")
    print("====================")
    print()
    print(f"Embedding model: {MODEL_NAME}")
    print("Vector database: IVF-Flat")
    print(f"Statements:      {len(statements)}")
    print(f"Dimension:       {dimension}")
    print(f"Clusters:        {N_CLUSTERS}")
    print(f"Probes:          {N_PROBE}")


if __name__ == "__main__":
    main()