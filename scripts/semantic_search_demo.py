"""Interactive semantic-search demo using the project's IVF-Flat index."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_db.approximate.ivf.index import IVFIndex


DIMENSION = 128
N_CLUSTERS = 4
N_PROBE = 2


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


def text_to_vector(text: str, dimension: int = DIMENSION) -> np.ndarray:
    """Convert text into a deterministic normalized vector.

    The embedding uses hashed word and character features. It is intentionally
    lightweight and deterministic for this educational demo; it is not a
    production-quality language-model embedding.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.strip().lower()

    if not text:
        raise ValueError("text must not be empty.")

    vector = np.zeros(dimension, dtype=np.float64)

    words = re.findall(r"[a-z0-9]+", text)

    if not words:
        raise ValueError("text must contain at least one alphanumeric token.")

    # Hashed word features.
    for word in words:
        digest = hashlib.sha256(
            f"word:{word}".encode("utf-8")
        ).digest()

        index = int.from_bytes(digest[:8], "little") % dimension
        vector[index] += 1.0

    # Hashed character n-gram features provide some robustness to
    # related word forms such as "program", "programming", and "programmer".
    padded = f"^{text}$"

    for size in (3, 4):
        for start in range(len(padded) - size + 1):
            ngram = padded[start : start + size]

            digest = hashlib.sha256(
                f"char:{ngram}".encode("utf-8")
            ).digest()

            index = int.from_bytes(digest[:8], "little") % dimension
            vector[index] += 0.25

    norm = np.linalg.norm(vector)

    if norm == 0.0:
        raise ValueError("Unable to create a non-zero embedding.")

    return vector / norm


def build_demo_index() -> tuple[IVFIndex, dict[int, str]]:
    """Build an IVF-Flat index from the sample statement collection."""
    vectors = np.vstack(
        [
            text_to_vector(statement)
            for statement in SAMPLE_STATEMENTS
        ]
    )

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

    return index, statements


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
    index, statements = build_demo_index()

    print("Semantic Search Demo")
    print("====================")
    print()
    print("Vector database: IVF-Flat")
    print(f"Statements:      {len(statements)}")
    print(f"Dimension:       {DIMENSION}")
    print(f"Clusters:        {N_CLUSTERS}")
    print(f"Probes:          {N_PROBE}")
    print()
    print("Type a statement to search.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        try:
            query_vector = text_to_vector(query)

            results = index.search(
                query_vector,
                k=3,
            )

            print_results(
                query=query,
                results=results,
                statements=statements,
            )

        except (TypeError, ValueError) as exc:
            print(f"Error: {exc}")

    print("\nDemo finished.")


if __name__ == "__main__":
    main()