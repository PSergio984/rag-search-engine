# Evaluation CLI for RRF hybrid search using a golden dataset.
#
# Flow:
#   1. Parse --limit (k for precision@k / recall@k).
#   2. Load golden_dataset.json — a set of queries with their known relevant docs.
#   3. Load the full movie catalog and initialise the HybridSearch engine
#      (BM25 + semantic search fused via Reciprocal Rank Fusion).
#   4. For each test case:
#      a. Run RRF search (k=60) to retrieve the top-k results.
#      b. Compare retrieved titles against the golden relevant set.
#      c. Compute precision@k and recall@k.
#      d. Print query, metrics, retrieved list, and relevant list.

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so cli.lib imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load the golden dataset containing query -> relevant document mappings
    golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    with golden_path.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    # Load movies and build hybrid search (BM25 + semantic embeddings)
    movies = load_movies()
    hs = HybridSearch(movies)

    print(f"k={limit}\n")

    for tc in golden["test_cases"]:
        query = tc["query"]
        relevant = set(tc["relevant_docs"])

        # Run RRF search with k=60 and the user-specified result limit
        results = hs.rrf_search(query, k=60, limit=limit)

        retrieved_titles = [r["title"] for r in results]
        num_relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant)
        precision = num_relevant_retrieved / limit if limit > 0 else 0.0
        recall = num_relevant_retrieved / len(relevant) if relevant else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(tc['relevant_docs'])}\n")


if __name__ == "__main__":
    main()