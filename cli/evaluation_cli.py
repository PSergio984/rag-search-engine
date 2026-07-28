import argparse
import json
import sys
from pathlib import Path

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

    golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    with golden_path.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    movies = load_movies()
    hs = HybridSearch(movies)

    print(f"k={limit}\n")

    for tc in golden["test_cases"]:
        query = tc["query"]
        relevant = set(tc["relevant_docs"])

        results = hs.rrf_search(query, k=60, limit=limit)

        retrieved_titles = [r["title"] for r in results]
        num_relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant)
        precision = num_relevant_retrieved / limit if limit > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(tc['relevant_docs'])}\n")


if __name__ == "__main__":
    main()