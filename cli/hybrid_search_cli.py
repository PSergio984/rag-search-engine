# ---------------------------------------------------------------------------
# Hybrid Search CLI — combines keyword (BM25) and semantic (embedding) search
# to produce more robust relevance rankings.
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that cli.lib imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies


def normalize_command(scores: list[float]) -> None:
    """Min-max normalize a list of scores and print each on its own line."""
    if not scores:
        return
    low = min(scores)
    high = max(scores)
    if low == high:
        for _ in scores:
            print(f"* {1.0:.4f}")
        return
    for s in scores:
        normalized = (s - low) / (high - low)
        print(f"* {normalized:.4f}")


def rrf_search_command(query: str, k: int, limit: int) -> None:
    """Run RRF hybrid search and print results.

    Each result includes the document title, the combined RRF score,
    the individual ranks from BM25 and semantic search, and the full description.
    """
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k, limit)
    for i, r in enumerate(results, 1):
        bm25_rank_str = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "-"
        sem_rank_str = str(r["semantic_rank"]) if r["semantic_rank"] is not None else "-"
        print(f"{i}. {r['title']}")
        print(f"  RRF Score: {r['rrf_score']:.3f}")
        print(f"  BM25 Rank: {bm25_rank_str}, Semantic Rank: {sem_rank_str}")
        print(f"  {r['description']}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    """Run weighted hybrid search and print results.

    Scores are min-max normalised before linear interpolation.
    *alpha* controls BM25 weight (1.0 = pure BM25, 0.0 = pure semantic).
    """
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"  Hybrid Score: {r['score']:.3f}")
        print(f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
        print(f"  {r['description']}")


def main() -> None:
    """Entrypoint: parse args and dispatch to the requested hybrid command."""
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    norm_parser.add_argument(
        "scores", type=float, nargs="*", help="Scores to normalize"
    )

    # Weighted-search subcommand: linearly interpolates BM25 + semantic scores
    ws_parser = subparsers.add_parser(
        "weighted-search", help="Weighted hybrid search (BM25 + semantic)"
    )
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument("--alpha", type=float, default=0.5, help="BM25 weight (0.0 = pure semantic, 1.0 = pure BM25)")
    ws_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    # RRF-search subcommand: combines BM25 + semantic via reciprocal rank fusion
    rrf_parser = subparsers.add_parser(
        "rrf-search", help="RRF hybrid search (BM25 + semantic via reciprocal rank fusion)"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF constant that dampens rank contributions (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()