# ---------------------------------------------------------------------------
# Hybrid Search CLI — combines keyword (BM25) and semantic (embedding) search
# to produce more robust relevance rankings.
# ---------------------------------------------------------------------------

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that cli.lib imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies
from dotenv import load_dotenv
from openai import OpenAI


def rewrite_query(query: str) -> str:
    """Send the query to the LLM and return a rewritten version optimized for search."""
    # Load the API key from .env and create an OpenRouter client
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Prompt the LLM to expand informal/slang terms into clear search keywords
    prompt = (
        f"Rewrite the user-provided movie search query below into a more effective "
        f"search query. Expand informal phrasing, slang, or vague references into "
        f"clear, searchable terms. Preserve the core intent. Do not add extra words "
        f"beyond what is needed. Output only the final query text, nothing else.\n"
        f'User query: "{query}"\n'
    )

    # Send the request and extract the rewritten query
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    enhanced = response.choices[0].message.content.strip()
    # Show the user the original vs. rewritten query
    print(f"Enhanced query (rewrite): '{query}' -> '{enhanced}'\n")
    return enhanced


def spell_correct_query(query: str) -> str:
    """Send the query to the LLM and return a spelling-corrected version."""
    # Load environment variables (API key) from .env file
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    # Create an OpenAI-compatible client pointed at OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Craft a prompt that asks the LLM to fix typos without rewriting
    prompt = (
        f"Fix any spelling errors in the user-provided movie search query below.\n"
        f"Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.\n"
        f"Preserve punctuation and capitalization unless a change is required for a typo fix.\n"
        f"If there are no spelling errors, or if you're unsure, output the original query unchanged.\n"
        f"Output only the final query text, nothing else.\n"
        f'User query: "{query}"\n'
    )

    # Send the request and extract the corrected query
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    enhanced = response.choices[0].message.content.strip()
    # Show the user what changed
    print(f"Enhanced query (spell): '{query}' -> '{enhanced}'\n")
    return enhanced


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


def rrf_search_command(query: str, k: int, limit: int, enhance: str | None = None) -> None:
    """Run RRF hybrid search and print results.

    If *enhance* is provided, the query is first sent through the LLM for
    spelling correction before being passed to the search engine.
    """
    # Optionally enhance the query before searching
    if enhance == "spell":
        query = spell_correct_query(query)
    elif enhance == "rewrite":
        query = rewrite_query(query)

    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k, limit)
    # Print each result with rank positions and RRF score
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
    # Print each result with its hybrid, BM25, and semantic scores
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"  Hybrid Score: {r['score']:.3f}")
        print(f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
        print(f"  {r['description']}")


def main() -> None:
    """Entrypoint: parse args and dispatch to the requested hybrid command."""
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize subcommand: min-max scale a list of scores
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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    # Dispatch to the appropriate handler based on the subcommand
    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()