# ---------------------------------------------------------------------------
# Hybrid Search CLI — combines keyword (BM25) and semantic (embedding) search
# to produce more robust relevance rankings.
#
# Pipeline flow:
#   1. Parse the subcommand (rrf-search, weighted-search, normalize) and flags.
#   2. Optionally enhance the query via LLM (spell-correct, rewrite, or expand).
#   3. Load the movie catalog and initialise the HybridSearch engine.
#   4. Run RRF search to fuse BM25 + semantic results into a ranked list.
#   5. Optionally re-rank the RRF results using one of three methods:
#      - individual: LLM scores each candidate separately.
#      - batch: LLM ranks all candidates in a single call.
#      - cross_encoder: local cross-encoder model scores (query, doc) pairs.
#   6. Print the final ranked results with scores and metadata.
#
# Debug logging ([DEBUG]) is emitted at each stage:
#   - Original query, enhanced query, RRF intermediate results, final reranked results.
# ---------------------------------------------------------------------------

import argparse
import json
import os
import sys
import time          # Throttle LLM API calls to avoid rate limits
from pathlib import Path

# Ensure the project root is on sys.path so that cli.lib imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder


def expand_query(query: str) -> str:
    """Send the query to the LLM and return it expanded with related terms."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = (
        f"Expand the user-provided movie search query below with related terms.\n\n"
        f"Add synonyms and related concepts that might appear in movie descriptions.\n"
        f"Keep expansions relevant and focused.\n"
        f"Output only the additional terms; they will be appended to the original query.\n"
        f"Do not include any disclaimers, safety notes, or extra text.\n\n"
        f"Examples:\n"
        f'- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"\n'
        f'- "action movie with bear" -> "action thriller bear chase fight adventure"\n'
        f'- "comedy with bear" -> "comedy funny bear humor lighthearted"\n\n'
        f'User query: "{query}"\n'
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    expanded_terms = response.choices[0].message.content.strip()
    enhanced = f"{query} {expanded_terms}"
    print(f"Enhanced query (expand): '{query}' -> '{enhanced}'\n")
    return enhanced


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


def individual_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Score each RRF result individually via LLM and re-sort by the new score.

    *results* should contain more than *limit* entries (typically 5× the
    user-facing limit) so the re-ranker has a deeper pool to choose from.
    """
    print(f"Re-ranking top {limit} results using individual method...")

    # ── LLM client setup ──────────────────────────────────────────────────
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # ── Score every candidate document ────────────────────────────────────
    for i, r in enumerate(results):
        # Ask the LLM to rate relevance on a 0–10 scale
        prompt = (
            f"Rate how well this movie matches the search query.\n\n"
            f'Query: "{query}"\n'
            f"Movie: {r.get('title', '')} - {r.get('description', '')}\n\n"
            f"Consider:\n"
            f"- Direct relevance to query\n"
            f"- User intent (what they're looking for)\n"
            f"- Content appropriateness\n\n"
            f"Rate 0-10 (10 = perfect match).\n"
            f"Output ONLY the number in your response, no other text or explanation.\n\n"
            f"Score:"
        )

        # Retry up to 3 times — free-tier OpenRouter routes to different
        # models on each request, so a retry often succeeds after a failure.
        llm_score = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content.strip()
                score = float(content)
                if 0.0 <= score <= 10.0:
                    llm_score = score
                    break
            except Exception:
                # Brief pause before retrying a failed request
                if attempt < 2:
                    time.sleep(1)
                continue

        # Default to 0.0 if all retries were exhausted
        r["llm_score"] = llm_score if llm_score is not None else 0.0

        # Throttle between documents so we don't hit OpenRouter rate limits
        if i < len(results) - 1:
            time.sleep(3)

    # ── Sort by LLM score and keep only the top *limit* ───────────────────
    results.sort(key=lambda x: x["llm_score"], reverse=True)
    return results[:limit]


def batch_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Rank all candidate documents in a single LLM call and re-sort by rank.

    The LLM receives every candidate formatted with its ID, title and
    description, and returns a JSON array of IDs ordered by relevance.
    Results are then re-ordered to match and truncated to *limit*.
    """
    print(f"Re-ranking top {limit} results using batch method...")

    # ── LLM client setup ──────────────────────────────────────────────────
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # ── Build the candidate listing for the prompt ────────────────────────
    doc_list_str = "\n".join(
        f"[ID {r['id']}] {r.get('title', '')} - {r.get('description', '')}"
        for r in results
    )

    prompt = (
        f"Rank the movies listed below by relevance to the following search query.\n\n"
        f'Query: "{query}"\n\n'
        f"Movies:\n"
        f"{doc_list_str}\n\n"
        f"Return the movie IDs in order of relevance, best match first.\n\n"
        f"Your response must be a raw JSON array of integers.\n"
        f"Do not wrap the JSON in Markdown. Do not use a ```json code block.\n"
        f"Do not include any explanatory text.\n\n"
        f"For example:\n"
        f"[75, 12, 34, 2, 1]\n\n"
        f"Ranking:"
    )

    # ── Send the prompt with retries ──────────────────────────────────────
    # Free-tier OpenRouter routes to different models on each request, so a
    # retry often succeeds after a parse failure.
    ranked_ids: list[int] | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            ranked_ids = json.loads(content)
            if isinstance(ranked_ids, list) and all(isinstance(i, int) for i in ranked_ids):
                break
        except Exception:
            if attempt < 2:
                time.sleep(1)
            continue

    # ── Re-order results to match the LLM ranking ─────────────────────────
    if ranked_ids:
        doc_map = {r["id"]: r for r in results}
        ordered: list[dict] = []
        for rank, doc_id in enumerate(ranked_ids, 1):
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            doc["llm_rank"] = rank
            ordered.append(doc)
        return ordered[:limit]

    # Fallback: return first *limit* results with default ranks
    for rank, r in enumerate(results[:limit], 1):
        r["llm_rank"] = rank
    return results[:limit]


def cross_encoder_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Re-rank results using a cross-encoder model for pairwise relevance.

    Each candidate is paired with the query and scored by a dedicated
    cross-encoder model in a single batch prediction call.
    """
    print(f"Re-ranking top {limit} results using cross_encoder method...")

    # ── Build query-document pairs ────────────────────────────────────────
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

    # ── Load the cross-encoder model ──────────────────────────────────────
    # Try GPU first; fall back to CPU if hardware acceleration is unavailable.
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    except Exception:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")

    # ── Score all pairs in one batch ──────────────────────────────────────
    scores = cross_encoder.predict(pairs)

    # ── Attach scores and sort ────────────────────────────────────────────
    for doc, score in zip(results, scores):
        doc["cross_encoder_score"] = float(score)

    results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return results[:limit]


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


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    """Send results to the LLM and get a relevance rating (0-3) for each."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    formatted_results = [
        f"{i}. {r['title']} - {r.get('description', '')[:200]}"
        for i, r in enumerate(results, 1)
    ]

    prompt = (
        f"Rate how relevant each result is to this query on a 0-3 scale:\n\n"
        f'Query: "{query}"\n\n'
        f"Results:\n"
        f"{chr(10).join(formatted_results)}\n\n"
        f"Scale:\n"
        f"- 3: Highly relevant\n"
        f"- 2: Relevant\n"
        f"- 1: Marginally relevant\n"
        f"- 0: Not relevant\n\n"
        f"Do NOT give any numbers other than 0, 1, 2, or 3.\n\n"
        f"Return ONLY the scores in the same order you were given the documents. "
        f"Return a valid JSON list, nothing else. For example:\n\n"
        f"[2, 0, 3, 2, 0, 1]"
    )

    scores = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            if isinstance(parsed, list) and all(isinstance(s, int) and 0 <= s <= 3 for s in parsed):
                scores = parsed
                break
        except Exception:
            if attempt < 2:
                time.sleep(1)
            continue

    return scores if scores is not None else [0] * len(results)


def rrf_search_command(query: str, k: int, limit: int, enhance: str | None = None, rerank_method: str | None = None, evaluate: bool = False) -> None:
    """Run RRF hybrid search and print results.

    If *enhance* is provided, the query is first sent through the LLM for
    spelling correction before being passed to the search engine.

    If *rerank_method* is ``"individual"``, each RRF result is individually
    scored by an LLM and results are re-sorted by that new score.

    If *rerank_method* is ``"batch"``, all candidates are sent in a single
    LLM call that returns a relevance-ordered list of IDs.

    If *rerank_method* is ``"cross_encoder"``, a cross-encoder model scores
    every (query, document) pair and results are re-sorted by that score.
    """
    # [DEBUG] Log the original query
    print(f"[DEBUG] Original query: '{query}'")

    # Optionally enhance the query before searching
    if enhance == "spell":
        query = spell_correct_query(query)
    elif enhance == "rewrite":
        query = rewrite_query(query)
    elif enhance == "expand":
        query = expand_query(query)

    # [DEBUG] Log the query after enhancements
    print(f"[DEBUG] Query after enhancements: '{query}'")

    movies = load_movies()
    hs = HybridSearch(movies)

    if rerank_method in ("individual", "batch", "cross_encoder"):
        # Gather 5× the desired limit so the re-ranker has a deeper pool
        gather_limit = limit * 5
        results = hs.rrf_search(query, k, gather_limit)

        # [DEBUG] Log the results after RRF search
        print(f"[DEBUG] Results after RRF search (top {len(results)}):")
        for i, r in enumerate(results, 1):
            print(f"[DEBUG]   {i}. {r['title']} (RRF Score: {r['rrf_score']:.3f})")

        if rerank_method == "individual":
            results = individual_rerank(query, results, limit)
        elif rerank_method == "batch":
            results = batch_rerank(query, results, limit)
        elif rerank_method == "cross_encoder":
            results = cross_encoder_rerank(query, results, limit)

        # [DEBUG] Log the final results after re-ranking
        print(f"[DEBUG] Final results after {rerank_method} re-ranking (top {len(results)}):")
        for i, r in enumerate(results, 1):
            score_key = {"individual": "llm_score", "batch": "llm_rank", "cross_encoder": "cross_encoder_score"}[rerank_method]
            print(f"[DEBUG]   {i}. {r['title']} ({score_key}: {r.get(score_key, 'N/A')})")
        print()
    else:
        # Standard RRF without re-ranking
        results = hs.rrf_search(query, k, limit)

        # [DEBUG] Log the results after RRF search
        print(f"[DEBUG] Results after RRF search (top {len(results)}):")
        for i, r in enumerate(results, 1):
            print(f"[DEBUG]   {i}. {r['title']} (RRF Score: {r['rrf_score']:.3f})")
        print()

    # Print the header and each result with rank positions and scores
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    for i, r in enumerate(results, 1):
        bm25_rank_str = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "-"
        sem_rank_str = str(r["semantic_rank"]) if r["semantic_rank"] is not None else "-"
        print(f"{i}. {r['title']}")
        if "llm_score" in r:
            print(f"   Re-rank Score: {r['llm_score']:.3f}/10")
        if "llm_rank" in r:
            print(f"   Re-rank Rank: {r['llm_rank']}")
        if "cross_encoder_score" in r:
            print(f"   Cross Encoder Score: {r['cross_encoder_score']:.3f}")
        print(f"   RRF Score: {r['rrf_score']:.3f}")
        print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {sem_rank_str}")
        print(f"   {r['description']}")

    if evaluate:
        print("\n--- LLM Evaluation ---")
        eval_scores = evaluate_results(query, results)
        for i, (r, score) in enumerate(zip(results, eval_scores), 1):
            print(f"{i}. {r['title']}: {score}/3")


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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    # Optional re-ranking to refine RRF results.
    # "individual"     scores each candidate with a separate LLM call.
    # "batch"          ranks all candidates in a single LLM call.
    # "cross_encoder"  uses a local cross-encoder model for pairwise scoring.
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to apply after initial RRF",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Rate results on a 0-3 relevance scale using an LLM",
    )

    args = parser.parse_args()

    # Dispatch to the appropriate handler based on the subcommand
    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()