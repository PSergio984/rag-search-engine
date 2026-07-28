# Augmented Generation CLI — Retrieval-Augmented Generation (RAG) pipeline.
#
# Commands:
#   rag        — RRF search + LLM-generated answer to the query.
#   summarize  — RRF search + LLM-generated multi-document summary.
#
# Flow (both commands):
#   1. Parse the subcommand (rag / summarize) with a user query and flags.
#   2. Load the movie catalog and initialise the HybridSearch engine
#      (BM25 + semantic search fused via Reciprocal Rank Fusion).
#   3. Run RRF search (k=60) to retrieve the top-k relevant documents.
#   4. Print the retrieved movie titles as "Search Results".
#   5. Build a prompt tailored to the subcommand that includes the query
#      and the retrieved documents.
#   6. Send the prompt to OpenRouter and print the LLM's response.

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Summarize multiple search results"
    )
    sum_parser.add_argument("query", type=str, help="Search query to summarize")
    sum_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to summarize"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            movies = load_movies()
            hs = HybridSearch(movies)

            results = hs.rrf_search(query, k=60, limit=5)
            titles = [r["title"] for r in results]
            docs = "\n".join(
                f"{i}. {r['title']} - {r.get('description', '')}"
                for i, r in enumerate(results, 1)
            )

            print("Search Results:")
            for t in titles:
                print(f"- {t}")
            print()

            client = _get_client()

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.choices[0].message.content.strip()
            print(f"RAG Response:\n{answer}")

        case "summarize":
            query = args.query
            limit = args.limit

            movies = load_movies()
            hs = HybridSearch(movies)

            results = hs.rrf_search(query, k=60, limit=limit)
            titles = [r["title"] for r in results]
            docs = "\n".join(
                f"{i}. {r['title']} - {r.get('description', '')}"
                for i, r in enumerate(results, 1)
            )

            print("Search Results:")
            for t in titles:
                print(f"  - {t}")
            print()

            client = _get_client()

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            summary = response.choices[0].message.content.strip()
            print(f"LLM Summary:\n{summary}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()