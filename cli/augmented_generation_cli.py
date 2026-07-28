# Augmented Generation CLI — Retrieval-Augmented Generation (RAG) pipeline.
#
# Flow:
#   1. Parse the "rag" subcommand with a user query.
#   2. Load the movie catalog and initialise the HybridSearch engine
#      (BM25 + semantic search fused via Reciprocal Rank Fusion).
#   3. Run RRF search (k=60) to retrieve the top 5 relevant documents.
#   4. Print the retrieved movie titles as "Search Results".
#   5. Build a prompt that includes the query and the retrieved documents,
#      instructing the LLM to act as a RAG agent for "Webflyx" streaming service.
#   6. Send the prompt to OpenRouter and print the LLM's generated answer.

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

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

            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()