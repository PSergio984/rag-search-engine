import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")
    subparsers.add_parser("verify_embeddings", help="Build and verify embeddings for the movie dataset")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for input text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            import json
            from pathlib import Path
            ss = SemanticSearch()
            documents = json.loads(Path(__file__).resolve().parent.parent.joinpath("data", "movies.json").read_text())["movies"]
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"  {r['description']}")
        case "chunk":
            words = args.text.split()
            chunks = []
            for i in range(0, len(words), args.chunk_size):
                chunks.append(" ".join(words[i:i + args.chunk_size]))
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help(sys.stderr)

if __name__ == "__main__":
    main()
