# ---------------------------------------------------------------------------
# Hybrid Search CLI — combines keyword (BM25) and semantic (embedding) search
# to produce more robust relevance rankings.
# ---------------------------------------------------------------------------

import argparse


def main() -> None:
    """Entrypoint: parse args and dispatch to the requested hybrid command."""
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_subparsers(dest="command", help="Available commands")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()