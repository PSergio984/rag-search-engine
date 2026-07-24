# ---------------------------------------------------------------------------
# Hybrid Search CLI — combines keyword (BM25) and semantic (embedding) search
# to produce more robust relevance rankings.
# ---------------------------------------------------------------------------

import argparse


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()