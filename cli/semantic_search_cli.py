import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.semantic_search import verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help(sys.stderr)

if __name__ == "__main__":
    main()
