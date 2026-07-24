# Re-export InvertedIndex from the main CLI module so it can be imported
# as cli.lib.keyword_search — keeping the lib namespace clean.
from ..keyword_search_cli import InvertedIndex

__all__ = ["InvertedIndex"]