"""
Hybrid search — merges keyword (BM25) and semantic (embedding) retrieval
into a single relevance pipeline.
"""

import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    """Combines BM25 keyword search with chunk-level semantic search.

    Two fusion strategies will be provided:
        - *weighted*: linear interpolation of normalised scores.
        - *RRF*: reciprocal rank fusion across both result sets.
    """

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        # Build or load semantic chunk embeddings up front
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Build or load the BM25 inverted index
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        """Run BM25 keyword search and return top-*limit* (doc_id, score) pairs."""
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Fuse BM25 and semantic scores via linear interpolation.

        *alpha* controls the weight: 1.0 = pure semantic, 0.0 = pure BM25.
        """
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        """Fuse BM25 and semantic results via reciprocal rank fusion.

        *k* is the RRF constant that dampens rank contributions.
        """
        raise NotImplementedError("RRF hybrid search is not implemented yet.")