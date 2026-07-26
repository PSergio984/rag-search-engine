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

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Run BM25 keyword search and return top-*limit* (doc_id, score) pairs."""
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        """Scale a list of floats to [0, 1] using min-max normalisation.

        Returns all 1.0 when every value is identical (or the list is empty).
        """
        if not values:
            return []
        low = min(values)
        high = max(values)
        if low == high:
            return [1.0] * len(values)
        return [(v - low) / (high - low) for v in values]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Fuse BM25 and semantic scores via linear interpolation.

        *alpha* controls the weight: 1.0 = pure BM25, 0.0 = pure semantic.
        """
        expanded = limit * 500

        # Fetch BM25 results and build doc_id -> raw score mapping
        bm25_pairs = self._bm25_search(query, expanded)
        bm25_map: dict[int, float] = dict(bm25_pairs)

        # Fetch semantic (chunk-level) results and build doc_id -> raw score mapping
        semantic_results = self.semantic_search.search_chunks(query, expanded)
        sem_map: dict[int, float] = {r["id"]: r["score"] for r in semantic_results}

        # Gather all unique document IDs that appeared in either result set
        all_ids = set(bm25_map.keys()) | set(sem_map.keys())

        if not all_ids:
            return []

        # Prepare parallel lists for min-max normalisation
        bm25_vals = [bm25_map.get(did, 0.0) for did in all_ids]
        sem_vals = [sem_map.get(did, 0.0) for did in all_ids]

        bm25_norm = self._min_max_normalize(bm25_vals)
        sem_norm = self._min_max_normalize(sem_vals)

        # Build the document map (id -> full doc dict)
        doc_map = {d["id"]: d for d in self.documents}

        combined: list[dict] = []
        for did, bn, sn in zip(all_ids, bm25_norm, sem_norm):
            doc = doc_map.get(did)
            if doc is None:
                continue
            hybrid = (1 - alpha) * sn + alpha * bn
            combined.append({
                "id": did,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "score": hybrid,
                "bm25_score": bn,
                "semantic_score": sn,
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        """Fuse BM25 and semantic results via reciprocal rank fusion.

        For each document appearing in either result set, an RRF score is
        computed as the sum of 1/(k + rank) across every system that returned
        that document.  Results are then sorted by RRF score descending.

        *k* is the RRF constant that dampens rank contributions (default 60).
        """
        expanded = limit * 500

        # Step 1: retrieve 500× the desired limit from each system
        bm25_pairs = self._bm25_search(query, expanded)
        semantic_results = self.semantic_search.search_chunks(query, expanded)

        # Step 2: map doc_id → 1-based rank for each system
        bm25_ranks: dict[int, int] = {}
        for rank, (doc_id, _) in enumerate(bm25_pairs, 1):
            bm25_ranks[doc_id] = rank

        sem_ranks: dict[int, int] = {}
        for rank, r in enumerate(semantic_results, 1):
            sem_ranks[r["id"]] = rank

        # Step 3: union of all document IDs that appeared in either system
        all_ids = set(bm25_ranks.keys()) | set(sem_ranks.keys())
        doc_map = {d["id"]: d for d in self.documents}

        # Step 4: compute RRF score for each document
        combined: list[dict] = []
        for did in all_ids:
            doc = doc_map.get(did)
            if doc is None:
                continue

            bm25_rank = bm25_ranks.get(did)
            sem_rank = sem_ranks.get(did)

            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (k + bm25_rank)
            if sem_rank is not None:
                rrf_score += 1.0 / (k + sem_rank)

            combined.append({
                "id": did,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank,
                "semantic_rank": sem_rank,
            })

        combined.sort(key=lambda x: x["rrf_score"], reverse=True)
        return combined[:limit]