import sys
from pathlib import Path
sys.path.insert(0, str(Path("cli").resolve().parent))

from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies

movies = load_movies()
hs = HybridSearch(movies)
alpha = 0.2
limit = 25
expanded = limit * 500

bm25_pairs = hs._bm25_search("British Bear", expanded)
bm25_map = dict(bm25_pairs)
sem_results = hs.semantic_search.search_chunks("British Bear", expanded)
sem_map = {r["id"]: r["score"] for r in sem_results}
all_ids = set(bm25_map.keys()) | set(sem_map.keys())

bm25_vals = [bm25_map.get(did, 0.0) for did in all_ids]
sem_vals = [sem_map.get(did, 0.0) for did in all_ids]

def mn(vals):
    low, high = min(vals), max(vals)
    if low == high:
        return [1.0]*len(vals)
    return [(v-low)/(high-low) for v in vals]

bm25_n, sem_n = mn(bm25_vals), mn(sem_vals)
doc_map = {d["id"]: d for d in movies}

# Approach: reversed interpretation — alpha=0.2 means 80% semantic
for label, alpha_sem, alpha_bm25 in [("alpha*sem + (1-alpha)*bm25", 0.2, 0.8), ("alpha*bm25 + (1-alpha)*sem (reversed)", 0.8, 0.2)]:
    combined = []
    for did, bn, sn in zip(all_ids, bm25_n, sem_n):
        doc = doc_map.get(did)
        if doc is None:
            continue
        hybrid = alpha_sem * sn + alpha_bm25 * bn
        combined.append({"title": doc["title"], "score": hybrid})
    combined.sort(key=lambda x: x["score"], reverse=True)
    found = any("Legends" in r["title"] for r in combined[:25])
    idx = next((i for i, r in enumerate(combined) if "Legends" in r["title"]), -1)
    print(f"{label}: found={found}, pos={idx+1 if idx>=0 else 'N/A'}")
    if found and idx < 25:
        r = combined[idx]
        print(f"  {r['title']} hybrid={r['score']:.4f}")
        # Also show what top/bottom of top 25 look like
        print(f"  #1: {combined[0]['title']} {combined[0]['score']:.4f}")
        print(f"  #25: {combined[24]['title']} {combined[24]['score']:.4f}")