import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SCORE_PRECISION = 4


def load_movies() -> list[dict]:
    with (DATA_DIR / "movies.json").open("r", encoding="utf-8") as f:
        return json.load(f).get("movies", [])


def format_search_result(
    doc: dict, score: float, metadata: dict | None = None
) -> dict:
    return {
        "id": doc["id"],
        "title": doc["title"],
        "document": doc["description"][:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
