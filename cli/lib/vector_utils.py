def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] - vec2[i] for i in range(len(vec1))]
