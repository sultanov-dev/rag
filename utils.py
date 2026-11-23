import math
import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.sub(r"['’ʻ]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return text


def dot_product(a: dict[str, float], b: dict[str, float]) -> float:
    result: float = 0.0

    for key, value in a.items():
        b_value = b.get(key, 0.0)
        inc = value * b_value
        result += inc

    return result


def vector_sqrt(a: dict[str, float]) -> float:
    result: float = 0.0

    for value in a.values():
        sqrt = value**2
        result += sqrt

    return math.sqrt(result)


def load_text(path: str) -> str:
    with open(path, mode="r", encoding="utf-8") as file:
        return file.read()
