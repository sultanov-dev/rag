import math
import re
from collections import Counter

from utils import dot_product, normalize_text, vector_sqrt


def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    chunks: list[str] = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlapda xatolik bor")

    step = chunk_size - overlap

    words = text.split()

    for t in range(0, len(words), step):
        end = t + chunk_size

        chunk = words[t:end]
        word = " ".join(chunk)
        chunks.append(word)

    return chunks


def tokenize(text: str) -> list[str]:
    normalized_text = normalize_text(text)
    tokens = re.findall(r"\w+", normalized_text)

    return tokens


def compute_tf(tokens: list[str]) -> dict[str, float]:
    tf_dict: dict[str, float] = {}

    counts = Counter(tokens)
    total_tokens = len(tokens)

    for key, value in counts.items():
        tf = value / total_tokens
        tf_dict[key] = tf

    return tf_dict


def compute_idf(tokenized_chunks: list[list[str]]) -> dict[str, float]:
    total_chunks = len(tokenized_chunks)
    doc_fr: Counter[str] = Counter()
    idf_dict: dict[str, float] = {}

    for chunk in tokenized_chunks:
        doc_fr.update(set(chunk))

    for word, df in doc_fr.items():
        idf_dict[word] = math.log(total_chunks / df)

    return idf_dict


def compute_tfidf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    tfidf_dict: dict[str, float] = {}

    for word, tf_value in tf.items():
        idf_value = idf.get(word, 0.0)

        tfidf_dict[word] = tf_value * idf_value

    return tfidf_dict


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = dot_product(a, b)
    norm_a = vector_sqrt(a)
    norm_b = vector_sqrt(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    result = dot / (norm_a * norm_b)

    return result


def build_index(text: str, chunk_size: int = 50, overlap: int = 10):
    chunks = chunk_text(text, chunk_size, overlap)
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    idf = compute_idf(tokenized_chunks)
    vectors = [compute_tfidf(compute_tf(tokens), idf) for tokens in tokenized_chunks]

    return chunks, idf, vectors


def retrieve(
    query: str,
    chunks: list[str],
    idf: dict[str, float],
    vectors: list[dict[str, float]],
    k: int = 3,
) -> list[tuple[float, str]]:
    query_tokens = tokenize(query)
    query_vector = compute_tfidf(compute_tf(query_tokens), idf)

    result: list[tuple[float, str]] = []

    for chunk, vector in zip(chunks, vectors):
        score = cosine_similarity(query_vector, vector)

        if score > 0:
            result.append((score, chunk))

    ranked_documents = sorted(result, key=lambda item: item[0], reverse=True)

    return ranked_documents[:k]


def build_prompt(query: str, retrieved: list[tuple[float, str]]) -> str:
    context_lines: list[str] = []

    for i, (_, chunk) in enumerate(retrieved, start=1):
        context_lines.append(f"[{i}] {chunk}")

    context = "\n".join(context_lines)
    prompt = f"""Quyidagi kontekst asosida savolga javob ber.
Agar javob kontekstda bo'lmasa, "Bilmayman" deb ayt.

Kontekst:
{context}

Savol: {query}
Javob:"""
    return prompt
