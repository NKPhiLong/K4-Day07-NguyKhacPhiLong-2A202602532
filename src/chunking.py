from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    # Tách ở vị trí NGAY SAU dấu câu (lookbehind) để không nuốt mất dấu.
    # Nhận ". ", "! ", "? " (kèm mọi khoảng trắng phía sau) và ".\n".
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return []

        n = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + n]) for i in range(0, len(sentences), n)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [c for c in self._split(text, list(self.separators)) if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: mảnh đã đủ nhỏ -> giữ nguyên.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: hết separator (hoặc separators=[] ngay từ đầu) -> cắt cứng theo chunk_size.
        if not remaining_separators:
            return self._hard_split(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]

        # Base case 3: separator rỗng "" nghĩa là "cắt theo ký tự" -> cắt cứng.
        if separator == "":
            return self._hard_split(current_text)

        parts = current_text.split(separator)
        if len(parts) == 1:
            # Separator này không xuất hiện trong text -> hạ xuống separator tiếp theo.
            return self._split(current_text, rest)

        # Gom các mảnh nhỏ liền kề lại cho tới sát chunk_size (chiều "gom lên").
        # Mảnh nào một mình vẫn quá dài thì đệ quy xuống separator nhỏ hơn (chiều "xuống sâu").
        chunks: list[str] = []
        buffer = ""
        for part in parts:
            if not part:
                continue
            candidate = part if not buffer else buffer + separator + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""

            if len(part) <= self.chunk_size:
                buffer = part
            else:
                chunks.extend(self._split(part, rest))

        if buffer:
            chunks.append(buffer)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Cắt cứng theo chunk_size khi không còn separator nào dùng được."""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
