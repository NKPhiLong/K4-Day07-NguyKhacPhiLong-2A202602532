"""Bài tập 3.3 — Dự đoán độ tương tự cosine trên 5 cặp câu.

Chạy:  python scripts/similarity_predictions.py
Backend đọc từ .env (EMBEDDING_PROVIDER=mock|local|openai|gemini), mặc định mock.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
    compute_similarity,
)

PAIRS = [
    # (câu A, câu B, dự đoán)
    ("Sinh viên phải nộp học phí trước ngày 15 tháng 9.",
     "Hạn chót đóng tiền học kỳ này là 15/9.", "cao"),
    ("Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần.",
     "Giờ phục vụ của thư viện là 7:30–21:00 từ thứ Hai đến thứ Sáu.", "cao"),
    ("Sinh viên phải nộp học phí trước ngày 15 tháng 9.",
     "Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần.", "thấp"),
    ("Sinh viên được mượn tối đa 5 cuốn sách trong 14 ngày.",
     "Giảng viên được mượn tối đa 20 cuốn sách trong 180 ngày.", "cao"),
    ("Đơn phúc khảo nộp trong vòng 7 ngày sau khi công bố điểm.",
     "Món phở bò cần ninh xương ít nhất 6 tiếng.", "thấp"),
]


def pick_embedder():
    load_dotenv(".env", override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    try:
        if provider == "local":
            return LocalEmbedder()
        if provider == "openai":
            return OpenAIEmbedder()
        if provider == "gemini":
            return GeminiEmbedder()
    except Exception as exc:  # thiếu thư viện / key -> quay về mock
        print(f"[warn] {provider} backend không khả dụng ({exc.__class__.__name__}), dùng mock.")
    return _mock_embed


def main() -> None:
    embedder = pick_embedder()
    print("Embedding backend:", getattr(embedder, "_backend_name", "mock embeddings fallback"))
    print()
    print(f"{'#':<3}{'Dự đoán':<10}{'Score':>8}  Câu A | Câu B")
    for i, (a, b, guess) in enumerate(PAIRS, start=1):
        score = compute_similarity(embedder(a), embedder(b))
        print(f"{i:<3}{guess:<10}{score:>8.3f}  {a} | {b}")


if __name__ == "__main__":
    main()
