"""
bench.py — Công cụ benchmark retrieval cho Lab 07 (K4-L3A).

Chạy:
    python bench.py                       # chiến lược của tôi (heading), ghi ket_qua_benchmark.txt
    python bench.py --strategy recursive  # một chiến lược khác
    python bench.py --all                 # cả 4 chiến lược, cùng backend, mỗi chiến lược một file

Backend embedding đọc từ .env (EMBEDDING_PROVIDER = gemini | openai | local | mock).
Embedding được cache theo hash nội dung vào .embedding_cache.json để chạy lại không tốn quota.

Luồng: file .md -> tách frontmatter -> chunk phần thân -> Document(id="doc#i", metadata={**fm, "doc_id": doc})
       -> EmbeddingStore -> 5 query (search / search_with_filter) -> chấm 2 mức -> A/B filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from src.agent import KnowledgeBaseAgent  # noqa: E402
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker  # noqa: E402
from src.embeddings import (  # noqa: E402
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
)
from src.models import Document  # noqa: E402
from src.store import EmbeddingStore  # noqa: E402

DATA_DIR = ROOT / "data" / "university"
CACHE_FILE = ROOT / ".embedding_cache.json"
TOP_K = 3

# ---------------------------------------------------------------------------
# 5 benchmark query của nhóm (REPORT_NHOM mục 3). gold_keyword là chuỗi đặc trưng
# PHẢI xuất hiện trong ngữ cảnh truy xuất được thì mới coi là "trả lời được".
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Sinh viên năm thứ mấy và cần đạt kết quả học tập thế nào để đủ điều kiện xét học bổng EVN?",
        "gold_doc_id": "hoc-bong-evn",
        "gold_keyword": "năm thứ 3",
        "gold_answer": "Sinh viên năm thứ 3; kết quả học tập 2025–2026 đạt loại giỏi trở lên, rèn luyện từ loại tốt, chưa nhận học bổng ngoài ngân sách khác.",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Thời gian nghỉ Tết Nguyên đán năm học 2025-2026 của sinh viên chính quy kéo dài từ ngày nào đến ngày nào?",
        "gold_doc_id": "ke-hoach-dao-tao-nam-hoc",
        "gold_keyword": "09/02-22/02/2026",
        "gold_answer": "Nghỉ Tết Nguyên đán từ 09/02/2026 đến 22/02/2026.",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Phương thức 2 của Trường ĐH Khoa học Tự nhiên năm 2026 áp dụng nhân hệ số 2 môn Toán cho những ngành nào?",
        "gold_doc_id": "phuong-thuc-xet-tuyen-hus",
        "gold_keyword": "Khoa học dữ liệu",
        "gold_answer": "Toán học, Toán tin, Khoa học máy tính và thông tin, Khoa học dữ liệu.",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Chương trình trao đổi sinh viên tại Đại học Osaka kỳ Xuân 2027 yêu cầu đã học bao nhiêu học kỳ và điểm GPA tối thiểu là bao nhiêu?",
        "gold_doc_id": "trao-doi-sinh-vien-osaka",
        "gold_keyword": "3,2/4,0",
        "gold_answer": "Hoàn thành ít nhất 02 học kỳ tại ĐHQGHN; GPA từ 3,2/4,0 trở lên.",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Số lượng và danh mục các chương trình đào tạo chuẩn và đặc thù của Trường Đại học Công nghệ là gì?",
        "gold_doc_id": "chuong-trinh-dao-tao-dai-hoc",
        # Dòng 18 của bảng UET — chỉ có mặt nếu chunk giữ được TRỌN bảng của trường.
        "gold_keyword": "Trí tuệ nhân tạo",
        "gold_answer": "18 chương trình: 13 chuẩn (CNTT, Kỹ thuật máy tính, Robot, Năng lượng, Cơ kỹ thuật, AI...) và 5 đặc thù (CLC).",
        "filter": {"audience": "student"},  # ràng buộc K4-L3A
    },
]

# Q5b — ĐỀ XUẤT thay cho Q5: câu hỏi KHÔNG nêu người hỏi là ai, trong khi corpus có 2 tài liệu
# cùng nhắc "Trường Đại học Công nghệ" nhưng khác audience (student: 18 CTĐT / faculty: 339 cán bộ).
# Không lọc thì bảng nhân sự (faculty) chiếm top-1 -> đây mới là câu "cần" metadata_filter.
EXTRA_QUERY = {
    "id": "5b",
    "query": "Theo thống kê của ĐHQGHN, Trường Đại học Công nghệ có tổng số bao nhiêu?",
    "gold_doc_id": "chuong-trinh-dao-tao-dai-hoc",
    "gold_keyword": "Trí tuệ nhân tạo",
    "gold_answer": "(với audience=student) 18 chương trình đào tạo; nếu không lọc sẽ trả về 339 cán bộ giảng viên (faculty).",
    "filter": {"audience": "student"},
}
AB_EXTRA_QUERY = EXTRA_QUERY["query"]


# ---------------------------------------------------------------------------
# Chiến lược của tôi: HeadingChunker — chunk theo tiêu đề/mục, header-aware với bảng.
# ---------------------------------------------------------------------------
class HeadingChunker:
    """Chunk theo tiêu đề/mục của văn bản quy định & thông báo đại học.

    Lý do thiết kế: văn bản do người soạn đã chia sẵn thành các mục ("Đối tượng và tiêu
    chuẩn", "Hồ sơ đăng ký", "CHỈ TIÊU & ĐỊA ĐIỂM"...), và các bảng lớn được nhóm theo
    đơn vị ("| I. Trường Đại học Công nghệ |"). Mỗi mục là một đơn vị ngữ nghĩa trọn vẹn,
    nên chunk theo mục giữ được câu hỏi-đáp án trong cùng một chunk.

    Nhận diện heading (3 dạng):
        1. Markdown heading  `# ...`, `## ...`
        2. Dòng tiêu đề trần: dòng ngắn (<= 90 ký tự), đứng riêng giữa 2 dòng trống,
           không kết thúc bằng dấu câu, không phải gạch đầu dòng / hàng bảng.
        3. Hàng nhóm trong bảng: `| I. ... |`, `| II. ... |` (số La Mã).

    Section quá dài -> cắt nhỏ, và GẮN LẠI tiêu đề vào từng mảnh con.
    Với bảng: mỗi mảnh con được gắn lại HEADER bảng (dòng tiêu đề cột + dòng `---`)
    để hàng dữ liệu không mất ngữ nghĩa cột (header-aware table chunking).
    """

    MD_HEADING = re.compile(r"^#{1,6}\s+\S")
    TABLE_GROUP = re.compile(r"^\|\s*[IVXLC]+\.\s+[^|]+\|")
    TABLE_ROW = re.compile(r"^\|")
    TABLE_SEP = re.compile(r"^\|\s*-{3,}")

    def __init__(self, max_len: int = 600, min_len: int = 150) -> None:
        self.max_len = max_len
        self.min_len = min_len
        self.fallback = RecursiveChunker(chunk_size=max_len)

    # --- nhận diện heading -------------------------------------------------
    def _is_bare_title(self, lines: list[str], i: int) -> bool:
        line = lines[i].strip()
        if not line or len(line) > 90 or line.endswith((".", ";", ":", ",")):
            return False
        if line.startswith(("-", "*", "|", ">")) or re.match(r"^\d+[.)]\s", line):
            return False
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        return prev_blank and next_blank

    def _split_sections(self, text: str) -> list[tuple[str, list[str]]]:
        """Trả về [(heading, [dòng thân])]. Heading rỗng cho phần mở đầu."""
        lines = text.splitlines()
        sections: list[tuple[str, list[str]]] = [("", [])]
        for i, raw in enumerate(lines):
            line = raw.rstrip()
            if self.MD_HEADING.match(line) or self.TABLE_GROUP.match(line) or self._is_bare_title(lines, i):
                title = line.strip("#").strip() if self.MD_HEADING.match(line) else line.strip()
                if self.TABLE_GROUP.match(line):
                    title = line.split("|")[1].strip()
                sections.append((title, []))
            else:
                sections[-1][1].append(line)
        return [(h, body) for h, body in sections if h or any(l.strip() for l in body)]

    # --- cắt section dài ---------------------------------------------------
    def _split_table(self, header: list[str], rows: list[str], prefix: str) -> list[str]:
        head = "\n".join(header)
        chunks, buf = [], []
        for row in rows:
            candidate = "\n".join([prefix, head, *buf, row]) if head else "\n".join([prefix, *buf, row])
            if buf and len(candidate) > self.max_len:
                chunks.append("\n".join([prefix, head, *buf]).strip() if head else "\n".join([prefix, *buf]).strip())
                buf = []
            buf.append(row)
        if buf:
            chunks.append("\n".join([prefix, head, *buf]).strip() if head else "\n".join([prefix, *buf]).strip())
        return chunks

    def _split_long_section(self, prefix: str, body_lines: list[str]) -> list[str]:
        table_rows = [l for l in body_lines if self.TABLE_ROW.match(l.strip())]
        if len(table_rows) >= 3:
            # header = dòng tiêu đề cột + dòng phân cách, nếu có trong section này
            header: list[str] = []
            data_rows = table_rows
            if len(table_rows) >= 2 and self.TABLE_SEP.match(table_rows[1].strip()):
                header, data_rows = table_rows[:2], table_rows[2:]
            text_part = "\n".join(l for l in body_lines if not self.TABLE_ROW.match(l.strip())).strip()
            out = [f"{prefix}\n{text_part}".strip()] if text_part else []
            return out + self._split_table(header, data_rows, prefix)
        body = "\n".join(body_lines).strip()
        return [f"{prefix}\n{piece}".strip() for piece in self.fallback.chunk(body)]

    # --- API --------------------------------------------------------------
    def chunk(self, text: str, table_header: list[str] | None = None) -> list[str]:
        if not text or not text.strip():
            return []
        sections = self._split_sections(text)
        doc_title = next((h for h, _ in sections if h), "")

        # Header bảng dùng chung: bảng lớn chỉ có header ở section đầu, các nhóm "| I. |" phía sau không có.
        shared_header: list[str] = []
        for _, body in sections:
            rows = [l for l in body if self.TABLE_ROW.match(l.strip())]
            if len(rows) >= 2 and self.TABLE_SEP.match(rows[1].strip()):
                shared_header = rows[:2]
                break

        chunks: list[str] = []
        for heading, body in sections:
            body_text = "\n".join(body).strip()
            prefix = heading if heading == doc_title or not doc_title else f"{doc_title} — {heading}"
            full = f"{prefix}\n{body_text}".strip() if body_text else prefix
            if len(full) <= self.max_len:
                chunks.append(full)
                continue
            rows = [l for l in body if self.TABLE_ROW.match(l.strip())]
            if rows and shared_header and not self.TABLE_SEP.match(rows[1].strip() if len(rows) > 1 else ""):
                body = shared_header + body  # gắn header bảng dùng chung vào nhóm không có header
            chunks.extend(self._split_long_section(prefix, body))

        # Gom mục quá ngắn (vd. tiêu đề không thân) vào mục kế tiếp để tránh chunk vụn.
        merged: list[str] = []
        for c in chunks:
            short = len(c) < self.min_len or len(merged[-1]) < self.min_len if merged else False
            if merged and short and len(merged[-1]) + len(c) + 1 <= self.max_len:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged


STRATEGIES: dict[str, Callable[[], Any]] = {
    "fixed_size": lambda: FixedSizeChunker(chunk_size=200, overlap=50),
    "by_sentences": lambda: SentenceChunker(max_sentences_per_chunk=3),
    "recursive": lambda: RecursiveChunker(chunk_size=400),
    "heading": lambda: HeadingChunker(max_len=600),
}
STRATEGY_LABELS = {
    "fixed_size": "FixedSizeChunker(chunk_size=200, overlap=50)",
    "by_sentences": "SentenceChunker(max_sentences_per_chunk=3)",
    "recursive": "RecursiveChunker(chunk_size=400)",
    "heading": "HeadingChunker(max_len=600) — chunk theo tiêu đề/mục, header-aware bảng",
}


# ---------------------------------------------------------------------------
# Backend: embedding có cache + LLM
# ---------------------------------------------------------------------------
class CachedEmbedder:
    def __init__(self, inner: Callable[[str], list[float]], name: str) -> None:
        self.inner, self.name = inner, name
        self.cache: dict[str, list[float]] = {}
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._dirty = 0

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256(f"{self.name}\x00{text}".encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.cache[key] = self._call_with_retry(text)
            self._dirty += 1
            if self._dirty % 20 == 0:
                self.flush()
        return self.cache[key]

    def _call_with_retry(self, text: str) -> list[float]:
        """Free tier (Gemini: 100 req/phút) hay trả 429 -> chờ rồi thử lại, không làm hỏng cả lượt chạy."""
        for attempt in range(6):
            try:
                return self.inner(text)
            except Exception as exc:
                msg = str(exc)
                if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "rate" not in msg.lower():
                    raise
                m = re.search(r"retry in ([0-9.]+)s", msg, re.I)
                wait = float(m.group(1)) + 2 if m else 30.0 * (attempt + 1)
                self.flush()
                print(f"   [rate-limit] chờ {wait:.0f}s rồi thử lại ({attempt + 1}/6)...", flush=True)
                time.sleep(wait)
        raise RuntimeError("Embedding API vẫn bị giới hạn sau 6 lần thử")

    def flush(self) -> None:
        CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")


def make_backend() -> tuple[CachedEmbedder, Callable[[str], str], str]:
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    try:
        if provider == "gemini":
            model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
            emb = CachedEmbedder(GeminiEmbedder(model_name=model), model)
            llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
            from google import genai

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

            def llm(prompt: str) -> str:
                r = client.models.generate_content(model=llm_model, contents=prompt)
                return (r.text or "").strip()

            return emb, with_retry(llm), f"Gemini ({model} + {llm_model})"
        if provider == "openai":
            model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            emb = CachedEmbedder(OpenAIEmbedder(model_name=model), model)
            from openai import OpenAI

            client = OpenAI()
            llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

            def llm(prompt: str) -> str:
                r = client.chat.completions.create(
                    model=llm_model, messages=[{"role": "user", "content": prompt}], temperature=0
                )
                return r.choices[0].message.content.strip()

            return emb, with_retry(llm), f"OpenAI ({model} + {llm_model})"
        if provider == "local":
            model = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            return CachedEmbedder(LocalEmbedder(model_name=model), model), _mock_llm, f"Local ({model}, LLM=mock)"
    except Exception as exc:
        print(f"[warn] backend '{provider}' không khả dụng ({exc.__class__.__name__}: {exc}); dùng mock.")
    return CachedEmbedder(MockEmbedder(), "mock"), _mock_llm, "MockEmbedder (MD5 hash, không có ngữ nghĩa) + LLM mock"


def with_retry(fn: Callable[[str], str]) -> Callable[[str], str]:
    """Bọc llm_fn: gặp 429 (free tier 15 req/phút) thì chờ theo retryDelay rồi gọi lại."""

    def wrapped(prompt: str) -> str:
        for attempt in range(6):
            try:
                return fn(prompt)
            except Exception as exc:
                msg = str(exc)
                if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                    raise
                m = re.search(r"retry in ([0-9.]+)s", msg, re.I)
                wait = float(m.group(1)) + 2 if m else 30.0 * (attempt + 1)
                print(f"   [rate-limit LLM] chờ {wait:.0f}s rồi thử lại ({attempt + 1}/6)...", flush=True)
                time.sleep(wait)
        raise RuntimeError("LLM API vẫn bị giới hạn sau 6 lần thử")

    return wrapped


def _mock_llm(prompt: str) -> str:
    return "[mock LLM] " + prompt[-200:].replace("\n", " ")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, val = line.split(":", 1)
        val = val.split("#", 1)[0].strip().strip('"').strip("'")  # bỏ comment cuối dòng + dấu nháy
        meta[key.strip()] = val
    return meta, body.strip()


def load_corpus(chunker: Any) -> tuple[list[Document], dict[str, int]]:
    docs: list[Document] = []
    per_file: dict[str, int] = {}
    for path in sorted(DATA_DIR.glob("*.md")):
        meta, body = parse_frontmatter(path)
        doc_id = meta.get("doc_id", path.stem)
        chunks = chunker.chunk(body)
        per_file[doc_id] = len(chunks)
        for i, chunk in enumerate(chunks):
            docs.append(Document(id=f"{doc_id}#{i}", content=chunk, metadata={**meta, "doc_id": doc_id, "chunk_index": i}))
    return docs, per_file


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def score_query(item: dict, results: list[dict]) -> tuple[int, int, bool, int]:
    """Trả về (điểm 2 mức, điểm ngây thơ theo doc_id, keyword_hit, hạng của gold doc)."""
    rank = next((i + 1 for i, r in enumerate(results) if r["metadata"].get("doc_id") == item["gold_doc_id"]), 0)
    keyword_hit = any(item["gold_keyword"].lower() in r["content"].lower() for r in results)
    naive = 2 if rank == 1 else (1 if rank else 0)
    if not keyword_hit:
        strict = 0
    elif rank == 1:
        strict = 2
    elif rank:
        strict = 1
    else:
        strict = 0
    return strict, naive, keyword_hit, rank


def fmt_results(results: list[dict], width: int = 110) -> list[str]:
    out = []
    for i, r in enumerate(results, 1):
        preview = r["content"].replace("\n", " ⏎ ")[:width]
        out.append(f"   top-{i}: {r['id']:<40} audience={r['metadata'].get('audience', '?'):<8} score={r['score']:.4f}")
        out.append(f"          {preview}...")
    return out


def run_benchmark(strategy: str, embedder: CachedEmbedder, llm: Callable[[str], str], backend: str) -> tuple[str, dict]:
    chunker = STRATEGIES[strategy]()
    docs, per_file = load_corpus(chunker)
    store = EmbeddingStore(collection_name=f"bench_{strategy}", embedding_fn=embedder)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    lengths = [len(d.content) for d in docs]

    L: list[str] = []
    L.append("=" * 100)
    L.append(f"BENCHMARK — chiến lược: {STRATEGY_LABELS[strategy]}")
    L.append(f"Backend: {backend}")
    L.append(f"Corpus: {len(per_file)} file -> {store.get_collection_size()} chunk | avg_length={sum(lengths)/len(lengths):.1f} | max={max(lengths)}")
    L.append("Chunk/file: " + ", ".join(f"{k}={v}" for k, v in per_file.items()))
    L.append("=" * 100)

    total_strict = total_naive = 0
    summary = {"strategy": strategy, "chunks": store.get_collection_size(), "avg_len": sum(lengths) / len(lengths), "per_query": []}
    for item in BENCHMARK_QUERIES:
        flt = item["filter"]
        results = store.search_with_filter(item["query"], top_k=TOP_K, metadata_filter=flt) if flt else store.search(item["query"], top_k=TOP_K)
        strict, naive, kw_hit, rank = score_query(item, results)
        total_strict += strict
        total_naive += naive
        # Agent: dùng đúng tập ngữ cảnh đã lọc để câu trả lời phản ánh filter.
        answer = llm(agent._build_prompt(item["query"], results)) if results else agent.answer(item["query"], top_k=TOP_K)

        L.append(f"\nQuery #{item['id']}: {item['query']}")
        if flt:
            L.append(f"   metadata_filter = {flt}")
        L.append(f"   gold_doc = {item['gold_doc_id']} | gold_keyword = \"{item['gold_keyword']}\"")
        L.extend(fmt_results(results))
        L.append(f"   -> gold doc ở hạng: {rank or 'ngoài top-3'} | keyword trong ngữ cảnh: {'CÓ' if kw_hit else 'KHÔNG'}")
        L.append(f"   -> điểm theo doc_id (ngây thơ): {naive}/2 | điểm 2 mức (doc_id + nội dung): {strict}/2")
        L.append(f"   Agent: {answer.replace(chr(10), ' ')[:600]}")
        summary["per_query"].append({"id": item["id"], "rank": rank, "kw": kw_hit, "strict": strict, "naive": naive, "top1": results[0]["id"] if results else None, "top1_score": results[0]["score"] if results else None, "answer": answer})

    # --- Q5b (đề xuất, không tính vào /10) ---
    r5b = store.search_with_filter(EXTRA_QUERY["query"], top_k=TOP_K, metadata_filter=EXTRA_QUERY["filter"])
    s5b, n5b, kw5b, rk5b = score_query(EXTRA_QUERY, r5b)
    ans5b = llm(agent._build_prompt(EXTRA_QUERY["query"], r5b)) if r5b else ""
    L.append(f"\nQuery #5b (ĐỀ XUẤT thay Q5, không tính vào tổng): {EXTRA_QUERY['query']}")
    L.append(f"   metadata_filter = {EXTRA_QUERY['filter']} | gold_doc = {EXTRA_QUERY['gold_doc_id']} | gold_keyword = \"{EXTRA_QUERY['gold_keyword']}\"")
    L.extend(fmt_results(r5b))
    L.append(f"   -> gold doc ở hạng: {rk5b or 'ngoài top-3'} | keyword trong ngữ cảnh: {'CÓ' if kw5b else 'KHÔNG'} | điểm 2 mức: {s5b}/2")
    L.append(f"   Agent: {ans5b.replace(chr(10), ' ')[:600]}")
    summary["q5b"] = {"rank": rk5b, "kw": kw5b, "strict": s5b, "answer": ans5b}

    L.append("\n" + "-" * 100)
    L.append(f"TỔNG: điểm 2 mức = {total_strict}/10 | điểm chỉ theo doc_id = {total_naive}/10 (chênh lệch = phần bị thổi phồng)")
    L.append("-" * 100)
    summary["strict"], summary["naive"] = total_strict, total_naive

    # --- A/B filter --------------------------------------------------------
    for label, q in [("Query #5 (câu chính thức)", BENCHMARK_QUERIES[4]["query"]), ("Query #5b (đề xuất — không nêu đối tượng)", AB_EXTRA_QUERY)]:
        L.append(f"\n=== A/B FILTER — {label}: \"{q}\"")
        no_f = store.search(q, top_k=TOP_K)
        with_f = store.search_with_filter(q, top_k=TOP_K, metadata_filter={"audience": "student"})
        L.append("  KHÔNG filter:")
        L.extend("  " + l for l in fmt_results(no_f, 80))
        L.append("  CÓ filter {'audience': 'student'}:")
        L.extend("  " + l for l in fmt_results(with_f, 80))
        same = [r["id"] for r in no_f] == [r["id"] for r in with_f]
        L.append(f"  -> top-3 {'GIỐNG HỆT (filter không đổi gì)' if same else 'KHÁC NHAU (filter có tác dụng)'}")
        summary[f"ab_{'main' if 'chính' in label else 'extra'}"] = {"same": same, "no_filter": [(r["id"], r["metadata"].get("audience")) for r in no_f], "with_filter": [(r["id"], r["metadata"].get("audience")) for r in with_f]}

    embedder.flush()
    return "\n".join(L), summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="heading")
    ap.add_argument("--all", action="store_true", help="chạy cả 4 chiến lược")
    ap.add_argument("--out", default="ket_qua_benchmark.txt")
    args = ap.parse_args()

    embedder, llm, backend = make_backend()
    print(f"Embedding backend: {backend}")
    strategies = list(STRATEGIES) if args.all else [args.strategy]
    summaries = []
    for s in strategies:
        report, summary = run_benchmark(s, embedder, llm, backend)
        print(report)
        out = Path(args.out) if (not args.all or s == "heading") else Path(f"ket_qua_benchmark_{s}.txt")
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\n>>> đã ghi {out}\n")
        summaries.append(summary)

    if args.all:
        print("\n" + "=" * 100)
        print(f"{'Chiến lược':<14}{'chunks':>7}{'avg_len':>9}{'2 mức':>8}{'doc_id':>8}  hạng gold theo query 1..5   A/B Q5 đổi?  A/B Q5b đổi?")
        for sm in summaries:
            ranks = " ".join(str(q["rank"] or "-") for q in sm["per_query"])
            print(f"{sm['strategy']:<14}{sm['chunks']:>7}{sm['avg_len']:>9.0f}{sm['strict']:>6}/10{sm['naive']:>6}/10   {ranks:<26}{'không' if sm['ab_main']['same'] else 'CÓ':<12}{'không' if sm['ab_extra']['same'] else 'CÓ'}")
        Path("ket_qua_benchmark_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
