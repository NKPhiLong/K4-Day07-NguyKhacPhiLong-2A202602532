# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguy Khắc Phi Long
**MSSV:** 2A202602532
**Nhóm:** Nhóm L3A - Quy Định & Dịch Vụ Đại Học (vai R3 · Strategy — chunk theo heading)
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding gần như cùng hướng trong không gian nhiều chiều, tức mô hình "hiểu" hai đoạn văn bản nói về cùng một ý/chủ đề — dù từ vựng có thể khác nhau hoàn toàn. Trong retrieval, đoạn có cosine cao với câu hỏi sẽ được xếp lên đầu top-k. Lưu ý: cosine cao chỉ nói lên *giống về nghĩa/chủ đề*, không đảm bảo đoạn đó *chứa câu trả lời*.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên phải nộp học phí trước ngày 15 tháng 9."
- Câu B: "Hạn chót đóng tiền học kỳ này là 15/9."
- Tại sao tương đồng: gần như không dùng chung từ ("nộp học phí" vs "đóng tiền học", "trước ngày" vs "hạn chót") nhưng cùng diễn đạt một sự kiện. Mô hình embedding ánh xạ chúng về vùng gần nhau — chứng tỏ embedding hiểu nghĩa chứ không so khớp từ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên phải nộp học phí trước ngày 15 tháng 9."
- Câu B: "Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần."
- Tại sao khác: khác chủ đề (tài chính vs giờ mở cửa thư viện), khác thực thể, khác loại thông tin — vector nằm ở hai hướng khác nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo *góc* (hướng) và bỏ qua độ lớn của vector — độ lớn thường không mang nghĩa (bị ảnh hưởng bởi độ dài văn bản, tần suất từ), nên hai đoạn dài/ngắn cùng nghĩa vẫn có cosine cao trong khi khoảng cách Euclid bị "kéo" xa. Ngoài ra, ở không gian hàng trăm–nghìn chiều, khoảng cách Euclid giữa mọi cặp điểm có xu hướng gần bằng nhau (curse of dimensionality) nên khó phân biệt; còn với vector đã chuẩn hoá (`||v|| = 1`), cosine = dot product, tính rất rẻ và đúng với cách các mô hình embedding được huấn luyện.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* mỗi chunk mới tiến thêm `step = chunk_size − overlap = 500 − 50 = 450` ký tự.
> `số_chunk = ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* **23 chunks**. Kiểm lại bằng code có sẵn: `len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000))` → **23** ✔.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Overlap = 100 → `step = 400`, `ceil(9900 / 400) = ceil(24.75) = 25` chunks (kiểm bằng `FixedSizeChunker` cũng ra **25**) — tăng 2 chunk, tức tốn thêm bộ nhớ và số lần embed. Vẫn muốn overlap lớn hơn vì mỗi ranh giới cắt là một chỗ có thể xé đôi một câu/ý; overlap lớn cho mỗi mẩu thông tin nằm trọn trong ít nhất một chunk, giữ ngữ cảnh liên tục giữa các chunk kề nhau và tăng cơ hội chunk chứa đáp án lọt top-k.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=[.!?])\s+` — *lookbehind* để tách ở vị trí **ngay sau** dấu câu mà vẫn giữ nguyên dấu (nếu split bằng `[.!?]\s+` thì dấu bị nuốt và mọi chunk thành câu cụt). Pattern này bao phủ cả `". "`, `"! "`, `"? "` lẫn `".\n"`. Sau khi tách, `strip()` từng câu, bỏ câu rỗng, rồi gom `max_sentences_per_chunk` câu liền kề thành một chunk nối bằng khoảng trắng. Edge case đã xử lý: text rỗng/toàn khoảng trắng → `[]`. **Edge case biết nhưng chưa xử lý**: chữ viết tắt (`TS.`, `v.v.`, `ThS.`) và số thập phân kiểu `3. 5` sẽ bị cắt sai vì regex không phân biệt được dấu chấm kết câu với dấu chấm viết tắt.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Với separator hiện tại, tách text thành các mảnh rồi chạy **hai chiều**: (1) *gom lên* — nối các mảnh nhỏ liền kề vào một buffer cho tới sát `chunk_size` (thiếu bước này, file nhiều dòng ngắn sẽ sinh hàng trăm chunk vụn); (2) *đệ quy xuống* — mảnh nào một mình vẫn dài hơn `chunk_size` thì gọi lại `_split` với các separator còn lại. Ba base case: text đã ≤ `chunk_size` → trả nguyên; hết separator (kể cả `separators=[]` truyền vào từ đầu) → cắt cứng theo `chunk_size`; separator là `""` → cũng cắt cứng theo ký tự. Nếu separator không xuất hiện trong text thì bỏ qua và hạ xuống separator kế tiếp.

**`compute_similarity`** — hướng tiếp cận:
> Tái sử dụng `_dot` có sẵn: `norm_a = sqrt(dot(a,a))`, `norm_b = sqrt(dot(b,b))`; nếu một trong hai bằng 0 thì trả `0.0` (chặn `ZeroDivisionError`), ngược lại trả `dot(a,b) / (norm_a * norm_b)`.

**`ChunkingStrategyComparator.compare`** — hướng tiếp cận:
> Khởi tạo ba chunker với cùng `chunk_size` (`FixedSizeChunker` dùng overlap = 10% chunk_size, `SentenceChunker` 3 câu/chunk, `RecursiveChunker`), chạy trên cùng text, trả dict đúng ba key `fixed_size` / `by_sentences` / `recursive`, mỗi key gồm `count`, `avg_length`, `chunks`. `avg_length` chỉ chia khi `count > 0`, text rỗng → `0.0`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Bỏ hẳn nhánh ChromaDB (`_use_chroma = False` cố định) để tránh bẫy máy chấm có cài `chromadb` làm code rẽ vào nhánh chưa cài đặt. Lưu trữ in-memory: mỗi `Document` → một record dict `{index, id, content, metadata, embedding}` qua helper `_make_record` (embed ngay lúc nạp; 1 Document = 1 record, không tự chunk). `search` chỉ là `_search_records(query, self._store, top_k)`: embed query, tính `_dot` với từng record (vector đã chuẩn hoá nên dot product = cosine), sort giảm dần theo score, cắt top-k, và **bỏ trường `embedding`** khỏi kết quả để output gọn.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, search sau**: lọc `self._store` giữ record có mọi cặp `key == value` trong `metadata_filter` khớp, rồi đưa tập ứng viên đó vào cùng `_search_records`. Nếu search top-k rồi mới lọc, k slot có thể bị tài liệu sai chiếm hết và trả rỗng dù store còn kết quả hợp lệ. Vì `search` và `search_with_filter` đi chung một đường code nên `metadata_filter=None` cho kết quả y hệt `search`. `delete_document` xây lại danh sách bỏ mọi record có `metadata['doc_id'] == doc_id`, trả `True` khi kích thước giảm. `_make_record` copy metadata (không sửa object của người gọi) và `setdefault('doc_id', doc.id)` để delete luôn có khoá để so; khi chunk ngoài (`bench.py`) tôi sẽ đặt `doc_id` trỏ về tên file gốc còn `Document.id` là `"file#i"`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba nhịp: `store.search(question, top_k)` → dựng prompt → `llm_fn(prompt)`. Nếu store rỗng, trả thẳng thông báo "không tìm thấy tài liệu" mà không gọi LLM. Prompt gồm: khối quy tắc (chỉ dùng ngữ cảnh được cấp, không dùng kiến thức ngoài, không có thì nói rõ không tìm thấy), khối **NGỮ CẢNH** trong đó mỗi chunk được đánh số `[1] [2] [3]` kèm nguồn (`metadata.source` / `doc_id`) và score, rồi câu hỏi và yêu cầu trả lời **kèm trích dẫn `[n]`** — để câu trả lời truy vết được về đúng chunk/file (tiêu chí *Source Traceability*).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ .venv/bin/python -m pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/nguykhacphilong/Downloads/K4-Day07-NguyKhacPhiLong-2A202602532/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nguykhacphilong/Downloads/K4-Day07-NguyKhacPhiLong-2A202602532
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.03s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

> Môi trường: Python 3.11.16 (cài qua `uv`), `pytest==9.1.1`, `sentence-transformers==5.6.0`, `google-genai` — đúng bản pin trong `requirements*.txt`. `python main.py "Chunking là gì?"` chạy trọn vẹn (dòng `Skipping missing file: data/customer_support_playbook.txt` là bình thường).

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Script: [`scripts/similarity_predictions.py`](../scripts/similarity_predictions.py) — gọi `compute_similarity()` trên 5 cặp câu, backend đọc từ `.env`. Dự đoán được ghi **trước** khi chạy.

**Backend:** `gemini-embedding-001` (3072 chiều) qua Gemini API. Cột cuối là điểm khi chạy bằng `MockEmbedder` để đối chiếu.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (Gemini) | Đúng? | Mock |
|------|-----------|-----------|---------|--------------|-------|------|
| 1 | Sinh viên phải nộp học phí trước ngày 15 tháng 9. | Hạn chót đóng tiền học kỳ này là 15/9. | cao | **0.906** | ✅ | 0.151 |
| 2 | Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần. | Giờ phục vụ của thư viện là 7:30–21:00 từ thứ Hai đến thứ Sáu. | cao | **0.914** | ✅ | 0.015 |
| 3 | Sinh viên phải nộp học phí trước ngày 15 tháng 9. | Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần. | thấp | **0.612** | ✅ (thấp nhất nhóm, nhưng trị tuyệt đối vẫn cao) | 0.010 |
| 4 | Sinh viên được mượn tối đa 5 cuốn sách trong 14 ngày. | Giảng viên được mượn tối đa 20 cuốn sách trong 180 ngày. | cao | **0.856** | ✅ | −0.094 |
| 5 | Đơn phúc khảo nộp trong vòng 7 ngày sau khi công bố điểm. | Món phở bò cần ninh xương ít nhất 6 tiếng. | thấp | **0.605** | ✅ | −0.206 |

Thứ tự xếp hạng đúng 5/5 so với dự đoán (cao: 2 > 1 > 4; thấp: 3 > 5).

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là **cặp 4 đạt 0.856** — hai câu nói về hai đối tượng khác nhau (sinh viên vs giảng viên) với hai đáp án hoàn toàn khác (5 cuốn/14 ngày vs 20 cuốn/180 ngày), nhưng embedding coi chúng gần như cùng một ý vì cùng khung "được mượn tối đa X cuốn trong Y ngày". Cosine đo *độ giống chủ đề/cấu trúc*, không đo *đúng đối tượng hay đúng con số* — nên nếu corpus có hai tài liệu như vậy, câu hỏi "được mượn bao nhiêu cuốn?" sẽ lẫn cả hai vào top-k và agent có thể trả lời sai đối tượng. Đây là bằng chứng trực tiếp cho việc `metadata_filter={"audience": "student"}` là bắt buộc chứ không phải tính năng phụ.
> Bất ngờ thứ hai: cặp "thấp" với Gemini vẫn ~0.60 chứ không gần 0 — mô hình có **sàn similarity cao** (mọi câu tiếng Việt ngắn cùng văn phong hành chính đều "hơi giống nhau"), nên khi chấm benchmark phải nhìn **chênh lệch tương đối** giữa các ứng viên thay vì đặt ngưỡng tuyệt đối. Đối chiếu với cột Mock: mock băm MD5 nên điểm là nhiễu thuần (cặp 4 âm, cặp 2 gần 0), xác nhận cảnh báo trong codelab rằng không được dùng mock để đo chất lượng retrieval.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** (trùng với `REPORT_NHOM.md` mục 3) trên `src` của tôi bằng [`bench.py`](../bench.py), output đầy đủ tại [`ket_qua_benchmark.txt`](../ket_qua_benchmark.txt).

- **Chiến lược của tôi:** `HeadingChunker(max_len=600)` — chunk theo tiêu đề/mục, gắn lại tiêu đề + header bảng vào từng mảnh con (code trong `bench.py`, giải thích ở REPORT_NHOM mục 2).
- **Backend:** Gemini `gemini-embedding-001` (embedding, cache theo SHA-256) + `gemini-3.5-flash-lite` (LLM của agent). Corpus 6 file → **61 chunk**, avg_length 459, max 652.
- **Cách chấm:** 2 mức theo `docs/SCORING.md` — gold doc ở top-1 **và** ngữ cảnh chứa chuỗi đáp án = 2; gold ở top-2/3 = 1; vắng hoặc không chứa đáp án = 0.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên năm thứ mấy và cần đạt kết quả học tập thế nào để đủ điều kiện xét học bổng EVN? | `hoc-bong-evn#1` — mục "Đối tượng và tiêu chuẩn xét chọn": năm thứ 3, loại giỏi, rèn luyện tốt | 0,8431 | CÓ (top-1; chuỗi "năm thứ 3" có trong ngữ cảnh) → **2/2** | Sinh viên năm thứ 3, kết quả học tập 2025–2026 đạt loại giỏi trở lên [1]. ✅ khớp gold |
| 2 | Thời gian nghỉ Tết Nguyên đán năm học 2025-2026 kéo dài từ ngày nào đến ngày nào? | `ke-hoach-dao-tao-nam-hoc#4` — mảnh bảng lịch trình (có header cột) chứa hàng 19 "Nghỉ Tết nguyên đán 09/02-22/02/2026" | 0,8122 | CÓ (top-1; "09/02-22/02/2026" có trong ngữ cảnh) → **2/2** | Từ 09/02 đến 22/02/2026 [1]. ✅ |
| 3 | Phương thức 2 của Trường ĐH KHTN năm 2026 nhân hệ số 2 môn Toán cho những ngành nào? | `phuong-thuc-xet-tuyen-hus#1` — mục "6 PHƯƠNG THỨC XÉT TUYỂN" chứa dòng "Lưu ý: Môn Toán nhân hệ số 2…" | 0,7982 | CÓ (top-1; "Khoa học dữ liệu" có) → **2/2** | Toán học, Toán tin, Khoa học máy tính và thông tin, Khoa học dữ liệu [1]. ✅ |
| 4 | Chương trình trao đổi ĐH Osaka kỳ Xuân 2027 yêu cầu đã học bao nhiêu học kỳ và GPA tối thiểu bao nhiêu? | `trao-doi-sinh-vien-osaka#3` — mục "Điều kiện chính": ≥ 02 học kỳ, GPA ≥ 3,2/4,0 | 0,8767 | CÓ (top-1; "3,2/4,0" có) → **2/2** | Đã hoàn thành ít nhất 02 học kỳ tại ĐHQGHN và GPA từ 3,2/4,0 trở lên [1]. ✅ |
| 5 | [filter `audience=student`] Số lượng và danh mục CTĐT chuẩn và đặc thù của Trường ĐH Công nghệ? | `chuong-trinh-dao-tao-dai-hoc#1` — "I. Trường Đại học Công nghệ" mảnh 1/3, kèm header cột; #2, #3 là hai mảnh còn lại của cùng bảng | 0,8376 | CÓ (top-1, top-2, top-3 đều là 3 mảnh của bảng; hàng 18 "Trí tuệ nhân tạo" có trong ngữ cảnh) → **2/2** | Liệt kê **đủ 18 ngành** (13 chuẩn + 5 đặc thù CLC), trích dẫn [1][2][3] theo từng mảnh. ✅ |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** — tổng **10/10** theo chấm 2 mức (và 10/10 theo doc_id).

**Đối chiếu với các chiến lược khác trên cùng backend** (`python bench.py --all`): Recursive(400) 8/10, FixedSize(200/50) 8/10 — cùng trượt Q5 vì chunk chỉ giữ 5 / 3 hàng đầu của bảng 18 hàng; Sentence(3) 10/10 nhưng vì cả file bảng thành một chunk 2.780 ký tự. Chiến lược của tôi là chiến lược duy nhất vừa giữ chunk ≤ 652 ký tự vừa trả lời đủ Q5. Điểm yếu tôi ghi nhận: với **Q5b** (câu hỏi không nêu đối tượng) filter đưa đúng bảng lên top-3 nhưng agent vẫn trả lời "không tìm thấy" vì bảng không ghi tường minh con số 18 — retrieval đúng mà câu hỏi/dữ liệu không đủ để LLM đếm.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Từ Nam: `RecursiveChunker` ưu tiên `\n` nên **không bao giờ xé một hàng bảng** — điểm mà FixedSize làm hỏng; tôi mượn ý này làm fallback cho section chữ dài. Nhưng khi so kết quả Q5 của hai người, tôi thấy "giữ trọn hàng" chưa đủ, phải "giữ trọn **bảng** hoặc gắn lại header" thì LLM mới đếm được — đó là lý do tôi thêm header-aware vào chunker. Bài học thứ hai: Nam chấm theo `doc_id` ra 5/5, tôi chấm theo nội dung ra 8/10 cho cùng chiến lược của Nam — cách chấm quyết định kết luận, nên nhóm thống nhất chấm 2 mức.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
