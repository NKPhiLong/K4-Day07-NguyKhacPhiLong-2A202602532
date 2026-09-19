# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** GO HOME
**Thành viên:**
- Chử Trần Phương Nam - 2A202602675 - Data & Pre-filtering (chiến lược: `RecursiveChunker`)
- Ngụy Khắc Phi Long - 2A202602532 - Benchmark & Evaluation (chiến lược: `FixedSizeChunker`; viết `bench.py` chạy/chấm chung cho cả nhóm)
- Nguyễn Đức Phát - 2A202602753 - Strategy - Chunking Comparator (chiến lược: `SentenceChunker`)
- Đỗ Thành Đạt - 2A202602874 - Report & Demo (chiến lược: `HeadingChunker` — chunk theo tiêu đề/mục, bắt buộc theo K4-L3A)
**Ngày:** 19/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

> **Cách đọc số liệu trong báo cáo này.** Mỗi thành viên chạy `bench.py` trên máy mình với backend riêng (Nam: OpenAI `text-embedding-3-small` + `gpt-4o-mini`; Long: Gemini `gemini-embedding-001` + `gemini-3.5-flash-lite`). Để bảng so sánh giữa các chiến lược **chỉ khác nhau ở chunker** như codelab yêu cầu, Long đã chạy thêm **cả 4 chiến lược trên cùng backend Gemini** (`python bench.py --all`, kết quả trong `ket_qua_benchmark_<strategy>.txt`). Các bảng ở mục 2–3 dùng bộ số liệu đồng nhất này; số riêng của từng người xem trong `REPORT_CANHAN.md`.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ và Quy định Đại học (Đại học Quốc gia Hà Nội - VNU)

**Tại sao nhóm chọn chủ đề này?**
> Nhóm chọn chủ đề quy định và dịch vụ của ĐHQGHN vì đây là chủ đề bắt buộc của phân ban K4-L3A. Dữ liệu thực tế phản ánh chính xác các văn bản pháp quy, quy chế xét tuyển, chương trình đào tạo, chính sách học bổng và lịch trình học vụ có cấu trúc bảng biểu phong phú, tính cập nhật cao và phục vụ trực tiếp nhu cầu tra cứu thông tin của sinh viên và cán bộ giảng viên. Bộ tài liệu cố ý trộn **văn bản thuần chữ** (học bổng, tuyển sinh, trao đổi) với **bảng markdown lớn** (danh mục CTĐT 187 hàng, lịch trình 42 hàng, thống kê nhân sự 41 hàng) để các chiến lược chunking bộc lộ điểm mạnh/yếu rõ ràng.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Chương trình học bổng EVN ĐHQGHN năm học 2025-2026 | https://vnu.edu.vn/dhqghn-thong-bao-chuong-trinh-hoc-bong-evn_dhqghn-nam-hoc-2025-2026-post40521.html | 2026-09-19 / 2025-2026 | 3.745 | audience: student, department: student-affairs, category: scholarship |
| 2 | Phương thức tuyển sinh năm 2026 của Trường ĐH Khoa học Tự nhiên | https://vnu.edu.vn/nam-2026-truong-dh-khoa-hoc-tu-nhien-dhqghn-tuyen-2510-chi-tieu-cho-28-nganh-dao-tao-post40194.html | 2026-09-19 / 2026.1 | 2.474 | audience: student, department: academic-affairs, category: admission |
| 3 | Danh mục các chương trình đào tạo bậc đại học tại ĐHQGHN | https://vnu.edu.vn/dao-tao/chuong-trinh-dao-tao-bac-dai-hoc | 2026-09-19 / 2026.1 | 10.387 | audience: student, department: academic-affairs, category: academic-program |
| 4 | Lịch trình và kế hoạch đào tạo năm học 2025-2026 của ĐHQGHN | https://vnu.edu.vn/dao-tao/ke-hoach-hoc-tap-va-giang-day | 2026-09-19 / 2025-2026 | 3.720 | audience: student, department: academic-affairs, category: academic-calendar |
| 5 | Chương trình trao đổi kỳ Xuân 2027 tại Đại học Osaka, Nhật Bản | https://vnu.edu.vn/chuong-trinh-trao-doi-ky-xuan-2027-tai-dai-hoc-osaka-nhat-ban-post40540.html | 2026-09-19 / 2026.1 | 2.158 | audience: student, department: international-relations, category: exchange |
| 6 | Thống kê đội ngũ cán bộ giảng viên theo chức danh khoa học | https://vnu.edu.vn/can-bo/so-lieu-thong-ke/theo-chuc-danh-khoa-hoc-va-trinh-do-dao-tao | 2026-09-19 / 2026.1 | 4.220 | audience: faculty, department: human-resources, category: faculty-staff |

File `.md` + `sources.csv` nằm trong `data/university/`; mỗi file có YAML frontmatter với đủ 6 trường bắt buộc (`doc_id`, `title`, `source_url`, `retrieved_at`, `document_version`, `audience`) + `department`, `category`, `language`.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.
- [x] Ràng buộc L3A: `audience` có 2 giá trị khác nhau (`student`: 5 file, `faculty`: 1 file). *Điểm yếu tự nhận:* chỉ có một tài liệu `faculty` và không có cặp tài liệu "cùng chủ đề, khác đối tượng" — xem hệ quả ở mục 3 (A/B filter).

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `hoc-bong-evn` | Định danh tài liệu nguồn; mọi chunk của một file mang cùng `doc_id` để `delete_document` và chấm benchmark theo tài liệu. |
| `audience` | string | `student` / `faculty` | Lọc đúng đối tượng khi câu hỏi không nêu người hỏi là ai (bảng CTĐT vs bảng nhân sự cùng nhắc "Trường ĐH Công nghệ"). |
| `department` | string | `academic-affairs` | Thu hẹp không gian tìm kiếm về đúng phòng ban chức năng (Đào tạo, CTSV, HTQT, Nhân sự). |
| `category` | string | `scholarship`, `admission` | Phân loại nghiệp vụ; lọc nhanh các tài liệu cùng mảng. |
| `document_version` | string | `2025-2026`, `2026.1` | Kiểm độ mới / phiên bản hiệu lực của văn bản. |
| `source_url`, `retrieved_at` | string | … | Truy vết nguồn để kiểm chứng gold answer. |
| `chunk_index` | int | `0`, `1`, … | Thêm lúc chunk (bench.py): biết vị trí chunk trong tài liệu, phục vụ phân tích lỗi. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên phần thân (đã bỏ frontmatter) của 3 tài liệu đại diện — số liệu được cả hai thành viên chạy độc lập trên `src` riêng và **khớp nhau**:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `hoc-bong-evn.md` | FixedSizeChunker (`fixed_size`) | 15 | 189,0 | Bị cắt ngang các câu quy định điều kiện hồ sơ. |
| `hoc-bong-evn.md` | SentenceChunker (`by_sentences`) | 5 | 505,2 | Giữ trọn câu nhưng các đoạn điều khoản bị dài. |
| `hoc-bong-evn.md` | RecursiveChunker (`recursive`) | 20 | 126,0 | Tốt, giữ nguyên các đoạn phân cách dòng đôi. |
| `phuong-thuc-xet-tuyen-hus.md` | FixedSizeChunker (`fixed_size`) | 9 | 196,6 | Cắt đứt mã phương thức (301, 401...) ở mép cắt. |
| `phuong-thuc-xet-tuyen-hus.md` | SentenceChunker (`by_sentences`) | 4 | 396,2 | Tốt cho văn bản dạng gạch đầu dòng ngắn. |
| `phuong-thuc-xet-tuyen-hus.md` | RecursiveChunker (`recursive`) | 11 | 144,5 | Tách gọn từng phương thức xét tuyển riêng biệt. |
| `ke-hoach-dao-tao-nam-hoc.md` | FixedSizeChunker (`fixed_size`) | 16 | 192,7 | Làm vỡ các dòng bảng markdown (table rows). |
| `ke-hoach-dao-tao-nam-hoc.md` | SentenceChunker (`by_sentences`) | 1 | 2.780,0 | **Thất bại**: bảng không có dấu chấm nên cả tài liệu thành 1 chunk. |
| `ke-hoach-dao-tao-nam-hoc.md` | RecursiveChunker (`recursive`) | 17 | 162,5 | Tách theo từng dòng `\n`, giữ trọn từng mốc thời gian. |

### Chiến lược của từng thành viên

**Thành viên 1 — Chử Trần Phương Nam (MSSV: 2A202602675)**
- **Loại chiến lược:** `RecursiveChunker(chunk_size=400)`
- **Mô tả & lý do chọn cho chủ đề này:** Phù hợp cho văn bản quy định và bảng biểu vì ưu tiên cắt theo `\n\n` và `\n` trước khi chia nhỏ theo ký tự, giúp từng dòng bảng hoặc từng điều khoản không bị vỡ vụn. Là "đường cơ sở mạnh" để các chiến lược khác so sánh.

**Thành viên 2 — Ngụy Khắc Phi Long (MSSV: 2A202602532)**
- **Loại chiến lược:** `FixedSizeChunker(chunk_size=200, overlap=50)`
- **Mô tả & lý do chọn cho chủ đề này:** Chia đều truyền thống có overlap 50 ký tự (25%) để hạn chế mất thông tin ở biên cắt giữa hai chunk liên tiếp; dùng làm đường cơ sở "ngây thơ" để đo overlap cứu được bao nhiêu trường hợp cắt giữa câu / giữa hàng bảng. Kèm theo vai Benchmark & Evaluation, Long viết `bench.py` dùng chung cho cả nhóm: đọc frontmatter, chunk ngoài store, cache embedding, chạy 5 query (+Q5b), **chấm 2 mức** (doc_id + chuỗi đáp án trong ngữ cảnh) và A/B filter; `python bench.py --all` chạy cả 4 chiến lược trên cùng backend để bảng so sánh chỉ khác nhau ở chunker.

**Thành viên 3 — Nguyễn Đức Phát (MSSV: 2A202602753)**
- **Loại chiến lược:** `SentenceChunker(max_sentences_per_chunk=3)`
- **Mô tả & lý do chọn cho chủ đề này:** Nhóm các câu hoàn chỉnh lại với nhau để giữ toàn vẹn ngữ pháp câu văn bản thông báo học vụ. Đồng thời chạy `ChunkingStrategyComparator` cho bảng Baseline.

**Thành viên 4 — Đỗ Thành Đạt (MSSV: 2A202602874) — chunk theo heading/section (ràng buộc L3A)**
- **Loại chiến lược:** custom — `HeadingChunker(max_len=600)`, code trong `bench.py`
- **Mô tả & lý do chọn cho chủ đề này:** Văn bản thông báo/quy định đại học đã được người soạn chia sẵn thành mục ("Đối tượng và tiêu chuẩn xét chọn", "Hồ sơ đăng ký", "CHỈ TIÊU & ĐỊA ĐIỂM HỌC TẬP"), và bảng lớn được nhóm theo đơn vị (`| I. Trường Đại học Công nghệ |`). Mỗi mục là một đơn vị ngữ nghĩa trọn vẹn nên câu hỏi và đáp án thường nằm chung một chunk. Chunker nhận diện 3 dạng heading: markdown `#`, **dòng tiêu đề trần** (dòng ngắn đứng riêng, không kết thúc bằng dấu câu — corpus này hầu như không dùng `##`), và **hàng nhóm số La Mã trong bảng**. Section dài hơn `max_len` thì cắt nhỏ và **gắn lại tiêu đề tài liệu + tiêu đề mục vào từng mảnh con**; với bảng, mỗi mảnh được gắn lại **header bảng** (dòng tên cột + dòng `---`) — chính là "header-aware table row chunking" mà nhóm ước có ở lần đầu. Mục quá ngắn (< 150 ký tự) được gom vào mục kề để tránh chunk vụn.
- **Code snippet (rút gọn — bản đầy đủ trong `bench.py`):**
```python
class HeadingChunker:
    MD_HEADING  = re.compile(r"^#{1,6}\s+\S")
    TABLE_GROUP = re.compile(r"^\|\s*[IVXLC]+\.\s+[^|]+\|")   # | I. Trường ĐH Công nghệ |
    TABLE_SEP   = re.compile(r"^\|\s*-{3,}")

    def _is_bare_title(self, lines, i):          # dòng ngắn, đứng riêng, không kết thúc bằng dấu câu
        line = lines[i].strip()
        if not line or len(line) > 90 or line.endswith((".", ";", ":", ",")): return False
        if line.startswith(("-", "*", "|", ">")): return False
        return (i == 0 or not lines[i-1].strip()) and (i+1 >= len(lines) or not lines[i+1].strip())

    def chunk(self, text):
        sections = self._split_sections(text)              # [(heading, [dòng thân]), ...]
        doc_title = next((h for h, _ in sections if h), "")
        shared_header = ...                                 # 2 dòng header của bảng lớn (nếu có)
        chunks = []
        for heading, body in sections:
            prefix = f"{doc_title} — {heading}" if heading != doc_title else heading
            full = prefix + "\n" + "\n".join(body).strip()
            if len(full) <= self.max_len:
                chunks.append(full); continue
            # section dài: bảng -> cắt theo hàng, gắn lại prefix + header bảng vào từng mảnh
            #               chữ  -> RecursiveChunker(max_len), gắn lại prefix vào từng mảnh
            chunks.extend(self._split_long_section(prefix, shared_header + body if is_table else body))
        return self._merge_short(chunks)
```

### So Sánh Giữa Các Thành Viên

Cùng corpus 6 file, cùng 5 query, cùng backend Gemini (`gemini-embedding-001` + `gemini-3.5-flash-lite`), chỉ đổi dòng chọn chunker — Long chạy `bench.py --all` một lần cho cả 4 chiến lược; mỗi thành viên chạy lại chiến lược của mình trên máy mình để đối chiếu (số riêng ghi trong `REPORT_CANHAN.md`). **Điểm 2 mức** = theo `docs/SCORING.md`: 2đ nếu gold doc ở top-1 **và** ngữ cảnh chứa chuỗi đáp án, 1đ nếu gold ở top-2/3, 0đ nếu vắng hoặc ngữ cảnh không chứa đáp án. **Điểm theo doc_id** = cách chấm ngây thơ (chỉ xem gold doc có trong top-3).

| Thành viên | Chiến lược (Strategy) | Số chunk / avg_len | Điểm truy xuất 2 mức (/10) | Điểm chỉ theo doc_id | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|------|------|-----------|----------|
| Phương Nam | RecursiveChunker (400) | 60 / 334 | **8/10** | 10/10 | Chunk gọn, giữ trọn dòng bảng và đoạn điều khoản; 4/5 câu đúng ở top-1 với ngữ cảnh đủ. | Q5: bảng 18 hàng của Trường ĐH Công nghệ bị cắt làm 3 chunk, top-3 chỉ chứa 5 hàng đầu → agent liệt kê 5/18 ngành. |
| Thành Đạt | HeadingChunker (600, header-aware) | 61 / 459 | **10/10** | 10/10 | Giữ trọn từng mục; mảnh con của bảng đều mang tiêu đề + header cột nên 3 chunk top-3 ghép lại đủ 18 hàng → agent liệt kê đủ 18 ngành, có trích dẫn [1][2][3]. Score top-1 cao nhất ở 4/5 câu (0,80–0,88). | Phụ thuộc heuristic nhận diện tiêu đề (dòng ngắn đứng riêng) — với văn bản không có cấu trúc mục sẽ rơi về Recursive. `max_len=600` vẫn phải cắt bảng lớn. |
| Phi Long | FixedSizeChunker (200/50) | 135 / 197 | **8/10** | 10/10 | Số chunk nhiều, kích thước đều; overlap giúp Q1–Q4 vẫn đúng top-1. | Q5: cắt ngang hàng bảng, top-3 chỉ có 3 ngành; agent tự nhận "tài liệu không cung cấp đầy đủ danh mục". |
| Đức Phát | SentenceChunker (3 câu) | 19 / 1.057 | 10/10 | 10/10 | Điểm cao nhưng **giả tạo**: bảng không có dấu chấm nên cả tài liệu 2.780–4.000 ký tự thành 1 chunk — "trúng" vì trả về nguyên file. | Chunk quá dài: vượt ngưỡng embedding lý tưởng, nhét cả file vào prompt; không mở rộng được khi corpus lớn hơn. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Với chấm ngây thơ theo `doc_id`, cả 4 chiến lược đều 10/10 — corpus nhỏ, 6 tài liệu chủ đề rất khác nhau nên tìm đúng *file* là chuyện dễ. Sự khác biệt chỉ lộ ra khi chấm ở mức **nội dung**: câu hỏi cần **cả một bảng** (Q5, 18 hàng) làm Recursive và FixedSize rơi xuống 8/10 vì chunk chỉ giữ được phần đầu bảng, còn `HeadingChunker` đạt 10/10 nhờ hai quyết định thiết kế: (1) cắt bảng theo **nhóm đơn vị** thay vì theo số ký tự, (2) **gắn lại tiêu đề và header cột** vào từng mảnh, nên dù bảng bị chia 3 thì cả 3 mảnh vẫn "biết" mình thuộc Trường ĐH Công nghệ và cùng lọt top-3. `SentenceChunker` cũng 10/10 nhưng bằng cách phản tác dụng (chunk = nguyên file). Kết luận: với văn bản quy định/thông báo đại học có cấu trúc mục và bảng, **chunk theo cấu trúc do người soạn để lại tốt hơn chunk theo kích thước**; Recursive là mặc định an toàn khi văn bản không có cấu trúc rõ.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng (điều kiện, tra mốc thời gian trong bảng, liệt kê, tra số liệu, liệt kê từ bảng lớn + filter). Mọi gold answer đều **trích nguyên văn được** từ file trong `data/university/`. Cột "Chunk" ghi theo `HeadingChunker`.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Sinh viên năm thứ mấy và cần đạt kết quả học tập thế nào để đủ điều kiện xét học bổng EVN? | Sinh viên năm thứ 3 các ngành năng lượng, kỹ thuật…; kết quả học tập 2025–2026 đạt loại giỏi trở lên, rèn luyện từ loại tốt, chưa nhận học bổng ngoài ngân sách khác. | `hoc-bong-evn#1` (mục "Đối tượng và tiêu chuẩn xét chọn") |
| 2 | Thời gian nghỉ Tết Nguyên đán năm học 2025-2026 của sinh viên chính quy kéo dài từ ngày nào đến ngày nào? | Nghỉ Tết Nguyên đán 09/02–22/02/2026. | `ke-hoach-dao-tao-nam-hoc#4` (hàng 19 của bảng) |
| 3 | Phương thức 2 của Trường ĐH Khoa học Tự nhiên năm 2026 áp dụng nhân hệ số 2 môn Toán cho những ngành nào? | Toán học, Toán tin, Khoa học máy tính và thông tin, Khoa học dữ liệu. | `phuong-thuc-xet-tuyen-hus#1` (mục "6 PHƯƠNG THỨC XÉT TUYỂN") |
| 4 | Chương trình trao đổi sinh viên tại Đại học Osaka kỳ Xuân 2027 yêu cầu đã học bao nhiêu học kỳ và điểm GPA tối thiểu là bao nhiêu? | Hoàn thành ít nhất 02 học kỳ tại ĐHQGHN; GPA từ 3,2/4,0 trở lên. | `trao-doi-sinh-vien-osaka#3` (mục "Điều kiện chính") |
| 5 | **[Filter `audience=student`]** Số lượng và danh mục các chương trình đào tạo chuẩn và đặc thù của Trường Đại học Công nghệ là gì? | 18 chương trình: 13 chuẩn (CNTT, CNTT định hướng Nhật Bản, Kỹ thuật máy tính, Robot, Năng lượng, Vật lý KT, Cơ KT, CNKT xây dựng, Hàng không vũ trụ, Điều khiển-TĐH, CN nông nghiệp, Trí tuệ nhân tạo…) và 5 đặc thù CLC. | `chuong-trinh-dao-tao-dai-hoc#1–#3` (nhóm "I. Trường Đại học Công nghệ", 3 mảnh) |

> **Hai chỉnh sửa so với bản đầu (cần nhóm xác nhận):**
> - **Q4** bản đầu hỏi "bao nhiêu chỉ tiêu" với gold "03 chỉ tiêu" — kiểm tra lại file `trao-doi-sinh-vien-osaka.md` **không có** thông tin chỉ tiêu (agent của cả hai thành viên đều trả lời "không nêu rõ"). Đã đổi thành hỏi số học kỳ + GPA, hai con số có thật trong tài liệu.
> - **Q5** giữ nguyên câu hỏi, nhưng A/B bên dưới cho thấy nó **không thực sự cần filter**. Nhóm đề xuất thay bằng **Q5b**: *"Theo thống kê của ĐHQGHN, Trường Đại học Công nghệ có tổng số bao nhiêu?"* — câu hỏi không nêu đối tượng, và corpus có 2 bảng cùng nhắc Trường ĐH Công nghệ (bảng CTĐT `student`: 18 chương trình; bảng nhân sự `faculty`: 339 cán bộ). `bench.py` chạy và chấm cả Q5b (không tính vào tổng).

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-1 là gold doc **và** ngữ cảnh chứa đáp án (2), gold ở top-2/3 (1), vắng hoặc ngữ cảnh không có đáp án (0). Số liệu: cùng backend Gemini, 4 chiến lược.

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Điều kiện học bổng EVN | Tất cả 2/2 — Heading top-1 score cao nhất (0,843) | CÓ (4/4 chiến lược) | Heading đưa đúng mục "Đối tượng và tiêu chuẩn" lên top-1; Recursive/Fixed đúng file nhưng top-1 là đoạn mở đầu, đáp án ở top-2/3. |
| 2 | Nghỉ Tết 09/02–22/02/2026 | Tất cả 2/2 | CÓ (4/4) | Sentence "đúng" vì cả bảng là 1 chunk 2.780 ký tự. |
| 3 | Ngành nhân hệ số 2 môn Toán | Tất cả 2/2 | CÓ (4/4) | Câu dễ: danh sách nằm gọn trong một dòng "Lưu ý". |
| 4 | Học kỳ + GPA trao đổi Osaka | Tất cả 2/2 — Heading top-1 0,877 (cao nhất toàn benchmark) | CÓ (4/4) | Mục "Điều kiện chính" là chunk riêng nên khớp gần như hoàn hảo. |
| 5 | Danh mục 18 CTĐT Trường ĐH Công nghệ (filter student) | **Heading 2/2**, Sentence 2/2 (nguyên file); **Recursive 0/2, Fixed 0/2** | Theo doc_id: CÓ (4/4); theo nội dung (hàng 18 "Trí tuệ nhân tạo"): chỉ Heading & Sentence | Failure case chính — xem mục 4. |
| **Tổng** | | | | **Heading 10/10 · Sentence 10/10 · Recursive 8/10 · Fixed 8/10** (chấm doc_id: cả 4 đều 10/10) |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Với Q5 chính thức: không.** A/B trên cả 4 chiến lược: top-3 *có* và *không* filter **giống hệt nhau** (cả 3 đều là mảnh bảng CTĐT, `audience=student`). Lý do: câu hỏi đã nêu rõ "chương trình đào tạo chuẩn và đặc thù", embedding tự loại tài liệu nhân sự mà không cần filter — filter chỉ có việc khi câu hỏi *thiếu* thông tin định hướng. Bản đầu của báo cáo ghi filter "loại bỏ 100% tài liệu nhân sự" là suy diễn, không khớp số liệu; nhóm sửa lại.
> **Với Q5b (câu không nêu đối tượng): có, ở cả 4 chiến lược.** Không filter → top-1 là `co-cau-doi-ngu-can-bo-giang-vien#0` (`faculty`, bảng có hàng "Trường Đại học Công nghệ | 339"); Recursive thậm chí cả 3 slot top-3 đều là bảng nhân sự. Có `metadata_filter={"audience":"student"}` → top-3 chuyển hoàn toàn sang bảng CTĐT và (với Heading/Sentence) chứa đủ hàng 18. Đây là bằng chứng A/B nhóm dùng cho ràng buộc L3A. Đánh đổi quan sát được: câu hỏi mơ hồ đến mức cần filter thì cũng mơ hồ với LLM — agent (prompt chống bịa) trả lời *"Không tìm thấy thông tin trong tài liệu"* ở cả 4 chiến lược vì bảng không ghi con số "18" mà chỉ liệt kê 18 hàng. Filter sửa được **truy xuất**, không sửa được **câu hỏi**.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
1. **Chấm theo `doc_id` thổi phồng kết quả.** Cả 4 chiến lược đều 10/10 nếu chỉ hỏi "gold doc có trong top-3 không", nhưng chấm theo nội dung thì Recursive và FixedSize rơi xuống 8/10. Chênh lệch 2 điểm đó chính là "top-3 đúng tài liệu nhưng sai section" — phát hiện đáng giá nhất của buổi lab.
2. **Cấu trúc do người soạn để lại là tín hiệu chunking tốt nhất.** `SentenceChunker` thất bại trên bảng (không có dấu chấm → 1 chunk 2.780 ký tự); `FixedSize` xé hàng bảng; `Recursive` giữ hàng nhưng không giữ *bảng*. `HeadingChunker` cắt theo mục/nhóm và gắn lại header cột nên là chiến lược duy nhất vừa có chunk ≤ 650 ký tự vừa trả lời đủ 18 ngành.
3. **Metadata filter chỉ có giá trị khi câu hỏi thiếu thông tin định hướng** — A/B Q5 (không đổi) vs Q5b (đổi hoàn toàn) trên cùng corpus, cùng chunker. Và filter không thay thế được câu hỏi rõ ràng: agent grounded vẫn từ chối trả lời Q5b.

**Phân tích lỗi (Failure Analysis — Bài 3.5):**
- **Câu hỏi hỏng:** Q5 với `RecursiveChunker(400)` và `FixedSizeChunker(200/50)` — top-3 đều đúng file `chuong-trinh-dao-tao-dai-hoc` nhưng chỉ chứa 5 (Recursive) hoặc 3 (Fixed) hàng đầu của bảng 18 hàng; agent liệt kê thiếu, Fixed còn tự nhận "tài liệu không cung cấp đầy đủ".
- **Vì sao:** cosine đo độ giống *chủ đề*, không đo *độ đầy đủ* — mảnh chứa "Công nghệ thông tin, Kỹ thuật máy tính…" giống câu hỏi nhất nên chiếm top-1, các mảnh còn lại của bảng (hàng 6–18) không mang tiêu đề "Trường Đại học Công nghệ" nên điểm thấp hơn cả bảng của trường khác. Chunk theo kích thước làm mất liên kết "mảnh này thuộc bảng nào".
- **Đề xuất sửa:** (a) gắn lại tiêu đề + header vào từng mảnh (đã làm trong `HeadingChunker` → 10/10); (b) tăng `top_k` cho câu hỏi dạng liệt kê; (c) với bảng, nhúng thêm một chunk "tóm tắt" (tên nhóm + số hàng) để câu hỏi "bao nhiêu" có đáp án trực tiếp.
- **Failure case thứ hai (Q5b):** filter đưa đúng bảng lên top-3 nhưng agent vẫn trả lời "không tìm thấy" vì con số 18 không xuất hiện tường minh — lỗi nằm ở **câu hỏi mơ hồ + dữ liệu không có tổng**, không phải ở retrieval.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng 6 tài liệu, cùng 5 câu hỏi, cùng backend: chọn sai chunker (cắt câu trên bảng, cắt cứng theo ký tự) không làm hỏng việc *tìm đúng file* nhưng làm hỏng việc *đưa đủ ngữ cảnh cho LLM* — và đó mới là thứ người dùng thấy. Số chunk và độ dài trung bình (19/1.057 – 135/197) không nói lên chất lượng; phải nhìn vào chunk thực sự được trả về. Ngoài ra, hai thành viên dùng hai backend khác nhau (OpenAI vs Gemini) cho score tuyệt đối khác nhau (0,58–0,70 vs 0,75–0,88) nhưng **thứ hạng** giống nhau — so sánh chiến lược phải dựa trên thứ hạng và nội dung, không dựa trên score.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> (1) Chọn corpus có **cặp tài liệu cùng chủ đề, khác `audience`** ngay từ đầu (vd. quy định mượn thư viện cho sinh viên và cho giảng viên) thay vì 5 student / 1 faculty, để câu hỏi filter là câu tự nhiên chứ không phải câu ép mơ hồ. (2) Thống nhất **một backend embedding** cho cả nhóm trước khi chạy (local `sentence-transformers` đa ngữ — miễn phí, không cần key, không rate-limit) để bảng so sánh sạch. (3) Với tài liệu dạng bảng, chuẩn hoá thêm một dòng tổng ("Tổng: 18 chương trình") lúc làm sạch dữ liệu, vì LLM đếm hàng bảng không đáng tin.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 9 / 10 — nguồn minh bạch, metadata đủ; trừ vì chỉ 1 tài liệu `faculty`, không có cặp đối chứng tự nhiên |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 — 4 chiến lược khác nhau, có custom chunker theo heading, so sánh cùng backend, giải thích được vì sao |
| Chất lượng truy xuất (Retrieval Quality) | 9 / 10 — Heading 10/10 theo chấm 2 mức; trừ vì Q5 chính thức chưa thực sự cần filter (đã có Q5b làm bằng chứng) |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **38 / 40** |
