from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Store rỗng: không gọi LLM vô ích, cũng không crash.
            return "Không tìm thấy tài liệu nào trong cơ sở tri thức để trả lời câu hỏi này."

        # 2. Build prompt — đánh số [1] [2] [3] kèm nguồn để câu trả lời truy vết được.
        prompt = self._build_prompt(question, results)

        # 3. Generate
        return self.llm_fn(prompt)

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("id", "unknown")
            context_blocks.append(
                f"[{index}] (nguồn: {source}, score={result.get('score', 0.0):.3f})\n{result['content']}"
            )
        context = "\n\n".join(context_blocks)

        return (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "QUY TẮC:\n"
            "- CHỈ dùng thông tin trong phần NGỮ CẢNH bên dưới, không dùng kiến thức bên ngoài.\n"
            "- Khi trả lời, trích dẫn số thứ tự của đoạn đã dùng, ví dụ [1] hoặc [2][3].\n"
            "- Nếu ngữ cảnh không đủ để trả lời, nói rõ: \"Không tìm thấy thông tin trong tài liệu.\"\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI (kèm trích dẫn [n]):"
        )
