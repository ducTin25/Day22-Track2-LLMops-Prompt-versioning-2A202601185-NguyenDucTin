"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


def validate_langsmith_access(client: Client) -> None:
    """Fail before paid LLM calls when the LangSmith credential is rejected."""
    try:
        next(client.list_projects(limit=1), None)
    except Exception as exc:
        raise RuntimeError(
            "LangSmith không chấp nhận API key hoặc key không có quyền "
            f"truy cập project: {exc}"
        ) from exc


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.

    Gợi ý:
        embeddings  = get_embeddings()
        text        = load_knowledge_base()
        chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
        vectorstore = build_vectorstore(chunks, embeddings)
    """
    embeddings = get_embeddings()

    text = load_knowledge_base()

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    return build_vectorstore(chunks, embeddings)


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời. "
        "Nếu context không có thông tin, hãy nói rõ là không biết.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Preserve the retrieved context in the final mapping so each top-level
    # trace exposes the complete question/context/answer triplet.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=RAG_PROMPT | llm | StrOutputParser()
        )
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> dict:
    """
    Chạy RAG chain và trả về question, retrieved context, answer.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng.
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    trace_client = Client(
        api_key=config.LANGSMITH_API_KEY,
        api_url=os.environ["LANGSMITH_ENDPOINT"],
    )
    try:
        validate_langsmith_access(trace_client)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    vectorstore = setup_vectorstore()

    chain, retriever = build_rag_chain(vectorstore)
    trace_started_at = datetime.now(timezone.utc)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        result = ask(
            chain,
            question,
            langsmith_extra={
                "client": trace_client,
                "project_name": config.LANGSMITH_PROJECT,
            },
        )
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {result['answer'][:100]}\n")

    trace_client.flush(timeout=30)
    trace_count = sum(
        1
        for _ in trace_client.list_runs(
            project_name=config.LANGSMITH_PROJECT,
            is_root=True,
            start_time=trace_started_at,
            limit=len(SAMPLE_QUESTIONS),
        )
    )
    if trace_count < len(SAMPLE_QUESTIONS):
        raise RuntimeError(
            f"LangSmith chỉ xác nhận {trace_count}/{len(SAMPLE_QUESTIONS)} traces."
        )

    print(f"\n✅ LangSmith đã xác nhận {trace_count} traces trong project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
