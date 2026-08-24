"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Stable, user-specific names make reruns update the same Hub prompts.
PROMPT_V1_NAME = "day22-nguyenductin-rag-v1"
PROMPT_V2_NAME = "day22-nguyenductin-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: concise, friendly, and strictly grounded in the retrieved context.
# Gợi ý: "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.
#          Giữ câu trả lời ngắn gọn (2-4 câu). ..."
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện và chính xác. Chỉ sử dụng thông tin trong "
    "context để trả lời câu hỏi. Trả lời ngắn gọn trong 2-4 câu; nếu context "
    "không đủ thông tin, hãy nói rõ là không biết thay vì suy đoán.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2: structured expert tone with explicit evidence and uncertainty.
# Gợi ý: "Bạn là chuyên gia AI. Đọc kỹ context, xác định facts liên quan,
#          viết câu trả lời rõ ràng và có tổ chức (3-5 câu). ..."
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Hãy đọc kỹ context và trả lời "
    "theo cấu trúc: (1) kết luận chính, (2) bằng chứng liên quan từ context, "
    "(3) mức độ chắc chắn hoặc giới hạn thông tin. Viết rõ ràng trong 3-5 câu, "
    "không thêm kiến thức bên ngoài context và không suy đoán.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def resolve_hub_prompt_names(client: Client) -> dict:
    """Return public owner-qualified or private tenant prompt identifiers."""
    owner = config.LANGSMITH_PROMPT_OWNER
    if not owner:
        settings = client._get_settings()
        owner = (settings.tenant_handle or "").strip()
    if not owner:
        # Private prompts belong to the current tenant and do not require a
        # public Prompt Hub handle.
        return {"v1": PROMPT_V1_NAME, "v2": PROMPT_V2_NAME}
    return {
        "v1": f"{owner}/{PROMPT_V1_NAME}",
        "v2": f"{owner}/{PROMPT_V2_NAME}",
    }


def push_prompts_to_hub(client: Client, hub_names: dict):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    pushed_urls = {}
    try:
        url = client.push_prompt(
            hub_names["v1"],
            object=PROMPT_V1,
            description="V1 - trợ lý thân thiện, trả lời ngắn gọn và grounded",
        )
        pushed_urls["v1"] = url
        print(f"✅ Đã push V1 → {url}")
    except Exception as exc:
        raise RuntimeError(
            f"Không thể push '{hub_names['v1']}' lên Prompt Hub: {exc}"
        ) from exc

    try:
        url = client.push_prompt(
            hub_names["v2"],
            object=PROMPT_V2,
            description="V2 - chuyên gia phân tích, trả lời có cấu trúc",
        )
        pushed_urls["v2"] = url
        print(f"✅ Đã push V2 → {url}")
    except Exception as exc:
        raise RuntimeError(
            f"Không thể push '{hub_names['v2']}' lên Prompt Hub: {exc}"
        ) from exc

    return pushed_urls


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client, hub_names: dict) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(hub_names["v1"])
        print(f"↓ Đã pull '{hub_names['v1']}' từ Hub")
    except Exception as exc:
        raise RuntimeError(
            f"Không thể pull '{hub_names['v1']}' từ Prompt Hub: {exc}"
        ) from exc

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(hub_names["v2"])
        print(f"↓ Đã pull '{hub_names['v2']}' từ Hub")
    except Exception as exc:
        raise RuntimeError(
            f"Không thể pull '{hub_names['v2']}' từ Prompt Hub: {exc}"
        ) from exc

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "version": version,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    hub_names = resolve_hub_prompt_names(client)
    if "/" in hub_names["v1"]:
        print(f"📦 Prompt Hub owner: {hub_names['v1'].split('/', 1)[0]}")
    else:
        print("📦 Prompt Hub scope: private workspace")

    push_prompts_to_hub(client, hub_names)

    prompts = pull_prompts_from_hub(client, hub_names)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    routing_log = ["A/B Routing Log", "=" * 60]
    trace_started_at = datetime.now(timezone.utc)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(
            retriever,
            llm,
            prompt,
            question,
            version_tag,
            langsmith_extra={
                "client": client,
                "project_name": config.LANGSMITH_PROJECT,
            },
        )

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        log_line = f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}..."
        print(log_line)
        routing_log.append(log_line)

    summary = (
        f"Routing: V1={v1_count} câu | V2={v2_count} câu | "
        f"Tổng={len(SAMPLE_QUESTIONS)}"
    )
    print(f"\n📊 {summary}")
    routing_log.append(summary)

    evidence_path = (
        Path(__file__).parent.parent / "evidence" / "02_ab_routing_log.txt"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("\n".join(routing_log) + "\n", encoding="utf-8")
    print(f"💾 Đã lưu log bằng chứng vào {evidence_path}")

    client.flush(timeout=30)
    trace_count = sum(
        1
        for _ in client.list_runs(
            project_name=config.LANGSMITH_PROJECT,
            is_root=True,
            start_time=trace_started_at,
            limit=len(SAMPLE_QUESTIONS),
        )
    )
    if trace_count < len(SAMPLE_QUESTIONS):
        raise RuntimeError(
            f"LangSmith chỉ xác nhận {trace_count}/{len(SAMPLE_QUESTIONS)} A/B traces."
        )
    print(f"✅ LangSmith đã xác nhận {trace_count} A/B traces.")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
