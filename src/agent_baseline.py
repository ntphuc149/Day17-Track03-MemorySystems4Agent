from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A - chỉ có short-term memory trong cùng một thread.

    - Không có User.md
    - Không nhớ facts qua thread mới
    - Là mốc so sánh công bằng cho AdvancedAgent
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI hữu ích. Hãy trả lời dựa trên lịch sử hội thoại hiện tại. "
        "Bạn không có khả năng nhớ thông tin từ các phiên trước."
    )

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0  # Baseline không compact

    # ------------------------------------------------------------------
    # Offline path (deterministic, không cần API key)
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)

        # Ghi nhận message người dùng
        session.messages.append({"role": "user", "content": message})

        # Ước tính prompt context = toàn bộ lịch sử session + system prompt
        prompt_ctx = self.SYSTEM_PROMPT + " ".join(
            m["content"] for m in session.messages
        )
        prompt_tokens = estimate_tokens(prompt_ctx)
        session.prompt_tokens_processed += prompt_tokens

        # Tạo phản hồi deterministic dựa trên lịch sử trong thread
        response_text = self._build_offline_response(session, message)

        agent_tokens = estimate_tokens(response_text)
        session.token_usage += agent_tokens

        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _build_offline_response(self, session: SessionState, message: str) -> str:
        """Trả lời dựa trên lịch sử thread hiện tại.

        Baseline chỉ biết những gì được nói trong thread này.
        """
        msg_lower = message.lower()

        # Tìm thông tin trong lịch sử thread hiện tại
        history_text = " ".join(
            m["content"] for m in session.messages if m["role"] == "user"
        )

        # Trả lời câu hỏi về tên
        if any(k in msg_lower for k in ["tên", "name", "mình là ai"]):
            import re
            name_match = re.search(
                r"(?:tên mình là|mình tên là|mình tên|tôi tên)\s+([A-ZÀ-Ỵa-zà-ỵ][A-ZÀ-Ỵa-zà-ỵ0-9_\-]+)",
                history_text, re.IGNORECASE
            )
            if name_match:
                name = name_match.group(1)
                return f"Trong cuộc trò chuyện này, bạn cho biết tên là {name}."
            return "Bạn chưa cho tôi biết tên của bạn trong cuộc trò chuyện này."

        # Trả lời câu hỏi về nghề nghiệp
        if any(k in msg_lower for k in ["nghề", "làm gì", "công việc", "engineer", "profession"]):
            import re
            prof_match = re.search(
                r"(?:mình là|mình đang làm|làm)\s+([a-zA-ZÀ-Ỵà-ỵ\s]+?engineer[a-zA-ZÀ-Ỵà-ỵ\s]*?)(?:\s+(?:cho|ở|,|\.|\n|$))",
                history_text, re.IGNORECASE
            )
            if prof_match:
                return f"Trong cuộc trò chuyện này, bạn đang làm {prof_match.group(1).strip()}."
            return "Bạn chưa đề cập đến nghề nghiệp trong cuộc trò chuyện này."

        # Trả lời câu hỏi về nơi ở
        if any(k in msg_lower for k in ["ở đâu", "nơi ở", "thành phố", "địa điểm"]):
            import re
            loc_match = re.search(
                r"(?:mình|tôi)\s+(?:đang\s+)?(?:ở|sống ở|hiện ở)\s+([A-ZÀ-Ỵa-zà-ỵ][A-ZÀ-Ỵa-zà-ỵ\s]+?)(?:\s+(?:và|,|\.|\n|$))",
                history_text, re.IGNORECASE
            )
            if loc_match:
                return f"Bạn đang ở {loc_match.group(1).strip()} theo thông tin trong cuộc trò chuyện này."
            return "Bạn chưa cho tôi biết nơi ở trong cuộc trò chuyện này."

        # Trả lời mặc định: echo lại nội dung chính
        turn_count = sum(1 for m in session.messages if m["role"] == "user")
        return (
            f"[Baseline - lượt {turn_count}] Tôi đã ghi nhận: \"{message[:80]}{'...' if len(message) > 80 else ''}\". "
            f"Tôi chỉ nhớ thông tin trong cuộc trò chuyện này."
        )

    # ------------------------------------------------------------------
    # Live path (cần LangChain + API key)
    # ------------------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Gọi LangGraph agent thật sự khi có API key."""
        try:
            from langchain_core.messages import HumanMessage
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
            response_text = result["messages"][-1].content

            session = self._session(thread_id)
            agent_tokens = estimate_tokens(response_text)
            prompt_tokens = estimate_tokens(message)
            session.token_usage += agent_tokens
            session.prompt_tokens_processed += prompt_tokens

            return {
                "response": response_text,
                "agent_tokens": agent_tokens,
                "prompt_tokens": prompt_tokens,
            }
        except Exception:
            return self._reply_offline(thread_id, message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _maybe_build_langchain_agent(self) -> None:
        """Khởi tạo LangGraph agent nếu dependencies đã cài."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            if llm is None:
                return

            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm,
                tools=[],
                checkpointer=checkpointer,
                prompt=self.SYSTEM_PROMPT,
            )
        except (ImportError, Exception):
            self.langchain_agent = None
