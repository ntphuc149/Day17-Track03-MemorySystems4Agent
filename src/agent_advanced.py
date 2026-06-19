from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B - có đủ 3 lớp memory:

    1. Short-term  : lịch sử hội thoại trong thread (qua CompactMemoryManager)
    2. Persistent  : User.md lưu facts ổn định của người dùng
    3. Compact     : tự động nén khi thread quá dài
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI có khả năng ghi nhớ thông tin người dùng. "
        "Bạn sử dụng hồ sơ người dùng (User.md) để trả lời chính xác hơn qua nhiều phiên. "
        "Hãy ưu tiên thông tin mới nhất khi có đính chính."
    )

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path (deterministic)
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Trích facts ổn định và lưu vào User.md
        new_facts = extract_profile_updates(message)
        for key, value in new_facts.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Thêm message vào compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Ước tính prompt context tokens
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Tạo response dựa trên persistent memory
        response_text = self._offline_response(user_id, thread_id, message)

        # 5. Lưu response vào compact memory và cập nhật token counter
        self.compact_memory.append(thread_id, "assistant", response_text)
        agent_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Ước tính số token mang vào mỗi lượt: User.md + summary + recent messages."""
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)

        summary_text = str(ctx.get("summary", ""))
        messages: list[dict] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(m["content"] for m in messages)

        return (
            estimate_tokens(profile_text)
            + estimate_tokens(summary_text)
            + estimate_tokens(recent_text)
            + estimate_tokens(self.SYSTEM_PROMPT)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Trả lời deterministic dựa trên User.md + compact memory."""
        msg_lower = message.lower()
        profile = self.profile_store.read_text(user_id)
        facts = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))

        # --- Câu hỏi về tên ---
        if any(k in msg_lower for k in ["tên", "mình là ai", "tôi là ai"]):
            name = facts.get("name")
            if name:
                return f"Tên của bạn là **{name}**."
            return "Tôi chưa có thông tin tên của bạn trong hồ sơ."

        # --- Câu hỏi về nghề nghiệp ---
        if any(k in msg_lower for k in ["nghề", "làm gì", "công việc", "profession", "engineer"]):
            prof = facts.get("profession")
            if prof:
                return f"Nghề nghiệp hiện tại của bạn là **{prof}**."
            return "Tôi chưa có thông tin nghề nghiệp của bạn."

        # --- Câu hỏi về nơi ở ---
        if any(k in msg_lower for k in ["ở đâu", "nơi ở", "thành phố", "địa điểm", "đang ở"]):
            loc = facts.get("location")
            if loc:
                return f"Bạn đang ở **{loc}**."
            return "Tôi chưa có thông tin nơi ở của bạn."

        # --- Câu hỏi về đồ uống ---
        if any(k in msg_lower for k in ["đồ uống", "thức uống", "uống gì"]):
            drink = facts.get("drink")
            if drink:
                return f"Đồ uống yêu thích của bạn là **{drink}**."
            return "Tôi chưa có thông tin đồ uống yêu thích của bạn."

        # --- Câu hỏi về món ăn ---
        if any(k in msg_lower for k in ["món ăn", "ăn gì", "món ruột", "thích ăn"]):
            food = facts.get("food")
            if food:
                return f"Món ăn yêu thích của bạn là **{food}**."
            return "Tôi chưa có thông tin món ăn yêu thích của bạn."

        # --- Câu hỏi về thú cưng ---
        if any(k in msg_lower for k in ["nuôi", "thú cưng", "pet", "chó", "mèo", "corgi"]):
            pet = facts.get("pet")
            if pet:
                return f"Bạn nuôi **{pet}**."
            return "Tôi chưa có thông tin về thú cưng của bạn."

        # --- Câu hỏi về style trả lời ---
        if any(k in msg_lower for k in ["style", "phong cách", "trả lời", "ngắn gọn"]):
            style = facts.get("style")
            if style:
                return f"Style trả lời bạn thích: **{style}**."
            return "Tôi chưa có thông tin về phong cách trả lời bạn muốn."

        # --- Câu hỏi tổng hợp (mô tả về user) ---
        if any(k in msg_lower for k in ["mô tả", "tóm tắt", "tóm lại", "biết gì về", "nhắc lại", "nhớ gì"]):
            if facts:
                lines = [f"- **{k}**: {v}" for k, v in facts.items()]
                return "Đây là những gì tôi nhớ về bạn:\n" + "\n".join(lines)
            return "Tôi chưa có đủ thông tin về bạn."

        # --- Response mặc định: xác nhận đã ghi nhớ và cho biết facts hiện có ---
        if facts:
            known = ", ".join(f"{k}: {v}" for k, v in list(facts.items())[:3])
            return (
                f"[Advanced] Đã ghi nhận thông tin. "
                f"Hồ sơ hiện tại: {known}{'...' if len(facts) > 3 else ''}."
            )

        return "[Advanced] Đã ghi nhận thông tin. Hãy cho tôi biết thêm về bạn để tôi có thể hỗ trợ tốt hơn."

    # ------------------------------------------------------------------
    # Live path (cần LangChain + API key)
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Gọi LangGraph agent với tools đọc/ghi User.md."""
        try:
            from langchain_core.messages import HumanMessage

            profile_text = self.profile_store.read_text(user_id)
            system_with_profile = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"## Hồ sơ người dùng (User.md)\n{profile_text}"
            )

            config_kwargs = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config_kwargs,
            )
            response_text = result["messages"][-1].content

            # Cập nhật facts từ message
            new_facts = extract_profile_updates(message)
            for key, value in new_facts.items():
                self.profile_store.upsert_fact(user_id, key, value)

            agent_tokens = estimate_tokens(response_text)
            prompt_tokens = estimate_tokens(system_with_profile + message)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

            return {
                "response": response_text,
                "agent_tokens": agent_tokens,
                "prompt_tokens": prompt_tokens,
            }
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    # ------------------------------------------------------------------
    # Build LangGraph agent (optional, cần dependencies)
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self) -> None:
        """Wire LangGraph agent với tool đọc/ghi User.md và compact memory."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from langchain_core.tools import tool

            llm = build_chat_model(self.config.model)
            if llm is None:
                return

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Doc ho so nguoi dung tu User.md."""
                return profile_store.read_text(user_id)

            @tool
            def write_user_fact(user_id: str, key: str, value: str) -> str:
                """Luu mot fact ve nguoi dung vao User.md."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Da luu: {key} = {value}"

            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm,
                tools=[read_user_profile, write_user_fact],
                checkpointer=checkpointer,
                prompt=self.SYSTEM_PROMPT,
            )
        except (ImportError, Exception):
            self.langchain_agent = None
