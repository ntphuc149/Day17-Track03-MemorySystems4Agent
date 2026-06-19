from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Ước lượng số token bằng heuristic đơn giản (chars / 4).

    Không cần chính xác theo tokenizer thật - đủ ổn định để benchmark offline.
    """
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore  -  quản lý file User.md bền vững
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE_TEMPLATE = """\
# User Profile

## Basic Info
- Name: (chưa rõ)
- Location: (chưa rõ)
- Profession: (chưa rõ)

## Preferences
- Response style: (chưa rõ)

## Interests
(chưa rõ)

## Other Facts
(chưa rõ)
"""


@dataclass
class UserProfileStore:
    """Lưu thông tin người dùng vào file User.md riêng cho từng user_id."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        # Sanitize user_id: chỉ giữ chữ, số, dấu gạch
        safe = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return _DEFAULT_PROFILE_TEMPLATE
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Thay thế đúng một lần xuất hiện đầu tiên của search_text."""
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    # --- Helper: đọc facts dưới dạng dict key->value ---
    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^-\s+([^:]+):\s*(.+)$", line)
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).strip()
                if val and val != "(chưa rõ)":
                    result[key] = val
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Cập nhật hoặc thêm một fact vào profile."""
        text = self.read_text(user_id)
        pattern = re.compile(rf"^(- {re.escape(key)}:).*$", re.MULTILINE | re.IGNORECASE)
        new_line = f"- {key}: {value}"
        if pattern.search(text):
            text = pattern.sub(new_line, text, count=1)
        else:
            # Thêm vào cuối section "Other Facts"
            if "## Other Facts" in text:
                text = text.replace(
                    "## Other Facts\n(chưa rõ)",
                    f"## Other Facts\n{new_line}",
                )
                text = text.replace(
                    "## Other Facts",
                    f"## Other Facts\n{new_line}" if new_line not in text else "## Other Facts",
                )
            else:
                text += f"\n{new_line}"
        self.write_text(user_id, text)


# ---------------------------------------------------------------------------
# extract_profile_updates  -  trích facts ổn định từ message
# ---------------------------------------------------------------------------

# Các pattern từ chối: câu hỏi thuần túy không chứa fact
_QUESTION_ONLY = re.compile(
    r"^\s*(bạn|mình|bạn có biết|bạn có thể|cho mình|hãy|nhắc|thử|kiểm tra|có thể|tại sao|vì sao|làm sao|như thế nào)[^.!?]*\?",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # (fact_key, regex, group_index_as_str)
    ("name",       re.compile(r"(?:tên mình là|tên tôi là|mình tên là|tôi tên là|mình tên)\s+([A-ZÀ-Ỵa-zà-ỵ][A-ZÀ-Ỵa-zà-ỵ0-9_\-]+)", re.IGNORECASE), "1"),
    ("location",   re.compile(r"(?:mình|tôi)\s+(?:đang\s+)?(?:ở|sống ở|làm việc ở|hiện ở|hiện tại ở|đang ở)\s+([A-ZÀ-Ỵa-zà-ỵ][A-ZÀ-Ỵa-zà-ỵ\s]+?)(?:\s+(?:và|,|\.|\n|$))", re.IGNORECASE),  "1"),
    ("profession", re.compile(r"(?:mình|tôi)\s+(?:là|đang làm|làm)\s+(?:một\s+)?([a-zA-ZÀ-Ỵà-ỵ][a-zA-ZÀ-Ỵà-ỵ\s]*?(?:engineer|developer|designer|manager|scientist|analyst|architect|researcher|MLOps|DevOps|backend|frontend|fullstack)[a-zA-ZÀ-Ỵà-ỵ\s]*?)(?=\s*(?:cho|ở|tại|và|,|\.|$))", re.IGNORECASE), "1"),
    ("drink",      re.compile(r"(?:đồ uống yêu thích|thức uống yêu thích|uống|thích uống)\s+(?:là\s+|nhất là\s+)?([a-zA-ZÀ-Ỵà-ỵ\s]+?)(?:\s*(?:và|,|\.|\n|$))", re.IGNORECASE), "1"),
    ("food",       re.compile(r"(?:món ăn yêu thích|món ruột|thích ăn|ăn)\s+(?:là\s+|nhất là\s+)?([a-zA-ZÀ-Ỵà-ỵ\s]+?)(?:\s*(?:và|,|\.|\n|$))", re.IGNORECASE), "1"),
    ("pet",        re.compile(r"(?:nuôi|có nuôi|có một\s+(?:bé|con))\s+(?:một\s+)?(?:bé\s+|con\s+)?([a-zA-ZÀ-Ỵà-ỵ]+)\s+(?:tên\s+)?([A-ZÀ-Ỵa-zà-ỵ][a-zA-ZÀ-Ỵà-ỵ]+)", re.IGNORECASE), "1+2"),
    ("style",      re.compile(r"(?:trả lời|câu trả lời|style|phong cách)\s+(?:mình thích|thích|mong muốn|là)\s+([a-zA-ZÀ-Ỵà-ỵ,\s]+?)(?:\s*(?:và|,|\.|\n|$))", re.IGNORECASE), "1"),
]

# Các từ khóa đính chính - khi có, cần cập nhật fact cũ
_CORRECTION_KEYWORDS = re.compile(
    r"(?:đính chính|không còn|giờ|bây giờ|hiện tại|chuyển sang|thay đổi|cập nhật|mới|đổi sang)",
    re.IGNORECASE,
)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Trích các fact ổn định từ message người dùng.

    Bỏ qua message là câu hỏi thuần túy.
    Trả về dict {fact_key: value}.
    """
    # Nếu message toàn là câu hỏi không có fact thì skip
    lines = [l.strip() for l in message.splitlines() if l.strip()]
    if all(_QUESTION_ONLY.match(l) for l in lines if l):
        return {}

    facts: dict[str, str] = {}

    for fact_key, pattern, group_spec in _PATTERNS:
        m = pattern.search(message)
        if not m:
            continue
        if group_spec == "1+2":
            value = f"{m.group(1).strip()} {m.group(2).strip()}"
        else:
            value = m.group(int(group_spec)).strip()
        value = value.rstrip(".,!?")
        if value:
            facts[fact_key] = value

    return facts


# ---------------------------------------------------------------------------
# summarize_messages  -  tóm tắt messages cũ thành text ngắn
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Tạo summary ngắn gọn từ danh sách messages cũ.

    Chỉ lấy content từ phía user, giữ tối đa max_items messages.
    """
    if not messages:
        return ""

    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    selected = user_msgs[-max_items:] if len(user_msgs) > max_items else user_msgs

    if not selected:
        return ""

    lines = []
    for i, msg in enumerate(selected, 1):
        # Cắt ngắn mỗi message xuống tối đa 120 ký tự
        snippet = msg[:120].replace("\n", " ").strip()
        if len(msg) > 120:
            snippet += "..."
        lines.append(f"- [{i}] {snippet}")

    return "Tóm tắt hội thoại trước:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# CompactMemoryManager  -  nén hội thoại dài
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Quản lý compact memory cho long threads.

    Khi tổng token vượt threshold, nén các messages cũ thành summary
    và chỉ giữ lại keep_messages messages gần nhất.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],      # list[dict] - messages gần nhất
                "summary": "",       # str - summary của messages đã compact
                "compactions": 0,    # int - số lần đã compact
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})  # type: ignore[index]

        # Kiểm tra có cần compact không
        total_tokens = self._estimate_thread_tokens(thread_id)
        if total_tokens > self.threshold_tokens:
            self._compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]

    def _estimate_thread_tokens(self, thread_id: str) -> int:
        thread = self.state[thread_id]
        total = estimate_tokens(str(thread["summary"]))
        for msg in thread["messages"]:  # type: ignore[union-attr]
            total += estimate_tokens(msg["content"])
        return total

    def _compact(self, thread_id: str) -> None:
        """Nén messages cũ thành summary, chỉ giữ keep_messages messages mới nhất."""
        thread = self.state[thread_id]
        messages: list[dict] = thread["messages"]  # type: ignore[assignment]

        if len(messages) <= self.keep_messages:
            return  # Chưa đủ để compact

        old_msgs = messages[: -self.keep_messages]
        recent_msgs = messages[-self.keep_messages :]

        new_summary_part = summarize_messages(old_msgs)
        existing_summary = str(thread["summary"])

        if existing_summary:
            combined = existing_summary + "\n\n" + new_summary_part
        else:
            combined = new_summary_part

        thread["summary"] = combined
        thread["messages"] = recent_msgs
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]
