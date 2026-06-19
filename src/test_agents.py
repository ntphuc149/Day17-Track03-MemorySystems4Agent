from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    """Config cô lập cho test: state_dir vào tmp_path, threshold nhỏ để dễ trigger compact."""
    provider_cfg = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "profiles").mkdir()

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,   # ngưỡng nhỏ để test compact dễ kích hoạt
        compact_keep_messages=2,
        model=provider_cfg,
        judge_model=provider_cfg,
    )


# ---------------------------------------------------------------------------
# Test 1: User.md read / write / edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Chưa có file -> trả về template mặc định
    default = store.read_text(user_id)
    assert "User Profile" in default

    # Ghi nội dung mới
    store.write_text(user_id, "# My Profile\n- name: DũngCT\n- location: Huế\n")
    content = store.read_text(user_id)
    assert "DũngCT" in content
    assert "Huế" in content

    # Edit: thay location
    changed = store.edit_text(user_id, "location: Huế", "location: Đà Nẵng")
    assert changed is True
    updated = store.read_text(user_id)
    assert "Đà Nẵng" in updated
    assert "Huế" not in updated

    # edit_text trả về False khi search_text không tồn tại
    not_changed = store.edit_text(user_id, "không có đoạn này", "thay thế")
    assert not_changed is False

    # file_size > 0 sau khi ghi
    assert store.file_size(user_id) > 0

    # upsert_fact
    store.upsert_fact(user_id, "profession", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"


# ---------------------------------------------------------------------------
# Test 2: CompactMemoryManager trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "t1"

    # Thêm đủ messages để vượt ngưỡng 50 tokens
    long_msg = "Đây là một tin nhắn khá dài để ép compact memory kích hoạt. " * 5
    for i in range(5):
        mgr.append(thread_id, "user", long_msg)
        mgr.append(thread_id, "assistant", f"Phản hồi {i}.")

    # Phải đã compact ít nhất 1 lần
    assert mgr.compaction_count(thread_id) >= 1

    # Chỉ giữ lại keep_messages messages gần nhất
    ctx = mgr.context(thread_id)
    assert len(ctx["messages"]) <= mgr.keep_messages * 2  # user + assistant

    # Summary phải khác rỗng
    assert ctx["summary"] != ""


# ---------------------------------------------------------------------------
# Test 3: Cross-session recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    user_id = "recall_user"

    # --- Session 1: giới thiệu thông tin ---
    advanced = AdvancedAgent(config=config, force_offline=True)
    advanced.reply(user_id, "thread-1", "Mình tên là DũngCT.")
    advanced.reply(user_id, "thread-1", "Mình đang làm MLOps engineer.")
    advanced.reply(user_id, "thread-1", "Mình ở Đà Nẵng.")

    # --- Session 2 (thread mới): hỏi lại ---
    advanced2 = AdvancedAgent(config=config, force_offline=True)
    result = advanced2.reply(user_id, "thread-2", "Mình tên gì?")
    assert "DũngCT" in result["response"], f"Expected DũngCT in: {result['response']}"

    result2 = advanced2.reply(user_id, "thread-2", "Mình đang làm nghề gì?")
    assert "MLOps" in result2["response"], f"Expected MLOps in: {result2['response']}"

    # --- Baseline: KHÔNG nhớ qua session ---
    baseline = BaselineAgent(config=config, force_offline=True)
    baseline.reply(user_id, "thread-b1", "Mình tên là DũngCT.")

    baseline2 = BaselineAgent(config=config, force_offline=True)
    result_b = baseline2.reply(user_id, "thread-b2", "Mình tên gì?")
    # Baseline không biết vì thread-b2 chưa có thông tin tên
    assert "DũngCT" not in result_b["response"], (
        f"Baseline should NOT recall name across sessions, got: {result_b['response']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Compact giảm prompt load so với Baseline
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    user_id = "stress_user"
    thread_id = "long-thread"

    long_msg = "Mình đang viết một đoạn hội thoại rất dài để kiểm tra compact memory. " * 8

    # Feed nhiều turns vào cả hai agent trên cùng thread_id
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    n_turns = 10
    for i in range(n_turns):
        baseline.reply(user_id, thread_id, long_msg)
        advanced.reply(user_id, thread_id, long_msg)

    b_prompt = baseline.prompt_token_usage(thread_id)
    a_prompt = advanced.prompt_token_usage(thread_id)
    a_compactions = advanced.compaction_count(thread_id)

    # Advanced phải đã compact ít nhất 1 lần
    assert a_compactions >= 1, f"Expected compactions >= 1, got {a_compactions}"

    # Baseline tích lũy prompt tokens nhiều hơn Advanced sau compact
    # (Baseline không compact nên prompt tăng tuyến tính)
    assert b_prompt >= a_prompt, (
        f"Baseline prompt ({b_prompt}) nên >= Advanced prompt ({a_prompt}) sau compact"
    )
