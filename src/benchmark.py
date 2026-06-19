from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """0 nếu không có fact nào, 0.5 nếu có một nửa, 1.0 nếu đủ hết."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for e in expected if e.lower() in answer_lower)
    ratio = found / len(expected)
    if ratio == 0:
        return 0.0
    if ratio < 1.0:
        return 0.5
    return 1.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Chất lượng đơn giản: recall_points + bonus nếu câu trả lời đủ dài và có cấu trúc."""
    base = recall_points(answer, expected)
    # Bonus nhỏ nếu câu trả lời có bullet hoặc đủ dài (dấu hiệu structured)
    bonus = 0.0
    if len(answer) > 50:
        bonus += 0.05
    if any(c in answer for c in ["**", "-", "•", "\n"]):
        bonus += 0.05
    return min(1.0, base + bonus)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    benchmark_label: str = "standard",
) -> BenchmarkRow:
    """Chạy một agent qua toàn bộ conversations và trả về BenchmarkRow."""

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    memory_growth = 0

    # Dùng user_id duy nhất cho benchmark này để reset profile giữa các suite
    benchmark_user = f"bench_{benchmark_label}"

    for conv in conversations:
        user_id = conv.get("user_id", benchmark_user)
        thread_id = conv["id"]

        # Feed tất cả turns vào agent
        for turn in conv.get("turns", []):
            agent.reply(user_id, thread_id, turn)

        # Thu thập token stats từ thread này
        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)

        # Đặt recall questions trong thread mới (cross-session test)
        recall_thread = f"{thread_id}_recall"
        for rq in conv.get("recall_questions", []):
            question = rq["question"]
            expected = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread, question)
            answer = result.get("response", "")

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # Memory file size (chỉ relevant với Advanced)
        if hasattr(agent, "memory_file_size"):
            memory_growth = max(memory_growth, agent.memory_file_size(user_id))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow], title: str = "") -> str:
    try:
        from tabulate import tabulate
        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        data = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.3f}",
                f"{r.response_quality:.3f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        table = tabulate(data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback nếu tabulate chưa cài
        lines = ["Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (B) | Compactions"]
        lines.append("-" * 90)
        for r in rows:
            lines.append(
                f"{r.agent_name:<20} | {r.agent_tokens_only:>12} | {r.prompt_tokens_processed:>13} "
                f"| {r.recall_score:.3f}  | {r.response_quality:.3f}   | {r.memory_growth_bytes:>10} "
                f"| {r.compactions:>11}"
            )
        table = "\n".join(lines)

    if title:
        separator = "=" * len(title)
        return f"\n{separator}\n{title}\n{separator}\n{table}\n"
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys, io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    std_conversations = load_conversations(std_path)
    stress_conversations = load_conversations(stress_path)

    print("\n" + "=" * 60)
    print("  MEMORY SYSTEMS BENCHMARK  -  Day 17 Track 03")
    print("=" * 60)

    # ---- Standard Benchmark ----
    print("\n[1/2] Running Standard Benchmark...")
    baseline_std = BaselineAgent(config=config)
    advanced_std = AdvancedAgent(config=config)

    rows_std = [
        run_agent_benchmark("Baseline", baseline_std, std_conversations, config, "std"),
        run_agent_benchmark("Advanced", advanced_std, std_conversations, config, "std"),
    ]
    print(format_rows(rows_std, title="Standard Benchmark (data/conversations.json)"))

    # ---- Long-Context Stress Benchmark ----
    print("[2/2] Running Long-Context Stress Benchmark...")
    baseline_stress = BaselineAgent(config=config)
    advanced_stress = AdvancedAgent(config=config)

    rows_stress = [
        run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config, "stress"),
        run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config, "stress"),
    ]
    print(format_rows(rows_stress, title="Long-Context Stress Benchmark (data/advanced_long_context.json)"))

    # ---- Phân tích kết quả ----
    _print_analysis(rows_std, rows_stress)


def _print_analysis(rows_std: list[BenchmarkRow], rows_stress: list[BenchmarkRow]) -> None:
    print("\n" + "=" * 60)
    print("  PHÂN TÍCH KẾT QUẢ")
    print("=" * 60)

    def get(rows, name, field):
        for r in rows:
            if r.agent_name == name:
                return getattr(r, field)
        return 0

    # Standard
    b_recall_std = get(rows_std, "Baseline", "recall_score")
    a_recall_std = get(rows_std, "Advanced", "recall_score")
    b_prompt_std = get(rows_std, "Baseline", "prompt_tokens_processed")
    a_prompt_std = get(rows_std, "Advanced", "prompt_tokens_processed")

    print(f"""
[Standard Benchmark]
- Advanced recall: {a_recall_std:.3f} vs Baseline recall: {b_recall_std:.3f}
  → Advanced nhớ facts qua session nhờ User.md, Baseline quên khi đổi thread.

- Advanced prompt tokens: {a_prompt_std} vs Baseline: {b_prompt_std}
  → Ở hội thoại ngắn, Advanced có thể tốn hơn vì mang User.md vào mỗi lượt.

[Long-Context Stress Benchmark]""")

    b_prompt_stress = get(rows_stress, "Baseline", "prompt_tokens_processed")
    a_prompt_stress = get(rows_stress, "Advanced", "prompt_tokens_processed")
    a_compact = get(rows_stress, "Advanced", "compactions")
    b_compact = get(rows_stress, "Baseline", "compactions")

    print(f"""- Advanced prompt tokens: {a_prompt_stress} vs Baseline: {b_prompt_stress}
  → Ở hội thoại rất dài, compact memory giảm prompt tokens của Advanced.
  → Compact chủ yếu tối ưu 'prompt tokens processed', không phải 'agent tokens'.

- Compactions: Advanced={a_compact}, Baseline={b_compact}
  → Baseline không compact, prompt tăng tuyến tính theo số lượt.

[Rủi ro]
- Memory file (User.md) tăng trưởng theo thời gian -> cần giới hạn kích thước.
- Lưu sai fact khi người dùng đặt câu hỏi thay vì cung cấp thông tin.
- Compact có thể làm mất chi tiết nếu summary quá ngắn.
""")


if __name__ == "__main__":
    main()
