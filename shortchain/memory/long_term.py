"""
长期记忆：以本地 JSON 文件持久化存储跨会话的信息。

特性：
- 每个 Agent 独享一个 JSON 文件（按 agent_id 区分）
- 支持两类数据：
    facts  —— 键值对事实（如用户偏好、背景信息）
    summaries —— 按时间排列的对话摘要文本
- 提供搜索接口（简单关键字匹配）
- 文件路径由 config.get_memory_dir() 决定，默认 .shortchain_memory/
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from shortchain.config import get_memory_dir


class LongTermMemory:
    """
    长期记忆，数据持久化到本地 JSON 文件。

    Parameters
    ----------
    agent_id:
        Agent 的唯一标识，用于区分不同 Agent 的记忆文件。
    memory_dir:
        存储目录，为 None 时使用配置文件中的默认目录。
    """

    def __init__(self, agent_id: str, memory_dir: Path | None = None) -> None:
        self.agent_id = agent_id
        self._dir = memory_dir or get_memory_dir()
        self._path = self._dir / f"{agent_id}.json"
        self._data = self._load()

    # ------------------------------------------------------------------ #
    # 事实（键值对）
    # ------------------------------------------------------------------ #

    def set_fact(self, key: str, value: Any) -> None:
        """存储一条事实。"""
        self._data["facts"][key] = value
        self._save()

    def get_fact(self, key: str, default: Any = None) -> Any:
        """读取一条事实。"""
        return self._data["facts"].get(key, default)

    def delete_fact(self, key: str) -> None:
        """删除一条事实。"""
        self._data["facts"].pop(key, None)
        self._save()

    def all_facts(self) -> dict[str, Any]:
        """返回所有事实的副本。"""
        return dict(self._data["facts"])

    # ------------------------------------------------------------------ #
    # 摘要（时间序列文本）
    # ------------------------------------------------------------------ #

    def add_summary(self, text: str) -> None:
        """追加一条对话摘要，自动附加时间戳。"""
        entry = {"timestamp": time.time(), "text": text}
        self._data["summaries"].append(entry)
        self._save()

    def get_summaries(self, last_n: int | None = None) -> list[dict]:
        """
        获取摘要列表。

        Parameters
        ----------
        last_n:
            只返回最近 n 条，为 None 时返回全部。
        """
        summaries = self._data["summaries"]
        if last_n is not None:
            summaries = summaries[-last_n:]
        return list(summaries)

    def get_summaries_text(self, last_n: int | None = None) -> str:
        """将摘要列表拼接为纯文本，方便注入 system prompt。"""
        entries = self.get_summaries(last_n)
        if not entries:
            return ""
        lines = []
        for e in entries:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["timestamp"]))
            lines.append(f"[{ts}] {e['text']}")
        return "\n".join(lines)

    def clear_summaries(self) -> None:
        """清空所有摘要。"""
        self._data["summaries"] = []
        self._save()

    # ------------------------------------------------------------------ #
    # 搜索
    # ------------------------------------------------------------------ #

    def search(self, keyword: str) -> dict[str, Any]:
        """
        简单关键字搜索，返回匹配的事实和摘要。

        Parameters
        ----------
        keyword:
            搜索关键字，大小写不敏感。
        """
        kw = keyword.lower()
        matched_facts = {
            k: v
            for k, v in self._data["facts"].items()
            if kw in k.lower() or kw in str(v).lower()
        }
        matched_summaries = [
            e for e in self._data["summaries"] if kw in e["text"].lower()
        ]
        return {"facts": matched_facts, "summaries": matched_summaries}

    # ------------------------------------------------------------------ #
    # 管理
    # ------------------------------------------------------------------ #

    def clear_all(self) -> None:
        """清空所有记忆并删除文件。"""
        self._data = self._empty_store()
        if self._path.exists():
            self._path.unlink()

    def __repr__(self) -> str:
        facts_n = len(self._data["facts"])
        summaries_n = len(self._data["summaries"])
        return (
            f"LongTermMemory(agent_id={self.agent_id!r}, "
            f"facts={facts_n}, summaries={summaries_n}, "
            f"path={self._path})"
        )

    # ------------------------------------------------------------------ #
    # 内部 I/O
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_store() -> dict:
        return {"facts": {}, "summaries": []}

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as f:
                    data = json.load(f)
                # 兼容旧版文件缺少字段的情况
                data.setdefault("facts", {})
                data.setdefault("summaries", [])
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return self._empty_store()

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
