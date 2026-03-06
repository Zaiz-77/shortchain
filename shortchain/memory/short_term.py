"""
短期记忆：维护当前会话内的消息历史。

特性：
- 消息按时间顺序存储在内存列表中
- 支持 max_messages 滑动窗口截断（保留 system 消息）
- 可序列化为 OpenAI API 所需的 dict 列表
"""

from __future__ import annotations

from shortchain.core.message import Message, Role


class ShortTermMemory:
    """
    会话级短期记忆。

    Parameters
    ----------
    max_messages:
        最多保留的消息条数（system 消息不计入，不受截断影响）。
        为 None 时不截断。
    """

    def __init__(self, max_messages: int | None = 50) -> None:
        self.max_messages = max_messages
        self._history: list[Message] = []

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def add(self, message: Message) -> None:
        """追加一条消息，并在必要时触发截断。"""
        self._history.append(message)
        self._truncate()

    def add_many(self, messages: list[Message]) -> None:
        """批量追加消息。"""
        self._history.extend(messages)
        self._truncate()

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def get_history(self) -> list[Message]:
        """返回完整消息列表的副本。"""
        return list(self._history)

    def to_openai_messages(self) -> list[dict]:
        """转换为 OpenAI API 接受的消息字典列表。"""
        return [m.to_openai_dict() for m in self._history]

    # ------------------------------------------------------------------ #
    # 管理
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """清空历史（保留 system 消息）。"""
        self._history = [m for m in self._history if m.role == Role.SYSTEM]

    def clear_all(self) -> None:
        """清空全部历史，包括 system 消息。"""
        self._history = []

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(messages={len(self._history)}, max={self.max_messages})"
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _truncate(self) -> None:
        """保留 system 消息，对其余消息应用滑动窗口截断。"""
        if self.max_messages is None:
            return

        system_msgs = [m for m in self._history if m.role == Role.SYSTEM]
        non_system = [m for m in self._history if m.role != Role.SYSTEM]

        if len(non_system) > self.max_messages:
            non_system = non_system[-self.max_messages :]

        self._history = system_msgs + non_system
