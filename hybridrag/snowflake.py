"""64 位雪花编号。

编号在切片第一次入库时生成，之后同时写入 Elasticsearch 和 FAISS。
进程重启不重编已有文档：旧编号已经落在两边的存储里，新编号由时间推出，不会从 1 重新计数。
"""

from __future__ import annotations

import threading
import time

# 41 位毫秒时间、10 位机器、12 位同一毫秒内的序号。最高位留 0，结果始终是正的 int64。
_EPOCH_MS = 1_704_067_200_000  # 2024-01-01 UTC
_WORKER_BITS = 10
_SEQUENCE_BITS = 12
_WORKER_MAX = (1 << _WORKER_BITS) - 1
_SEQUENCE_MASK = (1 << _SEQUENCE_BITS) - 1


class Snowflake:
    def __init__(self, worker_id: int = 1) -> None:
        if not 0 <= worker_id <= _WORKER_MAX:
            raise ValueError(f"worker_id 必须在 0 到 {_WORKER_MAX} 之间")
        self.worker_id = worker_id
        self._sequence = 0
        self._last_ms = -1
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            now = self._now_ms()
            if now < self._last_ms:
                now = self._wait_until(self._last_ms)
            if now == self._last_ms:
                self._sequence = (self._sequence + 1) & _SEQUENCE_MASK
                if self._sequence == 0:
                    now = self._wait_until(self._last_ms + 1)
            else:
                self._sequence = 0
            self._last_ms = now
            elapsed = now - _EPOCH_MS
            if elapsed < 0:
                raise RuntimeError("当前时间早于雪花编号的起始时间")
            return (elapsed << (_WORKER_BITS + _SEQUENCE_BITS)) | (
                self.worker_id << _SEQUENCE_BITS
            ) | self._sequence

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _wait_until(self, target_ms: int) -> int:
        now = self._now_ms()
        while now < target_ms:
            time.sleep((target_ms - now) / 1000)
            now = self._now_ms()
        return now
