"""倒放预览：把音频段反转写成临时 WAV。

打轴时用户需要「倒放预览」——把某段音频反向播放来听清实际歌词。方案：
切片原 PCM ``[::-1].copy()`` 写临时 WAV，交给音频引擎当作普通文件加载播放。
位置映射（``region_start + engine_local``）由 :class:`TimingService` 负责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def build_reversed_segment_file(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
) -> str:
    """切片 ``[start_ms, end_ms]`` 反转写临时 WAV，返回路径。

    ``samples`` 形状 ``(n_samples, channels)`` float32；负步长切片必须
    ``.copy()``（否则 soundfile 写入会因非连续内存报错）。
    文件名含起止毫秒便于区分与清理。
    """
    start = max(0, int(round(start_ms * sample_rate / 1000)))
    end = min(int(samples.shape[0]), int(round(end_ms * sample_rate / 1000)))
    if end <= start:
        raise ValueError(f"无效倒放区间: {start_ms}-{end_ms}ms")
    seg = samples[start:end][::-1].copy()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"reverse_preview_{start_ms}_{end_ms}.wav"
    sf.write(str(path), seg, sample_rate)
    return str(path)


def cleanup_reversed_segment_file(path: Optional[str]) -> None:
    """删除临时 WAV（忽略不存在/删除失败）。"""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
