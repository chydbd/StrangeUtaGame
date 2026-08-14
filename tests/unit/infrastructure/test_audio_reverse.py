"""倒放预览临时 WAV 生成测试。"""

import numpy as np
import pytest
import soundfile as sf

from strange_uta_game.backend.infrastructure.audio.reverse_preview import (
    build_reversed_segment_file,
    cleanup_reversed_segment_file,
)


def _ramp(sr: int = 1000, seconds: float = 2.0) -> np.ndarray:
    """归一化递增斜坡（频率特征便于验证反转，值域 [-1, 1] 适配 16-bit 写出）。"""
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / (sr * seconds)
    return np.stack([t, t], axis=1)  # 双声道


def test_reversed_segment_matches_flipped_pcm(tmp_path):
    sr = 1000
    samples = _ramp(sr)
    path = build_reversed_segment_file(samples, sr, 500, 1500, tmp_path)

    written, written_sr = sf.read(path)
    written_sr = int(written_sr)
    assert written_sr == sr
    # 16-bit PCM 写出后读出即归一化值，直接与归一化期望对比（int16 量化误差内）
    expected = samples[500:1500][::-1]
    np.testing.assert_allclose(
        np.asarray(written, dtype=np.float32), expected, atol=2.0 / 32767.0
    )
    # 时长 = 区域时长（1000ms）
    assert written.shape[0] == 1000


def test_reversed_segment_duration_matches_region(tmp_path):
    sr = 1000
    samples = _ramp(sr)
    path = build_reversed_segment_file(samples, sr, 0, 2000, tmp_path)
    written, written_sr = sf.read(path)
    assert int(written_sr) == sr
    assert written.shape[0] == 2000


def test_cleanup_removes_file(tmp_path):
    sr = 1000
    path = build_reversed_segment_file(_ramp(sr), sr, 0, 1000, tmp_path)
    assert tmp_path.joinpath(path).exists() or __import__("pathlib").Path(path).exists()
    cleanup_reversed_segment_file(path)
    assert not __import__("pathlib").Path(path).exists()


def test_invalid_region_raises(tmp_path):
    sr = 1000
    with pytest.raises(ValueError):
        build_reversed_segment_file(_ramp(sr), sr, 1500, 1000, tmp_path)
