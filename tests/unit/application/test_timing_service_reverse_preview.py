"""TimingService 倒放预览（[@reverse] 段打轴）测试。

覆盖：enter/exit 的 load 顺序与临时文件清理；位置映射（region_start + local）；
seek 映射；打轴 key 写入映射后的时间戳。
"""

from typing import Callable, Optional
from pathlib import Path

import numpy as np
import pytest

from strange_uta_game.backend.application.timing_service import TimingService
from strange_uta_game.backend.domain import Project, Sentence, Singer
from strange_uta_game.backend.infrastructure.audio.base import (
    AudioInfo,
    IAudioEngine,
    PlaybackState,
)


class FakeAudioEngine(IAudioEngine):
    """轻量音频引擎桩：记录 load 路径、位置、速度，提供反转用的采样。"""

    def __init__(self):
        self._position_ms = 0
        self._playing = False
        self._speed = 1.0
        self.loaded_paths: list[str] = []
        self._orig_path = "D:/orig.wav"
        self._duration_ms = 60000
        self._info = AudioInfo(
            file_path=self._orig_path,
            duration_ms=60000,
            sample_rate=1000,
            channels=2,
        )

    def load(self, file_path: str, progress_cb=None) -> None:
        self.loaded_paths.append(file_path)
        # 模拟真实引擎：加载临时反转文件后，引擎时长变成该文件时长
        self._duration_ms = 60000 if file_path == self._orig_path else 5000

    def play(self) -> None:
        self._playing = True

    def pause(self) -> None:
        self._playing = False

    def stop(self) -> None:
        self._playing = False
        self._position_ms = 0

    def get_position_ms(self) -> int:
        return self._position_ms

    def set_position_ms(self, position_ms: int) -> None:
        self._position_ms = position_ms

    def get_duration_ms(self) -> int:
        return self._duration_ms

    def get_playback_state(self) -> PlaybackState:
        return PlaybackState.PLAYING if self._playing else PlaybackState.STOPPED

    def is_playing(self) -> bool:
        return self._playing

    def set_speed(self, speed: float) -> None:
        self._speed = speed

    def get_speed(self) -> float:
        return self._speed

    def set_volume(self, volume: float) -> None:
        pass

    def get_volume(self) -> float:
        return 1.0

    def set_position_callback(self, callback: Callable[[int], None]) -> None:
        pass

    def clear_position_callback(self) -> None:
        pass

    def get_audio_info(self) -> Optional[AudioInfo]:
        return self._info

    def get_original_samples(self):
        n = 60000  # 60s @1000Hz
        t = np.arange(n, dtype=np.float32)
        return np.stack([t, t], axis=1)

    def release(self) -> None:
        pass


def _make_service(engine: FakeAudioEngine) -> TimingService:
    service = TimingService(audio_engine=engine)
    project = Project()
    singer = project.get_default_singer()
    service.set_project(project)
    return service


class TestTimingServiceReversePreview:
    def test_enter_loads_reversed_file_and_maps_position(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )

        ok = service.enter_reverse_preview(10_000, 15_000)
        assert ok is True
        assert service.is_reverse_preview_active()
        # 加载了临时反转文件
        assert len(engine.loaded_paths) == 1
        assert "reverse_preview_10000_15000.wav" in engine.loaded_paths[-1]
        assert Path(engine.loaded_paths[-1]).exists()

        # 本地位置 0 → 映射为 region_start
        engine.set_position_ms(0)
        assert service.get_position_ms() == 10_000
        # 本地 3000ms → region_start + 3000
        engine.set_position_ms(3_000)
        assert service.get_position_ms() == 13_000
        # duration 与位置同一坐标系：即使引擎已被替换为 5s 临时文件，
        # 对外仍报告完整原始时长（否则 UI 位置超过时长、播放头锁死在末尾）
        assert engine.get_duration_ms() == 5_000
        assert service.get_duration_ms() == 60_000

    def test_play_restarts_region_when_at_end(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        service.enter_reverse_preview(10_000, 15_000)
        # 本地位置 = 区间末尾：再次播放应从区间起点重播
        engine.set_position_ms(5_000)
        service.play()
        assert engine.get_position_ms() == 0
        assert engine.is_playing()

    def test_play_does_not_rewind_inside_region(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        service.enter_reverse_preview(10_000, 15_000)
        engine.set_position_ms(2_000)
        service.play()
        assert engine.get_position_ms() == 2_000

    def test_seek_maps_to_local(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        service.enter_reverse_preview(10_000, 15_000)
        service.seek(12_000)
        assert engine.get_position_ms() == 2_000

    def test_display_axis_mirrors_position(self, tmp_path, monkeypatch):
        """显示轴镜像：播放头在区间内从右向左走（反转音频视觉）。"""
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        # 非预览：显示轴 == 写入轴
        engine.set_position_ms(3_000)
        assert service.map_to_display(3_000) == 3_000
        assert service.get_display_position_ms() == 3_000

        service.enter_reverse_preview(10_000, 15_000)
        # local 0（反转起点，播放原区间末尾）→ 显示区间右端
        engine.set_position_ms(0)
        assert service.get_display_position_ms() == 15_000
        # local 3000 → 显示 12000；local 5000（区间末尾）→ 显示区间左端
        engine.set_position_ms(3_000)
        assert service.get_display_position_ms() == 12_000
        engine.set_position_ms(5_000)
        assert service.get_display_position_ms() == 10_000
        # 写入轴不受影响（打轴时间戳仍递增落在原始轴）
        engine.set_position_ms(3_000)
        assert service.get_position_ms() == 13_000

    def test_seek_display_maps_mirrored(self, tmp_path, monkeypatch):
        """显示轴 seek：拖到哪、播放头停在哪、听到的就是该处内容。"""
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        service.enter_reverse_preview(10_000, 15_000)
        # 显示 12000 → 本地 3000（原区间末尾倒推）
        service.seek_display(12_000)
        assert engine.get_position_ms() == 3_000
        assert service.get_display_position_ms() == 12_000
        # 显示区间右端 → 本地 0；区间左端 → 本地 5000
        service.seek_display(15_000)
        assert engine.get_position_ms() == 0
        service.seek_display(10_000)
        assert engine.get_position_ms() == 5_000
        # 拖出区间外钳制到区间边界
        service.seek_display(20_000)
        assert engine.get_position_ms() == 0
        service.seek_display(5_000)
        assert engine.get_position_ms() == 5_000

    def test_seek_display_outside_preview_equals_seek(self):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        service.seek_display(12_000)
        assert engine.get_position_ms() == 12_000

    def test_exit_restores_original_and_cleans_temp(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        service.enter_reverse_preview(10_000, 15_000)
        temp_path = engine.loaded_paths[-1]
        assert Path(temp_path).exists()

        service.exit_reverse_preview()
        assert not service.is_reverse_preview_active()
        # 恢复原文件
        assert engine.loaded_paths[-1] == "D:/orig.wav"
        assert not Path(temp_path).exists()
        # 位置恢复原时间轴
        assert service.get_position_ms() == engine.get_position_ms()
        # 时长恢复为原始完整时长
        assert service.get_duration_ms() == 60_000

    def test_timing_key_writes_mapped_timestamp(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        # 项目需含句子才有可写入的 checkpoint
        project = Project()
        singer = project.get_default_singer()
        sentence = Sentence.from_text("テスト", singer.id)
        project.add_sentence(sentence)
        service.set_project(project)
        service.set_timing_offset(0)

        service.enter_reverse_preview(10_000, 15_000)
        engine.set_position_ms(4_000)  # 本地 4s → 原始 14s

        service.on_timing_key_pressed("F1")
        written = sentence.characters[0].timestamps
        assert written == [14_000]

    def test_enter_fails_without_samples(self, tmp_path, monkeypatch):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        monkeypatch.setattr(engine, "get_original_samples", lambda: None)
        monkeypatch.setattr(
            "strange_uta_game.backend.application.timing_service.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        assert service.enter_reverse_preview(0, 1000) is False
        assert not service.is_reverse_preview_active()

    def test_mark_reverse_range_marks_sentences_in_range(self):
        engine = FakeAudioEngine()
        service = _make_service(engine)
        project = service._project
        singer = project.get_default_singer()
        s1 = Sentence.from_text("テスト", singer.id)
        s1.characters[0].add_timestamp(10_000)
        s2 = Sentence.from_text("倒放", singer.id)
        s2.characters[0].add_timestamp(20_000)
        project.add_sentence(s1)
        project.add_sentence(s2)

        changed = service.mark_reverse_range(15_000, 25_000)
        assert changed == 1
        assert s1.reverse_playback is False
        assert s2.reverse_playback is True

    def test_mark_sentence_reverse_via_command_is_undoable(self):
        from strange_uta_game.backend.application.command_manager import CommandManager

        engine = FakeAudioEngine()
        manager = CommandManager()
        service = TimingService(audio_engine=engine, command_manager=manager)
        project = Project()
        singer = project.get_default_singer()
        s = Sentence.from_text("テスト", singer.id)
        project.add_sentence(s)
        service.set_project(project)

        service.mark_sentence_reverse(s.id, True)
        assert s.reverse_playback is True
        manager.undo()
        assert s.reverse_playback is False
