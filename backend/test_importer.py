"""Unit tests for the transcript importer (app.importer).

Covers standard SRT / WebVTT block parsing (cue number / id, timing line,
multi-line body, cue settings, header blocks), CJK-aware continuation
joining, and backwards compatibility with the legacy inline-timestamp
plain-text format.

Run with: python3 test_importer.py  (or via pytest).
"""
from __future__ import annotations

import pytest

from app.importer import parse_document

SRT = """1
00:00:01,000 --> 00:00:03,500
访谈者: 感谢您接受访谈，
能先介绍一下自己吗？

2
00:00:04.000 --> 00:00:08,000
受访者: 好的，我叫李明。

3
00:00:09,000 --> 00:00:12,500
Interviewer: How are you
doing today?
"""

# same content in WebVTT: header, NOTE, non-numeric cue id, cue settings
VTT = """WEBVTT

NOTE 导出说明
本块应被忽略

cue-1
00:00:01.000 --> 00:00:03.500
访谈者: 感谢您接受访谈，
能先介绍一下自己吗？

00:00:04.000 --> 00:00:08.000 align:start position:20%
受访者: 好的，我叫李明。

cue-3
00:00:09.000 --> 00:00:12.500 line:0
Interviewer: How are you
doing today?
"""


def _segs(raw):
    return parse_document(raw)["segments"]


def _check_cues(segments):
    # sequence numbers / cue ids do not become segments, three cues total
    assert len(segments) == 3
    assert [round(s["start"], 3) for s in segments] == [1.0, 4.0, 9.0]
    assert [round(s["end"], 3) for s in segments] == [3.5, 8.0, 12.5]
    assert [s["speaker"] for s in segments] == ["访谈者", "受访者", "Interviewer"]
    # continuation lines are merged into the same segment; CJK joins without
    # a separator, latin lines keep a space
    assert segments[0]["text"] == "感谢您接受访谈，能先介绍一下自己吗？"
    assert segments[1]["text"] == "好的，我叫李明。"
    assert segments[2]["text"] == "How are you doing today?"
    assert all(s["id"].startswith("seg-") for s in segments)


def test_parse_standard_srt():
    _check_cues(_segs(SRT))


def test_parse_webvtt_header_note_cue_id_and_settings():
    _check_cues(_segs(VTT))


def test_srt_with_crlf_and_leading_bom_like_whitespace():
    raw = SRT.replace("\n", "\r\n")
    segments = _segs(raw)
    assert [s["text"] for s in segments] == [
        "感谢您接受访谈，能先介绍一下自己吗？",
        "好的，我叫李明。",
        "How are you doing today?",
    ]


def test_cue_without_speaker_prefix():
    raw = """WEBVTT

1
00:00:01,000 --> 00:00:02,000
这是没有说话人前缀的正文
第二行续接
"""
    seg = _segs(raw)[0]
    assert seg["speaker"] == ""
    assert seg["text"] == "这是没有说话人前缀的正文第二行续接"
    assert (seg["start"], seg["end"]) == (1.0, 2.0)


def test_timing_only_cue_keeps_empty_segment():
    raw = "1\n00:00:01,000 --> 00:00:02,000\n"
    seg = _segs(raw)[0]
    assert seg["text"] == ""
    assert (seg["start"], seg["end"]) == (1.0, 2.0)


def test_legacy_inline_timestamps_still_import():
    raw = (
        "[00:00:00] 访谈者: 感谢您今天接受访谈，能先简单介绍一下自己吗？\n"
        "[00:00:04] 受访者: 好的。我叫李明。\n"
        "[00:00:34] --> [00:00:40] 受访者: 后来 2003 年改制\n"
        "[01:06.200] 访谈者: 能具体说说吗？\n"
    )
    segments = _segs(raw)
    assert len(segments) == 4
    assert (segments[0]["start"], segments[0]["speaker"],
            segments[0]["text"]) == (0.0, "访谈者", "感谢您今天接受访谈，能先简单介绍一下自己吗？")
    # missing end -> auto-extended to next segment start
    assert segments[0]["end"] == 4.0
    # explicit [end] inline is locked
    assert (segments[2]["start"], segments[2]["end"]) == (34.0, 40.0)
    # mm:ss.mmm
    assert segments[3]["start"] == 66.2
    assert segments[3]["end"] == 67.2


def test_legacy_inline_mixed_with_srt_block():
    raw = (
        "[00:00:00] 访谈者: 先开场\n"
        "\n"
        "1\n"
        "00:00:05,000 --> 00:00:07,000\n"
        "受访者: 正式回答\n"
        "\n"
        "[00:00:09] 访谈者: 收尾\n"
    )
    segments = _segs(raw)
    assert len(segments) == 3
    assert segments[0]["end"] == 5.0  # auto-extended through the cue start
    assert (segments[1]["start"], segments[1]["end"],
            segments[1]["text"]) == (5.0, 7.0, "正式回答")
    assert segments[2]["start"] == 9.0


def test_empty_and_header_only_inputs_fail():
    with pytest.raises(ValueError):
        parse_document("   \n  ")
    with pytest.raises(ValueError):
        parse_document("WEBVTT\n\nNOTE 只有头和说明\n")


def test_text_importer_returns_empty_annotations():
    doc = parse_document(SRT)
    assert doc["annotations"] == []


def test_json_paths_unaffected():
    doc = parse_document([
        {"start": 1, "end": 2, "speaker": "甲", "text": "json 片段"},
    ])
    assert doc["segments"][0]["text"] == "json 片段"
