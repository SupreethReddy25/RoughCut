"""ASS generation, SRT, filter graph and audio helpers (no ffmpeg needed)."""
from __future__ import annotations

import re
from itertools import pairwise

import pytest

from app.render import ass as ass_mod
from app.render import audio
from app.render.graph import SourceInfo, build_filter_complex, crop_rect, fpath
from app.render.srt import build_srt
from app.schemas import EDL, CaptionLine, Captions, CaptionWord, Clip, OutputSpec, Point, TextOverlay


def mk_edl(style="bold_pop", position="lower_third", clips=None, texts=None, speed=1.0, dur=6.0):
    words1 = [("Three", 0.0, 0.3, False), ("mistakes", 0.3, 0.8, True), ("beginners", 0.8, 1.2, False)]
    words2 = [("make", 2.0, 2.3, False), ("when", 2.3, 2.6, False)]

    def line(ws):
        return CaptionLine(t0=ws[0][1], t1=ws[-1][2], words=[CaptionWord(t0=a, t1=b, text=t, emphasis=e)
                                                              for t, a, b, e in ws])

    return EDL(video_id="v", output=OutputSpec(speed=speed),
               clips=clips or [Clip(src_start=0.0, src_end=3.0)],
               captions=Captions(style_id=style, position=position, lines=[line(words1), line(words2)]),
               texts=texts or [], duration=dur)


def dialogues(text):
    return [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]


# ---------------------------------------------------------------- ASS
def test_ass_structure_and_bold_pop():
    out = ass_mod.build_ass(mk_edl())
    assert "PlayResX: 1080" in out and "PlayResY: 1920" in out
    st = next(ln for ln in out.splitlines() if ln.startswith("Style: Cap,"))
    assert "Montserrat ExtraBold,78," in st  # px at 1080x1920
    d = dialogues(out)
    assert len(d) == 5  # one event per word
    assert all("\\pos(540,1306)" in x for x in d)  # lower_third = 68% of 1920
    assert "THREE" in d[0] and "MISTAKES" in d[1]  # bold_pop is uppercase
    assert "\\fscx110\\fscy110" in d[1]  # active word pops
    assert d[1].count("\\fscx110") == 1
    assert "\\c&H0AD6FF&" in d[0] and "\\c&H0AD6FF&" in d[1]  # yellow emphasis (BGR) on 'mistakes'


def test_ass_events_contiguous_within_line_and_lines_do_not_overlap():
    d = dialogues(ass_mod.build_ass(mk_edl()))

    def t(s):
        h, m, rest = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    rows = [x.split(",")[1:3] for x in d]
    times = [(t(a), t(b)) for a, b in rows]
    for (_, a1), (b0, _) in pairwise(times[:3]):
        assert abs(a1 - b0) < 0.011  # words of a line hand over without a gap
    assert times[2][1] <= times[3][0] + 1e-6  # line 1 ends before line 2 starts


def test_ass_positions_and_scaling():
    assert "\\pos(540,960)" in dialogues(ass_mod.build_ass(mk_edl(position="center")))[0]
    assert "\\pos(540,269)" in dialogues(ass_mod.build_ass(mk_edl(position="top")))[0]
    edl = mk_edl()
    edl.output = OutputSpec(width=540, height=960)  # half-size output: sizes scale with height
    out = ass_mod.build_ass(edl)
    assert "PlayResX: 540" in out and "Montserrat ExtraBold,39," in out


def test_ass_other_styles():
    karaoke = ass_mod.build_ass(mk_edl("karaoke"))
    assert len(dialogues(karaoke)) == 2 and "\\kf" in karaoke
    clean = ass_mod.build_ass(mk_edl("clean"))
    assert "CapBox" in clean and "Inter SemiBold" in clean and "\\1a&HFF&" in clean
    minimal = ass_mod.build_ass(mk_edl("minimal"))
    assert "three" in minimal and "mistakes" in minimal and "THREE" not in minimal  # lowercase style


def test_ass_long_line_wraps_to_two_lines_and_disabled_has_no_events():
    edl = mk_edl()
    edl.captions.lines = [CaptionLine(t0=0, t1=1.4, words=[
        CaptionWord(t0=i * 0.3, t1=i * 0.3 + 0.3, text=t) for i, t in
        enumerate(["Entrepreneurship", "revolutionizes", "communication", "everywhere"])])]
    assert all("\\N" in x for x in dialogues(ass_mod.build_ass(edl)))
    short = mk_edl()
    assert all("\\N" not in x for x in dialogues(ass_mod.build_ass(short))[-2:])  # "make when" fits one line
    edl.captions.enabled = False
    assert dialogues(ass_mod.build_ass(edl)) == []


def test_ass_text_overlays_and_escaping():
    texts = [TextOverlay(id="t1", text="3 editing {mistakes}", t0=0.0, t1=3.0, position="top", style_id="title"),
             TextOverlay(id="t2", text="note", t0=1.0, t1=2.0, position="custom", y=0.9, style_id="label")]
    out = ass_mod.build_ass(mk_edl(texts=texts))
    d = [x for x in dialogues(out) if "Txt_" in x]
    assert len(d) == 2 and d[0].startswith("Dialogue: 1,")  # overlay layer above captions
    assert "\\pos(540,250)" in d[0] and "3 EDITING" in d[0] and "(MISTAKES)" in d[0] and "{mistakes" not in d[0].lower()  # upper; braces neutralised
    assert "\\pos(540,1728)" in d[1]
    assert "Style: Txt_title," in out and "Style: Txt_label," in out


def test_ass_color_and_time_formats():
    assert ass_mod.ass_color("#FFD60A") == "&H00" + "0AD6FF"
    assert ass_mod.ass_color("#000000", 0.55) == "&H73000000"
    assert ass_mod._fmt_time(3661.236) == "1:01:01.24"


def test_style_fonts_exist_and_metrics_work():
    cap, txt = ass_mod.load_styles()
    for st in [*cap["styles"].values(), *txt["styles"].values()]:
        assert (ass_mod.FONTS_DIR / st["font_file"]).exists()
    w = ass_mod.text_width("Montserrat-ExtraBold.ttf", 78, "HELLO")
    assert 150 < w < 450 and ass_mod.text_width("Montserrat-ExtraBold.ttf", 78, "HELLOHELLO") == pytest.approx(2 * w)


# ---------------------------------------------------------------- SRT
def test_srt():
    srt = build_srt(mk_edl())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,200\nThree mistakes beginners\n")
    assert "2\n00:00:02,000 --> 00:00:02,600\nmake when" in srt


# ---------------------------------------------------------------- graph
def test_crop_rect_landscape_zoom_and_clamp():
    src = SourceInfo(1920, 1080)
    full = crop_rect(Clip(src_start=0, src_end=1), src, 1080, 1920)
    assert full == (656, 0, 608, 1080)  # 9:16 window, centred
    z = crop_rect(Clip(src_start=0, src_end=1, zoom=1.15), src, 1080, 1920)
    assert z[2] == pytest.approx(608 / 1.15, abs=2) and z[3] == pytest.approx(1080 / 1.15, abs=2)
    left = crop_rect(Clip(src_start=0, src_end=1, crop_center=Point(x=0.0, y=0.0)), src, 1080, 1920)
    assert left[0] == 0 and left[1] == 0
    right = crop_rect(Clip(src_start=0, src_end=1, crop_center=Point(x=1.0, y=1.0)), src, 1080, 1920)
    assert right[0] + right[2] == 1920 and right[1] + right[3] == 1080
    face = crop_rect(Clip(src_start=0, src_end=1, crop_center=Point(x=0.3, y=0.5)), src, 1080, 1920)
    assert face[0] + face[2] // 2 == pytest.approx(0.3 * 1920, abs=2)


def test_crop_rect_portrait_sources():
    nine16 = crop_rect(Clip(src_start=0, src_end=1), SourceInfo(1080, 1920), 1080, 1920)
    assert nine16 == (0, 0, 1080, 1920)  # no crop at zoom 1
    z = crop_rect(Clip(src_start=0, src_end=1, zoom=1.3, crop_center=Point(x=0.4, y=0.3)), SourceInfo(1080, 1920),
                  1080, 1920)
    assert z[2] == pytest.approx(1080 / 1.3, abs=2) and 0 <= z[0] <= 1080 - z[2]  # zoom origin = face centre
    tall = crop_rect(Clip(src_start=0, src_end=1), SourceInfo(1080, 2400), 1080, 1920)
    assert tall[2] == 1080 and tall[3] == 1920


def test_filter_graph_contents():
    edl = mk_edl(clips=[Clip(src_start=1.0, src_end=2.0), Clip(src_start=3.0, src_end=4.5, zoom=1.15)], speed=1.25)
    g, v, a = build_filter_complex(edl, SourceInfo(1920, 1080), ass_path="C:\\tmp\\x.ass", fonts_dir="C:\\f",
                                   audio_chain=["highpass=f=80", "loudnorm=I=-14"])
    assert v == "vo" and a == "ao"
    assert "trim=start_frame=30:end_frame=60" in g and "trim=start_frame=90:end_frame=135" in g
    assert "atrim=start=1.0000:end=2.0000" in g and g.count("afade=t=in:st=0:d=0.0080") == 2
    assert "afade=t=out:st=0.9920:d=0.0080" in g
    assert "scale=1080:1920:flags=lanczos" in g and "concat=n=2:v=1:a=1[vc][ac]" in g
    assert "setpts=PTS/1.250000" in g and "atempo=1.250000" in g
    assert "ass=filename='C\\:/tmp/x.ass':fontsdir='C\\:/f'" in g
    assert g.rstrip().endswith("[ao]") and "highpass=f=80,loudnorm=I=-14" in g
    assert "[0:v]" not in build_filter_complex(edl, SourceInfo(1920, 1080), video=False)[0]
    assert re.search(r"concat=n=2:v=0:a=1\[ac\]", build_filter_complex(edl, SourceInfo(1920, 1080), video=False)[0])
    assert fpath("C:\\a b\\c.ass") == "'C\\:/a b/c.ass'"
    with pytest.raises(ValueError):
        build_filter_complex(EDL(video_id="v"), SourceInfo(1920, 1080))


# ---------------------------------------------------------------- audio helpers
def test_audio_chain_and_loudnorm_helpers():
    assert audio.enhance_chain(True) == ["highpass=f=80", "afftdn=nf=-25"]
    assert audio.enhance_chain(False) == ["highpass=f=80"]
    assert audio.should_denoise(None) and audio.should_denoise(-40) and not audio.should_denoise(-80)
    stderr = 'noise\n{\n\t"input_i" : "-20.15",\n\t"input_tp" : "-3.00",\n\t"input_lra" : "1.60",\n' \
             '\t"input_thresh" : "-30.5",\n\t"output_i" : "-14.1",\n\t"target_offset" : "0.10"\n}\n'
    m = audio.parse_loudnorm_json(stderr)
    p2 = audio.loudnorm_pass2(m)
    assert "I=-14.0:TP=-1.5:LRA=11.0" in p2 and "measured_I=-20.15" in p2 and "offset=0.10" in p2 and "linear=true" in p2
    assert "print_format=json" in audio.loudnorm_pass1()
    with pytest.raises(RuntimeError):
        audio.parse_loudnorm_json("nothing here")


def test_render_rejects_empty_and_overlong_edls(tmp_path):
    import subprocess

    from app import config
    from app.render.run import render

    src = tmp_path / "s.mp4"  # 2 s, 30 fps test pattern with silent audio
    subprocess.run([config.ffmpeg_path(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=2",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(src)], check=True)
    with pytest.raises(ValueError, match="no clips"):
        render(EDL(video_id="v"), src, tmp_path / "o.mp4", progress=lambda *_: None)
    long_edl = mk_edl(clips=[Clip(src_start=0.0, src_end=3.0)])
    with pytest.raises(ValueError, match="only 2"):
        render(long_edl, src, tmp_path / "o.mp4", progress=lambda *_: None)
