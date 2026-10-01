"""ASS subtitle generation (PLAN 8.2, 8.7): captions + text overlays from the EDL and the style JSON files.

The same JSON (render/caption_styles.json, text_styles.json) drives the browser preview. Sizes are px at
1080x1920 and scale with the output height. Fonts are the static OFL files in assets/fonts (libass gets them
through `fontsdir`); line wrapping is decided here with real glyph metrics so every per-word event of a line
wraps identically (no jitter when the active word scales up).
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from fontTools.ttLib import TTFont

from app import config
from app.schemas import EDL, CaptionLine

RENDER_DIR = Path(__file__).resolve().parent
FONTS_DIR = config.ASSETS_DIR / "fonts"
REF_H = 1920  # style sizes are defined at 1080x1920
HOLD = 0.12  # a caption line lingers this long after its last word (capped by the next line)


@cache
def load_styles() -> tuple[dict, dict]:
    cap = json.loads((RENDER_DIR / "caption_styles.json").read_text(encoding="utf-8"))
    txt = json.loads((RENDER_DIR / "text_styles.json").read_text(encoding="utf-8"))
    return cap, txt


@cache
def _font(font_file: str):
    f = TTFont(FONTS_DIR / font_file)
    return f.getBestCmap(), f["hmtx"], f["head"].unitsPerEm


def text_width(font_file: str, size: float, text: str) -> float:
    """Advance width in px (no kerning) of `text` set in the given static font."""
    cmap, hmtx, upem = _font(font_file)
    w = 0
    for ch in text:
        g = cmap.get(ord(ch))
        w += hmtx[g][0] if g in hmtx.metrics else upem * 0.5
    return w * size / upem


def ass_color(hex_rgb: str, opacity: float = 1.0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    a = round((1.0 - opacity) * 255)
    return f"&H{a:02X}{b}{g}{r}".upper()


def _tag_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _fmt_time(t: float) -> str:
    cs = max(0, round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _case(text: str, mode: str) -> str:
    return text.upper() if mode == "upper" else text.lower() if mode == "lower" else text


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")  # no override-tag injection


def wrap_index(style: dict, words: list[str], scale: float = 1.0, size_px: float | None = None) -> int | None:
    """Index at which to break a line of `words` into two balanced lines, or None if it fits on one line."""
    size = (size_px or style["size"]) * scale
    full = text_width(style["font_file"], size, " ".join(words))
    box = style.get("box")
    max_w = style.get("max_width", 900) - (2 * box["padding"] if box else 0)
    if full <= max_w or len(words) < 2:
        return None
    best, best_w = 1, float("inf")
    for k in range(1, len(words)):
        w = max(text_width(style["font_file"], size, " ".join(words[:k])),
                text_width(style["font_file"], size, " ".join(words[k:])))
        if w < best_w:
            best, best_w = k, w
    return best


def _style_line(name: str, st: dict, k: float, *, karaoke: bool = False) -> str:
    box = st.get("box")
    primary = ass_color(st.get("fill_color", st["color"]) if karaoke else st["color"])
    secondary = ass_color(st["color"])
    if box:
        border_style, outline = 3, round(box["padding"] * k)
        outline_c, back_c = ass_color(box["color"], box["opacity"]), ass_color("#000000", 0.0)
    else:
        border_style, outline = 1, round(st.get("outline", 0) * k)
        outline_c, back_c = ass_color(st.get("outline_color", "#000000")), ass_color("#000000", 0.55)
    shadow = round(st.get("shadow", 0) * k)
    return (f"Style: {name},{st['font']},{round(st['size'] * k)},{primary},{secondary},{outline_c},{back_c},"
            f"0,0,0,0,100,100,0,0,{border_style},{outline},{shadow},5,60,60,0,1")


def _wrapped_text(st: dict, parts: list[str], raw_words: list[str], size_px: float, scale: float = 1.0) -> str:
    """Join per-word ASS fragments with spaces, inserting \\N at the balanced break."""
    brk = wrap_index(st, raw_words, scale, size_px)
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append("\\N" if brk == i else " ")
        out.append(p)
    return "".join(out)


def _caption_events(edl: EDL, st: dict, positions: dict, k: float) -> list[tuple[float, str]]:
    cap = edl.captions
    W, H = edl.output.width, edl.output.height
    pos = f"{{\\an5\\pos({W // 2},{round(positions[cap.position] * H)})}}"
    size_px = st["size"] * k
    events: list[tuple[float, str]] = []
    emph = _tag_color(st.get("emphasis_color", st["color"]))
    lines: list[CaptionLine] = cap.lines
    for li, line in enumerate(lines):
        nxt = lines[li + 1].t0 if li + 1 < len(lines) else edl.duration
        end = max(line.t1, min(line.t1 + HOLD, nxt, edl.duration))
        words = [_case(_esc(w.text), st.get("case", "none")) for w in line.words]
        if not words:
            continue
        mode = st["mode"]
        if mode == "pop":
            scale = st.get("active_scale", 1.1)
            for j, w in enumerate(line.words):
                s = line.t0 if j == 0 else w.t0
                e = line.words[j + 1].t0 if j + 1 < len(line.words) else end
                parts = []
                for i, (txt, cw) in enumerate(zip(words, line.words)):
                    tags = ""
                    if cw.emphasis:
                        tags += f"\\c{emph}"
                    if i == j:
                        tags += f"\\fscx{round(scale * 100)}\\fscy{round(scale * 100)}"
                    parts.append(f"{{{tags}}}{txt}{{\\r}}" if tags else txt)
                events.append((s, f"0,{_fmt_time(s)},{_fmt_time(e)},Cap,,0,0,0,,{pos}"
                                  + _wrapped_text(st, parts, words, size_px, scale)))
        elif mode == "karaoke":
            parts = []
            for j, (txt, cw) in enumerate(zip(words, line.words)):
                nxt_t = line.words[j + 1].t0 if j + 1 < len(line.words) else end
                parts.append(f"{{\\kf{max(1, round((nxt_t - cw.t0) * 100))}}}{txt}")
            events.append((line.t0, f"0,{_fmt_time(line.t0)},{_fmt_time(end)},Cap,,0,0,0,,{pos}"
                                    + _wrapped_text(st, parts, words, size_px)))
        else:  # "line": the whole line at once, emphasised words coloured
            parts = [f"{{\\c{emph}}}{t}{{\\r}}" if cw.emphasis else t for t, cw in zip(words, line.words)]
            if st.get("box"):
                # the box is its own layer (invisible text on a BorderStyle-3 style); drawing it per colour run
                # would overlap the translucent boxes and leave dark seams
                events.append((line.t0, f"0,{_fmt_time(line.t0)},{_fmt_time(end)},CapBox,,0,0,0,,{pos}{{\\1a&HFF&}}"
                                        + _wrapped_text(st, words, words, size_px)))
            events.append((line.t0, f"1,{_fmt_time(line.t0)},{_fmt_time(end)},Cap,,0,0,0,,{pos}"
                                    + _wrapped_text(st, parts, words, size_px)))
    return events


def _text_events(edl: EDL, text_styles: dict, positions: dict, k: float) -> list[tuple[float, str]]:
    W, H = edl.output.width, edl.output.height
    events = []
    for t in edl.texts:
        st = text_styles.get(t.style_id) or text_styles["title"]
        y = t.y if (t.position == "custom" and t.y is not None) else positions.get(t.position, positions["top"])
        words = [_case(_esc(w), st.get("case", "none")) for w in t.text.split()]
        if not words or t.t1 <= t.t0:
            continue
        body = _wrapped_text(st, words, words, st["size"] * k)
        s, e = max(0.0, t.t0), min(t.t1, edl.duration) if edl.duration else t.t1
        name = f"Txt_{t.style_id if t.style_id in text_styles else 'title'}"
        head = f"1,{_fmt_time(s)},{_fmt_time(e)},{name},,0,0,0,,"
        events.append((s, head + f"{{\\an5\\pos({W // 2},{round(y * H)})}}{body}"))
    return events


def caption_breaks(edl: EDL) -> list[int | None]:
    """Where each caption line breaks into two visual lines (index of the first word of line 2, or None) -
    computed with the SAME function and arguments the ASS export uses, so the browser preview can break identically."""
    cap_json, _ = load_styles()
    st = cap_json["styles"].get(edl.captions.style_id) or cap_json["styles"]["bold_pop"]
    size_px = st["size"] * (edl.output.height / REF_H)
    scale = st.get("active_scale", 1.1) if st["mode"] == "pop" else 1.0
    out: list[int | None] = []
    for line in edl.captions.lines:
        words = [_case(_esc(w.text), st.get("case", "none")) for w in line.words]
        out.append(wrap_index(st, words, scale, size_px) if words else None)
    return out


def text_breaks(edl: EDL) -> dict[str, int | None]:
    """Same for text overlays (keyed by overlay id)."""
    _, txt_json = load_styles()
    k = edl.output.height / REF_H
    out: dict[str, int | None] = {}
    for t in edl.texts:
        st = txt_json["styles"].get(t.style_id) or txt_json["styles"]["title"]
        words = [_case(_esc(w), st.get("case", "none")) for w in t.text.split()]
        out[t.id] = wrap_index(st, words, 1.0, st["size"] * k) if words else None
    return out


def build_ass(edl: EDL, caption_styles: dict | None = None, text_styles: dict | None = None) -> str:
    cap_json, txt_json = load_styles()
    if caption_styles is not None:
        cap_json = caption_styles
    if text_styles is not None:
        txt_json = text_styles
    W, H = edl.output.width, edl.output.height
    k = H / REF_H
    style_lines, events = [], []
    if edl.captions.enabled and edl.captions.lines:
        st = cap_json["styles"].get(edl.captions.style_id) or cap_json["styles"]["bold_pop"]
        if st.get("box") and st["mode"] == "line":
            style_lines.append(_style_line("CapBox", st, k))
            st_text = {**st, "box": None, "outline": 0, "shadow": 0}
        else:
            st_text = st
        style_lines.append(_style_line("Cap", st_text, k, karaoke=st["mode"] == "karaoke"))
        events += _caption_events(edl, st, cap_json["positions"], k)
    used = {t.style_id if t.style_id in txt_json["styles"] else "title" for t in edl.texts}
    for name in sorted(used):
        style_lines.append(_style_line(f"Txt_{name}", txt_json["styles"][name], k))
    events += _text_events(edl, txt_json["styles"], txt_json["positions"], k)
    events.sort(key=lambda e: e[0])
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding\n"
    )
    body = "\n".join(style_lines) + "\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    return head + body + "\n".join(f"Dialogue: {e[1]}" for e in events) + "\n"
