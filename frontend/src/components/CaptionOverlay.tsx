import { Fragment, useEffect, useRef, useState } from "react";
import type { CaptionStyle, StylesOut, TextStyle } from "../api-types";
import type { EDL } from "../types";
import { activeLine, activeWord, applyCase, lineEnd, wordFill } from "../lib/captions";
import { Anchored, lookStyle } from "./StyledText";

export interface CaptionState {
  line: number;
  word: number;
  fill: number; // karaoke fill of the active word, quantised to 5%
}

/** Pure: which line / word / fill is on screen at output time t. */
export function captionState(edl: EDL, style: CaptionStyle | undefined, t: number): CaptionState {
  const line = edl.captions.enabled ? activeLine(edl, t) : -1;
  if (line < 0) return { line: -1, word: -1, fill: 0 };
  const l = edl.captions.lines[line];
  const word = activeWord(l, t);
  let fill = 0;
  if (style?.mode === "karaoke" && word >= 0) {
    fill = Math.round(wordFill(l, word, t, lineEnd(edl.captions.lines, line, edl.duration)) * 20) / 20;
  }
  return { line, word, fill };
}

interface Props {
  edl: EDL;
  breaks: (number | null)[];
  styles: StylesOut;
  /** frame width in px: sizes are defined at 1080 wide and scaled by this */
  frameW: number;
  /** output time of the playhead, read every animation frame */
  getOutTime: () => number;
}

/**
 * The live caption layer. It reads the playhead itself every animation frame and only re-renders when the line,
 * the active word or the karaoke fill changes, so captions follow the video to the frame without re-rendering the
 * rest of the editor. Line breaks come from the server (`breaks`, computed by the export's own wrapping code).
 */
export function CaptionOverlay({ edl, breaks, styles, frameW, getOutTime }: Props) {
  const style = styles.captions.styles[edl.captions.style_id] ?? styles.captions.styles.bold_pop;
  const [st, setSt] = useState<CaptionState>({ line: -1, word: -1, fill: 0 });
  const last = useRef(st);
  const edlRef = useRef(edl);
  edlRef.current = edl;

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const s = captionState(edlRef.current, style, getOutTime());
      const p = last.current;
      if (s.line !== p.line || s.word !== p.word || s.fill !== p.fill) {
        last.current = s;
        setSt(s);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [style, getOutTime]);

  // the edit changed under a stopped playhead: show the right thing immediately
  useEffect(() => {
    const s = captionState(edl, style, getOutTime());
    last.current = s;
    setSt(s);
  }, [edl, style, getOutTime]);

  if (st.line < 0 || st.line >= edl.captions.lines.length) return null;
  const line = edl.captions.lines[st.line];
  const brk = breaks[st.line] ?? null;
  const scale = frameW / 1080;
  const pos = styles.captions.positions[edl.captions.position] ?? 0.68;
  const pop = style.mode === "pop";
  const karaoke = style.mode === "karaoke";
  const look = {
    font: style.font, size: style.size, color: style.color, outline_color: style.outline_color,
    outline: style.outline, shadow: style.shadow, box: style.box,
  };
  return (
    <Anchored y={pos} style={lookStyle(look, scale)}>
      <span data-testid="caption-line" data-line={st.line}>
        {line.words.map((w, i) => {
          const active = i === st.word;
          const text = applyCase(w.text, style.case);
          let color: string | undefined = w.emphasis ? style.emphasis_color : undefined;
          const wordStyle: React.CSSProperties = { display: "inline-block", color };
          if (pop && active) wordStyle.transform = `scale(${style.active_scale ?? 1.1})`;
          let inner: React.ReactNode = text;
          if (karaoke) {
            // unspoken words in the base colour, spoken ones in the fill colour, the active one swept left to right
            const fillColor = style.fill_color ?? style.emphasis_color;
            const pct = i < st.word ? 100 : active ? st.fill * 100 : 0;
            inner = (
              <span style={{ position: "relative", display: "inline-block" }}>
                <span>{text}</span>
                <span
                  aria-hidden
                  style={{ position: "absolute", left: 0, top: 0, color: fillColor, clipPath: `inset(0 ${100 - pct}% 0 0)` }}
                >
                  {text}
                </span>
              </span>
            );
            color = undefined;
          }
          return (
            <Fragment key={i}>
              {i > 0 ? (brk === i ? <br /> : " ") : null}
              <span data-word={i} data-active={active ? "1" : "0"} data-emphasis={w.emphasis ? "1" : "0"} style={wordStyle}>
                {inner}
              </span>
            </Fragment>
          );
        })}
      </span>
    </Anchored>
  );
}

/** Ids of the text overlays visible at output time t. */
export function visibleTextIds(edl: EDL, t: number): string[] {
  return edl.texts.filter((o) => t >= o.t0 && t < o.t1).map((o) => o.id);
}

/** Text overlays (title, label, callout): shown between their start and end output time. */
export function TextOverlays({
  edl, breaks, styles, frameW, getOutTime,
}: {
  edl: EDL;
  breaks: Record<string, number | null>;
  styles: StylesOut;
  frameW: number;
  getOutTime: () => number;
}) {
  const [visible, setVisible] = useState("");
  useEffect(() => {
    const update = () => {
      const k = visibleTextIds(edl, getOutTime()).join("|");
      setVisible((prev) => (prev === k ? prev : k));
    };
    update();
    if (edl.texts.length === 0) return;
    let raf = 0;
    const loop = () => {
      update();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [edl, getOutTime]);
  const ids = new Set(visible ? visible.split("|") : []);
  const scale = frameW / 1080;
  return (
    <>
      {edl.texts
        .filter((o) => ids.has(o.id))
        .map((o) => {
          const st: TextStyle = styles.texts.styles[o.style_id] ?? styles.texts.styles.title;
          const y = o.position === "custom" && o.y !== null ? o.y : (styles.texts.positions[o.position] ?? 0.13);
          const words = applyCase(o.text, st.case).split(/\s+/).filter(Boolean);
          const brk = breaks[o.id] ?? null;
          return (
            <Anchored key={o.id} y={y} style={lookStyle(st, scale)}>
              <span data-testid="text-overlay">
                {words.map((w, i) => (
                  <Fragment key={i}>
                    {i > 0 ? (brk === i ? <br /> : " ") : null}
                    {w}
                  </Fragment>
                ))}
              </span>
            </Anchored>
          );
        })}
    </>
  );
}
