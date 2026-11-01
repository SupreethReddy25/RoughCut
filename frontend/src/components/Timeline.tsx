import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectOut, SuggestionOut } from "../api-types";
import { confidenceColor } from "../lib/describe";
import { fmtTime } from "../lib/format";
import { dragOverlay, type DragMode } from "../lib/overlay";
import type { Sel } from "./TranscriptEditor";

interface Props {
  project: ProjectOut;
  outTime: number;
  playing: boolean;
  selected: Sel | null;
  selectedText: string | null;
  onSeekOut: (o: number) => void;
  onSelect: (s: SuggestionOut) => void;
  onSelectText: (id: string | null) => void;
  /** a text bar was moved / resized on the text track (output seconds) */
  onOverlayChange?: (id: string, t0: number, t1: number) => void;
}

const EDGE_PX = 7; // grab zone at either end of a text bar

export interface ClipBlock {
  index: number;
  left: number; // output seconds
  width: number;
  zoom: number;
  unitIds: number[];
}

/** Clip blocks laid out on the OUTPUT timeline (what the Reel will be), in output seconds. */
export function layoutClips(project: ProjectOut): ClipBlock[] {
  const edl = project.edl;
  if (!edl) return [];
  let acc = 0;
  return edl.clips.map((c, index) => {
    const width = (c.src_end - c.src_start) / edl.output.speed;
    const b = { index, left: acc, width, zoom: c.zoom, unitIds: c.unit_ids };
    acc += width;
    return b;
  });
}

/** The weakest pending suggestion among the units of a clip (the one to show a confidence dot for). */
export function weakestIn(unitIds: number[], suggestions: SuggestionOut[]): SuggestionOut | null {
  const ids = new Set(unitIds);
  let worst: SuggestionOut | null = null;
  for (const s of suggestions) {
    if (s.status !== "pending" || s.unit_id === null || !ids.has(s.unit_id)) continue;
    if (!worst || s.confidence < worst.confidence) worst = s;
  }
  return worst;
}

const MIN_PPS = 6;
const MAX_PPS = 120;

/**
 * Timeline (PLAN Phase 4): clips with zoom badges and a confidence dot per clip, a ruler, the playhead, and the
 * text-overlay track. Click to seek; click a clip's dot to review its weakest decision.
 */
// a tiled waveform texture for clips (bars of varying height, centred)
const WAVE = `url("data:image/svg+xml,${encodeURIComponent(
  `<svg xmlns='http://www.w3.org/2000/svg' width='30' height='40' preserveAspectRatio='none'>${[14, 26, 18, 34, 22, 10, 28, 16, 30, 20]
    .map((h, i) => `<rect x='${i * 3}' y='${(40 - h) / 2}' width='1.6' height='${h}' rx='0.8' fill='white'/>`)
    .join("")}</svg>`,
)}")`;

export function Timeline({ project, outTime, playing, selected, selectedText, onSeekOut, onSelect, onSelectText, onOverlayChange }: Props) {
  const edl = project.edl;
  const [drag, setDrag] = useState<{ id: string; t0: number; t1: number } | null>(null);
  const dragStart = useRef<{ x: number; mode: DragMode; t0: number; t1: number } | null>(null);
  const moved = useRef(false);
  const scroller = useRef<HTMLDivElement>(null);
  const [pps, setPps] = useState(24);
  const blocks = useMemo(() => layoutClips(project), [project]);
  const duration = edl?.duration ?? 0;
  const width = Math.max(duration * pps, 200);

  // fit the whole programme into the visible width the first time it appears
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || !scroller.current || duration <= 0) return;
    const w = scroller.current.clientWidth;
    if (w > 0) {
      setPps(Math.min(MAX_PPS, Math.max(MIN_PPS, Math.floor((w - 24) / duration))));
      fitted.current = true;
    }
  }, [duration]);

  // keep the playhead in view while playing
  useEffect(() => {
    const el = scroller.current;
    if (!el || !playing) return;
    const x = outTime * pps;
    if (x < el.scrollLeft + 20 || x > el.scrollLeft + el.clientWidth - 40) el.scrollLeft = Math.max(0, x - el.clientWidth / 3);
  }, [outTime, pps, playing]);

  if (!edl) return null;
  const seek = (e: React.MouseEvent) => {
    const el = scroller.current!;
    const rect = el.getBoundingClientRect();
    onSeekOut(Math.min(Math.max((e.clientX - rect.left + el.scrollLeft) / pps, 0), duration));
  };
  const tick = pps >= 60 ? 1 : pps >= 24 ? 5 : pps >= 10 ? 10 : 30;
  const ticks: number[] = [];
  for (let t = 0; t <= duration; t += tick) ticks.push(t);

  return (
    <div className="border-t border-white/[0.06] bg-neutral-900/70" data-testid="timeline">
      <div className="flex items-center gap-2 px-3 pb-1 pt-1.5 text-[11px] text-neutral-400">
        <span className="rc-label text-neutral-400">Timeline</span>
        <span className="font-mono tabular-nums">{edl.clips.length} clips · {fmtTime(duration)}</span>
        <div className="ml-auto flex items-center rounded-md bg-white/[0.04] p-0.5 ring-1 ring-white/[0.06]">
          <button type="button" aria-label="Zoom timeline out" onClick={() => setPps((p) => Math.max(MIN_PPS, Math.round(p / 1.5)))} className="rc-press rounded px-2 hover:bg-white/10 hover:text-white">−</button>
          <button type="button" aria-label="Zoom timeline in" onClick={() => setPps((p) => Math.min(MAX_PPS, Math.round(p * 1.5)))} className="rc-press rounded px-2 hover:bg-white/10 hover:text-white">+</button>
        </div>
      </div>
      <div ref={scroller} className="relative overflow-x-auto overflow-y-hidden pb-2" onClick={seek}>
        <div className="relative" style={{ width, height: 112 }}>
          {ticks.map((t) => (
            <div key={t} className="absolute top-0 h-3 border-l border-white/15 pl-1 font-mono text-[9px] text-neutral-500" style={{ left: t * pps }}>
              {fmtTime(t).replace(".0", "")}
            </div>
          ))}
          {blocks.map((b) => {
            const weak = weakestIn(b.unitIds, project.suggestions);
            const hasSel = !!selected && project.suggestions.some(
              (s) => s.decision_type === selected.decision_type && s.target_id === selected.target_id && s.unit_id !== null && b.unitIds.includes(s.unit_id),
            );
            return (
              <div
                key={b.index}
                data-testid={`clip-${b.index}`}
                className={`absolute top-4 flex h-14 items-start justify-between overflow-hidden rounded-md border px-1 py-0.5 text-[10px] transition-colors ${
                  hasSel
                    ? "border-sky-400 bg-sky-500/25 shadow-[0_0_0_1px_rgb(157_143_255/0.5),0_6px_20px_-6px_rgb(111_92_246/0.6)]"
                    : b.zoom > 1
                      ? "border-sky-300/30 bg-sky-500/[0.14] hover:bg-sky-500/20"
                      : "border-white/[0.09] bg-white/[0.06] hover:bg-white/[0.09]"
                }`}
                style={{ left: b.left * pps, width: Math.max(b.width * pps - 1, 2) }}
              >
                {/* a waveform texture, so the track reads as audio/video at a glance */}
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-0 bottom-1.5 top-4 opacity-40"
                  style={{ backgroundImage: WAVE, backgroundSize: "30px 100%", backgroundRepeat: "repeat-x" }}
                />
                {b.zoom > 1 ? <span className="relative rounded bg-sky-500 px-1 font-medium text-white shadow">{b.zoom}×</span> : <span />}
                {weak && (
                  <button
                    type="button"
                    aria-label={`Review ${weak.decision_type} (${Math.round(weak.confidence * 100)}% sure)`}
                    data-testid={`clip-dot-${b.index}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect(weak);
                    }}
                    className={`relative mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full ring-2 ring-neutral-950/70 transition-transform hover:scale-150 ${confidenceColor(weak.confidence)}`}
                  />
                )}
              </div>
            );
          })}
          <div className="absolute left-1 top-[78px] text-[9px] font-semibold uppercase tracking-wider text-neutral-600">text</div>
          {edl.texts.filter((t) => !t.id.startsWith("pop_")).map((t) => {
            const a = drag?.id === t.id ? drag.t0 : t.t0;
            const b = drag?.id === t.id ? drag.t1 : t.t1;
            return (
              <button
                key={t.id}
                type="button"
                data-testid={`text-bar-${t.id}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (moved.current) return; // the click that ends a drag is not a selection
                  onSelectText(selectedText === t.id ? null : t.id);
                }}
                onPointerDown={(e) => {
                  if (!onOverlayChange || e.button !== 0) return;
                  e.stopPropagation();
                  const r = e.currentTarget.getBoundingClientRect();
                  const mode: DragMode = e.clientX - r.left <= EDGE_PX ? "start" : r.right - e.clientX <= EDGE_PX ? "end" : "move";
                  e.currentTarget.setPointerCapture?.(e.pointerId);
                  dragStart.current = { x: e.clientX, mode, t0: t.t0, t1: t.t1 };
                  moved.current = false;
                }}
                onPointerMove={(e) => {
                  const s = dragStart.current;
                  if (!s) return;
                  if (Math.abs(e.clientX - s.x) > 2) moved.current = true;
                  if (!moved.current) return;
                  const [n0, n1] = dragOverlay(s.mode, s.t0, s.t1, (e.clientX - s.x) / pps, duration);
                  setDrag({ id: t.id, t0: n0, t1: n1 });
                }}
                onPointerUp={() => {
                  const s = dragStart.current;
                  dragStart.current = null;
                  const d = drag;
                  setDrag(null);
                  setTimeout(() => (moved.current = false), 0); // after the click that ends the drag has been ignored
                  if (s && moved.current && d && d.id === t.id && (d.t0 !== s.t0 || d.t1 !== s.t1)) onOverlayChange?.(t.id, d.t0, d.t1);
                }}
                onPointerCancel={() => {
                  dragStart.current = null;
                  setDrag(null);
                }}
                className={`absolute top-[90px] h-5 touch-none truncate rounded-md px-1.5 text-left text-[10px] font-medium ${onOverlayChange ? "cursor-grab active:cursor-grabbing" : ""} ${
                  selectedText === t.id ? "bg-amber-300 text-neutral-950 shadow-lg shadow-amber-900/40" : "bg-amber-400/20 text-amber-100 ring-1 ring-amber-300/30 hover:bg-amber-400/30"
                }`}
                style={{ left: a * pps, width: Math.max((b - a) * pps, 12) }}
                title={`${t.text} (drag to move, pull an end to resize)`}
              >
                {t.text}
              </button>
            );
          })}
          <div
            data-testid="playhead"
            className="pointer-events-none absolute top-0 h-full w-px bg-rose-400 shadow-[0_0_8px_rgb(251_113_133/0.8)]"
            style={{ left: Math.min(outTime, duration) * pps }}
          >
            <span aria-hidden="true" className="absolute -left-[4px] -top-px h-2 w-[9px] rounded-b-sm bg-rose-400" />
          </div>
        </div>
      </div>
    </div>
  );
}
