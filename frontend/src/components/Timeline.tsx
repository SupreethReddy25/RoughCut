import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectOut, SuggestionOut } from "../api-types";
import { confidenceColor } from "../lib/describe";
import { fmtTime } from "../lib/format";
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
}

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
export function Timeline({ project, outTime, playing, selected, selectedText, onSeekOut, onSelect, onSelectText }: Props) {
  const edl = project.edl;
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
    <div className="border-t border-white/10 bg-neutral-900/60" data-testid="timeline">
      <div className="flex items-center gap-2 px-3 py-1 text-[11px] text-neutral-400">
        <span className="font-semibold text-neutral-300">Timeline</span>
        <span>{edl.clips.length} clips · {fmtTime(duration)}</span>
        <div className="ml-auto flex items-center gap-1">
          <button type="button" aria-label="Zoom timeline out" onClick={() => setPps((p) => Math.max(MIN_PPS, Math.round(p / 1.5)))} className="rounded bg-white/10 px-1.5 hover:bg-white/20">−</button>
          <button type="button" aria-label="Zoom timeline in" onClick={() => setPps((p) => Math.min(MAX_PPS, Math.round(p * 1.5)))} className="rounded bg-white/10 px-1.5 hover:bg-white/20">+</button>
        </div>
      </div>
      <div ref={scroller} className="relative overflow-x-auto overflow-y-hidden pb-2" onClick={seek}>
        <div className="relative" style={{ width, height: 112 }}>
          {ticks.map((t) => (
            <div key={t} className="absolute top-0 h-3 border-l border-white/20 pl-0.5 text-[9px] text-neutral-500" style={{ left: t * pps }}>
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
                className={`absolute top-4 flex h-14 items-start justify-between overflow-hidden rounded border px-1 text-[10px] ${
                  hasSel ? "border-sky-400 bg-sky-500/30" : b.zoom > 1 ? "border-violet-400/60 bg-violet-500/25" : "border-white/20 bg-white/10"
                }`}
                style={{ left: b.left * pps, width: Math.max(b.width * pps - 1, 2) }}
              >
                {b.zoom > 1 ? <span className="rounded bg-violet-500 px-1 text-white">{b.zoom}×</span> : <span />}
                {weak && (
                  <button
                    type="button"
                    aria-label={`Review ${weak.decision_type} (${Math.round(weak.confidence * 100)}% sure)`}
                    data-testid={`clip-dot-${b.index}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect(weak);
                    }}
                    className={`mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full ring-1 ring-black/40 ${confidenceColor(weak.confidence)}`}
                  />
                )}
              </div>
            );
          })}
          <div className="absolute left-0 top-[78px] text-[9px] uppercase tracking-wide text-neutral-500">text</div>
          {edl.texts.map((t) => (
            <button
              key={t.id}
              type="button"
              data-testid={`text-bar-${t.id}`}
              onClick={(e) => {
                e.stopPropagation();
                onSelectText(selectedText === t.id ? null : t.id);
              }}
              className={`absolute top-[90px] h-5 truncate rounded px-1 text-left text-[10px] ${
                selectedText === t.id ? "bg-amber-400 text-black" : "bg-amber-500/50 text-amber-50"
              }`}
              style={{ left: t.t0 * pps, width: Math.max((t.t1 - t.t0) * pps, 12) }}
              title={t.text}
            >
              {t.text}
            </button>
          ))}
          <div
            data-testid="playhead"
            className="pointer-events-none absolute top-0 h-full w-px bg-rose-400"
            style={{ left: Math.min(outTime, duration) * pps }}
          />
        </div>
      </div>
    </div>
  );
}
