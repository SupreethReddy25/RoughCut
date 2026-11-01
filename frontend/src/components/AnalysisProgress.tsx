import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisProgressOut } from "../api-types";

const BARS = 56;

/** Stable pseudo-waveform per video, so a project always shows the same strip. */
function bars(seed: number): number[] {
  let x = (seed * 2654435761) % 4294967296 || 1;
  return Array.from({ length: BARS }, (_, i) => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    const base = 0.25 + ((x % 1000) / 1000) * 0.75;
    return i % 11 === 7 || i % 17 === 3 ? 0.06 : base; // the quiet gaps the edit will close
  });
}

export function fmtEta(s: number): string {
  if (s < 8) return "almost there";
  if (s < 60) return `about ${Math.round(s / 5) * 5} s left`;
  const m = Math.floor(s / 60);
  const r = Math.round((s - m * 60) / 15) * 15;
  return r === 60 ? `about ${m + 1} min left` : r ? `about ${m} min ${r} s left` : `about ${m} min left`;
}

/**
 * A piece of film being edited while the analysis runs: the playhead moves along a strip of the video's waveform, the
 * quiet gaps it passes are snipped out and the strip closes up, so the screen shows what the analysis is for.
 */
export function AnalysisProgress({ id, progress }: { id: number; progress: AnalysisProgressOut | null }) {
  const shape = useMemo(() => bars(id), [id]);
  const target = progress?.fraction ?? 0;
  const [shown, setShown] = useState(0);
  const [left, setLeft] = useState<number | null>(null);
  const latest = useRef({ target, serverLeft: progress?.seconds_left ?? null, at: Date.now() });

  useEffect(() => {
    latest.current = { target, serverLeft: progress?.seconds_left ?? null, at: Date.now() };
  }, [target, progress?.seconds_left]);

  // smooth between the server's updates: the playhead creeps, the countdown ticks
  useEffect(() => {
    const t = setInterval(() => {
      const L = latest.current;
      setShown((v) => {
        const room = L.target - v;
        return room > 0 ? v + room * 0.08 : v + 0.0004; // never quite stands still while work is going on
      });
      if (L.serverLeft !== null) setLeft(Math.max(L.serverLeft - (Date.now() - L.at) / 1000, 0));
    }, 120);
    return () => clearInterval(t);
  }, []);

  const f = Math.min(shown, 0.995);
  const head = f * BARS;
  const running = progress?.stages.find((s) => s.state === "running");
  return (
    <div className="mx-auto w-full max-w-md" data-testid="analysis-progress">
      <div className="relative overflow-hidden rounded-xl bg-neutral-900/80 px-3 py-2 ring-1 ring-white/10">
        <Sprockets />
        <div className="relative my-1.5 flex h-16 items-center justify-center gap-[2px]" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(f * 100)} aria-label="Analysis progress">
          {shape.map((h, i) => {
            const passed = i < head;
            const gap = h < 0.1;
            return (
              <span
                key={i}
                aria-hidden="true"
                className="block rounded-full transition-all duration-500"
                style={{
                  width: passed && gap ? 0 : 4,
                  marginInline: passed && gap ? -1 : 0,
                  height: `${Math.max(h * 100, 8)}%`,
                  opacity: passed && gap ? 0 : passed ? 1 : gap ? 0.5 : 0.28,
                  background: passed ? "linear-gradient(to top, rgb(56 189 248), rgb(139 124 246))" : gap ? "rgb(251 113 133)" : "rgb(163 163 163)",
                }}
              />
            );
          })}
          <span aria-hidden="true" className="absolute inset-y-0 w-0.5 rounded-full bg-white shadow-[0_0_10px_2px_rgba(255,255,255,.5)]" style={{ left: `${f * 100}%` }} />
        </div>
        <Sprockets />
      </div>

      <div className="mt-3 flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate text-neutral-200">{running?.label ?? (progress?.waiting ? "Waiting for a free slot" : "Getting started")}</span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-neutral-400" data-testid="analysis-eta" title={progress?.learned ? "Estimated from how long earlier videos took on this computer" : "A first estimate: it gets better after each video"}>
          {left !== null && !progress?.waiting ? fmtEta(left) : "estimating…"}
        </span>
      </div>

      <ol className="mt-4 grid gap-1.5 text-left text-xs">
        {(progress?.stages ?? []).map((s) => (
          <li key={s.key} className={`flex items-center gap-2 ${s.state === "waiting" ? "text-neutral-600" : s.state === "running" ? "text-sky-200" : "text-neutral-400"}`}>
            <span aria-hidden="true" className={`flex h-3.5 w-3.5 items-center justify-center rounded-full text-[9px] ${s.state === "done" ? "bg-emerald-500/20 text-emerald-300" : s.state === "running" ? "rc-breathe bg-sky-400/30" : "bg-white/[0.05]"}`}>
              {s.state === "done" ? "✓" : ""}
            </span>
            <span className="flex-1">{s.label}</span>
            {s.state === "done" && <span className="font-mono tabular-nums text-neutral-600">{s.seconds.toFixed(0)} s</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}

function Sprockets() {
  return (
    <div aria-hidden="true" className="flex justify-between">
      {Array.from({ length: 22 }, (_, i) => (
        <span key={i} className="h-1.5 w-2 rounded-[2px] bg-white/[0.08]" />
      ))}
    </div>
  );
}
