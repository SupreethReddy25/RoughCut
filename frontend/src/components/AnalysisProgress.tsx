import { useEffect, useMemo, useRef, useState } from "react";
import { thumbnailUrl } from "../api";
import type { AnalysisProgressOut } from "../api-types";

const BARS = 56;
type Live = NonNullable<AnalysisProgressOut["live"]>;

/** A placeholder shape until the real waveform is known. */
function placeholder(seed: number): number[] {
  let x = (seed * 2654435761) % 4294967296 || 1;
  return Array.from({ length: BARS }, () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return 0.2 + ((x % 1000) / 1000) * 0.5;
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
 * Shows the work itself: every stage has its own view built from what the analysis has really found so far (the video's
 * frame and waveform, the words as they are heard, the gaps and fillers, the face, the topic words).
 */
export function AnalysisProgress({ id, progress }: { id: number; progress: AnalysisProgressOut | null }) {
  const live: Live = progress?.live ?? {};
  const target = progress?.fraction ?? 0;
  const [shown, setShown] = useState(0);
  const [left, setLeft] = useState<number | null>(null);
  const latest = useRef({ target, serverLeft: progress?.seconds_left ?? null, at: Date.now() });

  useEffect(() => {
    latest.current = { target, serverLeft: progress?.seconds_left ?? null, at: Date.now() };
  }, [target, progress?.seconds_left]);

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
  const running = progress?.stages.find((s) => s.state === "running");
  const stage = running?.key ?? "ingest";
  return (
    <div className="mx-auto w-full max-w-md" data-testid="analysis-progress">
      <div className="min-h-[9.5rem]" data-testid={`stage-${stage}`}>
        {stage === "ingest" && <Preparing />}
        {stage === "transcribe" && <Listening id={id} live={live} />}
        {stage === "align" && <Finding id={id} live={live} />}
        {stage === "vision" && <Framing id={id} live={live} />}
        {stage === "semantic" && <Reading live={live} />}
      </div>
      {(stage === "vision" || stage === "semantic") && <FoundSoFar live={live} />}

      <div className="mt-3 flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate text-neutral-200">{running?.label ?? (progress?.waiting ? "Waiting for a free slot" : "Getting started")}</span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-neutral-400" data-testid="analysis-eta" title={progress?.learned ? "Estimated from how long earlier videos took on this computer" : "A first estimate: it gets better after each video"}>
          {left !== null && !progress?.waiting ? fmtEta(left) : "estimating…"}
        </span>
      </div>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/[0.07]" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(f * 100)} aria-label="Analysis progress">
        <div className="h-full rounded-full bg-gradient-to-r from-sky-400 to-violet-400" style={{ width: `${f * 100}%` }} />
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

function Film({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-neutral-900/80 px-3 py-2 ring-1 ring-white/10">
      <Sprockets />
      <div className="relative my-1.5">{children}</div>
      <Sprockets />
    </div>
  );
}

function Wave({ id, live, head, snip }: { id: number; live: Live; head?: number; snip?: boolean }) {
  const shape = useMemo(() => (live.peaks?.length ? live.peaks : placeholder(id)), [live.peaks, id]);
  const n = shape.length;
  const inPause = (i: number) => (live.pause_marks ?? []).some(([a, b]) => (i + 0.5) / n >= a && (i + 0.5) / n < b);
  const hasFiller = (i: number) => (live.filler_marks ?? []).some((t) => t * n >= i && t * n < i + 1);
  return (
    <div className="relative flex h-16 items-center justify-center gap-[2px]" aria-hidden="true">
      {shape.map((h, i) => {
        const passed = head === undefined || i / n < head;
        const pause = !!snip && inPause(i);
        const filler = !!snip && hasFiller(i);
        return (
          <span
            key={i}
            className="block rounded-full transition-all duration-700"
            style={{
              width: pause ? 2 : 4,
              height: `${Math.max(h * 100, 8)}%`,
              opacity: pause ? 0.35 : passed ? 1 : 0.28,
              background: pause ? "rgb(251 113 133)" : filler ? "rgb(251 191 36)" : passed ? "linear-gradient(to top, rgb(56 189 248), rgb(139 124 246))" : "rgb(163 163 163)",
            }}
          />
        );
      })}
      {head !== undefined && <span className="absolute inset-y-0 w-0.5 rounded-full bg-white shadow-[0_0_10px_2px_rgba(255,255,255,.5)]" style={{ left: `${Math.min(head, 1) * 100}%` }} />}
    </div>
  );
}

function Preparing() {
  return (
    <Film>
      <div className="flex h-16 items-center justify-center gap-1.5 overflow-hidden" aria-hidden="true">
        {Array.from({ length: 9 }, (_, i) => (
          <span key={i} className="rc-breathe h-12 w-9 shrink-0 rounded-sm bg-white/[0.08] ring-1 ring-white/10" style={{ animationDelay: `${i * 180}ms` }} />
        ))}
      </div>
      <p className="mt-1 text-center text-[11px] text-neutral-500">Converting your footage to a steady frame rate and pulling out the sound</p>
    </Film>
  );
}

function Listening({ id, live }: { id: number; live: Live }) {
  const lines = live.heard ?? [];
  return (
    <div>
      <Film>
        <Wave id={id} live={live} head={live.heard_fraction ?? 0} />
      </Film>
      <div className="mt-2 h-[4.5rem] overflow-hidden rounded-lg bg-white/[0.03] p-2.5 text-left text-sm leading-snug text-neutral-300 ring-1 ring-white/[0.06]" data-testid="heard" aria-live="off">
        {lines.length === 0 ? <span className="text-neutral-500">Listening…</span> : lines.slice(-2).map((t, i, a) => (
          <p key={t + i} className={`rc-in ${i === a.length - 1 ? "text-neutral-100" : "text-neutral-500"}`}>“{t}”</p>
        ))}
      </div>
    </div>
  );
}

function Finding({ id, live }: { id: number; live: Live }) {
  return (
    <div>
      <Film>
        <Wave id={id} live={live} snip />
      </Film>
      {live.n_pauses === undefined ? (
        <p className="mt-2 text-xs text-neutral-500">Lining up every word with the sound, to the hundredth of a second…</p>
      ) : (
        <div className="rc-in mt-2 grid grid-cols-3 gap-2 text-center" data-testid="found-counts">
          <Stat value={live.n_pauses} label="pauses" color="text-rose-300" />
          <Stat value={live.n_fillers ?? 0} label="filler words" color="text-amber-300" />
          <Stat value={live.n_phrases ?? 0} label="phrases" color="text-sky-300" />
        </div>
      )}
      {!!live.filler_words?.length && <p className="mt-1.5 truncate text-left text-[11px] text-amber-200/70">{live.filler_words.map((w) => `“${w}”`).join("  ")}</p>}
    </div>
  );
}

function Stat({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div className="rounded-lg bg-white/[0.03] py-1.5 ring-1 ring-white/[0.06]">
      <div className={`font-mono text-lg tabular-nums ${color}`}>{value}</div>
      <div className="text-[10px] text-neutral-500">{label}</div>
    </div>
  );
}

function Framing({ id, live }: { id: number; live: Live }) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-36 flex-1 overflow-hidden rounded-xl bg-black ring-1 ring-white/10">
        {live.frame && <img src={thumbnailUrl(id)} alt="" className="absolute inset-0 h-full w-full object-contain opacity-80" />}
        <div className="rc-scan absolute inset-x-0 h-8 bg-gradient-to-b from-transparent via-sky-400/40 to-transparent" aria-hidden="true" />
      </div>
      <p className="w-36 text-left text-xs leading-relaxed text-neutral-400">Looking for your face in each moment, so the 9:16 frame stays on you.</p>
    </div>
  );
}

/** What has been found stays on screen once it is known. */
function FoundSoFar({ live }: { live: Live }) {
  if (live.n_pauses === undefined) return null;
  return (
    <div className="rc-in mt-2 text-left" data-testid="found-so-far">
      <p className="text-xs text-neutral-400">
        Found <span className="text-rose-300">{live.n_pauses} pauses</span>, <span className="text-amber-300">{live.n_fillers ?? 0} filler words</span> and{" "}
        <span className="text-sky-300">{live.n_phrases ?? 0} phrases</span>.
      </p>
      {!!live.keywords?.length && <p className="mt-0.5 truncate text-[11px] text-neutral-500">Talking about: {live.keywords.slice(0, 5).join(", ")}</p>}
    </div>
  );
}

function Reading({ live }: { live: Live }) {
  return (
    <div className="flex min-h-[9.5rem] flex-col justify-center rounded-xl bg-neutral-900/80 p-3 ring-1 ring-white/10">
      <p className="mb-2 text-left text-xs text-neutral-400">{live.face ? `Face found in ${Math.round(live.face.seen * 100)}% of the video. ` : ""}What this video keeps coming back to:</p>
      <div className="flex flex-wrap gap-1.5" data-testid="keywords">
        {(live.keywords ?? []).map((k, i) => (
          <span key={k} className="rc-in rounded-full bg-violet-500/15 px-2.5 py-1 text-xs text-violet-100 ring-1 ring-violet-400/25" style={{ animationDelay: `${i * 120}ms` }}>{k}</span>
        ))}
      </div>
    </div>
  );
}
