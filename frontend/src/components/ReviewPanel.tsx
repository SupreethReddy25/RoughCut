import { useEffect, useMemo, useState } from "react";
import { explain } from "../api";
import type { ExplainOut, ProjectOut, SuggestionOut } from "../api-types";
import type { Unit, Word } from "../types";
import { actionLabel, actionsFor, confidenceColor, DECISION_LABEL, describeTarget } from "../lib/describe";
import { reviewProgress, sameKey } from "../lib/decide";
import { pct } from "../lib/format";
import type { Sel } from "./TranscriptEditor";

interface Props {
  project: ProjectOut;
  selected: Sel | null;
  fresh: Set<number>;
  onSelect: (s: SuggestionOut) => void;
  onAccept: (s: SuggestionOut) => void;
  onReject: (s: SuggestionOut) => void;
  onChoose: (s: SuggestionOut, action: string) => void;
  /** a global nudge ("snappier", "calmer", "more_zooms", "fewer_zooms"); omit to hide the buttons */
  onNudge?: (name: string) => void;
  /** show only "Needs your eye" (a batch video opened from its "Needs review" badge) */
  onlyUncertain?: boolean;
  onShowAll?: () => void;
  /** "Your style so far: ..." (PLAN 7.2) */
  styleText?: string;
  /** after a decision, jump to the next uncertain suggestion */
  autoAdvance?: boolean;
  onAutoAdvance?: (on: boolean) => void;
}

export const NUDGES: { name: string; label: string; hint: string }[] = [
  { name: "snappier", label: "Snappier", hint: "Cut more: tighter pauses, more cut phrases and fillers (for this video)" },
  { name: "calmer", label: "Calmer", hint: "Keep more of the pauses and phrases (for this video)" },
  { name: "more_zooms", label: "More zooms", hint: "Punch in more often (for this video)" },
  { name: "fewer_zooms", label: "Fewer zooms", hint: "Fewer punch-ins (for this video)" },
];

interface RowProps extends Pick<Props, "selected" | "fresh" | "onSelect" | "onAccept" | "onReject" | "onChoose"> {
  s: SuggestionOut;
  units: Map<number, Unit>;
  words: Map<number, Word>;
  index?: number;
}

function QueueRow({ s, selected, fresh, onSelect, onAccept, onReject, onChoose, units, words, index = 0 }: RowProps) {
  const sel = sameKey(s, selected);
  const arms = actionsFor(s);
  const strip = s.status === "pending" ? confidenceColor(s.confidence) : s.status === "accepted" ? "bg-sky-400" : "bg-fuchsia-400";
  return (
    <li
      data-testid={`queue-item-${s.id}`}
      data-selected={sel ? "1" : "0"}
      style={{ animationDelay: `${Math.min(index, 12) * 25}ms` }}
      className={`rc-in rc-lift relative overflow-hidden rounded-xl border py-2 pl-3.5 pr-2 text-sm ${
        sel ? "border-sky-400/70 bg-sky-500/[0.09] shadow-lg shadow-sky-950/40" : "border-white/[0.06] bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
      } ${fresh.has(s.id) ? "animate-pulse ring-1 ring-amber-300" : ""}`}
    >
      <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-1 ${strip} ${sel ? "opacity-100" : "opacity-70"}`} />
      <button type="button" onClick={() => onSelect(s)} className="flex w-full items-center gap-2 text-left">
        <span
          title={s.status === "pending" ? `confidence ${pct(s.confidence)}` : s.status}
          className={`h-2 w-2 shrink-0 rounded-full ${strip}`}
        />
        <span className="shrink-0 rounded-md bg-white/[0.05] px-1.5 py-px text-[9.5px] font-semibold uppercase tracking-wider text-neutral-400">
          {DECISION_LABEL[s.decision_type]}
        </span>
        <span className="min-w-0 flex-1 truncate text-neutral-200">{describeTarget(s, units, words)}</span>
        {s.status === "pending" && <span className="shrink-0 font-mono text-[10px] text-neutral-500">{pct(s.confidence)}</span>}
      </button>
      <div className="mt-1.5 flex items-center gap-1 text-xs">
        <span className="text-neutral-500">{s.status === "pending" ? "Suggested:" : s.status === "accepted" ? "Accepted:" : "Changed to:"}</span>
        <span className="font-medium text-neutral-100">{actionLabel(s, s.current_action, words)}</span>
        {s.status === "pending" && (
          <>
            <button
              type="button"
              aria-label="Accept"
              title="Accept (A)"
              onClick={() => onAccept(s)}
              className="rc-press ml-auto inline-flex items-center gap-1 rounded-md bg-emerald-500/15 px-2 py-0.5 font-medium text-emerald-300 ring-1 ring-emerald-400/25 hover:bg-emerald-500/30 hover:text-emerald-100"
            >
              ✓ <kbd className="text-[9.5px] opacity-60">A</kbd>
            </button>
            <button
              type="button"
              aria-label="Reject"
              title="Reject (R)"
              onClick={() => onReject(s)}
              className="rc-press inline-flex items-center gap-1 rounded-md bg-rose-500/15 px-2 py-0.5 font-medium text-rose-300 ring-1 ring-rose-400/25 hover:bg-rose-500/30 hover:text-rose-100"
            >
              ✕ <kbd className="text-[9.5px] opacity-60">R</kbd>
            </button>
          </>
        )}
      </div>
      {sel && (
        <div className="rc-pop mt-2 flex flex-wrap gap-1" role="group" aria-label="Choose another action">
          {arms.slice(0, 4).map((a, i) => (
            <button
              key={a}
              type="button"
              onClick={() => onChoose(s, a)}
              className={`rc-press rounded-md border px-2 py-0.5 text-xs ${
                a === s.current_action ? "border-sky-400/70 bg-sky-500/15 text-sky-200" : "border-white/10 text-neutral-300 hover:border-white/25 hover:bg-white/[0.06]"
              }`}
            >
              <kbd className="mr-1 text-[9.5px] opacity-50">{i + 1}</kbd>
              {actionLabel(s, a, words)}
            </button>
          ))}
        </div>
      )}
    </li>
  );
}

/**
 * The review queue (PLAN 7.3): "Needs your eye" = confidence below 70 %, least confident first; the confident rest is
 * collapsed ("spot-check"); decided items are listed below. Live re-scoring re-sorts it after every feedback.
 */
export function ReviewPanel({ project, selected, fresh, onSelect, onAccept, onReject, onChoose, onNudge, onlyUncertain, onShowAll, styleText, autoAdvance, onAutoAdvance }: Props) {
  const byId = useMemo(() => new Map(project.suggestions.map((s) => [s.id, s])), [project.suggestions]);
  const units = useMemo(() => new Map(project.units.map((u) => [u.id, u])), [project.units]);
  const words = useMemo(() => new Map(project.words.map((w) => [w.id, w])), [project.words]);
  const pick = (ids: number[]) => ids.map((i) => byId.get(i)).filter((s): s is SuggestionOut => !!s);
  const needs = pick(project.queue.needs_review);
  const confident = pick(project.queue.confident);
  const done = project.suggestions.filter((s) => s.status !== "pending");
  const rowProps = { selected, fresh, onSelect, onAccept, onReject, onChoose, units, words };
  const progress = reviewProgress(project.suggestions, project.queue);
  const share = progress.total ? progress.done / progress.total : 0;

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="review-panel">
      <div className="space-y-2 border-b border-white/[0.06] px-3.5 py-3 text-xs" data-testid="review-progress">
        <div className="flex items-center gap-2">
          <span className="text-neutral-300">
            Reviewed <span className="font-semibold text-white">{progress.done}</span> of {progress.total}
          </span>
          <span
            className={`rounded-full px-1.5 py-px text-[10.5px] ring-1 ${
              progress.uncertain === 0 ? "bg-emerald-500/10 text-emerald-300 ring-emerald-400/25" : "bg-amber-400/10 text-amber-300 ring-amber-400/25"
            }`}
          >
            {progress.uncertain === 0 ? "nothing uncertain left" : `${progress.uncertain} uncertain left`}
          </span>
          {onAutoAdvance && (
            <label className="ml-auto flex items-center gap-1.5 text-neutral-400 hover:text-neutral-200" title="After accepting, rejecting or choosing, jump to the next uncertain suggestion">
              <input type="checkbox" checked={!!autoAdvance} onChange={(e) => onAutoAdvance(e.target.checked)} /> next after each
            </label>
          )}
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.done} aria-label="Review progress">
          <div className="rc-meter h-full rounded-full bg-gradient-to-r from-sky-600 to-sky-400" style={{ width: `${100 * share}%` }} />
        </div>
        {styleText && (
          <p className="flex items-start gap-1.5 text-neutral-400" data-testid="style-text">
            <span aria-hidden="true" className="mt-px text-sky-300">✦</span>
            <span>{styleText}</span>
          </p>
        )}
      </div>
      {onNudge && (
        <div className="flex flex-wrap items-center gap-1 border-b border-white/[0.06] px-3.5 py-2 text-xs" role="group" aria-label="Nudge the whole edit">
          <span className="rc-label mr-1">Nudge</span>
          {NUDGES.map((n) => (
            <button key={n.name} type="button" title={n.hint} onClick={() => onNudge(n.name)} className="rc-press rounded-full border border-white/10 px-2.5 py-0.5 text-neutral-300 hover:border-sky-400/50 hover:bg-sky-500/10 hover:text-white">
              {n.label}
            </button>
          ))}
        </div>
      )}
      {onlyUncertain && (
        <p className="border-b border-white/[0.06] bg-amber-400/10 px-3.5 py-1.5 text-xs text-amber-100">
          Showing only what needs your eye.{" "}
          <button type="button" onClick={onShowAll} className="underline hover:text-white">
            Show everything
          </button>
        </p>
      )}
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-amber-300">
            <span aria-hidden="true" className="rc-breathe h-1.5 w-1.5 rounded-full bg-amber-400" />
            Needs your eye <span className="rounded-full bg-amber-400/15 px-1.5 text-amber-200">{needs.length}</span>
            <span className="ml-auto text-[10px] font-normal normal-case tracking-normal text-neutral-500">
              <kbd className="rounded border border-white/10 px-1">J</kbd> = next
            </span>
          </h3>
          {needs.length === 0 ? (
            <p className="rc-in rounded-xl border border-emerald-400/15 bg-emerald-500/[0.06] px-3 py-2.5 text-sm text-emerald-200/90">
              Nothing uncertain left. Spot-check the confident edits below or finish.
            </p>
          ) : (
            <ul className="space-y-1.5">{needs.map((s, i) => <QueueRow key={s.id} s={s} index={i} {...rowProps} />)}</ul>
          )}
        </section>
        {!onlyUncertain && (
          <>
            <details className="group">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-neutral-400 hover:text-neutral-200">
                <span aria-hidden="true" className="transition-transform group-open:rotate-90">›</span>
                Confident edits ({confident.length}), spot-check
              </summary>
              <ul className="mt-2 space-y-1.5">{confident.map((s, i) => <QueueRow key={s.id} s={s} index={i} {...rowProps} />)}</ul>
            </details>
            <details className="group">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-neutral-400 hover:text-neutral-200">
                <span aria-hidden="true" className="transition-transform group-open:rotate-90">›</span>
                Reviewed ({done.length})
              </summary>
              <ul className="mt-2 space-y-1.5">{done.map((s, i) => <QueueRow key={s.id} s={s} index={i} {...rowProps} />)}</ul>
            </details>
          </>
        )}
      </div>
    </div>
  );
}

/** Human-readable name of a context feature ("hist_since_last_cut" -> "since last cut"). */
function featureName(f: string): string {
  return f
    .replace(/^(cand|none):/, "")
    .replace(/^hist_/, "")
    .replace(/^log_/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** "Why?": the top feature contributions behind the selected suggestion, as sentences (PLAN 7.7). */
export function WhyBox({ project, suggestion }: { project: ProjectOut; suggestion: SuggestionOut | null }) {
  const [cache] = useState(() => new Map<number, ExplainOut>());
  const [text, setText] = useState<ExplainOut | null>(null);
  const [failed, setFailed] = useState(false);
  const words = useMemo(() => new Map(project.words.map((w) => [w.id, w])), [project.words]);

  useEffect(() => {
    setFailed(false);
    if (!suggestion) {
      setText(null);
      return;
    }
    const hit = cache.get(suggestion.id);
    if (hit) {
      setText(hit);
      return;
    }
    setText(null);
    let alive = true;
    explain(project.id, suggestion.id)
      .then((e) => {
        cache.set(suggestion.id, e);
        if (alive) setText(e);
      })
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [project.id, suggestion?.id]);

  if (!suggestion) {
    return (
      <div className="flex items-center gap-2 border-t border-white/[0.06] px-3.5 py-3 text-xs text-neutral-500">
        <span aria-hidden="true" className="flex h-5 w-5 items-center justify-center rounded-full bg-white/[0.05] text-[10px]">?</span>
        Select a suggestion to see why it was made.
      </div>
    );
  }
  const changed = suggestion.current_action !== suggestion.proposed_action;
  const reasons = (text?.reasons ?? []).slice(0, 3);
  const maxC = Math.max(1e-6, ...reasons.map((r) => Math.abs(r.contribution)));
  return (
    <div key={suggestion.id} className="rc-in border-t border-white/[0.06] bg-white/[0.015] px-3.5 py-3 text-xs" data-testid="why-box">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-semibold text-neutral-200">Why?</span>
        <span className="truncate text-neutral-500">
          proposed {actionLabel(suggestion, suggestion.proposed_action, words)} · {pct(suggestion.confidence)} sure
          {changed ? ` · now ${actionLabel(suggestion, suggestion.current_action, words)}` : ""}
        </span>
      </div>
      <div className="mb-2 h-1 overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
        <div className={`rc-meter h-full rounded-full ${confidenceColor(suggestion.confidence)}`} style={{ width: pct(suggestion.confidence) }} />
      </div>
      {failed ? (
        <p className="text-neutral-500">No explanation is available for this one.</p>
      ) : text ? (
        <>
          <p className="leading-relaxed text-neutral-200">{text.text}</p>
          {reasons.length > 0 && (
            <ul className="mt-2.5 space-y-1" aria-label="What weighed most">
              {reasons.map((r) => (
                <li key={r.feature} className="flex items-center gap-2" title={r.sentence}>
                  <span className="w-28 shrink-0 truncate text-[10.5px] text-neutral-400">{featureName(r.feature)}</span>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.05]">
                    <span
                      className={`rc-meter absolute inset-y-0 left-0 rounded-full ${r.contribution >= 0 ? "bg-sky-400" : "bg-rose-400/80"}`}
                      style={{ width: `${(100 * Math.abs(r.contribution)) / maxC}%` }}
                    />
                  </span>
                  <span className="w-9 shrink-0 text-right font-mono text-[10px] text-neutral-500">
                    {r.contribution >= 0 ? "+" : "−"}
                    {Math.abs(r.contribution).toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <div className="space-y-1.5" aria-label="Thinking…">
          <p className="text-neutral-500">Thinking…</p>
          <div className="relative h-2 w-3/4 overflow-hidden rounded bg-white/[0.05]">
            <div className="rc-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
          </div>
        </div>
      )}
    </div>
  );
}
