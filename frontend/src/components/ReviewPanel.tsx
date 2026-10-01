import { useEffect, useMemo, useState } from "react";
import { explain } from "../api";
import type { ExplainOut, ProjectOut, SuggestionOut } from "../api-types";
import type { Unit, Word } from "../types";
import { actionLabel, actionsFor, confidenceColor, DECISION_LABEL, describeTarget } from "../lib/describe";
import { sameKey } from "../lib/decide";
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
}


interface RowProps extends Pick<Props, "selected" | "fresh" | "onSelect" | "onAccept" | "onReject" | "onChoose"> {
  s: SuggestionOut;
  units: Map<number, Unit>;
  words: Map<number, Word>;
}

function QueueRow({ s, selected, fresh, onSelect, onAccept, onReject, onChoose, units, words }: RowProps) {
  const sel = sameKey(s, selected);
  const arms = actionsFor(s);
  return (
    <li
      data-testid={`queue-item-${s.id}`}
      data-selected={sel ? "1" : "0"}
      className={`rounded-lg border px-2 py-1.5 text-sm ${
        sel ? "border-sky-400 bg-sky-500/10" : "border-white/10 hover:bg-white/5"
      } ${fresh.has(s.id) ? "animate-pulse ring-1 ring-amber-300" : ""}`}
    >
      <button type="button" onClick={() => onSelect(s)} className="flex w-full items-center gap-2 text-left">
        <span
          title={s.status === "pending" ? `confidence ${pct(s.confidence)}` : s.status}
          className={`h-2.5 w-2.5 shrink-0 rounded-full ${
            s.status === "pending" ? confidenceColor(s.confidence) : s.status === "accepted" ? "bg-sky-400" : "bg-violet-400"
          }`}
        />
        <span className="w-14 shrink-0 text-[10px] uppercase tracking-wide text-neutral-400">{DECISION_LABEL[s.decision_type]}</span>
        <span className="min-w-0 flex-1 truncate text-neutral-200">{describeTarget(s, units, words)}</span>
      </button>
      <div className="mt-1 flex items-center gap-1 pl-4 text-xs">
        <span className="text-neutral-400">{s.status === "pending" ? "Suggested:" : s.status === "accepted" ? "Accepted:" : "Changed to:"}</span>
        <span className="font-medium text-neutral-100">{actionLabel(s, s.current_action, words)}</span>
        {s.status === "pending" && (
          <>
            <button
              type="button"
              aria-label="Accept"
              onClick={() => onAccept(s)}
              className="ml-auto rounded bg-emerald-600/80 px-1.5 py-0.5 text-white hover:bg-emerald-500"
            >
              ✓ <kbd className="text-[10px] opacity-70">A</kbd>
            </button>
            <button
              type="button"
              aria-label="Reject"
              onClick={() => onReject(s)}
              className="rounded bg-rose-600/80 px-1.5 py-0.5 text-white hover:bg-rose-500"
            >
              ✗ <kbd className="text-[10px] opacity-70">R</kbd>
            </button>
          </>
        )}
      </div>
      {sel && (
        <div className="mt-1.5 flex flex-wrap gap-1 pl-4" role="group" aria-label="Choose another action">
          {arms.slice(0, 4).map((a, i) => (
            <button
              key={a}
              type="button"
              onClick={() => onChoose(s, a)}
              className={`rounded border px-1.5 py-0.5 text-xs ${
                a === s.current_action ? "border-sky-400 text-sky-300" : "border-white/15 text-neutral-300 hover:bg-white/10"
              }`}
            >
              <kbd className="mr-1 text-[10px] opacity-60">{i + 1}</kbd>
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
export function ReviewPanel({ project, selected, fresh, onSelect, onAccept, onReject, onChoose }: Props) {
  const byId = useMemo(() => new Map(project.suggestions.map((s) => [s.id, s])), [project.suggestions]);
  const units = useMemo(() => new Map(project.units.map((u) => [u.id, u])), [project.units]);
  const words = useMemo(() => new Map(project.words.map((w) => [w.id, w])), [project.words]);
  const pick = (ids: number[]) => ids.map((i) => byId.get(i)).filter((s): s is SuggestionOut => !!s);
  const needs = pick(project.queue.needs_review);
  const confident = pick(project.queue.confident);
  const done = project.suggestions.filter((s) => s.status !== "pending");
  const rowProps = { selected, fresh, onSelect, onAccept, onReject, onChoose, units, words };

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="review-panel">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <section>
          <h3 className="mb-1.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-amber-300">
            Needs your eye <span className="rounded bg-amber-400/20 px-1.5 text-amber-200">{needs.length}</span>
            <span className="ml-auto text-[10px] font-normal normal-case text-neutral-500">J = next</span>
          </h3>
          {needs.length === 0 ? (
            <p className="text-sm text-neutral-500">Nothing uncertain left. Spot-check the confident edits below or finish.</p>
          ) : (
            <ul className="space-y-1.5">{needs.map((s) => <QueueRow key={s.id} s={s} {...rowProps} />)}</ul>
          )}
        </section>
        <details>
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-neutral-300">
            Confident edits ({confident.length}), spot-check
          </summary>
          <ul className="mt-1.5 space-y-1.5">{confident.map((s) => <QueueRow key={s.id} s={s} {...rowProps} />)}</ul>
        </details>
        <details>
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-neutral-300">Reviewed ({done.length})</summary>
          <ul className="mt-1.5 space-y-1.5">{done.map((s) => <QueueRow key={s.id} s={s} {...rowProps} />)}</ul>
        </details>
      </div>
    </div>
  );
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
    return <div className="border-t border-white/10 p-3 text-xs text-neutral-500">Select a suggestion to see why it was made.</div>;
  }
  const changed = suggestion.current_action !== suggestion.proposed_action;
  return (
    <div className="border-t border-white/10 p-3 text-xs" data-testid="why-box">
      <div className="mb-1 flex items-center gap-2 font-semibold text-neutral-300">
        Why?
        <span className="font-normal text-neutral-500">
          proposed {actionLabel(suggestion, suggestion.proposed_action, words)} · {pct(suggestion.confidence)} sure
          {changed ? ` · now ${actionLabel(suggestion, suggestion.current_action, words)}` : ""}
        </span>
      </div>
      {failed ? (
        <p className="text-neutral-500">No explanation is available for this one.</p>
      ) : text ? (
        <p className="text-neutral-200">{text.text}</p>
      ) : (
        <p className="text-neutral-500">Thinking…</p>
      )}
    </div>
  );
}
