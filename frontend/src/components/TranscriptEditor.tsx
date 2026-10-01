import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { ManualEditIn, ProjectOut, SuggestionOut } from "../api-types";
import { confidenceColor } from "../lib/describe";
import { fmtGap } from "../lib/format";
import { groupWordsByUnit, keptWordIds, wordAt } from "../lib/words";
import type { Word } from "../types";

export interface TranscriptHandle {
  /** cut the selected (kept) words; false if nothing was selected */
  deleteSelection(): boolean;
  /** bring back the selected (cut) words */
  restoreSelection(): boolean;
}

export interface Sel {
  decision_type: string;
  target_id: number;
}

interface Props {
  project: ProjectOut;
  srcTime: number;
  playing: boolean;
  selected: Sel | null;
  onSelect: (s: SuggestionOut) => void;
  onSeek: (srcTime: number) => void;
  onEdit: (edit: ManualEditIn) => void;
  onCaptionText: (wordId: number, text: string) => void;
}

const PAUSE_OPTIONS: { action: string; label: string }[] = [
  { action: "KEEP", label: "Keep as is" },
  { action: "TIGHT_40", label: "Tighten to 0.4 s" },
  { action: "TIGHT_15", label: "Tighten to 0.15 s" },
  { action: "REMOVE", label: "Remove" },
];

export function ConfidenceDot({ s }: { s: SuggestionOut | undefined }) {
  if (!s) return null;
  return (
    <span
      data-testid={`dot-${s.decision_type}-${s.target_id}`}
      title={s.status === "pending" ? `confidence ${Math.round(s.confidence * 100)}%` : s.status}
      className={`mr-1 inline-block h-2 w-2 shrink-0 rounded-full align-middle ${
        s.status === "pending" ? confidenceColor(s.confidence) : s.status === "accepted" ? "bg-sky-400" : "bg-violet-400"
      }`}
    />
  );
}

/** Word ids (in document order) whose text the current text selection covers, restricted to `root`. */
export function selectedWordIds(root: HTMLElement, sel: Selection | null): number[] {
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return [];
  const range = sel.getRangeAt(0);
  if (!root.contains(range.commonAncestorContainer)) return [];
  const ids: number[] = [];
  root.querySelectorAll<HTMLElement>("[data-word-id]").forEach((span) => {
    const r = document.createRange();
    r.selectNodeContents(span);
    // overlap with positive length: the selection starts before the word ends and ends after the word starts
    if (range.compareBoundaryPoints(Range.END_TO_START, r) < 0 && range.compareBoundaryPoints(Range.START_TO_END, r) > 0) {
      ids.push(Number(span.dataset.wordId));
    }
  });
  return ids;
}

/**
 * The transcript as the main editing surface (PLAN 8.1): kept words are normal, cut words struck and grey, filler
 * tokens dotted, pauses are inline chips with an options popover, a selection is cut with Delete / restored, and a
 * double click fixes the caption text of a word. Every action goes out as a manual edit, which the server maps to
 * the right event (override / fine cut) of PLAN 6.4.
 */
export const TranscriptEditor = forwardRef<TranscriptHandle, Props>(function TranscriptEditor(props, ref) {
  const { project, srcTime, playing, selected, onSelect, onSeek, onEdit, onCaptionText } = props;
  const rootRef = useRef<HTMLDivElement>(null);
  const [selIds, setSelIds] = useState<number[]>([]);
  const [popover, setPopover] = useState<number | null>(null);
  const [editing, setEditing] = useState<{ id: number; value: string; original: string } | null>(null);

  const edl = project.edl;
  const kept = useMemo(() => (edl ? keptWordIds(project.words, edl.clips) : new Set<number>()), [project.words, edl]);
  const byUnit = useMemo(() => groupWordsByUnit(project.words), [project.words]);
  const sugBy = useMemo(() => {
    const m = new Map<string, SuggestionOut>();
    for (const s of project.suggestions) if (s.decision_type !== "emphasis") m.set(`${s.decision_type}:${s.target_id}`, s);
    return m;
  }, [project.suggestions]);
  const playingWord = playing ? wordAt(project.words, srcTime) : null;
  /** the filler decision of the filler token this word belongs to (filler units sit inside their phrase's span) */
  const fillerSuggestionFor = (w: Word): SuggestionOut | undefined => {
    const fu = project.units.find((u) => u.kind === "filler" && u.parent_unit_id === w.unit_id && u.start <= w.start + 1e-3 && u.end >= w.end - 1e-3);
    return fu ? sugBy.get(`filler:${fu.id}`) : undefined;
  };

  // ---- selection (a collapsed selection means "nothing selected")
  useEffect(() => {
    let raf = 0;
    const onChange = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const root = rootRef.current;
        if (root) setSelIds(selectedWordIds(root, window.getSelection()));
      });
    };
    document.addEventListener("selectionchange", onChange);
    return () => {
      document.removeEventListener("selectionchange", onChange);
      cancelAnimationFrame(raf);
    };
  }, []);

  const selKept = selIds.filter((i) => kept.has(i));
  const selCut = selIds.filter((i) => !kept.has(i));

  const clearSelection = () => {
    window.getSelection()?.removeAllRanges();
    setSelIds([]);
  };
  const doDelete = (): boolean => {
    const ids = selectedWordIds(rootRef.current!, window.getSelection()).filter((i) => kept.has(i));
    if (ids.length === 0) return false;
    onEdit({ kind: "delete_words", word_ids: ids });
    clearSelection();
    return true;
  };
  const doRestore = (): boolean => {
    const ids = selectedWordIds(rootRef.current!, window.getSelection()).filter((i) => !kept.has(i));
    if (ids.length === 0) return false;
    onEdit({ kind: "restore_words", word_ids: ids });
    clearSelection();
    return true;
  };
  useImperativeHandle(ref, () => ({ deleteSelection: doDelete, restoreSelection: doRestore }));

  // ---- keep the playing word in view
  useEffect(() => {
    if (!playingWord) return;
    rootRef.current?.querySelector(`[data-word-id="${playingWord.id}"]`)?.scrollIntoView({ block: "nearest" });
  }, [playingWord?.id]);

  // ---- close the pause popover on outside click / Escape
  useEffect(() => {
    if (popover === null) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !(e.target as HTMLElement).closest("[data-popover]")) setPopover(null);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [popover]);

  const isSel = (dt: string, id: number) => selected?.decision_type === dt && selected.target_id === id;

  const commitEdit = () => {
    if (!editing) return;
    const v = editing.value.trim();
    if (v && v !== editing.original) onCaptionText(editing.id, v);
    setEditing(null);
  };

  const renderWord = (w: Word) => {
    const isKept = kept.has(w.id);
    const cls = [
      "cursor-pointer rounded px-0.5",
      isKept ? "text-neutral-100" : "text-neutral-500 line-through decoration-neutral-500",
      w.is_filler ? "underline decoration-dotted decoration-amber-400 underline-offset-4" : "",
      playingWord?.id === w.id ? "bg-amber-400/40" : "hover:bg-white/10",
      selIds.includes(w.id) ? "bg-sky-500/30" : "",
    ].join(" ");
    if (editing?.id === w.id) {
      return (
        <input
          key={w.id}
          autoFocus
          aria-label="Caption text"
          value={editing.value}
          onChange={(e) => setEditing({ ...editing, value: e.target.value })}
          onBlur={commitEdit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitEdit();
            else if (e.key === "Escape") setEditing(null);
          }}
          className="w-28 rounded bg-neutral-700 px-1 text-neutral-100 outline outline-sky-400"
        />
      );
    }
    return (
      <span
        key={w.id}
        data-word-id={w.id}
        data-kept={isKept ? "1" : "0"}
        className={cls}
        onClick={() => {
          if (window.getSelection()?.isCollapsed === false) return; // a drag-selection ends in a click
          onSeek(w.start);
          const s = w.is_filler ? fillerSuggestionFor(w) : sugBy.get(`phrase:${w.unit_id}`);
          if (s) onSelect(s);
        }}
        onDoubleClick={() => {
          if (!isKept) return;
          const text = w.text_display.replace(/[.,;:!?]+$/, "");
          setEditing({ id: w.id, value: text, original: text });
        }}
      >
        {w.text_display}
      </span>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="transcript">
      <div className="flex items-center gap-2 border-b border-white/10 px-3 py-2 text-xs">
        <span className="font-semibold text-neutral-300">Transcript</span>
        <span className="text-neutral-500">
          {selIds.length ? `${selIds.length} word${selIds.length > 1 ? "s" : ""} selected` : "select words to cut them"}
        </span>
        <div className="ml-auto flex gap-1">
          <button
            type="button"
            disabled={selKept.length === 0}
            onClick={doDelete}
            className="rounded bg-rose-600 px-2 py-1 text-white disabled:opacity-30"
          >
            Delete <kbd className="opacity-70">Del</kbd>
          </button>
          <button
            type="button"
            disabled={selCut.length === 0}
            onClick={doRestore}
            className="rounded bg-emerald-600 px-2 py-1 text-white disabled:opacity-30"
          >
            Restore
          </button>
        </div>
      </div>
      <div ref={rootRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-[15px] leading-8">
        {project.units.map((u) => {
          if (u.kind === "pause") {
            const s = sugBy.get(`pause:${u.id}`);
            const gap = u.end - u.start;
            const take = u.features.is_take_boundary === 1;
            const action = s?.current_action ?? "KEEP";
            const removed = action === "REMOVE";
            return (
              <div key={u.id} className="relative my-1" data-testid={`pause-${u.id}`}>
                <button
                  type="button"
                  data-popover
                  aria-label={`Pause of ${fmtGap(gap)}, ${action}`}
                  onClick={() => {
                    setPopover(popover === u.id ? null : u.id);
                    if (s) onSelect(s);
                    onSeek(u.start);
                  }}
                  className={`inline-flex items-center rounded-full border px-2 text-xs ${
                    isSel("pause", u.id) ? "border-sky-400 bg-sky-500/20" : "border-white/15 hover:bg-white/10"
                  } ${removed ? "text-neutral-500 line-through" : "text-neutral-300"}`}
                >
                  <ConfidenceDot s={s} />
                  {take ? "✂ take join " : "⏸ "}
                  {fmtGap(gap)}
                  {s && action !== "KEEP" && <span className="ml-1 text-neutral-400">· {PAUSE_OPTIONS.find((o) => o.action === action)?.label}</span>}
                </button>
                {popover === u.id && (
                  <div data-popover role="menu" className="absolute left-0 top-7 z-20 w-44 rounded-lg border border-white/15 bg-neutral-800 py-1 shadow-xl">
                    {PAUSE_OPTIONS.map((o) => (
                      <button
                        key={o.action}
                        role="menuitem"
                        type="button"
                        onClick={() => {
                          setPopover(null);
                          if (o.action !== action) onEdit({ kind: "set_pause", unit_id: u.id, option: o.action });
                        }}
                        className={`block w-full px-3 py-1 text-left text-xs hover:bg-white/10 ${o.action === action ? "text-sky-300" : "text-neutral-200"}`}
                      >
                        {o.action === action ? "✓ " : ""}
                        {o.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          }
          if (u.kind !== "speech") return null;
          const ws = byUnit.get(u.id) ?? [];
          const s = sugBy.get(`phrase:${u.id}`);
          return (
            <p
              key={u.id}
              data-unit-id={u.id}
              className={`my-1 rounded px-1 ${isSel("phrase", u.id) ? "bg-sky-500/10 ring-1 ring-sky-400/60" : ""}`}
            >
              <ConfidenceDot s={s} />
              {ws.map((w, i) => (
                <span key={w.id}>
                  {i > 0 ? " " : null}
                  {renderWord(w)}
                </span>
              ))}
            </p>
          );
        })}
        {project.units.length === 0 && <p className="text-sm text-neutral-500">No transcript yet.</p>}
      </div>
      <div className="border-t border-white/10 px-3 py-1 text-[11px] text-neutral-500">
        Click a word to jump · double-click to fix its caption · <span className="underline decoration-dotted decoration-amber-400">dotted</span> = filler ·{" "}
        <span className="line-through">struck</span> = cut
      </div>
    </div>
  );
});
