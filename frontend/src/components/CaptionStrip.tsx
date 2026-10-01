import { useMemo } from "react";
import type { ManualEditIn, ProjectOut } from "../api-types";
import { chipsOf, emphasisEdit, lineAt } from "../lib/emphasis";

interface Props {
  project: ProjectOut;
  outTime: number;
  onEdit: (edit: ManualEditIn) => void;
}

/**
 * The words of the caption line under the playhead as chips (PLAN 8.2 / D5): click one to highlight it in the
 * captions, click the highlighted one to clear it. A dot marks the words the agent considered highlighting.
 */
export function CaptionStrip({ project, outTime, onEdit }: Props) {
  const edl = project.edl;
  const idx = edl ? lineAt(edl, outTime) : -1;
  const chips = useMemo(
    () => (edl && idx >= 0 ? chipsOf(edl, project.caption_word_ids, project.suggestions, idx) : []),
    [edl, idx, project.caption_word_ids, project.suggestions],
  );
  if (!edl || !edl.captions.enabled || chips.length === 0) return null;
  return (
    <div className="w-full" data-testid="caption-strip">
      <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-500">Click a caption word to highlight it</p>
      <div className="flex flex-wrap gap-1">
        {chips.map((c) => (
          <button
            key={c.wordId}
            type="button"
            aria-pressed={c.emphasis}
            title={c.candidate ? "the agent considered this word" : undefined}
            onClick={() => onEdit(emphasisEdit(chips, c))}
            className={`rounded px-1.5 py-0.5 text-xs ${c.emphasis ? "bg-yellow-400 font-semibold text-black" : "bg-white/10 text-neutral-200 hover:bg-white/20"}`}
          >
            {c.candidate && !c.emphasis && <span aria-hidden className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-amber-400 align-middle" />}
            {c.text}
          </button>
        ))}
      </div>
    </div>
  );
}
