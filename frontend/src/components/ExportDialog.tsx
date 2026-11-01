import { useEffect, useRef, useState } from "react";
import { downloadUrl, getJob, startRender } from "../api";
import type { StylesOut } from "../api-types";
import { fmtDuration } from "../lib/format";

interface Props {
  projectId: number;
  styles: StylesOut | null;
  defaultPreset: string;
  /** record the session in the learner (implicit accepts, effort metrics) before the export is made */
  onBeforeRender: () => Promise<boolean>;
  /** after a render finished (the project's export list changed) */
  onRendered: () => void;
  /** the one-tap verdict on the finished video: the rhythm agent's delayed reward */
  onRate?: (value: -1 | 0 | 1) => void;
  rating?: number | null;
  onClose: () => void;
}

type Phase =
  | { name: "idle" }
  | { name: "saving" }
  | { name: "rendering"; preset: string }
  | { name: "done"; preset: string; duration: number; warning: string | null }
  | { name: "failed"; message: string };

const POLL_MS = 800;

/** Finalize + render + download (PLAN 12): "without leaving the app". */
export function ExportDialog({ projectId, styles, defaultPreset, onBeforeRender, onRendered, onRate, rating, onClose }: Props) {
  const [preset, setPreset] = useState(defaultPreset);
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const busy = phase.name === "saving" || phase.name === "rendering";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !busy && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const run = async () => {
    setPhase({ name: "saving" });
    const ok = await onBeforeRender();
    if (!alive.current) return;
    if (!ok) {
      setPhase({ name: "failed", message: "Could not save your session before exporting. Nothing was rendered." });
      return;
    }
    let started;
    try {
      started = await startRender(projectId, preset);
    } catch (e) {
      if (alive.current) setPhase({ name: "failed", message: (e as Error).message });
      return;
    }
    if (!alive.current) return;
    setPhase({ name: "rendering", preset });
    const jobId = started.job.id;
    // poll until the job ends; a vanished server shows as an error rather than spinning forever
    let misses = 0;
    while (alive.current) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      if (!alive.current) return;
      try {
        const j = await getJob(jobId);
        misses = 0;
        if (j.status === "done") {
          setPhase({ name: "done", preset, duration: started.duration, warning: started.warning });
          onRendered();
          return;
        }
        if (j.status === "failed") {
          setPhase({ name: "failed", message: j.error ?? "The render failed." });
          return;
        }
      } catch (e) {
        if (++misses >= 5) {
          setPhase({ name: "failed", message: (e as Error).message });
          return;
        }
      }
    }
  };

  const presets = styles ? Object.entries(styles.presets) : [];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Export">
      <div className="rc-pop w-full max-w-md rounded-2xl bg-neutral-900 p-5 text-neutral-100 shadow-2xl shadow-black/60 ring-1 ring-white/10">
        <h2 className="mb-3 text-lg font-semibold">Export Reel</h2>
        {(phase.name === "idle" || phase.name === "failed") && (
          <>
            <fieldset className="mb-3 space-y-1.5">
              <legend className="mb-1 text-xs uppercase tracking-wide text-neutral-400">Format</legend>
              {presets.map(([id, p]) => (
                <label key={id} className="flex items-center gap-2 rounded border border-white/10 px-2 py-1.5 text-sm">
                  <input type="radio" name="preset" checked={preset === id} onChange={() => setPreset(id)} />
                  <span className="flex-1">{p.label}</span>
                  <span className="text-xs text-neutral-400">
                    {p.width}×{p.height} · max {p.max_duration_s}s
                  </span>
                </label>
              ))}
            </fieldset>
            {phase.name === "failed" && (
              <p role="alert" className="mb-3 rounded bg-red-500/20 p-2 text-sm text-red-200">
                {phase.message}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button type="button" onClick={onClose} className="rounded px-3 py-1.5 text-sm hover:bg-white/10">
                Cancel
              </button>
              <button type="button" onClick={run} className="rounded bg-sky-600 px-3 py-1.5 text-sm font-medium hover:bg-sky-500">
                {phase.name === "failed" ? "Try again" : "Render"}
              </button>
            </div>
          </>
        )}
        {busy && (
          <p role="status" className="py-6 text-center text-sm text-neutral-300">
            {phase.name === "saving" ? "Saving your edits…" : "Rendering your Reel… this can take a minute."}
          </p>
        )}
        {phase.name === "done" && (
          <>
            <p role="status" className="mb-2 text-sm text-emerald-300">
              Done: {fmtDuration(phase.duration)} long.
            </p>
            {phase.warning && <p className="mb-2 rounded bg-amber-500/20 p-2 text-sm text-amber-200">{phase.warning}</p>}
            {onRate && (
              <div className="mb-3 rounded-lg bg-white/[0.04] p-2.5 ring-1 ring-white/[0.06]" data-testid="export-rating">
                <p className="mb-1.5 text-xs text-neutral-400">How does this edit feel? It teaches your agent the rhythm you like.</p>
                <div className="flex gap-1.5">
                  {([[1, "👍", "Great"], [0, "😐", "Okay"], [-1, "👎", "Not me"]] as const).map(([v, icon, label]) => (
                    <button key={v} type="button" aria-pressed={rating === v} onClick={() => onRate(v)} className={`rc-press flex-1 rounded-md px-2 py-1.5 text-sm ring-1 ${rating === v ? "bg-sky-500/25 text-sky-100 ring-sky-400/50" : "bg-white/[0.04] text-neutral-300 ring-white/10 hover:bg-white/[0.08]"}`}>
                      <span aria-hidden="true">{icon}</span> {label}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex flex-wrap justify-end gap-2">
              <a
                href={downloadUrl(projectId, "srt", phase.preset)}
                download
                className="rounded border border-white/20 px-3 py-1.5 text-sm hover:bg-white/10"
              >
                Download captions (.srt)
              </a>
              <a href={downloadUrl(projectId, "mp4", phase.preset)} download className="rounded bg-sky-600 px-3 py-1.5 text-sm font-medium hover:bg-sky-500">
                Download video (.mp4)
              </a>
              <button type="button" onClick={onClose} className="rounded px-3 py-1.5 text-sm hover:bg-white/10">
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
