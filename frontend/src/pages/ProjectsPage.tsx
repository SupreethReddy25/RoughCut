import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { downloadUrl, getJob, listProjects, startRender, thumbnailUrl } from "../api";
import type { ProjectSummary } from "../api-types";
import { useUser } from "../hooks/useUser";
import { fmtDuration } from "../lib/format";

const LABEL: Record<ProjectSummary["status"], string> = {
  uploaded: "Queued",
  processing: "Analysing…",
  ready: "Ready to edit",
  edited: "Edited",
  exported: "Exported",
  failed: "Failed",
};

const BADGE: Record<ProjectSummary["status"], string> = {
  uploaded: "bg-neutral-500/30 text-neutral-200",
  processing: "bg-amber-500/25 text-amber-200",
  ready: "bg-sky-500/25 text-sky-200",
  edited: "bg-violet-500/25 text-violet-200",
  exported: "bg-emerald-500/25 text-emerald-200",
  failed: "bg-red-500/25 text-red-200",
};

const POLL_MS = 3000;

/** Still being prepared: analysing, or a batch video whose automatic edit is not there yet. */
export const isWorking = (p: ProjectSummary) =>
  p.status === "uploaded" || p.status === "processing" || (p.autopilot && p.status === "ready" && p.badge === null);

/** The badge of a batch (autopilot) video: ready to post, needs review (N), or still being auto-edited. */
export function AutopilotBadge({ p }: { p: ProjectSummary }) {
  if (!p.autopilot || p.status !== "ready") return null;
  if (p.badge === "ready") {
    return <span className="rounded bg-emerald-500/25 px-1.5 py-0.5 text-[11px] text-emerald-200">Ready to post</span>;
  }
  if (p.badge === "review") {
    return (
      <Link to={`/projects/${p.id}?review=1`} className="rounded bg-amber-500/25 px-1.5 py-0.5 text-[11px] text-amber-200 hover:bg-amber-500/40">
        Needs review ({p.n_review})
      </Link>
    );
  }
  return <span className="rounded bg-neutral-500/30 px-1.5 py-0.5 text-[11px] text-neutral-200">Auto-editing…</span>;
}

function Downloads({ id }: { id: number }) {
  return (
    <span className="flex gap-2 text-[11px]">
      <a href={downloadUrl(id, "mp4", "reels")} download className="text-sky-400 hover:underline">
        Download .mp4
      </a>
      <a href={downloadUrl(id, "srt", "reels")} download className="text-sky-400 hover:underline">
        .srt
      </a>
    </span>
  );
}

type Quick = { name: "idle" } | { name: "rendering" } | { name: "done" } | { name: "failed"; message: string };

/**
 * One-click export of a "Ready to post" video with its remembered defaults. It does NOT go through finalize: nothing
 * was reviewed, so nothing is learned from it (PLAN 8.8).
 */
export function QuickExport({ p, onDone }: { p: ProjectSummary; onDone: () => void }) {
  const [state, setState] = useState<Quick>({ name: "idle" });
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = async () => {
    setState({ name: "rendering" });
    try {
      const started = await startRender(p.id, "reels");
      let misses = 0;
      for (;;) {
        await new Promise((r) => setTimeout(r, 800));
        if (!alive.current) return;
        try {
          const j = await getJob(started.job.id);
          misses = 0;
          if (j.status === "done") break;
          if (j.status === "failed") throw new Error(j.error ?? "The render failed.");
        } catch (e) {
          if ((e as Error).message === "The render failed." || ++misses >= 5) throw e;
        }
      }
      if (!alive.current) return;
      setState({ name: "done" });
      onDone();
    } catch (e) {
      if (alive.current) setState({ name: "failed", message: (e as Error).message });
    }
  };

  if (state.name === "done") return <Downloads id={p.id} />;
  return (
    <span className="flex items-center gap-2">
      <button type="button" disabled={state.name === "rendering"} onClick={() => void run()} className="rounded bg-emerald-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-emerald-500 disabled:opacity-60">
        {state.name === "rendering" ? "Exporting…" : state.name === "failed" ? "Try again" : "Export now"}
      </button>
      {state.name === "failed" && (
        <span role="alert" className="truncate text-[11px] text-red-300" title={state.message}>
          {state.message}
        </span>
      )}
    </span>
  );
}

export default function ProjectsPage() {
  const { user } = useUser();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const userId = user?.id ?? null;
  const analysing = !!projects?.some(isWorking);
  const reloadList = () => {
    if (userId !== null) void listProjects(userId).then(setProjects).catch(() => undefined);
  };

  useEffect(() => {
    if (userId === null) return;
    let alive = true;
    const load = () =>
      listProjects(userId)
        .then((l) => {
          if (!alive) return;
          setProjects(l);
          setError(null);
        })
        .catch((e: Error) => alive && setError(e.message));
    setProjects(null);
    void load();
    return () => {
      alive = false;
    };
  }, [userId]);

  useEffect(() => {
    if (!analysing || userId === null) return;
    let alive = true;
    const t = setInterval(() => {
      listProjects(userId)
        .then((l) => alive && setProjects(l))
        .catch(() => undefined); // a missed poll is not worth an error: the next one will do
    }, POLL_MS);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [analysing, userId]);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <div className="mb-4 flex items-center">
        <h1 className="text-2xl font-semibold">Projects</h1>
        <Link to="/upload" className="ml-auto rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500">
          + New project
        </Link>
      </div>
      {error && (
        <p role="alert" className="mb-3 rounded bg-red-500/20 p-2 text-sm text-red-200">
          {error}
        </p>
      )}
      {projects === null && !error && <p className="text-neutral-400">Loading…</p>}
      {projects?.length === 0 && (
        <p className="rounded-lg border border-dashed border-white/20 p-10 text-center text-neutral-400">
          No projects yet. Upload a talking-head clip to get a first cut to review.
        </p>
      )}
      <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {projects?.map((p) => (
          <li key={p.id} className="overflow-hidden rounded-xl border border-white/10 bg-neutral-900 hover:border-sky-400/60">
            <Link to={`/projects/${p.id}`} className="block">
              <div className="flex aspect-video items-center justify-center bg-black">
                {p.has_thumbnail ? (
                  <img src={thumbnailUrl(p.id)} alt="" className="h-full w-full object-cover" loading="lazy" />
                ) : (
                  <span className="text-xs text-neutral-500">no preview yet</span>
                )}
              </div>
              <div className="p-3">
                <div className="flex items-center gap-2">
                  <h2 className="min-w-0 flex-1 truncate font-medium" title={p.title}>
                    {p.title || p.video_id}
                  </h2>
                  {!(p.autopilot && p.status === "ready") && (
                    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[11px] ${BADGE[p.status]}`}>{LABEL[p.status]}</span>
                  )}
                </div>
                <p className="mt-1 text-xs text-neutral-400">
                  {p.video_type} · {p.duration > 0 ? fmtDuration(p.duration) : "—"}
                  {p.target_duration ? ` · target ${p.target_duration}s` : ""}
                </p>
                {p.status === "failed" && p.error && <p className="mt-1 truncate text-xs text-red-300" title={p.error}>{p.error}</p>}
              </div>
            </Link>
            {p.autopilot && (p.status === "ready" || p.status === "exported") && (
              <div className="flex items-center gap-2 border-t border-white/10 px-3 py-2" data-testid={`autopilot-${p.id}`}>
                {p.status === "exported" && <Downloads id={p.id} />}
                <AutopilotBadge p={p} />
                {p.badge === "ready" && <QuickExport p={p} onDone={reloadList} />}
                {p.badge === "review" && p.expected_corrections !== null && (
                  <span className="text-[11px] text-neutral-500">~{p.expected_corrections.toFixed(1)} fixes expected</span>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
