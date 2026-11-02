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
  uploaded: "bg-neutral-950/70 text-neutral-200",
  processing: "bg-neutral-950/70 text-amber-200",
  ready: "bg-neutral-950/70 text-sky-200",
  edited: "bg-neutral-950/70 text-fuchsia-200",
  exported: "bg-neutral-950/70 text-emerald-200",
  failed: "bg-neutral-950/70 text-red-200",
};

const DOT: Record<ProjectSummary["status"], string> = {
  uploaded: "bg-neutral-400",
  processing: "bg-amber-400 animate-pulse",
  ready: "bg-sky-400",
  edited: "bg-fuchsia-400",
  exported: "bg-emerald-400",
  failed: "bg-red-400",
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
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="mb-6 flex items-end gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Projects</h1>
          <p className="mt-1 text-sm text-neutral-400">
            {projects && projects.length > 0
              ? `${projects.length} video${projects.length === 1 ? "" : "s"} · open one to review the first cut`
              : "Upload a talking-head clip and get a first cut in about a minute."}
          </p>
        </div>
        <Link to="/upload" className="ml-auto inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-lg shadow-sky-900/40 hover:bg-sky-500">
          <span aria-hidden="true" className="text-base leading-none">+</span> New project
        </Link>
      </div>
      {error && (
        <p role="alert" className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-sm text-red-200">
          {error}
        </p>
      )}
      {projects === null && !error && (
        <ul className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3" aria-hidden="true">
          {[0, 1, 2].map((k) => (
            <li key={k} className="rc-panel overflow-hidden">
              <div className="relative aspect-video overflow-hidden bg-white/[0.03]">
                <div className="rc-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/[0.05] to-transparent" />
              </div>
              <div className="space-y-2 p-4">
                <div className="h-3 w-2/3 rounded bg-white/[0.06]" />
                <div className="h-2.5 w-1/3 rounded bg-white/[0.04]" />
              </div>
            </li>
          ))}
        </ul>
      )}
      {projects?.length === 0 && (
        <div className="rc-panel flex flex-col items-center px-6 py-16 text-center">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-sky-500/10 text-xl text-sky-300 ring-1 ring-sky-400/20">▶</div>
          <p className="max-w-sm text-neutral-300">No projects yet. Upload a talking-head clip to get a first cut to review.</p>
          <Link to="/upload" className="mt-5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500">
            Upload a video
          </Link>
        </div>
      )}
      <ul className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {projects?.map((p, i) => (
          <li key={p.id} style={{ animationDelay: `${Math.min(i, 8) * 60}ms` }} className="group rc-panel rc-in overflow-hidden transition-[translate,border-color,box-shadow] duration-200 hover:-translate-y-0.5 hover:border-sky-400/40 hover:shadow-2xl hover:shadow-sky-950/50">
            <Link to={`/projects/${p.id}`} className="block">
              <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-neutral-900">
                {p.has_thumbnail ? (
                  <img src={thumbnailUrl(p.id)} alt="" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" loading="lazy" />
                ) : (
                  <span className="text-xs text-neutral-500">no preview yet</span>
                )}
                <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-neutral-950/80 via-transparent to-transparent" />
                {!(p.autopilot && p.status === "ready") && (
                  <span className={`absolute left-2.5 top-2.5 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-white/10 backdrop-blur ${BADGE[p.status]}`}>
                    <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${DOT[p.status]}`} />
                    {LABEL[p.status]}
                  </span>
                )}
                {p.duration > 0 && (
                  <span className="absolute bottom-2.5 right-2.5 rounded-md bg-neutral-950/75 px-1.5 py-0.5 font-mono text-[11px] text-neutral-200 ring-1 ring-white/10 backdrop-blur">
                    {fmtDuration(p.duration)}
                  </span>
                )}
              </div>
              <div className="px-4 py-3">
                <h2 className="truncate font-medium text-neutral-100" title={p.title}>
                  {p.title || p.video_id}
                </h2>
                <p className="mt-1 flex items-center gap-1.5 text-xs text-neutral-400">
                  <span className="capitalize">{p.video_type}</span>
                  {p.target_duration ? (
                    <span className="rounded-full bg-sky-500/10 px-1.5 py-px text-[10.5px] text-sky-300 ring-1 ring-sky-400/20">target {p.target_duration}s</span>
                  ) : null}
                </p>
                {p.status === "failed" && p.error && <p className="mt-1 truncate text-xs text-red-300" title={p.error}>{p.error}</p>}
              </div>
            </Link>
            {p.autopilot && (p.status === "ready" || p.status === "exported") && (
              <div className="flex items-center gap-2 border-t border-white/[0.06] px-4 py-2.5" data-testid={`autopilot-${p.id}`}>
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
