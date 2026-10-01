import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listProjects, thumbnailUrl } from "../api";
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

export default function ProjectsPage() {
  const { user } = useUser();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const userId = user?.id ?? null;
  const analysing = !!projects?.some((p) => p.status === "uploaded" || p.status === "processing");

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
          <li key={p.id}>
            <Link to={`/projects/${p.id}`} className="block overflow-hidden rounded-xl border border-white/10 bg-neutral-900 hover:border-sky-400/60">
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
                  <span className={`shrink-0 rounded px-1.5 py-0.5 text-[11px] ${BADGE[p.status]}`}>{LABEL[p.status]}</span>
                </div>
                <p className="mt-1 text-xs text-neutral-400">
                  {p.video_type} · {p.duration > 0 ? fmtDuration(p.duration) : "—"}
                  {p.target_duration ? ` · target ${p.target_duration}s` : ""}
                </p>
                {p.status === "failed" && p.error && <p className="mt-1 truncate text-xs text-red-300" title={p.error}>{p.error}</p>}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
