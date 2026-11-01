import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createProject } from "../api";
import { useUser } from "../hooks/useUser";
import { BATCH_MAX, moveItem, parseTarget, validateFiles } from "../lib/upload";
import type { VideoType } from "../types";

const TYPES: { id: VideoType; label: string; hint: string }[] = [
  { id: "tutorial", label: "Tutorial", hint: "explaining how to do something" },
  { id: "vlog", label: "Vlog", hint: "telling a story or your day" },
  { id: "opinion", label: "Opinion", hint: "a take or a rant" },
  { id: "other", label: "Other", hint: "anything else" },
];

const mb = (n: number) => `${(n / 2 ** 20).toFixed(1)} MB`;

export default function UploadPage() {
  const { user } = useUser();
  const nav = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [batch, setBatch] = useState(false);
  const [type, setType] = useState<VideoType>("other");
  const [target, setTarget] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(0);
  const [dragging, setDragging] = useState(false);

  const addFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    setFiles((cur) => [...cur, ...Array.from(list)]);
    setError(null);
    if (input.current) input.current.value = ""; // so choosing the same file again fires change
  };

  const form = (fs: File[], autopilot: boolean, name: string, t: number | null) => {
    const f = new FormData();
    f.append("user_id", String(user!.id));
    f.append("type", type);
    f.append("title", name);
    if (t !== null) f.append("target", String(t));
    if (autopilot) f.append("autopilot", "true");
    for (const file of fs) f.append("files", file, file.name); // in the order shown: that is the order they are joined in
    return f;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || !user) return;
    const problem = validateFiles(files, batch ? BATCH_MAX : undefined);
    const t = parseTarget(target);
    if (problem || "error" in t) {
      setError(problem ?? ("error" in t ? t.error : null));
      return;
    }
    setBusy(true);
    setError(null);
    setUploaded(0);
    if (!batch) {
      try {
        const p = await createProject(form(files, false, title.trim(), t.value));
        nav(`/projects/${p.id}`);
      } catch (err) {
        setError((err as Error).message);
        setBusy(false);
      }
      return;
    }
    // batch: every file is its own video, edited automatically with what the agent has learned about the user
    const failed: { file: File; reason: string }[] = [];
    let ok = 0;
    for (const [i, file] of files.entries()) {
      setProgress(`Uploading ${i + 1} of ${files.length}: ${file.name}`);
      try {
        await createProject(form([file], true, "", t.value));
        ok++;
      } catch (err) {
        failed.push({ file, reason: (err as Error).message });
      }
    }
    setProgress(null);
    setBusy(false);
    if (failed.length === 0) return nav("/");
    setUploaded((n) => n + ok);
    setFiles(failed.map((f) => f.file)); // what is left to retry
    setError(`${ok} of ${files.length} uploaded. ${failed.map((f) => `${f.file.name}: ${f.reason}`).join(" ")}`);
  };

  return (
    <form onSubmit={submit} className="rc-in mx-auto max-w-2xl space-y-6 px-6 py-8" aria-label="New project">
      <div className="flex items-end">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{batch ? "Batch upload" : "New project"}</h1>
          <p className="mt-1 text-sm text-neutral-400">Your agent makes the first cut; you only check what it is unsure about.</p>
        </div>
        <Link to="/" className="ml-auto text-sm text-neutral-400 hover:text-white">
          {uploaded > 0 ? "Go to projects" : "Cancel"}
        </Link>
      </div>

      <fieldset>
        <legend className="rc-label mb-2">What are you uploading?</legend>
        <div className="grid grid-cols-2 gap-2">
          {[
            { id: false, label: "One video", hint: "several takes are joined in order" },
            { id: true, label: `Batch (up to ${BATCH_MAX})`, hint: "each file is edited automatically; you review or post" },
          ].map((m) => (
            <label key={String(m.id)} className={`rc-lift cursor-pointer rounded-xl border px-3.5 py-2.5 text-sm ${batch === m.id ? "border-sky-400/70 bg-sky-500/10 shadow-lg shadow-sky-950/40" : "border-white/[0.08] bg-white/[0.02] hover:border-white/20"}`}>
              <input type="radio" name="mode" className="sr-only" checked={batch === m.id} onChange={() => setBatch(m.id)} />
              <div className="font-medium">{m.label}</div>
              <div className="text-xs text-neutral-400">{m.hint}</div>
            </label>
          ))}
        </div>
      </fieldset>

      <section>
        <h2 className="rc-label mb-1">Video clips</h2>
        <p className="mb-2 text-xs text-neutral-500">
          {batch ? `Up to ${BATCH_MAX} videos, 5 minutes each.` : "Several takes are joined in the order shown (5 minutes in total at most)."}
        </p>
        <input ref={input} type="file" accept="video/*,.mkv,.mov" multiple className="hidden" aria-label="Choose video files" onChange={(e) => addFiles(e.target.files)} />
        <button
          type="button"
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`group flex w-full flex-col items-center gap-2 rounded-2xl border border-dashed px-5 py-8 text-sm transition-colors ${
            dragging ? "border-sky-400 bg-sky-500/10 text-sky-100" : "border-white/15 bg-white/[0.015] text-neutral-300 hover:border-sky-400/50 hover:bg-sky-500/[0.05]"
          }`}
        >
          <span aria-hidden="true" className={`flex h-11 w-11 items-center justify-center rounded-xl bg-sky-500/10 text-lg text-sky-300 ring-1 ring-sky-400/20 transition-transform group-hover:-translate-y-0.5 ${dragging ? "scale-110" : ""}`}>
            ↑
          </span>
          <span className="font-medium">{files.length === 0 ? "Choose video files…" : "Add more files…"}</span>
          <span className="text-xs text-neutral-500">or drop them here · mp4, mov, mkv</span>
        </button>
        {files.length > 0 && (
          <ol className="mt-2 space-y-1" aria-label="Files in order">
            {files.map((f, i) => (
              <li key={`${f.name}-${f.size}-${i}`} className="rc-in flex items-center gap-2 rounded-lg border border-white/[0.07] bg-white/[0.02] px-2.5 py-1.5 text-sm">
                <span className="flex h-5 w-5 items-center justify-center rounded-md bg-white/[0.06] font-mono text-[10px] text-neutral-400">{i + 1}</span>
                <span className="min-w-0 flex-1 truncate" title={f.name}>
                  {f.name}
                </span>
                <span className="text-xs text-neutral-500">{mb(f.size)}</span>
                {!batch && (
                  <>
                    <button type="button" aria-label={`Move ${f.name} up`} disabled={i === 0} onClick={() => setFiles((l) => moveItem(l, i, -1))} className="rounded px-1.5 hover:bg-white/10 disabled:opacity-30">
                      ↑
                    </button>
                    <button type="button" aria-label={`Move ${f.name} down`} disabled={i === files.length - 1} onClick={() => setFiles((l) => moveItem(l, i, 1))} className="rounded px-1.5 hover:bg-white/10 disabled:opacity-30">
                      ↓
                    </button>
                  </>
                )}
                <button type="button" aria-label={`Remove ${f.name}`} onClick={() => setFiles((l) => l.filter((_, k) => k !== i))} className="rounded px-1.5 text-rose-300 hover:bg-white/10">
                  ×
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>

      <fieldset>
        <legend className="rc-label mb-2">What kind of video is it?</legend>
        <div className="grid grid-cols-2 gap-2">
          {TYPES.map((t) => (
            <label key={t.id} className={`rc-lift cursor-pointer rounded-xl border px-3.5 py-2.5 text-sm ${type === t.id ? "border-sky-400/70 bg-sky-500/10 shadow-lg shadow-sky-950/40" : "border-white/[0.08] bg-white/[0.02] hover:border-white/20"}`}>
              <input type="radio" name="type" className="sr-only" checked={type === t.id} onChange={() => setType(t.id)} />
              <div className="font-medium">{t.label}</div>
              <div className="text-xs text-neutral-400">{t.hint}</div>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="rc-label mb-1.5 block">Target length (seconds, optional)</span>
          <input value={target} onChange={(e) => setTarget(e.target.value)} inputMode="numeric" placeholder="e.g. 30" className="w-full rounded-lg bg-neutral-900 px-3 py-2" />
          <span className="mt-1 block text-[11px] text-neutral-500">The keep/cut plan is fitted to this length.</span>
        </label>
        {!batch && (
          <label className="block text-sm">
            <span className="rc-label mb-1.5 block">Title (optional)</span>
            <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} placeholder="defaults to the file name" className="w-full rounded-lg bg-neutral-900 px-3 py-2" />
          </label>
        )}
      </div>

      {error && (
        <p role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-2.5 text-sm text-red-200">
          {error}
        </p>
      )}
      {progress && (
        <p role="status" className="text-sm text-neutral-300">
          {progress}
        </p>
      )}
      <button type="submit" disabled={busy || files.length === 0} className="rc-press w-full rounded-xl bg-gradient-to-b from-sky-500 to-sky-600 py-2.5 font-medium text-white shadow-lg shadow-sky-900/50 ring-1 ring-white/10 hover:from-sky-400 hover:to-sky-500 disabled:opacity-40 disabled:shadow-none">
        {busy ? "Uploading…" : batch ? "Upload and auto-edit" : "Upload and analyse"}
      </button>
    </form>
  );
}
