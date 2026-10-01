import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createProject } from "../api";
import { useUser } from "../hooks/useUser";
import { moveItem, parseTarget, validateFiles } from "../lib/upload";
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
  const [type, setType] = useState<VideoType>("other");
  const [target, setTarget] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    setFiles((cur) => [...cur, ...Array.from(list)]);
    setError(null);
    if (input.current) input.current.value = ""; // so choosing the same file again fires change
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || !user) return;
    const problem = validateFiles(files);
    const t = parseTarget(target);
    if (problem || "error" in t) {
      setError(problem ?? ("error" in t ? t.error : null));
      return;
    }
    const form = new FormData();
    form.append("user_id", String(user.id));
    form.append("type", type);
    form.append("title", title.trim());
    if (t.value !== null) form.append("target", String(t.value));
    for (const f of files) form.append("files", f, f.name); // in the order shown: that is the order they are joined in
    setBusy(true);
    setError(null);
    try {
      const p = await createProject(form);
      nav(`/projects/${p.id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="mx-auto max-w-2xl space-y-5 p-6" aria-label="New project">
      <div className="flex items-center">
        <h1 className="text-2xl font-semibold">New project</h1>
        <Link to="/" className="ml-auto text-sm text-sky-400 hover:underline">
          Cancel
        </Link>
      </div>

      <section>
        <h2 className="mb-1 text-sm font-medium text-neutral-300">Video clips</h2>
        <p className="mb-2 text-xs text-neutral-500">Several takes are joined in the order shown (3 minutes in total at most).</p>
        <input ref={input} type="file" accept="video/*,.mkv,.mov" multiple className="hidden" aria-label="Choose video files" onChange={(e) => addFiles(e.target.files)} />
        <button type="button" onClick={() => input.current?.click()} className="w-full rounded-lg border border-dashed border-white/25 p-5 text-sm text-neutral-300 hover:bg-white/5">
          {files.length === 0 ? "Choose video files…" : "Add more files…"}
        </button>
        {files.length > 0 && (
          <ol className="mt-2 space-y-1" aria-label="Files in order">
            {files.map((f, i) => (
              <li key={`${f.name}-${f.size}-${i}`} className="flex items-center gap-2 rounded border border-white/10 px-2 py-1 text-sm">
                <span className="w-5 text-xs text-neutral-500">{i + 1}</span>
                <span className="min-w-0 flex-1 truncate" title={f.name}>
                  {f.name}
                </span>
                <span className="text-xs text-neutral-500">{mb(f.size)}</span>
                <button type="button" aria-label={`Move ${f.name} up`} disabled={i === 0} onClick={() => setFiles((l) => moveItem(l, i, -1))} className="rounded px-1.5 hover:bg-white/10 disabled:opacity-30">
                  ↑
                </button>
                <button type="button" aria-label={`Move ${f.name} down`} disabled={i === files.length - 1} onClick={() => setFiles((l) => moveItem(l, i, 1))} className="rounded px-1.5 hover:bg-white/10 disabled:opacity-30">
                  ↓
                </button>
                <button type="button" aria-label={`Remove ${f.name}`} onClick={() => setFiles((l) => l.filter((_, k) => k !== i))} className="rounded px-1.5 text-rose-300 hover:bg-white/10">
                  ×
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>

      <fieldset>
        <legend className="mb-1 text-sm font-medium text-neutral-300">What kind of video is it?</legend>
        <div className="grid grid-cols-2 gap-2">
          {TYPES.map((t) => (
            <label key={t.id} className={`cursor-pointer rounded-lg border px-3 py-2 text-sm ${type === t.id ? "border-sky-400 bg-sky-500/10" : "border-white/15 hover:bg-white/5"}`}>
              <input type="radio" name="type" className="sr-only" checked={type === t.id} onChange={() => setType(t.id)} />
              <div className="font-medium">{t.label}</div>
              <div className="text-xs text-neutral-400">{t.hint}</div>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block font-medium text-neutral-300">Target length (seconds, optional)</span>
          <input value={target} onChange={(e) => setTarget(e.target.value)} inputMode="numeric" placeholder="e.g. 30" className="w-full rounded bg-neutral-800 px-2 py-1.5" />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium text-neutral-300">Title (optional)</span>
          <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} placeholder="defaults to the file name" className="w-full rounded bg-neutral-800 px-2 py-1.5" />
        </label>
      </div>

      {error && (
        <p role="alert" className="rounded bg-red-500/20 p-2 text-sm text-red-200">
          {error}
        </p>
      )}
      <button type="submit" disabled={busy || files.length === 0} className="w-full rounded bg-sky-600 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-50">
        {busy ? "Uploading…" : "Upload and analyse"}
      </button>
    </form>
  );
}
