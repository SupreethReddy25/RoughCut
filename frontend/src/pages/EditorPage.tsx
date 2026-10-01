import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { ManualEditIn, ProjectOut, SuggestionOut } from "../api-types";
import { CaptionStrip } from "../components/CaptionStrip";
import { CaptionStylePanel } from "../components/CaptionStylePanel";
import { ExportDialog } from "../components/ExportDialog";
import { MusicPanel } from "../components/MusicPanel";
import { MusicPreview } from "../components/MusicPreview";
import { PreviewPlayer, type PlayerHandle, type TimeInfo } from "../components/PreviewPlayer";
import { ReviewPanel, WhyBox } from "../components/ReviewPanel";
import { Timeline } from "../components/Timeline";
import { TranscriptEditor, type Sel, type TranscriptHandle } from "../components/TranscriptEditor";
import { useActiveSeconds } from "../hooks/useActiveSeconds";
import { useProject, type ProjectApi } from "../hooks/useProject";
import { useFonts, useStyles } from "../hooks/useStyles";
import { useUser } from "../hooks/useUser";
import { acceptEvent, chooseEvent, keyOf, nextInQueue, rejectEvent, sameKey } from "../lib/decide";
import { actionsFor, targetStart } from "../lib/describe";
import { fmtTime } from "../lib/format";
import { isInteractiveTarget, isTypingTarget, resolveShortcut } from "../lib/shortcuts";

const PRESET = "reels";
const POLL_MS = 2000;

type Ready = ProjectOut & { edl: NonNullable<ProjectOut["edl"]> };

export default function EditorPage() {
  const id = Number(useParams().id);
  if (!Number.isInteger(id) || id <= 0) return <Notice title="Project not found" />;
  return <Editor key={id} id={id} />;
}

function Notice({ title, text, children }: { title: string; text?: string; children?: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-lg p-10 text-center text-neutral-100">
      <h1 className="text-xl font-semibold">{title}</h1>
      {text && <p className="mt-2 text-sm text-neutral-400">{text}</p>}
      <div className="mt-4">{children}</div>
      <Link to="/" className="mt-4 inline-block text-sm text-sky-400 hover:underline">
        ← All projects
      </Link>
    </div>
  );
}

function Editor({ id }: { id: number }) {
  const api = useProject(id);
  const { project, loadError, reload, suggest } = api;

  // while the analysis runs keep asking; once it is done make the suggestions (once)
  const status = project?.status;
  useEffect(() => {
    if (status !== "uploaded" && status !== "processing") return;
    const t = setInterval(() => void reload(), POLL_MS);
    return () => clearInterval(t);
  }, [status, reload]);
  // an analysed project without suggestions gets them made (once per visit; the server is idempotent)
  const started = useRef(false);
  const [prepared, setPrepared] = useState(false);
  const needsSuggest = !!project && project.status === "ready" && project.suggestions.length === 0;
  useEffect(() => {
    if (!needsSuggest || started.current) return;
    started.current = true;
    void suggest().finally(() => setPrepared(true));
  }, [needsSuggest, suggest]);

  if (loadError) {
    return (
      <Notice title="Could not open this project" text={loadError}>
        <button type="button" onClick={() => void reload()} className="rounded bg-sky-600 px-3 py-1.5 text-sm text-white hover:bg-sky-500">
          Try again
        </button>
      </Notice>
    );
  }
  if (!project) return <p className="p-10 text-center text-neutral-400">Loading…</p>;
  if (project.status === "failed") {
    return <Notice title="Processing failed" text={project.error ?? "The video could not be analysed."} />;
  }
  if (project.status === "uploaded" || project.status === "processing") {
    return (
      <Notice title="Analysing your video…" text="Transcribing, finding pauses and filler words. This takes a little while.">
        <div className="mx-auto h-1.5 w-48 overflow-hidden rounded bg-white/10">
          <div className="h-full w-1/3 animate-pulse rounded bg-sky-500" />
        </div>
      </Notice>
    );
  }
  if (needsSuggest && !prepared) return <Notice title="Preparing suggestions…" />;
  if (!project.edl) return <Notice title="Nothing to edit yet" text="The analysis of this project has no usable result." />;
  return <Workspace api={api} project={project as Ready} />;
}

function Workspace({ api, project }: { api: ProjectApi; project: Ready }) {
  const { styles } = useStyles();
  const fontsReady = useFonts(styles);
  const activeSeconds = useActiveSeconds();
  const player = useRef<PlayerHandle>(null);
  const transcript = useRef<TranscriptHandle>(null);
  const [time, setTime] = useState<TimeInfo>({ src: 0, out: 0, playing: false, clip: 0 });
  const [sel, setSel] = useState<Sel | null>(null);
  const [selText, setSelText] = useState<string | null>(null);
  const { user } = useUser();
  const [params, setParams] = useSearchParams();
  const onlyUncertain = params.get("review") === "1"; // opened from a batch video's "Needs review" badge
  const [tab, setTab] = useState<"review" | "style" | "music">("review");
  const [showSafe, setShowSafe] = useState(false);
  const [reframe, setReframe] = useState(false);
  const [exporting, setExporting] = useState(false);

  const units = useMemo(() => new Map(project.units.map((u) => [u.id, u])), [project.units]);
  const words = useMemo(() => new Map(project.words.map((w) => [w.id, w])), [project.words]);
  const selected: SuggestionOut | null = useMemo(() => project.suggestions.find((s) => sameKey(s, sel)) ?? null, [project.suggestions, sel]);

  // the global key handler reads the latest values from here
  const live = useRef({ api, project, selected, sel, selText, exporting, units, words });
  live.current = { api, project, selected, sel, selText, exporting, units, words };

  const select = useCallback((s: SuggestionOut) => {
    setSel(keyOf(s));
    setSelText(null);
    const t = targetStart(s, live.current.units, live.current.words);
    if (t !== null) player.current?.seekSrc(t);
  }, []);

  const { send, markSeen } = api;
  const onEdit = useCallback((edit: ManualEditIn) => void send({ kind: "manual", edit }), [send]);
  const onCaptionText = useCallback((word_id: number, text: string) => void send({ kind: "caption_text", payload: { word_id, text } }), [send]);
  const onSetting = useCallback((key: string, value: unknown) => void send({ kind: "setting", payload: { key, value } }), [send]);
  const onSpeed = useCallback((value: number) => void send({ kind: "speed", payload: { value } }), [send]);
  const onOverlay = useCallback((payload: Record<string, unknown>) => void send({ kind: "text_overlay", payload }), [send]);
  const onMusic = useCallback((payload: Record<string, unknown>) => void send({ kind: "music", payload }), [send]);
  const onNudge = useCallback((name: string) => void send({ kind: "nudge", payload: { name, magnitude: 2 } }), [send]);
  const onOverlayChange = useCallback(
    (id: string, t0: number, t1: number) => void send({ kind: "text_overlay", payload: { op: "update", id, t0, t1 } }),
    [send],
  );
  const onCrop = useCallback((unit_id: number, x: number, y: number) => void send({ kind: "crop", payload: { unit_id, x, y } }), [send]);
  const onSeekOut = useCallback((o: number) => player.current?.seekOut(o), []);
  const onSeekSrc = useCallback((t: number) => player.current?.seekSrc(t), []);

  // a text overlay that no longer exists cannot stay selected
  useEffect(() => {
    if (selText && !project.edl.texts.some((t) => t.id === selText)) setSelText(null);
  }, [project.edl.texts, selText]);

  // ---- keyboard (PLAN Phase 4)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const L = live.current;
      if (L.exporting) return;
      const act = resolveShortcut({
        key: e.key,
        ctrlKey: e.ctrlKey,
        metaKey: e.metaKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        repeat: e.repeat,
        typing: isTypingTarget(e.target),
        interactive: isInteractiveTarget(e.target),
      });
      if (!act) return;
      const p = player.current;
      const dur = L.project.edl.duration;
      switch (act.type) {
        case "play":
          p?.toggle();
          break;
        case "seek":
          if (p) p.seekOut(Math.min(Math.max(p.getOutTime() + act.delta, 0), dur));
          break;
        case "accept": {
          const ev = acceptEvent(L.selected);
          if (ev) void L.api.send(ev);
          break;
        }
        case "reject": {
          const ev = rejectEvent(L.selected);
          if (ev) void L.api.send(ev);
          break;
        }
        case "choose": {
          const ev = L.selected ? chooseEvent(L.selected, actionsFor(L.selected)[act.index]) : null;
          if (ev) void L.api.send(ev);
          break;
        }
        case "next": {
          const n = nextInQueue(L.project.suggestions, L.project.queue, L.sel);
          if (n) select(n);
          break;
        }
        case "delete":
          if (transcript.current?.deleteSelection()) break;
          if (L.selText) {
            void L.api.send({ kind: "text_overlay", payload: { op: "delete", id: L.selText } });
            break;
          }
          return; // nothing to delete: leave the key alone
        case "undo":
          if (L.project.can_undo) void L.api.undo();
          break;
        case "redo":
          if (L.project.can_redo) void L.api.redo();
          break;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [select]);

  // A button that was clicked with the mouse must not keep the focus, or Space would press it again instead of
  // playing. (detail is 0 for a keyboard activation, which keeps its focus.)
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.detail === 0) return;
      const b = (e.target as HTMLElement | null)?.closest?.("button");
      if (b && !b.closest("[role=dialog]")) b.blur();
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  const beforeRender = useCallback(async () => (await live.current.api.finalize(activeSeconds())) !== null, [activeSeconds]);
  const nPending = project.suggestions.filter((s) => s.status === "pending").length;

  return (
    <div className="flex h-screen flex-col bg-neutral-950 text-neutral-100">
      <header className="flex flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2 text-sm">
        <Link to="/" className="text-sky-400 hover:underline">
          ← Projects
        </Link>
        <h1 className="mr-2 max-w-xs truncate font-semibold" title={project.title}>
          {project.title || project.video_id}
        </h1>
        <span className="text-xs text-neutral-400">
          {fmtTime(time.out)} / {fmtTime(project.edl.duration)} · {nPending} to review
          {api.busy > 0 && <span className="ml-2 text-amber-300">saving…</span>}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <button type="button" onClick={() => void api.undo()} disabled={!project.can_undo} title="Undo (Ctrl+Z)" className="rounded border border-white/15 px-2 py-1 hover:bg-white/10 disabled:opacity-40">
            ↶ Undo
          </button>
          <button type="button" onClick={() => void api.redo()} disabled={!project.can_redo} title="Redo (Ctrl+Shift+Z)" className="rounded border border-white/15 px-2 py-1 hover:bg-white/10 disabled:opacity-40">
            ↷ Redo
          </button>
          <label className="ml-2 flex items-center gap-1 text-xs text-neutral-300">
            <input type="checkbox" checked={showSafe} onChange={(e) => setShowSafe(e.target.checked)} /> Safe zones
          </label>
          <label className="flex items-center gap-1 text-xs text-neutral-300">
            <input type="checkbox" checked={reframe} onChange={(e) => setReframe(e.target.checked)} /> Reframe
          </label>
          <button
            type="button"
            onClick={() => {
              player.current?.pause();
              setExporting(true);
            }}
            disabled={project.edl.clips.length === 0}
            className="ml-2 rounded bg-sky-600 px-3 py-1 font-medium hover:bg-sky-500 disabled:opacity-40"
          >
            Export…
          </button>
        </div>
      </header>

      {(api.error || api.notice) && (
        <div role={api.error ? "alert" : "status"} className={`flex items-center gap-2 px-3 py-1 text-xs ${api.error ? "bg-red-500/20 text-red-200" : "bg-sky-500/15 text-sky-200"}`}>
          <span className="flex-1">{api.error ?? api.notice}</span>
          <button type="button" aria-label="Dismiss" onClick={api.dismiss} className="px-1 hover:text-white">
            ×
          </button>
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(220px,300px)_minmax(0,1fr)_minmax(280px,340px)]">
        <div className="flex min-h-0 flex-col items-center gap-2 overflow-y-auto border-r border-white/10 p-3">
          <div className="relative min-h-0 w-full flex-1">
            <PreviewPlayer
              ref={player}
              project={project}
              styles={styles}
              presetId={PRESET}
              showSafe={showSafe}
              reframe={reframe}
              onTime={setTime}
              onSeen={markSeen}
              onCrop={onCrop}
            />
          </div>
          <button type="button" onClick={() => player.current?.toggle()} className="rounded bg-white/10 px-4 py-1 text-sm hover:bg-white/20">
            {time.playing ? "Pause" : "Play"} <kbd className="text-[10px] opacity-60">Space</kbd>
          </button>
          {!fontsReady && <p className="text-[10px] text-neutral-500">Loading caption fonts…</p>}
          <CaptionStrip project={project} outTime={time.out} onEdit={onEdit} />
          {user && <MusicPreview edl={project.edl} userId={user.id} time={time} />}
        </div>

        <div className="min-h-0 overflow-y-auto">
          <TranscriptEditor
            ref={transcript}
            project={project}
            srcTime={time.src}
            playing={time.playing}
            selected={sel}
            onSelect={select}
            onSeek={onSeekSrc}
            onEdit={onEdit}
            onCaptionText={onCaptionText}
          />
        </div>

        <div className="flex min-h-0 flex-col border-l border-white/10">
          <div className="flex border-b border-white/10 text-sm" role="tablist">
            {(["review", "style", "music"] as const).map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={`flex-1 px-3 py-1.5 ${tab === t ? "border-b-2 border-sky-400 text-white" : "text-neutral-400 hover:text-neutral-200"}`}
              >
                {t === "review" ? "Review" : t === "style" ? "Style" : "Music"}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-hidden">
            {tab === "review" ? (
              <ReviewPanel
                project={project}
                selected={sel}
                fresh={api.fresh}
                onNudge={onNudge}
                onlyUncertain={onlyUncertain}
                onShowAll={() => setParams({}, { replace: true })}
                onSelect={select}
                onAccept={(s) => {
                  const ev = acceptEvent(s);
                  if (ev) void api.send(ev);
                }}
                onReject={(s) => {
                  const ev = rejectEvent(s);
                  if (ev) void api.send(ev);
                }}
                onChoose={(s, a) => {
                  const ev = chooseEvent(s, a);
                  if (ev) void api.send(ev);
                }}
              />
            ) : tab === "music" ? (
              <div className="h-full overflow-y-auto p-3">{user && <MusicPanel edl={project.edl} userId={user.id} onMusic={onMusic} />}</div>
            ) : (
              <CaptionStylePanel
                project={project}
                styles={styles}
                outTime={time.out}
                selectedText={selText}
                onSelectText={setSelText}
                onSetting={onSetting}
                onSpeed={onSpeed}
                onOverlay={onOverlay}
              />
            )}
          </div>
          {tab === "review" && <WhyBox project={project} suggestion={selected} />}
        </div>
      </div>

      <Timeline
        project={project}
        outTime={time.out}
        playing={time.playing}
        selected={sel}
        selectedText={selText}
        onSeekOut={onSeekOut}
        onSelect={select}
        onSelectText={(t) => {
          setSelText(t);
          if (t) setTab("style");
        }}
        onOverlayChange={onOverlayChange}
      />

      {exporting && (
        <ExportDialog
          projectId={project.id}
          styles={styles}
          defaultPreset={PRESET}
          onBeforeRender={beforeRender}
          onRendered={() => void api.reload()}
          onClose={() => setExporting(false)}
        />
      )}
    </div>
  );
}
