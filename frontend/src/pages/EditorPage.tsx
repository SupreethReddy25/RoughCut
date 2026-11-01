import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { EventIn, ManualEditIn, ProjectOut, SuggestionOut } from "../api-types";
import { CaptionStrip } from "../components/CaptionStrip";
import { AnalysisProgress } from "../components/AnalysisProgress";
import { CaptionStylePanel } from "../components/CaptionStylePanel";
import { ComparePanel, type CompareHandle } from "../components/ComparePanel";
import { SfxPreview } from "../components/SfxPreview";
import { ExportDialog } from "../components/ExportDialog";
import { MusicPanel } from "../components/MusicPanel";
import { MusicPreview } from "../components/MusicPreview";
import { PreviewPlayer, type PlayerHandle, type TimeInfo } from "../components/PreviewPlayer";
import { LogoMark } from "../components/Brand";
import { ReviewPanel, WhyBox } from "../components/ReviewPanel";
import { Timeline } from "../components/Timeline";
import { TranscriptEditor, type Sel, type TranscriptHandle } from "../components/TranscriptEditor";
import { useActiveSeconds } from "../hooks/useActiveSeconds";
import { useProject, type ProjectApi } from "../hooks/useProject";
import { useFonts, useStyles } from "../hooks/useStyles";
import { useUser } from "../hooks/useUser";
import { acceptEvent, chooseEvent, keyOf, nextInQueue, nextUncertain, rejectEvent, sameKey } from "../lib/decide";
import { actionsFor, targetStart } from "../lib/describe";
import { fmtTime } from "../lib/format";
import { isInteractiveTarget, isTypingTarget, resolveShortcut } from "../lib/shortcuts";

const PRESET = "reels";
const AUTO_ADVANCE_KEY = "roughcut.autoAdvance";

/** "Next after each decision" is on unless the user turned it off (remembered in the browser). */
function readAutoAdvance(): boolean {
  try {
    return localStorage.getItem(AUTO_ADVANCE_KEY) !== "0";
  } catch {
    return true;
  }
}
const POLL_MS = 2000;

type Ready = ProjectOut & { edl: NonNullable<ProjectOut["edl"]> };

/** A pill switch around a real checkbox (so labels, keyboard and tests behave like a checkbox). */
function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (on: boolean) => void; label: string; hint: string }) {
  return (
    <label title={hint} className="relative">
      <input type="checkbox" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs text-neutral-400 ring-1 ring-white/[0.07] hover:text-neutral-200 peer-checked:bg-sky-500/15 peer-checked:text-sky-200 peer-checked:ring-sky-400/40 peer-focus-visible:outline-2 peer-focus-visible:outline-sky-400">
        <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${checked ? "bg-sky-300" : "bg-neutral-600"}`} />
        {label}
      </span>
    </label>
  );
}

/** Output length vs the target the creator set (the planner fits phrase/pause decisions to it, app/rl/plan.py). */
export function TargetChip({ duration, target }: { duration: number; target: number }) {
  const ok = duration <= target * 1.1; // a target is a limit: a shorter good edit is fine, it is never padded
  return (
    <span
      title="You set a target length. Keep/cut and pause decisions are planned over the whole video to stay within it, without leaving half a sentence (finite-horizon MDP solved by dynamic programming, with posterior sampling)."
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] ring-1 ${
        ok ? "bg-emerald-500/10 text-emerald-300 ring-emerald-400/25" : "bg-amber-400/10 text-amber-300 ring-amber-400/25"
      }`}
      data-testid="target-chip"
    >
      <span aria-hidden="true">◎</span>
      {duration.toFixed(1)}s of {target}s target
    </span>
  );
}

/** How long the upload was, how long the edit is. */
export function DurationChip({ before, after }: { before: number; after: number }) {
  const saved = before > 0 ? Math.max(0, Math.round((1 - after / before) * 100)) : 0;
  return (
    <span title="Length of your upload, then of this edit" data-testid="duration-chip" className="inline-flex items-center gap-1.5 rounded-md bg-white/[0.05] px-2 py-0.5 font-mono text-xs tabular-nums text-neutral-300 ring-1 ring-white/[0.06]">
      {fmtClock(before)} <span aria-hidden="true" className="text-neutral-500">→</span> <span className="text-white">{fmtClock(after)}</span>
      {saved > 0 && <span className="text-emerald-300">−{saved}%</span>}
    </span>
  );
}

const fmtClock = (t: number) => `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`.replace(/:60$/, ":59");

export default function EditorPage() {
  const id = Number(useParams().id);
  if (!Number.isInteger(id) || id <= 0) return <Notice title="Project not found" />;
  return <Editor key={id} id={id} />;
}

function Notice({ title, text, children }: { title: string; text?: string; children?: React.ReactNode }) {
  return (
    <div className="rc-glow flex min-h-screen items-center justify-center p-6 text-neutral-100">
      <div className="rc-in w-full max-w-md text-center">
        <div className="mb-5 flex justify-center">
          <LogoMark size={40} />
        </div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {text && <p className="mt-2 text-sm text-neutral-400">{text}</p>}
        <div className="mt-5">{children}</div>
        <Link to="/" className="mt-6 inline-block text-sm text-neutral-400 hover:text-white">
          ← All projects
        </Link>
      </div>
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
        <button type="button" onClick={() => void reload()} className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500">
          Try again
        </button>
      </Notice>
    );
  }
  if (!project) return <Notice title="Loading…" />;
  if (project.status === "failed") {
    return <Notice title="Processing failed" text={project.error ?? "The video could not be analysed."} />;
  }
  if (project.status === "uploaded" || project.status === "processing") {
    return (
      <Notice title="Analysing your video…">
        <AnalysisProgress id={project.id} progress={project.progress ?? null} />
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
  const [autoAdvance, setAutoAdvanceState] = useState(readAutoAdvance);
  const setAutoAdvance = useCallback((on: boolean) => {
    setAutoAdvanceState(on);
    try {
      localStorage.setItem(AUTO_ADVANCE_KEY, on ? "1" : "0");
    } catch {
      /* not remembered */
    }
  }, []);
  const [showSafe, setShowSafe] = useState(false);
  const [reframe, setReframe] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [comparing, setComparing] = useState(false);
  const compare = useRef<CompareHandle>(null);

  const units = useMemo(() => new Map(project.units.map((u) => [u.id, u])), [project.units]);
  const words = useMemo(() => new Map(project.words.map((w) => [w.id, w])), [project.words]);
  const selected: SuggestionOut | null = useMemo(() => project.suggestions.find((s) => sameKey(s, sel)) ?? null, [project.suggestions, sel]);

  // the global key handler reads the latest values from here
  const live = useRef({ api, project, selected, sel, selText, exporting, units, words, autoAdvance });
  live.current = { api, project, selected, sel, selText, exporting, units, words, autoAdvance };

  const select = useCallback((s: SuggestionOut) => {
    setSel(keyOf(s));
    setSelText(null);
    const t = targetStart(s, live.current.units, live.current.words);
    if (t !== null) player.current?.seekSrc(t);
  }, []);

  const { send, markSeen } = api;
  /** A review decision (accept / reject / choose). With "next after each" on, the selection then moves to the next
   *  uncertain suggestion of the state the server answered with. */
  const decide = useCallback(
    async (ev: EventIn) => {
      const out = await live.current.api.send(ev);
      if (!out || !live.current.autoAdvance) return;
      const next = nextUncertain(out.suggestions, out.queue);
      if (next) select(next);
    },
    [select],
  );
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
          if (ev) void decide(ev);
          break;
        }
        case "reject": {
          const ev = rejectEvent(L.selected);
          if (ev) void decide(ev);
          break;
        }
        case "choose": {
          const ev = L.selected ? chooseEvent(L.selected, actionsFor(L.selected)[act.index]) : null;
          if (ev) void decide(ev);
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
  }, [select, decide]);

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
  const nPending = project.queue.needs_review.length;
  const playBoth = useCallback(() => {
    player.current?.seekOut(0);
    compare.current?.playFromStart();
    player.current?.play();
  }, []);

  return (
    <div className="flex h-screen flex-col bg-neutral-950 text-neutral-100">
      <header className="relative z-20 flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-white/[0.06] bg-neutral-950/80 px-3 py-2 text-sm backdrop-blur-xl">
        <Link to="/" className="flex items-center gap-2 rounded-lg px-1.5 py-1 text-neutral-400 hover:bg-white/[0.05] hover:text-white" title="All projects">
          <LogoMark size={22} />
          <span className="hidden sm:inline">← Projects</span>
        </Link>
        <span aria-hidden="true" className="h-5 w-px bg-white/10" />
        <h1 className="max-w-[16rem] truncate font-semibold tracking-tight" title={project.title}>
          {project.title || project.video_id}
        </h1>
        <span className="rounded-md bg-white/[0.05] px-2 py-0.5 font-mono text-xs tabular-nums text-neutral-300 ring-1 ring-white/[0.06]">
          {fmtTime(time.out)} <span className="text-neutral-600">/</span> {fmtTime(project.edl.duration)}
        </span>
        <DurationChip before={project.duration} after={project.edl.duration} />
        <span className="text-xs text-neutral-400">{nPending} need{nPending === 1 ? "s" : ""} you</span>
        {project.target_duration ? <TargetChip duration={project.edl.duration} target={project.target_duration} /> : null}
        {api.busy > 0 && (
          <span className="flex items-center gap-1.5 text-xs text-amber-300">
            <span aria-hidden="true" className="h-1.5 w-1.5 animate-ping rounded-full bg-amber-300" />
            saving…
          </span>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          <div className="flex items-center rounded-lg bg-white/[0.04] p-0.5 ring-1 ring-white/[0.06]">
            <button type="button" onClick={() => void api.undo()} disabled={!project.can_undo} title="Undo (Ctrl+Z)" className="rc-press rounded-md px-2 py-1 text-neutral-300 hover:bg-white/[0.08] hover:text-white disabled:opacity-30 disabled:hover:bg-transparent">
              ↶ Undo
            </button>
            <button type="button" onClick={() => void api.redo()} disabled={!project.can_redo} title="Redo (Ctrl+Shift+Z)" className="rc-press rounded-md px-2 py-1 text-neutral-300 hover:bg-white/[0.08] hover:text-white disabled:opacity-30 disabled:hover:bg-transparent">
              ↷ Redo
            </button>
          </div>
          <Toggle checked={comparing} onChange={setComparing} label="Compare" hint="See the original next to the edit. The review panel moves below them." />
          <Toggle checked={showSafe} onChange={setShowSafe} label="Safe zones" hint="Show where the Reels / Shorts buttons cover the video" />
          <Toggle checked={reframe} onChange={setReframe} label="Reframe" hint="Drag the video inside the 9:16 frame to move the crop" />
          <button
            type="button"
            onClick={() => {
              player.current?.pause();
              setExporting(true);
            }}
            disabled={project.edl.clips.length === 0}
            className="rc-press ml-1 rounded-lg bg-gradient-to-b from-sky-500 to-sky-600 px-3.5 py-1.5 font-medium text-white shadow-lg shadow-sky-900/50 ring-1 ring-white/10 hover:from-sky-400 hover:to-sky-500 disabled:opacity-40"
          >
            Export…
          </button>
        </div>
      </header>

      {(api.error || api.notice) && (
        <div role={api.error ? "alert" : "status"} className={`rc-in flex items-center gap-2 border-b px-3 py-1.5 text-xs ${api.error ? "border-red-500/20 bg-red-500/10 text-red-200" : "border-sky-400/15 bg-sky-500/[0.08] text-sky-200"}`}>
          <span className="flex-1">{api.error ?? api.notice}</span>
          <button type="button" aria-label="Dismiss" onClick={api.dismiss} className="px-1 hover:text-white">
            ×
          </button>
        </div>
      )}

      {/* wide and laptop: preview | transcript | panel, so the video stays beside the review. Comparing: original | edit |
          transcript with the panel below. Narrow: stacked, the page scrolls. */}
      <div className={`grid min-h-0 flex-1 grid-cols-1 overflow-y-auto md:grid-cols-[minmax(220px,280px)_minmax(0,1fr)] lg:overflow-hidden ${
        comparing
          ? "lg:grid-cols-[minmax(0,1.2fr)_minmax(200px,0.8fr)_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)_minmax(0,32vh)]"
          : "lg:grid-cols-[minmax(200px,270px)_minmax(0,1fr)_minmax(290px,340px)] xl:grid-cols-[minmax(240px,320px)_minmax(0,1fr)_minmax(320px,380px)]"
      }`}>
        {comparing && <ComparePanel ref={compare} projectId={project.id} before={project.duration} after={project.edl.duration} srcTime={time.src} onPlayBoth={playBoth} />}
        <div className="flex h-[78vh] min-h-0 flex-col items-center gap-2.5 overflow-y-auto border-white/[0.06] bg-[radial-gradient(80%_60%_at_50%_30%,rgb(111_92_246/0.10),transparent_70%)] p-3 md:h-[64vh] md:border-r lg:h-auto">
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
          <button type="button" onClick={() => player.current?.toggle()} className="rc-press inline-flex items-center gap-2 rounded-full bg-white/[0.07] px-4 py-1.5 text-sm ring-1 ring-white/10 hover:bg-white/[0.12]">
            <span aria-hidden="true" className="text-[10px]">{time.playing ? "❚❚" : "▶"}</span>
            {time.playing ? "Pause" : "Play"} <kbd className="text-[10px] opacity-50">Space</kbd>
          </button>
          {!fontsReady && <p className="text-[10px] text-neutral-500">Loading caption fonts…</p>}
          <CaptionStrip project={project} outTime={time.out} onEdit={onEdit} />
          {user && <MusicPreview edl={project.edl} userId={user.id} time={time} />}
          <SfxPreview edl={project.edl} time={time} />
        </div>

        <div className="h-[64vh] min-h-0 overflow-y-auto border-t border-white/[0.06] md:border-t-0 lg:h-auto">
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

        <div className={`flex h-[80vh] min-h-0 flex-col border-t border-white/[0.06] bg-neutral-950 md:col-span-2 lg:h-auto ${comparing ? "lg:col-span-3" : "lg:col-span-1 lg:border-l lg:border-t-0"}`}>
          <div className="p-2 pb-0">
            <div className="flex rounded-lg bg-white/[0.04] p-0.5 text-sm ring-1 ring-white/[0.06]" role="tablist">
              {(["review", "style", "music"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  role="tab"
                  aria-selected={tab === t}
                  onClick={() => setTab(t)}
                  className={`flex-1 rounded-md px-3 py-1 ${tab === t ? "bg-white/[0.09] text-white shadow-sm shadow-black/40" : "text-neutral-400 hover:text-neutral-200"}`}
                >
                  {t === "review" ? "Review" : t === "style" ? "Style" : "Music"}
                </button>
              ))}
            </div>
          </div>
          <div key={tab} className="rc-in min-h-0 flex-1 overflow-hidden">
            {tab === "review" ? (
              <ReviewPanel
                project={project}
                selected={sel}
                fresh={api.fresh}
                onNudge={onNudge}
                styleText={project.style_mixture?.text}
                autoAdvance={autoAdvance}
                onAutoAdvance={setAutoAdvance}
                onlyUncertain={onlyUncertain}
                onShowAll={() => setParams({}, { replace: true })}
                onSelect={select}
                onAccept={(s) => {
                  const ev = acceptEvent(s);
                  if (ev) void decide(ev);
                }}
                onReject={(s) => {
                  const ev = rejectEvent(s);
                  if (ev) void decide(ev);
                }}
                onChoose={(s, a) => {
                  const ev = chooseEvent(s, a);
                  if (ev) void decide(ev);
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
          onRate={(value) => void send({ kind: "rating", payload: { value } })}
          rating={project.rating ?? null}
          onClose={() => setExporting(false)}
        />
      )}
    </div>
  );
}
