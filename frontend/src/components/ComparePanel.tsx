import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { proxyUrl } from "../api";
import { fmtDuration } from "../lib/format";

export interface CompareHandle {
  /** start the original from the beginning (the edit is started by the parent at the same moment) */
  playFromStart(): void;
  pause(): void;
}

interface Props {
  projectId: number;
  before: number;
  after: number;
  /** source time of the edit's playhead: the original follows it while it is not playing by itself */
  srcTime: number;
  /** the edit is playing, and where it is: pausing it pauses the original, its natural end does not */
  editPlaying: boolean;
  editOut: number;
  onPlayBoth: () => void;
}

/** The untouched upload next to the edit. It follows the edit's playhead, or both can run from the start in real time,
 *  which is what makes the difference in length felt. */
export const ComparePanel = forwardRef<CompareHandle, Props>(function ComparePanel({ projectId, before, after, srcTime, editPlaying, editOut, onPlayBoth }, ref) {
  const video = useRef<HTMLVideoElement>(null);
  const [muted, setMuted] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);

  useImperativeHandle(ref, () => ({
    playFromStart() {
      const v = video.current;
      if (!v) return;
      v.currentTime = 0;
      void v.play().catch(() => setPlaying(false));
    },
    pause() {
      video.current?.pause();
    },
  }), []);

  useEffect(() => {
    const v = video.current;
    if (v && !playing && Math.abs(v.currentTime - srcTime) > 0.25) v.currentTime = srcTime;
  }, [srcTime, playing]);

  const wasPlaying = useRef(false);
  useEffect(() => {
    if (wasPlaying.current && !editPlaying && editOut < after - 0.25) video.current?.pause();
    wasPlaying.current = editPlaying;
  }, [editPlaying, editOut, after]);

  const saved = before > 0 ? Math.max(0, Math.round((1 - after / before) * 100)) : 0;
  return (
    <div className="flex h-[60vh] min-h-0 flex-col gap-2 border-white/[0.06] p-3 md:col-span-2 lg:col-span-1 lg:h-auto lg:border-r" data-testid="compare-panel">
      <div className="flex items-center justify-between text-xs text-neutral-400">
        <span className="font-medium uppercase tracking-wide">Original</span>
        <span className="font-mono tabular-nums">{fmtDuration(now)} / {fmtDuration(before)}</span>
      </div>
      <div className="relative min-h-0 flex-1 overflow-hidden rounded-lg bg-black ring-1 ring-white/[0.08]">
        <video
          ref={video}
          src={proxyUrl(projectId)}
          muted={muted}
          playsInline
          preload="metadata"
          aria-label="Original video"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
          className="absolute inset-0 h-full w-full object-contain"
        />
      </div>
      <div className="grid gap-1 text-[11px] text-neutral-400" aria-hidden="true">
        <div className="h-1.5 rounded-full bg-white/20" />
        <div className="h-1.5 rounded-full bg-sky-400" style={{ width: `${before > 0 ? Math.min(100, (after / before) * 100) : 100}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <button type="button" onClick={onPlayBoth} className="rc-press rounded-full bg-sky-500/20 px-3 py-1 text-sky-100 ring-1 ring-sky-400/30 hover:bg-sky-500/30">
          ▶ Play both from start
        </button>
        <button type="button" onClick={() => setMuted((m) => !m)} aria-pressed={!muted} className="rc-press rounded-full bg-white/[0.06] px-3 py-1 text-neutral-300 ring-1 ring-white/10 hover:bg-white/[0.1]">
          {muted ? "Original muted" : "Original sound on"}
        </button>
        <span className="ml-auto text-emerald-300">{saved}% shorter</span>
      </div>
    </div>
  );
});
