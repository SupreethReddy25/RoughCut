import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { proxyUrl } from "../api";
import type { ProjectOut, StylesOut } from "../api-types";
import { cropRect, dragCenter, videoBox, type Rect } from "../lib/crop";
import { Playback } from "../lib/playback";
import { Remap } from "../lib/remap";
import { unitsBetween } from "../lib/words";
import { CaptionOverlay, TextOverlays } from "./CaptionOverlay";

export interface PlayerHandle {
  toggle(): void;
  play(): void;
  pause(): void;
  /** move to a source time (inside a cut: the start of the next kept range) */
  seekSrc(t: number): void;
  seekOut(o: number): void;
  getSrcTime(): number;
  getOutTime(): number;
  isPlaying(): boolean;
}

export interface TimeInfo {
  src: number;
  out: number;
  playing: boolean;
  clip: number;
}

interface Props {
  project: ProjectOut;
  styles: StylesOut | null;
  presetId: string;
  showSafe: boolean;
  reframe: boolean;
  /** ~15 Hz, plus on every seek / play / pause */
  onTime: (t: TimeInfo) => void;
  /** units the playhead has passed over (batched every 2 s) */
  onSeen: (unitIds: number[]) => void;
  /** the user dragged the picture of a clip: new crop centre for the clip that starts with `unitId` */
  onCrop: (unitId: number, x: number, y: number) => void;
}

const SEEN_FLUSH_MS = 2000;
const TIME_MS = 66;

/**
 * PLAN 12 "Browser preview": the proxy plays in a 9:16 frame; the rAF loop skips cut ranges; each clip gets its own
 * crop / zoom (CSS, same window maths as the export); captions and text overlays are laid over it.
 */
export const PreviewPlayer = forwardRef<PlayerHandle, Props>(function PreviewPlayer(props, ref) {
  const { project, styles, presetId, showSafe, reframe } = props;
  const edl = project.edl!;
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [frameW, setFrameW] = useState(360);
  const [clipIdx, setClipIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const [pendingCrop, setPendingCrop] = useState<{ unitId: number; x: number; y: number } | null>(null);
  const dragStart = useRef<{ px: number; py: number; cx: number; cy: number; rect: Rect; frameH: number } | null>(null);

  const remap = useMemo(
    () => new Remap(edl.clips.map((c): [number, number] => [c.src_start, c.src_end]), edl.output.speed),
    [edl.clips, edl.output.speed],
  );
  const pb = useRef(new Playback(remap));
  pb.current.setRemap(remap);
  const clipRef = useRef(0);
  const propsRef = useRef(props);
  propsRef.current = props;
  const unitsRef = useRef(project.units);
  unitsRef.current = project.units;
  const reported = useRef(new Set<number>(project.seen_unit_ids));
  const pending = useRef(new Set<number>());

  const srcDims = project.source;
  const outW = edl.output.width;
  const outH = edl.output.height;
  const frameH = (frameW * outH) / outW;

  // ---- frame size
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // the largest outW:outH frame that fits the space the parent gives us
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (!r || r.width <= 0 || r.height <= 0) return;
      setFrameW(Math.max(1, Math.floor(Math.min(r.width, (r.height * outW) / outH))));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [outW, outH]);

  const getOutTime = useCallback((): number => {
    const v = videoRef.current;
    return v ? pb.current.outTime(v) : 0;
  }, []);

  // ---- seen reporting
  const flushSeen = useCallback(() => {
    if (pending.current.size === 0) return;
    const ids = [...pending.current];
    pending.current.clear();
    ids.forEach((i) => reported.current.add(i));
    propsRef.current.onSeen(ids);
  }, []);
  useEffect(() => {
    const id = setInterval(flushSeen, SEEN_FLUSH_MS);
    return () => {
      clearInterval(id);
      flushSeen();
    };
  }, [flushSeen]);
  useEffect(() => {
    reported.current = new Set(project.seen_unit_ids);
  }, [project.id, project.seen_unit_ids]);

  const collect = (a: number, b: number) => {
    for (const id of unitsBetween(unitsRef.current, a, b)) if (!reported.current.has(id)) pending.current.add(id);
  };

  // ---- the playback loop
  useEffect(() => {
    let raf = 0;
    let lastSrc = -1;
    let lastReport = 0;
    let lastPlaying: boolean | null = null;
    const loop = (ts: number) => {
      const v = videoRef.current;
      if (v && pb.current.remap.ranges.length > 0) {
        const res = pb.current.tick(v);
        if (res.clip !== clipRef.current) {
          clipRef.current = res.clip;
          setClipIdx(res.clip);
        }
        const src = v.currentTime;
        if (!v.paused) {
          if (res.jumped || lastSrc < 0 || src < lastSrc || src - lastSrc > 0.5) collect(src, src);
          else collect(lastSrc, src);
        } else if (src !== lastSrc) collect(src, src);
        lastSrc = src;
        const isPlaying = !v.paused;
        if (ts - lastReport >= TIME_MS || isPlaying !== lastPlaying) {
          lastReport = ts;
          lastPlaying = isPlaying;
          propsRef.current.onTime({ src, out: pb.current.outTime(v), playing: isPlaying, clip: res.clip });
          if (!isPlaying) flushSeen();
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [flushSeen]);

  // ---- the edit changed (a cut was made / undone): stay on the same picture, out of newly cut ranges
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = edl.output.speed;
    if (v.paused && remap.ranges.length > 0) {
      const k = pb.current.seekSrc(v, v.currentTime);
      clipRef.current = k;
      setClipIdx(k);
    }
    setPendingCrop(null);
  }, [edl, remap]);

  // ---- imperative handle
  useImperativeHandle(
    ref,
    () => ({
      toggle() {
        const v = videoRef.current;
        if (!v || remap.ranges.length === 0) return;
        if (v.paused) {
          const end = remap.ranges[remap.ranges.length - 1][1];
          if (v.currentTime >= end - 0.05) pb.current.seekOut(v, 0);
          void Promise.resolve(v.play()).catch(() => setPlaying(false));
        } else v.pause();
      },
      play() {
        const v = videoRef.current;
        if (v && v.paused) void Promise.resolve(v.play()).catch(() => setPlaying(false));
      },
      pause() {
        videoRef.current?.pause();
      },
      seekSrc(t) {
        const v = videoRef.current;
        if (!v) return;
        const k = pb.current.seekSrc(v, t);
        clipRef.current = k;
        setClipIdx(Math.max(k, 0));
      },
      seekOut(o) {
        const v = videoRef.current;
        if (!v) return;
        const k = pb.current.seekOut(v, o);
        clipRef.current = k;
        setClipIdx(Math.max(k, 0));
      },
      getSrcTime: () => videoRef.current?.currentTime ?? 0,
      getOutTime,
      isPlaying: () => !!videoRef.current && !videoRef.current.paused,
    }),
    [remap, getOutTime],
  );

  // ---- crop / zoom of the current clip
  const clip = edl.clips[Math.min(clipIdx, edl.clips.length - 1)];
  const speechUnit = (c: typeof clip): number | null => {
    if (!c) return null;
    const kinds = new Map(project.units.map((u) => [u.id, u.kind]));
    return c.unit_ids.find((id) => kinds.get(id) === "speech") ?? null;
  };
  const anchor = speechUnit(clip);
  let center = clip ? clip.crop_center : { x: 0.5, y: 0.5 };
  if (pendingCrop && anchor === pendingCrop.unitId) center = { x: pendingCrop.x, y: pendingCrop.y };
  if (drag) center = drag;
  const rect: Rect = cropRect(clip?.zoom ?? 1, center.x, center.y, srcDims.width, srcDims.height, outW, outH);
  const box = videoBox(rect, srcDims.width, srcDims.height, frameW, frameH);

  const onPointerDown = (e: React.PointerEvent) => {
    if (!reframe || !clip || anchor === null) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragStart.current = { px: e.clientX, py: e.clientY, cx: center.x, cy: center.y, rect, frameH };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const s = dragStart.current;
    if (!s) return;
    const [x, y] = dragCenter(s.cx, s.cy, e.clientX - s.px, e.clientY - s.py, s.rect, srcDims.width, srcDims.height, frameW, s.frameH);
    setDrag({ x, y });
  };
  const onPointerUp = () => {
    const s = dragStart.current;
    dragStart.current = null;
    if (!s) return;
    const d = drag;
    setDrag(null);
    if (d && anchor !== null && (Math.abs(d.x - s.cx) > 1e-4 || Math.abs(d.y - s.cy) > 1e-4)) {
      setPendingCrop({ unitId: anchor, x: d.x, y: d.y });
      propsRef.current.onCrop(anchor, Math.round(d.x * 10000) / 10000, Math.round(d.y * 10000) / 10000);
    }
  };

  const preset = styles?.presets[presetId];
  return (
    <div ref={wrapRef} className="absolute inset-0 flex items-center justify-center">
    <div
      data-testid="preview-frame"
      style={{ width: frameW, height: frameH }}
      className={`relative shrink-0 select-none overflow-hidden rounded-lg bg-black ${
        reframe ? "cursor-grab active:cursor-grabbing" : ""
      }`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <video
        ref={videoRef}
        src={proxyUrl(project.id)}
        preload="auto"
        playsInline
        data-testid="preview-video"
        className="absolute left-0 top-0 max-w-none origin-top-left"
        style={{ width: box.width, height: box.height, transform: box.transform, touchAction: "none" }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onError={() => setVideoError(true)}
        onLoadedMetadata={() => {
          const v = videoRef.current;
          if (v) pb.current.seekSrc(v, v.currentTime);
        }}
      />
      {styles && (
        <div className="pointer-events-none absolute inset-0">
          <CaptionOverlay edl={edl} breaks={project.caption_breaks} styles={styles} frameW={frameW} getOutTime={getOutTime} />
          <TextOverlays edl={edl} breaks={project.text_breaks} styles={styles} frameW={frameW} getOutTime={getOutTime} />
        </div>
      )}
      {showSafe && preset && (
        <div className="pointer-events-none absolute inset-0" data-testid="safe-zones">
          <div className="absolute inset-x-0 top-0 border-b border-dashed border-rose-300/80 bg-rose-500/20" style={{ height: `${preset.safe_zones.top * 100}%` }}>
            <span className="m-1 text-[10px] text-rose-100">top bar</span>
          </div>
          <div className="absolute inset-x-0 bottom-0 border-t border-dashed border-rose-300/80 bg-rose-500/20" style={{ height: `${preset.safe_zones.bottom * 100}%` }}>
            <span className="m-1 text-[10px] text-rose-100">captions / buttons</span>
          </div>
          <div
            className="absolute right-0 border-l border-dashed border-rose-300/80 bg-rose-500/20"
            style={{ top: `${preset.safe_zones.top * 100}%`, bottom: `${preset.safe_zones.bottom * 100}%`, width: `${preset.safe_zones.right * 100}%` }}
          />
        </div>
      )}
      {reframe && (
        <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-[11px] text-white">
          Drag the picture to reframe this clip
        </div>
      )}
      {edl.clips.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-6 text-center text-sm text-neutral-300">
          Nothing is kept in this edit. Undo (Ctrl+Z) or restore some words in the transcript.
        </div>
      )}
      {videoError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 p-6 text-center text-sm text-red-300">
          The preview video could not be loaded.
        </div>
      )}
      {!playing && edl.clips.length > 0 && !videoError && (
        <div className="pointer-events-none absolute bottom-2 right-2 rounded bg-black/50 px-2 py-0.5 text-[10px] text-neutral-300">paused</div>
      )}
    </div>
    </div>
  );
});
