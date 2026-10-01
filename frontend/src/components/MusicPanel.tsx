import { useEffect, useRef, useState } from "react";
import { listMusic, musicUrl, uploadMusic } from "../api";
import type { TrackOut } from "../api-types";
import { DUCK_MAX, DUCK_MIN, GAIN_MAX, GAIN_MIN, PREVIEW_SECONDS } from "../lib/music";
import type { EDL } from "../types";

interface Props {
  edl: EDL;
  userId: number;
  /** send a `music` event: {track_id, gain_db?, duck_db?}; track_id null removes the music */
  onMusic: (payload: { track_id: string | null; gain_db?: number; duck_db?: number }) => void;
}

/** Music tab section (PLAN 8.6): track list with a 10 s listen, level and duck sliders, own upload. */
export function MusicPanel({ edl, userId, onMusic }: Props) {
  const music = edl.audio.music;
  const [tracks, setTracks] = useState<TrackOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    listMusic(userId)
      .then((t) => alive && setTracks(t))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [userId]);

  const stopListening = () => {
    if (stopTimer.current) clearTimeout(stopTimer.current);
    audio.current?.pause();
    audio.current = null;
    setListening(null);
  };
  useEffect(() => () => stopListening(), []);

  const listen = (t: TrackOut) => {
    if (listening === t.id) return stopListening();
    stopListening();
    const a = new Audio(musicUrl(t.id, userId));
    a.volume = 0.8;
    audio.current = a;
    setListening(t.id);
    a.onended = () => setListening(null);
    void a.play().catch(() => stopListening());
    stopTimer.current = setTimeout(stopListening, PREVIEW_SECONDS * 1000);
  };

  const upload = async (f: File | undefined) => {
    if (!f) return;
    setBusy(true);
    setError(null);
    try {
      const t = await uploadMusic(userId, f);
      setTracks((l) => (l.some((x) => x.id === t.id) ? l : [...l, t]));
      onMusic({ track_id: t.id });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (file.current) file.current.value = "";
    }
  };

  return (
    <section data-testid="music-panel">
      <h3 className="mb-1.5 flex items-center text-xs font-semibold uppercase tracking-wide text-neutral-300">
        Music
        {music && (
          <button type="button" onClick={() => onMusic({ track_id: null })} className="ml-auto rounded bg-white/10 px-2 py-0.5 text-[11px] normal-case text-neutral-100 hover:bg-white/20">
            Remove
          </button>
        )}
      </h3>
      <ul className="space-y-1" aria-label="Music tracks">
        {tracks.map((t) => {
          const on = music?.track_id === t.id;
          return (
            <li key={t.id} className={`flex items-center gap-1 rounded border px-2 py-1 text-xs ${on ? "border-sky-400 bg-sky-500/10" : "border-white/10"}`}>
              <button
                type="button"
                aria-label={listening === t.id ? `Stop ${t.name}` : `Listen to ${t.name}`}
                onClick={() => listen(t)}
                className="w-6 shrink-0 rounded bg-white/10 py-0.5 hover:bg-white/20"
              >
                {listening === t.id ? "■" : "▶"}
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => !on && onMusic({ track_id: t.id })}
                className="min-w-0 flex-1 truncate text-left text-neutral-100"
              >
                {t.name}
                <span className="ml-1 text-neutral-500">{t.source === "upload" ? "your track" : t.mood}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <input ref={file} type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" className="hidden" aria-label="Upload a music file" onChange={(e) => void upload(e.target.files?.[0])} />
      <button type="button" disabled={busy} onClick={() => file.current?.click()} className="mt-1.5 w-full rounded border border-dashed border-white/20 py-1 text-xs text-neutral-300 hover:bg-white/5 disabled:opacity-50">
        {busy ? "Uploading…" : "+ Use my own track (mp3 / wav, 10 MB)"}
      </button>
      {error && (
        <p role="alert" className="mt-1 text-xs text-red-300">
          {error}
        </p>
      )}
      {music && (
        <div className="mt-2 space-y-2 text-xs text-neutral-300">
          <Slider label="Music volume" unit="dB" min={GAIN_MIN} max={GAIN_MAX} value={music.gain_db} onCommit={(v) => onMusic({ track_id: music.track_id, gain_db: v, duck_db: music.duck_db })} />
          <Slider label="Dip under voice" unit="dB" min={DUCK_MIN} max={DUCK_MAX} value={music.duck_db} onCommit={(v) => onMusic({ track_id: music.track_id, gain_db: music.gain_db, duck_db: v })} />
          <p className="text-[11px] text-neutral-500">Fades in and out over 1 s. The dip is approximate in the preview; the export uses a sidechain compressor.</p>
        </div>
      )}
    </section>
  );
}

/** A slider that shows the value while dragging and sends it once, on release (one undo step, not fifty). */
function Slider({ label, unit, min, max, value, onCommit }: { label: string; unit: string; min: number; max: number; value: number; onCommit: (v: number) => void }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  const sent = useRef(value);
  useEffect(() => {
    sent.current = value;
  }, [value]);
  const commit = () => {
    if (v === sent.current) return;
    sent.current = v;
    onCommit(v);
  };
  return (
    <label className="block">
      <span className="flex justify-between">
        {label}
        <span className="tabular-nums text-neutral-400">
          {v} {unit}
        </span>
      </span>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={1}
        value={v}
        onChange={(e) => setV(Number(e.target.value))}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
        className="w-full"
      />
    </label>
  );
}
