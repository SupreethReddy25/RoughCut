import { useEffect, useState } from "react";
import type { ProjectOut, StylesOut } from "../api-types";
import { fmtTime } from "../lib/format";
import type { TextOverlay } from "../types";

interface Props {
  project: ProjectOut;
  styles: StylesOut | null;
  outTime: number;
  selectedText: string | null;
  onSelectText: (id: string | null) => void;
  onSetting: (key: string, value: unknown) => void;
  onSpeed: (value: number) => void;
  onOverlay: (payload: Record<string, unknown>) => void;
}

const SPEEDS = [1, 1.1, 1.15, 1.25];
const POSITIONS: { id: string; label: string }[] = [
  { id: "lower_third", label: "Lower third" },
  { id: "center", label: "Center" },
  { id: "top", label: "Top" },
];

/** Caption style, position, speed, audio and the text-overlay tool (PLAN 8.2, 8.3, 8.7). */
export function CaptionStylePanel({ project, styles, outTime, selectedText, onSelectText, onSetting, onSpeed, onOverlay }: Props) {
  const st = project.settings;
  const texts = project.edl?.texts ?? [];
  return (
    <div className="space-y-4 overflow-y-auto p-3 text-sm" data-testid="style-panel">
      <section>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-300">Captions</h3>
        <label className="mb-2 flex items-center gap-2 text-neutral-200">
          <input
            type="checkbox"
            checked={st.captions_enabled}
            onChange={(e) => onSetting("captions_enabled", e.target.checked)}
          />
          Burn captions into the video
        </label>
        <div className="grid grid-cols-2 gap-1.5" role="radiogroup" aria-label="Caption style">
          {styles &&
            Object.entries(styles.captions.styles).map(([id, s]) => (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={st.caption_style === id}
                onClick={() => st.caption_style !== id && onSetting("caption_style", id)}
                className={`rounded-lg border px-2 py-2 text-left ${
                  st.caption_style === id ? "border-sky-400 bg-sky-500/15" : "border-white/15 hover:bg-white/5"
                }`}
              >
                <div style={{ fontFamily: `"${s.font}", sans-serif`, textTransform: s.case === "upper" ? "uppercase" : s.case === "lower" ? "lowercase" : "none" }} className="text-sm text-white">
                  {s.label}
                </div>
                <div className="text-[10px] text-neutral-400">{s.mode === "pop" ? "word pops" : s.mode === "karaoke" ? "fills as spoken" : "whole line"}</div>
              </button>
            ))}
        </div>
        <div className="mt-2 flex gap-1" role="radiogroup" aria-label="Caption position">
          {POSITIONS.map((p) => (
            <button
              key={p.id}
              type="button"
              role="radio"
              aria-checked={st.caption_position === p.id}
              onClick={() => st.caption_position !== p.id && onSetting("caption_position", p.id)}
              className={`flex-1 rounded border px-2 py-1 text-xs ${
                st.caption_position === p.id ? "border-sky-400 text-sky-300" : "border-white/15 text-neutral-300 hover:bg-white/5"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-300">Speed</h3>
        <div className="flex gap-1" role="radiogroup" aria-label="Speed">
          {SPEEDS.map((v) => (
            <button
              key={v}
              type="button"
              role="radio"
              aria-checked={st.speed === v}
              onClick={() => st.speed !== v && onSpeed(v)}
              className={`flex-1 rounded border px-2 py-1 text-xs ${
                st.speed === v ? "border-sky-400 text-sky-300" : "border-white/15 text-neutral-300 hover:bg-white/5"
              }`}
            >
              {v}×
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-300">Audio</h3>
        <label className="flex items-center gap-2 text-neutral-200">
          <input type="checkbox" checked={st.audio_enhance} onChange={(e) => onSetting("audio_enhance", e.target.checked)} />
          Enhance audio
        </label>
        <p className="mt-1 text-[11px] text-neutral-500">
          Enhanced audio (rumble filter, light denoise, loudness −14 LUFS) is applied on export; the preview plays the original sound.
        </p>
      </section>

      <section>
        <h3 className="mb-1.5 flex items-center text-xs font-semibold uppercase tracking-wide text-neutral-300">
          Text
          <button
            type="button"
            onClick={() => {
              const t0 = Math.max(0, Math.round(outTime * 10) / 10);
              onOverlay({ op: "add", text: "Your title", t0, t1: Math.min(t0 + 3, Math.max(project.edl?.duration ?? 3, t0 + 0.5)) });
            }}
            className="ml-auto rounded bg-white/10 px-2 py-0.5 text-[11px] normal-case text-neutral-100 hover:bg-white/20"
          >
            + Add text
          </button>
        </h3>
        {texts.length === 0 && <p className="text-[11px] text-neutral-500">Add a title or label; it shows on the timeline text track.</p>}
        <ul className="space-y-2">
          {texts.map((t) => (
            <OverlayRow key={t.id} t={t} styles={styles} selected={selectedText === t.id} duration={project.edl?.duration ?? 0}
              onSelect={() => onSelectText(selectedText === t.id ? null : t.id)} onOverlay={onOverlay} />
          ))}
        </ul>
      </section>
    </div>
  );
}

function OverlayRow({
  t, styles, selected, duration, onSelect, onOverlay,
}: {
  t: TextOverlay;
  styles: StylesOut | null;
  selected: boolean;
  duration: number;
  onSelect: () => void;
  onOverlay: (payload: Record<string, unknown>) => void;
}) {
  const [text, setText] = useState(t.text);
  const [t0, setT0] = useState(String(t.t0));
  const [t1, setT1] = useState(String(t.t1));
  useEffect(() => {
    setText(t.text);
    setT0(String(t.t0));
    setT1(String(t.t1));
  }, [t.text, t.t0, t.t1]);
  const commit = (patch: Record<string, unknown>) => onOverlay({ op: "update", id: t.id, ...patch });
  const commitText = () => text.trim() && text !== t.text && commit({ text: text.trim() });
  const commitTimes = () => {
    const a = Number(t0);
    const b = Number(t1);
    if (!isFinite(a) || !isFinite(b) || a < 0 || b <= a) {
      setT0(String(t.t0));
      setT1(String(t.t1));
      return;
    }
    if (a !== t.t0 || b !== t.t1) commit({ t0: a, t1: b });
  };
  return (
    <li className={`rounded-lg border p-2 ${selected ? "border-amber-300 bg-amber-400/10" : "border-white/10"}`} data-testid={`overlay-${t.id}`}>
      <div className="flex gap-1">
        <input
          aria-label="Overlay text"
          value={text}
          onFocus={onSelect}
          onChange={(e) => setText(e.target.value)}
          onBlur={commitText}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          className="min-w-0 flex-1 rounded bg-neutral-800 px-2 py-1 text-neutral-100"
        />
        <button type="button" aria-label="Delete text" onClick={() => onOverlay({ op: "delete", id: t.id })} className="rounded bg-rose-600/80 px-2 text-white hover:bg-rose-500">
          ×
        </button>
      </div>
      <div className="mt-1 flex items-center gap-1 text-[11px] text-neutral-400">
        <input aria-label="Start" value={t0} onChange={(e) => setT0(e.target.value)} onBlur={commitTimes} className="w-14 rounded bg-neutral-800 px-1 text-neutral-100" />
        –
        <input aria-label="End" value={t1} onChange={(e) => setT1(e.target.value)} onBlur={commitTimes} className="w-14 rounded bg-neutral-800 px-1 text-neutral-100" />
        s of {fmtTime(duration)}
        <select
          aria-label="Position"
          value={t.position}
          onChange={(e) => commit({ position: e.target.value, ...(e.target.value === "custom" ? { y: 0.5 } : {}) })}
          className="ml-auto rounded bg-neutral-800 px-1 text-neutral-100"
        >
          <option value="top">top</option>
          <option value="center">center</option>
          <option value="custom">custom</option>
        </select>
        <select aria-label="Text style" value={t.style_id} onChange={(e) => commit({ style_id: e.target.value })} className="rounded bg-neutral-800 px-1 text-neutral-100">
          {Object.keys(styles?.texts.styles ?? { title: 1 }).map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </div>
    </li>
  );
}
