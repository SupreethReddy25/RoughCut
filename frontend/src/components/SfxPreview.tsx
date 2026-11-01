import { useEffect, useRef } from "react";
import type { EDL } from "../types";
import type { TimeInfo } from "./PreviewPlayer";

const SECONDS = 0.22;

/** The export's soft whoosh in the preview: a short burst of filtered noise when the playhead crosses a cue. */
export function SfxPreview({ edl, time }: { edl: EDL; time: TimeInfo }) {
  const ctx = useRef<AudioContext | null>(null);
  const last = useRef(time.out);

  useEffect(() => {
    const prev = last.current;
    last.current = time.out;
    if (!time.playing || time.out < prev || time.out - prev > 0.5) return;
    if (!(edl.audio.sfx ?? []).some((c) => c.t0 > prev && c.t0 <= time.out)) return;
    try {
      const ac = (ctx.current ??= new AudioContext());
      const n = Math.floor(ac.sampleRate * SECONDS);
      const buf = ac.createBuffer(1, n, ac.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * 0.25 * Math.min(i / (n * 0.2), 1) * (1 - i / n);
      const src = ac.createBufferSource();
      const hp = ac.createBiquadFilter();
      hp.type = "highpass";
      hp.frequency.value = 600;
      src.buffer = buf;
      src.connect(hp).connect(ac.destination);
      src.start();
    } catch {
      /* no audio in this browser: the export still has it */
    }
  }, [time, edl.audio.sfx]);

  return null;
}
