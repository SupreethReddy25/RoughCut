import { useEffect, useRef } from "react";
import { musicUrl } from "../api";
import { isSpeaking, loopDistance, musicPosition, smoothVolume, targetVolume } from "../lib/music";
import type { EDL } from "../types";
import type { TimeInfo } from "./PreviewPlayer";

const RESYNC_SECONDS = 0.6;

/**
 * Plays the background music of the edit in step with the preview (PLAN 8.6): looped, started and paused with the
 * video, put back in place after a seek, and dipped while somebody speaks. The export mixes it with a sidechain
 * compressor; this is the same idea driven by the EDL's caption words.
 */
export function MusicPreview({ edl, userId, time }: { edl: EDL; userId: number; time: TimeInfo }) {
  const music = edl.audio.music;
  const ref = useRef<HTMLAudioElement>(null);
  const volume = useRef(0);

  useEffect(() => {
    const a = ref.current;
    if (!a || !music) return;
    const len = isFinite(a.duration) && a.duration > 0 ? a.duration : 0;
    const start = music.start_s ?? 0;
    if (time.playing && time.out >= start) {
      const want = musicPosition(time.out - start, len);
      if (a.paused) {
        if (len) a.currentTime = want;
        void Promise.resolve(a.play()).catch(() => undefined); // blocked by the browser until the user has interacted
      } else if (len && loopDistance(a.currentTime, want, len) > RESYNC_SECONDS) {
        a.currentTime = want;
      }
    } else if (!a.paused) {
      a.pause();
    }
    volume.current = smoothVolume(volume.current, targetVolume(music.gain_db, music.duck_db, time.playing && isSpeaking(edl, time.out)));
    a.volume = volume.current;
  }, [time, edl, music]);

  // a different track starts from the right place
  useEffect(() => {
    volume.current = 0;
  }, [music?.track_id]);

  if (!music) return null;
  return <audio ref={ref} data-testid="music-preview" src={musicUrl(music.track_id, userId)} loop preload="auto" />;
}
