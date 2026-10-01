import { useEffect, useState } from "react";
import { fontUrl, getStyles } from "../api";
import type { StylesOut } from "../api-types";

let cached: Promise<StylesOut> | null = null;

/** The caption / text style JSON + presets the export renders from (loaded once per page). */
export function useStyles(): { styles: StylesOut | null; error: string | null } {
  const [styles, setStyles] = useState<StylesOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    cached ??= getStyles();
    cached
      .then((s) => alive && setStyles(s))
      .catch((e: Error) => {
        cached = null; // retry next time
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, []);
  return { styles, error };
}

const loaded = new Set<string>();

/**
 * Loads the exact font files the export uses (served from /api/fonts) so captions in the preview are set in the same
 * faces. Returns true once they are ready (or when the browser has no FontFace API).
 */
export function useFonts(styles: StylesOut | null): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (!styles) return;
    if (typeof FontFace === "undefined" || !document.fonts) {
      setReady(true);
      return;
    }
    let alive = true;
    const all = [...Object.values(styles.captions.styles), ...Object.values(styles.texts.styles)];
    const jobs = [...new Map(all.map((s) => [s.font_file, s.font])).entries()]
      .filter(([file]) => !loaded.has(file))
      .map(async ([file, family]) => {
        const face = new FontFace(family, `url(${fontUrl(file)})`);
        await face.load();
        document.fonts.add(face);
        loaded.add(file);
      });
    Promise.all(jobs)
      .catch(() => undefined) // a missing font only changes the look, never blocks editing
      .finally(() => alive && setReady(true));
    return () => {
      alive = false;
    };
  }, [styles]);
  return ready;
}
