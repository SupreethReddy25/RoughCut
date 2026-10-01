import { useEffect, useRef } from "react";

/**
 * Seconds the user has really spent in the editor (PLAN 16.1: "time in the editor with focus, not wall clock"):
 * counted once a second while the tab is visible AND the window has focus. Returns a getter.
 */
export function useActiveSeconds(enabled = true): () => number {
  const seconds = useRef(0);
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => {
      if (document.visibilityState === "visible" && document.hasFocus()) seconds.current += 1;
    }, 1000);
    return () => clearInterval(id);
  }, [enabled]);
  return () => seconds.current;
}
