import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import { ApiError } from "../api";
import type { ChangeOut, EventIn, FinalizeOut, ProjectOut, StateOut } from "../api-types";

export interface ProjectApi {
  project: ProjectOut | null;
  /** the project could not be loaded at all */
  loadError: string | null;
  /** the last action failed (shown as a dismissible banner) */
  error: string | null;
  /** number of requests in flight */
  busy: number;
  /** "Adjusted 3 suggestions based on your last change" and similar */
  notice: string | null;
  /** ids of suggestions that live re-scoring just replaced them with (to highlight in the queue) */
  fresh: Set<number>;
  reload: () => Promise<void>;
  suggest: () => Promise<void>;
  send: (ev: EventIn) => Promise<StateOut | null>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  markSeen: (unitIds: number[]) => Promise<void>;
  finalize: (secondsActive: number) => Promise<FinalizeOut | null>;
  dismiss: () => void;
}

/** The state fields every mutation response carries; transient ones (message, changes, events) are handled apart. */
export function mergeState(p: ProjectOut, s: StateOut): ProjectOut {
  return {
    ...p,
    status: s.status as ProjectOut["status"],
    settings: s.settings,
    suggestions: s.suggestions,
    queue: s.queue,
    edl: s.edl,
    style_mixture: s.style_mixture,
    can_undo: s.can_undo,
    can_redo: s.can_redo,
    n_events: s.n_events,
    caption_breaks: s.caption_breaks,
    text_breaks: s.text_breaks,
    caption_word_ids: s.caption_word_ids,
  };
}

const FRESH_MS = 6000;

/**
 * Loads a project and funnels every action through ONE request queue: the server derives the edit from the ordered
 * event log, so two quick key presses must reach it in the order they were made. The server is the source of truth -
 * the state is replaced by what each response says (no optimistic guessing, so undo/redo can never disagree).
 */
export function useProject(id: number): ProjectApi {
  const [project, setProject] = useState<ProjectOut | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [fresh, setFresh] = useState<Set<number>>(new Set());
  const chain = useRef<Promise<unknown>>(Promise.resolve());
  const alive = useRef(true);
  const idRef = useRef(id);
  idRef.current = id;
  const freshTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (freshTimer.current) clearTimeout(freshTimer.current);
    };
  }, []);

  const guard = <T,>(fn: () => Promise<T>): Promise<T | null> => {
    const run = async (): Promise<T | null> => {
      if (alive.current) setBusy((n) => n + 1);
      try {
        const out = await fn();
        if (alive.current) setError(null);
        return out;
      } catch (e) {
        if (alive.current) setError(e instanceof ApiError ? e.message : (e as Error).message);
        return null;
      } finally {
        if (alive.current) setBusy((n) => n - 1);
      }
    };
    const next = chain.current.then(run, run);
    chain.current = next;
    return next;
  };

  const apply = useCallback((s: StateOut) => {
    if (!alive.current) return;
    setProject((p) => (p ? mergeState(p, s) : p));
    setNotice(s.message);
    const ch: ChangeOut[] = s.changes ?? [];
    if (ch.length) {
      setFresh(new Set(ch.map((c) => c.new_id)));
      if (freshTimer.current) clearTimeout(freshTimer.current);
      freshTimer.current = setTimeout(() => alive.current && setFresh(new Set()), FRESH_MS);
    }
  }, []);

  const reload = useCallback(async () => {
    try {
      const p = await api.getProject(idRef.current);
      if (alive.current) {
        setProject(p);
        setLoadError(null);
      }
    } catch (e) {
      if (alive.current) setLoadError(e instanceof ApiError ? e.message : (e as Error).message);
    }
  }, []);

  useEffect(() => {
    setProject(null);
    setLoadError(null);
    void reload();
  }, [id, reload]);

  const suggest = useCallback(async () => {
    const out = await guard(() => api.suggest(idRef.current));
    if (out) apply(out);
  }, [apply]);

  const send = useCallback(
    async (ev: EventIn) => {
      const out = await guard(async () => {
        try {
          return await api.postEvent(idRef.current, ev);
        } catch (e) {
          // live re-scoring replaced the suggestion the user was looking at: the same decision, a newer row
          if (e instanceof ApiError && e.status === 409 && typeof e.body.replaced_by === "number" && ev.suggestion_id) {
            return await api.postEvent(idRef.current, { ...ev, suggestion_id: e.body.replaced_by });
          }
          throw e;
        }
      });
      if (out) apply(out);
      else void reload(); // the action failed: make sure the screen shows the server's truth
      return out;
    },
    [apply, reload],
  );

  const flip = useCallback(
    async (fn: (id: number) => Promise<StateOut>) => {
      const out = await guard(() => fn(idRef.current));
      if (out) apply(out);
    },
    [apply],
  );

  const undo = useCallback(() => flip(api.undo), [flip]);
  const redo = useCallback(() => flip(api.redo), [flip]);

  const markSeen = useCallback(async (unitIds: number[]) => {
    if (unitIds.length === 0) return;
    try {
      await api.markSeen(idRef.current, unitIds);
    } catch {
      return; // seen-tracking is best effort: it must never interrupt editing
    }
    if (!alive.current) return;
    const ids = new Set(unitIds);
    setProject((p) =>
      p && {
        ...p,
        seen_unit_ids: [...new Set([...p.seen_unit_ids, ...unitIds])].sort((a, b) => a - b),
        suggestions: p.suggestions.map((s) => (s.unit_id !== null && ids.has(s.unit_id) ? { ...s, seen: true } : s)),
      },
    );
  }, []);

  const finalize = useCallback(
    async (secondsActive: number) => {
      const out = await guard(() => api.finalize(idRef.current, secondsActive));
      if (out) apply(out);
      return out;
    },
    [apply],
  );

  return {
    project, loadError, error, busy, notice, fresh, reload, suggest, send, undo, redo, markSeen, finalize,
    dismiss: () => {
      setError(null);
      setNotice(null);
    },
  };
}
