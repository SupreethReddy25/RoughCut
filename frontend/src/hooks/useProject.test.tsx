import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ProjectOut, StateOut } from "../api-types";
import { makeProject } from "../test/factory";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    getProject: vi.fn(),
    postEvent: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    suggest: vi.fn(),
    finalize: vi.fn(),
    markSeen: vi.fn(),
  };
});

import * as api from "../api";
import { ApiError } from "../api";
import { mergeState, useProject } from "./useProject";

const mocked = vi.mocked(api);

/** The state a mutation answers with, derived from a project. */
function stateOf(p: ProjectOut, over: Partial<StateOut> = {}): StateOut {
  return {
    status: p.status, settings: p.settings, suggestions: p.suggestions, queue: p.queue, edl: p.edl, style_mixture: null,
    can_undo: p.can_undo, can_redo: p.can_redo, n_events: p.n_events, changes: [], message: null, events: [],
    caption_breaks: p.caption_breaks, text_breaks: p.text_breaks, caption_word_ids: p.caption_word_ids, ...over,
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  mocked.getProject.mockResolvedValue(makeProject());
});

describe("useProject", () => {
  it("loads the project and reports a load failure", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project?.id).toBe(7));

    mocked.getProject.mockRejectedValue(new ApiError(404, "project 8 not found"));
    const bad = renderHook(() => useProject(8));
    await waitFor(() => expect(bad.result.current.loadError).toBe("project 8 not found"));
    expect(bad.result.current.project).toBeNull();
  });

  it("replaces the state with what the server says, keeping the heavy parts", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    const next = stateOf(base, { can_undo: true, n_events: 1, message: "Adjusted 2 suggestions", suggestions: base.suggestions.slice(1) });
    mocked.postEvent.mockResolvedValue(next);
    await act(async () => {
      await result.current.send({ kind: "accept", suggestion_id: 1 });
    });
    expect(result.current.project!.can_undo).toBe(true);
    expect(result.current.project!.suggestions).toHaveLength(2);
    expect(result.current.project!.units).toBe(base.units); // not re-sent, not rebuilt
    expect(result.current.notice).toBe("Adjusted 2 suggestions");
  });

  it("sends actions in the order they were made, one at a time", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    const log: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    mocked.postEvent.mockImplementation(async (_id, ev) => {
      log.push(`start ${ev.suggestion_id}`);
      if (ev.suggestion_id === 1) await gate; // the first one is slow
      log.push(`end ${ev.suggestion_id}`);
      return stateOf(base);
    });
    let p1!: Promise<unknown>, p2!: Promise<unknown>;
    act(() => {
      p1 = result.current.send({ kind: "accept", suggestion_id: 1 });
      p2 = result.current.send({ kind: "accept", suggestion_id: 2 });
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(log).toEqual(["start 1"]); // the second waits for the first
    release();
    await act(async () => {
      await Promise.all([p1, p2]);
    });
    expect(log).toEqual(["start 1", "end 1", "start 2", "end 2"]);
  });

  it("retries once on the suggestion that replaced the one acted on", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    mocked.postEvent
      .mockRejectedValueOnce(new ApiError(409, "replaced", { replaced_by: 42 }))
      .mockResolvedValueOnce(stateOf(base, { n_events: 5 }));
    await act(async () => {
      await result.current.send({ kind: "reject", suggestion_id: 1 });
    });
    expect(mocked.postEvent).toHaveBeenCalledTimes(2);
    expect(mocked.postEvent.mock.calls[1][1]).toEqual({ kind: "reject", suggestion_id: 42 });
    expect(result.current.project!.n_events).toBe(5);
    expect(result.current.error).toBeNull();
  });

  it("shows a failure, keeps going afterwards, and re-reads the server's truth", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    mocked.postEvent.mockRejectedValueOnce(new ApiError(409, "this suggestion is already accepted")).mockResolvedValueOnce(stateOf(base, { n_events: 9 }));
    await act(async () => {
      expect(await result.current.send({ kind: "accept", suggestion_id: 1 })).toBeNull();
    });
    expect(result.current.error).toBe("this suggestion is already accepted");
    expect(mocked.getProject).toHaveBeenCalledTimes(2); // initial load + reload after the failure
    await act(async () => {
      await result.current.send({ kind: "accept", suggestion_id: 2 });
    });
    expect(result.current.error).toBeNull();
    expect(result.current.project!.n_events).toBe(9);
  });

  it("marks highlighted rows after re-scoring and undo/redo use the server answer", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    mocked.postEvent.mockResolvedValue(stateOf(base, { changes: [{ old_id: 1, new_id: 11, decision_type: "phrase", target_id: 2, from_action: "KEEP", to_action: "CUT" }] }));
    await act(async () => {
      await result.current.send({ kind: "accept", suggestion_id: 1 });
    });
    expect([...result.current.fresh]).toEqual([11]);

    mocked.undo.mockResolvedValue(stateOf(base, { can_undo: false, can_redo: true }));
    await act(async () => {
      await result.current.undo();
    });
    expect(result.current.project!.can_redo).toBe(true);
    mocked.redo.mockResolvedValue(stateOf(base, { can_undo: true, can_redo: false }));
    await act(async () => {
      await result.current.redo();
    });
    expect(result.current.project!.can_undo).toBe(true);
  });

  it("marks units as seen locally once the server has them, and ignores a failed report", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    mocked.markSeen.mockResolvedValueOnce({ marked: 1 });
    await act(async () => {
      await result.current.markSeen([2]);
    });
    expect(result.current.project!.seen_unit_ids).toEqual([2]);
    expect(result.current.project!.suggestions.find((s) => s.unit_id === 2)!.seen).toBe(true);
    expect(result.current.project!.suggestions.find((s) => s.unit_id === 4)!.seen).toBe(false);

    mocked.markSeen.mockRejectedValueOnce(new Error("offline"));
    await act(async () => {
      await result.current.markSeen([4]);
    });
    expect(result.current.project!.seen_unit_ids).toEqual([2]);
    expect(result.current.error).toBeNull(); // best effort: never interrupts editing
    await act(async () => {
      await result.current.markSeen([]);
    });
    expect(mocked.markSeen).toHaveBeenCalledTimes(2);
  });

  it("merges a state into a project without touching the transcript", () => {
    const p = makeProject();
    const merged = mergeState(p, stateOf(p, { can_redo: true }));
    expect(merged.can_redo).toBe(true);
    expect(merged.words).toBe(p.words);
    expect(merged.title).toBe(p.title);
  });

  it("finalizes with the active seconds", async () => {
    const { result } = renderHook(() => useProject(7));
    await waitFor(() => expect(result.current.project).not.toBeNull());
    const base = result.current.project!;
    mocked.finalize.mockResolvedValue({ ...stateOf(base, { status: "edited" }), metrics: {}, implicit_events: 3 });
    let out: unknown;
    await act(async () => {
      out = await result.current.finalize(41);
    });
    expect(mocked.finalize).toHaveBeenCalledWith(7, 41);
    expect(out).not.toBeNull();
    expect(result.current.project!.status).toBe("edited");
  });
});
