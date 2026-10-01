import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StateOut, StylesOut } from "../api-types";
import { makeProfile, makeProject } from "../test/factory";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    getProject: vi.fn(), getStyles: vi.fn(), postEvent: vi.fn(), undo: vi.fn(), redo: vi.fn(), suggest: vi.fn(), finalize: vi.fn(),
    markSeen: vi.fn(), explain: vi.fn(), listMusic: vi.fn(), getProfile: vi.fn(),
  };
});
vi.mock("../hooks/useUser", () => ({
  useUser: () => ({ user: { id: 1, name: "Tester", created_at: 0 }, users: [], loading: false, error: null, choose: vi.fn(), create: vi.fn() }),
}));
const player = vi.hoisted(() => ({ toggle: vi.fn(), seekSrc: vi.fn(), seekOut: vi.fn(), pause: vi.fn() }));
vi.mock("../components/PreviewPlayer", async () => {
  const React = await import("react");
  return {
    PreviewPlayer: React.forwardRef((_props: unknown, ref: React.Ref<unknown>) => {
      React.useImperativeHandle(ref, () => ({
        toggle: player.toggle, play: vi.fn(), pause: player.pause, seekSrc: player.seekSrc, seekOut: player.seekOut,
        getSrcTime: () => 0, getOutTime: () => 1, isPlaying: () => false,
      }));
      return <div data-testid="player" />;
    }),
  };
});

import * as api from "../api";
import EditorPage from "./EditorPage";
import ProfilePage from "./ProfilePage";

const mocked = vi.mocked(api);

const styles: StylesOut = {
  captions: { positions: {}, styles: { bold_pop: { label: "Bold pop", mode: "pop", font: "Inter", font_file: "i.ttf", size: 80, case: "upper", color: "#fff", emphasis_color: "#ff0", outline_color: "#000", outline: 8, shadow: 0, max_width: 0.85 } } },
  texts: { positions: {}, styles: {} },
  presets: { reels: { label: "Instagram Reels", width: 1080, height: 1920, fps: 30, crf: 20, audio_kbps: 160, max_duration_s: 90, safe_zones: { top: 0.1, bottom: 0.2, right: 0.1 } } },
};

const stateOf = (p = makeProject(), over: Partial<StateOut> = {}): StateOut => ({
  status: p.status, settings: p.settings, suggestions: p.suggestions, queue: p.queue, edl: p.edl, style_mixture: p.style_mixture,
  can_undo: false, can_redo: false, n_events: 0, changes: [], message: null, events: [], caption_breaks: [], text_breaks: {},
  caption_word_ids: p.caption_word_ids, ...over,
});

async function openEditor(project = makeProject()) {
  mocked.getProject.mockResolvedValue(project);
  mocked.getStyles.mockResolvedValue(styles);
  mocked.postEvent.mockResolvedValue(stateOf(project));
  mocked.explain.mockResolvedValue({ suggestion_id: 1, text: "because", confidence: 0.4, reasons: [] });
  mocked.listMusic.mockResolvedValue([]);
  render(
    <MemoryRouter initialEntries={["/projects/7"]}>
      <Routes>
        <Route path="/projects/:id" element={<EditorPage />} />
      </Routes>
    </MemoryRouter>,
  );
  await screen.findByTestId("player");
  await flush(); // passive effects (the keyboard listener) run after the commit that findBy saw
}
const key = (k: string, init: KeyboardEventInit = {}) => fireEvent.keyDown(document.body, { key: k, ...init });
const flush = () => act(async () => { for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 10)); }); // a few turns of the event loop, so a loaded machine cannot outrun it

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
  HTMLMediaElement.prototype.play = vi.fn(() => Promise.resolve()) as unknown as () => Promise<void>;
  HTMLMediaElement.prototype.pause = vi.fn();
});

describe("Editor: review polish", () => {
  /** What the server answers after suggestion 1 was accepted: it is decided, the next uncertain one is #2. */
  const afterAccept = () => {
    const p = makeProject();
    p.suggestions = p.suggestions.map((s) => (s.id === 1 ? { ...s, status: "accepted" as const } : s));
    p.queue = { needs_review: [2], confident: [3] };
    return stateOf(p);
  };

  it("shows how far the review is and the user's style", async () => {
    const p = makeProject();
    p.suggestions = p.suggestions.map((s) => (s.id === 3 ? { ...s, status: "accepted" as const } : s));
    p.queue = { needs_review: [1, 2], confident: [] };
    p.style_mixture = { weights: { snappy: 0.7, pooled: 0.3 }, n_events: 4, frozen: false, text: "Your style so far: 70% Snappy", top: "snappy", identified: false };
    await openEditor(p);
    const box = screen.getByTestId("review-progress");
    expect(box).toHaveTextContent("Reviewed 1 of 3");
    expect(box).toHaveTextContent("2 uncertain left");
    expect(within(box).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "1");
    expect(within(box).getByTestId("style-text")).toHaveTextContent("Your style so far: 70% Snappy");
  });

  it("moves on to the next uncertain suggestion after a decision, and seeks to it", async () => {
    await openEditor();
    key("j"); // suggestion 1 (the phrase at 1.4 s)
    mocked.postEvent.mockResolvedValueOnce(afterAccept());
    player.seekSrc.mockClear();
    key("a");
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "accept", suggestion_id: 1 });
    expect(screen.getByTestId("queue-item-2").dataset.selected).toBe("1");
    expect(player.seekSrc).toHaveBeenLastCalledWith(2.4); // the start of the pause that is next in the queue
  });

  it("does the same for the buttons of a row", async () => {
    await openEditor();
    mocked.postEvent.mockResolvedValueOnce(afterAccept());
    fireEvent.click(within(screen.getByTestId("queue-item-1")).getByLabelText("Accept"));
    await flush();
    expect(screen.getByTestId("queue-item-2").dataset.selected).toBe("1");
  });

  it("stays where it is when 'next after each' is off, and remembers the choice", async () => {
    await openEditor();
    fireEvent.click(screen.getByLabelText("next after each"));
    expect(localStorage.getItem("roughcut.autoAdvance")).toBe("0");
    key("j");
    mocked.postEvent.mockResolvedValueOnce(afterAccept());
    key("a");
    await flush();
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1");
    expect(screen.getByTestId("queue-item-2").dataset.selected).toBe("0");
  });

  it("starts with the setting that was saved", async () => {
    localStorage.setItem("roughcut.autoAdvance", "0");
    await openEditor();
    expect(screen.getByLabelText("next after each")).not.toBeChecked();
  });

  it("does not move when nothing uncertain is left", async () => {
    await openEditor();
    key("j");
    const p = makeProject();
    p.suggestions = p.suggestions.map((s) => (s.id === 3 ? s : { ...s, status: "accepted" as const }));
    p.queue = { needs_review: [], confident: [3] };
    mocked.postEvent.mockResolvedValueOnce(stateOf(p));
    key("a");
    await flush();
    expect(screen.getByTestId("review-progress")).toHaveTextContent("nothing uncertain left");
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1"); // unchanged: no uncertain item to go to
  });

  it("does not move after a refused action (the server said no)", async () => {
    await openEditor();
    key("j");
    mocked.postEvent.mockRejectedValueOnce(new api.ApiError(409, "already accepted"));
    key("a");
    await flush();
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1");
  });
});

describe("Profile: what it has learned", () => {
  const open = async (over = {}) => {
    mocked.getProfile.mockResolvedValue(makeProfile(over));
    render(<MemoryRouter><ProfilePage /></MemoryRouter>);
    await screen.findByTestId("style");
  };

  it("says there is no style yet and nothing learned for a new user", async () => {
    await open();
    expect(screen.getByTestId("style-text")).toHaveTextContent("No style yet");
    expect(screen.getByTestId("preferences")).toHaveTextContent("Not enough yet");
    expect(screen.getByTestId("stats")).toHaveTextContent("Finish and export a video");
    expect(screen.queryByText("identified")).toBeNull();
  });

  it("shows the style sentence, the bars with readable names, and the identified badge", async () => {
    await open({
      style_mixture: { weights: { pooled: 0.02, snappy: 0.9, relaxed: 0.08 }, n_events: 25, frozen: false, text: "Your style so far: 90% Snappy", top: "snappy", identified: true },
      style_labels: { pooled: { label: "Generic", blurb: "what most do" }, snappy: { label: "Snappy", blurb: "tight pace" }, relaxed: { label: "Relaxed", blurb: "breathing room" } },
    });
    const box = screen.getByTestId("style");
    expect(within(box).getByTestId("style-text")).toHaveTextContent("Your style so far: 90% Snappy");
    expect(within(box).getByText("identified")).toBeInTheDocument();
    expect(within(box).getByText("Snappy: tight pace.")).toBeInTheDocument();
    const names = within(box).getAllByRole("listitem").map((li) => li.textContent);
    expect(names[0]).toContain("Snappy");
    expect(names[0]).toContain("90%");
    expect(names[1]).toContain("Relaxed");
  });

  it("lists what it learned, grouped by decision in a fixed order", async () => {
    await open({
      preferences: [
        { decision_type: "zoom", text: "You rarely zoom." },
        { decision_type: "pause", text: "You remove pauses longer than ~0.5 s." },
        { decision_type: "filler", text: "You remove 'um' and 'uh' but keep the other filler words." },
      ],
    });
    const items = within(screen.getByTestId("preferences")).getAllByRole("listitem").map((li) => li.textContent ?? "");
    expect(items).toHaveLength(3);
    expect(items[0]).toContain("Pauses");
    expect(items[0]).toContain("You remove pauses longer than ~0.5 s.");
    expect(items[1]).toContain("Fillers");
    expect(items[2]).toContain("You rarely zoom.");
  });

  it("shows the editing statistics", async () => {
    await open({ stats: { n_videos: 2, avg_clip_seconds: 3.46, cuts_per_minute: 9.04, zooms_per_minute: 2, percent_kept: 61.6, fillers_removed_per_minute: 1.25 } });
    const box = screen.getByTestId("stats");
    expect(box).toHaveTextContent("3.5 s");
    expect(box).toHaveTextContent("average clip length");
    expect(box).toHaveTextContent("9");
    expect(box).toHaveTextContent("62%");
    expect(box).toHaveTextContent("1.3");
  });

  it("has the three curves: learning, effort and time in the editor", async () => {
    await open();
    for (const k of ["acceptance", "manual_per_minute", "minutes_active"]) expect(screen.getByTestId(`curve-${k}`)).toBeInTheDocument();
    expect(screen.getByText("Learning curve")).toBeInTheDocument();
    expect(screen.getByText("Effort curve")).toBeInTheDocument();
    expect(screen.getAllByText(/Export a first video/)).toHaveLength(3);
  });
});
