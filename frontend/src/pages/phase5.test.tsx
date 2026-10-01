import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ProfileOut, ProjectSummary, StateOut, StylesOut } from "../api-types";
import { makeProject } from "../test/factory";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    createProject: vi.fn(), listProjects: vi.fn(), getProject: vi.fn(), getStyles: vi.fn(), postEvent: vi.fn(), undo: vi.fn(),
    redo: vi.fn(), suggest: vi.fn(), finalize: vi.fn(), markSeen: vi.fn(), explain: vi.fn(), startRender: vi.fn(), getJob: vi.fn(),
    listMusic: vi.fn(), uploadMusic: vi.fn(), getProfile: vi.fn(),
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
import { ApiError } from "../api";
import EditorPage from "./EditorPage";
import ProfilePage from "./ProfilePage";
import ProjectsPage from "./ProjectsPage";
import UploadPage from "./UploadPage";

const mocked = vi.mocked(api);
const file = (name: string, size = 100) => new File([new Uint8Array(size)], name, { type: "video/mp4" });
const choose = (...files: File[]) => fireEvent.change(screen.getByLabelText("Choose video files"), { target: { files } });

beforeEach(() => {
  vi.resetAllMocks();
  HTMLMediaElement.prototype.play = vi.fn(() => Promise.resolve()) as unknown as () => Promise<void>;
  HTMLMediaElement.prototype.pause = vi.fn();
});

describe("Batch upload", () => {
  const renderUpload = () =>
    render(
      <MemoryRouter initialEntries={["/upload"]}>
        <Routes>
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/" element={<div>projects home</div>} />
        </Routes>
      </MemoryRouter>,
    );
  const batchMode = () => fireEvent.click(screen.getByRole("radio", { name: /Batch/ }));

  it("uploads every file as its own auto-edited video, then goes to the project list", async () => {
    mocked.createProject.mockResolvedValue({ id: 1 } as never);
    renderUpload();
    choose(file("a.mp4"), file("b.mp4"), file("c.mov"));
    batchMode();
    expect(screen.queryByLabelText("Move b.mp4 up")).toBeNull(); // takes are not joined: no ordering
    fireEvent.change(screen.getByPlaceholderText("e.g. 30"), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("button", { name: "Upload and auto-edit" }));
    await screen.findByText("projects home");
    expect(mocked.createProject).toHaveBeenCalledTimes(3);
    const forms = mocked.createProject.mock.calls.map((c) => c[0] as FormData);
    expect(forms.map((f) => (f.getAll("files") as File[]).map((x) => x.name))).toEqual([["a.mp4"], ["b.mp4"], ["c.mov"]]);
    expect(forms.every((f) => f.get("autopilot") === "true" && f.get("target") === "30" && f.get("user_id") === "1")).toBe(true);
  });

  it("refuses more than five videos without calling the server", () => {
    renderUpload();
    choose(...["1", "2", "3", "4", "5", "6"].map((n) => file(`v${n}.mp4`)));
    batchMode();
    fireEvent.click(screen.getByRole("button", { name: "Upload and auto-edit" }));
    expect(screen.getByRole("alert")).toHaveTextContent("at most 5");
    expect(mocked.createProject).not.toHaveBeenCalled();
  });

  it("keeps the files that failed so they can be tried again, and says what went through", async () => {
    mocked.createProject
      .mockResolvedValueOnce({ id: 1 } as never)
      .mockRejectedValueOnce(new ApiError(422, "no speech"))
      .mockResolvedValueOnce({ id: 3 } as never);
    renderUpload();
    choose(file("a.mp4"), file("b.mp4"), file("c.mp4"));
    batchMode();
    fireEvent.click(screen.getByRole("button", { name: "Upload and auto-edit" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("2 of 3 uploaded. b.mp4: no speech");
    expect(within(screen.getByLabelText("Files in order")).getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByRole("link", { name: "Go to projects" })).toBeInTheDocument();
    mocked.createProject.mockResolvedValueOnce({ id: 4 } as never);
    fireEvent.click(screen.getByRole("button", { name: "Upload and auto-edit" }));
    await screen.findByText("projects home");
    expect(mocked.createProject).toHaveBeenCalledTimes(4);
  });
});

const base: ProjectSummary = {
  id: 1, video_id: "v_1", user_id: 1, title: "A", status: "ready", error: null, duration: 20, video_type: "vlog", target_duration: null,
  created_at: 0, updated_at: 0, has_thumbnail: false, autopilot: true, badge: null, n_review: 0, expected_corrections: null,
};

describe("Projects page: batch videos", () => {
  const open = (list: ProjectSummary[]) => {
    mocked.listProjects.mockResolvedValue(list);
    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );
  };

  it("shows ready / needs review / still auto-editing", async () => {
    open([
      { ...base, id: 1, title: "Good", badge: "ready", expected_corrections: 1.2 },
      { ...base, id: 2, title: "Unsure", badge: "review", n_review: 4, expected_corrections: 7.5 },
      { ...base, id: 3, title: "Busy", badge: null },
      { ...base, id: 4, title: "Normal", autopilot: false },
    ]);
    const ready = await screen.findByTestId("autopilot-1");
    expect(within(ready).getByText("Ready to post")).toBeInTheDocument();
    expect(within(ready).getByRole("button", { name: "Export now" })).toBeInTheDocument();
    const review = screen.getByTestId("autopilot-2");
    expect(within(review).getByRole("link", { name: "Needs review (4)" })).toHaveAttribute("href", "/projects/2?review=1");
    expect(within(review).getByText("~7.5 fixes expected")).toBeInTheDocument();
    expect(within(review).queryByRole("button", { name: "Export now" })).toBeNull();
    expect(within(screen.getByTestId("autopilot-3")).getByText("Auto-editing…")).toBeInTheDocument();
    expect(screen.queryByTestId("autopilot-4")).toBeNull(); // an ordinary project has no batch footer
    expect(screen.getByText("Ready to edit")).toBeInTheDocument(); // ... and keeps its status pill
  });

  it("exports a ready video in one click without saving a review, then offers the downloads", async () => {
    open([{ ...base, badge: "ready", expected_corrections: 1 }]);
    mocked.startRender.mockResolvedValue({ job: { id: "j1", kind: "render", status: "queued", error: null, meta: {} }, preset: "reels", duration: 20, warning: null });
    mocked.getJob.mockResolvedValue({ id: "j1", kind: "render", status: "done", error: null, meta: {} });
    fireEvent.click(await screen.findByRole("button", { name: "Export now" }));
    expect(screen.getByRole("button", { name: "Exporting…" })).toBeDisabled();
    const link = await screen.findByRole("link", { name: "Download .mp4" }, { timeout: 4000 });
    expect(link).toHaveAttribute("href", "/api/projects/1/download?fmt=mp4&preset=reels");
    expect(mocked.startRender).toHaveBeenCalledWith(1, "reels");
    expect(mocked.finalize).not.toHaveBeenCalled(); // nothing was reviewed: nothing is learned
  });

  it("shows why a one-click export failed and lets the user try again", async () => {
    open([{ ...base, badge: "ready" }]);
    mocked.startRender.mockRejectedValueOnce(new ApiError(409, "the music track is not available any more"));
    fireEvent.click(await screen.findByRole("button", { name: "Export now" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("music track");
    expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
  });

  it("keeps the download links on an exported batch video", async () => {
    open([{ ...base, status: "exported", badge: null }]);
    expect(await screen.findByRole("link", { name: "Download .mp4" })).toBeInTheDocument();
    expect(screen.getByText("Exported")).toBeInTheDocument();
  });
});

describe("Profile: trust score", () => {
  const profile = (trust: number | null): ProfileOut => ({
    user: { id: 1, name: "Tester", created_at: 0 }, n_projects: 3, n_finalized: 2, n_learning_events: 10, style_mixture: null, metrics: [],
    trust, trust_threshold: 0.85,
  });
  const open = (p: ProfileOut) => {
    mocked.getProfile.mockResolvedValue(p);
    render(<MemoryRouter><ProfilePage /></MemoryRouter>);
  };

  it("explains when there is no trust yet", async () => {
    open(profile(null));
    expect(await screen.findByTestId("trust")).toHaveTextContent("Not yet");
    expect(screen.getByTestId("trust")).toHaveTextContent("85%");
  });

  it("shows the score and whether it is enough for “Ready to post”", async () => {
    open(profile(0.9));
    expect(await screen.findByTestId("trust")).toHaveTextContent("90%");
    expect(screen.getByTestId("trust")).toHaveTextContent("can be marked “Ready to post”");
  });

  it("says what is still missing", async () => {
    open(profile(0.6));
    expect(await screen.findByTestId("trust")).toHaveTextContent("from 85%");
  });
});

const styles: StylesOut = {
  captions: { positions: {}, styles: { bold_pop: { label: "Bold pop", mode: "pop", font: "Inter", font_file: "i.ttf", size: 80, case: "upper", color: "#fff", emphasis_color: "#ff0", outline_color: "#000", outline: 8, shadow: 0, max_width: 0.85 } } },
  texts: { positions: {}, styles: {} },
  presets: { reels: { label: "Instagram Reels", width: 1080, height: 1920, fps: 30, crf: 20, audio_kbps: 160, max_duration_s: 90, safe_zones: { top: 0.1, bottom: 0.2, right: 0.1 } } },
};
const stateOf = (p = makeProject(), over: Partial<StateOut> = {}): StateOut => ({
  status: p.status, settings: p.settings, suggestions: p.suggestions, queue: p.queue, edl: p.edl, style_mixture: null,
  can_undo: false, can_redo: false, n_events: 0, changes: [], message: null, events: [], caption_breaks: [], text_breaks: {},
  caption_word_ids: p.caption_word_ids, ...over,
});

async function openEditor(url = "/projects/7", project = makeProject()) {
  mocked.getProject.mockResolvedValue(project);
  mocked.getStyles.mockResolvedValue(styles);
  mocked.postEvent.mockResolvedValue(stateOf(project));
  mocked.explain.mockResolvedValue({ suggestion_id: 1, text: "because", confidence: 0.4, reasons: [] });
  mocked.listMusic.mockResolvedValue([{ id: "lib_sunrise", name: "Sunrise", mood: "calm", duration: 48, source: "library" }]);
  render(
    <MemoryRouter initialEntries={[url]}>
      <Routes>
        <Route path="/projects/:id" element={<EditorPage />} />
      </Routes>
    </MemoryRouter>,
  );
  await screen.findByTestId("player");
}
const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

describe("Editor: Phase 5 controls", () => {
  it("a global nudge is one nudge event", async () => {
    await openEditor();
    fireEvent.click(screen.getByRole("button", { name: "Snappier" }));
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "nudge", payload: { name: "snappier", magnitude: 2 } });
  });

  it("shows the note when a nudge re-scored suggestions", async () => {
    await openEditor();
    mocked.postEvent.mockResolvedValueOnce(stateOf(makeProject(), { message: "Adjusted 5 suggestions based on your last change" }));
    fireEvent.click(screen.getByRole("button", { name: "More zooms" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Adjusted 5 suggestions");
  });

  it("picking a track sends a music event; the tab lists the library", async () => {
    await openEditor();
    fireEvent.click(screen.getByRole("tab", { name: "Music" }));
    fireEvent.click(await screen.findByRole("radio", { name: /Sunrise/ }));
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "music", payload: { track_id: "lib_sunrise" } });
  });

  it("clicking a caption word sends an emphasis edit", async () => {
    const p = makeProject();
    p.edl!.captions.lines = [{ t0: 0, t1: 2, words: [{ t0: 0, t1: 1, text: "hello", emphasis: false }, { t0: 1, t1: 2, text: "there", emphasis: false }] }];
    p.caption_word_ids = [[0, 1]];
    await openEditor("/projects/7", p);
    fireEvent.click(within(screen.getByTestId("caption-strip")).getByRole("button", { name: "there" }));
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "manual", edit: { kind: "set_emphasis", word_id: 1, line_word_ids: [0, 1] } });
  });

  it("opens on the uncertain decisions only when sent from “Needs review”, and can show everything", async () => {
    await openEditor("/projects/7?review=1");
    expect(screen.queryByText(/Confident edits/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show everything" }));
    expect(await screen.findByText(/Confident edits/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show everything" })).toBeNull();
  });

  it("an ordinary open shows everything", async () => {
    await openEditor();
    expect(screen.getByText(/Confident edits/)).toBeInTheDocument();
  });

  it("the suggested title is shown as such in the Style tab and can be deleted", async () => {
    const p = makeProject();
    p.edl!.texts = [{ id: "hook", text: "Hello there", t0: 0, t1: 3, position: "top", y: null, style_id: "title" }];
    await openEditor("/projects/7", p);
    fireEvent.click(screen.getByRole("tab", { name: "Style" }));
    expect(screen.getByText(/Suggested title/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Delete text"));
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "text_overlay", payload: { op: "delete", id: "hook" } });
  });

  it("dragging a text bar on the timeline updates its times", async () => {
    const p = makeProject();
    p.edl!.texts = [{ id: "hook", text: "Hello", t0: 0.5, t1: 1.5, position: "top", y: null, style_id: "title" }];
    await openEditor("/projects/7", p);
    const bar = screen.getByTestId("text-bar-hook");
    bar.getBoundingClientRect = () => ({ left: 100, right: 124, top: 0, bottom: 20, width: 24, height: 20, x: 100, y: 0, toJSON: () => ({}) });
    fireEvent.pointerDown(bar, { clientX: 112, button: 0, pointerId: 1 });
    fireEvent.pointerMove(bar, { clientX: 136, pointerId: 1 });
    fireEvent.pointerUp(bar, { clientX: 136, pointerId: 1 });
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "text_overlay", payload: { op: "update", id: "hook", t0: 1.5, t1: 2.5 } });
  });
});

describe("Editor: remembered defaults show up as the starting state", () => {
  it("renders the settings and music the server started the project with", async () => {
    const p = makeProject();
    p.settings = { ...p.settings, caption_style: "bold_pop", speed: 1.15 };
    p.edl!.output.speed = 1.15;
    p.edl!.audio.music = { track_id: "lib_sunrise", gain_db: -22, duck_db: -8 };
    await openEditor("/projects/7", p);
    fireEvent.click(screen.getByRole("tab", { name: "Style" }));
    expect(screen.getByRole("radio", { name: "1.15×" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("tab", { name: "Music" }));
    expect(await screen.findByLabelText("Music volume")).toHaveValue("-22");
    expect(screen.getByLabelText("Dip under voice")).toHaveValue("-8");
    expect(screen.getByTestId("music-preview").getAttribute("src")).toBe("/api/music/lib_sunrise/audio?user_id=1");
    await waitFor(() => expect(mocked.postEvent).not.toHaveBeenCalled()); // defaults are not events
  });
});
