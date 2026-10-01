import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { makeProject } from "../test/factory";
import type { StateOut, StylesOut } from "../api-types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    createProject: vi.fn(),
    listProjects: vi.fn(),
    getProject: vi.fn(),
    getStyles: vi.fn(),
    postEvent: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    suggest: vi.fn(),
    finalize: vi.fn(),
    markSeen: vi.fn(),
    explain: vi.fn(),
    startRender: vi.fn(),
    getJob: vi.fn(),
  };
});
vi.mock("../hooks/useUser", () => ({
  useUser: () => ({ user: { id: 1, name: "Tester", created_at: 0 }, users: [], loading: false, error: null, choose: vi.fn(), create: vi.fn() }),
}));

// jsdom cannot play video: the real player is covered by the browser run; here only its handle matters.
const player = vi.hoisted(() => ({ toggle: vi.fn(), seekSrc: vi.fn(), seekOut: vi.fn(), pause: vi.fn(), out: 1 }));
vi.mock("../components/PreviewPlayer", async () => {
  const React = await import("react");
  return {
    PreviewPlayer: React.forwardRef((_props: unknown, ref: React.Ref<unknown>) => {
      React.useImperativeHandle(ref, () => ({
        toggle: player.toggle, play: vi.fn(), pause: player.pause, seekSrc: player.seekSrc, seekOut: player.seekOut,
        getSrcTime: () => 0, getOutTime: () => player.out, isPlaying: () => false,
      }));
      return <div data-testid="player" />;
    }),
  };
});

import * as api from "../api";
import { ApiError } from "../api";
import EditorPage from "./EditorPage";
import ProjectsPage from "./ProjectsPage";
import UploadPage from "./UploadPage";

const mocked = vi.mocked(api);

const file = (name: string, size = 100) => new File([new Uint8Array(size)], name, { type: "video/mp4" });

function renderUpload() {
  return render(
    <MemoryRouter initialEntries={["/upload"]}>
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/projects/:id" element={<div>editor opened</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const choose = (...files: File[]) => fireEvent.change(screen.getByLabelText("Choose video files"), { target: { files } });

beforeEach(() => {
  vi.resetAllMocks();
});

describe("Upload page", () => {
  it("joins the takes in the order shown and sends type, target and title", async () => {
    mocked.createProject.mockResolvedValue({ id: 5 } as never);
    renderUpload();
    choose(file("a.mp4"), file("b.mov"), file("c.mp4"));
    fireEvent.click(screen.getByLabelText("Move c.mp4 up"));
    fireEvent.click(screen.getByLabelText("Move a.mp4 down"));
    // [a, b, c] -> c up: [a, c, b] -> a down: [c, a, b]
    expect(within(screen.getByLabelText("Files in order")).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      expect.stringContaining("c.mp4"), expect.stringContaining("a.mp4"), expect.stringContaining("b.mov"),
    ]);
    fireEvent.click(screen.getByRole("radio", { name: /Opinion/ }));
    fireEvent.change(screen.getByPlaceholderText("e.g. 30"), { target: { value: "30" } });
    fireEvent.change(screen.getByPlaceholderText("defaults to the file name"), { target: { value: "  My reel " } });
    fireEvent.click(screen.getByRole("button", { name: "Upload and analyse" }));

    await screen.findByText("editor opened");
    const form = mocked.createProject.mock.calls[0][0] as FormData;
    expect((form.getAll("files") as File[]).map((f) => f.name)).toEqual(["c.mp4", "a.mp4", "b.mov"]);
    expect(form.get("user_id")).toBe("1");
    expect(form.get("type")).toBe("opinion");
    expect(form.get("target")).toBe("30");
    expect(form.get("title")).toBe("My reel");
  });

  it("removes a file and refuses an unsupported one without calling the server", async () => {
    renderUpload();
    choose(file("notes.txt"));
    fireEvent.click(screen.getByRole("button", { name: "Upload and analyse" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not a supported video file");
    fireEvent.click(screen.getByLabelText("Remove notes.txt"));
    expect(screen.queryByLabelText("Files in order")).toBeNull();
    expect(mocked.createProject).not.toHaveBeenCalled();
  });

  it("refuses a bad target length", () => {
    renderUpload();
    choose(file("a.mp4"));
    fireEvent.change(screen.getByPlaceholderText("e.g. 30"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Upload and analyse" }));
    expect(screen.getByRole("alert")).toHaveTextContent("5 to 180 seconds");
    expect(mocked.createProject).not.toHaveBeenCalled();
  });

  it("shows the server's reason and lets the user try again", async () => {
    mocked.createProject.mockRejectedValueOnce(new ApiError(422, "the clips are 4 minutes in total: at most 3"));
    renderUpload();
    choose(file("a.mp4"));
    fireEvent.click(screen.getByRole("button", { name: "Upload and analyse" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("at most 3");
    expect(screen.getByRole("button", { name: "Upload and analyse" })).toBeEnabled();
  });
});

describe("Projects page", () => {
  it("lists projects with their state, and says so when there are none", async () => {
    mocked.listProjects.mockResolvedValueOnce([]);
    const { unmount } = render(<MemoryRouter><ProjectsPage /></MemoryRouter>);
    expect(await screen.findByText(/No projects yet/)).toBeInTheDocument();
    unmount();

    const base = { video_id: "v_1", user_id: 1, error: null, duration: 12, video_type: "vlog", target_duration: null, created_at: 0, updated_at: 0, has_thumbnail: false, autopilot: false, badge: null, n_review: 0, expected_corrections: null } as const;
    mocked.listProjects.mockResolvedValueOnce([
      { ...base, id: 1, title: "First", status: "ready" },
      { ...base, id: 2, title: "Second", status: "failed", error: "no speech found" },
    ]);
    render(<MemoryRouter><ProjectsPage /></MemoryRouter>);
    expect(await screen.findByText("First")).toBeInTheDocument();
    expect(screen.getByText("Ready to edit")).toBeInTheDocument();
    expect(screen.getByText("no speech found")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /First/ })).toHaveAttribute("href", "/projects/1");
  });

  it("shows a load error", async () => {
    mocked.listProjects.mockRejectedValueOnce(new ApiError(0, "The server is unreachable"));
    render(<MemoryRouter><ProjectsPage /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("unreachable");
  });
});

const styles: StylesOut = {
  captions: { positions: {}, styles: { bold_pop: { label: "Bold pop", mode: "pop", font: "Inter", font_file: "i.ttf", size: 80, case: "upper", color: "#fff", emphasis_color: "#ff0", outline_color: "#000", outline: 8, shadow: 0, max_width: 0.85 } } },
  texts: { positions: {}, styles: {} },
  presets: { reels: { label: "Instagram Reels", width: 1080, height: 1920, fps: 30, crf: 20, audio_kbps: 160, max_duration_s: 90, safe_zones: { top: 0.1, bottom: 0.2, right: 0.1 } } },
};

const stateOf = (p = makeProject(), over: Partial<StateOut> = {}): StateOut => ({
  status: p.status, settings: p.settings, suggestions: p.suggestions, queue: p.queue, edl: p.edl, style_mixture: null,
  can_undo: p.can_undo, can_redo: p.can_redo, n_events: p.n_events, changes: [], message: null, events: [],
  caption_breaks: [], text_breaks: {}, caption_word_ids: [], ...over,
});

async function openEditor(project = makeProject()) {
  mocked.getProject.mockResolvedValue(project);
  mocked.getStyles.mockResolvedValue(styles);
  mocked.postEvent.mockResolvedValue(stateOf(project));
  mocked.undo.mockResolvedValue(stateOf(project));
  mocked.redo.mockResolvedValue(stateOf(project));
  mocked.explain.mockResolvedValue({ suggestion_id: 1, text: "because", confidence: 0.4, reasons: [] });
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

describe("Editor keyboard", () => {
  it("J selects the least confident suggestion and seeks to it; A accepts it", async () => {
    await openEditor();
    key("j");
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1");
    expect(player.seekSrc).toHaveBeenCalledWith(1.4); // the start of its phrase
    key("a");
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "accept", suggestion_id: 1 });
  });

  it("R rejects, 1-4 choose, and nothing happens with nothing selected", async () => {
    await openEditor();
    key("r");
    key("2");
    await flush();
    expect(mocked.postEvent).not.toHaveBeenCalled();

    key("j"); // phrase 1: KEEP / CUT
    key("r");
    await flush();
    expect(mocked.postEvent).toHaveBeenLastCalledWith(7, { kind: "reject", suggestion_id: 1 });
    key("2"); // CUT
    await flush();
    expect(mocked.postEvent).toHaveBeenLastCalledWith(7, { kind: "override", suggestion_id: 1, final_action: "CUT" });
    key("1"); // KEEP is the current action of a pending suggestion: an accept
    await flush();
    expect(mocked.postEvent).toHaveBeenLastCalledWith(7, { kind: "accept", suggestion_id: 1 });
    key("4"); // phrase decisions have two actions only
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledTimes(3);
  });

  it("J walks through the queue and wraps", async () => {
    await openEditor();
    key("j");
    key("j");
    expect(screen.getByTestId("queue-item-2").dataset.selected).toBe("1");
    key("j");
    key("j");
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1");
  });

  it("Space plays, arrows seek in output time, and typing in a field leaves keys alone", async () => {
    await openEditor();
    key(" ");
    expect(player.toggle).toHaveBeenCalledTimes(1);
    key("ArrowRight");
    expect(player.seekOut).toHaveBeenLastCalledWith(1.5);
    key("ArrowLeft", { shiftKey: true });
    expect(player.seekOut).toHaveBeenLastCalledWith(0); // clamped at the start

    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    fireEvent.keyDown(input, { key: " " });
    fireEvent.keyDown(input, { key: "j" });
    expect(player.toggle).toHaveBeenCalledTimes(1);
    input.remove();
  });

  it("a mouse-clicked button gives up focus (so Space plays), a keyboard-pressed one keeps it", async () => {
    await openEditor(makeProject({ can_undo: true }));
    const play = screen.getByRole("button", { name: /Play/ });
    play.focus();
    fireEvent.click(play, { detail: 1 });
    expect(document.activeElement).not.toBe(play);
    const undo = screen.getByRole("button", { name: /Undo/ });
    undo.focus();
    fireEvent.click(undo, { detail: 0 });
    expect(document.activeElement).toBe(undo);
  });

  it("Ctrl+Z / Ctrl+Shift+Z undo and redo only when there is something to do", async () => {
    await openEditor();
    key("z", { ctrlKey: true });
    key("z", { ctrlKey: true, shiftKey: true });
    await flush();
    expect(mocked.undo).not.toHaveBeenCalled();
    expect(mocked.redo).not.toHaveBeenCalled();
    key("a", { ctrlKey: true }); // not ours
  });

  it("undoes and redoes when the server says it can", async () => {
    await openEditor(makeProject({ can_undo: true, can_redo: true }));
    key("z", { ctrlKey: true });
    await flush();
    expect(mocked.undo).toHaveBeenCalledWith(7);
    key("z", { ctrlKey: true, shiftKey: true });
    await flush();
    expect(mocked.redo).toHaveBeenCalledWith(7);
  });

  it("Delete removes the selected text overlay", async () => {
    const p = makeProject();
    p.edl!.texts = [{ id: "t1", text: "Hello", t0: 0.5, t1: 2, position: "top", y: null, style_id: "title" }];
    await openEditor(p);
    fireEvent.click(screen.getByTestId("text-bar-t1"));
    key("Delete");
    await flush();
    expect(mocked.postEvent).toHaveBeenCalledWith(7, { kind: "text_overlay", payload: { op: "delete", id: "t1" } });
  });

  it("shows a refused action as a banner that can be dismissed", async () => {
    await openEditor();
    mocked.postEvent.mockRejectedValue(new ApiError(409, "this suggestion is already accepted"));
    key("j");
    key("a");
    expect(await screen.findByRole("alert")).toHaveTextContent("already accepted");
    fireEvent.click(screen.getByLabelText("Dismiss"));
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("Editor states", () => {
  it("polls while the video is being analysed, then opens the editor", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      mocked.getStyles.mockResolvedValue(styles);
      mocked.getProject.mockResolvedValueOnce(makeProject({ status: "processing", edl: null, suggestions: [] })).mockResolvedValue(makeProject());
      render(
        <MemoryRouter initialEntries={["/projects/7"]}>
          <Routes><Route path="/projects/:id" element={<EditorPage />} /></Routes>
        </MemoryRouter>,
      );
      expect(await screen.findByText(/Analysing your video/)).toBeInTheDocument();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });
      await screen.findByTestId("player");
    } finally {
      vi.useRealTimers();
    }
  });

  it("makes the suggestions once for an analysed project that has none", async () => {
    const bare = makeProject({ suggestions: [], queue: { needs_review: [], confident: [] } });
    mocked.getProject.mockResolvedValue(bare);
    mocked.getStyles.mockResolvedValue(styles);
    mocked.suggest.mockResolvedValue(stateOf(makeProject()));
    render(
      <MemoryRouter initialEntries={["/projects/7"]}>
        <Routes><Route path="/projects/:id" element={<EditorPage />} /></Routes>
      </MemoryRouter>,
    );
    await screen.findByTestId("player");
    expect(mocked.suggest).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByTestId("queue-item-1")).toBeInTheDocument());
  });

  it("shows why a project failed", async () => {
    mocked.getProject.mockResolvedValue(makeProject({ status: "failed", error: "no speech found" }));
    mocked.getStyles.mockResolvedValue(styles);
    render(
      <MemoryRouter initialEntries={["/projects/7"]}>
        <Routes><Route path="/projects/:id" element={<EditorPage />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText("no speech found")).toBeInTheDocument();
  });

  it("offers a retry when the project cannot be loaded", async () => {
    mocked.getProject.mockRejectedValueOnce(new ApiError(404, "project 7 not found")).mockResolvedValue(makeProject());
    mocked.getStyles.mockResolvedValue(styles);
    render(
      <MemoryRouter initialEntries={["/projects/7"]}>
        <Routes><Route path="/projects/:id" element={<EditorPage />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText("project 7 not found")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByTestId("player");
  });
});

describe("Export", () => {
  it("saves the session, renders, waits for the job and offers the downloads", async () => {
    await openEditor();
    mocked.finalize.mockResolvedValue({ ...stateOf(), metrics: {}, implicit_events: 0 });
    mocked.startRender.mockResolvedValue({ job: { id: "j1", kind: "render", status: "queued", error: null, meta: {} }, preset: "reels", duration: 3, warning: null });
    mocked.getJob.mockResolvedValue({ id: "j1", kind: "render", status: "done", error: null, meta: {} });
    fireEvent.click(screen.getByRole("button", { name: "Export…" }));
    expect(player.pause).toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "Render" }));
    expect(await screen.findByText(/Done: 3\.0s long/, undefined, { timeout: 4000 })).toBeInTheDocument();
    expect(mocked.finalize).toHaveBeenCalledWith(7, expect.any(Number));
    expect(mocked.startRender).toHaveBeenCalledWith(7, "reels");
    expect(screen.getByRole("link", { name: /\.mp4/ })).toHaveAttribute("href", "/api/projects/7/download?fmt=mp4&preset=reels");
    expect(screen.getByRole("link", { name: /\.srt/ })).toHaveAttribute("href", "/api/projects/7/download?fmt=srt&preset=reels");
  });

  it("does not render when the session could not be saved, and shows a failed render", async () => {
    await openEditor();
    mocked.finalize.mockRejectedValueOnce(new ApiError(500, "boom"));
    fireEvent.click(screen.getByRole("button", { name: "Export…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Render" }));
    expect(await within(screen.getByRole("dialog")).findByRole("alert")).toHaveTextContent("Could not save");
    expect(mocked.startRender).not.toHaveBeenCalled();

    mocked.finalize.mockResolvedValue({ ...stateOf(), metrics: {}, implicit_events: 0 });
    mocked.startRender.mockResolvedValue({ job: { id: "j2", kind: "render", status: "queued", error: null, meta: {} }, preset: "reels", duration: 3, warning: null });
    mocked.getJob.mockResolvedValue({ id: "j2", kind: "render", status: "failed", error: "ffmpeg exited with 1", meta: {} });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("ffmpeg exited with 1", undefined, { timeout: 4000 })).toBeInTheDocument();
  });
});
