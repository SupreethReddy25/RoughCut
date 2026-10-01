import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { TrackOut } from "../api-types";
import { makeProject, suggestion } from "../test/factory";
import { CaptionStrip } from "./CaptionStrip";
import { MusicPanel } from "./MusicPanel";
import { MusicPreview } from "./MusicPreview";
import { NUDGES, ReviewPanel } from "./ReviewPanel";
import { Timeline } from "./Timeline";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, listMusic: vi.fn(), uploadMusic: vi.fn() };
});
import * as api from "../api";
import { ApiError } from "../api";

const mocked = vi.mocked(api);

const play = vi.fn(() => Promise.resolve());
const pause = vi.fn();
beforeEach(() => {
  vi.resetAllMocks();
  play.mockImplementation(() => Promise.resolve());
  HTMLMediaElement.prototype.play = play as unknown as () => Promise<void>;
  HTMLMediaElement.prototype.pause = pause;
});

const withCaptions = () => {
  const p = makeProject();
  p.edl!.captions.lines = [
    { t0: 0, t1: 1, words: [{ t0: 0, t1: 0.5, text: "hello", emphasis: false }, { t0: 0.5, t1: 1, text: "there", emphasis: false }] },
    { t0: 1, t1: 2, words: [{ t0: 1, t1: 1.5, text: "second", emphasis: true }, { t0: 1.5, t1: 2, text: "part", emphasis: false }] },
  ];
  p.caption_word_ids = [[0, 1], [2, 3]];
  p.suggestions = [...p.suggestions, suggestion(20, { decision_type: "emphasis", target_id: 0, candidates: ["NONE", "CAND_0", "CAND_1"], candidate_word_ids: [0, 1] })];
  return p;
};

describe("CaptionStrip", () => {
  it("shows the words of the line under the playhead and sends the click as an emphasis edit", () => {
    const onEdit = vi.fn();
    render(<CaptionStrip project={withCaptions()} outTime={0.2} onEdit={onEdit} />);
    fireEvent.click(screen.getByRole("button", { name: "there" }));
    expect(onEdit).toHaveBeenCalledWith({ kind: "set_emphasis", word_id: 1, line_word_ids: [0, 1] });
  });

  it("clears a highlighted word on click and follows the playhead to the next line", () => {
    const onEdit = vi.fn();
    const { rerender } = render(<CaptionStrip project={withCaptions()} outTime={0.2} onEdit={onEdit} />);
    expect(screen.queryByRole("button", { name: "second" })).toBeNull();
    rerender(<CaptionStrip project={withCaptions()} outTime={1.2} onEdit={onEdit} />);
    const hi = screen.getByRole("button", { name: "second" });
    expect(hi).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(hi);
    expect(onEdit).toHaveBeenCalledWith({ kind: "set_emphasis", word_id: null, line_word_ids: [2, 3] });
  });

  it("marks the agent's candidate words", () => {
    render(<CaptionStrip project={withCaptions()} outTime={0.2} onEdit={() => {}} />);
    expect(screen.getByRole("button", { name: "hello" })).toHaveAttribute("title", "the agent considered this word");
  });

  it("is hidden when captions are off or the ids do not line up", () => {
    const off = withCaptions();
    off.edl!.captions.enabled = false;
    const { container, rerender } = render(<CaptionStrip project={off} outTime={0.2} onEdit={() => {}} />);
    expect(container).toBeEmptyDOMElement();
    const odd = withCaptions();
    odd.caption_word_ids = [[0], [2]];
    rerender(<CaptionStrip project={odd} outTime={0.2} onEdit={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ReviewPanel nudges and the filtered view", () => {
  const base = { selected: null, fresh: new Set<number>(), onSelect: vi.fn(), onAccept: vi.fn(), onReject: vi.fn(), onChoose: vi.fn() };

  it("offers the four global nudges", () => {
    const onNudge = vi.fn();
    render(<ReviewPanel project={makeProject()} {...base} onNudge={onNudge} />);
    const group = screen.getByRole("group", { name: "Nudge the whole edit" });
    for (const n of NUDGES) fireEvent.click(within(group).getByRole("button", { name: n.label }));
    expect(onNudge.mock.calls.map((c) => c[0])).toEqual(["snappier", "calmer", "more_zooms", "fewer_zooms"]);
  });

  it("has no nudge buttons unless asked", () => {
    render(<ReviewPanel project={makeProject()} {...base} />);
    expect(screen.queryByRole("group", { name: "Nudge the whole edit" })).toBeNull();
  });

  it("shows only what needs the eye for a batch video, with a way back to everything", () => {
    const onShowAll = vi.fn();
    render(<ReviewPanel project={makeProject()} {...base} onlyUncertain onShowAll={onShowAll} />);
    expect(screen.getByText(/Needs your eye/)).toBeInTheDocument();
    expect(screen.queryByText(/Confident edits/)).toBeNull();
    expect(screen.queryByText(/Reviewed \(/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show everything" }));
    expect(onShowAll).toHaveBeenCalled();
  });
});

describe("Timeline text track", () => {
  const withText = () => {
    const p = makeProject();
    p.edl!.texts = [{ id: "t1", text: "Hello", t0: 0.5, t1: 1.5, position: "top", y: null, style_id: "title" }];
    return p;
  };
  const bar = () => {
    const el = screen.getByTestId("text-bar-t1");
    el.getBoundingClientRect = () => ({ left: 100, right: 124, top: 0, bottom: 20, width: 24, height: 20, x: 100, y: 0, toJSON: () => ({}) });
    return el;
  };
  const setup = (onOverlayChange = vi.fn(), onSelectText = vi.fn()) => {
    render(<Timeline project={withText()} outTime={0} playing={false} selected={null} selectedText={null} onSeekOut={() => {}} onSelect={() => {}} onSelectText={onSelectText} onOverlayChange={onOverlayChange} />);
    return { onOverlayChange, onSelectText };
  };

  it("moves a text bar by dragging its body (24 px per second at the default zoom)", () => {
    const { onOverlayChange, onSelectText } = setup();
    const el = bar();
    fireEvent.pointerDown(el, { clientX: 112, button: 0, pointerId: 1 });
    fireEvent.pointerMove(el, { clientX: 136, pointerId: 1 });
    fireEvent.pointerUp(el, { clientX: 136, pointerId: 1 });
    fireEvent.click(el); // the click that follows a drag
    expect(onOverlayChange).toHaveBeenCalledWith("t1", 1.5, 2.5);
    expect(onSelectText).not.toHaveBeenCalled();
  });

  it("resizes from either end", () => {
    const { onOverlayChange } = setup();
    const el = bar();
    fireEvent.pointerDown(el, { clientX: 122, button: 0, pointerId: 1 }); // 2 px from the right edge
    fireEvent.pointerMove(el, { clientX: 146, pointerId: 1 });
    fireEvent.pointerUp(el, { clientX: 146, pointerId: 1 });
    expect(onOverlayChange).toHaveBeenLastCalledWith("t1", 0.5, 2.5);
    fireEvent.pointerDown(el, { clientX: 102, button: 0, pointerId: 1 }); // 2 px from the left edge
    fireEvent.pointerMove(el, { clientX: 114, pointerId: 1 });
    fireEvent.pointerUp(el, { clientX: 114, pointerId: 1 });
    expect(onOverlayChange).toHaveBeenLastCalledWith("t1", 1, 1.5);
  });

  it("a plain click only selects, and a tiny wobble changes nothing", async () => {
    const { onOverlayChange, onSelectText } = setup();
    const el = bar();
    fireEvent.pointerDown(el, { clientX: 112, button: 0, pointerId: 1 });
    fireEvent.pointerMove(el, { clientX: 113, pointerId: 1 });
    fireEvent.pointerUp(el, { clientX: 113, pointerId: 1 });
    fireEvent.click(el);
    expect(onOverlayChange).not.toHaveBeenCalled();
    expect(onSelectText).toHaveBeenCalledWith("t1");
  });

  it("a click after a drag selects again once the drag is over", async () => {
    const { onSelectText } = setup();
    const el = bar();
    fireEvent.pointerDown(el, { clientX: 112, button: 0, pointerId: 1 });
    fireEvent.pointerMove(el, { clientX: 140, pointerId: 1 });
    fireEvent.pointerUp(el, { clientX: 140, pointerId: 1 });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 5));
    });
    fireEvent.click(el); // e.g. Enter on the focused bar
    expect(onSelectText).toHaveBeenCalledWith("t1");
  });

  it("stays a plain button when there is no change handler", () => {
    render(<Timeline project={withText()} outTime={0} playing={false} selected={null} selectedText={null} onSeekOut={() => {}} onSelect={() => {}} onSelectText={() => {}} />);
    const el = bar();
    fireEvent.pointerDown(el, { clientX: 112, button: 0, pointerId: 1 });
    fireEvent.pointerMove(el, { clientX: 160, pointerId: 1 });
    expect(el.style.left).toBe("12px");
  });
});

const tracks: TrackOut[] = [
  { id: "lib_sunrise", name: "Sunrise", mood: "calm", duration: 48, source: "library" },
  { id: "lib_neon", name: "Neon Pulse", mood: "energetic", duration: 32, source: "library" },
  { id: "up_1_abcdef12", name: "my song", mood: "", duration: 90, source: "upload" },
];

describe("MusicPanel", () => {
  const edlWith = (music: { track_id: string; gain_db: number; duck_db: number } | null) => {
    const p = makeProject();
    p.edl!.audio.music = music;
    return p.edl!;
  };

  it("lists the tracks and picks one", async () => {
    mocked.listMusic.mockResolvedValue(tracks);
    const onMusic = vi.fn();
    render(<MusicPanel edl={edlWith(null)} userId={1} onMusic={onMusic} />);
    fireEvent.click(await screen.findByRole("radio", { name: /Neon Pulse/ }));
    expect(onMusic).toHaveBeenCalledWith({ track_id: "lib_neon" });
    expect(screen.getByText("your track")).toBeInTheDocument();
    expect(screen.queryByLabelText("Music volume")).toBeNull(); // sliders only once a track is chosen
  });

  it("shows the chosen track, does not re-send it, and removes the music", async () => {
    mocked.listMusic.mockResolvedValue(tracks);
    const onMusic = vi.fn();
    render(<MusicPanel edl={edlWith({ track_id: "lib_sunrise", gain_db: -18, duck_db: -10 })} userId={1} onMusic={onMusic} />);
    const on = await screen.findByRole("radio", { name: /Sunrise/ });
    expect(on).toHaveAttribute("aria-checked", "true");
    fireEvent.click(on);
    expect(onMusic).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(onMusic).toHaveBeenCalledWith({ track_id: null });
  });

  it("sends a level change once, on release, keeping the other value", async () => {
    mocked.listMusic.mockResolvedValue(tracks);
    const onMusic = vi.fn();
    render(<MusicPanel edl={edlWith({ track_id: "lib_sunrise", gain_db: -18, duck_db: -10 })} userId={1} onMusic={onMusic} />);
    const gain = await screen.findByLabelText("Music volume");
    fireEvent.change(gain, { target: { value: "-24" } });
    fireEvent.change(gain, { target: { value: "-26" } });
    expect(onMusic).not.toHaveBeenCalled(); // still dragging
    expect(screen.getByText("-26 dB")).toBeInTheDocument();
    fireEvent.pointerUp(gain);
    fireEvent.blur(gain); // a second "commit" for the same value is ignored
    expect(onMusic).toHaveBeenCalledTimes(1);
    expect(onMusic).toHaveBeenCalledWith({ track_id: "lib_sunrise", gain_db: -26, duck_db: -10 });
    fireEvent.change(screen.getByLabelText("Dip under voice"), { target: { value: "-16" } });
    fireEvent.keyUp(screen.getByLabelText("Dip under voice"));
    expect(onMusic).toHaveBeenLastCalledWith({ track_id: "lib_sunrise", gain_db: -18, duck_db: -16 });
  });

  it("plays a short listen and stops it", async () => {
    mocked.listMusic.mockResolvedValue(tracks);
    render(<MusicPanel edl={edlWith(null)} userId={1} onMusic={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Listen to Sunrise" }));
    expect(play).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Stop Sunrise" }));
    expect(pause).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Listen to Sunrise" })).toBeInTheDocument();
  });

  it("uploads the user's own track and selects it; shows why a bad file was refused", async () => {
    mocked.listMusic.mockResolvedValue(tracks.slice(0, 2));
    const onMusic = vi.fn();
    render(<MusicPanel edl={edlWith(null)} userId={1} onMusic={onMusic} />);
    await screen.findByRole("radio", { name: /Sunrise/ });
    mocked.uploadMusic.mockRejectedValueOnce(new ApiError(422, "only .mp3, .wav files are supported"));
    const input = screen.getByLabelText("Upload a music file");
    fireEvent.change(input, { target: { files: [new File(["x"], "song.ogg")] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("only .mp3");
    expect(onMusic).not.toHaveBeenCalled();

    mocked.uploadMusic.mockResolvedValueOnce(tracks[2]);
    fireEvent.change(input, { target: { files: [new File(["x"], "my song.mp3")] } });
    await waitFor(() => expect(onMusic).toHaveBeenCalledWith({ track_id: "up_1_abcdef12" }));
    expect(await screen.findByRole("radio", { name: /my song/ })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("reports a library that cannot be loaded", async () => {
    mocked.listMusic.mockRejectedValue(new ApiError(0, "The server is unreachable"));
    render(<MusicPanel edl={edlWith(null)} userId={1} onMusic={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("unreachable");
  });
});

describe("MusicPreview", () => {
  const edl = (music: { track_id: string; gain_db: number; duck_db: number } | null) => {
    const p = withCaptions();
    p.edl!.audio.music = music;
    return p.edl!;
  };
  const spec = { track_id: "lib_sunrise", gain_db: -20, duck_db: -10 };
  const info = (over: Partial<{ out: number; playing: boolean }>) => ({ src: 0, out: 0.2, playing: false, clip: 0, ...over });
  const audioEl = () => screen.getByTestId("music-preview") as HTMLAudioElement;

  it("renders nothing without music, and the right file with it", () => {
    const { container, rerender } = render(<MusicPreview edl={edl(null)} userId={3} time={info({})} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<MusicPreview edl={edl(spec)} userId={3} time={info({})} />);
    expect(audioEl().getAttribute("src")).toBe("/api/music/lib_sunrise/audio?user_id=3");
    expect(audioEl().loop).toBe(true);
  });

  it("starts with the video, dips while speaking, and stops with it", () => {
    const e = edl(spec);
    const { rerender } = render(<MusicPreview edl={e} userId={3} time={info({})} />);
    expect(play).not.toHaveBeenCalled(); // paused video: no music
    rerender(<MusicPreview edl={e} userId={3} time={info({ playing: true, out: 0.2 })} />); // inside "hello"
    expect(play).toHaveBeenCalledTimes(1);
    const speaking = audioEl().volume;
    expect(speaking).toBeLessThan(0.1);
    for (let i = 0; i < 40; i++) rerender(<MusicPreview edl={e} userId={3} time={info({ playing: true, out: 2.9 + i * 0.001 })} />); // a gap
    expect(audioEl().volume).toBeGreaterThan(speaking * 2);
    Object.defineProperty(audioEl(), "paused", { value: false, configurable: true });
    rerender(<MusicPreview edl={e} userId={3} time={info({ playing: false, out: 3 })} />);
    expect(pause).toHaveBeenCalled();
  });
});
