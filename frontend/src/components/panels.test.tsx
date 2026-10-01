import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeProject } from "../test/factory";
import { CaptionStylePanel } from "./CaptionStylePanel";
import { ReviewPanel } from "./ReviewPanel";
import { layoutClips, Timeline, weakestIn } from "./Timeline";
import type { StylesOut } from "../api-types";

describe("Timeline", () => {
  const p = makeProject();

  it("lays the clips out on the output axis, back to back", () => {
    const blocks = layoutClips(p);
    expect(blocks.map((b) => [b.left, b.width])).toEqual([[0, 1], [1, 1], [2, 1]]);
    expect(blocks[1].zoom).toBe(1.15);
  });

  it("divides by the speed", () => {
    const fast = makeProject();
    fast.edl!.output.speed = 1.25;
    expect(layoutClips(fast).map((b) => b.width)).toEqual([0.8, 0.8, 0.8]);
  });

  it("picks the least confident pending suggestion of a clip", () => {
    const withExtra = [...p.suggestions, { ...p.suggestions[0], id: 50, unit_id: 2, confidence: 0.2 }];
    expect(weakestIn([2], withExtra)!.id).toBe(50);
    expect(weakestIn([2], withExtra.map((s) => ({ ...s, status: "accepted" as const })))).toBeNull();
    expect(weakestIn([99], withExtra)).toBeNull();
  });

  it("shows zoom badges and confidence dots, and selects the weakest decision from the dot", () => {
    const onSelect = vi.fn();
    render(<Timeline project={p} outTime={0} playing={false} selected={null} selectedText={null} onSeekOut={() => {}} onSelect={onSelect} onSelectText={() => {}} />);
    expect(screen.getByTestId("clip-1")).toHaveTextContent("1.15×");
    expect(screen.queryByTestId("clip-0")).not.toHaveTextContent("×");
    fireEvent.click(screen.getByTestId("clip-dot-1"));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toMatchObject({ id: 1 });
    expect(screen.queryByTestId("clip-dot-0")).toBeNull(); // no decision on the first clip
  });

  it("shows the text track and lets a bar be selected", () => {
    const withText = makeProject();
    withText.edl!.texts = [{ id: "t1", text: "Hello", t0: 0.5, t1: 2, position: "top", y: null, style_id: "title" }];
    const onSelectText = vi.fn();
    render(<Timeline project={withText} outTime={0} playing={false} selected={null} selectedText={null} onSeekOut={() => {}} onSelect={() => {}} onSelectText={onSelectText} />);
    fireEvent.click(screen.getByTestId("text-bar-t1"));
    expect(onSelectText).toHaveBeenCalledWith("t1");
  });

  it("highlights the clip of the selected decision", () => {
    render(<Timeline project={p} outTime={0} playing={false} selected={{ decision_type: "phrase", target_id: 2 }} selectedText={null} onSeekOut={() => {}} onSelect={() => {}} onSelectText={() => {}} />);
    expect(screen.getByTestId("clip-1").className).toMatch(/border-sky-400/);
    expect(screen.getByTestId("clip-0").className).not.toMatch(/border-sky-400/);
  });
});

describe("ReviewPanel", () => {
  const baseProps = { selected: null, fresh: new Set<number>(), onSelect: vi.fn(), onAccept: vi.fn(), onReject: vi.fn(), onChoose: vi.fn() };

  it("lists the uncertain ones first, least confident on top, and collapses the confident rest", () => {
    const p = makeProject();
    render(<ReviewPanel project={p} {...baseProps} />);
    const needs = screen.getByTestId("review-panel").querySelector("section")!;
    const rows = within(needs).getAllByTestId(/queue-item-/);
    expect(rows.map((r) => r.dataset.testid)).toEqual(["queue-item-1", "queue-item-2"]);
    expect(screen.getByText(/Confident edits \(1\)/)).toBeInTheDocument();
  });

  it("accepts, rejects and chooses through the row buttons", () => {
    const onAccept = vi.fn();
    const onReject = vi.fn();
    const onChoose = vi.fn();
    const p = makeProject();
    render(<ReviewPanel project={p} {...baseProps} selected={{ decision_type: "pause", target_id: 3 }} onAccept={onAccept} onReject={onReject} onChoose={onChoose} />);
    const row = screen.getByTestId("queue-item-2");
    fireEvent.click(within(row).getByLabelText("Accept"));
    fireEvent.click(within(row).getByLabelText("Reject"));
    expect(onAccept).toHaveBeenCalledWith(expect.objectContaining({ id: 2 }));
    expect(onReject).toHaveBeenCalledWith(expect.objectContaining({ id: 2 }));
    fireEvent.click(within(row).getByText("Tighten to 0.15 s"));
    expect(onChoose).toHaveBeenCalledWith(expect.objectContaining({ id: 2 }), "TIGHT_15");
  });

  it("keeps a decided suggestion under Reviewed, without accept/reject buttons", () => {
    const p = makeProject();
    p.suggestions[0] = { ...p.suggestions[0], status: "corrected", current_action: "CUT" };
    p.queue = { needs_review: [2], confident: [3] };
    render(<ReviewPanel project={p} {...baseProps} />);
    expect(screen.getByText(/Reviewed \(1\)/)).toBeInTheDocument();
    const row = screen.getByTestId("queue-item-1");
    expect(within(row).queryByLabelText("Accept")).toBeNull();
    expect(row).toHaveTextContent("Changed to:");
  });

  it("marks a selection by decision, not by id (ids change when the model re-scores)", () => {
    const p = makeProject();
    render(<ReviewPanel project={p} {...baseProps} selected={{ decision_type: "phrase", target_id: 2 }} />);
    expect(screen.getByTestId("queue-item-1").dataset.selected).toBe("1");
    expect(screen.getByTestId("queue-item-2").dataset.selected).toBe("0");
  });

  it("highlights suggestions that re-scoring just replaced", () => {
    render(<ReviewPanel project={makeProject()} {...baseProps} fresh={new Set([2])} />);
    expect(screen.getByTestId("queue-item-2").className).toMatch(/animate-pulse/);
    expect(screen.getByTestId("queue-item-1").className).not.toMatch(/animate-pulse/);
  });
});

const styles: StylesOut = {
  captions: {
    positions: { lower_third: 0.68, center: 0.5, top: 0.15 },
    styles: {
      bold_pop: { label: "Bold pop", mode: "pop", font: "Montserrat", font_file: "m.ttf", size: 80, case: "upper", color: "#fff", emphasis_color: "#ff0", outline_color: "#000", outline: 8, shadow: 0, max_width: 0.85 },
      clean: { label: "Clean", mode: "line", font: "Inter", font_file: "i.ttf", size: 60, case: "none", color: "#fff", emphasis_color: "#ff0", outline_color: "#000", outline: 2, shadow: 0, max_width: 0.85 },
    },
  },
  texts: { positions: { top: 0.12, center: 0.5 }, styles: { title: { font: "Inter", font_file: "i.ttf", size: 70, case: "upper", color: "#fff", outline_color: "#000", outline: 4, shadow: 0, max_width: 0.8 } } },
  presets: {},
};

describe("CaptionStylePanel", () => {
  const handlers = () => ({ onSelectText: vi.fn(), onSetting: vi.fn(), onSpeed: vi.fn(), onOverlay: vi.fn() });

  it("sends a setting only when something changes", () => {
    const h = handlers();
    render(<CaptionStylePanel project={makeProject()} styles={styles} outTime={0} selectedText={null} {...h} />);
    fireEvent.click(screen.getByRole("radio", { name: /Bold pop/ })); // already the style
    expect(h.onSetting).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("radio", { name: /Clean/ }));
    expect(h.onSetting).toHaveBeenCalledWith("caption_style", "clean");
    fireEvent.click(screen.getByRole("radio", { name: "Top" }));
    expect(h.onSetting).toHaveBeenCalledWith("caption_position", "top");
    fireEvent.click(screen.getByLabelText("Burn captions into the video"));
    expect(h.onSetting).toHaveBeenCalledWith("captions_enabled", false);
    fireEvent.click(screen.getByLabelText("Enhance audio"));
    expect(h.onSetting).toHaveBeenCalledWith("audio_enhance", false);
  });

  it("changes the speed", () => {
    const h = handlers();
    render(<CaptionStylePanel project={makeProject()} styles={styles} outTime={0} selectedText={null} {...h} />);
    fireEvent.click(screen.getByRole("radio", { name: "1×" }));
    expect(h.onSpeed).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("radio", { name: "1.25×" }));
    expect(h.onSpeed).toHaveBeenCalledWith(1.25);
  });

  it("adds a text overlay at the playhead, ending with the programme", () => {
    const h = handlers();
    render(<CaptionStylePanel project={makeProject()} styles={styles} outTime={1.26} selectedText={null} {...h} />);
    fireEvent.click(screen.getByText("+ Add text"));
    expect(h.onOverlay).toHaveBeenCalledWith({ op: "add", text: "Your title", t0: 1.3, t1: 3 });
  });

  it("edits and deletes an overlay; a bad time range is put back", () => {
    const h = handlers();
    const p = makeProject();
    p.edl!.texts = [{ id: "t1", text: "Hello", t0: 0.5, t1: 2, position: "top", y: null, style_id: "title" }];
    render(<CaptionStylePanel project={p} styles={styles} outTime={0} selectedText={null} {...h} />);
    const text = screen.getByLabelText("Overlay text");
    fireEvent.change(text, { target: { value: "  New words " } });
    fireEvent.blur(text);
    expect(h.onOverlay).toHaveBeenLastCalledWith({ op: "update", id: "t1", text: "New words" });

    const end = screen.getByLabelText("End");
    fireEvent.change(end, { target: { value: "0.2" } }); // before the start: refused, value restored
    fireEvent.blur(end);
    expect(h.onOverlay).toHaveBeenCalledTimes(1);
    expect(end).toHaveValue("2");
    fireEvent.change(end, { target: { value: "2.5" } });
    fireEvent.blur(end);
    expect(h.onOverlay).toHaveBeenLastCalledWith({ op: "update", id: "t1", t0: 0.5, t1: 2.5 });

    fireEvent.click(screen.getByLabelText("Delete text"));
    expect(h.onOverlay).toHaveBeenLastCalledWith({ op: "delete", id: "t1" });
  });

  it("gives a custom position a vertical place", () => {
    const h = handlers();
    const p = makeProject();
    p.edl!.texts = [{ id: "t1", text: "Hello", t0: 0.5, t1: 2, position: "top", y: null, style_id: "title" }];
    render(<CaptionStylePanel project={p} styles={styles} outTime={0} selectedText={null} {...h} />);
    fireEvent.change(screen.getByLabelText("Position"), { target: { value: "custom" } });
    expect(h.onOverlay).toHaveBeenCalledWith({ op: "update", id: "t1", position: "custom", y: 0.5 });
  });
});
