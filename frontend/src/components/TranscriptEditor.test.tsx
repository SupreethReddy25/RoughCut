import { act, createRef } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeProject } from "../test/factory";
import { selectedWordIds, TranscriptEditor, type TranscriptHandle } from "./TranscriptEditor";

function setup(project = makeProject()) {
  const onEdit = vi.fn();
  const onSeek = vi.fn();
  const onSelect = vi.fn();
  const onCaptionText = vi.fn();
  const ref = createRef<TranscriptHandle>();
  render(
    <TranscriptEditor ref={ref} project={project} srcTime={0} playing={false} selected={null} onSelect={onSelect} onSeek={onSeek} onEdit={onEdit} onCaptionText={onCaptionText} />,
  );
  return { ref, onEdit, onSeek, onSelect, onCaptionText };
}

const word = (id: number) => document.querySelector<HTMLElement>(`[data-word-id="${id}"]`)!;

function selectWords(from: number, to: number) {
  const r = document.createRange();
  r.setStartBefore(word(from));
  r.setEndAfter(word(to));
  const s = window.getSelection()!;
  s.removeAllRanges();
  s.addRange(r);
  document.dispatchEvent(new Event("selectionchange"));
}

describe("TranscriptEditor", () => {
  it("shows kept words normally and cut words struck through", () => {
    const p = makeProject();
    p.edl!.clips = p.edl!.clips.slice(0, 2); // the third phrase is cut
    setup(p);
    expect(word(0).dataset.kept).toBe("1");
    expect(word(4).dataset.kept).toBe("0");
    expect(word(4).className).toMatch(/line-through/);
  });

  it("cuts a selection of kept words as a delete_words edit", async () => {
    const { ref, onEdit } = setup();
    selectWords(0, 1);
    await waitFor(() => expect(screen.getByText("2 words selected")).toBeInTheDocument());
    let done = false;
    act(() => {
      done = ref.current!.deleteSelection();
    });
    expect(done).toBe(true);
    expect(onEdit).toHaveBeenCalledWith({ kind: "delete_words", word_ids: [0, 1] });
  });

  it("restores a selection of cut words, and does nothing for the wrong kind", async () => {
    const p = makeProject();
    p.edl!.clips = p.edl!.clips.slice(0, 2);
    const { ref, onEdit } = setup(p);
    selectWords(4, 5);
    await waitFor(() => expect(screen.getByRole("button", { name: "Restore" })).toBeEnabled());
    expect(screen.getByRole("button", { name: /Delete/ })).toBeDisabled();
    act(() => {
      expect(ref.current!.deleteSelection()).toBe(false);
    });
    expect(onEdit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(onEdit).toHaveBeenCalledWith({ kind: "restore_words", word_ids: [4, 5] });
  });

  it("reports nothing for a collapsed selection or one outside the transcript", () => {
    setup();
    const root = document.querySelector<HTMLElement>("[data-word-id]")!.parentElement!.parentElement!.parentElement!;
    expect(selectedWordIds(root, null)).toEqual([]);
    window.getSelection()!.removeAllRanges();
    expect(selectedWordIds(root, window.getSelection())).toEqual([]);
    const outside = document.createElement("p");
    outside.textContent = "elsewhere";
    document.body.appendChild(outside);
    const r = document.createRange();
    r.selectNodeContents(outside);
    window.getSelection()!.addRange(r);
    expect(selectedWordIds(root, window.getSelection())).toEqual([]);
    outside.remove();
  });

  it("changes a pause through its menu, only when the choice differs", () => {
    const { onEdit, onSelect, onSeek } = setup();
    fireEvent.click(screen.getByRole("button", { name: /Pause of 0\.40s, KEEP/ }));
    expect(onSeek).toHaveBeenCalledWith(1);
    expect(onSelect).not.toHaveBeenCalled(); // this pause has no decision in the fixture
    const menu = screen.getByRole("menu");
    fireEvent.click(within(menu).getByText(/Keep as is/)); // the current action
    expect(onEdit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Pause of 0\.40s, KEEP/ }));
    fireEvent.click(within(screen.getByRole("menu")).getByText("Tighten to 0.15 s"));
    expect(onEdit).toHaveBeenCalledWith({ kind: "set_pause", unit_id: 1, option: "TIGHT_15" });
  });

  it("selects the decision of a pause and shows its confidence dot", () => {
    const { onSelect } = setup();
    fireEvent.click(screen.getByRole("button", { name: /Pause of 0\.60s, REMOVE/ }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ decision_type: "pause", target_id: 3 }));
    expect(screen.getByTestId("dot-pause-3")).toBeInTheDocument();
  });

  it("jumps to a word and selects its phrase decision", () => {
    const { onSeek, onSelect } = setup();
    fireEvent.click(word(2));
    expect(onSeek).toHaveBeenCalledWith(1.4);
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ decision_type: "phrase", target_id: 2 }));
  });

  it("fixes the caption text of a word on double click, and ignores an unchanged or empty edit", () => {
    const { onCaptionText } = setup();
    fireEvent.doubleClick(word(2));
    let input = screen.getByLabelText("Caption text");
    expect(input).toHaveValue("second");
    fireEvent.change(input, { target: { value: "2nd" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onCaptionText).toHaveBeenCalledWith(2, "2nd");

    fireEvent.doubleClick(word(3));
    input = screen.getByLabelText("Caption text");
    fireEvent.keyDown(input, { key: "Enter" }); // unchanged
    fireEvent.doubleClick(word(3));
    input = screen.getByLabelText("Caption text");
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.blur(input); // empty
    fireEvent.doubleClick(word(3));
    fireEvent.keyDown(screen.getByLabelText("Caption text"), { key: "Escape" });
    expect(onCaptionText).toHaveBeenCalledTimes(1);
  });

  it("does not offer to fix the caption of a cut word", () => {
    const p = makeProject();
    p.edl!.clips = p.edl!.clips.slice(0, 2);
    setup(p);
    fireEvent.doubleClick(word(4));
    expect(screen.queryByLabelText("Caption text")).toBeNull();
  });
});
