import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AnalysisProgressOut } from "../api-types";
import { Remap } from "../lib/remap";
import { AnalysisProgress, fmtEta } from "./AnalysisProgress";

vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  startRender: vi.fn(async () => ({ job: { id: "j" }, preset: "reels", duration: 20, warning: null })),
  getJob: vi.fn(async () => ({ id: "j", kind: "render", status: "done", error: null, meta: {} })),
}));
import { ExportDialog } from "./ExportDialog";

const stages = (running: string): AnalysisProgressOut["stages"] => {
  const keys = ["ingest", "transcribe", "align", "vision", "semantic"];
  const at = keys.indexOf(running);
  return keys.map((key, i) => ({ key, label: `label ${key}`, seconds: 3, state: i < at ? "done" : i === at ? "running" : "waiting" }));
};
const progress = (running: string, live: AnalysisProgressOut["live"] = {}): AnalysisProgressOut => ({
  stages: stages(running), fraction: 0.4, seconds_left: 70, video_seconds: 44, learned: true, waiting: false, live,
});

describe("analysing screen shows the real work", () => {
  it("shows the words as they are heard", () => {
    render(<AnalysisProgress id={1} progress={progress("transcribe", { heard: ["first line", "hello everyone"], heard_fraction: 0.5 })} />);
    expect(screen.getByTestId("heard")).toHaveTextContent("hello everyone");
  });

  it("shows what was found, and keeps it on screen in the next stage", () => {
    const { rerender } = render(<AnalysisProgress id={1} progress={progress("align", { n_pauses: 7, n_fillers: 3, n_phrases: 12 })} />);
    expect(screen.getByTestId("found-counts")).toHaveTextContent("7");
    rerender(<AnalysisProgress id={1} progress={progress("vision", { n_pauses: 7, n_fillers: 3, n_phrases: 12, keywords: ["lighting"] })} />);
    expect(screen.getByTestId("found-so-far")).toHaveTextContent("7 pauses");
    expect(screen.getByTestId("found-so-far")).toHaveTextContent("lighting");
  });

  it("words the time left", () => {
    expect(fmtEta(3)).toBe("almost there");
    expect(fmtEta(42)).toBe("about 40 s left");
    expect(fmtEta(125)).toBe("about 2 min 0 s left".replace(" 0 s", ""));
  });
});

describe("one-tap rating", () => {
  it("shows the choice right away and reports it", async () => {
    const onRate = vi.fn();
    render(<ExportDialog projectId={1} styles={null} defaultPreset="reels" onBeforeRender={async () => true} onRendered={() => undefined} onRate={onRate} onClose={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "Render" }));
    const great = await screen.findByRole("button", { name: /Great/ }, { timeout: 3000 });
    await userEvent.click(great);
    expect(onRate).toHaveBeenCalledWith(1);
    expect(great).toHaveAttribute("aria-pressed", "true");
  });
});

describe("remap with clip paces", () => {
  it("agrees with the Python maths", () => {
    const r = new Remap([[0, 2], [4, 6]], 1, [1, 2]);
    expect(r.duration).toBeCloseTo(3);
    expect(r.srcToOut(5)).toBeCloseTo(2.5);
    expect(r.outToSrc(2.5)).toBeCloseTo(5);
  });
});
