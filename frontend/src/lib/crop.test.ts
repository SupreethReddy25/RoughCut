import { describe, expect, it } from "vitest";
import fixtures from "./fixtures/parity.json";
import { cropRect, dragCenter, pyRound, videoBox, type Rect } from "./crop";

describe("cropRect (parity with backend/app/render/graph.py crop_rect)", () => {
  it(`equals the Python window for all ${fixtures.crop.length} golden cases`, () => {
    for (const c of fixtures.crop) {
      const got = cropRect(c.zoom, c.cx, c.cy, c.src[0], c.src[1], c.out[0], c.out[1]);
      expect(got, JSON.stringify(c)).toEqual(c.rect);
    }
  });

  it("rounds halves to even like Python", () => {
    expect([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(pyRound)).toEqual([0, 2, 2, 4, 2, 3]);
  });

  it("clamps the window inside the frame", () => {
    const [x, y, w, h] = cropRect(1.3, 1, 1, 1920, 1080);
    expect(x + w).toBeLessThanOrEqual(1920);
    expect(y + h).toBeLessThanOrEqual(1080);
    expect(cropRect(1, 0, 0, 1920, 1080).slice(0, 2)).toEqual([0, 0]);
  });
});

describe("videoBox / dragCenter", () => {
  const rect: Rect = [656, 0, 608, 1080];
  it("scales the source so the crop window fills the frame", () => {
    const b = videoBox(rect, 1920, 1080, 360, 640);
    expect(b.width).toBeCloseTo((1920 * 360) / 608, 6);
    expect(b.height).toBeCloseTo((1080 * 640) / 1080, 6);
    const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(b.transform)!;
    expect(Number(m[1])).toBeCloseTo((-656 * 360) / 608, 6);
    expect(Number(m[2])).toBeCloseTo(0, 9);
  });

  it("dragging the picture right moves the crop centre left (and stays in [0,1])", () => {
    const [nx, ny] = dragCenter(0.5, 0.5, 36, 0, rect, 1920, 1080, 360, 640);
    expect(nx).toBeLessThan(0.5);
    expect(ny).toBe(0.5);
    expect(dragCenter(0.02, 0.5, 5000, 0, rect, 1920, 1080, 360, 640)[0]).toBe(0);
    expect(dragCenter(0.98, 0.5, -5000, 0, rect, 1920, 1080, 360, 640)[0]).toBe(1);
    // moving by exactly the distance that corresponds to 10% of the source width
    const px = (0.1 * 1920 * 360) / 608;
    expect(dragCenter(0.5, 0.5, -px, 0, rect, 1920, 1080, 360, 640)[0]).toBeCloseTo(0.6, 9);
  });
});
