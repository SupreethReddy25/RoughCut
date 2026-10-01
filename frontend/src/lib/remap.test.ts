import { describe, expect, it } from "vitest";
import fixtures from "./fixtures/parity.json";
import { Remap, type Range, type Snap } from "./remap";

describe("Remap (parity with backend/app/edl/remap.py)", () => {
  for (const [i, c] of fixtures.remap.entries()) {
    const r = new Remap(c.ranges as Range[], c.speed);
    it(`case ${i}: duration, srcToOut, outToSrc, isKept equal the Python results`, () => {
      expect(r.duration).toBeCloseTo(c.duration, 12);
      for (const p of c.src_to_out) expect(r.srcToOut(p.t, p.snap as Snap), `${p.t} ${p.snap}`).toBeCloseTo(p.out, 9);
      for (const p of c.out_to_src) expect(r.outToSrc(p.o), `out ${p.o}`).toBeCloseTo(p.src, 9);
      for (const p of c.kept) expect(r.isKept(p.t), `kept ${p.t}`).toBe(p.kept);
    });
  }

  it("is exact on the documented example", () => {
    const r = new Remap([[1, 3], [5, 6], [8, 10]]);
    expect(r.duration).toBe(5);
    expect(r.srcToOut(2.5)).toBe(1.5);
    expect(r.srcToOut(4, "next")).toBe(2);
    expect(r.srcToOut(4, "prev")).toBe(2);
    expect(r.srcToOut(0.2, "next")).toBe(0);
    expect(r.outToSrc(2.5)).toBe(5.5);
  });

  it("round-trips kept times at any speed", () => {
    const r = new Remap([[1, 3], [5, 6], [8, 10]], 1.25);
    for (const t of [1, 2.2, 5.3, 8, 9.9]) expect(r.outToSrc(r.srcToOut(t))).toBeCloseTo(t, 9);
  });

  it("refuses an empty program", () => {
    expect(new Remap([]).duration).toBe(0);
    expect(() => new Remap([]).outToSrc(0)).toThrow(/empty program/);
  });
});
