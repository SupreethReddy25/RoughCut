import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TargetChip } from "./EditorPage";

describe("TargetChip", () => {
  it("is green within the limit, also well under it (a target is never padded)", () => {
    render(<TargetChip duration={12.04} target={30} />);
    const chip = screen.getByTestId("target-chip");
    expect(chip).toHaveTextContent("12.0s of 30s target");
    expect(chip.className).toMatch(/emerald/);
  });

  it("warns when the edit is over the limit", () => {
    render(<TargetChip duration={34} target={30} />);
    expect(screen.getByTestId("target-chip").className).toMatch(/amber/);
  });
});
