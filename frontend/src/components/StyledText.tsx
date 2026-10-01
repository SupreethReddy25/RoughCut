import type { CSSProperties, ReactNode } from "react";
import type { Box } from "../api-types";

export function rgba(hex: string, opacity = 1): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${opacity})`;
}

export interface TextLook {
  font: string;
  size: number; // px at 1080 x 1920
  color: string;
  outline_color: string;
  outline: number;
  shadow: number;
  box?: Box;
}

/**
 * CSS for text that looks like the ASS export: sizes are px at 1080x1920 scaled by `scale` (= frame width / 1080);
 * the ASS outline is drawn OUTSIDE the glyph, which is a centred stroke of twice the width painted under the fill.
 */
export function lookStyle(look: TextLook, scale: number): CSSProperties {
  const st: CSSProperties = {
    fontFamily: `"${look.font}", sans-serif`,
    fontSize: look.size * scale,
    color: look.color,
    lineHeight: 1.2,
    whiteSpace: "pre",
    textAlign: "center",
    fontWeight: 400,
  };
  if (look.outline > 0) {
    Object.assign(st, {
      paintOrder: "stroke fill",
      WebkitTextStroke: `${2 * look.outline * scale}px ${look.outline_color}`,
      strokeLinejoin: "round",
    });
  }
  if (look.shadow > 0) {
    st.textShadow = `${look.shadow * scale}px ${look.shadow * scale}px ${look.shadow * scale}px rgba(0, 0, 0, 0.6)`;
  }
  if (look.box) {
    st.background = rgba(look.box.color, look.box.opacity);
    st.padding = look.box.padding * scale;
  }
  return st;
}

/** A centred block at a vertical position (fraction of the frame height), like `\an5\pos(540, y)` in ASS. */
export function Anchored({ y, children, style }: { y: number; children: ReactNode; style: CSSProperties }) {
  return (
    <div
      className="absolute left-1/2"
      style={{ top: `${y * 100}%`, transform: "translate(-50%, -50%)", ...style }}
    >
      {children}
    </div>
  );
}
