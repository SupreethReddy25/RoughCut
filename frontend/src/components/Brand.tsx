/** The RoughCut mark: four waveform bars with a gap cut out of the middle ones (a jump cut). */
export function LogoMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
      <defs>
        <linearGradient id="rc-mark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#9d8fff" />
          <stop offset="1" stopColor="#5c48dc" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#rc-mark)" />
      <path d="M9.5 10v12M16 9v5.2M16 17.8V23M22.5 10v12" stroke="white" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

export function Wordmark({ size = 26 }: { size?: number }) {
  return (
    <span className="flex items-center gap-2">
      <LogoMark size={size} />
      <span className="rc-brand text-[15px] font-semibold tracking-tight">RoughCut</span>
    </span>
  );
}

/** Round initial avatar for a profile name. */
export function Avatar({ name, size = 28 }: { name: string; size?: number }) {
  const initial = name.trim().charAt(0).toUpperCase() || "?";
  return (
    <span
      aria-hidden="true"
      style={{ width: size, height: size, fontSize: size * 0.42 }}
      className="inline-flex shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-sky-400 to-sky-700 font-semibold text-white ring-1 ring-white/15"
    >
      {initial}
    </span>
  );
}
