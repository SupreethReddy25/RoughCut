import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getProfile } from "../api";
import type { ProfileOut } from "../api-types";
import { useUser } from "../hooks/useUser";
import { pct } from "../lib/format";
import { groupPreferences, statCards } from "../lib/profile";

type Row = ProfileOut["metrics"][number] & { minutes_active: number };

function Curve({ title, data, dataKey, label, percent, hint }: { title: string; data: Row[]; dataKey: "acceptance" | "manual_per_minute" | "minutes_active"; label: string; percent?: boolean; hint: string }) {
  return (
    <section className="rounded-xl border border-white/10 bg-neutral-900 p-3" data-testid={`curve-${dataKey}`}>
      <h2 className="text-sm font-medium text-neutral-200">{title}</h2>
      <p className="mb-1 text-[11px] text-neutral-500">{hint}</p>
      {data.length < 1 ? (
        <p className="py-8 text-center text-xs text-neutral-500">Export a first video to see this curve.</p>
      ) : (
        <div style={{ width: "100%", height: 190 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#ffffff22" />
              <XAxis dataKey="video_index" tick={{ fill: "#a3a3a3", fontSize: 11 }} allowDecimals={false} label={{ value: "video #", position: "insideBottom", offset: -2, fill: "#737373", fontSize: 11 }} />
              <YAxis tick={{ fill: "#a3a3a3", fontSize: 11 }} domain={percent ? [0, 1] : [0, "auto"]} tickFormatter={percent ? (v: number) => pct(v) : undefined} width={42} />
              <Tooltip
                contentStyle={{ background: "#171717", border: "1px solid #404040" }}
                formatter={(v) => [percent ? pct(Number(v)) : Number(v).toFixed(1), label]}
                labelFormatter={(i) => `Video ${i}`}
              />
              <Line type="monotone" dataKey={dataKey} stroke="#38bdf8" strokeWidth={2} dot isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

export default function ProfilePage() {
  const { user } = useUser();
  const [profile, setProfile] = useState<ProfileOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const userId = user?.id ?? null;

  useEffect(() => {
    if (userId === null) return;
    let alive = true;
    setProfile(null);
    getProfile(userId)
      .then((p) => alive && setProfile(p))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [userId]);

  const rows: Row[] = useMemo(() => (profile?.metrics ?? []).map((m) => ({ ...m, minutes_active: m.seconds_active / 60 })), [profile]);

  if (error) return <p role="alert" className="m-6 rounded bg-red-500/20 p-2 text-sm text-red-200">{error}</p>;
  if (!profile) return <p className="p-6 text-neutral-400">Loading…</p>;
  const style = profile.style_mixture;
  const mix = style ? Object.entries(style.weights).sort((a, b) => b[1] - a[1]) : [];
  const labelOf = (k: string) => profile.style_labels[k]?.label ?? k.replace(/_/g, " ");
  const groups = groupPreferences(profile.preferences);
  const cards = statCards(profile.stats);

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-6">
      <h1 className="text-2xl font-semibold">{profile.user.name}</h1>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {[
          ["Projects", profile.n_projects],
          ["Finished edits", profile.n_finalized],
          ["Things I have learned from", profile.n_learning_events],
        ].map(([k, v]) => (
          <div key={k} className="rounded-xl border border-white/10 bg-neutral-900 p-3">
            <div className="text-2xl font-semibold">{v}</div>
            <div className="text-xs text-neutral-400">{k}</div>
          </div>
        ))}
      </div>

      <section className="rounded-xl border border-white/10 bg-neutral-900 p-3" data-testid="style">
        <h2 className="mb-1 text-sm font-medium text-neutral-200">Your editing style</h2>
        <p className="mb-2 text-sm text-neutral-100" data-testid="style-text">
          {style?.text ?? "No style yet: review a few suggestions and RoughCut will learn how you like to edit."}
          {style?.identified && <span className="ml-2 rounded bg-emerald-500/25 px-1.5 py-0.5 text-[11px] text-emerald-200">identified</span>}
        </p>
        {mix.length > 0 && style && style.n_events > 0 && (
          <ul className="space-y-1.5">
            {mix.map(([name, w]) => (
              <li key={name} className="flex items-center gap-2 text-sm" title={profile.style_labels[name]?.blurb}>
                <span className="w-28">{labelOf(name)}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-white/10">
                  <div className="h-full bg-sky-500" style={{ width: pct(w) }} />
                </div>
                <span className="w-10 text-right text-xs text-neutral-400">{pct(w)}</span>
              </li>
            ))}
          </ul>
        )}
        {style?.top && profile.style_labels[style.top]?.blurb && (
          <p className="mt-2 text-xs text-neutral-500">
            {labelOf(style.top)}: {profile.style_labels[style.top].blurb}.
          </p>
        )}
      </section>

      <section className="rounded-xl border border-white/10 bg-neutral-900 p-3" data-testid="preferences">
        <h2 className="mb-1 text-sm font-medium text-neutral-200">What I have learned about how you edit</h2>
        {groups.length === 0 ? (
          <p className="text-xs text-neutral-500">Not enough yet: it needs a handful of decisions of each kind before it says anything about them.</p>
        ) : (
          <ul className="space-y-1 text-sm text-neutral-100">
            {groups.flatMap((g) => g.texts.map((t) => (
              <li key={t} className="flex gap-2">
                <span className="mt-0.5 w-16 shrink-0 text-[10px] uppercase tracking-wide text-neutral-500">{g.label}</span>
                <span>{t}</span>
              </li>
            )))}
          </ul>
        )}
      </section>

      <section className="rounded-xl border border-white/10 bg-neutral-900 p-3" data-testid="stats">
        <h2 className="mb-2 text-sm font-medium text-neutral-200">Your edits in numbers</h2>
        {cards.length === 0 ? (
          <p className="text-xs text-neutral-500">Finish and export a video to see these.</p>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
            {cards.map((c) => (
              <div key={c.label}>
                <div className="text-lg font-semibold">{c.value}</div>
                <div className="text-[11px] text-neutral-400">{c.label}</div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-xl border border-white/10 bg-neutral-900 p-3" data-testid="trust">
        <h2 className="mb-1 text-sm font-medium text-neutral-200">Trust score</h2>
        {profile.trust === null ? (
          <p className="text-xs text-neutral-500">
            Not yet: finish and export a video you reviewed. A batch video is called “Ready to post” once your trust is at least{" "}
            {pct(profile.trust_threshold)}.
          </p>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <span className="text-2xl font-semibold">{pct(profile.trust)}</span>
              <div className="relative h-2 flex-1 rounded bg-white/10">
                <div className={`h-full rounded ${profile.trust >= profile.trust_threshold ? "bg-emerald-500" : "bg-sky-500"}`} style={{ width: pct(profile.trust) }} />
                <div className="absolute -top-1 h-4 w-px bg-white/60" style={{ left: pct(profile.trust_threshold) }} title={`needed: ${pct(profile.trust_threshold)}`} />
              </div>
            </div>
            <p className="mt-1 text-xs text-neutral-500">
              How often you accepted what was suggested in your last 2 reviewed videos.{" "}
              {profile.trust >= profile.trust_threshold ? "Batch videos can be marked “Ready to post”." : `Batch videos are marked “Ready to post” from ${pct(profile.trust_threshold)}.`}
            </p>
          </>
        )}
      </section>

      <div className="grid gap-4 md:grid-cols-3">
        <Curve title="Learning curve" hint="How often you accept its suggestions" data={rows} dataKey="acceptance" label="acceptance" percent />
        <Curve title="Effort curve" hint="Manual edits per minute of footage" data={rows} dataKey="manual_per_minute" label="manual edits / min" />
        <Curve title="Time in the editor" hint="Minutes with the editor in focus" data={rows} dataKey="minutes_active" label="minutes" />
      </div>
    </div>
  );
}
