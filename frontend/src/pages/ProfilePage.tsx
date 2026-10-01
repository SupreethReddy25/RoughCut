import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getProfile } from "../api";
import type { ProfileOut } from "../api-types";
import { useUser } from "../hooks/useUser";
import { pct } from "../lib/format";

function Curve({ title, data, dataKey, label, percent }: { title: string; data: ProfileOut["metrics"]; dataKey: "acceptance" | "manual_per_minute"; label: string; percent?: boolean }) {
  return (
    <section className="rounded-xl border border-white/10 bg-neutral-900 p-3">
      <h2 className="mb-1 text-sm font-medium text-neutral-200">{title}</h2>
      {data.length < 1 ? (
        <p className="py-8 text-center text-xs text-neutral-500">Export a first video to see this curve.</p>
      ) : (
        <div style={{ width: "100%", height: 200 }}>
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

  if (error) return <p role="alert" className="m-6 rounded bg-red-500/20 p-2 text-sm text-red-200">{error}</p>;
  if (!profile) return <p className="p-6 text-neutral-400">Loading…</p>;
  const mix = profile.style_mixture ? Object.entries(profile.style_mixture.weights).sort((a, b) => b[1] - a[1]) : [];

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

      <section className="rounded-xl border border-white/10 bg-neutral-900 p-3">
        <h2 className="mb-2 text-sm font-medium text-neutral-200">Your editing style (what the agent thinks you like)</h2>
        {mix.length === 0 ? (
          <p className="text-xs text-neutral-500">Not enough feedback yet: review a few suggestions and it will start to show.</p>
        ) : (
          <ul className="space-y-1.5">
            {mix.map(([name, w]) => (
              <li key={name} className="flex items-center gap-2 text-sm">
                <span className="w-28 capitalize">{name.replace(/_/g, " ")}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-white/10">
                  <div className="h-full bg-sky-500" style={{ width: pct(w) }} />
                </div>
                <span className="w-10 text-right text-xs text-neutral-400">{pct(w)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <Curve title="How often you accept its suggestions" data={profile.metrics} dataKey="acceptance" label="acceptance" percent />
        <Curve title="Manual edits per minute of video" data={profile.metrics} dataKey="manual_per_minute" label="manual edits / min" />
      </div>
    </div>
  );
}
