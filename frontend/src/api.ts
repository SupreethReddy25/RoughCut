export async function getHealth(): Promise<{ status: string; ffmpeg: { ok: boolean } }> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return r.json();
}
