import { useState } from "react";
import { BrowserRouter, Link, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Avatar, LogoMark, Wordmark } from "./components/Brand";
import { UserProvider, useUser } from "./hooks/useUser";
import EditorPage from "./pages/EditorPage";
import ProfilePage from "./pages/ProfilePage";
import ProjectsPage from "./pages/ProjectsPage";
import UploadPage from "./pages/UploadPage";

/** Name-only profiles (PLAN 16.2): pick yours or create one. */
function UserGate() {
  const { users, loading, error, choose, create } = useUser();
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const n = name.trim();
    if (!n || busy) return;
    setBusy(true);
    setErr(null);
    try {
      await create(n);
    } catch (x) {
      setErr((x as Error).message);
      setBusy(false);
    }
  };

  if (loading) return <p className="p-10 text-center text-neutral-400">Loading…</p>;
  if (error) {
    return (
      <p role="alert" className="mx-auto mt-10 max-w-md rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-center text-sm text-red-200">
        Cannot reach the RoughCut server: {error}
      </p>
    );
  }
  return (
    <div className="rc-glow flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-sm">
        <div className="mb-7 flex flex-col items-center text-center">
          <LogoMark size={48} />
          <h1 className="rc-brand mt-4 text-3xl font-semibold tracking-tight">RoughCut</h1>
          <p className="mt-2 text-sm text-neutral-400">An editor that learns how you cut. Who is editing?</p>
        </div>
        <div className="rc-panel p-4 shadow-2xl shadow-black/40">
          {users.length > 0 && (
            <ul className="mb-4 space-y-1.5">
              {users.map((u) => (
                <li key={u.id}>
                  <button type="button" onClick={() => choose(u)} className="group flex w-full items-center gap-3 rounded-lg border border-white/8 bg-white/[0.02] px-3 py-2.5 text-left hover:border-sky-400/50 hover:bg-sky-500/10">
                    <Avatar name={u.name} />
                    <span className="flex-1 font-medium">{u.name}</span>
                    <span aria-hidden="true" className="text-neutral-500 group-hover:text-sky-300">→</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <form onSubmit={submit} className="space-y-2.5" aria-label="New profile">
            <label className="block text-xs font-medium text-neutral-400">
              New profile
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder="Your name" className="mt-1.5 w-full rounded-lg bg-neutral-900 px-3 py-2 text-sm text-neutral-100" />
            </label>
            {err && (
              <p role="alert" className="text-sm text-red-300">
                {err}
              </p>
            )}
            <button type="submit" disabled={!name.trim() || busy} className="w-full rounded-lg bg-sky-600 py-2.5 text-sm font-medium text-white shadow-lg shadow-sky-900/40 hover:bg-sky-500 disabled:opacity-40 disabled:shadow-none">
              Create and continue
            </button>
          </form>
        </div>
        <p className="mt-5 text-center text-[11px] text-neutral-600">Every profile has its own learner: your edits never train someone else&apos;s.</p>
      </div>
    </div>
  );
}

function Shell() {
  const { user, choose } = useUser();
  if (!user) return <UserGate />;
  return (
    <Routes>
      {/* the editor is full-screen and has its own header */}
      <Route path="/projects/:id" element={<EditorPage />} />
      <Route
        path="*"
        element={
          <div className="min-h-screen">
            <nav className="sticky top-0 z-30 flex items-center gap-1 border-b border-white/[0.06] bg-neutral-950/75 px-5 py-2.5 text-sm backdrop-blur-xl">
              <Link to="/" className="mr-5" aria-label="RoughCut home">
                <Wordmark />
              </Link>
              {(
                [
                  ["/", "Projects"],
                  ["/profile", "Profile"],
                ] as const
              ).map(([to, label]) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === "/"}
                  className={({ isActive }) =>
                    `rounded-full px-3 py-1.5 ${isActive ? "bg-white/[0.08] text-white" : "text-neutral-400 hover:bg-white/[0.04] hover:text-neutral-200"}`
                  }
                >
                  {label}
                </NavLink>
              ))}
              <span className="ml-auto flex items-center gap-2 text-neutral-300">
                <Avatar name={user.name} size={26} />
                <span className="hidden sm:inline">{user.name}</span>
              </span>
              <button type="button" onClick={() => choose(null)} className="ml-1 rounded-full px-2.5 py-1 text-xs text-neutral-400 hover:bg-white/[0.06] hover:text-white">
                Switch
              </button>
            </nav>
            <Routes>
              <Route path="/" element={<ProjectsPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <UserProvider>
        <div className="min-h-screen bg-neutral-950 text-neutral-100">
          <Shell />
        </div>
      </UserProvider>
    </BrowserRouter>
  );
}
