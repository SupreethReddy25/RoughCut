import { useState } from "react";
import { BrowserRouter, Link, NavLink, Navigate, Route, Routes } from "react-router-dom";
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
      <p role="alert" className="mx-auto mt-10 max-w-md rounded bg-red-500/20 p-3 text-center text-sm text-red-200">
        Cannot reach the RoughCut server: {error}
      </p>
    );
  }
  return (
    <div className="mx-auto max-w-sm p-8">
      <h1 className="mb-1 text-2xl font-semibold">RoughCut</h1>
      <p className="mb-5 text-sm text-neutral-400">Who is editing? RoughCut learns your taste per person.</p>
      {users.length > 0 && (
        <ul className="mb-5 space-y-1.5">
          {users.map((u) => (
            <li key={u.id}>
              <button type="button" onClick={() => choose(u)} className="w-full rounded border border-white/15 px-3 py-2 text-left hover:bg-white/10">
                {u.name}
              </button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={submit} className="space-y-2" aria-label="New profile">
        <label className="block text-sm text-neutral-300">
          New profile
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder="Your name" className="mt-1 w-full rounded bg-neutral-800 px-2 py-1.5" />
        </label>
        {err && (
          <p role="alert" className="text-sm text-red-300">
            {err}
          </p>
        )}
        <button type="submit" disabled={!name.trim() || busy} className="w-full rounded bg-sky-600 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-50">
          Create and continue
        </button>
      </form>
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
            <nav className="flex items-center gap-4 border-b border-white/10 px-6 py-3 text-sm">
              <Link to="/" className="mr-2 text-lg font-semibold">
                RoughCut
              </Link>
              <NavLink to="/" end className={({ isActive }) => (isActive ? "text-white" : "text-neutral-400 hover:text-neutral-200")}>
                Projects
              </NavLink>
              <NavLink to="/profile" className={({ isActive }) => (isActive ? "text-white" : "text-neutral-400 hover:text-neutral-200")}>
                Profile
              </NavLink>
              <span className="ml-auto text-neutral-400">{user.name}</span>
              <button type="button" onClick={() => choose(null)} className="text-sky-400 hover:underline">
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
