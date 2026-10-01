import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import * as api from "../api";
import type { UserOut } from "../api-types";

const KEY = "roughcut.user";

interface UserCtx {
  user: UserOut | null;
  users: UserOut[];
  loading: boolean;
  error: string | null;
  choose: (u: UserOut | null) => void;
  create: (name: string) => Promise<void>;
}

const Ctx = createContext<UserCtx | null>(null);

function remembered(): number | null {
  try {
    const v = localStorage.getItem(KEY);
    return v ? Number(v) : null;
  } catch {
    return null;
  }
}

/** Profiles are name-only (PLAN 16.2): the chosen one is remembered in the browser. */
export function UserProvider({ children }: { children: ReactNode }) {
  const [users, setUsers] = useState<UserOut[]>([]);
  const [user, setUser] = useState<UserOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .listUsers()
      .then((list) => {
        if (!alive) return;
        setUsers(list);
        const id = remembered();
        setUser(list.find((u) => u.id === id) ?? null);
      })
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const choose = useCallback((u: UserOut | null) => {
    setUser(u);
    try {
      if (u) localStorage.setItem(KEY, String(u.id));
      else localStorage.removeItem(KEY);
    } catch {
      /* private mode: just not remembered */
    }
  }, []);

  const create = useCallback(
    async (name: string) => {
      const u = await api.createUser(name);
      setUsers((l) => [...l, u]);
      choose(u);
    },
    [choose],
  );

  const value = useMemo(() => ({ user, users, loading, error, choose, create }), [user, users, loading, error, choose, create]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useUser(): UserCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useUser needs a UserProvider");
  return c;
}
