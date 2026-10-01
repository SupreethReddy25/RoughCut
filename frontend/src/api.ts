import type {
  EventIn,
  ExplainOut,
  FinalizeOut,
  JobOut,
  ProfileOut,
  ProjectOut,
  ProjectSummary,
  StateOut,
  StylesOut,
  TrackOut,
  UserOut,
} from "./api-types";

/** An error answer of the API: `status` is the HTTP code (0 = unreachable), `body` the JSON (e.g. {detail, replaced_by}). */
export class ApiError extends Error {
  readonly status: number;
  readonly body: Record<string, unknown>;

  constructor(status: number, message: string, body: Record<string, unknown> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(method: string, url: string, body?: unknown, form?: FormData): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
    });
  } catch (e) {
    throw new ApiError(0, `The server is unreachable (${(e as Error).message}).`);
  }
  if (!res.ok) {
    let data: Record<string, unknown> = {};
    try {
      data = await res.json();
    } catch {
      /* not JSON */
    }
    const detail = data.detail;
    const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? "The request was not valid." : res.statusText;
    throw new ApiError(res.status, msg || `HTTP ${res.status}`, data);
  }
  return (await res.json()) as T;
}

const get = <T,>(url: string) => request<T>("GET", url);
const post = <T,>(url: string, body?: unknown) => request<T>("POST", url, body ?? {});

export const getHealth = () => get<{ status: string; ffmpeg: { ok: boolean } }>("/api/health");
export const listUsers = () => get<UserOut[]>("/api/users");
export const createUser = (name: string) => post<UserOut>("/api/users", { name });
export const getProfile = (userId: number) => get<ProfileOut>(`/api/users/${userId}/profile`);
export const getStyles = () => get<StylesOut>("/api/styles");

export const listProjects = (userId: number) => get<ProjectSummary[]>(`/api/projects?user_id=${userId}`);
export const getProject = (id: number) => get<ProjectOut>(`/api/projects/${id}`);
export const createProject = (form: FormData) => request<ProjectSummary>("POST", "/api/projects", undefined, form);
export const suggest = (id: number) => post<StateOut>(`/api/projects/${id}/suggest`);
export const postEvent = (id: number, ev: EventIn) => post<StateOut>(`/api/projects/${id}/events`, ev);
export const undo = (id: number) => post<StateOut>(`/api/projects/${id}/undo`);
export const redo = (id: number) => post<StateOut>(`/api/projects/${id}/redo`);
export const markSeen = (id: number, unitIds: number[]) =>
  post<{ marked: number }>(`/api/projects/${id}/seen`, { unit_ids: unitIds });
export const finalize = (id: number, secondsActive: number) =>
  post<FinalizeOut>(`/api/projects/${id}/finalize`, { seconds_active: secondsActive });
export const startRender = (id: number, preset: string) =>
  post<{ job: JobOut; preset: string; duration: number; warning: string | null }>(`/api/projects/${id}/render`, { preset });
export const getJob = (jobId: string) => get<JobOut>(`/api/jobs/${jobId}`);
export const explain = (id: number, suggestionId: number) =>
  get<ExplainOut>(`/api/projects/${id}/suggestions/${suggestionId}/explain`);

export const listMusic = (userId: number) => get<TrackOut[]>(`/api/music?user_id=${userId}`);
export const uploadMusic = (userId: number, file: File) => {
  const form = new FormData();
  form.append("file", file, file.name);
  return request<TrackOut>("POST", `/api/users/${userId}/music`, undefined, form);
};
export const musicUrl = (trackId: string, userId: number) => `/api/music/${encodeURIComponent(trackId)}/audio?user_id=${userId}`;

export const proxyUrl = (id: number) => `/api/projects/${id}/proxy`;
export const thumbnailUrl = (id: number) => `/api/projects/${id}/thumbnail`;
export const downloadUrl = (id: number, fmt: "mp4" | "srt", preset?: string) =>
  `/api/projects/${id}/download?fmt=${fmt}${preset ? `&preset=${preset}` : ""}`;
export const fontUrl = (file: string) => `/api/fonts/${encodeURIComponent(file)}`;
