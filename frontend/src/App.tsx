import { useEffect, useState } from "react";
import { getHealth } from "./api";

export default function App() {
  const [msg, setMsg] = useState("checking backend...");
  useEffect(() => {
    getHealth()
      .then((h) => setMsg(`backend: ${h.status}, ffmpeg: ${h.ffmpeg.ok ? "ok" : "missing"}`))
      .catch((e) => setMsg(`backend unreachable: ${e.message}`));
  }, []);
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold">RoughCut</h1>
      <p className="mt-2 text-gray-600">{msg}</p>
    </div>
  );
}
