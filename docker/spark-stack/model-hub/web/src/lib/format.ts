export const gb = (bytes: number) => (bytes / 1e9).toFixed(bytes >= 10e9 ? 1 : 2) + " GB";

/** A GB figure already in GB (memory view), "?" when unknown. */
export const gbs = (v: number | null | undefined) => (v == null ? "?" : `${Number(v).toFixed(1)} GB`);

export const date = (s: string | null | undefined) => (s ? String(s).slice(0, 10) : "");

export const num = (n: number | null | undefined) => (n == null ? "" : Number(n).toLocaleString());

export function fmtDuration(seconds: number) {
  const s = Math.round(seconds);
  if (s < 60) return s + " s";
  if (s < 3600) return Math.floor(s / 60) + " min " + (s % 60) + " s";
  return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
}

/** llama-swap id suggested from a repo id: `owner/Qwen3-GGUF` -> `qwen3`. */
export function suggestId(repo: string) {
  return repo
    .split("/")[1]
    .toLowerCase()
    .replace(/-gguf$/, "")
    .replace(/[^a-z0-9._-]+/g, "-")
    .slice(0, 60);
}
