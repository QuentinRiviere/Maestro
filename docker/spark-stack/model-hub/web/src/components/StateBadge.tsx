import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export type Tone = "ok" | "warn" | "bad" | "info" | "neutral";

/** The tone of a known state word (fit badges, job states, hash states...). */
export function toneOf(state: string): Tone {
  switch (state) {
    case "fits":
    case "ok":
    case "done":
    case "ready":
    case "trusted":
    case "mapped":
    case "installed":
    case "pinned":
      return "ok";
    case "tight":
    case "running":
    case "queued":
    case "loading":
    case "starting":
    case "not mapped":
      return "warn";
    case "too big":
    case "mismatch":
    case "untrusted":
    case "failed":
    case "error":
    case "cancelled":
    case "stopping":
      return "bad";
    case "resident":
    case "MTP":
    case "vision":
      return "info";
    default:
      return "neutral";
  }
}

const TONE_CLASS: Record<Tone, string> = {
  ok: "border-ok/50 bg-ok/10 text-ok",
  warn: "border-warn/50 bg-warn/10 text-warn",
  bad: "border-bad/50 bg-bad/10 text-bad",
  info: "border-primary/50 bg-primary/10 text-primary",
  neutral: "border-border bg-transparent text-muted-foreground",
};

export function StateBadge({
  children,
  tone,
  className,
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
  title?: string;
}) {
  const t = tone ?? (typeof children === "string" ? toneOf(children) : "neutral");
  return (
    <Badge variant="outline" title={title} className={cn("font-medium", TONE_CLASS[t], className)}>
      {children}
    </Badge>
  );
}
