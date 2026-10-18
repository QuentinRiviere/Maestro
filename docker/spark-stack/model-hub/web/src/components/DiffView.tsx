import { cn } from "@/lib/utils";

/** A unified diff with added and removed lines coloured. */
export function DiffView({ text, className }: { text: string; className?: string }) {
  return (
    <pre className={cn("code-block", className)}>
      {text.split("\n").map((l, i) => (
        <div
          key={i}
          className={cn(
            l.startsWith("+") && !l.startsWith("+++") && "text-ok",
            l.startsWith("-") && !l.startsWith("---") && "text-bad",
            l.startsWith("@@") && "text-primary",
          )}
        >
          {l || " "}
        </div>
      ))}
    </pre>
  );
}

export function CodeBlock({ text, className }: { text: string; className?: string }) {
  return <pre className={cn("code-block", className)}>{text}</pre>;
}
