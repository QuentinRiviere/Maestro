// A small toaster: the hub's CSP (style-src 'self') forbids the inline
// stylesheet sonner injects, so this one is plain Tailwind.
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

type Kind = "info" | "success" | "error";

interface Toast {
  id: number;
  kind: Kind;
  title: string;
  description?: string;
}

interface ToastApi {
  toast: (title: string, opts?: { description?: string; kind?: Kind }) => void;
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast outside ToastProvider");
  return ctx;
}

const ICONS: Record<Kind, typeof Info> = { info: Info, success: CheckCircle2, error: AlertTriangle };

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seq = useRef(0);

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const toast = useCallback<ToastApi["toast"]>(
    (title, opts = {}) => {
      const id = ++seq.current;
      const kind = opts.kind || "info";
      setToasts((t) => [...t.slice(-4), { id, kind, title, description: opts.description }]);
      window.setTimeout(() => dismiss(id), kind === "error" ? 9000 : 5000);
    },
    [dismiss],
  );

  const value = useMemo<ToastApi>(
    () => ({
      toast,
      success: (title, description) => toast(title, { description, kind: "success" }),
      error: (title, description) => toast(title, { description, kind: "error" }),
    }),
    [toast],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-[100] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2">
        {toasts.map((t) => {
          const Icon = ICONS[t.kind];
          return (
            <div
              key={t.id}
              role="status"
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-lg border bg-popover p-3 text-sm text-popover-foreground shadow-lg animate-in fade-in-0 slide-in-from-bottom-2",
                t.kind === "error" && "border-bad/40",
                t.kind === "success" && "border-ok/40",
              )}
            >
              <Icon
                className={cn(
                  "mt-0.5 size-4 shrink-0",
                  t.kind === "error" ? "text-bad" : t.kind === "success" ? "text-ok" : "text-muted-foreground",
                )}
              />
              <div className="min-w-0 flex-1">
                <div className="font-medium break-words">{t.title}</div>
                {t.description ? (
                  <div className="mt-0.5 text-xs break-words text-muted-foreground">{t.description}</div>
                ) : null}
              </div>
              <button
                type="button"
                aria-label="Dismiss"
                onClick={() => dismiss(t.id)}
                className="rounded p-0.5 text-muted-foreground hover:text-foreground"
              >
                <X className="size-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
