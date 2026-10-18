import { useCallback, useEffect, useState } from "react";
import { Boxes, Compass, Download, FileCode2, HardDrive, LogOut, Server } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ToastProvider, useToast } from "@/components/toast";
import { api, errorMessage, forgetToken, isLoginRequired, login, onUnauthorized, setCsrf, storedToken } from "@/lib/api";
import { gb } from "@/lib/format";
import type { Status } from "@/lib/types";
import { Login } from "@/views/Login";
import { Catalog } from "@/views/Catalog";
import { Explore } from "@/views/Explore";
import { Downloads } from "@/views/Downloads";
import { Installed } from "@/views/Installed";
import { ConfigView } from "@/views/ConfigView";
import { RepoDialog, type RepoRequest } from "@/views/RepoDialog";

export type TabName = "catalog" | "explore" | "downloads" | "installed" | "config";

const TABS: { id: TabName; label: string; icon: typeof Boxes }[] = [
  { id: "catalog", label: "Catalog", icon: Boxes },
  { id: "explore", label: "Explore", icon: Compass },
  { id: "downloads", label: "Downloads", icon: Download },
  { id: "installed", label: "Installed", icon: HardDrive },
  { id: "config", label: "llama-swap config", icon: FileCode2 },
];

function statusLine(st: Status) {
  const b = st.budget;
  let line =
    `disk free ${gb(st.disk.free)} · memory ${b.total_gb} GB − ${b.reserved_gb} reserved − ` +
    `${b.resident_gb.toFixed(1)} resident = ${b.available_gb.toFixed(1)} GB for on-demand models`;
  if (b.swap_reserve_gb) line += ` (${b.resident_available_gb.toFixed(1)} GB for resident ones: ${b.swap_reserve_gb} GB swap reserve)`;
  return line;
}

function Hub() {
  const toast = useToast();
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [tab, setTab] = useState<TabName>("catalog");
  const [repo, setRepo] = useState<RepoRequest | null>(null);

  const report = useCallback(
    (err: unknown) => {
      if (!isLoginRequired(err)) toast.error(errorMessage(err));
    },
    [toast],
  );

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api<Status>("/api/status"));
    } catch (e) {
      report(e);
    }
  }, [report]);

  const boot = useCallback(async () => {
    let s = await api<{ authenticated: boolean; csrf?: string } | null>("/api/session", { allow401: true, allowError: true });
    if (!s || !s.authenticated) {
      // A token kept in this tab logs in again (the hub forgot the session, e.g. after a restart).
      const token = storedToken();
      if (token && (await login(token)).ok) {
        s = await api<{ authenticated: boolean; csrf?: string } | null>("/api/session", { allow401: true, allowError: true });
      } else if (token) {
        forgetToken();
      }
    }
    if (!s || !s.authenticated) {
      setAuthed(false);
      return;
    }
    setCsrf(s.csrf || "");
    setAuthed(true);
    await refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    onUnauthorized(() => {
      setAuthed(false);
      setRepo(null);
    });
    boot().catch(report);
    return () => onUnauthorized(null);
  }, [boot, report]);

  const logout = async () => {
    try {
      await api("/api/logout", { method: "POST" });
    } catch {
      /* ignore */
    }
    forgetToken();
    setCsrf("");
    setAuthed(false);
    setStatus(null);
  };

  const openRepo = useCallback((req: RepoRequest) => setRepo(req), []);
  const goToDownloads = useCallback(() => {
    setRepo(null);
    setTab("downloads");
  }, []);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b bg-card/95 backdrop-blur supports-backdrop-filter:bg-card/80">
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5">
          <div className="flex items-center gap-2 font-semibold">
            <Server className="size-4 text-primary" />
            Spark model hub
          </div>
          {authed ? (
            <>
              <nav className="order-last w-full sm:order-none sm:w-auto">
                <Tabs value={tab} onValueChange={(v) => setTab(v as TabName)}>
                  <TabsList>
                    {TABS.map((t) => (
                      <TabsTrigger key={t.id} value={t.id}>
                        <t.icon data-icon="inline-start" />
                        {t.label}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                </Tabs>
              </nav>
              <div className="ml-auto flex items-center gap-3">
                {status ? (
                  <span className="hidden max-w-[520px] truncate text-xs text-muted-foreground lg:inline" title={statusLine(status)}>
                    {statusLine(status)}
                    {status.base_error ? <span className="text-bad"> · base config error: {status.base_error}</span> : null}
                  </span>
                ) : null}
                <Button variant="ghost" size="sm" onClick={logout}>
                  <LogOut data-icon="inline-start" />
                  Log out
                </Button>
              </div>
            </>
          ) : null}
        </div>
      </header>

      <main className="mx-auto max-w-[1200px] px-4 py-4">
        {authed === null ? null : !authed ? (
          <Login onLoggedIn={boot} />
        ) : (
          <Tabs value={tab} onValueChange={(v) => setTab(v as TabName)}>
            {status?.base_error ? (
              <p className="mb-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad lg:hidden">
                base config error: {status.base_error}
              </p>
            ) : null}
            <TabsContent value="catalog">
              {tab === "catalog" ? <Catalog onOpen={openRepo} report={report} /> : null}
            </TabsContent>
            <TabsContent value="explore">
              {tab === "explore" ? <Explore status={status} onOpen={openRepo} /> : null}
            </TabsContent>
            <TabsContent value="downloads">{tab === "downloads" ? <Downloads report={report} /> : null}</TabsContent>
            <TabsContent value="installed">
              {tab === "installed" ? <Installed status={status} report={report} /> : null}
            </TabsContent>
            <TabsContent value="config">{tab === "config" ? <ConfigView report={report} /> : null}</TabsContent>
          </Tabs>
        )}
      </main>

      <RepoDialog request={repo} status={status} onClose={() => setRepo(null)} onQueued={goToDownloads} />
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <TooltipProvider>
        <Hub />
      </TooltipProvider>
    </ToastProvider>
  );
}
