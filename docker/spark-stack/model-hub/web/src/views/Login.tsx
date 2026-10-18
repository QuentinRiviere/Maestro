import { useState, type FormEvent } from "react";
import { KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { errorMessage, login } from "@/lib/api";

export function Login({ onLoggedIn }: { onLoggedIn: () => Promise<void> }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    setError("");
    setBusy(true);
    try {
      const r = await login(token);
      if (!r.ok) {
        setError(r.detail || "Login failed");
        return;
      }
      setToken("");
      await onLoggedIn();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto mt-16 max-w-sm">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-primary" />
            Log in
          </CardTitle>
          <CardDescription>The hub's access token (MODEL_HUB_TOKEN).</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="grid gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="token">Access token</Label>
              <Input
                id="token"
                type="password"
                autoComplete="current-password"
                autoFocus
                required
                value={token}
                onChange={(e) => setToken(e.target.value)}
              />
            </div>
            <Button type="submit" disabled={busy}>
              {busy ? "Logging in..." : "Log in"}
            </Button>
            {error ? <p className="text-xs text-bad">{error}</p> : null}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
