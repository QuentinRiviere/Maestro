"""Keep slow Vast.ai hosts out of the worker's offers.

dstack cannot filter on network speed, and a pod on a slow host spends its
billed time pulling tens of GB of weights. Before a run is submitted, the
offers dstack would consider are planned, each Vast.ai offer is looked up in
the Vast.ai API (dstack names a Vast offer by its ``ask_contract_id``) to read
the host's measured download speed, and the run is pinned with
``instance_types`` to the offers that pass: every non-Vast offer plus the fast
Vast ones. dstack then picks the cheapest of those as usual.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

import requests

log = logging.getLogger("maestro.remote_worker.vast_network")

VAST_BUNDLES = "https://console.vast.ai/api/v0/bundles/"
DSTACK_SERVER_CONFIG = "/root/.dstack/server/config.yml"
PLAN_OFFERS = 100


class VastFilterError(RuntimeError):
    pass


def vast_api_key(settings: Any, config_path: str = DSTACK_SERVER_CONFIG) -> str:
    """The Vast.ai key from settings, else from the dstack server config Maestro can read."""
    if getattr(settings, "vast_api_key", ""):
        return settings.vast_api_key
    try:
        import yaml

        with open(os.path.expanduser(config_path), encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, ValueError, ImportError):
        return ""
    for project in config.get("projects") or []:
        for backend in project.get("backends") or []:
            if isinstance(backend, Mapping) and backend.get("type") == "vastai":
                return str((backend.get("creds") or {}).get("api_key") or "")
    return ""


def download_speeds(offer_ids: list[int], api_key: str,
                    session: requests.Session | None = None) -> dict[int, float]:
    """Measured download speed (Mb/s) of each Vast offer still listed."""
    if not offer_ids:
        return {}
    try:
        response = (session or requests).post(
            VAST_BUNDLES, json={"ask_contract_id": {"in": offer_ids}, "limit": len(offer_ids) + 10},
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30,
        )
        response.raise_for_status()
        offers = response.json().get("offers") or []
    except (requests.RequestException, ValueError) as exc:
        raise VastFilterError(f"Vast.ai API unreachable: {exc}") from exc
    return {int(o["ask_contract_id"]): float(o.get("inet_down") or 0) for o in offers if o.get("ask_contract_id")}


def allowed_instance_types(
    plan_offers: list[Mapping[str, Any]],
    min_mbps: float,
    speeds: Callable[[list[int]], dict[int, float]],
) -> tuple[list[str], dict[str, Any]]:
    """Instance names the run may use, and a summary for the logs and UI."""
    allowed: list[str] = []
    vast_ids: list[int] = []
    for offer in plan_offers:
        name = str((offer.get("instance") or {}).get("name") or "")
        if not name:
            continue
        if offer.get("backend") == "vastai" and name.isdigit():
            vast_ids.append(int(name))
        elif name not in allowed:
            allowed.append(name)
    measured = speeds(vast_ids) if vast_ids else {}
    fast = [str(i) for i in vast_ids if measured.get(i, 0) >= min_mbps]
    allowed += [name for name in fast if name not in allowed]
    return allowed, {"vast_offers": len(vast_ids), "vast_fast": len(fast), "min_mbps": min_mbps,
                     "other_offers": len(allowed) - len(fast)}


def pin_fast_offers(run_spec: dict[str, Any], plan: Callable[[dict[str, Any]], Mapping[str, Any]],
                    settings: Any, session: requests.Session | None = None) -> dict[str, Any]:
    """Restrict ``run_spec`` to non-Vast offers and Vast hosts at least ``min_download_mbps`` fast."""
    min_mbps = float(getattr(settings, "min_download_mbps", 0) or 0)
    backends = run_spec["configuration"].get("backends") or []
    if min_mbps <= 0 or "vastai" not in backends:
        return {"skipped": True}
    key = vast_api_key(settings)
    if not key:
        raise VastFilterError("No Vast.ai API key to check host network speed (dstack config or settings)")
    probe = {**run_spec, "configuration": {k: v for k, v in run_spec["configuration"].items() if k != "fleets"}}
    job_plans = plan(probe).get("job_plans") or [{}]
    offers = job_plans[0].get("offers") or []
    allowed, summary = allowed_instance_types(offers, min_mbps, lambda ids: download_speeds(ids, key, session))
    if not allowed:
        raise VastFilterError(f"No offer left once Vast.ai hosts under {min_mbps:g} Mb/s are excluded")
    run_spec["configuration"]["instance_types"] = allowed
    log.info("network filter: %s", summary)
    return summary
