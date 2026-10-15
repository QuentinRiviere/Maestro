"""Paginated, searchable list of saved Director runs.

``GET /api/v1/director/pipelines`` used to return every saved run. Run pickers
now page through it newest first instead: ``limit`` starts paging, ``cursor``
continues after the last row of the previous page, ``q`` searches the id,
scene description, pipeline type, label and workspace, and ``status`` keeps
only the given comma-separated statuses (``repairing`` matches runs whose
repair is active). Without any of those parameters the endpoint still returns
the whole list, so older callers keep working.
"""
from __future__ import annotations

import base64
import binascii
import json
from typing import Callable, Iterable, Optional
from urllib.parse import quote

from services import director_pipeline as _pipeline

MAX_PAGE_SIZE = 200
REPAIRING_STATUS = "repairing"


class InvalidCursorError(ValueError):
    pass


def encode_cursor(summary: dict) -> str:
    raw = json.dumps([_pipeline.pipeline_activity_time(summary), str(summary.get("id") or "")])
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[float, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        activity, pid = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return float(activity), str(pid)
    except (ValueError, TypeError, binascii.Error, UnicodeError):
        raise InvalidCursorError("Invalid cursor") from None


def _parse_statuses(status: Optional[str]) -> set[str]:
    return {part.strip().lower() for part in (status or "").split(",") if part.strip()}


def _matches(summary: dict, terms: list[str], statuses: set[str]) -> bool:
    if statuses:
        status_ok = str(summary.get("status") or "").lower() in statuses
        if not status_ok and REPAIRING_STATUS in statuses:
            status_ok = summary.get("repair_status") in _pipeline._REPAIR_ACTIVE_STATUSES
        if not status_ok:
            return False
    if terms:
        text = summary.get("_search_text") or str(summary.get("id") or "").lower()
        if not all(term in text for term in terms):
            return False
    return True


def public_summary(summary: dict) -> dict:
    """Drop server-only keys and add the thumbnail URL."""
    item = {key: value for key, value in summary.items() if not key.startswith("_")}
    if item.get("output_count") or item.get("clip_count"):
        item["thumbnail_url"] = (
            f"/api/v1/director/pipelines/{quote(str(item.get('id') or ''), safe='')}/thumbnail"
        )
    return item


def page_pipeline_summaries(
    summaries: Iterable[dict],
    *,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    q: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """Filter, sort newest first and slice ``summaries``.

    Sorting happens before slicing and the cursor is a position in that
    order, (activity time, id), so rows are neither skipped nor repeated
    when other runs are added between two page requests.
    """
    terms = [term for term in (q or "").lower().split() if term]
    statuses = _parse_statuses(status)
    rows = sorted(
        (summary for summary in summaries if _matches(summary, terms, statuses)),
        key=_pipeline.pipeline_sort_key,
    )
    total = len(rows)
    if cursor:
        after = decode_cursor(cursor)
        after_key = (-after[0], after[1])
        rows = [row for row in rows if _pipeline.pipeline_sort_key(row) > after_key]
    next_cursor = None
    if limit is not None:
        limit = max(1, min(int(limit), MAX_PAGE_SIZE))
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = encode_cursor(rows[-1])
    return {
        "pipelines": [public_summary(row) for row in rows],
        "next_cursor": next_cursor,
        "total": total,
    }


def register_routes(api, *, get_base_dir: Callable[[], str]) -> None:
    from fastapi.responses import JSONResponse

    @api.get("/api/v1/director/pipelines")
    def list_saved_pipelines(
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        q: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """List saved Director runs, newest first, optionally paged."""
        summaries = _pipeline.list_pipeline_states(get_base_dir())
        try:
            return page_pipeline_summaries(
                summaries, limit=limit, cursor=cursor, q=q, status=status,
            )
        except InvalidCursorError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
