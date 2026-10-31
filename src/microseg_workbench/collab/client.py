"""Client for the optional metadata service (standard library ``urllib``)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ServiceError(RuntimeError):
    pass


class ServiceClient:
    def __init__(self, base_url: str, token: str, timeout: float = 10.0, retries: int = 3) -> None:
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        """All endpoints are idempotent (registrations of identical records are no-ops), so retrying is safe."""
        for attempt in range(self.retries):
            try:
                return self._once(method, path, payload)
            except ServiceError as exc:
                if not str(exc).startswith("cannot reach") or attempt == self.retries - 1:
                    raise
                time.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")

    def _once(self, method: str, path: str, payload: Any = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - intranet URL by config
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", "")
            except ValueError:
                detail = ""
            raise ServiceError(f"{exc.code} {detail or exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ServiceError(f"cannot reach {self.base}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def whoami(self) -> str:
        return self._request("GET", "/api/whoami")["user"]

    def datasets(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/datasets")

    def runs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/runs")

    def approved(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/approved")

    def register(self, kind: str, record: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/{kind}", record)

    def sync_from_share(self, store) -> int:
        """Register every shared record (datasets, then runs, then reviews). Identical records are no-ops."""
        sent = 0
        for d in store.list_datasets():
            self.register("datasets", d)
            sent += 1
        runs = store.list_runs()
        for r in runs:
            record = {k: v for k, v in r.items() if k != "review"}
            self.register("runs", record)
            sent += 1
        for r in runs:
            for review in store.reviews(r["run_id"]):
                try:
                    self.register("reviews", review)
                    sent += 1
                except ServiceError as exc:
                    if not str(exc).startswith("403"):
                        raise
        return sent
