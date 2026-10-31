"""Optional lightweight metadata service for the Ubuntu VM (standard library only).

It indexes what is published on the shared drive (datasets, runs, reviews), authenticates users with tokens
and records who submitted what. Bulk data never passes through it. Its SQLite database must live on the VM's
local disk: the service is the only writer (one write lock, WAL mode). The shared drive remains the authority;
the index can always be rebuilt with ``msw share sync``.

Run::

    python -m microseg_workbench.collab.server add-user alice --db /var/lib/msw/meta.sqlite   # prints a token
    python -m microseg_workbench.collab.server serve --db /var/lib/msw/meta.sqlite --host 0.0.0.0 --port 5080
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .. import __version__

MAX_BODY = 2 * 1024 * 1024
KINDS = {
    "datasets": ("microseg-workbench.shared-dataset/1", "dataset_id"),
    "runs": ("microseg-workbench.shared-run/1", "run_id"),
    "reviews": ("microseg-workbench.review/1", None),
}
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL UNIQUE, created_at TEXT);
CREATE TABLE IF NOT EXISTS records (
    kind TEXT NOT NULL, key TEXT NOT NULL, record TEXT NOT NULL, submitted_by TEXT NOT NULL,
    submitted_at TEXT NOT NULL, PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, user TEXT, action TEXT, detail TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class MetadataDB:
    def __init__(self, path: Path) -> None:
        text = str(path)
        if text.startswith("\\\\") or text.startswith("//"):
            raise ValueError("The metadata database must be on the server's local disk, not a network share.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.lock = threading.Lock()
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def add_user(self, name: str) -> str:
        token = secrets.token_urlsafe(24)
        with self.lock, self._connect() as con:
            con.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
                        (name, hashlib.sha256(token.encode()).hexdigest(), _now()))
        return token

    def user_for(self, token: str) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT name FROM users WHERE token_sha256 = ?",
                              (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        return row["name"] if row else None

    def put(self, kind: str, record: dict[str, Any], user: str) -> tuple[int, dict[str, Any]]:
        schema, key_field = KINDS[kind]
        if record.get("schema") != schema:
            return HTTPStatus.BAD_REQUEST, {"error": f"expected schema {schema}"}
        if kind == "reviews":
            key = hashlib.sha256(_canonical(record).encode()).hexdigest()[:32]
            run_id = record.get("run_id")
            if not run_id or record.get("decision") not in ("approved", "rejected", "needs_changes"):
                return HTTPStatus.BAD_REQUEST, {"error": "review needs run_id and a valid decision"}
        else:
            key = record.get(key_field)
            if not isinstance(key, str) or not key:
                return HTTPStatus.BAD_REQUEST, {"error": f"missing {key_field}"}
        with self.lock, self._connect() as con:
            if kind == "reviews":
                run = con.execute("SELECT record FROM records WHERE kind='runs' AND key=?", (run_id,)).fetchone()
                if run is None:
                    return HTTPStatus.CONFLICT, {"error": f"run {run_id} is not registered"}
                publisher = json.loads(run["record"])["published_by"]
                if record["decision"] == "approved" and publisher in (record.get("reviewer"), user):
                    return HTTPStatus.FORBIDDEN, {"error": "self-approval is not allowed"}
            existing = con.execute("SELECT record, submitted_by FROM records WHERE kind=? AND key=?",
                                   (kind, key)).fetchone()
            if existing is not None:
                if _canonical(json.loads(existing["record"])) == _canonical(record):
                    return HTTPStatus.OK, {"status": "exists", "key": key}
                return HTTPStatus.CONFLICT, {"error": f"{kind[:-1]} {key} is already registered with different content"}
            con.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?)", (kind, key, _canonical(record), user, _now()))
            con.execute("INSERT INTO audit (at, user, action, detail) VALUES (?, ?, ?, ?)",
                        (_now(), user, f"register {kind}", key))
        return HTTPStatus.CREATED, {"status": "created", "key": key}

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT record, submitted_by, submitted_at FROM records WHERE kind=? ORDER BY key",
                               (kind,)).fetchall()
        out = []
        for r in rows:
            rec = json.loads(r["record"])
            identity = rec.get("reviewer") if kind == "reviews" else rec.get("published_by")
            out.append({"record": rec, "submitted_by": r["submitted_by"], "submitted_at": r["submitted_at"],
                        "identity_verified": identity == r["submitted_by"]})
        return out

    def runs_with_status(self) -> list[dict[str, Any]]:
        reviews: dict[str, list[dict[str, Any]]] = {}
        for item in self.list("reviews"):
            reviews.setdefault(item["record"]["run_id"], []).append(item)
        out = []
        for item in self.list("runs"):
            rid = item["record"]["run_id"]
            ordered = sorted(reviews.get(rid, []), key=lambda r: r["record"]["reviewed_at"])
            item["review_status"] = ordered[-1]["record"]["decision"] if ordered else "unreviewed"
            item["reviews"] = ordered
            out.append(item)
        return out


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else html.escape(str(v))


def registry_page(db: MetadataDB) -> str:
    """Read-only HTML overview (for an intranet portal tile). Server-rendered; no scripts or external assets."""
    runs = db.runs_with_status()
    datasets = db.list("datasets")

    def run_rows(items: list[dict[str, Any]], approved: bool) -> str:
        rows = []
        for item in items:
            r = item["record"]
            summary = r.get("summary") or {}
            classes = ", ".join(f"{html.escape(k)} {_fmt(v)}" for k, v in (summary.get("per_class_iou") or {}).items())
            last = item["reviews"][-1]["record"] if item["reviews"] else None
            review = (f"{html.escape(last['decision'])} by {html.escape(last['reviewer'])} "
                      f"{html.escape(last['reviewed_at'][:10])}" if last else "unreviewed")
            if approved and item["review_status"] != "approved":
                continue
            rows.append(f"<tr><td>{html.escape(r['run_id'])}</td><td>{html.escape(r['dataset_id'])}</td>"
                        f"<td>{_fmt(summary.get('mean_iou'))}</td><td>{classes}</td>"
                        f"<td>{html.escape(r['published_by'])}</td><td>{review}</td></tr>")
        return "".join(rows) or "<tr><td colspan='6'>none</td></tr>"

    head = "<tr><th>Run</th><th>Dataset version</th><th>Test mIoU</th><th>Per-class IoU</th><th>Published by</th>" \
           "<th>Latest review</th></tr>"
    ds_rows = "".join(
        f"<tr><td>{html.escape(d['record']['dataset_id'])}</td><td>{html.escape(str(d['record'].get('split_counts')))}"
        f"</td><td>{html.escape(', '.join(d['record'].get('class_names', [])))}</td>"
        f"<td>{html.escape(d['record']['published_by'])}</td></tr>" for d in datasets) or \
        "<tr><td colspan='4'>none</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MicroSegWorkbench models</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;background:#fff}}
table{{border-collapse:collapse;margin-bottom:24px;width:100%}}th,td{{border:1px solid #ccc;padding:4px 8px;
text-align:left;font-size:14px}}th{{background:#f2f2f2}}.note{{color:#555}}</style></head><body>
<h1>MicroSegWorkbench model registry</h1>
<p class="note">Read-only index of the shared folder. Use the MicroSegWorkbench desktop app to download,
compare, review or apply models. Only models approved by someone other than their publisher are deployment
candidates. Version {html.escape(__version__)}.</p>
<h2>Approved models</h2><table>{head}{run_rows(runs, True)}</table>
<h2>All published runs</h2><table>{head}{run_rows(runs, False)}</table>
<h2>Dataset versions</h2><table><tr><th>Dataset version</th><th>Images (train/val/test)</th><th>Classes</th>
<th>Published by</th></tr>{ds_rows}</table></body></html>"""


class Handler(BaseHTTPRequestHandler):
    db: MetadataDB
    public_read: bool = False
    server_version = f"msw-metadata/{__version__}"

    def _send_html(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default; audit table records writes
        pass

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _user(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        return self.db.user_for(auth[7:]) if auth.startswith("Bearer ") else None

    def do_GET(self) -> None:  # noqa: N802
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
        if parts == ["api", "health"]:
            return self._send(HTTPStatus.OK, {"status": "ok", "version": __version__})
        if parts in ([""], ["registry"]):
            if self.public_read or self._user():
                return self._send_html(HTTPStatus.OK, registry_page(self.db))
            return self._send_html(HTTPStatus.UNAUTHORIZED, "<!doctype html><title>Sign-in required</title>"
                                   "<p>This registry is not public. Use the MicroSegWorkbench desktop app.</p>")
        user = self._user()
        if user is None:
            return self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid token"})
        if parts == ["api", "whoami"]:
            return self._send(HTTPStatus.OK, {"user": user})
        if parts == ["api", "datasets"]:
            return self._send(HTTPStatus.OK, self.db.list("datasets"))
        if parts == ["api", "runs"]:
            return self._send(HTTPStatus.OK, self.db.runs_with_status())
        if parts == ["api", "approved"]:
            return self._send(HTTPStatus.OK, [r for r in self.db.runs_with_status()
                                              if r["review_status"] == "approved"])
        if len(parts) == 3 and parts[:2] == ["api", "runs"]:
            match = [r for r in self.db.runs_with_status() if r["record"]["run_id"] == parts[2]]
            return self._send(HTTPStatus.OK, match[0]) if match else self._send(HTTPStatus.NOT_FOUND,
                                                                                 {"error": "unknown run"})
        return self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        user = self._user()
        if user is None:
            return self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid token"})
        if len(parts) != 2 or parts[0] != "api" or parts[1] not in KINDS:
            return self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body missing or too large"})
        try:
            record = json.loads(self.rfile.read(length))
        except ValueError:
            return self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
        if not isinstance(record, dict):
            return self._send(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON object"})
        status, payload = self.db.put(parts[1], record, user)
        return self._send(status, payload)


class MetadataServer(ThreadingHTTPServer):
    request_queue_size = 128  # default 5 resets connections when many colleagues sync at once
    daemon_threads = True


def make_server(db_path: Path, host: str = "127.0.0.1", port: int = 5080,
                public_read: bool = False) -> ThreadingHTTPServer:
    db = MetadataDB(db_path)
    handler = type("BoundHandler", (Handler,), {"db": db, "public_read": public_read})
    server = MetadataServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="msw-service", description="MicroSegWorkbench metadata service")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--db", required=True, help="SQLite file on the VM's LOCAL disk")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5080, help="default 5080 (8765 is used by pytex in the ML suite)")
    s.add_argument("--public-read", action="store_true",
                   help="let anyone on the intranet view the read-only registry page at / (for a portal tile)")
    a = sub.add_parser("add-user", help="create/replace a user and print their token")
    a.add_argument("name")
    a.add_argument("--db", required=True)
    args = p.parse_args(argv)
    if args.cmd == "add-user":
        print(MetadataDB(Path(args.db)).add_user(args.name))
        return 0
    server = make_server(Path(args.db), args.host, args.port, public_read=args.public_read)
    print(f"Metadata service on http://{args.host}:{server.server_address[1]} (database {args.db})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
