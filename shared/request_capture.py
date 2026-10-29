"""Persist uploaded files and complete API responses independently of handlers.

SPEECH_CAPTURE_DIR enables capture. Wrap the completed ASGI app so validation,
authentication and unhandled errors are captured as well as successful inference.
Request bodies are replayed unchanged; credentials are omitted from saved metadata.
"""
import contextvars
import datetime
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import traceback
import uuid
import wave
from pathlib import Path
from urllib.parse import parse_qsl


capture_context = contextvars.ContextVar("speech_capture_context", default=None)
LOG = logging.getLogger("speech_capture")


def _secret_key(key):
    key = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if key in {"tokenid", "recordid"}:  # SpeechSuper request IDs, not credentials.
        return False
    return any(s in key for s in ("token", "secret", "password", "signature", "authorization")) or key in {
        "key", "apikey", "appkey", "cookie", "credential", "credentials", "sessionkey", "accesskey",
    }


def _clean(value, secrets=()):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _secret_key(k) else _clean(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def _json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _credentials(value, into):
    if isinstance(value, dict):
        for k, v in value.items():
            if _secret_key(k) and isinstance(v, str) and v:
                into.add(v)
            else:
                _credentials(v, into)
    elif isinstance(value, list):
        for v in value:
            _credentials(v, into)


def _store_file(source, directory, index, filename, field, content_type):
    extension = Path(filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
        extension = ".bin"
    target = directory / ("audio-%02d%s" % (index, extension))
    digest, size = hashlib.sha256(), 0
    source.seek(0)
    with target.open("wb") as handle:
        os.chmod(target, 0o600)
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    metadata = {"field": field, "filename": filename, "content_type": content_type,
                "path": target.name, "bytes": size, "sha256": digest.hexdigest()}
    try:
        with wave.open(str(target), "rb") as audio:
            metadata["wav"] = {"sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
                               "sample_width": audio.getsampwidth(), "frames": audio.getnframes(),
                               "sec": audio.getnframes() / audio.getframerate()}
    except (wave.Error, EOFError):
        pass
    return metadata


async def _parse_body(body, headers, directory, record, secrets):
    content_type = headers.get("content-type", "")
    record["files"], record["form"] = [], {}
    body.seek(0)
    if content_type.startswith("multipart/form-data"):
        from starlette.datastructures import Headers, UploadFile
        from starlette.formparsers import MultiPartParser

        async def stream():
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

        form = await MultiPartParser(Headers({"content-type": content_type}), stream()).parse()
        try:
            for key, value in form.multi_items():
                if isinstance(value, UploadFile):
                    record["files"].append(_store_file(value.file, directory, len(record["files"]),
                                                       value.filename, key, value.content_type))
                else:
                    try:
                        parsed = json.loads(value)
                    except (ValueError, TypeError):
                        parsed = value
                    _credentials({key: parsed}, secrets)
                    record["form"].setdefault(key, []).append(parsed)
        finally:
            await form.close()
    elif content_type.startswith("application/json"):
        record["form"] = json.load(body)
        _credentials(record["form"], secrets)
    elif content_type.startswith("application/x-www-form-urlencoded"):
        for key, value in parse_qsl(body.read().decode("utf-8", "replace"), keep_blank_values=True):
            record["form"].setdefault(key, []).append(value)
            _credentials({key: value}, secrets)
    elif content_type.startswith("audio/") or content_type.startswith("application/octet-stream"):
        record["files"].append(_store_file(body, directory, 0, "upload.bin", "body", content_type))


class SpeechCaptureMiddleware:
    """Outer ASGI wrapper: capture responses including FastAPI's 422/500 replies."""

    def __init__(self, app, root, service):
        self.app, self.root, self.service = app, Path(root), service
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started, moment = time.perf_counter(), datetime.datetime.now().astimezone()
        rid = moment.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
        directory = self.root / moment.strftime("%Y%m%d") / rid
        try:
            directory.mkdir(parents=True, mode=0o700)
            os.chmod(directory.parent, 0o700)
        except OSError:
            LOG.exception("[capture] cannot create request directory")
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        secrets = set()
        for key in ("authorization", "x-api-key", "cookie"):
            if headers.get(key):
                secrets.add(headers[key])
                if key == "authorization" and " " in headers[key]:
                    secrets.add(headers[key].split(" ", 1)[1])
        query = {}
        for key, value in parse_qsl(scope.get("query_string", b"").decode("utf-8", "replace"), keep_blank_values=True):
            query.setdefault(key, []).append(value)
            _credentials({key: value}, secrets)
        record = {"schema_version": 1, "rid": rid, "time": moment.isoformat(), "service": self.service,
                  "deployment_id": os.environ.get("SPEECH_DEPLOYMENT_ID", "unknown"),
                  "method": scope["method"], "path": scope["path"], "query": _clean(query, secrets),
                  "headers": {k: v for k, v in headers.items() if k in {
                      "content-type", "content-length", "user-agent", "origin", "host", "x-request-id",
                      "x-client-type", "x-client-version", "x-app-version", "x-session-id"}},
                  "client": (scope.get("client") or [None])[0], "state": "receiving"}
        _json_write(directory / "request.json", record)
        # Unnamed temporary files disappear even if the process is killed. The
        # original multipart body can contain credentials; it is never archived.
        body = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b", dir=str(self.root))
        response = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b", dir=str(self.root))
        state = {"rid": rid}
        context_token = capture_context.set(state)
        disconnected, response_complete = False, False
        response_type, error = "", None
        try:
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected = True
                    break
                if message["type"] == "http.request":
                    chunk = message.get("body", b"")
                    body.write(chunk)
                    size += len(chunk)
                    if not message.get("more_body", False):
                        break
            record["request_bytes"] = size
            try:
                await _parse_body(body, headers, directory, record, secrets)
            except Exception as exc:
                record["parse_error"] = "%s: %s" % (type(exc).__name__, str(exc))
                LOG.exception("[capture] upload parsing failed rid=%s", rid)
            record["state"] = "processing"
            _json_write(directory / "request.json", _clean(record, secrets))
            body.seek(0)
            replay_complete = False

            async def replay():
                nonlocal replay_complete
                if replay_complete:
                    return {"type": "http.disconnect"} if disconnected else await receive()
                chunk = body.read(1024 * 1024)
                replay_complete = body.tell() >= size
                return {"type": "http.request", "body": chunk, "more_body": not replay_complete}

            async def capture_send(message):
                nonlocal response_type, response_complete
                if message["type"] == "http.response.start":
                    record["status"] = message["status"]
                    response_headers = list(message.get("headers", []))
                    response_type = dict(response_headers).get(b"content-type", b"").decode("latin-1")
                    message = {**message, "headers": response_headers + [(b"x-capture-id", rid.encode("ascii"))]}
                elif message["type"] == "http.response.body":
                    # Static HTML may contain injected auth tokens. Only JSON API
                    # responses and error text belong in analysis records.
                    if "json" in response_type or record.get("status", 0) >= 400:
                        response.write(message.get("body", b""))
                    response_complete = not message.get("more_body", False)
                await send(message)

            if disconnected:
                record["status"] = 499
            else:
                await self.app(scope, replay, capture_send)
        except BaseException as exc:
            error = exc
            record.setdefault("status", 500)
            record["exception"] = {"type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc()}
            raise
        finally:
            try:
                response.seek(0)
                response_bytes = response.read()
                if response_bytes:
                    try:
                        record["result"] = json.loads(response_bytes)
                    except (ValueError, UnicodeDecodeError):
                        record["result"] = {"body": response_bytes.decode("utf-8", "replace")}
                record["response_complete"] = response_complete
                record["ms"] = round((time.perf_counter() - started) * 1000, 2)
                record["state"] = "disconnected" if disconnected else "exception" if error else "complete"
                record["diagnostics"] = {k: v for k, v in state.items() if k != "rid"}
                saved = _clean(record, secrets)
                _json_write(directory / "request.json", saved)
                line = (json.dumps(saved, ensure_ascii=False) + "\n").encode("utf-8")
                fd = os.open(str(directory.parent / "captures.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    # Advisory locking keeps complete records together across workers.
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    with os.fdopen(fd, "ab", closefd=False) as handle:
                        handle.write(line)
                finally:
                    os.close(fd)
            except Exception:
                LOG.exception("[capture] finalization failed rid=%s", rid)
            finally:
                body.close()
                response.close()
                capture_context.reset(context_token)


def capture_app(app, service):
    root = os.environ.get("SPEECH_CAPTURE_DIR")
    return SpeechCaptureMiddleware(app, root, service) if root else app
