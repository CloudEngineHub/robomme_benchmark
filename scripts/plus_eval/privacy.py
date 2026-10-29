"""Redaction at artifact boundaries; never modify live model inputs or outputs.

This module uses only the standard library, including for filtering subprocess
logs: ``python privacy.py filter < server-output``. It does not inspect images.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
from urllib.parse import quote, quote_plus, unquote, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
LOCAL_PATH = "[LOCAL_PATH]"
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".csv", ".md", ".yaml", ".yml"}
_PRIVATE_FIELDS = {
    "authorization", "proxyauthorization", "apikey", "accesstoken", "refreshtoken",
    "idtoken", "bearertoken", "token", "password", "passwd", "secret", "clientsecret",
    "secretkey", "privatekey", "credentials", "credential", "cookie", "setcookie",
    "sessiontoken", "awsaccesskeyid", "awssecretaccesskey", "username", "userid", "user",
    "email", "hostname", "machineid", "organizationid", "projectid",
}
_SECRET_NAME = r"(?:[\w.-]+[-_])?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|auth[-_]?token|client[-_]?secret|secret[-_]?key|private[-_]?key|password|passwd|secret|token|authorization|cookie|set-cookie)"
_ASSIGNMENT = re.compile(r"(?i)(\b" + _SECRET_NAME + r"[\"']?\s*[:=]\s*)((?:Bearer|Basic)\s+[^\s,;\"']+|\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)")
_URL = re.compile(r"\b(?:https?|wss?|s3|gs|ssh)://[^\s<>\"']+")
# Rooted filesystem paths (including Windows/UNC), excluding URL route paths.
_PATH = re.compile(r"(?<![\w/.\]-])(?:/(?:home|Users|nfs|mnt|media|root|tmp|private|opt|usr|var|workspace|workspaces|data|srv|etc|scratch|shared|lustre|gpfs|net)(?:/[^\s\"'<>\[\]{},;:()]+)*|[A-Za-z]:[\\/][^\s\"'<>\[\]{},;()]+|\\\\[\w.-]+\\[^\s\"'<>]+|~/[^\s\"'<>]+)")
# Additional mount points; /api and /files are the viewer's portable routes.
_OTHER_PATH = re.compile(r"(?<![\w/:.\]-])/(?!api(?:/|\b)|files(?:/|\b))[A-Za-z0-9_.-]+(?:/[^\s\"'<>\[\]{},;:()]+)+")
_SECRETS = ()
_HOSTS = ()


def sensitive_key(key):
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return normalized in _PRIVATE_FIELDS or normalized.endswith(("apikey", "password", "accesstoken", "refreshtoken", "clientsecret", "secretaccesskey", "sessiontoken", "authtoken", "apitoken", "hftoken", "githubtoken", "huggingfacetoken"))


def refresh_secrets():
    """Snapshot credential environment variables without recording their values."""
    global _SECRETS, _HOSTS
    values = {v for k, v in os.environ.items() if sensitive_key(k) and len(v) >= 4}
    _SECRETS = tuple(sorted({form for value in values for form in
                            (value, quote(value, safe=""), quote_plus(value), json.dumps(value)[1:-1])}, key=len, reverse=True))
    _HOSTS = tuple(h for h in {socket.gethostname(), socket.getfqdn()} if h not in {"localhost", ""} and len(h) > 3)


def _url(match):
    text = match.group()
    try:
        parts = urlsplit(text)
        host = parts.netloc.rsplit("@", 1)[-1]
        # Keep query spelling/order, but omit credential and signed URL values.
        query = re.sub(r"([^&=]+)=([^&]*)", lambda m: m[1] + "=" +
                       (REDACTED if sensitive_key(unquote(m[1])) or any(s in unquote(m[1]).lower() for s in ("signature", "credential", "security-token")) else m[2]), parts.query)
        return urlunsplit((parts.scheme, host, parts.path, query, parts.fragment))
    except ValueError:
        return REDACTED


def scrub(value):
    text = str(value)
    # Protect embedded media from text patterns; these are generated camera data.
    if text.startswith("data:image/") and ";base64," in text:
        return text
    text = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|\Z)", REDACTED, text, flags=re.S)
    text = re.sub(r"\bfile://[^\s<>\"']+", LOCAL_PATH, text)
    # URLs must be protected before filesystem matching; public source links stay useful.
    urls = []
    def protect(match):
        urls.append(_url(match))
        return f"\x00URL{len(urls)-1}\x00"
    text = _URL.sub(protect, text)
    text = _PATH.sub(LOCAL_PATH, text)
    text = _OTHER_PATH.sub(LOCAL_PATH, text)
    text = re.sub(r"(?im)(\b(?:cookie|set-cookie)\s*:\s*)[^\r\n]+", lambda m: m[1] + REDACTED, text)
    text = _ASSIGNMENT.sub(lambda m: m[1] + '"' + REDACTED + '"', text)
    for i, url in enumerate(urls):
        text = text.replace(f"\x00URL{i}\x00", url)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]+|AKIA[A-Z0-9]{16})", REDACTED, text)
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9+/._~=-]+", lambda m: m[1] + " " + REDACTED, text)
    for host in _HOSTS:
        text = re.sub(r"(?<![\w.-])" + re.escape(host) + r"(?![\w.-])", "local-host.invalid", text)
    text = re.sub(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b", "redacted-host.invalid", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", lambda m: m[0] if m[0] == "git@github.com" else "[EMAIL]", text)
    for secret in _SECRETS:
        text = text.replace(secret, REDACTED)
    return text


def safe_items(value):
    """Redact mapping keys too, without losing entries when paths become equal."""
    used = set()
    for key, item in value.items():
        name = scrub(key)
        base, index = name, 1
        while name in used:
            index += 1
            name = f"{base}#{index}"
        used.add(name)
        yield name, REDACTED if sensitive_key(key) else item


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in safe_items(value)}
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    if isinstance(value, (str, Path)):
        return scrub(value)
    return value


def sanitize_text(text, suffix=".txt"):
    """Parse structured artifacts so arbitrary credential values are removed."""
    if suffix == ".json":
        return json.dumps(sanitize(json.loads(text)), indent=2, ensure_ascii=False) + "\n"
    if suffix == ".jsonl":
        return "".join(json.dumps(sanitize(json.loads(line)), ensure_ascii=False) + "\n"
                       for line in text.splitlines() if line.strip())
    return scrub(text)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text), encoding="utf-8")


class RedactingWriter(io.TextIOBase):
    """Buffer complete lines, so split writes/flushes cannot disclose a key.

    Unterminated lines are emitted at finish(), not flush(). Oversized lines are
    omitted to bound memory. PEM blocks are suppressed through their end marker.
    """
    def __init__(self, *streams):
        self.streams, self.pending, self.in_key, self.dropping = streams, "", False, False
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self._line(line + "\n")
            if len(self.pending) > 1024 * 1024:
                self.pending = ""
                self.dropping = True
        return len(text)

    def _line(self, line):
        if self.dropping:
            line, self.dropping = "[OVERSIZED LOG LINE OMITTED]\n", False
        if "-----BEGIN " in line and "PRIVATE KEY-----" in line:
            self.in_key = True
            for stream in self.streams: stream.write(REDACTED + "\n")
        if self.in_key:
            if "-----END " in line and "PRIVATE KEY-----" in line: self.in_key = False
            return
        for stream in self.streams:
            stream.write(scrub(line))
            stream.flush()

    def flush(self):
        for stream in self.streams:
            if not stream.closed: stream.flush()

    def finish(self):
        with self.lock:
            if self.pending or self.dropping:
                self._line(self.pending)
                self.pending = ""
            self.flush()


def sanitize_tree(root, apply=False):
    """Audit/migrate text artifacts only; never follow symlinks or change media."""
    root = Path(root)
    checked = changed = 0
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = [c for c in children if not (Path(directory) / c).is_symlink() and not c.startswith('.')]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or path.suffix not in TEXT_SUFFIXES:
                continue
            original = path.read_text(encoding="utf-8")
            clean = sanitize_text(original, path.suffix)
            checked += 1
            # Formatting alone isn't a privacy finding.
            different = (sanitize(json.loads(original)) != json.loads(original) if path.suffix == ".json"
                         else any(sanitize(json.loads(line)) != json.loads(line) for line in original.splitlines() if line.strip()) if path.suffix == ".jsonl"
                         else clean != original)
            if not different:
                continue
            changed += 1
            if apply:
                temporary = path.with_name(path.name + ".redacting")
                temporary.write_text(clean, encoding="utf-8")
                temporary.replace(path)
    return {"text_files_checked": checked, "files_changed" if apply else "files_needing_redaction": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("filter", help="Redact subprocess stdout/stderr before saving it")
    audit = sub.add_parser("audit", help="Check existing result text without displaying private values")
    audit.add_argument("root", type=Path)
    audit.add_argument("--apply", action="store_true", help="Sanitize text artifacts in place")
    args = parser.parse_args()
    if args.command == "filter":
        writer = RedactingWriter(sys.stdout)
        try:
            for line in sys.stdin: writer.write(line)
        finally:
            writer.finish()
    else:
        report = sanitize_tree(args.root, apply=args.apply)
        print(json.dumps(report))
        if not args.apply and report["files_needing_redaction"]: raise SystemExit(1)


refresh_secrets()

if __name__ == "__main__":
    main()
