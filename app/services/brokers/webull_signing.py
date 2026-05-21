"""Webull OpenAPI signing helpers — shared by market-data and trading paths.

The signing recipe matches the official webull-inc/openapi-python-sdk
Python SDK (sha_hmac1.py + default_signature_composer.py).

Recipe:
  1. Build sign_headers: x-app-key, x-timestamp (ISO-8601 UTC), x-signature-version,
     x-signature-algorithm=HMAC-SHA1, x-signature-nonce (UUID5), Host.
  2. Lowercase header keys, merge with query params; sort keys; join k=v with '&'.
  3. Prefix with the request URI ("/path") + '&'.
  4. For POST/PUT with body, append uppercase md5-hex of
     json.dumps(body, ensure_ascii=False, separators=(',',':')) to string_to_sign.
  5. URL-encode the whole string_to_sign with quote(safe='').
  6. HMAC-SHA1(encoded_string_to_sign, secret=app_secret + "&"); base64-encode.
  7. Send signature in the x-signature header.

This module owns the HMAC bits only — endpoint URLs and host-specific behavior
live in the caller (webull_md.py for market data, webull.py for trading).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import socket
import uuid
from datetime import datetime, timezone
from typing import Mapping
from urllib.parse import quote

SIGNATURE_VERSION = "1.0"
SIGNATURE_ALGORITHM = "HMAC-SHA1"
API_VERSION = "v1"


def webull_uuid() -> str:
    """UUID5 nonce — matches the SDK's get_uuid() helper."""
    name = socket.gethostname() + str(uuid.uuid1())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def iso8601_utc_now() -> str:
    """ISO-8601 UTC timestamp without microseconds (matches SDK FORMAT_ISO_8601)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_signed_headers(
    *,
    app_key: str,
    app_secret: str,
    host: str,
    uri: str,
    query: Mapping[str, str] | None = None,
    body: object | None = None,
) -> dict[str, str]:
    """Return wire headers (including x-signature) for a Webull OpenAPI request.

    For POST/PUT, pass the request body (dict/list) — it will be JSON-serialized
    with the same compact format the server expects and appended to the string
    that gets signed.

    Returns headers ready to attach to a requests/httpx call. The signature is
    bound to ``host`` and ``uri``, so changing the path means re-signing.
    """
    query = dict(query or {})
    sign_headers = {
        "x-app-key": app_key,
        "x-timestamp": iso8601_utc_now(),
        "x-signature-version": SIGNATURE_VERSION,
        "x-signature-algorithm": SIGNATURE_ALGORITHM,
        "x-signature-nonce": webull_uuid(),
        "Host": host,
    }

    # Merge lowercased headers with query; collisions concatenate with '&'.
    merged: dict[str, str] = {}
    for k, v in sign_headers.items():
        merged[k.lower()] = v
    for k, v in query.items():
        existing = merged.get(k)
        merged[k] = (str(existing) + "&" + str(v)) if existing is not None else str(v)

    sorted_kv = "&".join(f"{k}={merged[k]}" for k in sorted(merged.keys()))
    string_to_sign = uri + "&" + sorted_kv

    if body is not None:
        # SDK recipe (default_signature_composer._get_body_string):
        #   md5(json_dumps_compact(body)).hexdigest().upper()
        # json_dumps_compact = json.dumps(content, ensure_ascii=False, separators=(',',':'))
        body_json = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        body_digest = hashlib.md5(body_json.encode("utf-8")).hexdigest().upper()
        string_to_sign = string_to_sign + "&" + body_digest

    encoded = quote(string_to_sign, safe="")
    digest = hmac.new(
        (app_secret + "&").encode("utf-8"),
        encoded.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.standard_b64encode(digest).decode("ascii").strip()

    # Drop synthetic Host (the HTTP client sets it); add x-version + x-signature.
    wire = {k: v for k, v in sign_headers.items() if k != "Host"}
    wire["x-version"] = API_VERSION
    wire["x-signature"] = signature
    return wire


def json_body_bytes(body: object) -> bytes:
    """Serialize a body exactly the way build_signed_headers hashes it.

    Use this for the actual HTTP request body so the wire bytes match the
    bytes whose md5-hex was appended to the string-to-sign. Mixing dumps
    options across signer and sender desyncs the signature.
    """
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
