#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, Optional

import aiohttp
from dotenv import load_dotenv

DEFAULT_ENDPOINT = "https://api.brightdata.com/request"
CONFIG_DIR = Path.home() / ".bright-unlocker"
CONFIG_FILE = CONFIG_DIR / ".env"


def _load_config() -> None:
    """Load config from ~/.bright-unlocker/.env if it exists."""
    if CONFIG_FILE.exists():
        load_dotenv(CONFIG_FILE)


def _init_config(api_key: str, zone: str | None = None) -> None:
    """Initialize config file with API key and optional zone."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = [f"BRIGHT_API_KEY={api_key}"]
    if zone:
        lines.append(f"BRIGHT_ZONE={zone}")

    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    CONFIG_FILE.chmod(0o600)  # Restrict permissions since it contains secrets
    print(f"Config saved to {CONFIG_FILE}")


CHUNK_SIZE = 64 * 1024


class BrightDataError(RuntimeError):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
        request_id: Optional[str] = None,
        response_snippet: Optional[bytes] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.error_detail = error_detail
        self.request_id = request_id
        self.response_snippet = response_snippet


@dataclass(frozen=True)
class BrightResponse:
    status: int
    headers: Dict[str, str]
    body: bytes

    def text(self, encoding: Optional[str] = None) -> str:
        # Best-effort decode. For HTML/markdown this is usually fine.
        enc = encoding or "utf-8"
        return self.body.decode(enc, errors="replace")


def _build_payload(url: str, zone: str, data_format: str) -> Dict[str, Any]:
    # Bright Data examples keep "format":"raw" and use "data_format" for markdown/screenshot.
    payload: Dict[str, Any] = {
        "zone": zone,
        "url": url,
        "format": "raw",
    }
    if data_format == "markdown":
        payload["data_format"] = "markdown"
    elif data_format == "screenshot":
        payload["data_format"] = "screenshot"
    # data_format == "raw" => omit
    return payload


def _pick_bright_headers(headers: aiohttp.typedefs.LooseHeaders) -> Dict[str, str]:
    # Normalize to simple dict[str,str]
    out: Dict[str, str] = {}
    for k, v in dict(headers).items():
        try:
            out[str(k)] = str(v)
        except Exception:
            pass
    return out


def _extract_error_headers(
    h: Dict[str, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    # BrightData may use either x-brd-* or older x-luminati-*.
    code = h.get("x-brd-error-code") or h.get("x-luminati-error-code")
    msg = h.get("x-brd-error") or h.get("x-luminati-error")
    req_id = None
    dbg = h.get("x-brd-debug")
    if dbg:
        # dbg looks like: "req_id=...; bytes_up=...; ..."
        parts = [p.strip() for p in dbg.split(";")]
        for p in parts:
            if p.startswith("req_id="):
                req_id = p.split("=", 1)[1].strip()
                break
    return code, msg, req_id


class BrightUnlocker:
    """
    Thin async wrapper around Bright Data Web Unlocker HTTP API.

    Usage:
        async with BrightUnlocker(api_key=..., zone=...) as c:
            resp = await c.fetch("https://example.com", data_format="markdown")
            print(resp.text())
    """

    def __init__(
        self,
        *,
        api_key: str,
        zone: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = 120.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.api_key = api_key
        self.zone = zone
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = session

    async def __aenter__(self) -> "BrightUnlocker":
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None and self._external_session is None:
            await self._session.close()
            self._session = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def fetch(self, url: str, *, data_format: str = "raw") -> BrightResponse:
        """
        Fetch and return the full response body as bytes.

        data_format: "raw" | "markdown" | "screenshot"
        """
        if self._session is None:
            raise RuntimeError(
                "BrightUnlocker must be used in an 'async with' block, or provide a session=..."
            )

        payload = _build_payload(url=url, zone=self.zone, data_format=data_format)

        async with self._session.post(
            self.endpoint, headers=self._headers(), json=payload
        ) as resp:
            headers = _pick_bright_headers(resp.headers)

            if resp.status >= 400:
                snippet = await resp.content.read(2000)
                code, msg, req_id = _extract_error_headers(headers)
                raise BrightDataError(
                    resp.status,
                    f"Bright Data request failed (HTTP {resp.status})",
                    error_code=code,
                    error_detail=msg,
                    request_id=req_id,
                    response_snippet=snippet,
                )

            body = await resp.read()
            return BrightResponse(status=resp.status, headers=headers, body=body)

    async def stream(
        self, url: str, *, data_format: str = "raw"
    ) -> AsyncIterator[bytes]:
        """
        Stream the response body in chunks (useful for large outputs).
        """
        if self._session is None:
            raise RuntimeError(
                "BrightUnlocker must be used in an 'async with' block, or provide a session=..."
            )

        payload = _build_payload(url=url, zone=self.zone, data_format=data_format)

        async with self._session.post(
            self.endpoint, headers=self._headers(), json=payload
        ) as resp:
            headers = _pick_bright_headers(resp.headers)

            if resp.status >= 400:
                snippet = await resp.content.read(2000)
                code, msg, req_id = _extract_error_headers(headers)
                raise BrightDataError(
                    resp.status,
                    f"Bright Data request failed (HTTP {resp.status})",
                    error_code=code,
                    error_detail=msg,
                    request_id=req_id,
                    response_snippet=snippet,
                )

            async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                yield chunk


async def _cli_scrape(args: argparse.Namespace) -> int:
    api_key = args.api_key
    zone = args.zone
    endpoint = args.endpoint
    data_format = args.format
    url = args.url
    out = args.out
    verbose = args.verbose

    if not api_key:
        print(
            "Missing API key. Set BRIGHT_API_KEY in .env or pass --api-key.",
            file=sys.stderr,
        )
        return 2
    if not zone:
        print("Missing zone. Set BRIGHT_ZONE in .env or pass --zone.", file=sys.stderr)
        return 2

    # Choose sink
    sink: Optional[BinaryIO]
    close_sink = False
    if out is None or out == "-":
        sink = sys.stdout.buffer
    else:
        sink = open(out, "wb")
        close_sink = True

    try:
        async with BrightUnlocker(
            api_key=api_key, zone=zone, endpoint=endpoint, timeout_s=args.timeout
        ) as client:
            try:
                async for chunk in client.stream(url, data_format=data_format):
                    sink.write(chunk)
            except BrightDataError as e:
                print(f"Bright Data error: HTTP {e.status}", file=sys.stderr)
                if e.error_code or e.error_detail:
                    print(f"  code: {e.error_code or '(none)'}", file=sys.stderr)
                    print(f"  message: {e.error_detail or '(none)'}", file=sys.stderr)
                if e.request_id:
                    print(f"  req_id: {e.request_id}", file=sys.stderr)
                if verbose and e.response_snippet:
                    print("  response body (first 2000 bytes):", file=sys.stderr)
                    try:
                        # Best effort: print bytes as-is (may be binary)
                        sys.stderr.buffer.write(e.response_snippet + b"\n")
                    except Exception:
                        pass
                return 1
    finally:
        if close_sink and sink is not None:
            sink.close()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    _load_config()

    p = argparse.ArgumentParser(
        prog="bright",
        description="Thin wrapper around Bright Data Web Unlocker HTTP API.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # init subcommand
    init_p = sub.add_parser("init", help="Initialize config with API key and zone")
    init_p.add_argument("api_key", help="Bright Data API key")
    init_p.add_argument("--zone", default=None, help="Default zone name (optional)")

    # scrape subcommand
    s = sub.add_parser("scrape", help="Scrape a URL through Bright Data Unlocker API")
    s.add_argument("url", help="Target URL to fetch")
    s.add_argument(
        "--format",
        choices=["raw", "markdown", "screenshot"],
        default="raw",
        help="Output format: raw (HTML), markdown, or screenshot (PNG)",
    )
    s.add_argument(
        "--out", default=None, help="Output file path. Use '-' or omit for stdout."
    )
    s.add_argument(
        "--zone",
        default=os.getenv("BRIGHT_ZONE"),
        help="Unlocker zone name (or set via 'bright init')",
    )
    s.add_argument(
        "--api-key",
        default=os.getenv("BRIGHT_API_KEY"),
        help="Bright Data API key (or set via 'bright init')",
    )
    s.add_argument(
        "--endpoint",
        default=os.getenv("BRIGHT_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"API endpoint (default: {DEFAULT_ENDPOINT})",
    )
    s.add_argument(
        "--timeout", type=float, default=120.0, help="Request timeout in seconds"
    )
    s.add_argument(
        "-v", "--verbose", action="store_true", help="Print more debug info to stderr"
    )

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "init":
        _init_config(args.api_key, args.zone)
        return 0

    if args.cmd == "scrape":
        return asyncio.run(_cli_scrape(args))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
