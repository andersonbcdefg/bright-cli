#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional

import aiohttp
import pandas as pd
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


def _load_urls(file_path: str, field: str | None = None) -> List[str]:
    """Load URLs from a txt, csv, or jsonl file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".csv":
        if not field:
            raise ValueError("--field is required for CSV files")
        df = pd.read_csv(path)
        return df[field].dropna().astype(str).tolist()
    elif suffix in (".jsonl", ".ndjson"):
        if not field:
            raise ValueError("--field is required for JSONL files")
        df = pd.read_json(path, lines=True)
        return df[field].dropna().astype(str).tolist()
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .csv, or .jsonl")


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


def _load_completed_urls(output_path: str) -> set[str]:
    """Load URLs that have already been successfully scraped from output file."""
    completed = set()
    path = Path(output_path)
    if not path.exists():
        return completed

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            # Only count as done if it succeeded (no error)
            if record.get("error") is None:
                completed.add(record["url"])
        except json.JSONDecodeError:
            continue
    return completed


async def _cli_batch(args: argparse.Namespace) -> int:
    api_key = args.api_key
    zone = args.zone
    endpoint = args.endpoint
    data_format = args.format
    verbose = args.verbose
    concurrency = args.concurrency
    max_retries = 3

    if not api_key:
        print("Missing API key. Run 'bright init' or pass --api-key.", file=sys.stderr)
        return 2
    if not zone:
        print("Missing zone. Run 'bright init' or pass --zone.", file=sys.stderr)
        return 2

    try:
        urls = _load_urls(args.input, args.field)
    except ValueError as e:
        print(f"Error loading URLs: {e}", file=sys.stderr)
        return 2

    if not urls:
        print("No URLs found in input file.", file=sys.stderr)
        return 2

    # Check for already completed URLs
    already_done = _load_completed_urls(args.output)
    urls_to_scrape = [u for u in urls if u not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} already done, {len(urls_to_scrape)} remaining", file=sys.stderr)
    else:
        print(f"Loaded {len(urls_to_scrape)} URLs", file=sys.stderr)

    if not urls_to_scrape:
        print("All URLs already scraped.", file=sys.stderr)
        return 0

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(urls_to_scrape)

    async def scrape_one(client: BrightUnlocker, url: str) -> dict:
        nonlocal completed
        async with semaphore:
            last_error = None
            for attempt in range(max_retries):
                try:
                    resp = await client.fetch(url, data_format=data_format)
                    content = resp.text() if data_format != "screenshot" else resp.body.hex()
                    result = {"url": url, "content": content, "error": None}
                    break
                except BrightDataError as e:
                    last_error = f"HTTP {e.status}: {e.error_detail or e.error_code or 'unknown'}"
                    if verbose:
                        print(f"\nRetry {attempt + 1}/{max_retries} for {url}: {last_error}", file=sys.stderr)
                except Exception as e:
                    last_error = str(e) or type(e).__name__
                    if verbose:
                        print(f"\nRetry {attempt + 1}/{max_retries} for {url}: {last_error}", file=sys.stderr)
            else:
                # All retries exhausted
                result = {"url": url, "content": None, "error": last_error}
                if verbose:
                    print(f"\nFailed after {max_retries} attempts: {url}", file=sys.stderr)

            completed += 1
            print(f"\rProgress: {completed}/{total}", end="", file=sys.stderr)
            return result

    # Append mode for resumability
    with open(args.output, "a") as f:
        async with BrightUnlocker(
            api_key=api_key, zone=zone, endpoint=endpoint, timeout_s=args.timeout
        ) as client:
            tasks = [scrape_one(client, url) for url in urls_to_scrape]
            results = await asyncio.gather(*tasks)

            for result in results:
                f.write(json.dumps(result) + "\n")

    print(file=sys.stderr)  # newline after progress
    print(f"Results written to {args.output}", file=sys.stderr)
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

    # batch subcommand
    b = sub.add_parser("batch", help="Batch scrape URLs from a file")
    b.add_argument("input", help="Input file (.txt, .csv, or .jsonl)")
    b.add_argument("output", help="Output JSONL file")
    b.add_argument(
        "--field",
        default=None,
        help="Field/column name containing URLs (required for csv/jsonl)",
    )
    b.add_argument(
        "--format",
        choices=["raw", "markdown", "screenshot"],
        default="raw",
        help="Output format: raw (HTML), markdown, or screenshot (PNG)",
    )
    b.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
    )
    b.add_argument(
        "--zone",
        default=os.getenv("BRIGHT_ZONE"),
        help="Unlocker zone name (or set via 'bright init')",
    )
    b.add_argument(
        "--api-key",
        default=os.getenv("BRIGHT_API_KEY"),
        help="Bright Data API key (or set via 'bright init')",
    )
    b.add_argument(
        "--endpoint",
        default=os.getenv("BRIGHT_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"API endpoint (default: {DEFAULT_ENDPOINT})",
    )
    b.add_argument(
        "--timeout", type=float, default=120.0, help="Request timeout in seconds"
    )
    b.add_argument(
        "-v", "--verbose", action="store_true", help="Print errors to stderr"
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

    if args.cmd == "batch":
        return asyncio.run(_cli_batch(args))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
