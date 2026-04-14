"""Serve the local demo page for screen recording.

This script serves the repository root so the demo can fetch existing artifacts under
`outputs/eval_pack/...` without copying data.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import socket
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Slightly quieter static-file handler for local demo serving."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Keep local terminal output clean for recording setup.
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the self-calibrating-spatiallm demo page")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to serve (default: auto-detected repo root)",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Open URL with autoplay enabled",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset id passed to the demo URL (for autoplay or quick selection)",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=None,
        help="Autoplay step duration in seconds",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Enable autoplay loop via URL parameter",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open browser automatically",
    )
    return parser


def _build_demo_url(args: argparse.Namespace) -> str:
    query: dict[str, str] = {}
    if args.autoplay:
        query["autoplay"] = "1"
    if args.preset:
        query["preset"] = str(args.preset)
    if args.step_seconds is not None:
        query["stepSec"] = str(max(2, int(args.step_seconds)))
    if args.loop:
        query["loop"] = "1"

    query_string = urllib.parse.urlencode(query)
    base = f"http://{args.host}:{args.port}/demo/index.html"
    return f"{base}?{query_string}" if query_string else base


def _port_in_use(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    args = _build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"serve root does not exist: {root}")

    if _port_in_use(args.host, args.port):
        raise RuntimeError(
            f"Port {args.port} is already in use on {args.host}. "
            "Choose another port with --port."
        )

    handler_class = lambda *handler_args, **handler_kwargs: QuietHandler(  # noqa: E731
        *handler_args,
        directory=str(root),
        **handler_kwargs,
    )

    with socketserver.TCPServer((args.host, args.port), handler_class) as httpd:
        demo_url = _build_demo_url(args)
        print(f"Serving demo from: {root}")
        print(f"Demo URL: {demo_url}")
        print("Press Ctrl+C to stop.")

        if args.open_browser:
            webbrowser.open(demo_url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDemo server stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
