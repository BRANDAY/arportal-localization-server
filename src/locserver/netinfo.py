"""Small network helpers for the start-up banner and the doctor. Standard library only."""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request

from . import API_VERSION, SERVICE_ID


def filter_lan(addresses) -> list[str]:
    """Usable LAN IPv4 addresses: no loopback, no link-local; private ranges first."""
    out = []
    for a in set(addresses):
        try:
            ip = ipaddress.IPv4Address(a)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            continue
        out.append(ip)
    # 192.168.x.x (home Wi-Fi) first, then other private ranges, then the rest
    return [str(ip) for ip in sorted(out, key=lambda ip: (not str(ip).startswith("192.168."), not ip.is_private, int(ip)))]


def lan_ipv4_addresses() -> list[str]:
    """IPv4 addresses of this machine's network interfaces (best effort, no dependencies)."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # the interface of the default route; connecting a UDP socket sends nothing
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return filter_lan(found)


def banner_lines(host: str, port: int, addresses: list[str], server_name: str) -> list[str]:
    """What the server prints at start-up so the user knows what to type into the app."""
    lines = ["", f"AR Portal localization server \"{server_name}\" - service {SERVICE_ID}, API v{API_VERSION}"]
    if host in ("127.0.0.1", "localhost", "::1"):
        lines += [
            f"  listening on http://127.0.0.1:{port}  (this PC only)",
            "  A phone cannot connect: start the server with --host 0.0.0.0 to accept LAN connections.",
        ]
        return lines + [""]
    urls = [f"http://{a}:{port}" for a in (addresses if host in ("0.0.0.0", "::", "") else [host])]
    if not urls:
        lines += [
            "  No LAN address found: this PC is not connected to a network.",
            "  Connect it to the same Wi-Fi as the phone and restart the server.",
        ]
        return lines + [""]
    lines.append("  Reachable on the local network at:")
    lines += [f"    {u}" for u in urls]
    lines.append(
        f"  In the app: type {'that address' if len(urls) == 1 else 'the address that is in the same network as the phone'}"
        ' into the server field, or tap "Find automatically".'
    )
    lines.append(f"  Check from the phone's browser: {urls[0]}/health   (no answer? run: uv run locserver-doctor)")
    return lines + [""]


def ready_lines(host: str, port: int, addresses: list[str], server_name: str, version: str = "") -> list[str]:
    """The block printed once the server really answers: that it runs, what to type into the phone, how to stop it."""
    bar = "=" * 76
    loop = host in ("127.0.0.1", "localhost", "::1")
    urls = [] if loop else [f"http://{a}:{port}" for a in (addresses if host in ("0.0.0.0", "::", "") else [host])]
    lines = ["", bar, "", f"   SERVER IS RUNNING      (\"{server_name}\"{', version ' + version if version else ''})", ""]
    if urls:
        lines.append("   ADDRESS FOR THE PHONE:")
        lines += [f"       {u}" for u in urls]
        if len(urls) > 1:
            lines.append("     (several networks: use the address that is in the same network as the phone)")
        lines += ['     The app finds it by itself ("Find automatically"); type it in if it does not.',
                  f"     Test from the phone's browser: {urls[0]}/health   (no answer? run: uv run locserver-doctor)"]
    elif loop:
        lines += ["   THIS PC ONLY - a phone cannot connect (started with --host 127.0.0.1)."]
    else:
        lines += ["   NO NETWORK: this PC has no address in a local network.",
                  "     Connect it to the same Wi-Fi as the phone and start the server again."]
    lines += ["", f"   Panel on this PC:  http://127.0.0.1:{port}/ui", "",
              "   TO STOP THE SERVER: close this window (or press Ctrl+C).", "", bar, ""]
    return lines


def port_free(host: str, port: int) -> bool:
    """Can a server listen on ``host:port`` right now?"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host if host not in ("", "::") else "0.0.0.0", port))
            return True
    except OSError:
        return False


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_health(host: str, port: int, timeout: float = 2.0) -> dict | None:
    """``GET /health`` of whatever listens there; None if it is not JSON over HTTP."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            data = json.loads(r.read(65536).decode("utf-8", "replace"))
            return data if isinstance(data, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def is_our_server(health: dict | None) -> bool:
    return bool(health) and health.get("service") == SERVICE_ID
