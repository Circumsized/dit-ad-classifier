"""Safe retrieval of public data files.

AI4AD data is distributed outside this repository.  When this CLI downloads a
file it performs a server-side request, so the request target is validated
before any socket is opened and re-validated on every redirect hop.

Policy:
    * only ``http`` and ``https`` schemes;
    * the host is resolved before the request and **every** resolved address
      must be a public global address — loopback, private, link-local,
      reserved, benchmarking, multicast and unspecified ranges are refused;
    * ``localhost`` and other non-routable hostnames are refused before DNS;
    * redirects to an unsafe target are not followed.

The resolver is injectable so tests can assert the policy without touching the
network.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from dit import __version__

OFFICIAL_DATA_URL = "https://github.com/YongLiuLab/AI4AD_AFQ"
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = {80, 443}
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = 30.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB safety ceiling

# Hostnames that never point at a public address, regardless of DNS answer.
_NON_ROUTABLE_NAMES = {"localhost", "local", "broadcasthost"}
_NON_ROUTABLE_SUFFIXES = (".local", ".internal", ".home.arpa", ".corp", ".lan")

# ``ipaddress.is_global`` is True for multicast and for the IETF special-use
# ranges it does not track, so the flags alone would let 224.0.0.1, 239.255.
# 255.250 and 192.88.99.1 through.  These networks are blocked explicitly.
_REFUSED_NETWORKS = tuple(
    ipaddress.ip_network(item)
    for item in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "233.252.0.0/24",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "::/48",
        "::/96",
        "100::/64",
        "2001:10::/28",
        "2001:20::/28",
        "2001:db8::/32",
        "2002::/16",
        "64:ff9b::/96",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)


class UnsafeURL(ValueError):
    """Raised when a request target violates the fetch policy."""


class FetchError(RuntimeError):
    """Raised when a validated request fails to complete."""


@dataclass(frozen=True)
class FetchTarget:
    """A validated request target."""

    scheme: str
    host: str
    port: int
    path: str
    url: str


Resolver = Callable[[str], list[ipaddress._BaseAddress]]


def _split_target(url: str) -> FetchTarget:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise UnsafeURL(
            f"malformed bracketed hostname in {url!r}: {exc}"
            if "[" in url
            else f"malformed url {url!r}: {exc}"
        ) from exc
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURL(
            f"scheme {scheme!r} is not allowed; only {', '.join(ALLOWED_SCHEMES)}"
        )
    netloc = parsed.netloc
    if "@" in netloc:
        raise UnsafeURL("credentials in the URL are not allowed")

    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        # urlsplit returns no hostname for an unbracketed IPv6 literal, so this
        # branch has to tell "no host" from "host written without brackets".
        if ":" in netloc and "[" not in netloc:
            raise UnsafeURL(f"IPv6 literals must be bracketed: {url!r}")
        raise UnsafeURL(f"url has no hostname: {url!r}")
    if "[" not in netloc and netloc.count(":") >= 2:
        raise UnsafeURL(f"IPv6 literals must be bracketed: {url!r}")
    if netloc.count("[") + netloc.count("]") and not (
        netloc.count("[") == 1
        and netloc.count("]") == 1
        and netloc.startswith("[")
        and netloc.index("]") > netloc.index("[")
    ):
        raise UnsafeURL(f"malformed bracketed hostname: {netloc!r}")

    if host in _NON_ROUTABLE_NAMES or host.endswith(_NON_ROUTABLE_SUFFIXES):
        raise UnsafeURL(f"host {host!r} is not routable")
    if _is_ip_literal(host):
        if not re.fullmatch(r"[0-9a-f:.]+", host):
            raise UnsafeURL(f"malformed address literal {host!r}")
    elif not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*", host):
        raise UnsafeURL(f"host {host!r} is not a valid DNS name")

    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURL(f"invalid port in {url!r}") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed; use {sorted(ALLOWED_PORTS)}")

    return FetchTarget(scheme, host, port, parsed.path or "/", url)


def resolve_host(host: str, resolver: Resolver | None = None) -> list[ipaddress._BaseAddress]:
    """Resolve ``host`` to addresses, refusing when nothing public is returned."""

    if resolver is None:
        resolver = _system_resolver
    try:
        infos = resolver(host)
    except (socket.gaierror, OSError, socket.herror) as exc:
        raise UnsafeURL(f"host {host!r} could not be resolved: {exc}") from exc
    if not infos:
        raise UnsafeURL(f"host {host!r} resolved to no addresses")
    return infos


def validate_url(url: str, *, resolver: Resolver | None = None) -> FetchTarget:
    """Validate ``url`` against the fetch policy and return the target.

    The host is resolved as part of validation so the check covers DNS
    answers, not just literal addresses.
    """

    target = _split_target(url)
    if _is_ip_literal(target.host):
        addresses = [ipaddress.ip_address(target.host)]
    else:
        addresses = resolve_host(target.host, resolver=resolver)
    for address in addresses:
        if not _is_global(address):
            raise UnsafeURL(
                f"host {target.host!r} resolves to non-public address {address}"
            )
    return target


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _is_global(address: ipaddress._BaseAddress) -> bool:
    """True only for public, globally routable addresses.

    Combines the ``ipaddress`` flags with an explicit special-use range list.
    ``is_global`` alone is not sufficient: it reports multicast and several
    IETF reserved ranges as global, which would allow a request to
    224.0.0.1 or 192.88.99.1.
    """

    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return _is_global(address.ipv4_mapped)
    if not address.is_global:
        return False
    for flag in ("is_multicast", "is_reserved", "is_unspecified", "is_loopback",
                 "is_private", "is_link_local"):
        if bool(getattr(address, flag)):
            return False
    for network in _REFUSED_NETWORKS:
        if address.version == network.version and address in network:
            return False
    return True


def _system_resolver(host: str) -> list[ipaddress._BaseAddress]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    addresses: list[ipaddress._BaseAddress] = []
    seen: set[ipaddress._BaseAddress] = set()
    for family, _, _, _, sockaddr in infos:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if address not in seen:
            seen.add(address)
            addresses.append(address)
    return addresses


def safe_request(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_RESPONSE_BYTES,
    resolver: Resolver | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[bytes, FetchTarget]:
    """Fetch ``url`` after validating the target and every redirect hop."""

    import http.client
    import ssl

    current = url
    for hop in range(MAX_REDIRECTS + 1):
        target = validate_url(current, resolver=resolver)
        try:
            connection = (
                http.client.HTTPSConnection(
                    target.host,
                    target.port,
                    timeout=timeout,
                    context=ssl.create_default_context(),
                    blocksize=65536,
                )
                if target.scheme == "https"
                else http.client.HTTPConnection(
                    target.host, target.port, timeout=timeout, blocksize=65536
                )
            )
        except OSError as exc:
            raise FetchError(f"could not connect to {target.host}: {exc}") from exc

        try:
            connection.request(
                "GET",
                target.path,
                headers={"User-Agent": f"dit-afq/{__version__}", **(headers or {})},
            )
            response = connection.getresponse()

            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("location")
                if not location:
                    raise FetchError(f"{target.host} returned {response.status} without a location")
                if hop == MAX_REDIRECTS:
                    raise FetchError(f"too many redirects from {url!r}")
                from urllib.parse import urljoin

                current = urljoin(target.url, location)
                continue

            if response.status < 200 or response.status >= 300:
                raise FetchError(f"{target.host} returned HTTP {response.status}")

            declared = response.getheader("content-length")
            if declared is not None:
                try:
                    if int(declared) > max_bytes:
                        raise FetchError(
                            f"content-length {declared} exceeds limit {max_bytes}"
                        )
                except ValueError:
                    pass

            chunks = []
            total = 0
            try:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"response exceeds limit of {max_bytes} bytes")
                    chunks.append(chunk)
            except OSError as exc:
                raise FetchError(f"reading {target.host} failed: {exc}") from exc

            return b"".join(chunks), target
        except (OSError, ssl.SSLError) as exc:
            raise FetchError(f"request to {target.host} failed: {exc}") from exc
        finally:
            connection.close()
