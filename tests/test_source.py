"""Server-side request policy: no localhost, loopback, private or reserved targets.

The fetch path is the one place this project opens a socket, so the policy is
asserted against every class of address a resolver could return. The resolver
is injected, which means these tests make no network calls and cannot be
affected by the machine they run on.
"""

from __future__ import annotations

import ipaddress

import pytest

from dit.data.source import (
    FetchError,
    UnsafeURL,
    resolve_host,
    safe_request,
    validate_url,
)

PUBLIC = "203.0.113.10"  # TEST-NET-3 in the literal, treated as an address below


def _resolver(*addresses: str):
    return lambda host: [ipaddress.ip_address(address) for address in addresses]


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "local",
        "broadcasthost",
        "example.local",
        "svc.internal",
        "host.home.arpa",
        "foo.corp",
        "svc.lan",
    ],
)
def test_non_routable_hostnames_are_refused_before_dns(host: str) -> None:
    with pytest.raises(UnsafeURL, match="not routable"):
        validate_url("https://" + host + "/file.mat", resolver=_resolver(PUBLIC))


@pytest.mark.parametrize(
    "scheme", ["file", "ftp", "gopher", "data", "ws", "httpx", ""]
)
def test_only_http_and_https_are_allowed(scheme: str) -> None:
    with pytest.raises(UnsafeURL, match="scheme"):
        validate_url(scheme + "://8.8.8.8/file.mat", resolver=_resolver(PUBLIC))


@pytest.mark.parametrize(
    "url",
    [
        "https://8.8.8.8:8443/f.mat",
        "http://8.8.8.8:3000/f.mat",
        "https://8.8.8.8:0/f.mat",
    ],
)
def test_non_standard_ports_are_refused(url: str) -> None:
    with pytest.raises(UnsafeURL, match="port"):
        validate_url(url, resolver=_resolver(PUBLIC))


def test_credentials_in_the_url_are_refused() -> None:
    with pytest.raises(UnsafeURL, match="credentials"):
        validate_url("https://user:pass@8.8.8.8/f.mat", resolver=_resolver(PUBLIC))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",          # loopback
        "127.5.5.5",          # rest of loopback block
        "10.0.0.5",           # private
        "172.16.1.1",         # private
        "192.168.1.1",        # private
        "169.254.16.4",       # link-local / mDNS
        "100.64.0.1",         # carrier-grade NAT
        "0.0.0.0",            # unspecified
        "192.0.0.8",          # IANA reserved
        "192.0.2.1",          # TEST-NET-1
        "192.88.99.1",        # IETF assigned
        "224.0.0.1",          # multicast
        "239.255.255.250",    # multicast
        "240.0.0.1",          # reserved
        "255.255.255.255",    # broadcast
        "198.51.100.7",       # TEST-NET-2
        "203.0.113.10",       # TEST-NET-3
        "198.18.0.1",         # benchmarking
        "233.252.0.1",        # AS112
        "[::1]",              # IPv6 loopback
        "[::]",               # IPv6 unspecified
        "[fc00::1]",          # IPv6 private (ULA)
        "[fe80::1]",          # IPv6 link-local
        "[ff02::1]",          # IPv6 multicast
        "[::ffff:7f00:1]",    # IPv4-mapped loopback
        # Transition mechanisms embedding a loopback/private v4 address must
        # not slip past ipaddress.is_global (which reports 64:ff9b::/96 as
        # global); _REFUSED_NETWORKS covers them explicitly.
        "[64:ff9b::7f00:1]",  # NAT64 embedding loopback
        "[2002:7f00:1::]",    # 6to4 embedding loopback
        "[2001:0::1]",        # Teredo
    ],
)
def test_every_private_and_reserved_family_is_refused(address: str) -> None:
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://" + address + "/file.mat")


def test_multicast_is_refused_despite_ipaddress_flag() -> None:
    """Regression: ipaddress.is_global reports multicast as global."""

    assert ipaddress.ip_address("224.0.0.1").is_global is True
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://224.0.0.1/file.mat")


def test_ietf_reserved_is_refused_despite_ipaddress_flag() -> None:
    assert ipaddress.ip_address("192.88.99.1").is_global is True
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://192.88.99.1/file.mat")


@pytest.mark.parametrize(
    "address",
    [
        "8.8.8.8",
        "1.1.1.1",
        "9.9.9.9",
        "93.184.216.34",
        "[2606:4700:4700::1111]",
    ],
)
def test_public_addresses_pass(address: str) -> None:
    target = validate_url("https://" + address + "/file.mat")
    assert target.host == address.strip("[]").lower()
    assert target.port == 443
    assert target.scheme == "https"


def test_decimal_ipv4_hostnames_are_not_decoded_locally() -> None:
    """'2130706433' and friends must stay opaque hostnames to this module.

    The policy deliberately does NOT string-decode alternate IPv4 forms:
    a system resolver maps them to loopback (Linux glibc) or fails to
    resolve them (Windows), and either way the refusal is resolver-backed;
    a resolver answering a public address yields a connection pinned to
    exactly that validated address. Lock the contract so a future "helpful"
    local decoder cannot silently change it.
    """
    target = validate_url("http://2130706433/", resolver=_resolver("93.184.216.34"))
    assert target.host == "2130706433"
    assert target.port == 80


def test_zone_id_in_bracketed_ipv6_is_refused_as_malformed() -> None:
    # The module does not URL-decode %25 into the address parser, so a scoped
    # literal is rejected as malformed before the global/private check runs.
    with pytest.raises(UnsafeURL, match="malformed"):
        validate_url("https://[fe80::1%25eth0]/file.mat", resolver=_resolver(PUBLIC))


def test_unbracketed_ipv6_is_refused_as_malformed() -> None:
    with pytest.raises(UnsafeURL, match="bracketed"):
        validate_url("https://::1/file.mat", resolver=_resolver(PUBLIC))


def test_malformed_bracket_is_refused() -> None:
    with pytest.raises(UnsafeURL, match="malformed bracket"):
        validate_url("https://[::1:extra]/file.mat", resolver=_resolver(PUBLIC))


def test_http_defaults_to_port_80() -> None:
    target = validate_url("http://8.8.8.8/data", resolver=_resolver("8.8.8.8"))
    assert target.port == 80
    assert target.path == "/data"


def test_dns_returning_only_private_answers_is_refused() -> None:
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://research.example/f.mat", resolver=_resolver("10.1.2.3", "192.168.0.9"))


def test_dns_returning_nothing_is_refused() -> None:
    with pytest.raises(UnsafeURL, match="no addresses"):
        resolve_host("nowhere.invalid", resolver=lambda host: [])


def test_dns_failure_is_refused_not_silently_ignored() -> None:
    def failing(host: str):
        raise OSError("name resolution failed")

    with pytest.raises(UnsafeURL, match="could not be resolved"):
        validate_url("https://broken.invalid/f.mat", resolver=failing)


def test_mixed_dns_answers_fail_closed() -> None:
    # A resolver that returns one public and one private address must not
    # allow the request: the client could reach either.
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://research.example/f.mat", resolver=_resolver("93.184.216.34", "127.0.0.1"))


def test_ipv4_mapped_ipv6_loopback_is_refused() -> None:
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://[::ffff:127.0.0.1]/f.mat")


def test_mapped_ipv6_private_is_refused() -> None:
    with pytest.raises(UnsafeURL, match="non-public"):
        validate_url("https://[::ffff:10.0.0.1]/f.mat")


def test_socket_is_pinned_to_the_validated_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """The socket must dial the address validation approved, never a fresh DNS answer.

    Validating one resolution and then letting the HTTP client resolve the
    hostname again is the rebinding window OWASP's SSRF cheat sheet calls out:
    an attacker with a short-TTL record flips the second answer to an internal
    address. The fix pins the socket to the validated address, so this test's
    resolver answers differently on every lookup — only the first (validated)
    answer may ever be dialed, and there must be no second lookup at all.
    """

    import http.client
    import socket as socket_module

    dialed: list[tuple[str, int]] = []
    lookups = {"count": 0}

    def fake_create_connection(address, *args, **kwargs):
        class DummySocket:
            def setsockopt(self, *args: object) -> None:
                pass

            def close(self) -> None:
                pass

        dialed.append(tuple(address))
        return DummySocket()

    def rebinding_resolver(host: str):
        lookups["count"] += 1
        if lookups["count"] == 1:
            return [ipaddress.ip_address("93.184.216.34")]
        return [ipaddress.ip_address("93.184.216.35")]

    class FakeResponse:
        status = 200

        def getheader(self, name: str):
            return None

        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, host, *args, **kwargs) -> None:
            self.host = host
            self.port = args[0] if args else kwargs.get("port")
            self.timeout = kwargs.get("timeout")

        def request(self, *args, **kwargs) -> None:
            # http.client connects lazily inside request; the fake has to do
            # the same for the pinning to be observable.
            self.connect()

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket_module, "create_connection", fake_create_connection)
    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)

    body, target = safe_request("http://research.example/f.mat", resolver=rebinding_resolver)
    assert body == b""
    assert dialed == [("93.184.216.34", 80)]
    assert lookups["count"] == 1


def test_tls_identity_stays_on_the_hostname_while_the_socket_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SNI and certificate verification must use the hostname, not the pinned IP."""

    import http.client
    import socket as socket_module
    import ssl as ssl_module

    dialed: list[tuple[str, int]] = []
    captured: dict[str, object] = {}

    def fake_create_connection(address, *args, **kwargs):
        class DummySocket:
            def setsockopt(self, *args: object) -> None:
                pass

            def close(self) -> None:
                pass

        dialed.append(tuple(address))
        return DummySocket()

    class FakeContext:
        def wrap_socket(self, sock, server_hostname=None):
            captured["server_hostname"] = server_hostname
            return sock

    class FakeResponse:
        status = 200

        def getheader(self, name: str):
            return None

        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, host, *args, **kwargs) -> None:
            self.host = host
            self.port = args[0] if args else kwargs.get("port")
            self.timeout = kwargs.get("timeout")

        def request(self, *args, **kwargs) -> None:
            self.connect()

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket_module, "create_connection", fake_create_connection)
    monkeypatch.setattr(ssl_module, "create_default_context", lambda **kw: FakeContext())
    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)

    safe_request("https://research.example/f.mat", resolver=_resolver("93.184.216.34"))
    assert dialed == [("93.184.216.34", 443)]
    assert captured["server_hostname"] == "research.example"


def test_redirect_to_loopback_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the hop-by-hop recheck, not just the first request."""

    import http.client

    seen: list[str] = []

    class FakeResponse:
        def __init__(self, status: int, headers: dict) -> None:
            self.status = status
            self._headers = headers

        def getheader(self, name: str):
            return self._headers.get(name)

        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.host = args[0]

        def request(self, method: str, path: str, headers=None) -> None:
            seen.append(self.host)

        def getresponse(self) -> FakeResponse:
            if len(seen) == 1:
                return FakeResponse(302, {"location": "http://127.0.0.1/steal"})
            return FakeResponse(200, {})

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(UnsafeURL, match="non-public"):
        safe_request("https://93.184.216.34/data", resolver=_resolver("93.184.216.34"))
    assert seen == ["93.184.216.34"]


def test_fetch_user_agent_uses_the_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request version must follow dit.__version__, never a duplicate literal."""

    import http.client
    from dit import __version__

    captured: dict[str, str] = {}

    class FakeResponse:
        status = 200

        def getheader(self, name: str):
            return None

        def read(self, size: int = -1) -> bytes:
            return b""

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def request(self, method: str, path: str, headers=None) -> None:
            captured.update(headers or {})

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    safe_request("https://93.184.216.34/data", resolver=_resolver("93.184.216.34"))
    assert captured["User-Agent"] == "dit-afq/" + __version__


def test_too_many_redirects_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    class FakeResponse:
        status = 302

        def getheader(self, name: str):
            return "http://93.184.216.34/again" if name.lower() == "location" else None

        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def request(self, *args, **kwargs) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(FetchError, match="too many redirects"):
        safe_request(
            "https://93.184.216.34/a",
            resolver=lambda host: [ipaddress.ip_address("93.184.216.34")],
        )


def test_response_size_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    class FakeResponse:
        status = 200
        _first = True

        def getheader(self, name: str):
            return None

        def read(self, size: int = -1) -> bytes:
            self._first = False
            return b"x" * size

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def request(self, *args, **kwargs) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(FetchError, match="exceeds limit"):
        safe_request(
            "https://93.184.216.34/big",
            max_bytes=100,
            resolver=lambda host: [ipaddress.ip_address("93.184.216.34")],
        )


def test_first_validated_address_is_preferred_and_a_refusal_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-homed host keeps getaddrinfo-style fallback over *validated* addresses.

    Pinning must not cost availability: when the first validated address
    refuses the connection, the next validated address is tried. Both
    addresses came from the same validated resolution, so the rebinding
    guarantee (never dial an address validation did not approve) holds.
    """

    import http.client
    import socket as socket_module

    dialed: list[tuple[str, int]] = []

    def fake_create_connection(address, *args, **kwargs):
        class DummySocket:
            def setsockopt(self, *args: object) -> None:
                pass

            def close(self) -> None:
                pass

        dialed.append(tuple(address))
        if len(dialed) == 1:
            raise ConnectionRefusedError("first address refused")
        return DummySocket()

    class FakeResponse:
        status = 200

        def getheader(self, name: str):
            return None

        def read(self, size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeConnection:
        def __init__(self, host, *args, **kwargs) -> None:
            self.host = host
            self.port = args[0] if args else kwargs.get("port")
            self.timeout = kwargs.get("timeout")

        def request(self, *args, **kwargs) -> None:
            self.connect()

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket_module, "create_connection", fake_create_connection)
    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)

    body, _ = safe_request(
        "http://research.example/f.mat",
        resolver=lambda host: [
            ipaddress.ip_address("93.184.216.34"),
            ipaddress.ip_address("93.184.216.35"),
        ],
    )
    assert body == b""
    assert dialed == [("93.184.216.34", 80), ("93.184.216.35", 80)]


def test_all_validated_addresses_refusing_raises_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import http.client
    import socket as socket_module

    def fake_create_connection(address, *args, **kwargs):
        raise ConnectionRefusedError("refused")

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.port = 80
            self.timeout = kwargs.get("timeout")

        def request(self, *args, **kwargs) -> None:
            self.connect()

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket_module, "create_connection", fake_create_connection)
    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)

    with pytest.raises(FetchError, match="request to research.example failed"):
        safe_request(
            "http://research.example/f.mat",
            resolver=lambda host: [ipaddress.ip_address("93.184.216.34")],
        )
