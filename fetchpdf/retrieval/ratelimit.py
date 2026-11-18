"""Per-host token buckets.

Per-host and not global, because the configured limits differ by more than an
order of magnitude across the APIs this package touches: NCBI E-utilities is
3/s unkeyed and 10/s keyed, Semantic Scholar is 1/s unkeyed, Crossref's polite
pool is 10/s, and arXiv now wants roughly one request every three seconds. A
single global limiter set safe enough for arXiv would make a PMC-heavy batch
take all day; set fast enough for Crossref it would get us 429'd off arXiv.

Buckets are process-wide and lock-guarded so the ThreadPoolExecutor in
batch_fetch_pdfs shares one budget per host rather than one per worker.
"""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter


class TokenBucket:
    """Classic token bucket: `rate` tokens/second, capped at `burst`."""

    def __init__(self, rate: float, burst: float):
        self.rate = max(float(rate), 0.001)
        self.burst = max(float(burst), 1.0)
        self._tokens = self.burst
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available. Returns seconds actually slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.burst, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return slept
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            time.sleep(wait)
            slept += wait

    def penalize(self, seconds: float) -> None:
        """Drain the bucket after a 429, so the next caller waits it out."""
        with self._lock:
            self._tokens = 0.0
            self._last = time.monotonic() + max(0.0, seconds)


#: The process-wide limiter. Politeness is a property of the PROCESS, not of a
#: batch: an API's allowance does not multiply because we started a second
#: walk. Three call sites used to build their own -- the tiered engine, the
#: supplementary pass and the legacy chain's Crossref helper -- so a run that
#: touched two of them had two budgets for the same host and could serve
#: Crossref 20 requests a second against a published allowance of 10.
#:
#: First config wins, which is correct here because every caller passes the
#: same `ladder.rate_limits`. Tests that want an isolated limiter construct
#: HostRateLimiter directly.
_SHARED_LIMITER = None
_SHARED_LOCK = threading.Lock()


def shared_host_limiter(config: Optional[dict] = None) -> "HostRateLimiter":
    """The one limiter every production caller should use."""
    global _SHARED_LIMITER
    with _SHARED_LOCK:
        if _SHARED_LIMITER is None:
            _SHARED_LIMITER = HostRateLimiter(config)
        return _SHARED_LIMITER


def _normalize_host(host: Optional[str]) -> str:
    """Lower-case hostname without port or credentials; '' when there is none."""
    host = (host or "").strip().lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if host.startswith("["):              # IPv6 literal
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0].rstrip(".")


#: Second-level labels under a two-letter country TLD that are registries, not
#: organisations: ``ac.at``, ``co.uk``. Enough to group a publisher's hosts for
#: blocking without a public-suffix dependency.
_REGISTRY_SLDS = frozenset({"ac", "co", "com", "edu", "gov", "gv", "net", "or", "org"})


def _site_of(host: str) -> str:
    """The registrable domain of *host*: ``content.ebscohost.com`` -> ``ebscohost.com``."""
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _REGISTRY_SLDS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


class PublisherBlocked(requests.RequestException):
    """A request to a publisher this run has stopped asking.

    A ``RequestException`` so every existing ``except RequestException``
    already treats it as "this route did not work" and moves on.
    """

    def __init__(self, key: str, reason: str):
        super().__init__(f"{key}: requests stopped for this run ({reason})")
        self.key = key
        self.reason = reason


#: Consecutive 429s from one bucket, after the backoff each already waited
#: out, that end requests to it for the run. Not 403: a publisher answers 403
#: for every journal the library does not subscribe to, so a run of 403s says
#: "not licensed", not "stop".
BLOCK_AFTER_CONSECUTIVE_429 = 5


class HostRateLimiter:
    """Resolves a host to its bucket, honoring the keyed/unkeyed distinction.

    A host's spec is the exact entry in the config, else the nearest parent
    domain's (``onlinelibrary.wiley.com`` -> ``wiley.com``), else ``_default``.
    Specs naming the same ``"bucket"`` share one ``TokenBucket``, so a
    publisher's PDF CDN and article pages draw on one allowance. A host with no
    entry of its own keeps a bucket of its own at the default rate.

    It also holds the run's stop list: buckets a publisher told us to leave
    alone (a challenge a real browser could not get past, or a run of 429s).
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = dict(config or {})
        self._default = self._config.get("_default", {"rps": 3, "burst": 3})
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._blocked: Dict[str, dict] = {}
        self._streak_429: Dict[str, int] = {}

    def _spec_for(self, host: str) -> dict:
        return self._resolve(host)[1]

    def _resolve(self, host: Optional[str]) -> Tuple[str, dict]:
        """``(bucket key, spec)`` for *host*."""
        name = _normalize_host(host)
        if not name:
            return "_default", self._default
        labels = name.split(".")
        for i in range(len(labels) - 1):          # never a bare TLD
            candidate = ".".join(labels[i:])
            spec = self._config.get(candidate)
            if isinstance(spec, dict) and not candidate.startswith("_"):
                group = spec.get("bucket")
                if group:
                    return f"bucket:{group}", spec
                # A parent-domain entry covers its subdomains as one bucket.
                return candidate, spec
        return name, self._default

    def bucket_key(self, host: Optional[str]) -> str:
        """The name of the bucket *host* draws from (what logs and reports show)."""
        return self._resolve(host)[0]

    def bucket(self, host: Optional[str]) -> TokenBucket:
        key, spec = self._resolve(host)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is not None:
                return bucket
            rate = spec.get("rps", 3)
            key_env = spec.get("key_env")
            # The keyed allowance only applies if we actually hold the key.
            if key_env and os.getenv(key_env) and spec.get("rps_keyed"):
                rate = spec["rps_keyed"]
            bucket = TokenBucket(rate=rate, burst=spec.get("burst", rate))
            self._buckets[key] = bucket
            return bucket

    def acquire(self, host: Optional[str]) -> float:
        return self.bucket(host).acquire()

    def penalize(self, host: Optional[str], seconds: float) -> None:
        self.bucket(host).penalize(seconds)

    # -- the stop list --------------------------------------------------------

    def _block_key(self, host: Optional[str]) -> str:
        """Configured publishers block as their bucket; others as their whole site."""
        key, spec = self._resolve(host)
        if spec is self._default:
            return _site_of(_normalize_host(host)) or key
        return key

    def mark_blocked(self, host: Optional[str], reason: str) -> str:
        """Stop requests to *host*'s publisher for the rest of the run."""
        key = self._block_key(host)
        with self._lock:
            entry = self._blocked.get(key)
            if entry is None:
                self._blocked[key] = {"reason": reason, "host": _normalize_host(host),
                                      "since": time.time(), "refused": 0}
                print(f"⛔ {key}: stopping requests for the rest of this run ({reason}). "
                      f"Its DOIs are recorded as blocked; check by hand before retrying.")
        return key

    def is_blocked(self, host: Optional[str]) -> bool:
        with self._lock:
            return self._block_key(host) in self._blocked

    def check_not_blocked(self, host: Optional[str]) -> None:
        """Raise ``PublisherBlocked`` when *host*'s publisher is on the stop list."""
        key = self._block_key(host)
        with self._lock:
            entry = self._blocked.get(key)
            if entry is None:
                return
            entry["refused"] += 1
            reason = entry["reason"]
        raise PublisherBlocked(key, reason)

    def note_status(self, host: Optional[str], status: int) -> None:
        """Count consecutive 429s per bucket; enough of them stop the bucket."""
        key = self._block_key(host)
        with self._lock:
            if status == 429:
                streak = self._streak_429.get(key, 0) + 1
                self._streak_429[key] = streak
            else:
                self._streak_429.pop(key, None)
                return
        if streak >= BLOCK_AFTER_CONSECUTIVE_429:
            self.mark_blocked(host, f"{streak} consecutive HTTP 429")

    def blocked_summary(self) -> List[str]:
        """One line per stopped publisher, for the end-of-run report."""
        with self._lock:
            items = sorted(self._blocked.items(), key=lambda kv: kv[1]["since"])
        return [f"{key}: stopped at {time.strftime('%H:%M:%S', time.localtime(e['since']))} "
                f"({e['reason']}; first seen on {e['host']}), "
                f"{e['refused']} later request(s) not sent"
                for key, e in items]

    def reset_blocks(self) -> None:
        with self._lock:
            self._blocked.clear()
            self._streak_429.clear()


# ---------------------------------------------------------------------------
# One choke point for every request the package makes
# ---------------------------------------------------------------------------
#: Waits longer than this are reported under --verbose, so a slow run reads as
#: deliberately paced rather than hung.
_REPORT_WAIT_S = 2.0
_verbose = False


def set_verbose(on: bool) -> None:
    """Report long rate-limit waits (``--verbose``)."""
    global _verbose
    _verbose = bool(on)


def process_limiter() -> "HostRateLimiter":
    """The shared limiter, built from ladder.json's ``rate_limits``.

    The config-first lookup ``_crossref_bucket`` uses: a caller that asks for
    the shared limiter before anything has loaded the ladder must not freeze it
    with an empty config and the 3/s default for every publisher.
    """
    if _SHARED_LIMITER is not None:
        return _SHARED_LIMITER
    try:
        from .tiers import load_ladder
        limits = load_ladder().rate_limits
    except Exception:      # noqa: BLE001 - a broken ladder must not stop HTTP
        limits = None
    return shared_host_limiter(limits)


#: Per worker thread: the stopped publishers the current record would have
#: asked. Batch workers are threads and each runs one record at a time, so a
#: thread-local scoped by begin_record() is exactly "this record".
_local = threading.local()


def begin_record() -> None:
    """Start counting refusals for the record this thread is about to fetch."""
    _local.refused = []


def record_refusals() -> List[str]:
    """Stopped publishers this thread's current record was kept away from."""
    return list(getattr(_local, "refused", None) or [])


def _wait_turn(limiter: "HostRateLimiter", host: str) -> None:
    try:
        limiter.check_not_blocked(host)
    except PublisherBlocked as exc:
        refused = getattr(_local, "refused", None)
        if refused is not None and exc.key not in refused:
            refused.append(exc.key)
        raise
    slept = limiter.acquire(host)
    if _verbose and slept >= _REPORT_WAIT_S:
        print(f"    rate-limit: {limiter.bucket_key(host)} waited {slept:.1f}s")


def _retry_after_s(response) -> Optional[float]:
    raw = (getattr(response, "headers", None) or {}).get("Retry-After")
    try:
        return min(float(raw), 60.0) if raw else None
    except (TypeError, ValueError):
        return None


class PacedAdapter(HTTPAdapter):
    """An ``HTTPAdapter`` that takes a token before every request it sends.

    Mounted on a session, it sees every hop of a redirect chain --
    ``doi.org`` -> ``onlinelibrary.wiley.com`` -> ``.../pdfdirect`` -- which a
    wrapper around ``session.get`` cannot: those hops happen inside requests.
    """

    def __init__(self, limiter: Optional["HostRateLimiter"] = None, **kwargs):
        self._limiter = limiter
        super().__init__(**kwargs)

    def _get_limiter(self) -> "HostRateLimiter":
        return self._limiter or process_limiter()

    def send(self, request, **kwargs):
        limiter = self._get_limiter()
        host = urlsplit(request.url).hostname or ""
        _wait_turn(limiter, host)
        response = super().send(request, **kwargs)
        limiter.note_status(host, response.status_code)
        if response.status_code in (429, 503):
            wait = _retry_after_s(response)
            if wait:
                # Every worker waits, not just the one that was told to.
                limiter.penalize(host, wait)
        return response


def _mount_paced(session: requests.Session,
                 limiter: Optional["HostRateLimiter"] = None) -> requests.Session:
    adapter = PacedAdapter(limiter)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class PacedSession(requests.Session):
    """``requests.Session`` with ``PacedAdapter`` mounted for http and https."""

    def __init__(self, limiter: Optional["HostRateLimiter"] = None):
        super().__init__()
        _mount_paced(self, limiter)


def paced_session(limiter: Optional["HostRateLimiter"] = None) -> PacedSession:
    return PacedSession(limiter)


def paced_request(method: str, url: str, **kwargs):
    """``requests.request`` through a paced session (a fresh one, as requests does)."""
    with PacedSession() as session:
        return session.request(method=method, url=url, **kwargs)


class _PacedRequests:
    """Stands in for the ``requests`` module in modules that call it directly.

    ``get``/``head``/``post``/``request`` go through a paced session and
    ``Session`` is ``PacedSession``; everything else (exceptions, ``cookies``,
    ``utils``) is the real module. A module binds ``requests = paced_requests``
    and its ~40 call sites are paced without being rewritten -- and tests that
    patch ``module.requests.get`` keep working, because that is now this
    object's attribute.
    """

    Session = PacedSession

    @staticmethod
    def request(method, url, **kwargs):
        return paced_request(method, url, **kwargs)

    @staticmethod
    def get(url, params=None, **kwargs):
        return paced_request("get", url, params=params, **kwargs)

    @staticmethod
    def head(url, **kwargs):
        kwargs.setdefault("allow_redirects", False)
        return paced_request("head", url, **kwargs)

    @staticmethod
    def post(url, data=None, json=None, **kwargs):
        return paced_request("post", url, data=data, json=json, **kwargs)

    def __getattr__(self, name):
        return getattr(requests, name)


paced_requests = _PacedRequests()


# ---------------------------------------------------------------------------
# Browser routes: one token per navigation
# ---------------------------------------------------------------------------

def pace_navigation(url: str) -> None:
    """Wait for *url*'s turn before a browser navigates there.

    One token per top-level navigation (or in-page PDF fetch), not per
    sub-resource: a page load is dozens of requests the publisher serves as one
    reader action. Raises ``PublisherBlocked`` when the publisher is stopped.
    """
    _wait_turn(process_limiter(), urlsplit(url or "").hostname or "")


def mark_blocked(url_or_host: str, reason: str) -> str:
    """Put the publisher serving *url_or_host* on the run's stop list."""
    host = urlsplit(url_or_host).hostname if "//" in (url_or_host or "") else url_or_host
    return process_limiter().mark_blocked(host or "", reason)


def is_blocked(url_or_host: str) -> bool:
    host = urlsplit(url_or_host).hostname if "//" in (url_or_host or "") else url_or_host
    return process_limiter().is_blocked(host or "")


def blocked_summary() -> List[str]:
    return process_limiter().blocked_summary() if _SHARED_LIMITER is not None else []
