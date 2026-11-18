"""Publisher rate limits reach every request, not just the tiered engine's.

Before: the per-host buckets were taken only by HttpClient and two legacy
helpers (Crossref, DataCite). The legacy chain's publisher PDF requests (Wiley
pdfdirect, Springer content/pdf), the institutional cookie route and the
browser routes (LibKey, Atypon) asked publishers at whatever rate the workers
produced, and a publisher's several hosts each had an allowance of their own.

Now a paced transport adapter takes a token for every hop, hosts of one
publisher share one bucket, browser routes take one per navigation, and a
publisher that refuses a real browser (or 429s repeatedly) is left alone for
the rest of the run.
"""
import io
import re
from pathlib import Path

import pytest
import requests
import urllib3
from requests.adapters import HTTPAdapter

from fetchpdf.retrieval import ratelimit as rl
from fetchpdf.retrieval.ratelimit import (BLOCK_AFTER_CONSECUTIVE_429, HostRateLimiter,
                                          PacedAdapter, PacedSession, PublisherBlocked)
from fetchpdf.retrieval.tiers import load_ladder

FAST = {"rps": 10_000, "burst": 10_000}


class CountingLimiter(HostRateLimiter):
    """A fast limiter that records which bucket each token came from."""

    def __init__(self, config=None):
        super().__init__(config if config is not None else {"_default": FAST})
        self.taken = []

    def acquire(self, host):
        self.taken.append(self.bucket_key(host))
        return 0.0


@pytest.fixture
def limiter(monkeypatch):
    """Install a counting limiter as the process-wide one, for this test only."""
    lim = CountingLimiter()
    monkeypatch.setattr(rl, "_SHARED_LIMITER", lim)
    return lim


def _fake_transport(monkeypatch, routes):
    """Answer requests from *routes* ({url: (status, headers)}) at the adapter.

    Patches the base HTTPAdapter.send, which PacedAdapter.send calls after
    taking its token -- so pacing is exercised and nothing leaves the machine.
    """
    sent = []

    def send(self, request, **kwargs):
        sent.append(request.url)
        status, headers = routes.get(request.url, (404, {}))
        raw = urllib3.HTTPResponse(body=io.BytesIO(b"%PDF-1.4 x"), status=status,
                                   headers=headers, preload_content=False)
        return self.build_response(request, raw)

    monkeypatch.setattr(HTTPAdapter, "send", send)
    return sent


# -- which bucket a host draws from -------------------------------------------

def test_a_publishers_hosts_share_one_bucket():
    lim = HostRateLimiter(load_ladder().rate_limits)
    springer = {lim.bucket(h) for h in ("link.springer.com", "static-content.springer.com",
                                        "media.springernature.com", "www.nature.com")}
    assert len(springer) == 1, "Springer's 1 s rule is per publisher, not per hostname"
    assert springer.pop().rate <= 1.0


def test_a_parent_entry_covers_subdomains_and_an_exact_entry_wins():
    lim = HostRateLimiter({"example.com": {"rps": 1, "burst": 1},
                           "api.example.com": {"rps": 5, "burst": 5},
                           "_default": {"rps": 3, "burst": 3}})
    assert lim.bucket_key("www.example.com") == "example.com"
    assert lim.bucket_key("deep.cdn.example.com") == "example.com"
    assert lim.bucket_key("api.example.com") == "api.example.com"
    assert lim.bucket("api.example.com").rate == 5
    # Unlisted hosts keep a bucket of their own at the default rate, as before.
    assert lim.bucket_key("other.org") == "other.org"
    assert lim.bucket("other.org").rate == 3
    assert lim.bucket_key("WWW.Example.COM:443") == "example.com"


def test_wiley_is_one_request_per_ten_seconds_on_every_wiley_host():
    lim = HostRateLimiter(load_ladder().rate_limits)
    for host in ("onlinelibrary.wiley.com", "agupubs.onlinelibrary.wiley.com",
                 "api.wiley.com"):
        assert lim.bucket_key(host) == "bucket:wiley"
    assert lim.bucket("onlinelibrary.wiley.com").rate <= 0.1


def test_members_of_one_bucket_agree_on_its_rate():
    """The first host to create a shared bucket sets its rate; members must agree."""
    groups = {}
    for host, spec in load_ladder().rate_limits.items():
        if isinstance(spec, dict) and spec.get("bucket"):
            groups.setdefault(spec["bucket"], set()).add((spec.get("rps"), spec.get("burst")))
    assert groups, "expected publisher buckets in ladder.json"
    for name, rates in groups.items():
        assert len(rates) == 1, f"bucket {name!r} has members with different rates: {rates}"


# -- every hop takes a token ----------------------------------------------------

def test_every_redirect_hop_takes_a_token_from_its_own_host(monkeypatch):
    lim = CountingLimiter(load_ladder().rate_limits)
    sent = _fake_transport(monkeypatch, {
        "https://doi.org/10.1002/x": (302, {"Location": "https://onlinelibrary.wiley.com/doi/10.1002/x"}),
        "https://onlinelibrary.wiley.com/doi/10.1002/x": (
            302, {"Location": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/x"}),
        "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/x": (200, {"Content-Type": "application/pdf"}),
    })
    with PacedSession(lim) as session:
        r = session.get("https://doi.org/10.1002/x")
    assert r.status_code == 200
    assert len(sent) == 3
    assert lim.taken == ["doi.org", "bucket:wiley", "bucket:wiley"]


def test_module_level_requests_calls_are_paced(monkeypatch, limiter):
    """The legacy chain calls `requests.get` ~40 times; all of them are paced."""
    import fetchpdf.fetchpdf as fpd
    import fetchpdf.retrieval.institutional as inst

    _fake_transport(monkeypatch, {
        "https://link.springer.com/content/pdf/10.1007/x.pdf": (200, {}),
    })
    fpd.requests.get("https://link.springer.com/content/pdf/10.1007/x.pdf", timeout=5)
    inst.requests.get("https://link.springer.com/content/pdf/10.1007/x.pdf", timeout=5)
    assert limiter.taken == ["link.springer.com", "link.springer.com"]


def test_modules_that_call_requests_directly_use_the_paced_stand_in():
    import importlib

    # import_module, not `import a.b as c`: the package re-exports functions
    # named like these modules, which shadows the module attribute.
    modules = [importlib.import_module(name) for name in (
        "fetchpdf.fetchpdf", "fetchpdf.fetch_abstract_from_doi",
        "fetchpdf.fetch_metadata_from_doi", "fetchpdf.retrieval.institutional")]
    fpd = modules[0]
    for module in modules:
        assert module.requests is rl.paced_requests, module.__name__
    # The stand-in still exposes the real module's exceptions and helpers.
    assert fpd.requests.RequestException is requests.RequestException
    assert fpd.requests.Session is PacedSession


def test_no_route_module_bypasses_the_limiter_with_a_bare_requests_import():
    """A new `import requests` in a route module would silently skip the limits."""
    root = Path(__file__).resolve().parents[1] / "fetchpdf"
    checked = [root / "fetchpdf.py", root / "fetch_abstract_from_doi.py",
               root / "fetch_metadata_from_doi.py"]
    checked += [root / "retrieval" / n for n in
                ("institutional.py", "libkey.py", "supplement_atypon.py", "repository_waf.py")]
    checked += sorted((root / "retrieval" / "sources").glob("*.py"))
    checked += sorted((root / "retrieval").glob("supplement*.py"))
    bare = re.compile(r"^\s*import requests\s*$|^\s*from requests import (get|Session)\b", re.M)
    offenders = [str(p.relative_to(root)) for p in checked if bare.search(p.read_text())]
    assert not offenders, f"route modules importing requests unpaced: {offenders}"


def test_http_client_paces_at_the_adapter_with_its_own_limiter(monkeypatch):
    from fetchpdf.retrieval.http import HttpClient

    lim = CountingLimiter()
    _fake_transport(monkeypatch, {"https://api.crossref.org/works/x": (200, {})})
    http = HttpClient(lim)
    assert isinstance(http.session.get_adapter("https://api.crossref.org/"), PacedAdapter)
    http.get("https://api.crossref.org/works/x", polite=False)
    assert lim.taken == ["api.crossref.org"], "one token per request, not two"


def test_the_legacy_crossref_helper_takes_one_token_per_request(monkeypatch, limiter):
    """The helper used to take a token itself; under the paced adapter that
    would be two per request -- half the published allowance."""
    import fetchpdf.fetchpdf as fpd

    url = "https://api.crossref.org/works/10.1/x"
    _fake_transport(monkeypatch, {})

    def send(self, request, **kwargs):
        raw = urllib3.HTTPResponse(body=io.BytesIO(b'{"message": {"title": ["T"]}}'),
                                   status=200, preload_content=False,
                                   headers={"Content-Type": "application/json"})
        return self.build_response(request, raw)

    monkeypatch.setattr(HTTPAdapter, "send", send)
    assert fpd._crossref_get(url) == {"title": ["T"]}
    assert limiter.taken == ["api.crossref.org"]


# -- the stop list -----------------------------------------------------------------

def test_a_blocked_publisher_is_not_asked_again(monkeypatch, limiter):
    sent = _fake_transport(monkeypatch, {})
    rl.mark_blocked("https://onlinelibrary.wiley.com/doi/x", "test")
    rl.begin_record()
    with pytest.raises(PublisherBlocked):
        rl.paced_requests.get("https://onlinelibrary.wiley.com/doi/pdfdirect/x")
    with pytest.raises(PublisherBlocked):
        rl.pace_navigation("https://onlinelibrary.wiley.com/doi/epdf/x")
    assert sent == [], "a stopped publisher must not receive the request"
    # This limiter has no Wiley entry, so the whole site is what stopped.
    assert rl.record_refusals() == ["wiley.com"]
    # Other publishers carry on.
    rl.pace_navigation("https://link.springer.com/x")


def test_blocking_a_configured_publisher_stops_its_whole_bucket(monkeypatch):
    lim = HostRateLimiter(load_ladder().rate_limits)
    lim.mark_blocked("www.nature.com", "test")
    assert lim.is_blocked("link.springer.com")
    assert not lim.is_blocked("api.elsevier.com")


def test_an_unlisted_providers_whole_site_is_stopped():
    """EBSCO's bot check on one host stops content.ebscohost.com as well."""
    lim = HostRateLimiter({"_default": FAST})
    lim.mark_blocked("research.ebsco.com", "test")
    assert lim.is_blocked("research.ebsco.com")
    lim.mark_blocked("search.ebscohost.com", "test")
    assert lim.is_blocked("content.ebscohost.com")
    assert not lim.is_blocked("ovid.com")


def test_a_run_of_429s_stops_the_publisher_but_403s_do_not():
    lim = HostRateLimiter({"_default": FAST})
    for _ in range(BLOCK_AFTER_CONSECUTIVE_429 * 3):
        lim.note_status("onlinelibrary.wiley.com", 403)     # not licensed: an answer
    assert not lim.is_blocked("onlinelibrary.wiley.com")
    for _ in range(BLOCK_AFTER_CONSECUTIVE_429 - 1):
        lim.note_status("onlinelibrary.wiley.com", 429)
    lim.note_status("onlinelibrary.wiley.com", 200)          # a success resets the run
    for _ in range(BLOCK_AFTER_CONSECUTIVE_429 - 1):
        lim.note_status("onlinelibrary.wiley.com", 429)
    assert not lim.is_blocked("onlinelibrary.wiley.com")
    lim.note_status("onlinelibrary.wiley.com", 429)
    assert lim.is_blocked("onlinelibrary.wiley.com")


def test_the_adapter_counts_429s_and_honours_retry_after(monkeypatch):
    lim = CountingLimiter()
    penalties = []
    monkeypatch.setattr(lim, "penalize", lambda host, s: penalties.append((host, s)))
    url = "https://www.example.org/x.pdf"
    _fake_transport(monkeypatch, {url: (429, {"Retry-After": "7"})})
    with PacedSession(lim) as session:
        for _ in range(BLOCK_AFTER_CONSECUTIVE_429):
            assert session.get(url).status_code == 429
        with pytest.raises(PublisherBlocked):
            session.get(url)
    assert penalties[0] == ("www.example.org", 7.0)


def test_the_summary_names_each_stopped_publisher_and_what_it_cost():
    lim = HostRateLimiter(load_ladder().rate_limits)
    lim.mark_blocked("onlinelibrary.wiley.com", "bot challenge did not clear in a real browser")
    for _ in range(3):
        with pytest.raises(PublisherBlocked):
            lim.check_not_blocked("onlinelibrary.wiley.com")
    (line,) = lim.blocked_summary()
    assert "bucket:wiley" in line and "3 later request(s) not sent" in line


def test_the_institutional_cookie_route_never_contacts_a_stopped_publisher(monkeypatch, limiter):
    import fetchpdf.retrieval.institutional as inst

    sent = _fake_transport(monkeypatch, {})
    doi = "10.1002/anie.202400001"
    urls = inst.publisher_pdf_urls(doi)
    assert urls, "Wiley PDF URLs are derivable from the DOI"
    session = inst._session(urls[0], [], None, "application/pdf")
    assert isinstance(session.get_adapter(urls[0]), PacedAdapter)
    rl.mark_blocked(urls[0], "test")
    url, challenged = inst.fetch_with_cookies(doi, "/nonexistent/x.pdf", [])
    assert url is None and not challenged
    assert sent == []


def test_the_batch_failure_row_says_blocked(tmp_path):
    """failed_dois.csv tells "stopped by us" from "nothing found"."""
    import threading
    import fetchpdf.fetchpdf as fpd

    fpd._append_failed_to_csv(str(tmp_path), "10.1002/x", category="blocked",
                              detail="publisher stopped for this run: bucket:wiley",
                              lock=threading.Lock())
    row = (tmp_path / "failed_dois.csv").read_text().splitlines()[1]
    assert ",blocked," in row
