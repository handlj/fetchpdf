"""Shared HTTP constants and response checks.

Its own module because fetchpdf imports download_from_aa, so the
latter cannot import back from the former at module scope without a cycle.
"""
import re
import threading
from typing import Optional

# Keep the Chrome version current: some hosts (Zenodo among them) 403 stale
# Chrome UAs as a bot signal, which silently breaks every source on that host.
#
# Defined once and referenced everywhere -- previously four copies drifted apart,
# leaving three Playwright/requests paths on Chrome/120 while the main header
# said 131, so the bot-signal problem this string exists to avoid still applied
# to exactly the browser-driven sources most likely to be fingerprinted.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


#: Polite, self-identifying UA for hosts that BLOCK browser-impersonating
#: clients. Springer and OUP invert the usual bot logic on their direct
#: ``/content/pdf`` routes: a request carrying the Chrome UA above is answered
#: with a ~3 KB HTML interstitial (HTTP 200, ``text/html``), while the very same
#: URL asked with any non-browser UA returns the real ``application/pdf``.
#: Verified 2026-09-03 across several 10.1186 and 10.1093 DOIs -- every one of
#: (no UA), ``curl/8.5.0``, ``python-requests/2.31.0`` and this string served
#: the PDF; only Chrome/131 served HTML.
POLITE_USER_AGENT = "fetchpdf/1.0 (+https://github.com/The-Metascience-Observatory/fetchpdf)"

#: Hosts that must be asked with :data:`POLITE_USER_AGENT` instead of
#: :data:`USER_AGENT`. Deliberately a narrow allow-list rather than a global
#: swap: the Chrome UA exists because other hosts (Zenodo) 403 non-browser or
#: stale-Chrome clients as a bot signal, so flipping it everywhere would trade
#: one silent failure mode for another. Matched on the exact host or any
#: subdomain of it.
POLITE_UA_HOSTS = (
    "link.springer.com",
    "academic.oup.com",
)


def user_agent_for(url: str) -> str:
    """The UA to send for *url* -- polite on :data:`POLITE_UA_HOSTS`, else Chrome."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return USER_AGENT
    for polite in POLITE_UA_HOSTS:
        if host == polite or host.endswith("." + polite):
            return POLITE_USER_AGENT
    return USER_AGENT


def headers_for(url: str, base: dict) -> dict:
    """*base* with its User-Agent swapped for whatever *url*'s host needs."""
    out = dict(base or {})
    out["User-Agent"] = user_agent_for(url)
    return out


def looks_like_pdf(content: bytes, content_type: str = "") -> bool:
    """True when a response body really is a PDF.

    A PDF fetch that comes back as HTML is the failure this guards: the host
    answered 200 with a login wall, a cookie interstitial or a landing page, and
    a caller that only checks ``status_code`` records a success and writes
    3 KB of markup into a ``.pdf``. Checks the magic bytes first -- the header
    is advisory, the body is the fact -- and treats an explicitly HTML
    content-type with no ``%PDF`` magic as a definite miss.
    """
    if content and content[:5].startswith(b"%PDF"):
        return True
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in ("text/html", "application/xhtml+xml"):
        return False
    # No magic and no HTML claim: only believe it if it claims to be a PDF.
    return ctype in ("application/pdf", "application/octet-stream") and bool(content)


def elsevier_headers(api_key: str, extra: dict = None) -> dict:
    """Request headers for any api.elsevier.com call that carries the TDM key.

    The one place the credentials are spelled, so a request can never carry the
    key and forget the institutional token. ``X-ELS-Insttoken`` is added when
    ELSEVIER_INSTTOKEN is set and omitted otherwise -- an empty header would be
    read as a malformed token and refused. It is looked up at call time, not
    bound at import, so a process that loads its .env late still gets it.

    Both go in headers, never the URL: a URL surfaces in exception text, logs
    and provenance, and a credential there is a credential leaked.

    The key is a parameter, not read here, because each caller holds its own
    binding of it (and its tests pin that binding).
    """
    from . import _env

    out = {"X-ELS-APIKey": api_key}
    if _env.ELSEVIER_INSTTOKEN:
        out["X-ELS-Insttoken"] = _env.ELSEVIER_INSTTOKEN
    out.update(extra or {})
    return out


_ELSEVIER_ERROR_CODE_RES = (
    re.compile(rb"<error-code>\s*([A-Z_]+)\s*</error-code>"),
    re.compile(rb'"statusCode"\s*:\s*"([A-Z_]+)"'),
)

#: What each Elsevier refusal means for the user, printed once per run.
_ELSEVIER_ADVICE = {
    "APIKEY_INVALID": "the API key is invalid -- check ELSEVIER_TDM_API_KEY "
                      "(a 32-character key from dev.elsevier.com)",
    "INSTTOKEN_INVALID": "the institutional token is invalid -- check or clear "
                         "ELSEVIER_INSTTOKEN",
    "AUTHENTICATION_ERROR": "the key is valid but not linked to your institution -- ask "
                            "your library for an Elsevier institutional token "
                            "(ELSEVIER_INSTTOKEN), or run on the campus network/VPN",
    "FIRST_PAGE_ONLY": "only first-page previews are served to this key/network -- if "
                       "this repeats for journals your library subscribes to, Elsevier "
                       "does not recognise you as the institution: ask your library for "
                       "an institutional token (ELSEVIER_INSTTOKEN)",
}
_elsevier_warned: set = set()
_elsevier_warn_lock = threading.Lock()


def elsevier_refusal_code(status, content=b"", headers=None) -> Optional[str]:
    """Why Elsevier refused a request, as a code, or None if it did not.

    APIKEY_INVALID / INSTTOKEN_INVALID / AUTHENTICATION_ERROR come from the error
    payload of a 401/403 (XML ``<error-code>`` or JSON ``statusCode``);
    FIRST_PAGE_ONLY from the X-ELS-Status header of a partial 200. Without a
    code these all used to read "not entitled", which sent users looking for a
    missing subscription when the key itself was the problem (seen 2026-11-17:
    a 17-character placeholder key, 401 APIKEY_INVALID on every request).
    """
    if status in (401, 403):
        head = (content or b"")[:4096]
        if isinstance(head, str):
            head = head.encode("utf-8", "replace")
        for rx in _ELSEVIER_ERROR_CODE_RES:
            m = rx.search(head)
            if m:
                return m.group(1).decode()
        return "AUTHENTICATION_ERROR" if status == 403 else "APIKEY_INVALID"
    if elsevier_first_page_only(headers):
        return "FIRST_PAGE_ONLY"
    return None


def warn_elsevier_refusal(code: Optional[str]) -> None:
    """Print what an Elsevier refusal means -- once per code per process.

    Printed without --verbose: an unusable key silently costs every 10.1016
    record its best route.
    """
    if not code:
        return
    with _elsevier_warn_lock:
        if code in _elsevier_warned:
            return
        _elsevier_warned.add(code)
    advice = _ELSEVIER_ADVICE.get(code, "see the Elsevier API response")
    print(f"⚠️  Elsevier API ({code}): {advice}")


def elsevier_first_page_only(headers) -> bool:
    """True when Elsevier's X-ELS-Status header marks a first-page-only PDF.

    A key entitled to a record's XML but not its PDF does not get an error
    from `Accept: application/pdf` -- it gets a genuine typeset FIRST PAGE:
    HTTP 200, %PDF magic, megabyte-scale, real text layer. Nothing in the body
    distinguishes it from the full article; the only in-band signal is

        X-ELS-Status: WARNING - Response limited to first page because
                      requestor not entitled to resource

    (observed 2026-08-16 on PII S0022510X13030578, where the "success" was
    1 of 8 pages). Matched case-insensitively on key and value: the live API
    serves the header lowercased over HTTP/2, while requests' own header dict
    is case-insensitive -- callers pass either.
    """
    for key, value in (headers or {}).items():
        if key.lower() == "x-els-status":
            return "limited to first page" in str(value).lower()
    return False
