"""Institutional retrieval through LibKey (Third Iron), for IP-authenticated libraries.

Some subscriptions are not on the publisher's own site but at an aggregator --
at the University of Graz, Hogrefe journals are licensed through Ovid and
INFORMS journals through EBSCO. LibKey knows each library's linking rules, so
instead of hard-coding them this route follows LibKey, as the LibKey Nomad
"Download PDF" button does:

    https://libkey.io/libraries/<id>/<doi>        (JS page; lists the links)
      -> .../articles/<articleId>/full-text-file  (DOWNLOAD PDF)
      -> publisher / Ovid / EBSCO, which serve the PDF to the campus IP

The PDF is taken from the browser's own network traffic -- the first response
that really is a PDF -- or from a download, so no host-specific code is needed
(Ovid loads it from assets.ovid.com, EBSCO's viewer from content.ebscohost.com
about 20 s after landing).

Off unless ``FETCHPDF_LIBKEY_LIBRARY`` (the library's LibKey ID) is set. No
cookies are sent: access is by IP, so the route only works on the library's
network or VPN, like the browser it imitates.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urljoin, urlsplit

from . import institutional as inst
from . import ratelimit as _ratelimit
from .blocked import CHALLENGE_TITLE_RE
from .display import browser_launch_args, browser_launch_env

#: The library's LibKey landing page for a DOI.
LIBKEY_BASE = "https://libkey.io/libraries/{id}/{doi}"
#: How long to wait for the PDF after following LibKey's link (EBSCO ~21 s).
LIBKEY_WAIT_S = 60
#: How long to wait for LibKey's page to render its links.
LIBKEY_LINKS_WAIT_S = 15
#: Extra time DOWNLOAD PDF gets once ARTICLE LINK has rendered.
LIBKEY_PDF_GRACE_S = 5
#: Seconds a viewer gets to deliver a PDF it requested before we refetch it.
LIBKEY_REFETCH_AFTER_S = 3


def library_id() -> Optional[str]:
    """The LibKey library ID from the environment, or None when the route is off."""
    lib = (os.environ.get("FETCHPDF_LIBKEY_LIBRARY") or "").strip()
    return lib if lib.isdigit() else None


def _libkey_link(page, timeout_s: float) -> Optional[str]:
    """The DOWNLOAD PDF link on LibKey's page, else ARTICLE LINK, else None.

    The page renders ARTICLE LINK before DOWNLOAD PDF (seen 2026-11-17 on
    INFORMS/EBSCO), so once any link is there, DOWNLOAD PDF still gets
    LIBKEY_PDF_GRACE_S to appear before ARTICLE LINK is taken.
    """
    def href(selector):
        try:
            value = page.get_attribute(selector, "href", timeout=250)
        except Exception:      # noqa: BLE001 - not rendered yet
            value = None
        return urljoin(page.url, value) if value else None

    deadline = time.time() + timeout_s
    article_link, grace_until = None, None
    while time.time() < deadline:
        pdf_link = href('a[href*="full-text-file"]')
        if pdf_link:
            return pdf_link
        article_link = article_link or href('a[href*="content-location"]')
        if article_link:
            grace_until = grace_until or time.time() + LIBKEY_PDF_GRACE_S
            if time.time() >= grace_until:
                return article_link
        page.wait_for_timeout(500)
    return article_link


#: Hosts whose bot wall may say "Are you a robot?" in the body rather than the
#: title (ScienceDirect's pdfft, seen 2026-11-17 on every 10.1016 DOI).
_BODY_CHECKED_HOSTS = ("sciencedirect.com", "elsevier.com")
_ROBOT_TEXT_RE = re.compile(r"are you a robot|verify you are human|unusual traffic", re.I)


def _bot_wall(page) -> bool:
    """True when *page* is a bot check rather than content.

    Matched by title (blocked.CHALLENGE_TITLE_RE, the test the rest of fetchpdf
    uses) and, on ScienceDirect, by the head of the visible text. Detection
    only: the route gives up on a wall, it never interacts with one.
    """
    try:
        if CHALLENGE_TITLE_RE.match(page.title() or ""):
            return True
        host = urlsplit(page.url or "").hostname or ""
        if any(host == h or host.endswith("." + h) for h in _BODY_CHECKED_HOSTS):
            text = page.evaluate(
                "() => ((document.body && document.body.innerText) || '').slice(0, 2000)")
            return bool(_ROBOT_TEXT_RE.search(text or ""))
    except Exception:      # noqa: BLE001 - mid-navigation
        return False
    return False


def fetch_via_libkey(doi, save_path, user_agent=None, verbose=False) -> Optional[str]:
    """Follow LibKey to the library's copy of *doi*. Returns the PDF's URL, or None."""
    lib = library_id()
    if not lib or not doi:
        return None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        if verbose:
            print("  libkey: playwright not installed")
        return None
    env = browser_launch_env()
    if env is None:
        if verbose:
            print("  libkey: no display for a headed browser")
        return None

    captured: List[tuple] = []          # (url, bytes)
    candidates: List[tuple] = []        # (seen at, url) of PDFs not read whole
    downloads: list = []

    def _note_candidate(url):
        if url and url not in [u for _t, u in candidates]:
            candidates.append((time.time(), url))

    def _on_response(response):
        # Read the body here: once the page navigates on, it is gone.
        if captured:
            return
        ctype = (response.headers.get("content-type") or "").lower()
        if "html" in ctype and "viewer.html" in response.url:
            # A pdf.js viewer names its file in ?file= (Ovid).
            _note_candidate((parse_qs(urlsplit(response.url).query).get("file") or [None])[0])
            return
        if "pdf" not in ctype:
            return
        try:
            body = response.body()
        except Exception:      # noqa: BLE001 - redirect, aborted, or already gone
            body = b""
        if response.status == 200 and body[:4] == b"%PDF" \
                and len(body) <= inst.MAX_PDF_BYTES:
            captured.append((response.url, body))
        else:
            # pdf.js may abort its first request and switch to range requests
            # (206 chunks): the URL is right, the body is not whole.
            _note_candidate(response.url)

    def _refetch(page, context, url):
        """The whole PDF at *url*, asked for again from this browser."""
        try:
            data, _info = inst.fetch_pdf_in_page(page, url)
            if data:
                return data
        except Exception:      # noqa: BLE001 - CORS, page mid-navigation
            pass
        try:
            _ratelimit.pace_navigation(url)
            response = context.request.get(url, timeout=60000)
            body = response.body() if response.ok else b""
        except Exception:      # noqa: BLE001
            return None
        return body if body[:4] == b"%PDF" and len(body) <= inst.MAX_PDF_BYTES else None

    def _on_download(download):
        downloads.append(download)

    def _on_page(new_page):
        new_page.on("download", _on_download)

    start = LIBKEY_BASE.format(id=lib, doi=doi)
    try:
        with sync_playwright() as driver:
            launch = dict(headless=False, env=env,
                          ignore_default_args=["--enable-automation"],
                          args=browser_launch_args(
                              ["--disable-blink-features=AutomationControlled"]))
            try:
                browser = driver.chromium.launch(channel="chrome", **launch)
            except Exception:      # noqa: BLE001 - no Chrome; Playwright's Chromium
                browser = driver.chromium.launch(**launch)
            try:
                context = browser.new_context(user_agent=user_agent or None,
                                              viewport={"width": 1400, "height": 900},
                                              accept_downloads=True)
                context.on("response", _on_response)
                context.on("page", _on_page)
                page = context.new_page()
                # One token per navigation, from libkey.io's bucket; the hop
                # to the provider takes the next one (see ladder.json).
                _ratelimit.pace_navigation(start)
                page.goto(start, wait_until="domcontentloaded", timeout=60000)
                link = _libkey_link(page, LIBKEY_LINKS_WAIT_S)
                if not link:
                    if verbose:
                        print(f"  libkey: no PDF or article link for {doi} "
                              f"(not licensed, or unknown to LibKey)")
                    return None
                if verbose:
                    print(f"  libkey: {start} -> {link}")
                _ratelimit.pace_navigation(link)
                try:
                    page.goto(link, wait_until="domcontentloaded", timeout=60000)
                except Exception:      # noqa: BLE001 - "Download is starting", slow IdP
                    pass

                deadline = time.time() + LIBKEY_WAIT_S
                tried = set()
                while time.time() < deadline and not captured and not downloads:
                    page.wait_for_timeout(500)
                    # A PDF URL seen but not read whole: once the viewer had
                    # a few seconds to deliver it, fetch it ourselves.
                    for seen, cand in list(candidates):
                        if cand in tried or time.time() - seen < LIBKEY_REFETCH_AFTER_S:
                            continue
                        tried.add(cand)
                        data = _refetch(page, context, cand)
                        if data:
                            captured.append((cand, data))
                            break
                    if captured or downloads:
                        break
                    if _bot_wall(page):
                        # Give an automatic challenge its usual chance to clear
                        # by itself; never click or solve anything.
                        inst.wait_past_challenge(page)
                        page.wait_for_timeout(1000)
                        if _bot_wall(page) and not captured and not downloads:
                            host = urlsplit(page.url or "").hostname or ""
                            # The provider (EBSCO, Ovid, ScienceDirect, ...)
                            # refused a real browser: stop asking it this run.
                            _ratelimit.mark_blocked(
                                host, "bot check did not clear in a real browser")
                            if verbose:
                                if "sciencedirect.com" in host:
                                    print("  libkey: ScienceDirect bot check -- use the "
                                          "Elsevier API (ELSEVIER_TDM_API_KEY)")
                                else:
                                    print(f"  libkey: bot check at {host} did not clear")
                            return None

                data, url = None, None
                if captured:
                    url, data = captured[0]
                elif downloads:
                    download = downloads[0]
                    path = download.path()
                    if path and os.path.getsize(path) <= inst.MAX_PDF_BYTES:
                        data, url = Path(path).read_bytes(), download.url
                else:
                    # Landed on a publisher the cookie route knows (Wiley,
                    # Atypon) without the PDF arriving: ask for it the way the
                    # publisher route does.
                    host = inst.registrable_domain(urlsplit(page.url or "").hostname or "")
                    if host and host in inst.publisher_sites(doi):
                        landing = page.url
                        for pdf_url in inst.publisher_pdf_urls(doi):
                            data, _info = inst.navigate_for_pdf(page, pdf_url, referer=landing)
                            if data:
                                url = pdf_url
                                break
                if verbose:
                    print(f"  libkey: {'PDF from ' + url if data else 'no PDF'} "
                          f"(at {(page.url or '')[:100]})")
                if data and inst._write_pdf_bytes(data, save_path):
                    return url
            finally:
                browser.close()
    except Exception as exc:      # noqa: BLE001 - never fail a record on this
        if verbose:
            print(f"  libkey: failed ({type(exc).__name__}: {str(exc)[:80]})")
    return None
