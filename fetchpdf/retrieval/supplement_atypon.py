"""Supplementary files from Atypon-hosted journals, via a real browser tab.

Europe PMC holds supplements for many non-open-access articles and is not
permitted to serve them: `supplementaryFiles` answers HTTP 200 wrapping
`<errorBean>...Article with id PMC... is not open access one</errorBean>`.
Those files still exist on the publisher's site, and for PNAS and its Atypon
siblings they are reachable -- but only from something that behaves like a
person with a browser open.

What was measured against 10.1073/pnas.1209746109, all on the same machine and
network:

    direct GET of the SI url (curl, guessed name)         403
    Playwright headless, GET the real url                 403
    Playwright headless, in-page fetch() with 6 cookies   403
    Playwright headless, persistent profile, click        download canceled
    Playwright HEADED (xvfb), click the link              261 KB %PDF   <-- works
    same, fresh profile, on an open-access control        341 KB %PDF   <-- works

Two things are load-bearing and neither is obvious:

  * **Headed, not headless.** This is the discriminating factor. Cookies, a
    persistent profile, a referer and a browser user-agent were all tried
    headless and all refused. On a headless server that means xvfb.
  * **Click, do not navigate.** A direct navigation to the same URL still 403s
    even when headed. The download has to originate from a user gesture on the
    listing page.

The redirect chain explains why: `302 -> ?cookieSet=1 -> 302 ->
/action/cookieAbsent` is Atypon's cookie handshake behind Cloudflare, and it
only completes for a client that looks entirely ordinary.

This is expensive -- one browser per record -- so it is deliberately the last
resort, and the caller only reaches it for the handful of records EPMC has
already said it is withholding.
"""

import os
import shutil
import tempfile
from typing import List, Optional

from .supplement_index import ROLE_SUPPLEMENT, SupplementFile

#: Publishers on the Atypon platform, whose SI lives at /doi/suppl/<doi> and
#: whose files hang off a[href*="suppl_file"]. PNAS is the case that prompted
#: this; the others share the platform and the same URL shape.
_ATYPON_HOSTS = {
    "10.1073": "https://www.pnas.org",          # PNAS
    "10.1126": "https://www.science.org",       # Science / AAAS
    "10.1146": "https://www.annualreviews.org",  # Annual Reviews
    "10.1177": "https://journals.sagepub.com",  # SAGE
    "10.1080": "https://www.tandfonline.com",   # Taylor & Francis
}

_SUPPL_PATH = "{host}/doi/suppl/{doi}"

#: Long enough for Cloudflare's handshake plus a real page render. The listing
#: page took ~5 s to settle in testing; the download itself is fast.
_PAGE_TIMEOUT_MS = 90000
_SETTLE_MS = 5000
_DOWNLOAD_TIMEOUT_MS = 45000

#: One browser per record is the whole cost of this provider, so cap how many
#: files a single record may pull through it.
_MAX_FILES = 12


def atypon_host_for(doi: str) -> Optional[str]:
    """The publisher base URL for a DOI, or None if not an Atypon journal."""
    if not doi:
        return None
    prefix = str(doi).split("/", 1)[0].strip()
    return _ATYPON_HOSTS.get(prefix)


def note_blocked(ctx, provider: str, url: str, detail: str) -> None:
    """Record that a provider was refused, so the manifest can say "blocked".

    An empty list alone reads as "nothing listed", which claims the article has
    no supplements. A refused listing page proves nothing of the sort.
    """
    scratch = getattr(ctx, "scratch", None)
    if isinstance(scratch, dict):
        scratch.setdefault("_provider_blocked", {})[provider] = {"url": url, "detail": detail}


def headed_browser_available() -> bool:
    """Whether a headed Chromium can run in the current --browser-windows mode.

    Kept here for callers that import it; the decision lives in display.py.
    """
    from .display import headed_browser_available as _available
    return _available()


def fetch_atypon_supplements(ids, ctx) -> List[SupplementFile]:
    """Download SI from an Atypon publisher into a temp dir, as local files.

    Unlike every other enumerator this cannot return remote URLs for the
    engine to fetch: the URLs only work from inside the browser session that
    clicked them. So the files are downloaded here and handed back as
    `file://` entries the normal pipeline can take.
    """
    doi = getattr(ids, "doi", None)
    host = atypon_host_for(doi)
    if not host:
        return []
    if not headed_browser_available():
        ctx.log("    atypon SI: no display and no xvfb; skipping headed fetch")
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        ctx.log("    atypon SI: playwright not installed")
        return []

    # Headed Chromium needs a display: the private Xvfb when windows are hidden
    # (the default), the user's own when --browser-windows visible. Passed as
    # the launch's env, never set in os.environ, which other workers share.
    from .display import browser_launch_args, browser_launch_env
    env = browser_launch_env()
    if env is None:
        ctx.log("    atypon SI: no display for a headed browser")
        return []

    listing = _SUPPL_PATH.format(host=host, doi=doi)
    scratch = tempfile.mkdtemp(prefix=".fetchpdf-atypon-")
    files: List[SupplementFile] = []

    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                os.path.join(scratch, "profile"),
                headless=False,          # load-bearing; see module docstring
                env=env,
                accept_downloads=True,
                args=browser_launch_args(
                    ["--no-sandbox", "--disable-setuid-sandbox",
                     "--disable-blink-features=AutomationControlled"]),
                viewport={"width": 1400, "height": 900},
                user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
                response = page.goto(listing, wait_until="domcontentloaded",
                                     timeout=_PAGE_TIMEOUT_MS)
                status = response.status if response else None
                if status != 200:
                    ctx.log(f"    atypon SI: listing page HTTP {status}")
                    note_blocked(ctx, "atypon_suppl", listing, f"listing page HTTP {status}")
                    return []
                page.wait_for_timeout(_SETTLE_MS)

                links = page.eval_on_selector_all(
                    'a[href*="suppl_file"]', "els => els.map(e => e.href)")
                if not links:
                    ctx.log("    atypon SI: no suppl_file links on the listing page")
                    return []

                for index, href in enumerate(dict.fromkeys(links)):
                    if index >= _MAX_FILES:
                        ctx.log(f"    atypon SI: stopping at {_MAX_FILES} files")
                        break
                    name = href.rsplit("/", 1)[-1].split("?")[0] or f"si_{index+1}"
                    try:
                        # A click, not a navigation -- see the module docstring.
                        # Selected by nth match rather than by full href: the
                        # DOM href is often relative while eval returns it
                        # absolute, so an [href="<absolute>"] selector matches
                        # nothing and times out.
                        with page.expect_download(
                                timeout=_DOWNLOAD_TIMEOUT_MS) as download:
                            page.locator('a[href*="suppl_file"]').nth(index).click(
                                timeout=20000)
                        saved = os.path.join(scratch, name)
                        download.value.save_as(saved)
                    except Exception as error:
                        ctx.log(f"    atypon SI: {name}: "
                                f"{str(error).splitlines()[0][:80]}")
                        continue
                    if not os.path.exists(saved) or os.path.getsize(saved) == 0:
                        continue
                    files.append(SupplementFile(
                        name=name,
                        url="file://" + saved,
                        provider="atypon_suppl",
                        listing_index=index,
                        size_bytes=os.path.getsize(saved),
                        role=ROLE_SUPPLEMENT,
                        origin_doi=doi,
                        extra={"listing_url": listing, "local": saved,
                               "scratch_dir": scratch},
                    ))
                if files:
                    ctx.log(f"    atypon SI: {len(files)} file(s) via headed browser")
            finally:
                try:
                    context.close()
                except Exception:
                    pass
    except Exception as error:
        ctx.log(f"    atypon SI: {str(error).splitlines()[0][:100]}")

    if not files:
        # Nothing to keep the scratch dir for.
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        # The browser profile is large and is dead weight once the files are
        # downloaded; the files themselves must survive until the pipeline has
        # copied them out, so only the profile goes now.
        shutil.rmtree(os.path.join(scratch, "profile"), ignore_errors=True)
        ctx.scratch.setdefault("_atypon_scratch_dirs", []).append(scratch)
    return files


def cleanup_scratch(ctx) -> None:
    """Remove the temp dirs this provider downloaded into.

    Called once the pipeline has copied the files to their real homes. Kept
    separate from fetch because the `file://` entries must stay readable until
    then -- deleting on the way out would race the download.
    """
    for path in ctx.scratch.pop("_atypon_scratch_dirs", []) or []:
        shutil.rmtree(path, ignore_errors=True)
