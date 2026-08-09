from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from eta.logging import get_logger

__all__ = ["download"]

log = get_logger(__name__)

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_ATTEMPTS = 4
_CHUNK = 1 << 20


def download(
    url: str,
    dest: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Path:
    if dest.exists():
        log.info("download_skipped", url=url, path=str(dest))
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log.info("download_start", url=url, attempt=attempt)
            with urllib.request.urlopen(url, timeout=timeout_s) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh, _CHUNK)
            tmp.rename(dest)
            log.info("download_done", url=url, bytes=dest.stat().st_size)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            log.warning("download_failed", url=url, attempt=attempt, error=str(exc))
        else:
            return dest

    msg = f"download failed after {attempts} attempts: {url}"
    raise RuntimeError(msg) from last
