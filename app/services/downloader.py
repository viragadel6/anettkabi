"""Streaming media downloader with SSRF guards, hashing, and atomic temp writes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import get_settings
from app.constants import DOWNLOAD_REDIRECT_LIMIT, HTTP_USER_AGENT
from app.errors import ErrorCode, ServiceError
from app.utils.files import ensure_dir, remove_quietly
from app.utils.hashing import sha256_bytes

__all__ = ["DownloadResult", "Downloader", "decode_data_uri"]

_CHUNK = 1024 * 512


@dataclass(slots=True)
class DownloadResult:
    """Outcome of a successful download.

    Attributes:
        path: Local file path.
        sha256: Content digest.
        size_bytes: Bytes written.
        content_type: Effective MIME type from response headers.
        filename: Best-effort original filename.
    """

    path: Path
    sha256: str
    size_bytes: int
    content_type: str
    filename: str


def decode_data_uri(uri: str, destination: Path) -> tuple[str, int]:
    """Decode a `data:` URI to a file.

    Parameters:
        uri: The full data: URI.
        destination: Target path.

    Returns:
        (sha256, size_bytes).

    Raises:
        ServiceError: invalid_video_url for malformed URIs.
    """
    try:
        header, _, payload = uri[5:].partition(",")
        if not payload:
            raise ValueError("empty payload")
        if ";base64" in header:
            data = base64.b64decode(payload, validate=False)
        else:
            data = payload.encode("utf-8")
    except (ValueError, binascii.Error) as exc:
        raise ServiceError(ErrorCode.INVALID_VIDEO_URL, f"malformed data: URI: {exc}") from exc
    destination.write_bytes(data)
    return sha256_bytes(data), len(data)


class Downloader:
    """HTTP/data-URI fetcher enforcing size, timeout, and network safety."""

    def __init__(self, work_dir: Path) -> None:
        """Create a downloader bound to a scratch directory.

        Parameters:
            work_dir: Directory for temp files.
        """
        self._work_dir = ensure_dir(work_dir)

    async def download(
        self,
        url: str,
        *,
        destination: Path | None = None,
        max_bytes: int | None = None,
        timeout_s: float | None = None,
    ) -> DownloadResult:
        """Stream `url` to disk with all guards applied.

        Parameters:
            url: http(s) or data: URI.
            destination: Explicit destination path.
            max_bytes: Byte ceiling override.
            timeout_s: Timeout override.

        Returns:
            DownloadResult with digest and metadata.

        Raises:
            ServiceError: download_forbidden_host, download_failed, or video_too_large.
        """
        settings = get_settings()
        if url.startswith("data:"):
            return self._download_data_uri(url, destination)
        ceiling = max_bytes or settings.media.download_max_bytes
        timeout = timeout_s or settings.media.download_timeout_s
        current = httpx.URL(url)
        self._assert_scheme(current)
        target = destination or (self._work_dir / self._derive_name(current))
        self._prepare_target(target)
        digest = hashlib.sha256()
        content_type = "application/octet-stream"
        filename = current.path.rsplit("/", 1)[-1] or "video"
        redirects = 0
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={"user-agent": HTTP_USER_AGENT},
        ) as client:
            while True:
                self._assert_public(current.host, self._default_port(current))
                response = await client.send(client.build_request("GET", current))
                if response.has_redirect_location:
                    redirects += 1
                    if redirects > DOWNLOAD_REDIRECT_LIMIT:
                        raise ServiceError(
                            ErrorCode.DOWNLOAD_FAILED,
                            f"too many redirects (> {DOWNLOAD_REDIRECT_LIMIT})",
                        )
                    next_url = httpx.URL(response.headers["location"])
                    if next_url.is_relative_url:
                        next_url = current.join(next_url)
                    current = next_url
                    self._assert_scheme(current)
                    continue
                if response.status_code >= 400:
                    raise ServiceError(
                        ErrorCode.DOWNLOAD_FAILED,
                        f"remote server returned {response.status_code}",
                    )
                content_type = response.headers.get("content-type", content_type)
                disposition = response.headers.get("content-disposition", "")
                if "filename=" in disposition:
                    filename = disposition.split("filename=", 1)[1].strip('" ;')
                await self._write_body(response, target, digest, ceiling)
                break
        return DownloadResult(
            path=target,
            sha256=digest.hexdigest(),
            size_bytes=target.stat().st_size,
            content_type=content_type.split(";")[0].strip().lower(),
            filename=filename,
        )

    async def _write_body(
        self, response: httpx.Response, target: Path, digest: hashlib._Hash, ceiling: int
    ) -> None:
        """Stream one response body to disk under the byte ceiling.

        Parameters:
            response: The non-redirect response.
            target: Destination path.
            digest: Running SHA-256 accumulator.
            ceiling: Maximum bytes allowed.

        Raises:
            ServiceError: video_too_large or download_failed on I/O errors.
        """
        written = 0
        try:
            with target.open("wb") as handle:
                async for chunk in response.aiter_bytes(_CHUNK):
                    written += len(chunk)
                    if written > ceiling:
                        raise ServiceError(
                            ErrorCode.VIDEO_TOO_LARGE,
                            f"download exceeds {ceiling}-byte limit",
                        )
                    digest.update(chunk)
                    handle.write(chunk)
        except ServiceError:
            remove_quietly(target)
            raise
        except (httpx.HTTPError, OSError) as exc:
            remove_quietly(target)
            raise ServiceError(
                ErrorCode.DOWNLOAD_FAILED,
                f"failed to download media: {exc}",
            ) from exc

    def _download_data_uri(self, uri: str, destination: Path | None) -> DownloadResult:
        """Persist a data: URI to disk.

        Parameters:
            uri: The data: URI.
            destination: Optional explicit target.

        Returns:
            DownloadResult describing the file.
        """
        from app.utils.mime import decode_data_uri_suffix

        target = destination or (self._work_dir / f"upload-{abs(hash(uri))}{decode_data_uri_suffix(uri)}")
        self._prepare_target(target)
        digest, size = decode_data_uri(uri, target)
        return DownloadResult(
            path=target,
            sha256=digest,
            size_bytes=size,
            content_type=uri[5:].split(";", 1)[0] or "application/octet-stream",
            filename=f"upload{decode_data_uri_suffix(uri)}",
        )

    def _prepare_target(self, target: Path) -> Path:
        """Ensure a fresh target file path with parent dirs.

        Parameters:
            target: Path to prepare.

        Returns:
            The prepared path.
        """
        ensure_dir(target.parent)
        remove_quietly(target)
        return target

    @staticmethod
    def _derive_name(url: httpx.URL) -> str:
        """Derive a scratch filename from a URL.

        Parameters:
            url: Source URL.

        Returns:
            A filesystem-safe filename.
        """
        stem = url.path.rsplit("/", 1)[-1][:80] or "video"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        digest = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:12]
        return f"dl-{digest}-{safe}"

    @staticmethod
    def _default_port(url: httpx.URL) -> int:
        """Return the effective TCP port of a URL.

        Parameters:
            url: Parsed URL.

        Returns:
            Explicit port or scheme default.
        """
        if url.port is not None:
            return int(url.port)
        return 443 if url.scheme == "https" else 80

    def _assert_scheme(self, url: httpx.URL) -> None:
        """Enforce the configured scheme allowlist.

        Parameters:
            url: Parsed URL.

        Raises:
            ServiceError: invalid_video_url for disallowed schemes.
        """
        settings = get_settings()
        if url.scheme not in settings.media.download_allowed_schemes:
            raise ServiceError(
                ErrorCode.INVALID_VIDEO_URL,
                f"URL scheme {url.scheme!r} is not allowed; use one of "
                f"{settings.media.download_allowed_schemes}",
            )

    def _assert_public(self, host: str, port: int) -> None:
        """Reject non-public destinations (SSRF guard, per redirect hop).

        Parameters:
            host: Hostname or IP.
            port: Port number.

        Raises:
            ServiceError: download_forbidden_host for blocked addresses.
        """
        settings = get_settings()
        if not settings.media.download_block_private_cidrs:
            return
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise ServiceError(
                ErrorCode.DOWNLOAD_FORBIDDEN_HOST,
                f"cannot resolve host {host!r}",
            ) from exc
        for info in infos:
            address = info[4][0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise ServiceError(
                    ErrorCode.DOWNLOAD_FORBIDDEN_HOST,
                    f"host {host!r} resolves to a blocked address ({ip})",
                )
