"""Asyncio TCP server wrapping the IMAP session handler."""
from __future__ import annotations
import asyncio
import logging
import ssl
from typing import Optional

from .config import Config
from .protocol import IMAPSession

log = logging.getLogger(__name__)


def _build_ssl_context(cert: str, key: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    config: Config,
    tls_ctx: Optional[ssl.SSLContext],
):
    peer = writer.get_extra_info("peername")
    log.info("Connection from %s", peer)
    try:
        session = IMAPSession(reader, writer, config, tls_ctx)
        await session.run()
    except Exception:
        log.exception("Unhandled error for %s", peer)
    finally:
        log.info("Disconnected %s", peer)


async def run_server(config: Config):
    """Start IMAP (143) and IMAPS (993) listeners."""
    tls_ctx: Optional[ssl.SSLContext] = None
    if config.tls_available():
        tls_ctx = _build_ssl_context(config.tls_cert, config.tls_key)
        log.info("TLS enabled: cert=%s", config.tls_cert)
    else:
        log.warning("TLS certificate not found — STARTTLS/IMAPS disabled")

    servers = []

    # Plain IMAP (STARTTLS offered when cert is present)
    imap_server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, config, tls_ctx),
        host=config.host,
        port=config.port_imap,
    )
    servers.append(imap_server)
    log.info("IMAP listening on %s:%d", config.host, config.port_imap)

    # IMAPS (implicit TLS)
    if tls_ctx is not None:
        imaps_server = await asyncio.start_server(
            lambda r, w: _handle_client(r, w, config, None),  # already TLS
            host=config.host,
            port=config.port_imaps,
            ssl=tls_ctx,
        )
        servers.append(imaps_server)
        log.info("IMAPS listening on %s:%d", config.host, config.port_imaps)

    async with asyncio.TaskGroup() as tg:
        for srv in servers:
            tg.create_task(srv.serve_forever())
