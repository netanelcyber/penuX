#!/usr/bin/env python3
"""Entry point: python run_imap.py"""
import asyncio
import logging
import sys

from imap_server.config import Config
from imap_server.server import run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)


def main():
    config = Config()
    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
