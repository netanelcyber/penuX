#!/usr/bin/env python3
"""Entry point for the PenuX webmail server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webmail import run

if __name__ == "__main__":
    run()
