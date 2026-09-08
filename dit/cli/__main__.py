"""Allow ``python -m dit.cli`` to reach the same entry point as ``dit``."""

from __future__ import annotations

from dit.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
