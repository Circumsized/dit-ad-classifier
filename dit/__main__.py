"""Allow ``python -m dit`` to reach the same entry point as the ``dit`` script."""

from __future__ import annotations

from dit.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
