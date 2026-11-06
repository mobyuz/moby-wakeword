"""Download licence-approved open data into D:\\moby-cache\\wakeword-data (never into the repo).

Thin wrapper over mobyww.data.fetch (install the package first: pip install -e . --no-deps).
Run with --help. MUSAN is only downloaded with --musan; Common Voice is never downloaded here.
"""

from mobyww.data.fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
