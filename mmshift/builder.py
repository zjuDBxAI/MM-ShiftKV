"""Compatibility wrapper for the offline statistics builder script."""

from tools.analysis.builder import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.analysis.builder import main

    main()
