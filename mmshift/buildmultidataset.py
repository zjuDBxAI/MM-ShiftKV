"""Compatibility wrapper for the offline multi-dataset builder script."""

from tools.analysis.buildmultidataset import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.analysis.buildmultidataset import main

    main()
