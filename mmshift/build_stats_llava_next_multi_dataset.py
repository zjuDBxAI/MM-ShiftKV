"""Compatibility wrapper for the offline LLaVA-NeXT statistics builder."""

from tools.analysis.build_stats_llava_next_multi_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.analysis.build_stats_llava_next_multi_dataset import main

    main()
