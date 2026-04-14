"""Compatibility wrapper for the offline Qwen2-VL statistics builder."""

from tools.analysis.build_stats_qwen2vl_multi import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.analysis.build_stats_qwen2vl_multi import main

    main()
