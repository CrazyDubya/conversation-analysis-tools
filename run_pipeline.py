#!/usr/bin/env python3
"""Compatibility entry point for the content analysis pipeline."""

from conversation_analysis.pipeline.cli import load_config, setup_logging, main


if __name__ == '__main__':
    main()
