"""Compilation of the MuMin datasets."""

import logging
import os
import sys

from dotenv import load_dotenv

from src.mumin import MuminDataset

load_dotenv()


def compile_mumin(size: str = "small") -> None:
    """Compile the MuMiN dataset.

    Args:
        size (str):
            The size of the dataset to compile. Can be "small", "medium", or "large".
    """
    logging.info(f"Compiling {size} MuMin dataset...")
    bearer_token = str(os.getenv("TWITTER_API_KEY"))
    dataset = MuminDataset(twitter_bearer_token=bearer_token, size=size)
    dataset.compile()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compile_mumin(sys.argv[1])
    else:
        compile_mumin()
