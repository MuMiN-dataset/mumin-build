"""Functions related to processing images"""

import io
import time
import warnings
from typing import Union

import numpy as np
import requests
from PIL import Image
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout


@timeout(10)
def download_image_with_timeout(url: str) -> np.ndarray:
    while True:
        # Get the data from the URL, and try again if it fails
        response = requests.get(url)
        if response.status_code != 200:
            time.sleep(1)
            continue

        # Convert the data to a NumPy array
        byte_file = io.BytesIO(response.content)
        image = np.asarray(Image.open(byte_file))
        return image


def process_image_url(url: str) -> Union[None, dict]:
    """Process the URL and extract the article.

    Args:
        url (str): The URL.

    Returns:
        dict or None:
            The processed article, or None if the URL could not be parsed.
    """
    # Ignore warnings while processing images
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            image = download_image_with_timeout(url)
        except:  # noqa
            return None

        if image is None:
            return None
        else:
            try:
                return dict(
                    url=url, pixels=image, height=image.shape[0], width=image.shape[1]
                )
            except IndexError:
                return None
