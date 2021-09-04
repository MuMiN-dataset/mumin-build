'''Functions related to processing images'''

from typing import Union
import requests
from requests.exceptions import ConnectionError
from timeout_decorator import timeout, TimeoutError
import numpy as np
import warnings
import time
import cv2


@timeout(5)
def download_image_with_timeout(url: str):
    while True:
        # Get the data from the URL, and try again if it fails
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            time.sleep(1)
            continue

        # Read the response as a raw array of bytes
        byte_arr = bytearray(response.raw.read())
        return byte_arr


def process_image_url(url: str) -> Union[None, dict]:
    '''Process the URL and extract the article.

    Args:
        url (str): The URL.

    Returns:
        dict or None:
            The processed article, or None if the URL could not be parsed.
    '''
    # Ignore warnings while processing images
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        try:
            byte_arr = download_image_with_timeout(url)
        except (TimeoutError, ConnectionError):
            return None

        # Read byte array as a one-dimensional numpy array of unsigned integers
        onedim_arr = np.asarray(byte_arr, dtype='uint8')

        # Convert the array to (pixels, channels) matrix
        image = cv2.imdecode(onedim_arr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        # `cv2.imdecode` converted array into BGR format, convert it to RGB
        # format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            return None
        else:
            return dict(url=url,
                        pixels=image,
                        height=image.shape[0],
                        width=image.shape[1])
