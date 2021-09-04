'''Functions related to processing images'''

from typing import Union
from timeout_decorator import timeout, TimeoutError
import requests
from urllib.error import HTTPError, URLError
import numpy as np
from http.client import InvalidURL
import warnings


@timeout(5)
def download_image_with_timeout(url: str):
    while True:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            continue
        return np.asarray(bytearray(response.raw.read()), dtype='uint8')


def process_image_url(url: str) -> Union[None, dict]:
    '''Process the URL and extract the article.

    Args:
        url (str): The URL.

    Returns:
        dict or None:
            The processed article, or None if the URL could not be parsed.
    '''
    # Ignore warnings while processing articles
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        #try:
        pixel_array = download_image_with_timeout(url)
        #except (ValueError, HTTPError, URLError, TimeoutError,
        #        OSError, InvalidURL, IndexError):
        #    return None

        return dict(url=url, pixels=pixel_array, height=pixel_array.shape[0],
                    width=pixel_array.shape[1])
