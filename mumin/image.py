'''Functions related to processing images'''

from typing import Union
from timeout_decorator import timeout, TimeoutError
import wget
from urllib.error import HTTPError, URLError
import cv2
from http.client import InvalidURL
import warnings


@timeout(5)
def download_image_with_timeout(url: str):
    return wget.download(url, bar=None)


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

        try:
            filename = download_image_with_timeout(url)
            pixel_array = cv2.imread(filename)
        except (ValueError, HTTPError, URLError, TimeoutError,
                OSError, InvalidURL, IndexError):
            return None

        if pixel_array is None:
            return None

        return dict(url=url, pixels=pixel_array, height=pixel_array.shape[0],
                    width=pixel_array.shape[1])
