'''Script related to media processing'''

from typing import Union
import logging
import numpy as np


logger = logging.getLogger(__name__)


def download_media(url: str) -> Union[np.ndarray, None]:
    '''Downloads an image or video.

    Args:
        url (str):
            The URL to the image or video.

    Returns:
        NumPy array or None:
            The image as an (image_size, 3) matrix, or video as an
            (video_length, image_size, 3). None if `url` does not
            point to an image or video.
    '''
    pass
