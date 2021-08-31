'''Script related to the fetching of article data'''

from typing import Dict
import logging


logger = logging.getLogger(__name__)


def download_article(url: str) -> Dict[str, str]:
    '''Downloads an article.

    Args:
        url (str):
            The URL of the article.

    Returns:
        dict:
            A dictionary with keys 'title', 'content' and 'top_image', with
            associated values the title of the article, the content of the
            article and the URL to the top image of the article, respectively.
    '''
    pass
