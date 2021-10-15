'''Functions related to processing articles'''

from typing import Union
from newspaper import Article, ArticleException
from timeout_decorator import timeout, TimeoutError
from urllib3.exceptions import ReadTimeoutError
import re
import datetime as dt
import warnings


@timeout(5)
def download_article_with_timeout(article: Article):
    article.download()
    return article


def process_article_url(url: str) -> Union[None, dict]:
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

        # Remove GET arguments from the URL
        stripped_url = re.sub(r'(\?.*"|\/$)', '', url)

        try:
            article = Article(stripped_url)
            article = download_article_with_timeout(article)
            article.parse()
        except (ArticleException, ValueError, ReadTimeoutError,
                RuntimeError, TimeoutError):
            return None

        # Extract the title and skip URL if it is empty
        title = article.title
        if title == '':
            return None
        else:
            title = re.sub('\n+', '\n', title)
            title = re.sub(' +', ' ', title)
            title = title.strip()

        # Extract the content and skip URL if it is empty
        content = article.text.strip()
        if content == '':
            return None
        else:
            content = re.sub('\n+', '\n', content)
            content = re.sub(' +', ' ', content)
            content = content.strip()

        # Extract the authors, the publishing date and the top image
        authors = list(article.authors)
        if article.publish_date is not None:
            date = article.publish_date
            publish_date = dt.datetime.strftime(date, '%Y-%m-%d')
        else:
            publish_date = None
        try:
            top_image_url = article.top_image
        except AttributeError:
            top_image_url = None

        return dict(url=stripped_url, title=title, content=content,
                    authors=authors, publish_date=publish_date,
                    top_image_url=top_image_url)
