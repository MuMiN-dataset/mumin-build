"""Unit tests for the `article` module."""

import os

import pytest
from newspaper import Article

from mumin.article import download_article_with_timeout, process_article_url

WORKING_URL: str = "https://www.bbc.com/news/62261164"
BAD_GATEWAY_URL: str = (
    "https://www.diariodocentrodomundo.com.br/finlandia-tera-jornada-"
    "de-trabalho-de-seis-horas-quatro-dias-por-semana"
)
BIG_FILE_URL: str = (
    "https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/arxiv_data.db"
)


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[WORKING_URL, BAD_GATEWAY_URL],
    ids=["working_url", "bad_gateway_url"],
)
def test_download_article(url):
    article = Article(url)
    assert article.html == ""
    article = download_article_with_timeout(article)
    assert isinstance(article, Article)
    assert article.html != ""


@pytest.mark.skipif(os.name != "posix", reason="Linux and MacOS only")
def test_download_article_timeout_empty_html():
    article = Article(BIG_FILE_URL)
    assert article.html == ""
    download_article_with_timeout(article)
    assert article.html == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_download_article_timeout_exception():
    with pytest.raises(TimeoutError):
        article = Article(BIG_FILE_URL)
        download_article_with_timeout(article)


def test_process_article_url_bad_url():
    assert process_article_url(BIG_FILE_URL) is None


@pytest.mark.parametrize(
    argnames="article_config",
    argvalues=[
        dict(
            url=WORKING_URL,
            title="Capitol riot: Trump ignored pleas to condemn attack, hearing told",
            authors=[],
            publish_date=None,
            top_image_url="https://ichef.bbci.co.uk/news/1024/branded_news/11A62/"
            "production/_126009227_capitol_riot_2_getty.jpg",
        ),
        dict(
            url=BAD_GATEWAY_URL,
            title="Finlândia terá jornada de trabalho de seis horas, quatro "
            "dias por semana",
            authors=["Diario Do Centro Do Mundo", "Publicado Por"],
            publish_date="2020-01-06",
            top_image_url="https://www.diariodocentrodomundo.com.br/wp-content/"
            "uploads/2019/12/premie.jpg",
        ),
    ],
    ids=[
        "working_url",
        "bad_gateway_url",
    ],
    scope="class",
)
class TestProcessArticleUrl:
    @pytest.fixture(scope="class")
    def processed_article(self, article_config):
        yield process_article_url(article_config["url"])

    def test_is_dict(self, processed_article, article_config):
        assert isinstance(processed_article, dict)

    def test_url(self, processed_article, article_config):
        assert processed_article["url"] == article_config["url"]

    def test_title(self, processed_article, article_config):
        assert processed_article["title"] == article_config["title"]

    def test_authors(self, processed_article, article_config):
        assert processed_article["authors"] == article_config["authors"]

    def test_publish_date(self, processed_article, article_config):
        assert processed_article["publish_date"] == article_config["publish_date"]

    def test_top_image_url(self, processed_article, article_config):
        assert processed_article["top_image_url"] == article_config["top_image_url"]
