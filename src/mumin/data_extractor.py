"""Extract node and relation data from the rehydrated Twitter data"""

import multiprocessing as mp
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .article import process_article_url
from .image import process_image_url


class DataExtractor:
    """Extract node and relation data from the rehydrated Twitter data.

    Args:
        include_replies (bool):
            Whether to include replies.
        include_articles (bool):
            Whether to include articles.
        include_tweet_images (bool):
            Whether to include tweet images.
        include_extra_images (bool):
            Whether to include extra images.
        include_hashtags (bool):
            Whether to include hashtags.
        include_mentions (bool):
            Whether to include mentions.
        n_jobs (int):
            The number of parallel jobs to use.
        chunksize (int):
            The number of samples to process at a time.

    Attributes:
        include_replies (bool):
            Whether to include replies.
        include_articles (bool):
            Whether to include articles.
        include_tweet_images (bool):
            Whether to include tweet images.
        include_extra_images (bool):
            Whether to include extra images.
        include_hashtags (bool):
            Whether to include hashtags.
        include_mentions (bool):
            Whether to include mentions.
    """

    def __init__(
        self,
        include_replies: bool,
        include_articles: bool,
        include_tweet_images: bool,
        include_extra_images: bool,
        include_hashtags: bool,
        include_mentions: bool,
        n_jobs: int,
        chunksize: int,
    ):
        self.include_replies = include_replies
        self.include_articles = include_articles
        self.include_tweet_images = include_tweet_images
        self.include_extra_images = include_extra_images
        self.include_hashtags = include_hashtags
        self.include_mentions = include_mentions
        self.n_jobs = n_jobs
        self.chunksize = chunksize

    def extract_all(
        self,
        nodes: Dict[str, pd.DataFrame],
        rels: Dict[Tuple[str, str, str], pd.DataFrame],
    ) -> Tuple[dict, dict]:
        """Extract all node and relation data.

        Args:
            nodes (Dict[str, pd.DataFrame]):
                A dictionary of node dataframes.
            rels (Dict[Tuple[str, str, str], pd.DataFrame]):
                A dictionary of relation dataframes.

        Returns:
            pair of dicts:
                A tuple of updated node and relation dictionaries.
        """
        # Extract data directly from the rehydrated data
        nodes["hashtag"] = self._extract_hashtags(
            tweet_df=nodes["tweet"], user_df=nodes["user"]
        )
        rels[("user", "posted", "tweet")] = self._extract_user_posted_tweet(
            tweet_df=nodes["tweet"], user_df=nodes["user"]
        )
        rels[("user", "posted", "reply")] = self._extract_user_posted_reply(
            reply_df=nodes["reply"], user_df=nodes["user"]
        )
        rels[("tweet", "mentions", "user")] = self._extract_tweet_mentions_user(
            tweet_df=nodes["tweet"], user_df=nodes["user"]
        )
        rels[("user", "mentions", "user")] = self._extract_user_mentions_user(
            user_df=nodes["user"]
        )

        # Extract data relying on hashtag data
        rels[
            ("tweet", "has_hashtag", "hashtag")
        ] = self._extract_tweet_has_hashtag_hashtag(
            tweet_df=nodes["tweet"], hashtag_df=nodes["hashtag"]
        )
        rels[
            ("user", "has_hashtag", "hashtag")
        ] = self._extract_user_has_hashtag_hashtag(
            user_df=nodes["user"], hashtag_df=nodes["hashtag"]
        )

        # Extract data relying on the pre-extracted article and image data, as well has
        # the (:User)-[:POSTED]->(:Tweet) relation
        nodes["url"] = self._extract_urls(
            tweet_dicusses_claim_df=rels[("tweet", "discusses", "claim")],
            user_posted_tweet_df=rels[("user", "posted", "tweet")],
            user_df=nodes["user"],
            tweet_df=nodes["tweet"],
            article_df=nodes.get("article"),
            image_df=nodes.get("image"),
        )

        # Extract data relying on the URL data without the URLs from the articles
        nodes["article"] = self._extract_articles(
            url_df=nodes["url"],
        )

        # Update the URLs with URLs from the articles
        nodes["url"] = self._update_urls_from_articles(
            url_df=nodes["url"], article_df=nodes["article"]
        )

        # Extract data relying on url data
        rels[("user", "has_url", "url")] = self._extract_user_has_url_url(
            user_df=nodes["user"], url_df=nodes["url"]
        )
        rels[
            ("user", "has_profile_picture_url", "url")
        ] = self._extract_user_has_profile_picture_url_url(
            user_df=nodes["user"], url_df=nodes["url"]
        )

        # Extract data relying on url and pre-extracted image data
        rels[("tweet", "has_url", "url")] = self._extract_tweet_has_url_url(
            tweet_df=nodes["tweet"], url_df=nodes["url"], image_df=nodes.get("image")
        )

        # Extract data relying on article and url data
        nodes["image"] = self._extract_images(
            url_df=nodes["url"], article_df=nodes["article"]
        )
        rels[
            ("article", "has_top_image_url", "url")
        ] = self._extract_article_has_top_image_url_url(
            article_df=nodes["article"], url_df=nodes["url"]
        )

        # Extract data relying on article and url data, as well has the
        # (:Tweet)-[:HAS_URL]->(:Url) relation
        rels[
            ("tweet", "has_article", "article")
        ] = self._extract_tweet_has_article_article(
            tweet_has_url_url_df=rels[("tweet", "has_url", "url")],
            article_df=nodes["article"],
            url_df=nodes["url"],
        )

        # Extract data relying on image and url data, as well has the
        # (:Tweet)-[:HAS_URL]->(:Url) relation
        rels[("tweet", "has_image", "image")] = self._extract_tweet_has_image_image(
            tweet_has_url_url_df=rels[("tweet", "has_url", "url")],
            url_df=nodes["url"],
            image_df=nodes["image"],
        )

        # Extract data relying on article, image and url data, as well has the
        # (:Article)-[:HAS_TOP_IMAGE_URL]->(:Url) relation
        top_image_url_rel = rels[("article", "has_top_image_url", "url")]
        rels[
            ("article", "has_top_image", "image")
        ] = self._extract_article_has_top_image_image(
            article_has_top_image_url_url_df=top_image_url_rel,
            url_df=nodes["url"],
            image_df=nodes["image"],
        )

        # Extract data relying on image and url data, as well has the
        # (:User)-[:HAS_PROFILE_PICTURE_URL]->(:Url) relation
        has_profile_pic_df = rels[("user", "has_profile_picture_url", "url")]
        rels[
            ("user", "has_profile_picture", "image")
        ] = self._extract_user_has_profile_picture_image(
            user_has_profile_picture_url_url_df=has_profile_pic_df,
            url_df=nodes["url"],
            image_df=nodes["image"],
        )

        return nodes, rels

    def _extract_user_posted_tweet(
        self, tweet_df: pd.DataFrame, user_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:User)-[:POSTED]->(:Tweet) relation data.

        Args:
            tweet_df (pd.DataFrame):
                A dataframe of tweets.
            user_df (pd.DataFrame):
                A dataframe of users.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        if len(tweet_df) and len(user_df):
            merged = (
                tweet_df[["author_id"]]
                .dropna()
                .reset_index()
                .rename(columns=dict(index="tweet_idx"))
                .astype({"author_id": np.uint64})
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx")),
                    left_on="author_id",
                    right_on="user_id",
                )
            )
            data_dict = dict(
                src=merged.user_idx.tolist(), tgt=merged.tweet_idx.tolist()
            )
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_user_posted_reply(
        self, reply_df: pd.DataFrame, user_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:User)-[:POSTED]->(:Reply) relation data.

        Args:
            reply_df (pd.DataFrame):
                A dataframe of replies.
            user_df (pd.DataFrame):
                A dataframe of users.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        if self.include_replies and len(reply_df) and len(user_df):
            merged = (
                reply_df[["author_id"]]
                .dropna()
                .reset_index()
                .rename(columns=dict(index="reply_idx"))
                .astype({"author_id": np.uint64})
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx")),
                    left_on="author_id",
                    right_on="user_id",
                )
            )
            data_dict = dict(
                src=merged.user_idx.tolist(), tgt=merged.reply_idx.tolist()
            )
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_tweet_mentions_user(
        self, tweet_df: pd.DataFrame, user_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:Tweet)-[:MENTIONS]->(:User) relation data.

        Args:
            tweet_df (pd.DataFrame):
                A dataframe of tweets.
            user_df (pd.DataFrame):
                A dataframe of users.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        mentions_exist = "entities.mentions" in tweet_df.columns
        if self.include_mentions and mentions_exist and len(tweet_df) and len(user_df):

            def extract_mention(dcts: List[dict]) -> List[Union[int, str]]:
                """Extracts user ids from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[Union[int, str]]:
                        A list of user ids.
                """
                return [dct["id"] for dct in dcts]

            merged = (
                tweet_df[["entities.mentions"]]
                .dropna()
                .applymap(extract_mention)
                .reset_index()
                .rename(columns=dict(index="tweet_idx"))
                .explode("entities.mentions")
                .astype({"entities.mentions": np.uint64})
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx")),
                    left_on="entities.mentions",
                    right_on="user_id",
                )
            )
            data_dict = dict(
                src=merged.tweet_idx.tolist(), tgt=merged.user_idx.tolist()
            )
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_user_mentions_user(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """Extract (:User)-[:MENTIONS]->(:User) relation data.

        Args:
            user_df (pd.DataFrame):
                A dataframe of users.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        user_cols = user_df.columns
        mentions_exist = "entities.description.mentions" in user_cols
        if self.include_mentions and mentions_exist and len(user_df):

            def extract_mention(dcts: List[dict]) -> List[str]:
                """Extracts user ids from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[str]:
                        A list of user ids.
                """
                return [dct["username"] for dct in dcts]

            merged = (
                user_df[["entities.description.mentions"]]
                .dropna()
                .applymap(extract_mention)
                .reset_index()
                .rename(columns=dict(index="user_idx1"))
                .explode("entities.description.mentions")
                .merge(
                    user_df[["username"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx2")),
                    left_on="entities.description.mentions",
                    right_on="username",
                )
            )
            data_dict = dict(
                src=merged.user_idx1.tolist(), tgt=merged.user_idx2.tolist()
            )
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_tweet_has_hashtag_hashtag(
        self, tweet_df: pd.DataFrame, hashtag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:Tweet)-[:HAS_HASHTAG]->(:Hashtag) relation data.

        Args:
            tweet_df (pd.DataFrame):
                A dataframe of tweets.
            hashtag_df (pd.DataFrame):
                A dataframe of hashtags.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        hashtags_exist = "entities.hashtags" in tweet_df.columns
        if (
            self.include_hashtags
            and hashtags_exist
            and len(tweet_df)
            and len(hashtag_df)
        ):

            def extract_hashtag(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (list of dict):
                        A list of dictionaries.

                Returns:
                    list of str or None:
                        A list of hashtags.
                """
                return [dct.get("tag") for dct in dcts]

            merged = (
                tweet_df[["entities.hashtags"]]
                .dropna()
                .applymap(extract_hashtag)
                .reset_index()
                .rename(columns=dict(index="tweet_idx"))
                .explode("entities.hashtags")
                .merge(
                    hashtag_df[["tag"]]
                    .reset_index()
                    .rename(columns=dict(index="tag_idx")),
                    left_on="entities.hashtags",
                    right_on="tag",
                )
            )
            data_dict = dict(src=merged.tweet_idx.tolist(), tgt=merged.tag_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_user_has_hashtag_hashtag(
        self, user_df: pd.DataFrame, hashtag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:User)-[:HAS_HASHTAG]->(:Hashtag) relation data.

        Args:
            user_df (pd.DataFrame):
                A dataframe of users.
            hashtag_df (pd.DataFrame):
                A dataframe of hashtags.

        Returns:
            pd.DataFrame: A dataframe of relations.
        """
        user_cols = user_df.columns
        hashtags_exist = "entities.description.hashtags" in user_cols
        if (
            self.include_hashtags
            and hashtags_exist
            and len(user_df)
            and len(hashtag_df)
        ):

            def extract_hashtag(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (list of dict):
                        A list of dictionaries.

                Returns:
                    list of str or None:
                        A list of hashtags.
                """
                return [dct.get("tag") for dct in dcts]

            merged = (
                user_df[["entities.description.hashtags"]]
                .dropna()
                .applymap(extract_hashtag)
                .reset_index()
                .rename(columns=dict(index="user_idx"))
                .explode("entities.description.hashtags")
                .merge(
                    hashtag_df[["tag"]]
                    .reset_index()
                    .rename(columns=dict(index="tag_idx")),
                    left_on="entities.description.hashtags",
                    right_on="tag",
                )
            )
            data_dict = dict(src=merged.user_idx.tolist(), tgt=merged.tag_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_tweet_has_url_url(
        self,
        tweet_df: pd.DataFrame,
        url_df: pd.DataFrame,
        image_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract (:Tweet)-[:HAS_URL]->(:Url) relation data.

        Args:
            tweet_df (pd.DataFrame):
                A dataframe of tweets.
            url_df (pd.DataFrame):
                A dataframe of urls.
            image_df (pd.DataFrame or None, optional):
                A dataframe of images, or None if it is not available. Defaults
                to None.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        urls_exist = (
            "entities.urls" in tweet_df.columns
            or "attachments.media_keys" in tweet_df.columns
        )
        include_images = self.include_tweet_images or self.include_extra_images
        if (
            (self.include_articles or include_images)
            and urls_exist
            and len(tweet_df)
            and len(url_df)
        ):

            # Add the urls from the tweets themselves
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[str]:
                        A list of urls.
                """
                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            merged = (
                tweet_df[["entities.urls"]]
                .dropna()
                .applymap(extract_url)
                .reset_index()
                .rename(columns=dict(index="tweet_idx"))
                .explode("entities.urls")
                .merge(
                    url_df[["url"]].reset_index().rename(columns=dict(index="ul_idx")),
                    left_on="entities.urls",
                    right_on="url",
                )
            )
            data_dict = dict(src=merged.tweet_idx.tolist(), tgt=merged.ul_idx.tolist())
            rel_df = pd.DataFrame(data_dict)

            # Append the urls from the images
            if image_df is not None and len(image_df) and include_images:
                merged = (
                    tweet_df.reset_index()
                    .rename(columns=dict(index="tweet_idx"))
                    .explode("attachments.media_keys")
                    .merge(
                        image_df[["media_key", "url"]],
                        left_on="attachments.media_keys",
                        right_on="media_key",
                    )
                    .merge(
                        url_df[["url"]]
                        .reset_index()
                        .rename(columns=dict(index="ul_idx")),
                        on="url",
                    )
                )
                data_dict = dict(
                    src=merged.tweet_idx.tolist(), tgt=merged.ul_idx.tolist()
                )
                rel_df = pd.concat((rel_df, pd.DataFrame(data_dict))).drop_duplicates()

            return rel_df

    def _extract_user_has_url_url(
        self, user_df: pd.DataFrame, url_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:User)-[:HAS_URL]->(:Url) relation data.

        Args:
            user_df (pd.DataFrame):
                A dataframe of users.
            url_df (pd.DataFrame):
                A dataframe of urls.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        user_cols = user_df.columns
        url_urls_exist = "entities.url.urls" in user_cols
        desc_urls_exist = "entities.description.urls" in user_cols
        if url_urls_exist or desc_urls_exist and len(user_df) and len(url_df):

            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (list of dict):
                        A list of dictionaries.

                Returns:
                    list of str or None:
                        A list of urls.
                """

                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            # Initialise empty relation, which will be populated below
            rel_df = pd.DataFrame()

            if url_urls_exist:
                merged = (
                    user_df[["entities.url.urls"]]
                    .dropna()
                    .applymap(extract_url)
                    .reset_index()
                    .rename(columns=dict(index="user_idx"))
                    .explode("entities.url.urls")
                    .merge(
                        url_df[["url"]]
                        .reset_index()
                        .rename(columns=dict(index="ul_idx")),
                        left_on="entities.url.urls",
                        right_on="url",
                    )
                )
                data_dict = dict(
                    src=merged.user_idx.tolist(), tgt=merged.ul_idx.tolist()
                )
                rel_df = (
                    pd.concat((rel_df, pd.DataFrame(data_dict)), axis=0)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

            if desc_urls_exist:
                merged = (
                    user_df[["entities.description.urls"]]
                    .dropna()
                    .applymap(extract_url)
                    .reset_index()
                    .rename(columns=dict(index="user_idx"))
                    .explode("entities.description.urls")
                    .merge(
                        url_df[["url"]]
                        .reset_index()
                        .rename(columns=dict(index="ul_idx")),
                        left_on="entities.description.urls",
                        right_on="url",
                    )
                )
                data_dict = dict(
                    src=merged.user_idx.tolist(), tgt=merged.ul_idx.tolist()
                )
                rel_df = (
                    pd.concat((rel_df, pd.DataFrame(data_dict)), axis=0)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

            return rel_df
        else:
            return pd.DataFrame()

    def _extract_user_has_profile_picture_url_url(
        self, user_df: pd.DataFrame, url_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:User)-[:HAS_PROFILE_PICTURE]->(:Url) relation data.

        Args:
            user_df (pd.DataFrame):
                A dataframe of users.
            url_df (pd.DataFrame):
                A dataframe of urls.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        user_cols = user_df.columns
        profile_images_exist = "profile_image_url" in user_cols
        if (
            self.include_extra_images
            and profile_images_exist
            and len(user_df)
            and len(url_df)
        ):
            merged = (
                user_df[["profile_image_url"]]
                .dropna()
                .reset_index()
                .rename(columns=dict(index="user_idx"))
                .merge(
                    url_df[["url"]].reset_index().rename(columns=dict(index="ul_idx")),
                    left_on="profile_image_url",
                    right_on="url",
                )
            )
            data_dict = dict(src=merged.user_idx.tolist(), tgt=merged.ul_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_articles(self, url_df: pd.DataFrame) -> pd.DataFrame:
        """Extract (:Article) nodes.

        Args:
            url_df (pd.DataFrame):
                A dataframe of urls.

        Returns:
            pd.DataFrame:
                A dataframe of articles.
        """
        if self.include_articles and len(url_df):
            # Create regex that filters out non-articles. These are common
            # images, videos and social media websites
            non_article_regexs = [
                "youtu[.]*be",
                "vimeo",
                "spotify",
                "twitter",
                "instagram",
                "tiktok",
                "gab[.]com",
                "https://t[.]me",
                "imgur",
                "/photo/",
                "mp4",
                "mov",
                "jpg",
                "jpeg",
                "bmp",
                "png",
                "gif",
                "pdf",
            ]
            non_article_regex = "(" + "|".join(non_article_regexs) + ")"

            # Filter out the URLs to get the potential article URLs
            article_urls = [
                url
                for url in url_df.url.tolist()
                if re.search(non_article_regex, url) is None
            ]

            # Loop over all the Url nodes
            data_dict = defaultdict(list)
            with mp.Pool(processes=self.n_jobs) as pool:
                for result in tqdm(
                    pool.imap_unordered(
                        process_article_url, article_urls, chunksize=self.chunksize
                    ),
                    desc="Parsing articles",
                    total=len(article_urls),
                ):

                    # Skip result if URL is not parseable
                    if result is None:
                        continue

                    # Store the data in the data dictionary
                    data_dict["url"].append(result["url"])
                    data_dict["title"].append(result["title"])
                    data_dict["content"].append(result["content"])
                    data_dict["authors"].append(result["authors"])
                    data_dict["publish_date"].append(result["publish_date"])
                    data_dict["top_image_url"].append(result["top_image_url"])

            # Convert the data dictionary to a dataframe and return it
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_article_has_top_image_url_url(
        self, article_df: pd.DataFrame, url_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:Article)-[:HAS_TOP_IMAGE_URL]->(:Url) relation data.

        Args:
            article_df (pd.DataFrame):
                A dataframe of articles.
            url_df (pd.DataFrame):
                A dataframe of urls.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        if (
            self.include_articles
            and self.include_extra_images
            and len(article_df)
            and len(url_df)
        ):

            # Create relation
            merged = (
                article_df[["top_image_url"]]
                .dropna()
                .reset_index()
                .rename(columns=dict(index="art_idx"))
                .merge(
                    url_df[["url"]].reset_index().rename(columns=dict(index="ul_idx")),
                    left_on="top_image_url",
                    right_on="url",
                )
            )
            data_dict = dict(src=merged.art_idx.tolist(), tgt=merged.ul_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_tweet_has_article_article(
        self,
        tweet_has_url_url_df: pd.DataFrame,
        article_df: pd.DataFrame,
        url_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract (:Tweet)-[:HAS_ARTICLE]->(:Article) relation data.

        Args:
            tweet_has_url_url_df (pd.DataFrame):
                A dataframe of relations.
            article_df (pd.DataFrame):
                A dataframe of articles.
            url_df (pd.DataFrame):
                A dataframe of urls.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        if (
            self.include_articles
            and len(tweet_has_url_url_df)
            and len(article_df)
            and len(url_df)
        ):
            merged = (
                tweet_has_url_url_df.rename(columns=dict(src="tweet_idx", tgt="ul_idx"))
                .merge(
                    url_df[["url"]].reset_index().rename(columns=dict(index="ul_idx")),
                    on="ul_idx",
                )
                .merge(
                    article_df[["url"]]
                    .reset_index()
                    .rename(columns=dict(index="art_idx")),
                    on="url",
                )
            )
            data_dict = dict(src=merged.tweet_idx.tolist(), tgt=merged.art_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_images(
        self, url_df: pd.DataFrame, article_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:Image) nodes.

        Args:
            url_df (pd.DataFrame):
                A dataframe of urls.
            article_df (pd.DataFrame):
                A dataframe of articles.

        Returns:
            pd.DataFrame:
                A dataframe of images.
        """
        if self.include_tweet_images or self.include_extra_images and len(url_df):

            # If there are no articles then set `article_df` to have an empty `url`
            # column
            if not len(article_df):
                article_df = pd.DataFrame(columns=["url"])

            # Start with all the URLs that have not already been parsed as articles
            image_urls = [
                url for url in url_df.url.tolist() if url not in article_df.url.tolist()
            ]

            # Filter the resulting list of URLs using a hardcoded list of image formats
            regex = "|".join(
                [
                    "png",
                    "jpg",
                    "jpeg",
                    "bmp",
                    "pdf",
                    "jfif",
                    "tiff",
                    "ppm",
                    "pgm",
                    "pbm",
                    "pnm",
                    "webp",
                    "hdr",
                    "heif",
                ]
            )
            image_urls = [
                url for url in image_urls if re.search(regex, url) is not None
            ]

            # Loop over all the Url nodes
            data_dict = defaultdict(list)
            with mp.Pool(processes=self.n_jobs) as pool:
                for result in tqdm(
                    pool.imap_unordered(
                        process_image_url, image_urls, chunksize=self.chunksize
                    ),
                    desc="Parsing images",
                    total=len(image_urls),
                ):

                    # Store the data in the data dictionary if it was parseable
                    if (
                        result is not None
                        and len(result["pixels"].shape) == 3
                        and result["pixels"].shape[2] == 3
                    ):
                        data_dict["url"].append(result["url"])
                        data_dict["pixels"].append(result["pixels"])
                        data_dict["height"].append(result["height"])
                        data_dict["width"].append(result["width"])

            # Convert the data dictionary to a dataframe and store it as the `Image`
            # node.
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_tweet_has_image_image(
        self,
        tweet_has_url_url_df: pd.DataFrame,
        url_df: pd.DataFrame,
        image_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract (:Tweet)-[:HAS_IMAGE]->(:Image) relation data.

        Args:
            tweet_has_url_url_df (pd.DataFrame):
                A dataframe of relations.
            url_df (pd.DataFrame):
                A dataframe of urls.
            image_df (pd.DataFrame):
                A dataframe of images.

        Returns:
            pd.DataFrame:
                A dataframe of relations.
        """
        if (
            len(tweet_has_url_url_df)
            and len(url_df)
            and len(image_df)
            and self.include_tweet_images
        ):
            url_idx = dict(index="ul_idx")
            img_idx = dict(index="im_idx")
            if len(tweet_has_url_url_df):
                merged = (
                    tweet_has_url_url_df.rename(
                        columns=dict(src="tweet_idx", tgt="ul_idx")
                    )
                    .merge(
                        url_df[["url"]].reset_index().rename(columns=url_idx),
                        on="ul_idx",
                    )
                    .merge(
                        image_df[["url"]].reset_index().rename(columns=img_idx),
                        on="url",
                    )
                )
                data_dict = dict(
                    src=merged.tweet_idx.tolist(), tgt=merged.im_idx.tolist()
                )
                return pd.DataFrame(data_dict)
            else:
                return pd.DataFrame()

    def _extract_article_has_top_image_image(
        self,
        article_has_top_image_url_url_df: pd.DataFrame,
        url_df: pd.DataFrame,
        image_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract (:Article)-[:HAS_TOP_IMAGE]->(:Image) relation data.

        Args:
            article_has_top_image_url_url_df (pd.DataFrame):
                A dataframe of relations.
            url_df (pd.DataFrame):
                A dataframe of urls.
            image_df (pd.DataFrame):
                A dataframe of images.

        Returns:
            pd.DataFrame: A dataframe of relations.
        """
        if (
            self.include_articles
            and self.include_extra_images
            and len(article_has_top_image_url_url_df)
            and len(url_df)
            and len(image_df)
        ):
            url_idx = dict(index="ul_idx")
            img_idx = dict(index="im_idx")
            if self.include_articles:
                merged = (
                    article_has_top_image_url_url_df.rename(
                        columns=dict(src="art_idx", tgt="ul_idx")
                    )
                    .merge(
                        url_df[["url"]].reset_index().rename(columns=url_idx),
                        on="ul_idx",
                    )
                    .merge(
                        image_df[["url"]].reset_index().rename(columns=img_idx),
                        on="url",
                    )
                )
                data_dict = dict(
                    src=merged.art_idx.tolist(), tgt=merged.im_idx.tolist()
                )
                return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_user_has_profile_picture_image(
        self,
        user_has_profile_picture_url_url_df: pd.DataFrame,
        url_df: pd.DataFrame,
        image_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract (:User)-[:HAS_PROFILE_PICTURE]->(:Image) relation data.

        Args:
            user_has_profile_picture_url_url_df (pd.DataFrame):
                A dataframe of relations.
            url_df (pd.DataFrame):
                A dataframe of urls.
            image_df (pd.DataFrame):
                A dataframe of images.

        Returns:
            pd.DataFrame: A dataframe of relations.
        """
        if (
            self.include_extra_images
            and len(image_df)
            and len(url_df)
            and len(user_has_profile_picture_url_url_df)
        ):
            merged = (
                user_has_profile_picture_url_url_df.rename(
                    columns=dict(src="user_idx", tgt="ul_idx")
                )
                .merge(
                    url_df[["url"]].reset_index().rename(columns=dict(index="ul_idx")),
                    on="ul_idx",
                )
                .merge(
                    image_df[["url"]]
                    .reset_index()
                    .rename(columns=dict(index="im_idx")),
                    on="url",
                )
            )
            data_dict = dict(src=merged.user_idx.tolist(), tgt=merged.im_idx.tolist())
            return pd.DataFrame(data_dict)
        else:
            return pd.DataFrame()

    def _extract_hashtags(
        self, tweet_df: pd.DataFrame, user_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract (:Hashtag) node data.

        Args:
            tweet_df (pd.DataFrame):
                A dataframe of tweets.
            user_df (pd.DataFrame):
                A dataframe of users.

        Returns:
            pd.DataFrame:
                A dataframe of hashtags.
        """
        if self.include_hashtags and len(tweet_df) and len(user_df):

            # Initialise the hashtag dataframe
            hashtag_df = pd.DataFrame()

            def extract_hashtag(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (list of dict):
                        A list of dictionaries.

                Returns:
                    list of str or None:
                        A list of hashtags.
                """
                return [dct.get("tag") for dct in dcts]

            # Add hashtags from tweets
            if "entities.hashtags" in tweet_df.columns:
                hashtags = (
                    tweet_df["entities.hashtags"]
                    .dropna()
                    .map(extract_hashtag)
                    .explode()
                    .tolist()
                )
                hashtag_dict = pd.DataFrame(dict(tag=hashtags))
                hashtag_df = (
                    pd.concat((hashtag_df, pd.DataFrame(hashtag_dict)), axis=0)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

            # Add hashtags from users
            if "entities.description.hashtags" in user_df.columns:
                hashtags = (
                    user_df["entities.description.hashtags"]
                    .dropna()
                    .map(extract_hashtag)
                    .explode()
                    .tolist()
                )
                hashtag_dict = pd.DataFrame(dict(tag=hashtags))
                hashtag_df = (
                    pd.concat((hashtag_df, pd.DataFrame(hashtag_dict)), axis=0)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

            return hashtag_df
        else:
            return pd.DataFrame()

    def _extract_urls(
        self,
        tweet_dicusses_claim_df: pd.DataFrame,
        user_posted_tweet_df: pd.DataFrame,
        user_df: pd.DataFrame,
        tweet_df: pd.DataFrame,
        article_df: Optional[pd.DataFrame] = None,
        image_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract (:Url) node data.

        Note that this does not extract the top image urls from the articles.
        This will be added with the _update_urls_from_articles method, after
        the articles have been extracted.

        Args:
            tweet_dicusses_claim_df (pd.DataFrame):
                A dataframe of tweet_dicusses_claim nodes.
            user_posted_tweet_df (pd.DataFrame):
                A dataframe of user_posted_tweet nodes.
            user_df (pd.DataFrame):
                A dataframe of user nodes.
            tweet_df (pd.DataFrame):
                A dataframe of tweet nodes.
            article_df (pd.DataFrame or None, optional):
                A dataframe of article nodes, of None if no articles are
                available. Defaults to None.
            image_df (pd.DataFrame, or None, optional):
                A dataframe of image nodes, or None if no images are available.
                Defaults to None.

        Returns:
            pd.DataFrame: A dataframe of url nodes.
        """
        if (
            len(tweet_dicusses_claim_df) == 0
            or len(user_df) == 0
            or len(tweet_df) == 0
            or len(user_posted_tweet_df) == 0
        ):
            return pd.DataFrame()

        # Define dataframe with the source tweets
        is_src = tweet_df.index.isin(tweet_dicusses_claim_df.src.tolist())
        src_tweets = tweet_df[is_src]

        # Define dataframe with the source users
        posted_rel = user_posted_tweet_df
        is_src = posted_rel[posted_rel.tgt.isin(src_tweets.index.tolist())].src.tolist()
        src_users = user_df.loc[is_src]

        # Initialise empty URL dataframe
        url_df = pd.DataFrame()

        # Add urls from source tweets
        if "entities.urls" in tweet_df.columns:

            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[Union[str, None]]:
                        A list of urls.
                """
                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            urls = (
                src_tweets["entities.urls"].dropna().map(extract_url).explode().tolist()
            )
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Add urls from source user urls
        if "entities.url.urls" in user_df.columns:

            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[Union[str, None]]:
                        A list of urls.
                """
                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            urls = (
                src_users["entities.url.urls"]
                .dropna()
                .map(extract_url)
                .explode()
                .tolist()
            )
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Add urls from source user descriptions
        if "entities.description.urls" in user_df.columns:

            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[Union[str, None]]:
                        A list of urls.
                """
                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            urls = (
                src_users["entities.description.urls"]
                .dropna()
                .map(extract_url)
                .explode()
                .tolist()
            )
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Add urls from images
        if (
            image_df is not None
            and len(image_df)
            and self.include_tweet_images
            and "attachments.media_keys" in tweet_df.columns
        ):
            urls = (
                src_tweets[["attachments.media_keys"]]
                .dropna()
                .explode("attachments.media_keys")
                .merge(
                    image_df[["media_key", "url"]],
                    left_on="attachments.media_keys",
                    right_on="media_key",
                )
                .url.tolist()
            )
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Add urls from profile pictures
        if self.include_extra_images and "profile_image_url" in user_df.columns:

            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                """Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]):
                        A list of dictionaries.

                Returns:
                    List[Union[str, None]]:
                        A list of urls.
                """
                return [dct.get("expanded_url") or dct.get("url") for dct in dcts]

            urls = src_users["profile_image_url"].dropna().tolist()
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Add urls from articles
        if article_df is not None and len(article_df) and self.include_articles:
            urls = article_df.url.dropna().tolist()
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        return url_df

    def _update_urls_from_articles(
        self, url_df: pd.DataFrame, article_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Updates the url_df with the urls from the articles.

        Args:
            url_df (pd.DataFrame):
                A dataframe with the urls.
            article_df (pd.DataFrame):
                A dataframe with the articles.

        Returns:
            pd.DataFrame:
                A dataframe with the urls.
        """
        if self.include_extra_images and len(article_df) and len(url_df):
            urls = article_df.top_image_url.dropna().tolist()
            url_dict = pd.DataFrame(dict(url=urls))
            url_df = (
                pd.concat((url_df, pd.DataFrame(url_dict)), axis=0)
                .drop_duplicates()
                .reset_index(drop=True)
            )
        return url_df
