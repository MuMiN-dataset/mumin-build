"""Wrapper for the Twitter API"""

import logging
import time
from json import JSONDecodeError
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class Twitter:
    """A wrapper for the Twitter API.

    Args:
        twitter_bearer_token (str):
            The bearer token from the Twitter API.
    """

    tweet_lookup_url: str = "https://api.twitter.com/2/tweets"

    def __init__(self, twitter_bearer_token: str):
        self.api_key = twitter_bearer_token
        self.headers = dict(Authorization=f"Bearer {self.api_key}")
        self.expansions = [
            "attachments.poll_ids",
            "attachments.media_keys",
            "author_id",
            "entities.mentions.username",
            "geo.place_id",
            "in_reply_to_user_id",
            "referenced_tweets.id",
            "referenced_tweets.id.author_id",
        ]
        self.tweet_fields = [
            "attachments",
            "author_id",
            "conversation_id",
            "created_at",
            "entities",
            "geo",
            "id",
            "in_reply_to_user_id",
            "lang",
            "public_metrics",
            "possibly_sensitive",
            "referenced_tweets",
            "reply_settings",
            "source",
            "text",
            "withheld",
        ]
        self.user_fields = [
            "created_at",
            "description",
            "entities",
            "id",
            "location",
            "name",
            "pinned_tweet_id",
            "profile_image_url",
            "protected",
            "public_metrics",
            "url",
            "username",
            "verified",
            "withheld",
        ]
        self.media_fields = [
            "duration_ms",
            "height",
            "media_key",
            "preview_image_url",
            "type",
            "url",
            "width",
            "public_metrics",
        ]
        self.place_fields = [
            "contained_within",
            "country",
            "country_code",
            "full_name",
            "geo",
            "id",
            "name",
            "place_type",
        ]
        self.poll_fields = [
            "duration_minutes",
            "end_datetime",
            "id",
            "options",
            "voting_status",
        ]

    def rehydrate_tweets(
        self, tweet_ids: List[Union[str, int]]
    ) -> Dict[str, pd.DataFrame]:
        """Rehydrates the tweets for the given tweet IDs.

        Args:
            tweet_ids (list of either str or int):
                The tweet IDs to rehydrate.

        Returns:
            dict:
                A dictionary with keys 'tweets', 'users', 'media',
                'polls' and 'places', where the values are the
                associated Pandas DataFrame objects.
        """
        # Ensure that the tweet IDs are strings
        tweet_ids = [str(tweet_id) for tweet_id in tweet_ids]

        # Set up the params for the GET request
        get_params = {
            "expansions": ",".join(self.expansions),
            "media.fields": ",".join(self.media_fields),
            "place.fields": ",".join(self.place_fields),
            "poll.fields": ",".join(self.poll_fields),
            "tweet.fields": ",".join(self.tweet_fields),
            "user.fields": ",".join(self.user_fields),
        }

        # Split `tweet_ids` into batches of at most 100, as this is the
        # maximum number allowed by the API
        num_batches = len(tweet_ids) // 100
        if len(tweet_ids) % 100 != 0:
            num_batches += 1
        batches = np.array_split(tweet_ids, num_batches)

        # Initialise dataframes
        tweet_df = pd.DataFrame()
        user_df = pd.DataFrame()
        media_df = pd.DataFrame()
        poll_df = pd.DataFrame()
        place_df = pd.DataFrame()

        # Initialise progress bar
        if len(batches) > 1:
            pbar = tqdm(total=len(tweet_ids), desc="Rehydrating")

        # Loop over all the batches
        for batch in batches:

            # Add the batch tweet IDs to the batch
            get_params["ids"] = ",".join(batch)

            # Perform the GET request
            try:
                response = requests.get(
                    self.tweet_lookup_url, params=get_params, headers=self.headers
                )
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"[{e}] Error in rehydrating tweets.\nThe "
                    f"parameters used were {get_params}."
                )
                continue

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug("Request limit reached. Waiting...")
                time.sleep(1)
                try:
                    response = requests.get(
                        self.tweet_lookup_url, params=get_params, headers=self.headers
                    )
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"[{e}] Error in rehydrating tweets.\nThe "
                        f"parameters used were {get_params}."
                    )
                    continue

            # If we are not authorised then continue to the next batch
            if response.status_code == 401:
                continue

            # If the GET request failed then continue to the next batch
            elif response.status_code != 200:
                msg = f"[{response.status_code}] {response.text}"
                logger.error(msg)
                continue

            # Convert the response to a dict
            try:
                data_dict = response.json()
            except JSONDecodeError as e:
                logger.error(
                    f"[{e}] Error in unpacking tweets.\nThe "
                    f"parameters used were {get_params}."
                )
                continue

            # If the query returned errors then continue to the next batch
            if "data" not in data_dict and "errors" in data_dict:
                error = data_dict["errors"][0]
                logger.error(error["detail"])
                continue

            # Tweet dataframe
            if "data" in data_dict:
                df = pd.json_normalize(data_dict["data"]).rename(
                    columns=dict(id="tweet_id")
                )
                tweet_df = (
                    pd.concat((tweet_df, df))
                    .drop_duplicates(subset="tweet_id")
                    .astype(dict(tweet_id=int))
                    .reset_index(drop=True)
                )

            # User dataframe
            if "includes" in data_dict and "users" in data_dict["includes"]:
                users = data_dict["includes"]["users"]
                df = pd.json_normalize(users).rename(columns=dict(id="user_id"))
                user_df = (
                    pd.concat((user_df, df))
                    .drop_duplicates(subset="user_id")
                    .astype(dict(user_id=int))
                    .reset_index(drop=True)
                )

            # Media dataframe
            if "includes" in data_dict and "media" in data_dict["includes"]:
                media = data_dict["includes"]["media"]
                df = pd.json_normalize(media)
                media_df = (
                    pd.concat((media_df, df))
                    .drop_duplicates(subset="media_key")
                    .reset_index(drop=True)
                )

            # Poll dataframe
            if "includes" in data_dict and "polls" in data_dict["includes"]:
                polls = data_dict["includes"]["polls"]
                df = pd.json_normalize(polls).rename(columns=dict(id="poll_id"))
                poll_df = (
                    pd.concat((poll_df, df))
                    .drop_duplicates(subset="poll_id")
                    .astype(dict(poll_id=int))
                    .reset_index(drop=True)
                )

            # Places dataframe
            if "includes" in data_dict and "places" in data_dict["includes"]:
                places = data_dict["includes"]["places"]
                df = pd.json_normalize(places).rename(columns=dict(id="place_id"))
                place_df = (
                    pd.concat((place_df, df))
                    .drop_duplicates(subset="place_id")
                    .reset_index(drop=True)
                )

            # Update the progress bar
            if len(batches) > 1:
                pbar.update(len(batch))

        # Close the progress bar
        if len(batches) > 1:
            pbar.close()

        # Collect all the resulting dataframes
        all_dfs = dict(
            tweets=tweet_df,
            users=user_df,
            media=media_df,
            polls=poll_df,
            places=place_df,
        )

        # Return the dictionary containing all the dataframes
        return all_dfs
