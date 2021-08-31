'''Script containing functions related to the rehydration of Twitter data'''

from typing import List, Union
import logging


logger = logging.getLogger(__name__)


def rehydrate_tweets(tweet_ids: List[Union[str, int]]) -> List[dict]:
    '''Rehydrate tweets from the Twitter API.

    Args:
        tweet_ids (list of str or int):
            List of tweet IDs that needs to be rehydrated.

    Returns:
        list of dict:
            Each entry contains a dict with the tweet information pertaining to
            a single tweet ID.
    '''
    pass


def rehydrate_users(user_ids: List[Union[str, int]]) -> List[dict]:
    '''Rehydrate users from the Twitter API.

    Args:
        user_ids (list of str or int):
            List of user IDs that needs to be rehydrated.

    Returns:
        list of dict:
            Each entry contains a dict with the user information pertaining to
            a single user ID.
    '''
    pass
