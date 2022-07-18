"""Class that updates the precomputed IDs"""

from typing import Dict, Tuple

import pandas as pd


class IdUpdator:
    """Class that updates the IDs of nodes and relations"""

    def update_all(
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
        rel = ("tweet", "discusses", "claim")
        if rel in rels.keys():
            rels[rel] = self._update_tweet_discusses_claim(
                rel_df=rels[rel], tweet_df=nodes["tweet"], claim_df=nodes["claim"]
            )

        rel = ("article", "discusses", "claim")
        if rel in rels.keys():
            rels[rel] = self._update_article_discusses_claim(
                rel_df=rels[rel], article_df=nodes["article"], claim_df=nodes["claim"]
            )

        rel = ("user", "follows", "user")
        if rel in rels.keys():
            rels[rel] = self._update_user_follows_user(
                rel_df=rels[rel], user_df=nodes["user"]
            )

        rel = ("reply", "reply_to", "tweet")
        if rel in rels.keys():
            rels[rel] = self._update_reply_reply_to_tweet(
                rel_df=rels[rel], reply_df=nodes["reply"], tweet_df=nodes["tweet"]
            )

        rel = ("reply", "quote_of", "tweet")
        if rel in rels.keys():
            rels[rel] = self._update_reply_quote_of_tweet(
                rel_df=rels[rel], reply_df=nodes["reply"], tweet_df=nodes["tweet"]
            )

        rel = ("user", "retweeted", "tweet")
        if rel in rels.keys():
            rels[rel] = self._update_user_retweeted_tweet(
                rel_df=rels[rel], user_df=nodes["user"], tweet_df=nodes["tweet"]
            )

        # Remove ID columns from the claim and article dataframes
        nodes["claim"] = self._remove_id_column(node_df=nodes["claim"])
        if "article" in nodes.keys():
            nodes["article"] = self._remove_id_column(node_df=nodes["article"])

        return nodes, rels

    def _update_tweet_discusses_claim(
        self, rel_df: pd.DataFrame, tweet_df: pd.DataFrame, claim_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:Tweet)-[:DISCUSSES]->(:Claim) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            tweet_df (pd.DataFrame): The tweet dataframe.
            claim_df (pd.DataFrame): The claim dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    tweet_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="tweet_idx")),
                    left_on="src",
                    right_on="tweet_id",
                )
                .merge(
                    claim_df[["id"]]
                    .reset_index()
                    .rename(columns=dict(index="claim_idx")),
                    left_on="tgt",
                    right_on="id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.tweet_idx.tolist(), tgt=merged.claim_idx.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _update_article_discusses_claim(
        self, rel_df: pd.DataFrame, article_df: pd.DataFrame, claim_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:Article)-[:DISCUSSES]->(:Claim) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            article_df (pd.DataFrame): The article dataframe.
            claim_df (pd.DataFrame): The claim dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    article_df[["id"]]
                    .reset_index()
                    .rename(columns=dict(index="art_idx")),
                    left_on="src",
                    right_on="id",
                )
                .merge(
                    claim_df[["id"]]
                    .reset_index()
                    .rename(columns=dict(index="claim_idx")),
                    left_on="tgt",
                    right_on="id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.art_idx.tolist(), tgt=merged.claim_idx.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _update_user_follows_user(
        self, rel_df: pd.DataFrame, user_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:User)-[:FOLLOWS]->(:User) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            user_df (pd.DataFrame): The user dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx1")),
                    left_on="src",
                    right_on="user_id",
                )
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx2")),
                    left_on="tgt",
                    right_on="user_id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.user_idx1.tolist(), tgt=merged.user_idx2.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _update_reply_reply_to_tweet(
        self, rel_df: pd.DataFrame, reply_df: pd.DataFrame, tweet_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:Reply)-[:REPLY_TO]->(:Tweet) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            reply_df (pd.DataFrame): The reply dataframe.
            tweet_df (pd.DataFrame): The tweet dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    reply_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="reply_idx")),
                    left_on="src",
                    right_on="tweet_id",
                )
                .merge(
                    tweet_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="tweet_idx")),
                    left_on="tgt",
                    right_on="tweet_id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.reply_idx.tolist(), tgt=merged.tweet_idx.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _update_reply_quote_of_tweet(
        self, rel_df: pd.DataFrame, reply_df: pd.DataFrame, tweet_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:Reply)-[:QUOTE_OF]->(:Tweet) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            reply_df (pd.DataFrame): The reply dataframe.
            tweet_df (pd.DataFrame): The tweet dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    reply_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="reply_idx")),
                    left_on="src",
                    right_on="tweet_id",
                )
                .merge(
                    tweet_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="tweet_idx")),
                    left_on="tgt",
                    right_on="tweet_id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.reply_idx.tolist(), tgt=merged.tweet_idx.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _update_user_retweeted_tweet(
        self, rel_df: pd.DataFrame, user_df: pd.DataFrame, tweet_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update the (:User)-[:RETWEETED]->(:Tweet) relation.

        Args:
            rel_df (pd.DataFrame): The relation dataframe.
            user_df (pd.DataFrame): The user dataframe.
            tweet_df (pd.DataFrame): The tweet dataframe.

        Returns:
            pd.DataFrame: The updated relation dataframe.
        """
        if len(rel_df) > 0:
            merged = (
                rel_df.astype(dict(src=int, tgt=int))
                .merge(
                    user_df[["user_id"]]
                    .reset_index()
                    .rename(columns=dict(index="user_idx")),
                    left_on="src",
                    right_on="user_id",
                )
                .merge(
                    tweet_df[["tweet_id"]]
                    .reset_index()
                    .rename(columns=dict(index="tweet_idx")),
                    left_on="tgt",
                    right_on="tweet_id",
                )
            )
            if len(merged) > 0:
                data_dict = dict(
                    src=merged.user_idx.tolist(), tgt=merged.tweet_idx.tolist()
                )
                rel_df = pd.DataFrame(data_dict)
            else:
                rel_df = pd.DataFrame()

        return rel_df

    def _remove_id_column(self, node_df: pd.DataFrame) -> pd.DataFrame:
        """Remove the id column from the node dataframe.

        Args:
            node_df (pd.DataFrame): The node dataframe.

        Returns:
            pd.DataFrame: The node dataframe without the id column.
        """
        if len(node_df) > 0:
            node_df = node_df.drop(columns="id")
        return node_df
