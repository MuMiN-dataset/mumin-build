'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional
import pandas as pd
import zipfile
import io
import numpy as np
import logging
import requests
from collections import defaultdict
import re
import os
import multiprocessing as mp
from tqdm.auto import tqdm
import warnings
import json
from functools import partial
from shutil import rmtree

from .twitter import Twitter
from .article import process_article_url
from .image import process_image_url
from .dgl import build_dgl_dataset


# Set up logging
logger = logging.getLogger(__name__)


# Disable tokenizer parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Allows progress bars with `pd.DataFrame.progress_apply`
tqdm.pandas()


class MuminDataset:
    '''The MuMiN misinformation dataset, from [1].

    Args:
        twitter_bearer_token (str):
            The Twitter bearer token.
        size (str, optional):
            The size of the dataset. Can be either 'small', 'medium' or
            'large'. Defaults to 'small'.
        include_replies (bool, optional):
            Whether to include replies and quote tweets in the dataset.
            Defaults to True.
        include_articles (bool, optional):
            Whether to include articles in the dataset. This will mean that
            compilation of the dataset will take a bit longer, as these need to
            be downloaded and parsed. Defaults to True.
        include_images (bool, optional):
            Whether to include images in the dataset. This will mean that
            compilation of the dataset will take a bit longer, as these need to
            be downloaded and parsed. Defaults to True.
        include_hashtags (bool, optional):
            Whether to include hashtags in the dataset. Defaults to True.
        include_mentions (bool, optional):
            Whether to include mentions in the dataset. Defaults to True.
        text_embedding_model_id (str, optional):
            The HuggingFace Hub model ID to use when embedding texts. Defaults
            to 'xlm-roberta-base'.
        image_embedding_model_id (str, optional):
            The HuggingFace Hub model ID to use when embedding images. Defaults
            to 'google/vit-base-patch16-224-in21k'.
        dataset_path (str, pathlib Path or None, optional):
            The path to the file where the dataset should be stored. If None
            then the dataset will be stored at './mumin-<size>.zip'. Defaults
            to None.
        verbose (bool, optional):
            Whether extra information should be outputted. Defaults to True.

    Attributes:
        include_replies (bool): Whether to include replies in the dataset.
        include_articles (bool): Whether to include articles in the dataset.
        include_images (bool): Whether to include images in the dataset.
        include_hashtags (bool): Whether to include hashtags in the dataset.
        include_mentions (bool): Whether to include mentions in the dataset.
        size (str): The size of the dataset.
        dataset_path (pathlib Path): The dataset file.
        text_embedding_model_id (str): The model ID used for embedding text.
        image_embedding_model_id (str): The model ID used for embedding images.
        nodes (dict): The nodes of the dataset.
        rels (dict): The relations of the dataset.
        compiled (bool): Whether the dataset has been compiled.
        verbose (bool): Whether extra information should be outputted.
        download_url (str): The URL to download the dataset from.

    References:
        - [1] Nielsen and McConville: _MuMiN: A Large-Scale Multilingual
              Multimodal Fact-Checked Misinformation Dataset with Linked Social
              Network Posts_ (2021)
    '''
    download_url: str = ('https://github.com/CLARITI-REPHRAIN/mumin-build/raw/'
                         'c6700f77c4c6cffcdb0bbfad21feb40c153f9a75'
                         '/data/mumin.zip')
    _node_dump: List[str] = ['claim', 'tweet', 'user', 'image', 'article',
                             'hashtag', 'reply']
    _rel_dump: List[Tuple[str, str, str]] = [
        ('tweet', 'discusses', 'claim'),
        ('tweet', 'mentions', 'user'),
        ('tweet', 'has_image', 'image'),
        ('tweet', 'has_hashtag', 'hashtag'),
        ('tweet', 'has_article', 'article'),
        ('reply', 'reply_to', 'tweet'),
        ('reply', 'quote_of', 'tweet'),
        ('user', 'posted', 'tweet'),
        ('user', 'posted', 'reply'),
        ('user', 'mentions', 'user'),
        ('user', 'has_hashtag', 'hashtag'),
        ('user', 'has_profile_picture', 'image'),
        ('user', 'retweeted', 'tweet'),
        ('user', 'follows', 'user'),
        ('article', 'has_top_image', 'image'),
    ]

    def __init__(self,
                 twitter_bearer_token: str,
                 size: str = 'small',
                 include_replies: bool = True,
                 include_articles: bool = True,
                 include_images: bool = True,
                 include_hashtags: bool = True,
                 include_mentions: bool = True,
                 text_embedding_model_id: str = 'xlm-roberta-base',
                 image_embedding_model_id: str = ('google/vit-base-patch16-'
                                                  '224-in21k'),
                 dataset_path: Optional[Union[str, Path]] = None,
                 verbose: bool = True):
        self.compiled = False
        self._twitter = Twitter(twitter_bearer_token=twitter_bearer_token)
        self.size = size
        self.include_replies = include_replies
        self.include_articles = include_articles
        self.include_images = include_images
        self.include_hashtags = include_hashtags
        self.include_mentions = include_mentions
        self.text_embedding_model_id = text_embedding_model_id
        self.image_embedding_model_id = image_embedding_model_id
        self.verbose = verbose
        self.nodes: Dict[str, pd.DataFrame] = dict()
        self.rels: Dict[Tuple[str, str, str], pd.DataFrame] = dict()

        if dataset_path is None:
            dataset_path = f'./mumin-{size}.zip'
        self.dataset_path = Path(dataset_path)

        # Set up logging verbosity
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    def __repr__(self) -> str:
        '''A string representation of the dataset.

        Returns:
            str: The representation of the dataset.
        '''
        if len(self.nodes) == 0 or len(self.rels) == 0:
            return f'MuminDataset(size={self.size}, compiled={self.compiled})'
        else:
            num_nodes = sum([len(df) for df in self.nodes.values()])
            num_rels = sum([len(df) for df in self.rels.values()])
            return (f'MuminDataset(num_nodes={num_nodes:,}, '
                    f'num_relations={num_rels:,}, '
                    f'size=\'{self.size}\', '
                    f'compiled={self.compiled})')

    def compile(self, overwrite: bool = False):
        '''Compiles the dataset.

        This entails downloading the dataset, rehydrating the Twitter data and
        downloading the relevant associated data, such as articles and images.

        Args:
            overwrite (bool, optional):
                Whether the dataset directory should be overwritten, in case it
                already exists. Defaults to False.
        '''
        self._download(overwrite=overwrite)
        self._load_dataset()

        # Variable to check if dataset has been compiled
        compiled = self.compiled or ('text' in self.nodes['tweet'].columns)

        # Only compile the dataset if it has not already been compiled
        if not compiled:

            # Shrink dataset to the correct size
            self._shrink_dataset()

            # Rehydrate the tweets
            self._rehydrate(node_type='tweet')
            self._rehydrate(node_type='reply')

            # Update the IDs of the data that was there pre-hydration
            self._update_precomputed_ids()

            # Extract data from the rehydrated tweets
            self._extract_nodes()
            self._extract_relations()
            self._extract_articles()
            self._extract_images()

            # Filter the data
            self._filter_node_features()
            self._filter_relations()

        # Remove unnecessary bits
        self._remove_auxilliaries()
        self._remove_islands()

        # Save dataset
        if not compiled:
            self._dump_dataset()

        # Mark dataset as compiled
        self.compiled = True

        return self

    def _download(self, overwrite: bool = False):
        '''Downloads the dataset.

        Args:
            overwrite (bool, optional):
                Whether the dataset directory should be overwritten, in case it
                already exists. Defaults to False.
        '''
        if (not self.dataset_path.exists() or
                (self.dataset_path.exists() and overwrite)):

            logger.info('Downloading dataset')

            # Remove existing directory if we are overwriting
            if self.dataset_path.exists() and overwrite:
                self.dataset_path.unlink()

            # Set up download stream of dataset
            with requests.get(self.download_url, stream=True) as response:

                # If the response was unsuccessful then raise an error
                if response.status_code != 200:
                    msg = f'[{response.status_code}] {response.content}'
                    raise RuntimeError(msg)

                # Download dataset with progress bar
                total = int(response.headers['Content-Length'])
                with tqdm(total=total,
                          unit='iB',
                          unit_scale=True,
                          desc='Downloading MuMiN') as pbar:
                    with Path(self.dataset_path).open('wb') as f:
                        for data in response.iter_content(1024):
                            pbar.update(len(data))
                            f.write(data)

            logger.info('Converting dataset to less compressed format')

            # Open the zip file containing the dataset
            data_dict = dict()
            with zipfile.ZipFile(self.dataset_path,
                                 mode='r',
                                 compression=zipfile.ZIP_DEFLATED) as zip_file:


                # Loop over all the files in the zipped file
                for name in zip_file.namelist():

                    # Extract the dataframe in the file
                    byte_data = zip_file.read(name=name)
                    df = pd.read_pickle(io.BytesIO(byte_data),
                                        compression='xz')
                    data_dict[name] = df

            # Overwrite the zip file in a less compressed way, to make io
            # operations faster
            with zipfile.ZipFile(self.dataset_path,
                                 mode='w',
                                 compression=zipfile.ZIP_STORED) as zip_file:
                for name, df in data_dict.items():
                    buffer = io.BytesIO()
                    df.to_pickle(buffer, protocol=4)
                    zip_file.writestr(name, data=buffer.getvalue())

        return self

    def _load_dataset(self):
        '''Loads the dataset files into memory.

        Raises:
            RuntimeError:
                If the dataset has not been downloaded yet.
        '''
        # Raise error if the dataset has not been downloaded yet
        if not self.dataset_path.exists():
            raise RuntimeError('Dataset has not been downloaded yet!')

        logger.info('Loading dataset')

        # Reset `nodes` and `relations` to ensure a fresh start
        self.nodes = dict()
        self.rels = dict()

        # Open the zip file containing the dataset
        with zipfile.ZipFile(self.dataset_path,
                             mode='r',
                             compression=zipfile.ZIP_STORED) as zip_file:

            # Loop over all the files in the zipped file
            for name in zip_file.namelist():

                # Extract the dataframe in the file
                byte_data = zip_file.read(name=name)
                df = pd.read_pickle(io.BytesIO(byte_data))

                # If there are no underscores in the filename then we assume
                # that it contains node data
                if '_' not in name:
                    self.nodes[name.replace('.pickle', '')] = df.copy()

                # Otherwise, with underscores in the filename then we assume it
                # contains relation data
                else:
                    splits = name.replace('.pickle', '').split('_')
                    src = splits[0]
                    tgt = splits[-1]
                    rel = '_'.join(splits[1:-1])
                    self.rels[(src, rel, tgt)] = df.copy()

            # Ensure that claims are present in the dataset
            if 'claim' not in self.nodes.keys():
                raise RuntimeError('No claims are present in the file!')

            # Ensure that tweets are present in the dataset, and also that the
            # tweet IDs are unique
            if 'tweet' not in self.nodes.keys():
                raise RuntimeError('No tweets are present in the file!')
            else:
                tweet_df = self.nodes['tweet']
                duplicated = (tweet_df[tweet_df.tweet_id.duplicated()].tweet_id
                                                                      .tolist())
                if len(duplicated) > 0:
                    raise RuntimeError(f'The tweet IDs {duplicated} are '
                                       f'duplicate in the dataset!')

        return self

    def _shrink_dataset(self):
        '''Shrink dataset if `size` is 'small' or 'medium'''
        if self.size != 'large':
            logger.info('Shrinking dataset')

            # Define the `relevance` threshold
            if self.size == 'small':
                threshold = 0.80
            elif self.size == 'medium':
                threshold = 0.75
            elif self.size == 'test':
                threshold = 0.995

            # Filter (:Tweet)-[:DISCUSSES]->(:Claim)
            discusses_rel = (self.rels[('tweet', 'discusses', 'claim')]
                             .query(f'relevance > {threshold}')
                             .reset_index(drop=True))
            self.rels[('tweet', 'discusses', 'claim')] = discusses_rel

            # Filter tweets
            tweet_df = self.nodes['tweet']
            include_tweet = tweet_df.tweet_id.isin(discusses_rel.src.tolist())
            tweet_df = tweet_df[include_tweet].reset_index(drop=True)
            self.nodes['tweet'] = tweet_df

            # Filter (:Reply)-[:REPLY_TO]->(:Tweet)
            reply_rel = self.rels[('reply', 'reply_to', 'tweet')]
            include = reply_rel.tgt.isin(tweet_df.tweet_id.tolist())
            reply_rel = reply_rel[include].reset_index(drop=True)
            self.rels[('reply', 'reply_to', 'tweet')] = reply_rel

            # Filter (:Reply)-[:QUOTE_OF]->(:Tweet)
            quote_rel = self.rels[('reply', 'quote_of', 'tweet')]
            include = quote_rel.tgt.isin(tweet_df.tweet_id.tolist())
            quote_rel = quote_rel[include].reset_index(drop=True)
            self.rels[('reply', 'quote_of', 'tweet')] = quote_rel

            # Filter replies
            reply_df = self.nodes['reply']
            include_reply = reply_df.tweet_id.isin(reply_rel.src.tolist())
            include_quote = reply_df.tweet_id.isin(quote_rel.src.tolist())
            self.nodes['reply'] = (reply_df[include_reply | include_quote]
                                   .reset_index(drop=True))

            # Filter claims
            claim_df = self.nodes['claim']
            include_claim = claim_df.id.isin(discusses_rel.tgt.tolist())
            self.nodes['claim'] = (claim_df[include_claim]
                                   .reset_index(drop=True))

            # Filter (:Article)-[:DISCUSSES]->(:Claim)
            discusses_rel = (self.rels[('article', 'discusses', 'claim')]
                             .query(f'relevance > {threshold}')
                             .reset_index(drop=True))
            self.rels[('article', 'discusses', 'claim')] = discusses_rel

            # Filter articles
            article_df = self.nodes['article']
            include_article = article_df.id.isin(discusses_rel.src.tolist())
            self.nodes['article'] = (article_df[include_article]
                                     .reset_index(drop=True))

            # Filter (:User)-[:POSTED]->(:Tweet)
            posted_rel = self.rels[('user', 'posted', 'tweet')]
            posted_rel = posted_rel[posted_rel.tgt.isin(self.nodes['tweet']
                                                            .tweet_id
                                                            .tolist())]
            posted_rel = posted_rel.reset_index(drop=True)
            self.rels[('user', 'posted', 'tweet')] = posted_rel

            # Filter (:User)-[:POSTED]->(:Reply)
            rposted_rel = self.rels[('user', 'posted', 'reply')]
            rposted_rel = rposted_rel[rposted_rel.tgt.isin(self.nodes['reply']
                                                               .tweet_id
                                                               .tolist())]
            rposted_rel = rposted_rel.reset_index(drop=True)
            self.rels[('user', 'posted', 'reply')] = rposted_rel

            # Filter (:Tweet)-[:MENTIONS]->(:User)
            mentions_rel = self.rels[('tweet', 'mentions', 'user')]
            mentions_rel = mentions_rel[mentions_rel
                                        .src
                                        .isin(self.nodes['tweet']
                                                  .tweet_id
                                                  .tolist())]
            mentions_rel = mentions_rel.reset_index(drop=True)
            self.rels[('tweet', 'mentions', 'user')] = mentions_rel

            # Filter (:User)-[:FOLLOWS]->(:User)
            follows_rel = self.rels[('user', 'follows', 'user')]
            user_df = self.nodes['user']
            has_posted = user_df.user_id.isin(posted_rel.src.tolist())
            has_rposted = user_df.user_id.isin(rposted_rel.src.tolist())
            was_mentioned = user_df.user_id.isin(mentions_rel.tgt.tolist())
            filtered_users = (user_df[has_posted | has_rposted | was_mentioned]
                              .reset_index(drop=True)
                              .user_id
                              .tolist())
            follows_rel = follows_rel[follows_rel.src.isin(filtered_users)]
            follows_rel = follows_rel.reset_index(drop=True)
            self.rels[('user', 'follows', 'user')] = follows_rel

            # Filter users
            user_df = self.nodes['user']
            has_posted = user_df.user_id.isin(posted_rel.src.tolist())
            has_rposted = user_df.user_id.isin(rposted_rel.src.tolist())
            was_mentioned = user_df.user_id.isin(mentions_rel.tgt.tolist())
            is_followed = user_df.user_id.isin(follows_rel.tgt.tolist())
            self.nodes['user'] = (user_df[has_posted |
                                          has_rposted |
                                          was_mentioned |
                                          is_followed]
                                  .reset_index(drop=True))

            # Filter (:User)-[:MENTIONS]->(:User)
            mentions_rel = self.rels[('user', 'mentions', 'user')]
            mentions_rel = mentions_rel[mentions_rel
                                        .src
                                        .isin(self.nodes['user']
                                                  .user_id
                                                  .tolist())]
            mentions_rel = mentions_rel[mentions_rel
                                        .tgt
                                        .isin(self.nodes['user']
                                                  .user_id
                                                  .tolist())]
            mentions_rel = mentions_rel.reset_index(drop=True)
            self.rels[('user', 'mentions', 'user')] = mentions_rel

            return self

    def _rehydrate(self, node_type: str):
        '''Rehydrate the tweets and users in the dataset.

        Args:
            node_type (str): The type of node to rehydrate.
        '''

        if (node_type in self.nodes.keys() and
                (node_type != 'reply' or self.include_replies)):

            logger.info(f'Rehydrating {node_type} nodes')

            # Get the tweet IDs
            tweet_ids = self.nodes[node_type].tweet_id.tolist()

            # Store any features the nodes might have had before hydration
            prehydration_df = self.nodes[node_type].copy()

            # Rehydrate the tweets
            tweet_dfs = self._twitter.rehydrate_tweets(tweet_ids=tweet_ids)

            # Extract and store tweets and users
            self.nodes[node_type] = (tweet_dfs['tweets']
                                     .drop_duplicates(subset='tweet_id')
                                     .reset_index(drop=True))
            if ('user' in self.nodes.keys() and
                    'username' in self.nodes['user'].columns):
                user_df = (self.nodes['user']
                               .append(tweet_dfs['users'])
                               .drop_duplicates(subset='user_id')
                               .reset_index(drop=True))
            else:
                user_df = tweet_dfs['users']
            self.nodes['user'] = user_df

            # Add prehydration tweet features back to the tweets
            self.nodes[node_type] = pd.merge(left=self.nodes[node_type],
                                             right=prehydration_df,
                                             how='left',
                                             on='tweet_id')

            # Extract and store images
            if self.include_images and len(tweet_dfs['media']):
                image_df = (tweet_dfs['media']
                            .query('type == "photo"')
                            .drop_duplicates(subset='media_key')
                            .reset_index(drop=True))

                video_query = '(type == "video") or (type == "animated gif")'
                if len(tweet_dfs['media'].query(video_query)):
                    video_df = (tweet_dfs['media']
                                .query(video_query)
                                .drop(columns=['url', 'duration_ms',
                                               'public_metrics.view_count'])
                                .rename(columns=dict(preview_image_url='url')))
                    image_df = image_df.append(video_df)

                if 'media' in self.nodes.keys():
                    image_df = (self.nodes['media']
                                    .append(image_df)
                                    .drop_duplicates(subset='media_key')
                                    .reset_index(drop=True))
                self.nodes['image'] = image_df

            return self

    def _update_precomputed_ids(self):
        '''Update the node IDs in the pre-hydrated dataset.

        In the dataset nodes are uniquely characterised using their Twitter
        IDs, like node IDs and user IDs, and articles and claims have IDs from
        Neo4j. After rehydration we use a simple enumeration as the IDs of the
        node types, and this updates the nodes and relations to those IDs.
        '''
        logger.info('Updating precomputed IDs')

        # Update the (:Tweet)-[:DISCUSSES]->(:Claim) relation
        rel_type = ('tweet', 'discusses', 'claim')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['tweet'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='tweet_idx')),
                                left_on='src',
                                right_on='tweet_id')
                         .merge(self.nodes['claim'][['id']]
                                    .reset_index()
                                    .rename(columns=dict(index='claim_idx')),
                                left_on='tgt',
                                right_on='id'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.claim_idx.tolist(),
                             relevance=merged.relevance.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:Article)-[:DISCUSSES]->(:Claim) relation
        rel_type = ('article', 'discusses', 'claim')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['article'][['id']]
                                    .reset_index()
                                    .rename(columns=dict(index='art_idx')),
                                left_on='src',
                                right_on='id')
                         .merge(self.nodes['claim'][['id']]
                                    .reset_index()
                                    .rename(columns=dict(index='claim_idx')),
                                left_on='tgt',
                                right_on='id'))
            data_dict = dict(src=merged.art_idx.tolist(),
                             tgt=merged.claim_idx.tolist(),
                             relevance=merged.relevance.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:User)-[:FOLLOWS]->(:User) relation
        rel_type = ('user', 'follows', 'user')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['user'][['user_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='user_idx1')),
                                left_on='src',
                                right_on='user_id')
                         .merge(self.nodes['user'][['user_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='user_idx2')),
                                left_on='tgt',
                                right_on='user_id'))
            data_dict = dict(src=merged.user_idx1.tolist(),
                             tgt=merged.user_idx2.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:Reply)-[:REPLY_TO]->(:Tweet) relation
        rel_type = ('reply', 'reply_to', 'tweet')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['reply'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='reply_idx')),
                                left_on='src',
                                right_on='tweet_id')
                         .merge(self.nodes['tweet'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='tweet_idx')),
                                left_on='tgt',
                                right_on='tweet_id'))
            data_dict = dict(src=merged.reply_idx.tolist(),
                             tgt=merged.tweet_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:Reply)-[:QUOTE_OF]->(:Tweet) relation
        rel_type = ('reply', 'quote_of', 'tweet')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['reply'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='reply_idx')),
                                left_on='src',
                                right_on='tweet_id')
                         .merge(self.nodes['tweet'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='tweet_idx')),
                                left_on='tgt',
                                right_on='tweet_id'))
            data_dict = dict(src=merged.reply_idx.tolist(),
                             tgt=merged.tweet_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:Reply)-[:REPLY_TO]->(:Reply) relation
        rel_type = ('reply', 'reply_to', 'reply')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['reply'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='reply_idx1')),
                                left_on='src',
                                right_on='tweet_id')
                         .merge(self.nodes['reply'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='reply_idx2')),
                                left_on='tgt',
                                right_on='tweet_id'))
            data_dict = dict(src=merged.reply_idx1.tolist(),
                             tgt=merged.reply_idx2.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:User)-[:RETWEETED]->(:Tweet) relation
        rel_type = ('user', 'retweeted', 'tweet')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['user'][['user_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='user_idx')),
                                left_on='src',
                                right_on='user_id')
                         .merge(self.nodes['tweet'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='tweet_idx')),
                                left_on='tgt',
                                right_on='tweet_id'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.tweet_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Update the (:User)-[:LIKED]->(:Tweet) relation
        rel_type = ('user', 'liked', 'tweet')
        if rel_type in self.rels.keys():
            rel = self.rels[rel_type]
            merged = (rel.merge(self.nodes['user'][['user_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='user_idx')),
                                left_on='src',
                                right_on='user_id')
                         .merge(self.nodes['tweet'][['tweet_id']]
                                    .reset_index()
                                    .rename(columns=dict(index='tweet_idx')),
                                left_on='tgt',
                                right_on='tweet_id'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.tweet_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[rel_type] = rel_df

        # Remove the ID columns of the Claim and Article nodes
        if 'claim' in self.nodes.keys():
            self.nodes['claim'] = self.nodes['claim'].drop(columns='id')
        if 'article' in self.nodes.keys():
            self.nodes['article'] = self.nodes['article'].drop(columns='id')

        return self

    def _extract_nodes(self):
        '''Extracts nodes from the raw Twitter data'''
        logger.info('Extracting nodes')

        # Hashtags
        if self.include_hashtags:
            def extract_hashtag(dcts: List[dict]) -> List[str]:
                '''Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of hashtags.
                '''
                return [dct.get('tag') for dct in dcts]

            # Add hashtags from tweets
            if 'entities.hashtags' in self.nodes['tweet'].columns:
                hashtags = (self.nodes['tweet']['entities.hashtags']
                                .dropna()
                                .map(extract_hashtag)
                                .explode()
                                .tolist())
                node_df = pd.DataFrame(dict(tag=hashtags))
                if 'hashtag' in self.nodes.keys():
                    node_df = (self.nodes['hashtag']
                                   .append(node_df)
                                   .drop_duplicates()
                                   .reset_index(drop=True))
                self.nodes['hashtag'] = node_df

            # Add hashtags from users
            if 'entities.description.hashtags' in self.nodes['user'].columns:
                hashtags = (self.nodes['user']['entities.description.hashtags']
                                .dropna()
                                .map(extract_hashtag)
                                .explode()
                                .tolist())
                node_df = pd.DataFrame(dict(tag=hashtags))
                if 'hashtag' in self.nodes.keys():
                    node_df = (self.nodes['hashtag']
                                   .append(node_df)
                                   .drop_duplicates()
                                   .reset_index(drop=True))
                self.nodes['hashtag'] = node_df

        # Add urls from tweets
        if 'entities.urls' in self.nodes['tweet'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[Union[str, None]]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['tweet']['entities.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url']
                               .append(node_df)
                               .drop_duplicates()
                               .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from user urls
        if 'entities.url.urls' in self.nodes['user'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[Union[str, None]]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['entities.url.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url']
                               .append(node_df)
                               .drop_duplicates()
                               .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from user descriptions
        if 'entities.description.urls' in self.nodes['user'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[Union[str, None]]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['entities.description.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url']
                               .append(node_df)
                               .drop_duplicates()
                               .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from profile pictures
        if (self.include_images and
                'profile_image_url' in self.nodes['user'].columns):
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[Union[str, None]]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['profile_image_url']
                        .dropna()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url']
                               .append(node_df)
                               .drop_duplicates()
                               .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from articles
        if self.include_articles:
            urls = self.nodes['article'].url.dropna().tolist()
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url']
                               .append(node_df)
                               .drop_duplicates()
                               .reset_index(drop=True))
            self.nodes['url'] = node_df

        return self

    def _extract_relations(self):
        '''Extracts relations from the raw Twitter data'''
        logger.info('Extracting relations')

        # (:User)-[:POSTED]->(:Tweet)
        merged = (self.nodes['tweet'][['author_id']]
                      .dropna()
                      .reset_index()
                      .rename(columns=dict(index='tweet_idx'))
                      .astype({'author_id': int})
                      .merge(self.nodes['user'][['user_id']]
                                 .reset_index()
                                 .rename(columns=dict(index='user_idx')),
                             left_on='author_id',
                             right_on='user_id'))
        data_dict = dict(src=merged.user_idx.tolist(),
                         tgt=merged.tweet_idx.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.rels[('user', 'posted', 'tweet')] = rel_df

        # (:User)-[:POSTED]->(:Reply)
        if self.include_replies:
            merged = (self.nodes['reply'][['author_id']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='reply_idx'))
                          .astype({'author_id': int})
                          .merge(self.nodes['user'][['user_id']]
                                     .reset_index()
                                     .rename(columns=dict(index='user_idx')),
                                 left_on='author_id',
                                 right_on='user_id'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.reply_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'posted', 'reply')] = rel_df

        # (:Tweet)-[:MENTIONS]->(:User)
        mentions_exist = 'entities.mentions' in self.nodes['tweet'].columns
        if self.include_mentions and mentions_exist:

            def extract_mention(dcts: List[dict]) -> List[int]:
                '''Extracts user ids from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[int]: A list of user ids.
                '''
                return [int(dct['id']) for dct in dcts]

            merged = (self.nodes['tweet'][['entities.mentions']]
                          .dropna()
                          .applymap(extract_mention)
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .explode('entities.mentions')
                          .astype({'entities.mentions': int})
                          .merge(self.nodes['user'][['user_id']]
                                     .reset_index()
                                     .rename(columns=dict(index='user_idx')),
                                 left_on='entities.mentions',
                                 right_on='user_id'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.user_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'mentions', 'user')] = rel_df

        # (:User)-[:MENTIONS]->(:User)
        user_cols = self.nodes['user'].columns
        mentions_exist = 'entities.description.mentions' in user_cols
        if self.include_mentions and mentions_exist:

            def extract_mention(dcts: List[dict]) -> List[str]:
                '''Extracts user ids from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of user ids.
                '''
                return [dct['username'] for dct in dcts]

            merged = (self.nodes['user'][['entities.description.mentions']]
                          .dropna()
                          .applymap(extract_mention)
                          .reset_index()
                          .rename(columns=dict(index='user_idx1'))
                          .explode('entities.description.mentions')
                          .merge(self.nodes['user'][['username']]
                                     .reset_index()
                                     .rename(columns=dict(index='user_idx2')),
                                 left_on='entities.description.mentions',
                                 right_on='username'))
            data_dict = dict(src=merged.user_idx1.tolist(),
                             tgt=merged.user_idx2.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'mentions', 'user')] = rel_df

        # (:Tweet)-[:HAS_IMAGE]->(:Image)
        images_exist = 'attachments.media_keys' in self.nodes['tweet'].columns
        if self.include_images and images_exist:
            merged = (self.nodes['tweet'][['attachments.media_keys']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .explode('attachments.media_keys')
                          .merge(self.nodes['image'][['media_key']]
                                     .reset_index()
                                     .rename(columns=dict(index='im_idx')),
                                 left_on='attachments.media_keys',
                                 right_on='media_key'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.im_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_image', 'image')] = rel_df

        # (:Tweet)-[:HAS_HASHTAG]->(:Hashtag)
        hashtags_exist = 'entities.hashtags' in self.nodes['tweet'].columns
        if self.include_hashtags and hashtags_exist:
            def extract_hashtag(dcts: List[dict]) -> List[str]:
                '''Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of hashtags.
                '''
                return [dct.get('tag') for dct in dcts]
            merged = (self.nodes['tweet'][['entities.hashtags']]
                          .dropna()
                          .applymap(extract_hashtag)
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .explode('entities.hashtags')
                          .merge(self.nodes['hashtag'][['tag']]
                                     .reset_index()
                                     .rename(columns=dict(index='tag_idx')),
                                 left_on='entities.hashtags',
                                 right_on='tag'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.tag_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_hashtag', 'hashtag')] = rel_df

        # (:User)-[:HAS_HASHTAG]->(:Hashtag)
        user_cols = self.nodes['user'].columns
        hashtags_exist = 'entities.description.hashtags' in user_cols
        if self.include_hashtags and hashtags_exist:
            def extract_hashtag(dcts: List[dict]) -> List[str]:
                '''Extracts hashtags from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of hashtags.
                '''
                return [dct.get('tag') for dct in dcts]
            merged = (self.nodes['user'][['entities.description.hashtags']]
                          .dropna()
                          .applymap(extract_hashtag)
                          .reset_index()
                          .rename(columns=dict(index='user_idx'))
                          .explode('entities.description.hashtags')
                          .merge(self.nodes['hashtag'][['tag']]
                                     .reset_index()
                                     .rename(columns=dict(index='tag_idx')),
                                 left_on='entities.description.hashtags',
                                 right_on='tag'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.tag_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'has_hashtag', 'hashtag')] = rel_df

        # (:Tweet)-[:HAS_URL]->(:Url)
        urls_exist = 'entities.urls' in self.nodes['tweet'].columns
        if (self.include_articles or self.include_images) and urls_exist:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            merged = (self.nodes['tweet'][['entities.urls']]
                          .dropna()
                          .applymap(extract_url)
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .explode('entities.urls')
                          .merge(self.nodes['url'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='ul_idx')),
                                 left_on='entities.urls',
                                 right_on='url'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.ul_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_URL]->(:Url)
        user_cols = self.nodes['user'].columns
        url_urls_exist = 'entities.url.urls' in user_cols
        desc_urls_exist = 'entities.description.urls' in user_cols
        if self.include_images and (url_urls_exist or desc_urls_exist):
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                '''Extracts urls from a list of dictionaries.

                Args:
                    dcts (List[dict]): A list of dictionaries.

                Returns:
                    List[str]: A list of urls.
                '''
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]

            # Initialise empty relation, which will be populated below
            rel_df = pd.DataFrame()

            if url_urls_exist:
                merged = (self.nodes['user'][['entities.url.urls']]
                              .dropna()
                              .applymap(extract_url)
                              .reset_index()
                              .rename(columns=dict(index='user_idx'))
                              .explode('entities.url.urls')
                              .merge(self.nodes['url'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='ul_idx')),
                                     left_on='entities.url.urls',
                                     right_on='url'))
                data_dict = dict(src=merged.user_idx.tolist(),
                                 tgt=merged.ul_idx.tolist())
                rel_df = (rel_df.append(pd.DataFrame(data_dict))
                                .drop_duplicates()
                                .reset_index(drop=True))

            if desc_urls_exist:
                merged = (self.nodes['user'][['entities.description.urls']]
                              .dropna()
                              .applymap(extract_url)
                              .reset_index()
                              .rename(columns=dict(index='user_idx'))
                              .explode('entities.description.urls')
                              .merge(self.nodes['url'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='ul_idx')),
                                     left_on='entities.description.urls',
                                     right_on='url'))
                data_dict = dict(src=merged.user_idx.tolist(),
                                 tgt=merged.ul_idx.tolist())
                rel_df = (rel_df.append(pd.DataFrame(data_dict))
                                .drop_duplicates()
                                .reset_index(drop=True))

            self.rels[('tweet', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_PROFILE_PICTURE_URL]->(:Url)
        user_cols = self.nodes['user'].columns
        profile_images_exist = 'profile_image_url' in user_cols
        if self.include_images and profile_images_exist:
            merged = (self.nodes['user'][['profile_image_url']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='user_idx'))
                          .merge(self.nodes['url'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='ul_idx')),
                                 left_on='profile_image_url',
                                 right_on='url'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.ul_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'has_profile_picture_url', 'url')] = rel_df

        return self

    def _extract_articles(self):
        '''Downloads the articles in the dataset'''
        if self.include_articles:
            logger.info('Extracting articles')

            # Create regex that filters out non-articles. These are common
            # images, videos and social media websites
            non_article_regexs = ['youtu[.]*be', 'vimeo', 'spotify', 'twitter',
                                  'instagram', 'tiktok', 'gab[.]com',
                                  'https://t[.]me', 'imgur', '/photo/',
                                  'mp4', 'mov', 'jpg', 'jpeg', 'bmp', 'png',
                                  'gif', 'pdf']
            non_article_regex = '(' + '|'.join(non_article_regexs) + ')'

            # Filter out the URLs to get the potential article URLs
            article_urls = [url for url in self.nodes['url'].url.tolist()
                            if re.search(non_article_regex, url) is None]

            # Loop over all the Url nodes
            data_dict = defaultdict(list)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                for result in tqdm(pool.imap_unordered(process_article_url,
                                                       article_urls,
                                                       chunksize=50),
                                   desc='Parsing articles',
                                   total=len(article_urls)):

                    # Skip result if URL is not parseable
                    if result is None:
                        continue

                    # Store the data in the data dictionary
                    data_dict['url'].append(result['url'])
                    data_dict['title'].append(result['title'])
                    data_dict['content'].append(result['content'])
                    data_dict['authors'].append(result['authors'])
                    data_dict['publish_date'].append(result['publish_date'])
                    data_dict['top_image_url'].append(result['top_image_url'])

            # Convert the data dictionary to a dataframe and store it as the
            # `Article` node
            article_df = pd.DataFrame(data_dict)
            self.nodes['article'] = article_df

            # Extract top images of the articles
            if self.include_images:

                # Create Url node for each top image url
                urls = article_df.top_image_url.dropna().tolist()
                node_df = pd.DataFrame(dict(url=urls))
                if 'url' in self.nodes.keys():
                    node_df = (self.nodes['url'].append(node_df)
                                                .drop_duplicates()
                                                .reset_index(drop=True))
                self.nodes['url'] = node_df

                # (:Article)-[:HAS_TOP_IMAGE_URL]->(:Url)
                merged = (self.nodes['article'][['top_image_url']]
                              .dropna()
                              .reset_index()
                              .rename(columns=dict(index='art_idx'))
                              .merge(self.nodes['url'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='ul_idx')),
                                     left_on='top_image_url',
                                     right_on='url'))
                data_dict = dict(src=merged.art_idx.tolist(),
                                 tgt=merged.ul_idx.tolist())
                rel_df = pd.DataFrame(data_dict)
                self.rels[('article', 'has_top_image_url', 'url')] = rel_df

            # (:Tweet)-[:HAS_ARTICLE]->(:Article)
            merged = (self.rels[('tweet', 'has_url', 'url')]
                          .rename(columns=dict(src='tweet_idx', tgt='ul_idx'))
                          .merge(self.nodes['url'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='ul_idx')),
                                 on='ul_idx')
                          .merge(self.nodes['article'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='art_idx')),
                                 on='url'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.art_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_article', 'article')] = rel_df

        return self

    def _extract_images(self):
        '''Downloads the images in the dataset'''
        if self.include_images:
            logger.info('Extracting images')

            # Start with all the URLs that have not already been parsed as
            # articles
            image_urls = [url for url in self.nodes['url'].url.tolist()
                          if url not in self.nodes['article'].url.tolist()]
            if 'image' in self.nodes.keys() and len(self.nodes['image']):
                image_urls.extend(self.nodes['image'].url.tolist())

            # Filter the resulting list of URLs using a hardcoded list of image
            # formats
            regex = '|'.join(['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'jfif',
                              'tiff', 'ppm', 'pgm', 'pbm', 'pnm', 'webp',
                              'hdr', 'heif'])
            image_urls = [url for url in image_urls
                          if re.search(regex, url) is not None]

            # Loop over all the Url nodes
            data_dict = defaultdict(list)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                for result in tqdm(pool.imap_unordered(process_image_url,
                                                       image_urls,
                                                       chunksize=50),
                                   desc='Parsing images',
                                   total=len(image_urls)):

                    # Store the data in the data dictionary if it was parseable
                    if (result is not None and
                            len(result['pixels'].shape) == 3 and
                            result['pixels'].shape[2] == 3):
                        data_dict['url'].append(result['url'])
                        data_dict['pixels'].append(result['pixels'])
                        data_dict['height'].append(result['height'])
                        data_dict['width'].append(result['width'])

            # Convert the data dictionary to a dataframe and store it as the
            # `Image` node
            image_df = pd.DataFrame(data_dict)
            self.nodes['image'] = image_df

            # (:Tweet)-[:HAS_IMAGE]->(:Image)
            rel = ('tweet', 'has_url', 'url')
            if rel in self.rels.keys() and len(self.rels[rel]):
                merged = (self.rels[rel]
                              .rename(columns=dict(src='tweet_idx',
                                                   tgt='ul_idx'))
                              .merge(self.nodes['url'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='ul_idx')),
                                     on='ul_idx')
                              .merge(self.nodes['image'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='im_idx')),
                                     on='url'))
                data_dict = dict(src=merged.tweet_idx.tolist(),
                                 tgt=merged.im_idx.tolist())
                rel_df = pd.DataFrame(data_dict)
                rel_type = ('tweet', 'has_image', 'image')
                if rel_type in self.rels.keys():
                    rel_df = (self.rels[rel_type].append(rel_df)
                                                 .drop_duplicates()
                                                 .reset_index(drop=True))
                self.rels[rel_type] = rel_df

            # (:Article)-[:HAS_TOP_IMAGE]->(:Image)
            rel = ('article', 'has_top_image_url', 'url')
            if self.include_articles and rel in self.rels.keys():
                merged = (self.rels[rel]
                              .rename(columns=dict(src='art_idx',
                                                   tgt='ul_idx'))
                              .merge(self.nodes['url'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='ul_idx')),
                                     on='ul_idx')
                              .merge(self.nodes['image'][['url']]
                                         .reset_index()
                                         .rename(columns=dict(index='im_idx')),
                                     on='url'))
                data_dict = dict(src=merged.art_idx.tolist(),
                                 tgt=merged.im_idx.tolist())
                rel_df = pd.DataFrame(data_dict)
                self.rels[('article', 'has_top_image', 'image')] = rel_df

            # (:User)-[:HAS_PROFILE_PICTURE]->(:Image)
            merged = (self.rels[('user', 'has_profile_picture_url', 'url')]
                          .rename(columns=dict(src='user_idx', tgt='ul_idx'))
                          .merge(self.nodes['url'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='ul_idx')),
                                 on='ul_idx')
                          .merge(self.nodes['image'][['url']]
                                     .reset_index()
                                     .rename(columns=dict(index='im_idx')),
                                 on='url'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.im_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'has_profile_picture', 'image')] = rel_df

        return self

    def add_embeddings(self,
                       nodes_to_embed: List[str] = ['tweet', 'reply', 'user',
                                                    'claim', 'article',
                                                    'image']):
        '''Computes, stores and dumps embeddings of node features.

        Args:
            nodes_to_embed (list of str):
                The node types which needs to be embedded. If a node type does
                not exist in the graph it will be ignored. Defaults to
                ['tweet', 'reply', 'user', 'claim', 'article', 'image'].
        '''
        # Throw error if `transformers` has not been installed
        try:
            import transformers  # noqa
            from transformers import logging as tf_logging
            tf_logging.set_verbosity_error()
        except ModuleNotFoundError:
            msg = ('You have opted to include embeddings, but you have '
                   'not installed the `transformers` library. Have you '
                   'installed the `mumin` library with the `embeddings` '
                   'extension, via `pip install mumin[embeddings]`, or via '
                   'the `dgl` extension, via `pip install mumin[dgl]`?')
            raise ModuleNotFoundError(msg)

        # Embed tweets
        if ('tweet' in nodes_to_embed and
                not 'text_emb' in self.nodes['tweet'].columns):
            self._embed_tweets()
            self._dump_dataset()

        # Embed replies
        if ('reply' in nodes_to_embed:
                not 'text_emb' in self.nodes['reply'].columns):
            self._embed_replies()
            self._dump_dataset()

        # Embed users
        if ('user' in nodes_to_embed:
                not 'description_emb' in self.nodes['user'].columns):
            self._embed_users()
            self._dump_dataset()

        # Embed articles
        if ('article' in nodes_to_embed:
                not 'content_emb' in self.nodes['article'].columns):
            self._embed_articles()
            self._dump_dataset()

        # Embed images
        if ('image' in nodes_to_embed:
                not 'pixels_emb' in self.nodes['image'].columns):
            self._embed_images()
            self._dump_dataset()

        # Embed claims
        if ('claim' in nodes_to_embed:
                not 'reviewer_emb' in self.nodes['claim'].columns):
            self._embed_claims()
            self._dump_dataset()

        return self

    @staticmethod
    def _embed_text(text: str, tokenizer, model) -> np.ndarray:
        '''Extract a text embedding.

        Args:
            text (str): The text to embed.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            model (transformers.PreTrainedModel): The model to use.

        Returns:
            np.ndarray: The embedding of the text.
        '''
        import torch
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, return_tensors='pt')
            result = model(**inputs)
            return result.pooler_output[0].numpy()

    def _embed_tweets(self):
        '''Embeds all the tweets in the dataset'''
        from transformers import AutoModel, AutoTokenizer

        logger.info('Embedding tweets')

        # Load text embedding model
        model_id = self.text_embedding_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        # Define embedding function
        embed = partial(self._embed_text, tokenizer=tokenizer, model=model)

        # Embed tweet text using the pretrained transformer
        text_embs = self.nodes['tweet'].text.progress_apply(embed)
        self.nodes['tweet']['text_emb'] = text_embs

        # Embed tweet language using a one-hot encoding
        languages = self.nodes['tweet'].lang.tolist()
        one_hotted = [np.asarray(lst)
                      for lst in pd.get_dummies(languages).to_numpy().tolist()]
        self.nodes['tweet']['lang_emb'] = one_hotted

        return self

    def _embed_replies(self):
        '''Embeds all the replies in the dataset'''
        from transformers import AutoModel, AutoTokenizer

        logger.info('Embedding replies')

        # Load text embedding model
        model_id = self.text_embedding_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        # Define embedding function
        embed = partial(self._embed_text, tokenizer=tokenizer, model=model)

        # Embed tweet text using the pretrained transformer
        text_embs = self.nodes['reply'].text.progress_apply(embed)
        self.nodes['reply']['text_emb'] = text_embs

        # Embed tweet language using a one-hot encoding
        languages = self.nodes['reply'].lang.tolist()
        one_hotted = [np.asarray(lst)
                      for lst in pd.get_dummies(languages).to_numpy().tolist()]
        self.nodes['reply']['lang_emb'] = one_hotted

        return self

    def _embed_users(self):
        '''Embeds all the users in the dataset'''
        from transformers import AutoModel, AutoTokenizer

        logger.info('Embedding users')

        # Load text embedding model
        model_id = self.text_embedding_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        # Define embedding function
        def embed(text: str):
            '''Extract a text embedding'''
            if text != text:
                return np.zeros(model.config.hidden_size)
            else:
                return self._embed_text(text, tokenizer=tokenizer, model=model)

        # Embed user description using the pretrained transformer
        desc_embs = self.nodes['user'].description.progress_apply(embed)
        self.nodes['user']['description_emb'] = desc_embs

        return self

    def _embed_articles(self):
        '''Embeds all the tweets in the dataset'''
        if self.include_articles:
            from transformers import AutoModel, AutoTokenizer

            logger.info('Embedding articles')

            # Load text embedding model
            model_id = self.text_embedding_model_id
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)

            # Define embedding function
            def embed(text: Union[str, List[str]]):
                '''Extract a text embedding'''
                params = dict(tokenizer=tokenizer, model=model)
                if isinstance(text, str):
                    return self._embed_text(text, **params)
                else:
                    return np.mean([self._embed_text(doc, **params)
                                    for doc in text], axis=0)

            def split_content(doc: str) -> List[str]:
                '''Split up a string into smaller chunks'''
                if '.' in doc:
                    return doc.split('.')
                else:
                    end = min(len(doc) - 1000, 0)
                    return [doc[i:i+1000]
                            for i in range(0, end, 1000)] + [doc[end:-1]]

            # Embed titles using the pretrained transformer
            title_embs = self.nodes['article'].title.progress_apply(embed)
            self.nodes['article']['title_emb'] = title_embs

            # Embed contents using the pretrained transformer
            contents = self.nodes['article'].content
            content_embs = contents.map(split_content).progress_apply(embed)
            self.nodes['article']['content_emb'] = content_embs

        return self

    def _embed_images(self):
        '''Embeds all the images in the dataset'''
        if self.include_images:
            from transformers import (AutoFeatureExtractor,
                                      AutoModelForImageClassification)
            import torch

            logger.info('Embedding images')

            # Load image embedding model
            model_id = self.image_embedding_model_id
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)

            # Define embedding function
            def embed(image):
                '''Extract the last hiden state of image model'''
                with torch.no_grad():

                    # Ensure that the input has shape (C, H, W)
                    image = np.transpose(image, (2, 0, 1))

                    # Extract the features to be used in the model
                    inputs = feature_extractor(images=image,
                                               return_tensors='pt')

                    # Get the embedding
                    outputs = model(**inputs, output_hidden_states=True)
                    penultimate_embedding = outputs.hidden_states[-1]
                    cls_embedding = penultimate_embedding[0, 0, :]

                    # Convert to NumPy and return
                    return cls_embedding.numpy()

            # Embed pixels using the pretrained transformer
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.nodes['image']['pixels_emb'] = (self.nodes['image']
                                                         .pixels
                                                         .progress_apply(embed)
                                                         .tolist())

        return self

    def _embed_claims(self):
        '''Embeds all the claims in the dataset'''
        logger.info('Embedding claims')

        # Ensure that `reviewers` is a list
        if isinstance(self.nodes['claim'].reviewers.iloc[0], str):

            def string_to_list(string: str) -> list:
                '''Convert a string to a list.

                Args:
                    string: A string to be converted to a list.

                Returns:
                    list: A list of strings.
                '''
                string = string.replace('\'', '\"')
                return json.loads(string)

            self.nodes['claim']['reviewers'] = (self.nodes['claim']
                                                    .reviewers
                                                    .map(string_to_list))

        # Set up one-hot encoding of claim reviewers
        reviewers = self.nodes['claim'].reviewers.explode().unique().tolist()
        one_hotted = [np.asarray(lst)
                      for lst in pd.get_dummies(reviewers).to_numpy().tolist()]
        one_hot_dict = {reviewer: array
                        for reviewer, array in zip(reviewers, one_hotted)}

        def embed_reviewers(revs: List[str]) -> np.ndarray:
            '''One-hot encoding of multiple reviewers.

            Args:
                revs: A list of reviewers.

            Returns:
                np.ndarray: A one-hot encoded array.
            '''
            arrays = [one_hot_dict[rev] for rev in revs]
            return np.stack(arrays, axis=0).sum(axis=0)

        # Embed claim reviewer using a one-hot encoding
        reviewer_emb = self.nodes['claim'].reviewers.map(embed_reviewers)
        self.nodes['claim']['reviewer_emb'] = reviewer_emb

        return self

    def _filter_node_features(self):
        '''Filters the node features to avoid redundancies and noise'''
        logger.info('Filters node features')

        # Set up the node features that should be kept
        size = 'small' if self.size == 'test' else self.size
        node_feats = dict(claim=['embedding', 'label', 'reviewers', 'date',
                                 'language', 'keywords', 'cluster_keywords',
                                 'cluster',
                                 f'{size}_train_mask',
                                 f'{size}_val_mask',
                                 f'{size}_test_mask'],
                          tweet=['tweet_id', 'text', 'created_at', 'lang',
                                 'source', 'public_metrics.retweet_count',
                                 'public_metrics.reply_count',
                                 'public_metrics.quote_count', 'label',
                                 f'{size}_train_mask',
                                 f'{size}_val_mask',
                                 f'{size}_test_mask'],
                          reply=['tweet_id', 'text', 'created_at', 'lang',
                                 'source', 'public_metrics.retweet_count',
                                 'public_metrics.reply_count',
                                 'public_metrics.quote_count'],
                          user=['user_id', 'verified', 'protected',
                                'created_at', 'username', 'description', 'url',
                                'name', 'public_metrics.followers_count',
                                'public_metrics.following_count',
                                'public_metrics.tweet_count',
                                'public_metrics.listed_count', 'location'],
                          image=['url', 'pixels', 'width', 'height'],
                          article=['url', 'title', 'content'],
                          place=['place_id', 'name', 'full_name',
                                 'country_code', 'country', 'place_type',
                                 'lat', 'lng'],
                          hashtag=['tag'],
                          poll=['poll_id', 'labels', 'votes', 'end_datetime',
                                'voting_status', 'duration_minutes'])

        # Set up renaming of node features that should be kept
        node_feat_renaming = {
            'public_metrics.retweet_count': 'num_retweets',
            'public_metrics.reply_count': 'num_replies',
            'public_metrics.quote_count': 'num_quote_tweets',
            'public_metrics.followers_count': 'num_followers',
            'public_metrics.following_count': 'num_followees',
            'public_metrics.tweet_count': 'num_tweets',
            'public_metrics.listed_count': 'num_listed',
            f'{size}_train_mask': 'train_mask',
            f'{size}_val_mask': 'val_mask',
            f'{size}_test_mask': 'test_mask'
        }

        # Filter and rename the node features
        for node_type, features in node_feats.items():
            if node_type in self.nodes.keys():
                filtered_feats = [feat for feat in features
                                  if feat in self.nodes[node_type].columns]
                renaming_dict = {old: new
                                 for old, new in node_feat_renaming.items()
                                 if old in features}
                self.nodes[node_type] = (self.nodes[node_type][filtered_feats]
                                             .rename(columns=renaming_dict))

        return self

    def _filter_relations(self):
        '''Filters the relations to only include node IDs that exist'''
        logger.info('Filters relations')

        # Remove article relations if they are not included
        if not self.include_articles:
            rels_to_pop = list()
            for rel_type in self.rels.keys():
                src, _, tgt = rel_type
                if src == 'article' or tgt == 'article':
                    rels_to_pop.append(rel_type)
            for rel_type in rels_to_pop:
                self.rels.pop(rel_type)

        # Remove reply relations if they are not included
        if not self.include_replies:
            rels_to_pop = list()
            for rel_type in self.rels.keys():
                src, _, tgt = rel_type
                if src == 'reply' or tgt == 'reply':
                    rels_to_pop.append(rel_type)
            for rel_type in rels_to_pop:
                self.rels.pop(rel_type)

        # Remove mention relations if they are not included
        if not self.include_mentions:
            rels_to_pop = list()
            for rel_type in self.rels.keys():
                _, rel, _ = rel_type
                if rel == 'mentions':
                    rels_to_pop.append(rel_type)
            for rel_type in rels_to_pop:
                self.rels.pop(rel_type)

        # Loop over the relations, extract the associated node IDs and filter
        # the relation dataframe to only include relations between nodes that
        # exist
        rels_to_pop = list()
        for rel_type, rel_df in self.rels.items():
            src, _, tgt = rel_type
            if src not in self.nodes.keys() or tgt not in self.nodes.keys():
                rels_to_pop.append(rel_type)
            else:
                src_ids = self.nodes[src].index.tolist()
                tgt_ids = self.nodes[tgt].index.tolist()
                rel_df = rel_df[rel_df.src.isin(src_ids)]
                rel_df = rel_df[rel_df.tgt.isin(tgt_ids)]
                self.rels[rel_type] = rel_df

        for rel_type in rels_to_pop:
            self.rels.pop(rel_type)

    def _remove_auxilliaries(self):
        '''Removes node types that are not in use anymore'''
        # Remove auxilliary node types
        nodes_to_remove = [node_type for node_type in self.nodes.keys()
                           if node_type not in self._node_dump]
        for node_type in nodes_to_remove:
            self.nodes.pop(node_type)

        # Remove auxilliary relation types
        rels_to_remove = [rel_type for rel_type in self.rels.keys()
                          if rel_type not in self._rel_dump]
        for rel_type in rels_to_remove:
            self.rels.pop(rel_type)

        return self

    def _remove_islands(self):
        '''Removes nodes and relations that are not connected to anything'''
        # Loop over all the node types
        for node_type, node_df in self.nodes.items():

            # For each node type, loop over all the relations, to see what
            # nodes of that node type does not appear in any of the relations
            for rel_type, rel_df in self.rels.items():
                src, _, tgt = rel_type

                # If the node is the source of the relation
                if node_type == src:

                    # Store all the nodes connected to the relation (or any of
                    # the previously checked relations)
                    connected = node_df.index.isin(rel_df.src.tolist())
                    if 'connected' in node_df.columns:
                        connected = (node_df.connected | connected)
                    node_df['connected'] = connected

                # If the node is the source of the relation
                if node_type == tgt:

                    # Store all the nodes connected to the relation (or any of
                    # the previously checked relations)
                    connected = node_df.index.isin(rel_df.tgt.tolist())
                    if 'connected' in node_df.columns:
                        connected = (node_df.connected | connected)
                    node_df['connected'] = connected

            # Filter the node dataframe to only keep the connected ones
            if ('connected' in node_df.columns and
                    'index' not in node_df.columns):
                self.nodes[node_type] = (node_df.query('connected == True')
                                                .drop(columns='connected')
                                                .reset_index())

            # Update the relevant relations
            for rel_type, rel_df in self.rels.items():
                src, _, tgt = rel_type

                # If islands have been removed from the source, then update
                # those indices
                if node_type == src and 'index' in self.nodes[node_type]:
                    node_df = (self.nodes[node_type]
                                   .rename(columns=dict(index='old_idx'))
                                   .reset_index())

                    rel_df = (rel_df.merge(node_df[['index', 'old_idx']],
                                           left_on='src',
                                           right_on='old_idx')
                                    .drop(columns=['src', 'old_idx'])
                                    .rename(columns=dict(index='src')))
                    self.rels[rel_type] = rel_df[['src', 'tgt']]
                    self.nodes[node_type] = self.nodes[node_type]

                # If islands have been removed from the target, then update
                # those indices
                if node_type == tgt and 'index' in self.nodes[node_type]:
                    node_df = (self.nodes[node_type]
                                   .rename(columns=dict(index='old_idx'))
                                   .reset_index())
                    rel_df = (rel_df.merge(node_df[['index', 'old_idx']],
                                           left_on='tgt',
                                           right_on='old_idx')
                                    .drop(columns=['tgt', 'old_idx'])
                                    .rename(columns=dict(index='tgt')))
                    self.rels[rel_type] = rel_df[['src', 'tgt']]
                    self.nodes[node_type] = self.nodes[node_type]

            if 'index' in self.nodes[node_type]:
                self.nodes[node_type] = (self.nodes[node_type]
                                             .drop(columns='index'))

        return self

    def _dump_dataset(self):
        '''Dumps the dataset to a zip file'''
        logger.info('Dumping dataset')

        # Create a temporary pickle folder
        temp_pickle_folder = Path('temp_pickle_folder')
        if not temp_pickle_folder.exists():
            temp_pickle_folder.mkdir()

        # Make temporary pickle list
        pickle_list = list()

        # Create progress bar
        total = len(self._node_dump) + len(self._rel_dump) + 1
        pbar = tqdm(total=total)

        # Store the nodes
        for node_type in self._node_dump:
            pbar.set_description(f'Storing {node_type} nodes')
            if node_type in self.nodes.keys():
                pickle_list.append(node_type)
                pickle_path = temp_pickle_folder / f'{node_type}.pickle'
                self.nodes[node_type].to_pickle(pickle_path, protocol=4)
            pbar.update()

        # Store the relations
        for rel_type in self._rel_dump:
            pbar.set_description(f'Storing {rel_type} relations')
            if rel_type in self.rels.keys():
                name = '_'.join(rel_type)
                pickle_list.append(name)
                pickle_path = temp_pickle_folder / f'{name}.pickle'
                self.rels[rel_type].to_pickle(pickle_path, protocol=4)
            pbar.update()

        # Zip the nodes and relations, and save the zip file
        with zipfile.ZipFile(self.dataset_path,
                             mode='w',
                             compression=zipfile.ZIP_STORED) as zip_file:
            pbar.set_description('Dumping dataset')
            for name in pickle_list:
                fname = f'{name}.pickle'
                zip_file.write(temp_pickle_folder / fname, arcname=fname)

        # Remove the temporary pickle folder
        rmtree(temp_pickle_folder)

        # Final progress bar update and close it
        pbar.update()
        pbar.close()

        return self

    def to_dgl(self):
        '''Convert the dataset to a DGL dataset.

        Returns:
            DGLHeteroGraph:
                The graph in DGL format.
        '''
        logger.info('Outputting to DGL')
        return build_dgl_dataset(nodes=self.nodes, relations=self.rels)
