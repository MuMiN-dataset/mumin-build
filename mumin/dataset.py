'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd
import numpy as np
import logging
import requests
import zipfile
import io
import shutil
from collections import defaultdict
import re
import multiprocessing as mp
from tqdm.auto import tqdm
import warnings

from .twitter import Twitter
from .article import process_article_url
from .image import process_image_url
from .dgl import build_dgl_dataset


# Set up logging
logger = logging.getLogger(__name__)


# Allows progress bars with `pd.DataFrame.progress_apply`
tqdm.pandas()


class MuminDataset:
    '''The MuMiN misinformation dataset, from [1].

    Args:
        twitter_bearer_token (str):
            The Twitter bearer token.
        size (str, optional):
            The size of the dataset. Can be either 'small', 'medium' or
            'large'. Defaults to 'large'.
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
        include_places (bool, optional):
            Whether to include places in the dataset. Defaults to True.
        include_polls (bool, optional):
            Whether to include polls in the dataset. Defaults to True.
        text_embedding_model_id (str, optional):
            The HuggingFace Hub model ID to use when embedding texts. Defaults
            to 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
        image_embedding_model_id (str, optional):
            The HuggingFace Hub model ID to use when embedding images. Defaults
            to 'facebook/deit-base-distilled-patch16-224'.
        dataset_dir (str or pathlib Path, optional):
            The path to the folder where the dataset should be stored. Defaults
            to './mumin'.
        verbose (bool, optional):
            Whether extra information should be outputted. Defaults to False.

    Attributes:
        include_replies (bool): Whether to include replies in the dataset.
        include_articles (bool): Whether to include articles in the dataset.
        include_images (bool): Whether to include images in the dataset.
        include_hashtags (bool): Whether to include hashtags in the dataset.
        include_mentions (bool): Whether to include mentions in the dataset.
        include_places (bool): Whether to include places in the dataset.
        include_polls (bool): Whether to include polls in the dataset.
        size (str): The size of the dataset.
        dataset_dir (pathlib Path): The dataset directory.
        text_embedding_model_id (str): The model ID used for embedding text.
        image_embedding_model_id (str): The model ID used for embedding images.
        nodes (dict): The nodes of the dataset.
        rels (dict): The relations of the dataset.

    References:
        - [1] Nielsen and McConville: _MuMiN: A Large-Scale Multilingual
              Multimodal Fact-Checked Misinformation Dataset with Linked Social
              Network Posts_ (2021)
    '''

    download_url: str = ('https://github.com/CLARITI-REPHRAIN/mumin-build/'
                         'raw/main/data/mumin.zip')
    _node_dump: List[str] = ['claim', 'tweet', 'user', 'image', 'article',
                             'place', 'hashtag', 'poll', 'reply']
    _rel_dump: List[Tuple[str, str, str]] = [
        ('tweet', 'discusses', 'claim'),
        ('tweet', 'mentions', 'user'),
        ('tweet', 'located_in', 'place'),
        ('tweet', 'has_image', 'image'),
        ('tweet', 'has_hashtag', 'hashtag'),
        ('tweet', 'has_article', 'article'),
        ('tweet', 'has_poll', 'poll'),
        ('reply', 'reply_to', 'tweet'),
        ('reply', 'reply_to', 'reply'),
        ('reply', 'quote_of', 'tweet'),
        ('user', 'posted', 'tweet'),
        ('user', 'mentions', 'user'),
        ('user', 'has_pinned', 'tweet'),
        ('user', 'has_hashtag', 'hashtag'),
        ('user', 'has_profile_picture', 'image'),
        ('user', 'retweeted', 'tweet'),
        ('user', 'liked', 'tweet'),
        ('user', 'follows', 'user'),
        ('article', 'has_top_image', 'image'),
    ]

    def __init__(self,
                 twitter_bearer_token: str,
                 size: str = 'large',
                 include_replies: bool = True,
                 include_articles: bool = True,
                 include_images: bool = True,
                 include_hashtags: bool = True,
                 include_mentions: bool = True,
                 include_places: bool = True,
                 include_polls: bool = True,
                 text_embedding_model_id: str = ('sentence-transformers/'
                                                 'paraphrase-multilingual-'
                                                 'MiniLM-L12-v2'),
                 image_embedding_model_id: str = ('facebook/deit-base-'
                                                  'distilled-patch16-224'),
                 dataset_dir: Union[str, Path] = './mumin',
                 verbose: bool = False):
        self._twitter = Twitter(twitter_bearer_token=twitter_bearer_token)
        self.size = size
        self.include_replies = include_replies
        self.include_articles = include_articles
        self.include_images = include_images
        self.include_hashtags = include_hashtags
        self.include_mentions = include_mentions
        self.include_places = include_places
        self.include_polls = include_polls
        self.text_embedding_model_id = text_embedding_model_id
        self.image_embedding_model_id = image_embedding_model_id
        self.dataset_dir = Path(dataset_dir)
        self.verbose = verbose
        self.nodes: Dict[str, pd.DataFrame] = dict()
        self.rels: Dict[Tuple[str, str, str], pd.DataFrame] = dict()

        # Set up logging verbosity
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Raise warning
        warnings.warn('This dataset is currently under review at NeuRIPS 2021 '
                      'Datasets and Benchmarks Track (Round 2). This dataset '
                      'must not be used until this warning is removed, as the '
                      'dataset is subject to change, for example, during the '
                      'review period.')

    def __repr__(self) -> str:
        '''A string representation of the dataaset.

        Returns:
            str: The representation of the dataset.
        '''
        if len(self.nodes) == 0 or len(self.rels) == 0:
            return f'MuminDataset(size={self.size}, compiled=False)'
        else:
            num_nodes = sum([len(df) for df in self.nodes.values()])
            num_rels = sum([len(df) for df in self.rels.values()])
            return (f'MuminDataset(num_nodes={num_nodes:,}, '
                    f'num_relations={num_rels:,}, '
                    f'size=\'{self.size}\', '
                    f'compiled=False)')

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

        # Only compile the dataset if it has not already been compiled
        if 'text' not in self.nodes['tweet'].columns:
            self._shrink_dataset()
            self._rehydrate(node_type='tweet')
            self._rehydrate(node_type='reply')
            self._update_precomputed_ids()
            self._extract_nodes()
            self._extract_relations()
            self._extract_articles()
            self._extract_images()
            self._filter_node_features()
            self._filter_relations()
            self._remove_auxilliaries()
            self._remove_islands()
            self._dump_to_csv()

        return self

    def _download(self, overwrite: bool = False):
        '''Downloads and unzips the dataset.

        Args:
            overwrite (bool, optional):
                Whether the dataset directory should be overwritten, in case it
                already exists. Defaults to False.
        '''
        if (not self.dataset_dir.exists() or
                (self.dataset_dir.exists() and overwrite)):

            logger.info('Downloading dataset')

            # Remove existing directory if we are overwriting
            if self.dataset_dir.exists() and overwrite:
                shutil.rmtree(self.dataset_dir)

            response = requests.get(self.download_url)

            # If the response was unsuccessful then raise an error
            if response.status_code != 200:
                msg = f'[{response.status_code}] {response.content}'
                raise RuntimeError(msg)

            # Otherwise unzip the in-memory zip file to `self.dataset_dir`
            else:
                zipped = response.content
                with zipfile.ZipFile(io.BytesIO(zipped)) as zip_file:
                    zip_file.extractall(self.dataset_dir)

        return self

    def _load_dataset(self):
        '''Loads the dataset files into memory.

        Raises:
            RuntimeError:
                If the dataset has not been downloaded yet.
        '''
        # Raise error if the dataset has not been downloaded yet
        if not self.dataset_dir.exists():
            raise RuntimeError('Dataset has not been downloaded yet!')

        logger.info('Loading dataset')

        # Reset `nodes` and `relations` to ensure a fresh start
        self.nodes = dict()
        self.rels = dict()

        # Loop over the files in the dataset directory
        csv_paths = [path for path in self.dataset_dir.iterdir()
                     if str(path)[-4:] == '.csv']
        for path in csv_paths:
            fname = path.stem

            # Node case: no underscores in file name
            if len(fname.split('_')) == 1:
                self.nodes[fname] = pd.read_csv(path)

            # Relation case: exactly two underscores in file name
            elif len(fname.split('_')) > 2:
                splits = fname.split('_')
                src = splits[0]
                tgt = splits[-1]
                rel = '_'.join(splits[1:-1])
                self.rels[(src, rel, tgt)] = pd.read_csv(path)

            # Otherwise raise error
            else:
                raise RuntimeError(f'Could not recognise {fname} as a node '
                                   f'or relation.')

        # Ensure that claims are present in the dataset
        if 'claim' not in self.nodes.keys():
            raise RuntimeError('No claims are present in the zipfile!')

        # Ensure that tweets are present in the dataset, and also that the
        # tweet IDs are unique
        if 'tweet' not in self.nodes.keys():
            raise RuntimeError('No tweets are present in the zipfile!')
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
                threshold = 0.99

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
            reply_rel = reply_rel[include]
            self.rels[('reply', 'reply_to', 'tweet')] = reply_rel

            # Filter (:Reply)-[:QUOTE_OF]->(:Tweet)
            quote_rel = self.rels[('reply', 'quote_of', 'tweet')]
            include = quote_rel.tgt.isin(tweet_df.tweet_id.tolist())
            quote_rel = quote_rel[include]
            include = quote_rel.tgt.isin(tweet_df.tweet_id.tolist())
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
        '''Rehydrate the tweets and users in the dataset'''

        if (node_type in self.nodes.keys() and
                (node_type != 'reply' or self.include_replies)):

            logger.info(f'Rehydrating {node_type} nodes')

            # Get the tweet IDs
            tweet_ids = self.nodes[node_type].tweet_id.tolist()

            # Rehydrate the tweets
            tweet_dfs = self._twitter.rehydrate_tweets(tweet_ids=tweet_ids)

            # Extract and store tweets and users
            self.nodes[node_type] = tweet_dfs['tweets']
            if ('user' in self.nodes.keys() and
                    'username' in self.nodes['user'].columns):
                user_df = self.nodes['user'].append(tweet_dfs['users'])
            else:
                user_df = tweet_dfs['users']
            self.nodes['user'] = user_df

            # Extract and store images
            if self.include_images and len(tweet_dfs['media']):
                video_query = '(type == "video") or (type == "animated gif")'
                video_df = (tweet_dfs['media']
                            .query(video_query)
                            .drop(columns=['url', 'duration_ms',
                                           'public_metrics.view_count'])
                            .rename(columns=dict(preview_image_url='url')))
                image_df = (tweet_dfs['media']
                            .query('type == "photo"')
                            .append(video_df))

                if 'media' in self.nodes.keys():
                    image_df = self.nodes['media'].append(image_df)
                self.nodes['image'] = image_df

            # Extract and store polls
            if self.include_polls and len(tweet_dfs['polls']):
                if 'poll' in self.nodes.keys():
                    poll_df = self.nodes['poll'].append(tweet_dfs['polls'])
                else:
                    poll_df = tweet_dfs['polls']
                self.nodes['poll'] = poll_df

            # Extract and store places
            if self.include_places and len(tweet_dfs['places']):
                if 'place' in self.nodes.keys():
                    place_df = self.nodes['place'].append(tweet_dfs['places'])
                else:
                    place_df = tweet_dfs['places']
                self.nodes['place'] = place_df

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
                    node_df = (self.nodes['hashtag'].append(node_df)
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
                    node_df = (self.nodes['hashtag'].append(node_df)
                                                    .drop_duplicates()
                                                    .reset_index(drop=True))
                self.nodes['hashtag'] = node_df

        # Add urls from tweets
        if 'entities.urls' in self.nodes['tweet'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['tweet']['entities.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url'].append(node_df)
                                            .drop_duplicates()
                                            .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from user urls
        if 'entities.url.urls' in self.nodes['user'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['entities.url.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url'].append(node_df)
                                            .drop_duplicates()
                                            .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from user descriptions
        if 'entities.description.urls' in self.nodes['user'].columns:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['entities.description.urls']
                        .dropna()
                        .map(extract_url)
                        .explode()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url'].append(node_df)
                                            .drop_duplicates()
                                            .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from profile pictures
        if (self.include_images and
                'profile_image_url' in self.nodes['user'].columns):
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['user']['profile_image_url']
                        .dropna()
                        .tolist())
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url'].append(node_df)
                                            .drop_duplicates()
                                            .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add urls from articles
        if self.include_articles:
            urls = self.nodes['article'].url.dropna().tolist()
            node_df = pd.DataFrame(dict(url=urls))
            if 'url' in self.nodes.keys():
                node_df = (self.nodes['url'].append(node_df)
                                            .drop_duplicates()
                                            .reset_index(drop=True))
            self.nodes['url'] = node_df

        # Add place features
        if self.include_places and 'place' in self.nodes.keys():

            def get_lat(bbox: list) -> float:
                return (bbox[1] + bbox[3]) / 2

            def get_lng(bbox: list) -> float:
                return (bbox[0] + bbox[2]) / 2

            place_df = self.nodes['place']
            place_df['lat'] = place_df['geo.bbox'].map(get_lat)
            place_df['lng'] = place_df['geo.bbox'].map(get_lng)
            self.nodes['place'] = place_df

        # Add poll features
        if self.include_polls and 'poll' in self.nodes.keys():

            def get_labels(options: List[dict]) -> List[str]:
                return [dct['label'] for dct in options]

            def get_votes(options: List[dict]) -> List[int]:
                return [dct['votes'] for dct in options]

            poll_df = self.nodes['poll']
            poll_df['labels'] = poll_df.options.map(get_labels)
            poll_df['votes'] = poll_df.options.map(get_votes)
            self.nodes['poll'] = poll_df

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

        # (:Tweet)-[:MENTIONS]->(:User)
        mentions_exist = 'entities.mentions' in self.nodes['tweet'].columns
        if self.include_mentions and mentions_exist:

            def extract_mention(dcts: List[dict]) -> List[int]:
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

        # (:User)-[:HAS_PINNED]->(:Tweet)
        pinned_exist = 'pinned_tweet_id' in self.nodes['user'].columns
        if pinned_exist:
            merged = (self.nodes['user'][['pinned_tweet_id']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='user_idx'))
                          .astype({'pinned_tweet_id': int})
                          .merge(self.nodes['tweet'][['tweet_id']]
                                     .reset_index()
                                     .rename(columns=dict(index='tweet_idx')),
                                 left_on='pinned_tweet_id',
                                 right_on='tweet_id'))
            data_dict = dict(src=merged.user_idx.tolist(),
                             tgt=merged.tweet_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'has_pinned', 'tweet')] = rel_df

        # (:Tweet)-[:LOCATED_IN]->(:Place)
        places_exist = 'geo.place_id' in self.nodes['tweet'].columns
        if self.include_places and places_exist:
            merged = (self.nodes['tweet'][['geo.place_id']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .merge(self.nodes['place'][['place_id']]
                                     .reset_index()
                                     .rename(columns=dict(index='place_idx')),
                                 left_on='geo.place_id',
                                 right_on='place_id'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.place_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'located_in', 'place')] = rel_df

        # (:Tweet)-[:HAS_POLL]->(:Poll)
        polls_exist = 'attachments.poll_ids' in self.nodes['tweet'].columns
        if self.include_polls and polls_exist:
            merged = (self.nodes['tweet'][['attachments.poll_ids']]
                          .dropna()
                          .reset_index()
                          .rename(columns=dict(index='tweet_idx'))
                          .explode('attachments.poll_ids')
                          .astype({'attachments.poll_ids': int})
                          .merge(self.nodes['poll'][['poll_id']]
                                     .reset_index()
                                     .rename(columns=dict(index='poll_idx')),
                                 left_on='attachments.poll_ids',
                                 right_on='poll_id'))
            data_dict = dict(src=merged.tweet_idx.tolist(),
                             tgt=merged.poll_idx.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_poll', 'poll')] = rel_df

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
                rel_df = rel_df.append(pd.DataFrame(data_dict))

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
                rel_df = rel_df.append(pd.DataFrame(data_dict))

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
                                                       chunksize=5),
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
                                                       chunksize=5),
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
            merged = (self.rels[('tweet', 'has_url', 'url')]
                          .rename(columns=dict(src='tweet_idx', tgt='ul_idx'))
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
                                             .reset_index())
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
                       nodes_to_embed: List[str] = ['tweet', 'user', 'claim',
                                                    'article', 'image']):
        '''Computes, stores and dumps embeddings of node features.

        Args:
            nodes_to_embed (list of str):
                The node types which needs to be embedded. If a node type does
                not exist in the graph it will be ignored. Defaults to
                ['tweet', 'user', 'claim', 'article', 'image'].
        '''
        # Throw error if `transformers` has not been installed
        try:
            import transformers  # noqa
        except ModuleNotFoundError:
            msg = ('You have opted to include embeddings, but you have '
                   'not installed the `transformers` library. Have you '
                   'installed the `mumin` library with the `embeddings` '
                   'extension, via `pip install mumin[embeddings]`, or via '
                   'the `dgl` extension, via `pip install mumin[dgl]`?')
            raise ModuleNotFoundError(msg)

        # Embed tweets
        if 'tweet' in nodes_to_embed:
            self._embed_tweets()

        # Embed users
        if 'user' in nodes_to_embed:
            self._embed_users()

        # Embed articles
        if 'article' in nodes_to_embed:
            self._embed_articles()

        # Embed images
        if 'image' in nodes_to_embed:
            self._embed_images()

        # Embed claims
        if 'claim' in nodes_to_embed:
            self._embed_claims()

        # Dump the nodes with all the embeddings
        self._dump_to_csv()

        return self

    def _embed_tweets(self):
        '''Embeds all the tweets in the dataset'''
        import transformers

        logger.info('Embedding tweets')

        # Load text embedding model
        model_id = self.text_embedding_model_id
        pipeline = transformers.pipeline(task='feature-extraction',
                                         model=model_id,
                                         tokenizer=model_id)

        # Define embedding function
        def embed(text: str):
            '''Extract a text embedding'''
            return np.asarray(pipeline(text))[0, 0, :]

        # Embed tweet text using the pretrained transformer
        text_embs = self.nodes['tweet'].text.progress_apply(embed)
        self.nodes['tweet']['text_emb'] = text_embs

        # Embed tweet language using a one-hot encoding
        languages = self.nodes['tweet'].lang.tolist()
        one_hotted = [np.asarray(lst)
                      for lst in pd.get_dummies(languages).to_numpy().tolist()]
        self.nodes['tweet']['lang_emb'] = one_hotted

        return self

    def _embed_users(self):
        '''Embeds all the users in the dataset'''
        import transformers

        logger.info('Embedding users')

        # Load text embedding model
        model_id = self.text_embedding_model_id
        pipeline = transformers.pipeline(task='feature-extraction',
                                         model=model_id,
                                         tokenizer=model_id)

        # Define embedding function
        def embed(text: Union[float, str]):
            '''Extract a text embedding'''
            if text != text:
                return None
            else:
                return np.asarray(pipeline(text))[0, 0, :]

        # Embed user description using the pretrained transformer
        desc_embs = self.nodes['user'].description.progress_apply(embed)
        emb_dim = desc_embs.dropna().iloc[0].shape
        desc_embs[desc_embs.isna()] = np.zeros(emb_dim)
        self.nodes['user']['description_emb'] = desc_embs

        return self

    def _embed_articles(self):
        '''Embeds all the tweets in the dataset'''
        if self.include_articles:
            import transformers

            logger.info('Embedding articles')

            # Load text embedding model
            model_id = self.text_embedding_model_id
            pipeline = transformers.pipeline(task='feature-extraction',
                                             model=model_id,
                                             tokenizer=model_id)

            # Define embedding function
            def embed(text: Union[str, List[str]]):
                '''Extract a text embedding'''
                if isinstance(text, str):
                    return np.asarray(pipeline(text))[0, 0, :]
                else:
                    arrays = [np.asarray(pipeline(doc))[0, 0, :]
                              for doc in text]
                    return np.mean(arrays)

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

                    # Get the embedding, being the last hidden state of the
                    # model (we return the first sequence element, as this
                    # corresponds to the [HEAD] tag)
                    outputs = model(**inputs,
                                    output_hidden_states=True,
                                    output_attentions=True)
                    torch_embedding = outputs.hidden_states[-1][0, 0, :]

                    # Convert to NumPy and return
                    return torch_embedding.numpy()

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

        # Embed claim reviewer using a one-hot encoding
        reviewers = self.nodes['claim'].reviewer.tolist()
        one_hotted = [np.asarray(lst)
                      for lst in pd.get_dummies(reviewers).to_numpy().tolist()]
        self.nodes['claim']['reviewer_emb'] = one_hotted

        return self

    def _filter_node_features(self):
        '''Filters the node features to avoid redundancies and noise'''
        logger.info('Filters node features')

        # Set up the node features that should be kept
        node_feats = dict(claim=['raw_verdict', 'predicted_verdict',
                                 'reviewer', 'date'],
                          tweet=['tweet_id', 'text', 'created_at', 'lang',
                                 'source', 'public_metrics.retweet_count',
                                 'public_metrics.reply_count',
                                 'public_metrics.quote_count'],
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
        logger.info('Removing auxilliary nodes')

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
        logger.info('Removing island nodes')

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
                    rel_df = (rel_df.merge(node_df,
                                           left_on='src',
                                           right_on='old_idx')
                                    .drop(columns=['src', 'old_idx'])
                                    .rename(columns=dict(index='src')))
                    self.rels[rel_type] = rel_df
                    self.nodes[node_type] = (self.nodes[node_type]
                                                 .drop(columns='index'))

                # If islands have been removed from the target, then update
                # those indices
                if node_type == tgt and 'index' in self.nodes[node_type]:
                    node_df = (self.nodes[node_type]
                                   .rename(columns=dict(index='old_idx'))
                                   .reset_index())
                    rel_df = (rel_df.merge(node_df,
                                           left_on='tgt',
                                           right_on='old_idx')
                                    .drop(columns=['tgt', 'old_idx'])
                                    .rename(columns=dict(index='tgt')))
                    self.rels[rel_type] = rel_df
                    self.nodes[node_type] = (self.nodes[node_type]
                                                 .drop(columns='index'))

        return self

    def _dump_to_csv(self):
        '''Dumps the dataset to CSV files'''
        logger.info('Dumping to CSV')

        # Dump the nodes
        for node_type in self._node_dump:
            if node_type in self.nodes.keys():
                path = self.dataset_dir / f'{node_type}.csv'
                self.nodes[node_type].to_csv(path, index=False)

        # Dump the relations
        for rel_type in self._rel_dump:
            if rel_type in self.rels.keys():
                path = self.dataset_dir / f'{"_".join(rel_type)}.csv'
                self.rels[rel_type].to_csv(path, index=False)

        return self

    def to_dgl(self):
        '''Convert the dataset to a DGL dataset.

        Returns:
            DGLHeteroGraph:
                The graph in DGL format.
        '''
        logger.info('Outputting to DGL')
        return build_dgl_dataset(nodes=self.nodes, relations=self.rels)
