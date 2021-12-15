'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional
import pandas as pd
import zipfile
import io
import numpy as np
import logging
import requests
import os
from tqdm.auto import tqdm
from shutil import rmtree

from .twitter import Twitter
from .dgl import build_dgl_dataset
from .data_extractor import DataExtractor
from .id_updator import IdUpdator
from .embedder import Embedder


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
        include_tweet_images (bool, optional):
            Whether to include images from the tweets in the dataset. This will
            mean that compilation of the dataset will take a bit longer, as
            these need to be downloaded and parsed. Defaults to True.
        include_extra_images (bool, optional):
            Whether to include images from the articles and users in the
            dataset. This will mean that compilation of the dataset will take a
            bit longer, as these need to be downloaded and parsed. Defaults to
            False.
        include_hashtags (bool, optional):
            Whether to include hashtags in the dataset. Defaults to True.
        include_mentions (bool, optional):
            Whether to include mentions in the dataset. Defaults to True.
        include_timelines (bool, optional):
            Whether to include timelines in the dataset. Defaults to False.
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
        include_replies (bool): Whether to include replies.
        include_articles (bool): Whether to include articles.
        include_tweet_images (bool): Whether to include tweet images.
        include_extra_images (bool): Whether to include user/article images.
        include_hashtags (bool): Whether to include hashtags.
        include_mentions (bool): Whether to include mentions.
        include_timelines (bool): Whether to include timelines.
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
                         '22cbc45ad014464a0ee37338a61de1e5e64e756a'
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
                 include_tweet_images: bool = True,
                 include_extra_images: bool = False,
                 include_hashtags: bool = True,
                 include_mentions: bool = True,
                 include_timelines: bool = False,
                 text_embedding_model_id: str = 'xlm-roberta-base',
                 image_embedding_model_id: str = ('google/vit-base-patch16-'
                                                  '224-in21k'),
                 dataset_path: Optional[Union[str, Path]] = None,
                 verbose: bool = True):
        self.size = size
        self.include_replies = include_replies
        self.include_articles = include_articles
        self.include_tweet_images = include_tweet_images
        self.include_extra_images = include_extra_images
        self.include_hashtags = include_hashtags
        self.include_mentions = include_mentions
        self.include_timelines = include_timelines
        self.text_embedding_model_id = text_embedding_model_id
        self.image_embedding_model_id = image_embedding_model_id
        self.verbose = verbose

        self.compiled = False
        self.nodes: Dict[str, pd.DataFrame] = dict()
        self.rels: Dict[Tuple[str, str, str], pd.DataFrame] = dict()

        self._twitter = Twitter(twitter_bearer_token=twitter_bearer_token)
        self._extractor = DataExtractor(
            include_replies=include_replies,
            include_articles=include_articles,
            include_tweet_images=include_tweet_images,
            include_extra_images=include_extra_images,
            include_hashtags=include_hashtags,
            include_mentions=include_mentions
        )
        self._updator = IdUpdator()
        self._embedder = Embedder(
            text_embedding_model_id=text_embedding_model_id,
            image_embedding_model_id=image_embedding_model_id,
            include_articles=include_articles,
            include_tweet_images=include_tweet_images,
            include_extra_images=include_extra_images
        )

        if dataset_path is None:
            dataset_path = f'./mumin-{size}.zip'
        self.dataset_path = Path(dataset_path)

        # Set up logging verbosity
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

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
            self.nodes, self.rels = self._updator.update_all(nodes=self.nodes,
                                                             rels=self.rels)

            # Extract data from the rehydrated tweets
            self.nodes, self.rels = self._extractor.extract_all(
                nodes=self.nodes,
                rels=self.rels
            )

            # Filter the data
            self._filter_node_features()
            self._filter_relations()

            # Set datatypes
            self._set_datatypes()

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
                duplicated = (tweet_df[tweet_df.tweet_id.duplicated()]
                              .tweet_id
                              .tolist())
                if len(duplicated) > 0:
                    raise RuntimeError(f'The tweet IDs {duplicated} are '
                                       f'duplicate in the dataset!')

        return self

    def _shrink_dataset(self):
        '''Shrink dataset if `size` is 'small' or 'medium'''
        logger.info('Shrinking dataset')

        # Define the `relevance` threshold
        if self.size == 'small':
            threshold = 0.80  # noqa
        elif self.size == 'medium':
            threshold = 0.75  # noqa
        elif self.size == 'large':
            threshold = 0.70  # noqa
        elif self.size == 'test':
            threshold = 0.995  # noqa

        # Filter nodes
        ntypes = ['tweet', 'reply', 'user', 'article']
        for ntype in ntypes:
            self.nodes[ntype] = (self.nodes[ntype]
                                 .query('relevance > @threshold')
                                 .drop(columns=['relevance'])
                                 .reset_index(drop=True))

        # Filter relations
        etypes = [('reply', 'reply_to', 'tweet'),
                  ('reply', 'quote_of', 'tweet'),
                  ('user', 'retweeted', 'tweet'),
                  ('user', 'follows', 'user'),
                  ('tweet', 'discusses', 'claim'),
                  ('article', 'discusses', 'claim')]
        for etype in etypes:
            self.rels[etype] = (self.rels[etype]
                                .query('relevance > @threshold')
                                .drop(columns=['relevance'])
                                .reset_index(drop=True))

        # Filter claims
        claim_df = self.nodes['claim']
        discusses_rel = self.rels[('tweet', 'discusses', 'claim')]
        include_claim = claim_df.id.isin(discusses_rel.tgt.tolist())
        self.nodes['claim'] = claim_df[include_claim].reset_index(drop=True)

        # Filter timeline tweets
        if not self.include_timelines:
            src_tweet_ids = (self.rels[('tweet', 'discusses', 'claim')]
                                 .src
                                 .astype(int)
                                 .tolist())
            is_src = self.nodes['tweet'].tweet_id.isin(src_tweet_ids)
            self.nodes['tweet'] = self.nodes['tweet'].loc[is_src]

        return self

    def _rehydrate(self, node_type: str):
        '''Rehydrate the tweets and users in the dataset.

        Args:
            node_type (str): The type of node to rehydrate.
        '''

        if (node_type in self.nodes.keys() and
                (node_type != 'reply' or self.include_replies)):

            logger.info(f'Rehydrating {node_type} nodes')

            # Get the tweet IDs, and if the node type is a tweet then separate
            # these into source tweets and the rest (i.e., timeline tweets)
            if node_type == 'tweet':
                source_tweet_ids = (self.rels[('tweet', 'discusses', 'claim')]
                                        .astype(int)
                                        .src
                                        .tolist())
                tweet_ids = [tweet_id
                             for tweet_id in self.nodes[node_type]
                                                 .tweet_id
                                                 .tolist()
                             if tweet_id not in source_tweet_ids]
            else:
                source_tweet_ids = list()
                tweet_ids = self.nodes[node_type].tweet_id.tolist()

            # Store any features the nodes might have had before hydration
            prehydration_df = self.nodes[node_type].copy()

            # Rehydrate the source tweets
            if len(source_tweet_ids) > 0:
                params = dict(tweet_ids=source_tweet_ids)
                source_tweet_dfs = self._twitter.rehydrate_tweets(**params)

                # Return error if there are no tweets were rehydrated. This is
                # probably because the bearer token is wrong
                if len(source_tweet_dfs) == 0:
                    raise RuntimeError('No tweets were rehydrated. Check if '
                                       'the bearer token is correct.')

                if len(tweet_ids) == 0:
                    tweet_dfs = {key: pd.DataFrame()
                                 for key in source_tweet_dfs.keys()}

            # Rehydrate the other tweets
            if len(tweet_ids) > 0:
                params = dict(tweet_ids=tweet_ids)
                tweet_dfs = self._twitter.rehydrate_tweets(**params)

                # Return error if there are no tweets were rehydrated. This is
                # probably because the bearer token is wrong
                if len(tweet_dfs) == 0:
                    raise RuntimeError('No tweets were rehydrated. Check if '
                                       'the bearer token is correct.')

                if len(source_tweet_ids) == 0:
                    source_tweet_dfs = {key: pd.DataFrame()
                                        for key in tweet_dfs.keys()}

            # Extract and store tweets and users
            tweet_df = pd.concat([source_tweet_dfs['tweets'],
                                  tweet_dfs['tweets']],
                                 ignore_index=True)
            self.nodes[node_type] = (tweet_df
                                     .drop_duplicates(subset='tweet_id')
                                     .reset_index(drop=True))
            user_df = pd.concat([source_tweet_dfs['users'],
                                 tweet_dfs['users']],
                                ignore_index=True)
            if ('user' in self.nodes.keys() and
                    'username' in self.nodes['user'].columns):
                user_df = (self.nodes['user']
                               .append(user_df)
                               .drop_duplicates(subset='user_id')
                               .reset_index(drop=True))
            self.nodes['user'] = user_df

            # Add prehydration tweet features back to the tweets
            self.nodes[node_type] = (self.nodes[node_type]
                                         .merge(prehydration_df,
                                                on='tweet_id',
                                                how='outer')
                                         .reset_index(drop=True))

            # Extract and store images
            # Note: This will store `self.nodes['image']`, but this is only
            #       to enable extraction of URLs later on. The
            #       `self.nodes['image']` will be overwritten later on.
            if (node_type == 'tweet' and self.include_tweet_images and
                    len(source_tweet_dfs['media'])):

                image_df = (source_tweet_dfs['media']
                            .query('type == "photo"')
                            .drop_duplicates(subset='media_key')
                            .reset_index(drop=True))

                if 'image' in self.nodes.keys():
                    image_df = (self.nodes['image']
                                    .append(image_df)
                                    .drop_duplicates(subset='media_key')
                                    .reset_index(drop=True))

                self.nodes['image'] = image_df

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
        # Compute the embeddings
        self.nodes = self._embedder.embed_all(nodes=self.nodes,
                                              nodes_to_embed=nodes_to_embed)

        # Store dataset
        self._dump_dataset()

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

            # Pop the relation if the dataframe does not exist
            if rel_df is None or len(rel_df) == 0:
                rels_to_pop.append(rel_type)
                continue

            # Pop the relation if the source or target node does not exist
            src, _, tgt = rel_type
            if src not in self.nodes.keys() or tgt not in self.nodes.keys():
                rels_to_pop.append(rel_type)

            # Otherwise filter the relation dataframe to only include nodes
            # that exist
            else:
                src_ids = self.nodes[src].index.tolist()
                tgt_ids = self.nodes[tgt].index.tolist()
                rel_df = rel_df[rel_df.src.isin(src_ids)]
                rel_df = rel_df[rel_df.tgt.isin(tgt_ids)]
                self.rels[rel_type] = rel_df

        # Pop the relations that has been assigned to be popped
        for rel_type in rels_to_pop:
            self.rels.pop(rel_type)

    def _set_datatypes(self):
        '''Set datatypes in the dataframes, to use less memory'''

        # Set up all the dtypes of the columns
        dtypes = dict(tweet=dict(tweet_id='uint64',
                                 text='str',
                                 created_at={'created_at': 'datetime64[ns]'},
                                 lang='category',
                                 source='str',
                                 num_retweets='uint64',
                                 num_replies='uint64',
                                 num_quote_tweets='uint64'),
                      user=dict(user_id='uint64',
                                verified='bool',
                                protected='bool',
                                created_at={'created_at': 'datetime64[ns]'},
                                username='str',
                                description='str',
                                url='str',
                                name='str',
                                num_followers='uint64',
                                num_followees='uint64',
                                num_tweets='uint64',
                                num_listed='uint64',
                                location='category'))

        if self.include_hashtags:
            dtypes['hashtag'] = dict(tag='str')

        if self.include_replies:
            dtypes['reply'] = dict(tweet_id='uint64',
                                   text='str',
                                   created_at={'created_at': 'datetime64[ns]'},
                                   lang='category',
                                   source='str',
                                   num_retweets='uint64',
                                   num_replies='uint64',
                                   num_quote_tweets='uint64')

        if self.include_tweet_images or self.include_extra_images:
            dtypes['image'] = dict(url='str',
                                   pixels='numpy',
                                   width='uint64',
                                   height='uint64')

        if self.include_articles:
            dtypes['article'] = dict(url='str',
                                     title='str',
                                     content='str')

        # Create conversion function for missing values
        def fill_na_values(dtype: Union[str, dict]):
            if dtype == 'uint64':
                return 0
            elif dtype == 'bool':
                return False
            elif dtype == dict(created_at='datetime64[ns]'):
                return np.datetime64('NaT')
            elif dtype == 'category':
                return 'NaN'
            elif dtype == 'str':
                return ''
            else:
                return np.nan

        # Loop over all nodes
        for ntype, dtype_dict in dtypes.items():
            if ntype in self.nodes.keys():

                # Set the dtypes for non-numpy columns
                dtype_dict_no_numpy = {col: dtype
                                       for col, dtype in dtype_dict.items()
                                       if dtype != 'numpy'}
                for col, dtype in dtype_dict_no_numpy.items():
                    if col in self.nodes[ntype].columns:

                        # Fill NaN values with canonical values in accordance
                        # with the datatype
                        self.nodes[ntype][col].fillna(fill_na_values(dtype),
                                                      inplace=True)

                        # Set the dtype
                        self.nodes[ntype][col] = (self.nodes[ntype][col]
                                                  .astype(dtype))

                # For numpy columns, set the type manually
                def numpy_fn(x):
                    return np.asarray(x)

                for col, dtype in dtype_dict.items():
                    if dtype == 'numpy' and col in self.nodes[ntype].columns:

                        # Fill NaN values with canonical values in accordance
                        # with the datatype
                        self.nodes[ntype][col].fillna(fill_na_values(dtype),
                                                      inplace=True)

                        # Set the dtype
                        self.nodes[ntype][col] = (self.nodes[ntype][col]
                                                      .map(numpy_fn))

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
