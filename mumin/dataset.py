'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd
import logging
import requests
import zipfile
import io
import shutil

from .twitter import Twitter


logger = logging.getLogger(__name__)


class MuminDataset:
    '''The MuMiN misinformation dataset, from [1].

    Args:
        twitter_bearer_token(str):
            The Twitter bearer token.
        size (str, optional):
            The size of the dataset. Can be either 'small', 'medium' or
            'large'. Defaults to 'large'.
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
        dataset_dir (str or pathlib Path, optional):
            The path to the folder where the dataset should be stored. Defaults
            to './mumin'.

    Attributes:
        twitter (Twitter object): A wrapper for the Twitter API.
        include_articles (bool): Whether to include articles in the dataset.
        include_images (bool): Whether to include images in the dataset.
        include_hashtags (bool): Whether to include hashtags in the dataset.
        include_mentions (bool): Whether to include mentions in the dataset.
        include_places (bool): Whether to include places in the dataset.
        include_polls (bool): Whether to include polls in the dataset.
        size (str): The size of the dataset.
        dataset_dir (pathlib Path): The dataset directory.
        nodes (dict): The nodes of the dataset.
        rels (dict): The relations of the dataset.

    References:
        - [1] Nielsen and McConville: _MuMiN: A Large-Scale Multilingual
              Multimodal Fact-Checked Misinformation Dataset with Linked Social
              Network Posts_ (2021)
    '''

    download_url: str = ('https://github.com/CLARITI-REPHRAIN/mumin-build/'
                         'raw/main/data/mumin.zip')

    def __init__(self,
                 twitter_bearer_token: str,
                 size: str = 'large',
                 include_articles: bool = True,
                 include_images: bool = True,
                 include_hashtags: bool = True,
                 include_mentions: bool = True,
                 include_places: bool = True,
                 include_polls: bool = True,
                 dataset_dir: Union[str, Path] = './mumin'):
        self.twitter = Twitter(twitter_bearer_token=twitter_bearer_token)
        self.include_articles = include_articles
        self.include_images = include_images
        self.include_hashtags = include_hashtags
        self.include_mentions = include_mentions
        self.include_places = include_places
        self.include_polls = include_polls
        self.size = size
        self.dataset_dir = Path(dataset_dir)
        self.nodes: Dict[str, pd.DataFrame] = dict()
        self.rels: Dict[Tuple[str, str, str], pd.DataFrame] = dict()

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
        self._rehydrate()
        self._extract_relations()
        self._extract_articles()
        self._extract_images()
        self._filter_node_features()
        self._dump_to_csv()

    def _download(self, overwrite: bool = False):
        '''Downloads and unzips the dataset.

        Args:
            overwrite (bool, optional):
                Whether the dataset directory should be overwritten, in case it
                already exists. Defaults to False.
        '''
        if (not self.dataset_dir.exists() or
                (self.dataset_dir.exists() and overwrite)):
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

    def _load_dataset(self):
        '''Loads the dataset files into memory.

        Raises:
            RuntimeError:
                If the dataset has not been downloaded yet.
        '''

        # Raise error if the dataset has not been downloaded yet
        if not self.dataset_dir.exists():
            raise RuntimeError('Dataset has not been downloaded yet!')

        # Loop over the files in the dataset directory
        csv_paths = [path for path in self.dataset_dir.iterdir()
                     if path.stem[-4:]('.csv')]
        for path in csv_paths:
            fname = path.stem

            # Node case: no underscores in file name
            if len(fname.split('_')) == 1:
                self.nodes[fname] = pd.DataFrame(pd.read_csv(path))

            # Relation case: exactly two underscores in file name
            elif len(fname.split('_')) > 2:
                splits = fname.split('_')
                src = splits[0]
                tgt = splits[-1]
                rel = '_'.join(splits[1:-1])
                self.rels[(src, rel, tgt)] = pd.DataFrame(pd.read_csv(path))

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
            duplicated = tweet_df[tweet_df.index.duplicated()].index.tolist()
            if len(duplicated) > 0:
                raise RuntimeError(f'The tweet IDs {duplicated} are '
                                   f'duplicate in the dataset!')

        # Ensure that the `id` column is set as the index, if it exists
        for node_type, df in self.nodes.items():
            if 'id' in df.columns:
                self.nodes[node_type] = pd.DataFrame(df.set_index('id'))

    def _rehydrate(self):
        '''Rehydrate the tweets and users in the dataset'''

        # Ensure that the tweet and user IDs have been loaded into memory
        if 'tweet' not in self.nodes.keys():
            raise RuntimeError('Tweet IDs have not been loaded yet! '
                               'Load the dataset first.')

        # Only rehydrate if we have not rehydrated already; a simple way to
        # check this is to see if the tweet dataframe has the 'tweet_id'
        # column, as this is stored as an index after rehydration
        elif 'tweet_id' in self.nodes['tweet'].columns:
            # Get the tweet IDs
            tweet_ids = self.nodes['tweet'].tweet_id.tolist()

            # Rehydrate the tweets
            tweet_dfs = self.twitter.rehydrate_tweets(tweet_ids=tweet_ids)

            # Extract and store tweets and users
            self.nodes['tweet'] = tweet_dfs['tweets']
            self.nodes['user'] = tweet_dfs['users']

            # Extract and store images
            if self.include_images:
                video_query = '(type == "video") or (type == "animated gif")'
                video_df = (tweet_dfs['media']
                            .query(video_query)
                            .rename(dict(preview_image_url='url')))
                image_df = (tweet_dfs['media']
                            .query('type == "photo"')
                            .append(video_df))
                self.nodes['image'] = image_df

            # Extract and store polls
            if self.include_polls:
                self.nodes['poll'] = tweet_dfs['polls']

            # Extract and store places
            if self.include_places:
                self.nodes['place'] = tweet_dfs['places']

            # TODO: Rehydrate quote tweets and replies

    def _extract_relations(self):
        '''Extracts relations from the raw Twitter data'''

        # (:User)-[:POSTED]->(:Tweet)
        data_dict = dict(src=self.nodes['tweet'].author_id.tolist(),
                         tgt=self.nodes['tweet'].index.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.rels[('user', 'posted', 'tweet')] = rel_df

        # (:Tweet)-[:MENTIONS]->(:User)
        if self.include_mentions:
            extract_mention = lambda dcts: [int(dct['id']) for dct in dcts]
            mentions = (self.nodes['tweet']['entities.mentions']
                            .dropna()
                            .map(extract_mention)
                            .explode())
            data_dict = dict(src=mentions.index.tolist(),
                             tgt=mentions.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'mentions', 'user')] = rel_df

        # (:User)-[:MENTIONS]->(:User)
        if self.include_mentions:
            extract_mention = lambda dcts: [dct['username'] for dct in dcts]
            mentions = (self.nodes['user']['entities.description.mentions']
                            .dropna()
                            .map(extract_mention)
                            .explode())
            existing_usernames = self.nodes['user'].username.tolist()
            mentions = mentions[mentions.isin(existing_usernames)]
            data_dict = dict(src=mentions.index.tolist(),
                             tgt=mentions.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'mentions', 'user')] = rel_df

        # (:Tweet)-[:LOCATED_IN]->(:Place)
        if self.include_places:
            place_ids = self.nodes['tweet']['geo.place_id'].dropna()
            data_dict = dict(src=place_ids.index.tolist(),
                             tgt=place_ids.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'located_in', 'place')] = rel_df

        # (:Tweet)-[:HAS_POLL]->(:Poll)
        if self.include_polls:
            poll_ids = (self.nodes['tweet']['attachments.poll_ids']
                            .dropna()
                            .explode())
            data_dict = dict(src=poll_ids.index.tolist(),
                             tgt=poll_ids.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_poll', 'poll')] = rel_df

        # (:Tweet)-[:HAS_IMAGE]->(:Image)
        if self.include_images:
            media_ids = (self.nodes['tweet']['attachments.media_keys']
                             .dropna()
                             .explode())
            is_image = media_ids.isin(self.nodes['image'].index.tolist())
            image_ids = media_ids[is_image]
            data_dict = dict(src=image_ids.index.tolist(),
                             tgt=image_ids.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_image', 'image')] = rel_df

        # (:Tweet)-[:HAS_HASHTAG]->(:Hashtag)
        if self.include_hashtags:
            def extract_hashtag(dcts: List[dct]) -> List[str]:
                return [dct.get('tag') for dct in dcts]
            hashtags = (self.nodes['tweet']['entities.hashtags']
                            .dropna()
                            .map(extract_hashtag)
                            .explode())
            node_df  = pd.DataFrame(index=hashtags.tolist())
            data_dict = dict(src=hashtags.index.tolist(),
                             tgt=hashtags.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'hashtag' in self.nodes.keys():
                self.nodes['hashtag'] = self.nodes['hashtag'].append(node_df)
            else:
                self.nodes['hashtag'] = node_df
            self.rels[('tweet', 'has_hashtag', 'hashtag')] = rel_df

        # (:User)-[:HAS_HASHTAG]->(:Hashtag)
        if self.include_hashtags:
            def extract_hashtag(dcts: List[dct]) -> List[str]:
                return [dct.get('tag') for dct in dcts]
            hashtags = (self.nodes['user']['entities.description.hashtags']
                            .dropna()
                            .map(extract_hashtag)
                            .explode())
            node_df = pd.DataFrame(index=hashtags.tolist())
            data_dict = dict(src=hashtags.index.tolist(),
                             tgt=hashtags.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'hashtag' in self.nodes.keys():
                self.nodes['hashtag'] = self.nodes['hashtag'].append(node_df)
            else:
                self.nodes['hashtag'] = node_df
            self.rels[('user', 'has_hashtag', 'hashtag')] = rel_df

        # (:Tweet)-[:HAS_URL]->(:Url)
        if self.include_articles or self.include_images:
            def extract_url(dcts: List[dct]) -> List[str]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            urls = (self.nodes['tweet']['entities.urls']
                        .dropna()
                        .map(extract_url)
                        .explode())
            node_df = pd.DataFrame(index=urls.tolist())
            data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'url' in self.nodes.keys():
                self.nodes['url'] = self.nodes['urls'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('tweet', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_URL]->(:Url)
        if self.include_images:
            def extract_url(dcts: List[dct]) -> List[str]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            url_urls = (self.nodes['user']['entities.url.urls']
                            .dropna()
                            .map(extract_url)
                            .explode())
            desc_urls = (self.nodes['user']['entities.description.urls']
                             .dropna()
                             .map(extract_url)
                             .explode())
            urls = url_urls.append(desc_urls)
            node_df = pd.DataFrame(index=urls.tolist())
            data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'url' in self.nodes.keys():
                self.nodes['url'] = self.nodes['urls'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('user', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_PROFILE_PICTURE_URL]->(:Url)
        if self.include_images:
            urls = self.nodes['user']['profile_image_url'].dropna()
            node_df = pd.DataFrame(index=urls.tolist())
            data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'url' in self.nodes.keys():
                self.nodes['url'] = self.nodes['urls'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('user', 'has_profile_picture_url', 'url')] = rel_df

    def _extract_articles(self):
        '''Downloads the articles in the dataset'''
        pass

    def _extract_images(self):
        '''Downloads the images in the dataset'''
        pass

    def _filter_node_features(self):
        '''Filters the node features to avoid redundancies and noise'''

        # Set up the node features that should be kept
        node_feats = dict(claim=['raw_verdict', 'predicted_verdict',
                                 'reviewer', 'date'],
                          tweet=['text', 'created_at', 'lang', 'source',
                                 'public_metrics.retweet_count',
                                 'public_metrics.reply_count',
                                 'public_metrics.quote_count'],
                          user=['verified', 'protected', 'created_at',
                                'username', 'description', 'url', 'name',
                                'public_metrics.followers_count',
                                'public_metrics.following_count',
                                'public_metrics.tweet_count',
                                'public_metrics.listed_count',
                                'location'],
                          image=['url'],
                          article=['url', 'title', 'content'],
                          place=[],
                          hashtag=['tag'],
                          poll=[])

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
                renaming_dict = {old: new
                                 for old, new in node_feat_renaming.items()
                                 if old in features}
                self.nodes[node_type] = (self.nodes[node_type][features]
                                         .rename(renaming_dict))

    def _dump_to_csv(self):
        '''Dumps the dataset to CSV files'''

        # Set up the node types and relation types to dump
        nodes_to_dump = ['claim', 'tweet', 'user', 'image', 'article',
                        'place', 'hashtag', 'poll']
        rels_to_dump = []

        # Dump the nodes
        for node_type in nodes_to_dump:
            path = self.dataset_dir / f'{node_type}.csv'
            self.nodes[node_type].to_csv(path, index=True)

        # Dump the relations
        for rel_type in rels_to_dump:
            path = self.dataset_dir / f'{"_".join(rel_type)}.csv'
            self.rels[rel_type].to_csv(path, index=False)

    def to_dgl(self,
               output_format: str = 'thread-level-graphs'
               ) -> 'DGLDataset':
        '''Convert the dataset to a DGL dataset.

        Args:
            output_format (str, optional):
                The format the dataset should be outputted in. Can be
                'thread-level-graphs', 'claim-level-graphs' and 'single-graph'.
                Defaults to 'thread-level-graphs'.

        Returns:
            DGLDataset:
                The dataset in DGL format.
        '''
        pass

    def to_pyg(self,
               output_format: str = 'thread-level-graphs'
               ) -> 'InMemoryDataset':
        '''Convert the dataset to a PyTorch Geometric dataset.

        Args:
            output_format (str, optional):
                The format the dataset should be outputted in. Can be
                'thread-level-graphs', 'claim-level-graphs' and 'single-graph'.
                Defaults to 'thread-level-graphs'.

        Returns:
            PyTorch Geometric InMemoryDataset:
                The dataset in PyTorch Geometric format.
        '''
        pass
