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
        dataset_dir (str or pathlib Path, optional):
            The path to the folder where the dataset should be stored. Defaults
            to './mumin'.

    Attributes:
        twitter (Twitter object): A wrapper for the Twitter API.
        size (str): The size of the dataset.
        dataset_dir (pathlib Path): The dataset directory.
        nodes (dict): The nodes of the dataset.
        rels (list): The relations of the dataset.

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
                 dataset_dir: Union[str, Path] = './mumin'):
        self.twitter = Twitter(twitter_bearer_token=twitter_bearer_token)
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
        downloading the relevant associated data, such as articles, images and
        videos.

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
        self._extract_media()
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
        for path in self.dataset_dir.iterdir():
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

    def _rehydrate(self):
        '''Rehydrate the tweets and users in the dataset'''

        # Ensure that the tweet and user IDs have been loaded into memory
        if 'tweet' not in self.nodes.keys():
            raise RuntimeError('Tweet IDs have not been loaded yet! '
                               'Load the dataset first.')
        else:
            # Get the tweet IDs
            tweet_ids = self.nodes['tweet'].tweet_id.tolist()

            # Rehydrate the tweets
            tweet_dfs = self.twitter.rehydrate_tweets(tweet_ids=tweet_ids)

            # Extract and store the node types
            self.nodes['tweet'] = tweet_dfs['tweets']
            self.nodes['user'] = tweet_dfs['users']
            self.nodes['media'] = tweet_dfs['media']
            self.nodes['poll'] = tweet_dfs['polls']
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
        extract_mention = lambda dcts: [int(dct['id']) for dct in dcts]
        mentions = (self.nodes['tweet']['entities.mentions']
                        .dropna()
                        .map(extract_mention)
                        .explode())
        data_dict = dict(src=mentions.index.tolist(), tgt=mentions.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.rels[('tweet', 'mentions', 'user')] = rel_df

        # (:User)-[:MENTIONS]->(:User)
        extract_mention = lambda dcts: [dct['username'] for dct in dcts]
        mentions = (self.nodes['user']['entities.description.mentions']
                        .dropna()
                        .map(extract_mention)
                        .explode())
        existing_usernames = self.nodes['user'].username.tolist()
        mentions = mentions[mentions.isin(existing_usernames)]
        data_dict = dict(src=mentions.index.tolist(), tgt=mentions.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.rels[('user', 'mentions', 'user')] = rel_df

        # (:Tweet)-[:HAS_MEDIA]->(:Media)
        media_ids = (self.nodes['tweet']['attachments.media_keys']
                         .dropna()
                         .explode())
        data_dict = dict(src=media_ids.index.tolist(), tgt=media_ids.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.rels[('tweet', 'has_media', 'media')] = rel_df

        # (:Tweet)-[:HAS_HASHTAG]->(:Hashtag)
        extract_hashtag = lambda dcts: [dct['tag'] for dct in dcts]
        hashtags = (self.nodes['tweet']['entities.hashtags']
                        .dropna()
                        .map(extract_hashtag)
                        .explode())
        data_dict = dict(src=hashtags.index.tolist(), tgt=hashtags.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.nodes['hashtag'] = pd.DataFrame(index=hashtags.tolist())
        self.rels[('tweet', 'has_hashtag', 'hashtag')] = rel_df

        # (:User)-[:HAS_HASHTAG]->(:Hashtag)
        extract_hashtag = lambda dcts: [dct['tag'] for dct in dcts]
        hashtags = (self.nodes['user']['entities.descriptin.hashtags']
                        .dropna()
                        .map(extract_hashtag)
                        .explode())
        node_df = pd.DataFrame(index=hashtags.tolist())
        data_dict = dict(src=hashtags.index.tolist(), tgt=hashtags.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.nodes['hashtag'] = self.nodes['hashtag'].append(node_df)
        self.rels[('user', 'has_hashtag', 'hashtag')] = rel_df

        # (:Tweet)-[:HAS_URL]->(:Url)
        extract_url = lambda dcts: [dct['expanded_url'] for dct in dcts]
        urls = (self.nodes['tweet']['entities.urls']
                    .dropna()
                    .map(extract_url)
                    .explode())
        data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.nodes['url'] = pd.DataFrame(index=urls.tolist())
        self.rels[('tweet', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_PROFILE_PICTURE_URL]->(:Url)
        urls = self.nodes['user']['profile_image_url'].dropna()
        node_df = pd.DataFrame(index=urls.tolist())
        data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
        rel_df = pd.DataFrame(data_dict)
        self.nodes['url'] = self.nodes['url'].append(node_df)
        self.rels[('tweet', 'has_profile_picture_url', 'url')] = rel_df

    def _extract_articles(self):
        '''Downloads the articles in the dataset'''
        pass

    def _extract_media(self):
        '''Downloads the images and videos in the dataset'''
        pass

    def _dump_to_csv(self):
        '''Dumps the dataset to CSV files'''
        pass

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
