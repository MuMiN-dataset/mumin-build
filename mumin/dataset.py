'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd
import logging

from .rehydrate import rehydrate_tweets, rehydrate_users
from .articles import download_article
from .media import download_media
from .download import download_dataset


logger = logging.getLogger(__name__)


class MuminDataset:
    '''The MuMiN misinformation dataset, from [1].

    Args:
        twitter_api_key (str):
            The Twitter API key.
        twitter_api_secret (str):
            The Twitter API secret.
        twitter_access_token (str):
            The Twitter access token.
        twitter_access_secret (str):
            The Twitter access secret.
        size (str, optional):
            The size of the dataset. Can be either 'small', 'medium' or
            'large'. Defaults to 'large'.
        dataset_dir (str, optional):
            The path to the folder where the dataset should be stored. Defaults
            to './mumin'.

    References:
        - [1] Nielsen and McConville: _MuMiN: A Large-Scale Multilingual
              Multimodal Fact-Checked Misinformation Dataset with Linked Social
              Network Posts_ (2021)
    '''
    def __init__(self,
                 twitter_api_key: str,
                 twitter_api_secret: str,
                 twitter_access_token: str,
                 twitter_access_secret: str,
                 size: str = 'large',
                 dataset_dir: Union[str, Path] = './mumin'):
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_access_token = twitter_access_token
        self.twitter_access_secret = twitter_access_secret
        self.size = size
        self.dataset_dir = Path(dataset_dir)
        self.nodes: Dict[str, pd.DataFrame] = dict()
        self.rels: List[Tuple[str, str, pd.DataFrame]] = list()

    def __repr__(self) -> str:
        '''A string representation of the dataaset.

        Returns:
            str: The representation of the dataset.
        '''
        if len(self.nodes) == 0 or len(self.rels) == 0:
            return f'MuMiNDataset(size={self.size}, compiled=False)'
        else:
            num_nodes = sum([len(df) for df in self.nodes.values()])
            num_rels = sum([len(df) for _, _, df in self.rels])
            return (f'MuMiNDataset(num_nodes={num_nodes:,}, '
                    f'num_relations={num_rels:,}, '
                    f'size=\'{self.size}\', '
                    f'compiled=False)')

    def compile(self):
        '''Compiles the dataset.

        This entails downloading the dataset, rehydrating the Twitter data and
        downloading the relevant associated data, such as articles, images and
        videos.
        '''
        self._download()
        self._load_dataset()
        self._rehydrate()
        self._extract_twitter_data()
        self._populate_articles()
        self._populate_media()
        self._dump_to_csv()

    def _download(self):
        '''Downloads the dataset'''
        pass

    def _load_dataset(self):
        '''Loads the dataset files'''
        pass

    def _rehydrate(self):
        '''Rehydrate the tweets and users in the dataset'''
        pass

    def _extract_twitter_data(self):
        '''Extracts data from the raw Twitter data'''
        pass

    def _populate_articles(self):
        '''Downloads the articles in the dataset'''
        pass

    def _populate_media(self):
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
