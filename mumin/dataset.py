'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd
import logging
import requests
import zipfile
import io

from .rehydrate import rehydrate_tweets, rehydrate_users
from .articles import download_article
from .media import download_media
from .download import download_dataset
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
                         'tree/main/data')

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
            num_rels = sum([len(df) for _, _, df in self.rels])
            return (f'MuminDataset(num_nodes={num_nodes:,}, '
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
        '''Downloads and unzips the dataset'''
        response = requests.get(self.download_url)

        # If the response was unsuccessful then raise an error
        if response.status_code != 200:
            raise RuntimeError(f'[{response.status_code}] {response.content}')

        # Otherwise unzip the in-memory zip file to `self.dataset_dir`
        else:
            zipped = response.raw.read()
            with zipfile.ZipFile(io.BytesIO(zipped)) as zip_file:
                zip_file.extractall(self.dataset_dir)

    def _load_dataset(self):
        '''Loads the dataset files into memory'''

        # Create the dataset directory if it does not already exist
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir()

        # Loop over the files in the dataset directory
        for path in self.dataset_dir.iterdir():
            fname = path.stem

            # Node case: no underscores in file name
            if len(fname.split('_')) == 0:
                self.nodes[fname] = pd.DataFrame(pd.read_csv(path))

            # Relation case: exactly two underscores in file name
            elif len(fname.split('_')) == 2:
                src, rel, tgt = tuple(fname.split('_'))
                self.rels[(src, rel, tgt)] = pd.DataFrame(pd.read_csv(path))

            # Otherwise raise error
            else:
                raise RuntimeError(f'Could not recognise {fname} as a node '
                                   f'or relation.')

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
