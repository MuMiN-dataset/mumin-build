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


class MuMiNDataset:
    '''MuMiN dataset'''
    def __init__(self, dataset_dir: Union[str, Path] = '.'):
        self.dataset_dir = Path(dataset_dir)
        self.nodes = None
        self.relations = None

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
        self._dump()

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

    def _dump(self):
        '''Dumps the dataset to CSV files'''
        pass
