'''Script containing the main dataset class'''

from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd
import logging
import requests
import zipfile
import io
import shutil
from collections import defaultdict
import re
import multiprocessing as mp
from tqdm.auto import tqdm

from .twitter import Twitter
from .article import process_article_url
from .image import process_image_url


logging.getLogger('jieba').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


class MuminDataset:
    '''The MuMiN misinformation dataset, from [1].

    Args:
        twitter_bearer_token (str):
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
    _node_dump: List[str] = ['claim', 'tweet', 'user', 'image', 'article',
                             'place', 'hashtag', 'poll']
    _rel_dump: List[Tuple[str, str, str]] = [
        ('tweet', 'discusses', 'claim'),
        ('tweet', 'mentions', 'user'),
        ('tweet', 'located_in', 'place'),
        ('tweet', 'has_image', 'image'),
        ('tweet', 'has_hashtag', 'hashtag'),
        ('tweet', 'has_article', 'article'),
        ('tweet', 'has_poll', 'poll'),
        ('user', 'posted', 'tweet'),
        ('user', 'mentions', 'user'),
        ('user', 'has_pinned', 'tweet'),
        ('user', 'has_hashtag', 'hashtag'),
        ('user', 'has_profile_picture', 'image'),
        ('article', 'has_top_image', 'image'),
    ]

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
        self._shrink_dataset()
        self._rehydrate()
        self._extract_relations()
        self._extract_places()
        self._extract_polls()
        self._extract_articles()
        self._extract_images()
        self._filter_node_features()
        self._remove_auxilliaries()
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
            duplicated = tweet_df[tweet_df.index.duplicated()].index.tolist()
            if len(duplicated) > 0:
                raise RuntimeError(f'The tweet IDs {duplicated} are '
                                   f'duplicate in the dataset!')

        # Ensure that the `id` column is set as the index, if it exists
        for node_type, df in self.nodes.items():
            if 'id' in df.columns:
                self.nodes[node_type] = pd.DataFrame(df.set_index('id'))

    def _shrink_dataset(self):
        '''Shrink dataset if `size` is 'small' or 'medium'''
        if self.size == 'small' or self.size == 'medium':

            # Define the `relevance` threshold
            if self.size == 'small':
                threshold = 0.8
            else:
                threshold = 0.75

            # Filter (:Tweet)-[:DISCUSSES]->(:Claim)
            discusses_rel = (self.rels[('tweet', 'discusses', 'claim')]
                             .query(f'relevance > {threshold}'))
            self.rels[('tweet', 'discusses', 'claim')] = discusses_rel

            # Filter tweets
            tweet_df = self.nodes['tweet']
            include_tweet = tweet_df.tweet_id.isin(discusses_rel.src.tolist())
            self.nodes['tweet'] = tweet_df[include_tweet]

            # Filter claims
            claim_df = self.nodes['claim']
            include_claim = claim_df.index.isin(discusses_rel.tgt.tolist())
            self.nodes['claim'] = claim_df[include_claim]

            # Filter (:Tweet)-[:DISCUSSES]->(:Claim)
            discusses_rel = (self.rels[('article', 'discusses', 'claim')]
                             .query(f'relevance > {threshold}'))
            self.rels[('article', 'discusses', 'claim')] = discusses_rel

            # Filter articles
            article_df = self.nodes['article']
            include_article = article_df.index.isin(discusses_rel.src.tolist())
            self.nodes['article'] = article_df[include_article]

            # Filter (:User)-[:POSTED]->(:Tweet)
            posted_rel = self.rels[('user', 'posted', 'tweet')]
            posted_rel = posted_rel[posted_rel.tgt.isin(self.nodes['tweet'])]
            self.rels[('user', 'posted', 'tweet')] = posted_rel

            # Filter (:Tweet)-[:MENTIONS]->(:User)
            mentions_rel = self.rels[('tweet', 'mentions', 'user')]
            mentions_rel = mentions_rel[mentions_rel
                                        .src
                                        .isin(self.nodes['tweet'])]
            self.rels[('tweet', 'mentions', 'user')] = mentions_rel

            # Filter users
            user_df = self.nodes['user']
            has_posted = user_df.index.isin(posted_rel.src.tolist())
            was_mentioned = user_df.index.isin(mentions_rel.tgt.tolist())
            self.nodes['user'] = user_df[has_posted | was_mentioned]

            # Filter (:User)-[:MENTIONS]->(:User)
            mentions_rel = self.rels[('user', 'mentions', 'user')]
            mentions_rel = mentions_rel[mentions_rel
                                        .src
                                        .isin(self.nodes['user'])]
            mentions_rel = mentions_rel[mentions_rel
                                        .tgt
                                        .isin(self.nodes['user'])]
            self.rels[('user', 'mentions', 'user')] = mentions_rel

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
            if self.include_images and len(tweet_dfs['media']):
                video_query = '(type == "video") or (type == "animated gif")'
                video_df = (tweet_dfs['media']
                            .query(video_query)
                            .drop(columns=['url', 'duration_ms',
                                           'public_metrics.view_count'])
                            .rename(dict(preview_image_url='url')))
                image_df = (tweet_dfs['media']
                            .query('type == "photo"')
                            .append(video_df)
                            .set_index('url'))
                self.nodes['image'] = image_df

            # Extract and store polls
            if self.include_polls and len(tweet_dfs['polls']):
                self.nodes['poll'] = tweet_dfs['polls']

            # Extract and store places
            if self.include_places and len(tweet_dfs['places']):
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
        mentions_exist = 'entities.mentions' in self.nodes['tweet'].columns
        if self.include_mentions and mentions_exist:
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
        user_cols = self.nodes['user'].columns
        mentions_exist = 'entities.description.mentions' in user_cols
        if self.include_mentions and mentions_exist:
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

        # (:User)-[:HAS_PINNED]->(:Tweet)
        pinned_exist = 'TODO' in self.nodes['user'].columns
        if pinned_exist:
            extract_mention = lambda dcts: [dct['username'] for dct in dcts]
            pinned = self.nodes['user']['TODO'].dropna()
            data_dict = dict(src=pinned.index.tolist(), tgt=pinned.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('user', 'has_pinned', 'tweet')] = rel_df

        # (:Tweet)-[:LOCATED_IN]->(:Place)
        places_exist = 'geo.place_id' in self.nodes['tweet'].columns
        if self.include_places and places_exist:
            place_ids = self.nodes['tweet']['geo.place_id'].dropna()
            data_dict = dict(src=place_ids.index.tolist(),
                             tgt=place_ids.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'located_in', 'place')] = rel_df

        # (:Tweet)-[:HAS_POLL]->(:Poll)
        polls_exist = 'attachments.poll_ids' in self.nodes['tweet'].columns
        if self.include_polls and polls_exist:
            poll_ids = (self.nodes['tweet']['attachments.poll_ids']
                            .dropna()
                            .explode())
            data_dict = dict(src=poll_ids.index.tolist(),
                             tgt=poll_ids.tolist())
            rel_df = pd.DataFrame(data_dict)
            self.rels[('tweet', 'has_poll', 'poll')] = rel_df

        # (:Tweet)-[:HAS_IMAGE]->(:Image)
        images_exist = 'attachments.media_keys' in self.nodes['tweet'].columns
        if self.include_images and images_exist:
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
        hashtags_exist = 'entities.hashtags' in self.nodes['tweet'].columns
        if self.include_hashtags and hashtags_exist:
            def extract_hashtag(dcts: List[dict]) -> List[str]:
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
        user_cols = self.nodes['user'].columns
        hashtags_exist = 'entities.description.hashtags' in user_cols
        if self.include_hashtags and hashtags_exist:
            def extract_hashtag(dcts: List[dict]) -> List[str]:
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
        urls_exist = 'entities.urls' in self.nodes['tweet'].columns
        if (self.include_articles or self.include_images) and urls_exist:
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
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
                self.nodes['url'] = self.nodes['url'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('tweet', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_URL]->(:Url)
        user_cols = self.nodes['user'].columns
        url_urls_exist = 'entities.url.urls' in user_cols
        desc_urls_exist = 'entities.description.urls' in user_cols
        if self.include_images and (url_urls_exist or desc_urls_exist):
            def extract_url(dcts: List[dict]) -> List[Union[str, None]]:
                return [dct.get('expanded_url') or dct.get('url')
                        for dct in dcts]
            if url_urls_exist:
                url_urls = (self.nodes['user']['entities.url.urls']
                                .dropna()
                                .map(extract_url)
                                .explode())
            else:
                url_urls = pd.Series()
            if desc_urls_exist:
                desc_urls = (self.nodes['user']['entities.description.urls']
                                 .dropna()
                                 .map(extract_url)
                                 .explode())
            else:
                desc_urls = pd.Series()
            urls = pd.concat((url_urls, desc_urls))
            node_df = pd.DataFrame(index=urls.tolist())
            data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'url' in self.nodes.keys():
                self.nodes['url'] = self.nodes['url'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('user', 'has_url', 'url')] = rel_df

        # (:User)-[:HAS_PROFILE_PICTURE_URL]->(:Url)
        user_cols = self.nodes['user'].columns
        profile_images_exist = 'profile_image_url' in user_cols
        if self.include_images and profile_images_exist:
            urls = self.nodes['user']['profile_image_url'].dropna()
            node_df = pd.DataFrame(index=urls.tolist())
            data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
            rel_df = pd.DataFrame(data_dict)
            if 'url' in self.nodes.keys():
                self.nodes['url'] = self.nodes['url'].append(node_df)
            else:
                self.nodes['url'] = node_df
            self.rels[('user', 'has_profile_picture_url', 'url')] = rel_df

    def _extract_places(self):
        '''Computes extra features for Place nodes'''
        if self.include_places and 'place' in self.nodes.keys():
            def get_lat(bbox: list) -> float:
                return (bbox[1] + bbox[3]) / 2
            def get_lng(bbox: list) -> float:
                return (bbox[0] + bbox[2]) / 2
            place_df = self.nodes['place']
            place_df['lat'] = place_df['geo.bbox'].map(get_lat)
            place_df['lng'] = place_df['geo.bbox'].map(get_lng)
            self.nodes['place'] = place_df

    def _extract_polls(self):
        '''Computes extra features for Poll nodes'''
        if self.include_polls and 'poll' in self.nodes.keys():
            def get_labels(options: List[dict]) -> List[str]:
                return [dct['label'] for dct in options]
            def get_votes(options: List[dict]) -> List[int]:
                return [dct['votes'] for dct in options]
            poll_df = self.nodes['poll']
            poll_df['labels'] = poll_df.options.map(get_labels)
            poll_df['votes'] = poll_df.options.map(get_votes)
            self.nodes['poll'] = poll_df

    def _extract_articles(self):
        '''Downloads the articles in the dataset'''
        if self.include_articles:

            # Create regex that filters out non-articles. These are common
            # images, videos and social media websites
            non_article_regexs = ['youtu[.]*be', 'vimeo', 'spotify', 'twitter',
                                  'instagram', 'tiktok', 'gab[.]com',
                                  'https://t[.]me', 'imgur', '/photo/',
                                  'mp4', 'mov', 'jpg', 'jpeg', 'bmp', 'png',
                                  'gif', 'pdf']
            non_article_regex = '(' + '|'.join(non_article_regexs) + ')'

            # Filter out the URLs to get the potential article URLs
            article_urls = [url for url in self.nodes['url'].index.tolist()
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
            article_urls = data_dict.pop('url')
            article_df = pd.DataFrame(data_dict, index=article_urls)
            self.nodes['article'] = article_df

            # (:Article)-[:HAS_TOP_IMAGE_URL]->(:Url)
            if self.include_images:
                urls = article_df.top_image_url.dropna()
                node_df = pd.DataFrame(index=urls.tolist())
                data_dict = dict(src=urls.index.tolist(), tgt=urls.tolist())
                rel_df = pd.DataFrame(data_dict)
                if 'url' in self.nodes.keys():
                    self.nodes['url'] = self.nodes['url'].append(node_df)
                else:
                    self.nodes['url'] = node_df
                self.rels[('article', 'has_top_image_url', 'url')] = rel_df

            # (:Tweet)-[:HAS_ARTICLE]->(:Article)
            is_article_url = (self.rels[('tweet', 'has_url', 'url')]
                                  .tgt
                                  .isin(article_df))
            rel_df = (self.rels[('tweet', 'has_url', 'url')][is_article_url]
                          .reset_index(drop=True))
            self.rels[('tweet', 'has_article', 'article')] = rel_df


    def _extract_images(self):
        '''Downloads the images in the dataset'''
        if self.include_images:

            # Create regex that filters out article urls
            image_urls = [url for url in self.nodes['url'].index.tolist()
                          if url not in self.nodes['article'].index.tolist()]

            # Loop over all the Url nodes
            data_dict = defaultdict(list)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                for result in tqdm(pool.imap_unordered(process_image_url,
                                                       image_urls,
                                                       chunksize=5),
                                   desc='Parsing images',
                                   total=len(image_urls)):

                    # Skip result if URL is not parseable
                    if result is None:
                        continue

                    # Store the data in the data dictionary
                    data_dict['url'].append(result['url'])
                    data_dict['pixels'].append(result['pixels'])
                    data_dict['height'].append(result['height'])
                    data_dict['width'].append(result['width'])

            # Convert the data dictionary to a dataframe and store it as the
            # `Image` node
            image_urls = data_dict.pop('url')
            image_df = pd.DataFrame(data_dict, index=image_urls)
            if 'image' in self.nodes:
                image_df = self.nodes['image'].append(image_df)
                image_df = image_df[~image_df.index.duplicated()]
                self.nodes['image'] = image_df
            else:
                self.nodes['image'] = image_df

            # (:Tweet)-[:HAS_IMAGE]->(:Image)
            rel = ('tweet', 'has_url', 'url')
            is_image_url = self.rels[rel].tgt.isin(image_df)
            rel_df = self.rels[rel][is_image_url].reset_index(drop=True)
            self.rels[('tweet', 'has_image', 'image')] = rel_df

            # (:Article)-[:HAS_TOP_IMAGE]->(:Image)
            rel = ('article', 'has_top_image_url', 'url')
            is_image_url = self.rels[rel].tgt.isin(image_df)
            rel_df = self.rels[rel][is_image_url].reset_index(drop=True)
            self.rels[('article', 'has_top_image', 'image')] = rel_df

            # (:User)-[:HAS_PROFILE_PICTURE]->(:Image)
            rel = ('user', 'has_profile_picture_url', 'url')
            is_image_url = self.rels[rel].tgt.isin(image_df)
            rel_df = self.rels[rel][is_image_url].reset_index(drop=True)
            self.rels[('user', 'has_profile_picture', 'image')] = rel_df

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
                          image=['url', 'pixels'],
                          article=['url', 'title', 'content'],
                          place=['name', 'full_name', 'country_code',
                                 'country', 'place_type', 'lat', 'lng'],
                          hashtag=[],  # Only contains the hashtag as index
                          poll=['labels', 'votes', 'end_datetime',
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
                                         .rename(renaming_dict))

    def _remove_auxilliaries(self):
        '''Removes node types that are not in use anymore'''

        # Remove auxilliary node types
        for node_type in self.nodes.keys():
            if node_type not in self._node_dump:
                self.nodes.pop(node_type)

        # Remove auxilliary relation types
        for rel_type in self.rels.keys():
            if rel_type not in self._rel_dump:
                self.rels.pop(rel_type)

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
