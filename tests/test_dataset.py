'''Unit tests for the MuminDataset class'''

from dotenv import load_dotenv
import os
from mumin.dataset import MuminDataset
from dgl import DGLHeteroGraph
import pytest


load_dotenv()


class TestMuminDataset:

    @pytest.fixture(scope='class')
    def dataset(self):
        yield MuminDataset(str(os.getenv('TWITTER_API_KEY')),
                           size='test',
                           verbose=True)

    @pytest.fixture(scope='class')
    def compiled_dataset(self, dataset):
        yield dataset.compile(overwrite=True)

    def test_init(self, dataset):
        assert dataset.nodes == dict()
        assert dataset.rels == dict()

    def test_compile(self, compiled_dataset):
        nodes = ['claim', 'tweet', 'reply', 'user', 'article',
                 'image', 'hashtag']
        for node in nodes:
            assert node in compiled_dataset.nodes.keys()
        rels = [('tweet', 'discusses', 'claim'),
                ('tweet', 'has_hashtag', 'hashtag'),
                ('tweet', 'has_article', 'article'),
                ('reply', 'reply_to', 'tweet'),
                ('reply', 'quote_of', 'tweet'),
                ('user', 'posted', 'tweet'),
                ('user', 'posted', 'reply'),
                ('user', 'mentions', 'user')]
        for rel in rels:
            assert rel in compiled_dataset.rels.keys()

    def test_to_dgl(self, compiled_dataset):
        dgl_graph = compiled_dataset.to_dgl()
        assert isinstance(dgl_graph, DGLHeteroGraph)
