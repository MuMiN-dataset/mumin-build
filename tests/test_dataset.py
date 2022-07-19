"""Unit tests for the MuminDataset class"""

import os
import warnings
from copy import deepcopy
from pathlib import Path

import pytest
from dotenv import load_dotenv

from mumin import MuminDataset, load_dgl_graph, save_dgl_graph

load_dotenv()


class TestMuminDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        yield MuminDataset(str(os.getenv("TWITTER_API_KEY")), size="test", verbose=True)

    @pytest.fixture(scope="class")
    def compiled_dataset(self, dataset):
        yield dataset.compile(overwrite=True)

    @pytest.fixture(scope="class")
    def dataset_with_embeddings(self, compiled_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            copied_dataset = deepcopy(compiled_dataset)
            copied_dataset.add_embeddings()
            yield copied_dataset

    @pytest.fixture(scope="class")
    def dgl_graph(self, compiled_dataset):
        yield compiled_dataset.to_dgl()

    @pytest.fixture(scope="class")
    def dgl_graph_path(self):
        yield Path("data/mumin-test.dgl")

    def test_nodes_and_rels_are_empty_dicts(self, dataset):
        assert dataset.nodes == dict()
        assert dataset.rels == dict()

    def test_initialize_dataset_without_bearer_token(self, dataset):
        new_dataset = MuminDataset(size="test")
        assert dataset._twitter.api_key == new_dataset._twitter.api_key

    @pytest.mark.parametrize(
        argnames="node_type",
        argvalues=["claim", "tweet", "reply", "user", "article", "image", "hashtag"],
        ids=["claim", "tweet", "reply", "user", "article", "image", "hashtag"],
    )
    def test_compiled_dataset_contains_node_type(self, compiled_dataset, node_type):
        assert node_type in compiled_dataset.nodes.keys()

    @pytest.mark.parametrize(
        argnames="rel_type",
        argvalues=[
            ("tweet", "discusses", "claim"),
            ("tweet", "has_hashtag", "hashtag"),
            ("tweet", "has_article", "article"),
            ("reply", "reply_to", "tweet"),
            ("reply", "quote_of", "tweet"),
            ("user", "posted", "tweet"),
            ("user", "posted", "reply"),
            ("user", "mentions", "user"),
            ("tweet", "has_image", "image"),
        ],
        ids=[
            "tweet_discusses_claim",
            "tweet_has_hashtag_hashtag",
            "tweet_has_article_article",
            "reply_reply_to_tweet",
            "reply_quote_of_tweet",
            "user_posted_tweet",
            "user_posted_reply",
            "user_mentions_user",
            "tweet_has_image_image",
        ],
    )
    def test_compiled_dataset_contains_rel_type(self, compiled_dataset, rel_type):
        assert rel_type in compiled_dataset.rels.keys()

    @pytest.mark.skip(reason="DGL not available")
    def test_to_dgl(self, dgl_graph):
        from dgl import DGLHeteroGraph

        assert isinstance(dgl_graph, DGLHeteroGraph)

    @pytest.mark.skip(reason="DGL not available")
    def test_save_dgl(self, dgl_graph, dgl_graph_path):
        if dgl_graph_path.exists():
            dgl_graph_path.unlink()
        save_dgl_graph(dgl_graph, dgl_graph_path)
        assert dgl_graph_path.exists()

    @pytest.mark.skip(reason="DGL not available")
    def test_load_dgl(self, dgl_graph_path):
        from dgl import DGLHeteroGraph

        if dgl_graph_path.exists():
            dgl_graph = load_dgl_graph(dgl_graph_path)
            assert isinstance(dgl_graph, DGLHeteroGraph)
            dgl_graph_path.unlink()

    @pytest.mark.parametrize(
        argnames="node_type, embedding_name",
        argvalues=[
            ("tweet", "text_emb"),
            ("reply", "text_emb"),
            ("user", "description_emb"),
            ("article", "content_emb"),
            ("image", "pixels_emb"),
            ("claim", "reviewer_emb"),
        ],
        ids=[
            "tweet",
            "reply",
            "user",
            "article",
            "image",
            "claim",
        ],
    )
    def test_embed(self, dataset_with_embeddings, node_type, embedding_name):
        assert (
            len(dataset_with_embeddings.nodes[node_type]) == 0
            or embedding_name in dataset_with_embeddings.nodes[node_type].columns
        )
