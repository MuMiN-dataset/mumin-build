"""Unit tests for the MuminDataset class"""

import os
import warnings
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
    def dgl_graph(self, compiled_dataset):
        yield compiled_dataset.to_dgl()

    @pytest.fixture(scope="class")
    def dgl_graph_path(self):
        yield Path("data/mumin-test.dgl")

    def test_init(self, dataset):
        assert dataset.nodes == dict()
        assert dataset.rels == dict()

    def test_init_no_bearer_token(self, dataset):
        new_dataset = MuminDataset(size="test")
        assert dataset._twitter.api_key == new_dataset._twitter.api_key

    def test_compile(self, compiled_dataset):
        nodes = ["claim", "tweet", "reply", "user", "article", "image", "hashtag"]
        for node in nodes:
            assert node in compiled_dataset.nodes.keys()
        rels = [
            ("tweet", "discusses", "claim"),
            ("tweet", "has_hashtag", "hashtag"),
            ("tweet", "has_article", "article"),
            ("reply", "reply_to", "tweet"),
            ("reply", "quote_of", "tweet"),
            ("user", "posted", "tweet"),
            ("user", "posted", "reply"),
            ("user", "mentions", "user"),
            ("tweet", "has_image", "image"),
        ]
        for rel in rels:
            assert rel in compiled_dataset.rels.keys()

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

    def test_embed(self, compiled_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            compiled_dataset.add_embeddings()
            assert (
                len(compiled_dataset.nodes["tweet"]) == 0
                or "text_emb" in compiled_dataset.nodes["tweet"].columns
            )
            assert (
                len(compiled_dataset.nodes["reply"]) == 0
                or "text_emb" in compiled_dataset.nodes["reply"].columns
            )
            assert (
                len(compiled_dataset.nodes["user"]) == 0
                or "description_emb" in compiled_dataset.nodes["user"].columns
            )
            assert (
                len(compiled_dataset.nodes["article"]) == 0
                or "content_emb" in compiled_dataset.nodes["article"].columns
            )
            assert (
                len(compiled_dataset.nodes["image"]) == 0
                or "pixels_emb" in compiled_dataset.nodes["image"].columns
            )
            assert (
                len(compiled_dataset.nodes["claim"]) == 0
                or "reviewer_emb" in compiled_dataset.nodes["claim"].columns
            )
