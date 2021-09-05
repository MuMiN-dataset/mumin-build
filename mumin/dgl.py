'''Functions related to exporting the dataset to the Deep Graph Library'''

from typing import Dict, Tuple
import pandas as pd
import numpy as np


def build_dgl_dataset(nodes: Dict[str, pd.DataFrame],
                      relations: Dict[Tuple[str, str, str], pd.DataFrame],
                      ) -> 'DGLDataset':
    '''Convert the dataset to a DGL dataset.

    This assumes that the dataset has been compiled and thus also dumped to a
    local file.

    Args:
        nodes (dict):
            The nodes of the dataset, with keys the node types and NumPy arrays
            as the values.
        relations (dict):
            The relations of the dataset, with keys being triples of strings
            (source_node_type, relation_type, target_node_type) and NumPy
            arrays as the values.

    Returns:
        DGLDataset:
            The dataset in DGL format.

    Raises:
        ModuleNotFoundError:
            If `dgl` has not been installed, due to `mumin` not having been
            installed with the `dgl` extension, like so: `pip install
            mumin[dgl]`.
    '''
    # Import the needed libraries, and raise an error if they have not yet been
    # installed
    try:
        import dgl
        from dgl.data import DGLDataset
        import torch
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Could not find the `dgl` library. Try '
                                  'installing the `mumin` library with the '
                                  '`dgl` extension, like so: `pip install '
                                  'mumin[dgl]`')

    # Set up the graph as a DGL graph
    graph_data = dict()
    for canonical_etype, rel_arr in relations.items():
        rel_arr = relations[canonical_etype].to_numpy()
        src_tensor = torch.from_numpy(rel_arr[:, 0]).int()
        tgt_tensor = torch.from_numpy(rel_arr[:, 1]).int()
        graph_data[canonical_etype] = (src_tensor, tgt_tensor)
    dgl_graph = dgl.heterograph(graph_data)

    def emb_to_tensor(df: pd.DataFrame, col_name: str):
        np_array = np.stack(df[col_name].tolist())
        return torch.from_numpy(np_array)

    # Add node features to the Tweet nodes
    cols = ['num_retweets', 'num_replies', 'num_quote_tweets']
    tweet_feats = torch.from_numpy(nodes['tweet'][cols].to_numpy())
    if ('text_emb' in nodes['tweet'].columns and
            'lang_emb' in nodes['tweet'].columns):
        tweet_embs = emb_to_tensor(nodes['tweet'], 'text_emb')
        lang_embs = emb_to_tensor(nodes['tweet'], 'lang_emb')
        tensors = (tweet_embs, lang_embs, tweet_feats)
    else:
        tensors = (tweet_feats,)
    dgl_graph.nodes['tweet'].data['feat'] = torch.cat(tensors, dim=1)

    # Add node features to the User nodes
    nodes['user']['verified'] = nodes['user'].verified.astype(int)
    nodes['user']['protected'] = nodes['user'].verified.astype(int)
    cols = ['verified', 'protected', 'num_followers', 'num_followees',
            'num_tweets', 'num_listed']
    user_feats = torch.from_numpy(nodes['user'][cols].to_numpy())
    if 'description_emb' in nodes['user'].columns:
        user_embs = emb_to_tensor(nodes['user'],
                                               'description_emb')
        tensors = (user_embs, user_feats)
    else:
        tensors = (user_feats,)
    dgl_graph.nodes['user'].data['feat'] = torch.cat(tensors, dim=1)

    # Add node features to the Article nodes
    if 'article' in nodes.keys():
        if ('title_emb' in nodes['article'].columns and
                'content_emb' in nodes['article'].columns):
            title_embs = emb_to_tensor(nodes['article'], 'title_emb')
            content_embs = emb_to_tensor(nodes['article'], 'content_emb')
            tensors = (title_embs, content_embs)
            dgl_graph.nodes['article'].data['feat'] = torch.cat(tensors, dim=1)
        else:
            num_articles = dgl_graph.num_nodes('article')
            ones = torch.ones(num_articles, 1)
            dgl_graph.nodes['article'].data['feat'] = ones

    # Add node features to the Image nodes
    if 'image' in nodes.keys():
        if 'pixels_emb' in nodes['image'].columns:
            image_embs = emb_to_tensor(nodes['image'], 'pixels_emb')
            dgl_graph.nodes['image'].data['feat'] = image_embs
        else:
            num_images = dgl_graph.num_nodes('image')
            dgl_graph.nodes['image'].data['feat'] = torch.ones(num_images, 1)

    # Add node features to the Place nodes
    if 'place' in nodes.keys():
        cols = ['lat', 'lng']
        place_feats = torch.from_numpy(nodes['place'][cols].to_numpy())
        dgl_graph.nodes['place'].data['feat'] = place_feats

    # Add node features to the Poll nodes
    if 'poll' in nodes.keys():
        num_polls = dgl_graph.num_nodes('poll')
        dgl_graph.nodes['poll'].data['feat'] = torch.ones(num_polls, 1)

    # Add node features to the Hashtag nodes
    if 'hashtag' in nodes.keys():
        num_hashtags = dgl_graph.num_nodes('hashtag')
        dgl_graph.nodes['hashtag'].data['feat'] = torch.ones(num_hashtags, 1)

    # Add node features to the Claim nodes
    if 'claim' in nodes.keys():
        if 'reviewer_emb' in nodes['claim'].columns:
            rev_embs = emb_to_tensor(nodes['claim'], 'reviewer_emb')
            dgl_graph.nodes['claim'].data['feat'] = rev_embs
        else:
            num_claims = dgl_graph.num_nodes('claim')
            dgl_graph.nodes['claim'].data['feat'] = torch.ones(num_claims, 1)

    return dgl_graph
