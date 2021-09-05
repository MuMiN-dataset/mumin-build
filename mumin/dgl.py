'''Functions related to exporting the dataset to the Deep Graph Library'''

from typing import Dict, Tuple
import pandas as pd


def build_dgl_dataset(nodes: Dict[str, pd.DataFrame],
                      relations: Dict[Tuple[str, str, str], pd.DataFrame],
                      output_format: str) -> 'DGLDataset':
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
        output_format (str, optional):
            The format the dataset should be outputted in. Can be
            'thread-level-graphs', 'claim-level-graphs' and 'single-graph'.
            Defaults to 'thread-level-graphs'.

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

    # Case where we want to do graph classification for each Twitter thread
    if output_format == 'thread-level-graphs':

        class MuminDGLDataset(DGLDataset):
            def __init__(self):
                super().__init__(name='mumin')

            def process(self):

                # Get the list of all source tweets, as these are effectively a
                # unique ID per graph, and get the associated labels
                discusses_rel = relations[('tweet', 'discusses', 'claim')]
                source_tweets = discusses_rel.src.tolist()
                claim_ids = discusses_rel.tgt.tolist()
                labels = nodes['claim'][claim_ids].predicted_verdict.tolist()

                # Build a graph data dict, used to build the `dgl` graph without any
                # features
                graph_data = dict()
                for canonical_etype, rel_arr in relations.items():
                    rel_arr = relations[canonical_etype]
                    src_tensor = torch.from_numpy(rel_arr[:, 0])
                    tgt_tensor = torch.from_numpy(rel_arr[:, 1])
                    graph_data[canonical_etype] = (src_tensor, tgt_tensor)
                dgl_graph = dgl.heterograph(graph_data)

                # Add the node features
                for node_type, node_arr in nodes.items():
                    if node_arr.size > 0:
                        node_feats = torch.from_numpy(node_arr)
                        dgl_graph.nodes[node_type].data['feat'] = node_feats

                # Add the edge features
                for canonical_etype, rel_arr in nodes.items():
                    if rel_arr.shape[1] > 2:
                        edge_feats = torch.from_numpy(rel_arr[:, 2:])
                        dgl_graph.edges[canonical_etype].data['feat'] = edge_feats

                # Return the `dgl` graph
                return dgl_graph
