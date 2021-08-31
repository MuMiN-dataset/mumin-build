# MuMiN-Build
This repository contains the package used to build the MuMiN dataset from the
paper [Nielsen and McConville: _MuMiN: A Large-Scale Multilingual Multimodal
Fact-Checked Misinformation Dataset with Linked Social Network Posts_
(2021)](todo).


## Installation
The `mumin` package can be installed with `pip` as follows:
```shell
$ pip install mumin
```

To be able to build the dataset Twitter data needs to be downloaded, which
requires a Twitter API key. You can get one
[for free here](https://developer.twitter.com/en/portal/dashboard).


## Quickstart
The main class of the package is the `MuMiNDataset` class:
```python
>>> from mumin import MuMiNDataset
>>> dataset = MuMiNDataset(twitter_api_key=XXXXX,
                           twitter_api_secret=XXXXX,
                           twitter_access_token=XXXXX,
                           twitter_access_secret=XXXXX)
>>> dataset
MuMiNDataset(compiled=False)
```

To begin using the dataset, it first needs to be compiled. This will download
the dataset, rehydrate the tweets and users, and download all the associated
news articles, images and videos. This usually takes a while.
```python
>>> dataset.compile()
>>> dataset
MuMiNDataset(num_nodes=3,000,000, num_relations=5,000,000, compiled=True)
```

With the dataset compiled, it can then be exported to the format you require.
The following formats are supported:
- [Deep Graph Library](https://www.dgl.ai/): The `to_dgl` method will return a
  `DGLDataset`,
  which can be used directly with the `dgl` library.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/): The
  `to_pyg` method will return an `InMemoryDataset`, which can be used directly
  with the `pytorch_geometric` library.

After compilation the dataset can also be found in the `./mumin` folder as
separate CSV files. This path can be changed using the `dataset_dir` argument
when initialising the `MuMiNDataset` class.


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/CLARITI-REPHRAIN/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
