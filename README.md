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
MuMiNDataset(size=large, compiled=False)
```

By default, this loads the large version of the dataset. This can be changed by
setting the `size` argument to one of 'small', 'medium' or 'large'. To begin
using the dataset, it first needs to be compiled. This will download the
dataset, rehydrate the tweets and users, and download all the associated news
articles, images and videos. This usually takes a while.
```python
>>> dataset.compile()
>>> dataset
MuMiNDataset(num_nodes=9,535,121, num_relations=15,232,212, size=large, compiled=True)
```

With the dataset compiled, it can then be exported to the format you require.
If you want to use the dataset in the [Deep Graph Library](https://www.dgl.ai/), then simply use the `to_dgl` method:
```python
>>> dgl_dataset = dataset.to_dgl()
>>> type(dgl_dataset)
<class 'dgl.data.dgl_dataset.DGLDataset'>
```

If you want to work the dataset in the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/), then analogously use the `to_pyg` method:
```python
>>> pyg_dataset = dataset.to_pyg()
>>> type(pyg_dataset)
<class 'torch_geometric.data.in_memory_dataset.InMemoryDataset'>
```

After compilation, the dataset can also be found in the `./mumin` folder as
separate CSV files. This path can be changed using the `dataset_dir` argument
when initialising the `MuMiNDataset` class.


## Dataset Statistics


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/CLARITI-REPHRAIN/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
