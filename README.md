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
This depends on both the _library_ that you are working in and the _format_ you
want the data in. For the library aspect, `mumin` currently supports the
[Deep Graph Library](https://www.dgl.ai/) and
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). They
can be exported using the `to_dgl` and `to_pyg` methods as follows:
```python
>>> dgl_dataset = dataset.to_dgl()
>>> type(dgl_dataset)
<class 'dgl.data.dgl_dataset.DGLDataset'>
>>> pyg_dataset = dataset.to_pyg()
>>> type(pyg_dataset)
<class 'torch_geometric.data.in_memory_dataset.InMemoryDataset'>
```

By default, this assumes that you want to perform graph classification on the
graphs pertaining to each Twitter thread. You can change this using the
`output_format` argument in the `to_dgl` and `to_pyg` methods. We currently
support the following formats:
- `thread-level-graphs`: The default value, which outputs a separate graph per
  Twitter thread. This can be used to do thread-level graph classification.
- `claim-level-graphs`: A separate graph per claim; i.e., each graph contains
  multiple Twitter threads, which might be connected. This can be used to do
  claim-level graph classification.
- `single-graph`: The entire dataset in a single graph. This can be used to do
  various node classification and link prediction tasks.

After compilation, the dataset can also be found in the `./mumin` folder as
separate CSV files. This path can be changed using the `dataset_dir` argument
when initialising the `MuMiNDataset` class.


## Dataset Statistics

| Similarity threshold | #Claims | #Tweets   | #Users    | #Languages | %`misinformation` |
| ---:                 | ---:    | ---:      | ---:      | ---:       | ---:              |
| 0.70                 | 10,187  | 2,755,168 | 4,978,321 | 41         | 94.80%            |
| 0.75                 | 4,526   | 1,578,492 | 2,627,006 | 36         | 94.26%            |
| 0.80                 | 1,837   | 999,539   | 1,485,367 | 34         | 92.98%            |


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/CLARITI-REPHRAIN/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
