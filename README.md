# MuMiN-Build
This repository contains the package used to build the MuMiN dataset from the
paper [Nielsen and McConville: _MuMiN: A Large-Scale Multilingual Multimodal
Fact-Checked Misinformation Dataset with Linked Social Network Posts_
(2021)](https://openreview.net/forum?id=sOLdMFkQe7).

### This is currently under review at NeurIPS 2021 Datasets and Benchmarks Track (Round 2). This dataset must not be used until this warning is removed as the dataset is subject to change, for example, during the review period.

## Installation
The `mumin` package can be installed from source:
```shell
$ pip install git+https://github.com/CLARITI-REPHRAIN/mumin-build
```

To be able to build the dataset Twitter data needs to be downloaded, which
requires a Twitter API key. You can get one
[for free here](https://developer.twitter.com/en/portal/dashboard). You will
need the _Bearer Token_.


## Quickstart
The main class of the package is the `MuminDataset` class:
```python
>>> from mumin import MuminDataset
>>> dataset = MuminDataset(twitter_bearer_token=XXXXX)
>>> dataset
MuminDataset(size='large', compiled=False)
```

By default, this loads the large version of the dataset. This can be changed by
setting the `size` argument to one of 'small', 'medium' or 'large'. To begin
using the dataset, it first needs to be compiled. This will download the
dataset, rehydrate the tweets and users, and download all the associated news
articles, images and videos. This usually takes a while.
```python
>>> dataset.compile()
>>> dataset
MuminDataset(num_nodes=9,535,121, num_relations=15,232,212, size='large', compiled=True)
```

After compilation, the dataset can also be found in the `./mumin` folder as
separate `csv` files. This path can be changed using the `dataset_dir` argument
when initialising the `MuminDataset` class.

It is possible to export the dataset to a library-specific class for your
convenience. Such exports depends on both the _library_ that you are working in
and the _format_ you want the data in. For the library aspect, `mumin`
currently supports the [Deep Graph Library](https://www.dgl.ai/) and
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

**Note**: If you need to use the `to_dgl` or `to_pyg` methods, you need to
install the `mumin` package as `pip install mumin[dgl]` or `pip install
mumin[pyg]`, respectively. This is to avoid unnecessary library downloads if
all you require are the `csv` files.

By default, the above `dgl`/`pyg` export assumes that you want to perform graph
classification on the graphs pertaining to each Twitter thread. You can change
this using the `output_format` argument in the `to_dgl` and `to_pyg` methods.
We currently support the following formats:
- `thread-level-graphs`: The default value, which outputs a separate graph per
  Twitter thread. This can be used to do thread-level graph classification.
- `claim-level-graphs`: A separate graph per claim; i.e., each graph contains
  multiple Twitter threads, which might be connected. This can be used to do
  claim-level graph classification.
- `single-graph`: The entire dataset in a single graph. This can be used to do
  various node classification and link prediction tasks.


## Dataset Statistics

| Similarity threshold | #Claims | #Threads | #Replies  | #Retweets |  #Users    | #Languages | %`misinformation` |
| :---:                | ---:    | ---:     | ---:      | :---:     | :---:      | :---:      | :---:             |
| 0.70                 | 12,242  | 23,856   | 798,259   | 2,251,263 | 5,525,194  | 41         | 94.81%            |
| 0.75                 | 5,244   | 9,863    | 427,472   | 1,299,096 | 2,894,456  | 37         | 94.34%            |
| 0.80                 | 2,079   | 4,018	| 258,455   | 811,078   | 1,611,344  | 35         | 93.20%            |


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/CLARITI-REPHRAIN/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
