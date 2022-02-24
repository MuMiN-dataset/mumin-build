# MuMiN-Build
This repository contains the package used to build the MuMiN dataset from the
paper [Nielsen and McConville: _MuMiN: A Large-Scale Multilingual Multimodal
Fact-Checked Misinformation Social Network Dataset_
(2021)](https://arxiv.org/abs/2202.11684).

See [the MuMiN website](https://mumin-dataset.github.io/) for more information,
including a leaderboard of the top performing models.


## Installation
The `mumin` package can be installed using `pip`:
```shell
$ pip install mumin
```

To be able to build the dataset, Twitter data needs to be downloaded, which
requires a Twitter API key. You can get one
[for free here](https://developer.twitter.com/en/portal/dashboard). You will
need the _Bearer Token_.


## Quickstart
The main class of the package is the `MuminDataset` class:
```python
>>> from mumin import MuminDataset
>>> dataset = MuminDataset(twitter_bearer_token=XXXXX)
>>> dataset
MuminDataset(size='small', compiled=False)
```

By default, this loads the small version of the dataset. This can be changed by
setting the `size` argument of `MuminDataset` to one of 'small', 'medium' or
'large'. To begin using the dataset, it first needs to be compiled. This will
download the dataset, rehydrate the tweets and users, and download all the
associated news articles, images and videos. This usually takes a while.
```python
>>> dataset.compile()
MuminDataset(num_nodes=388,149, num_relations=475,490, size='small', compiled=True)
```

Note that this dataset does not contain _all_ the nodes and relations in
MuMiN-small, as that would take way longer to compile. The data left out are
timelines, profile pictures and article images. These can be included by
specifying `include_extra_images=True` and/or `include_timelines=True` in the
constructor of `MuminDataset`.

After compilation, the dataset can also be found in the `mumin-<size>.zip`
file. This file name can be changed using the `dataset_path` argument when
initialising the `MuminDataset` class. If you need embeddings of the nodes, for
instance for use in machine learning models, then you can simply call the
`add_embeddings` method:
```python
>>> dataset.add_embeddings()
MuminDataset(num_nodes=388,149, num_relations=475,490, size='small', compiled=True)
```

**Note**: If you need to use the `add_embeddings` method, you need to install
the `mumin` package as either `pip install mumin[embeddings]` or `pip install
mumin[all]`, which will install the `transformers` and `torch` libraries. This
is to ensure that such large libraries are only downloaded if needed.

It is possible to export the dataset to the
[Deep Graph Library](https://www.dgl.ai/), using the `to_dgl` method:
```python
>>> dgl_graph = dataset.to_dgl()
>>> type(dgl_graph)
dgl.heterograph.DGLHeteroGraph
```

**Note**: If you need to use the `to_dgl` method, you need to install the
`mumin` package as `pip install mumin[dgl]` or `pip install mumin[all]`, which
will install the `dgl` and `torch` libraries.

For a more in-depth tutorial of how to work with the dataset, including
training multiple different misinformation classifiers, see [the
tutorial](https://colab.research.google.com/drive/1Kz0EQtySYQTo1ui8F2KZ6ERneZVf5TIN).


## Dataset Statistics

| Dataset | #Claims | #Threads | #Tweets | #Users | #Articles | #Images | #Languages | %Misinfo |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MuMiN-large | 12,914 | 26,048 | 21,565,018 | 1,986,354 | 10,920 | 6,573 | 41 | 94.57% |
| MuMiN-medium | 5,565 | 10,832 | 12,650,371 | 1,150,259 | 4,212 | 2,510 | 37 | 94.07% |
| MuMiN-small | 2,183 | 4,344 | 7,202,506 | 639,559 | 1,497 | 1,036 | 35 | 92.87% |


## Related Repositories
- [MuMiN website](https://mumin-dataset.github.io/), the central place for the
  MuMiN ecosystem, containing tutorials, leaderboards and links to the paper
  and related repositories.
- [MuMiN](https://github.com/MuMiN-dataset/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/MuMiN-dataset/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
- [MuMiN-baseline](https://github.com/MuMiN-dataset/mumin-baseline),
  containing the source code for the baselines.
