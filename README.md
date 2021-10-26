# MuMiN-Build
This repository contains the package used to build the MuMiN dataset from the
paper [Nielsen and McConville: _MuMiN: A Large-Scale Multilingual Multimodal
Fact-Checked Misinformation Dataset with Linked Social Network Posts_
(2021)](https://openreview.net/forum?id=sOLdMFkQe7).


## Installation
Firstly, as the dataset is packaged in the HDF5 format, this needs to be
installed. It comes installed with most operating systems, but to be sure that
you have it you can install it as follows:

<details>
<summary>HDF5 installation for Ubuntu.</summary>

```shell
$ apt install libhdf5-dev libhdf5-serial-dev
```

</details>
<details>
<summary>HDF5 installation for CentOS.</summary>

```shell
$ yum install hdf5-devel
```

</details>
<details>
<summary>HDF5 installation for MacOS.</summary>

```shell
$ brew install hdf5
```

</details>

With HDF5 installed, the `mumin` package can be installed using `pip`:
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
MuminDataset(size='large', compiled=False)
```

By default, this loads the large version of the dataset. This can be changed by
setting the `size` argument of `MuminDataset` to one of 'small', 'medium' or
'large'. To begin using the dataset, it first needs to be compiled. This will
download the dataset, rehydrate the tweets and users, and download all the
associated news articles, images and videos. This usually takes a while.
```python
>>> dataset.compile()
MuminDataset(num_nodes=XXXXX, num_relations=XXXXX, size='large', compiled=True)
```

After compilation, the dataset can also be found in the `mumin-<size>.hdf`
file. This file name can be changed using the `dataset_path` argument when
initialising the `MuminDataset` class. If you need embeddings of the nodes, for
instance for use in machine learning models, then you can simply call the
`add_embeddings` method:
```python
>>> dataset.add_embeddings()
MuminDataset(num_nodes=XXXXX, num_relations=XXXXX, size='large', compiled=True)
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


## Dataset Statistics

| Size   | #Claims | #Threads | #Replies  | #Retweets |  #Users    | #Languages | %Misinfo |
| :---:  | ---:    | ---:     | ---:      | :---:     | :---:      | :---:      | :---:    |
| Large  | 12,347  | 24,773   | 1,024,070 | 695,924   | 4,306,272  | 41         | 94.57%   |
| Medium | 5,265   | 10,195   | 480,249   | 305,300   | 2,004,300  | 37         | 94.07%   |
| Small  | 2,089   | 4,126    | 220,862   | 132,561   | 916,697    | 35         | 92.87%   |


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-trawl](https://github.com/CLARITI-REPHRAIN/mumin-trawl),
  containing the source code used to construct the dataset from scratch.
