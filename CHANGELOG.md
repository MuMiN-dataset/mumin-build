# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Now allows setting `twitter_bearer_token=None` in the constructor of
  `MuminDataset`, which uses the environment variable `TWITTER_API_KEY`
  instead, which can be stored in a separate `.env` file. This is now the
  default value of `twitter_bearer_token`.


## [v1.6.2] - 2022-03-21
### Fixed
- Now removes claims that are only connected to deleted tweets when calling
  `to_dgl`. This previously caused a bug that was due to a mismatch between
  nodes in the dataset (which includes deleted ones) and nodes in the DGL graph
  (which does not contain the deleted ones).


## [v1.6.1] - 2022-03-17
### Fixed
- Now correctly catches JSONDecodeError during rehydration.


## [v1.6.0] - 2022-03-10
### Changed
- Changed the download link from Git-LFS to the official data.bris data
  repository, with URI
  [https://doi.org/10.5523/bris.23yv276we2mll25fjakkfim2ml](https://doi.org/10.5523/bris.23yv276we2mll25fjakkfim2ml).


## [v1.5.0] - 2022-02-19
### Changed
- Now using dicts rather than Series in `to_dgl`. This improved the wall time
  from 1.5 hours to 2 seconds!

### Fixed
- There was a bug in the call to `dgl.data.utils.load_graphs` causing
  `load_dgl_graph` to fail. This is fixed now.


## [v1.4.1] - 2022-02-19
### Changed
- Now only saves dataset at the end of `add_embeddings` if any embeddings were
  added.


## [v1.4.0] - 2022-02-19
### Added
- The `to_dgl` method is now being parallelised, speeding export up
  significantly.
- Added convenience functions `save_dgl_graph` and `load_dgl_graph`, which
  stores the Boolean train/val/test masks as unsigned 8-bit integers and
  handles the conversion. Using the `dgl`-native `save_graphs` and
  `load_graphs` causes an error, as it cannot handle Boolean tensors. These two
  convenience functions can be loaded simply as
  `from mumin import save_dgl_graph, load_dgl_graph`.


## [v1.3.0] - 2022-02-18
### Added
- Now uses GPU to embed all the text and images, if available.


## [v1.2.5] - 2022-02-06
### Fixed
- Now does not raise an error if we are not authorised to rehydrate a tweet,
  and instead merely skips it.


## [v1.2.4] - 2022-01-24
### Fixed
- Changed the minimum Python version compatible with `mumin` to 3.7, rather
  than 3.4.


## [v1.2.3] - 2021-12-15
### Fixed
- During rehydration, the authors of the source tweets were not included, and
  the images from tweets were not included either. They are now included.


## [v1.2.2] - 2021-12-15
### Fixed
- Now replacing NaN values for Numpy features with `np.nan` instead of an
  array, as `fillna` does not accept that. These are then converted in a scalar
  array with a `np.nan` value.


## [v1.2.1] - 2021-12-15
### Fixed
- When running `add_embeddings`, only embeddings to existing nodes will be
  added. This caused an error when e.g. images were not included in the
  dataset.


## [v1.2.0] - 2021-12-14
### Changed
- If tweets have been deleted (and thus cannot be rehydrated) then we keep them
  along with their related entities, just without being able to populate their
  features. When exporting to DGL then neither these tweets nor their replies
  are included.

### Added
- Now includes a check that tweets are actually rehydrated, and raises an error
  if they are not. Such an error is usually due to the inputted Twitter Bearer
  Token being invalid.


## [v1.1.1] - 2021-12-13
### Fixed
- Fixed bug in producing embeddings


## [v1.1.0] - 2021-12-12
### Fixed
- Updated the dataset with deduplicated entries. The deduplication is done such
  that the duplicate with the largest `relevance` parameter is kept.
- Include checks of whether nodes and relations exist, before extracting data
  from them.

### Added
- Added `include_timelines` option, which allows one to not include all the
  extra tweets in the timelines if not needed. As this greatly increases the
  amount of tweets needed to rehydrate, it defaults to False.


## [v1.0.2] - 2021-12-09
### Fixed
- Removed the relations from the dump which we are getting through compilation
  anyway.
- Updated the filtering mechanism, so that the `relevance` parameter is built
  in to all nodes and relations upon download.
- Deal with the situation where no relations exist of a certain type, above a
  specified threshold.


## [v1.0.1] - 2021-12-05
### Fixed
- Added in the `POSTED` relation, as leaving this out effectively meant that
  all the new tweets were filtered out during compilation.


## [v1.0.0] - 2021-12-03
### Changed
- Added new version of the dataset, which now includes a sample of ~100
  timeline tweets for every user. This approximately doubles the dataset size,
  to ~200MB before compilation. This new dataset includes different
  train/val/test splits as well, which is now 80/10/10 rather than 60/10/30.
  This means that the training dataset will see a much more varied amount of
  events (6-7) compared to the previous 2.


## [v0.7.0] - 2021-12-02
### Changed
- Changed `include_images` to `include_tweet_images`, which now only includes
  the images from the tweets themselves. Further, `include_user_images` is
  changed to `include_extra_images`, which now includes both profile pictures
  and the top images from articles. The tweet pictures are included by default,
  and the extras are not. This is to reduce the size of the default dataset, to
  make it easier to use.


## [v0.6.0] - 2021-12-01
### Changed
- Split up the `include_images` into `include_images` and
  `include_user_images`, with the former including images from tweets and
  articles, and the latter being profile pictures. The former has been set to
  True by default, and the latter False. This is due to the large amount of
  profile pictures making the dataset excessively large.

### Fixed
- Now catches connection errors when attempting to rehydrate tweets.


## [v0.5.3] - 2021-11-26
### Fixed
- Masks have been changed to boolean tensors, as otherwise indexing did not
  work properly.
- In the case where a claim/tweet does not have any label, this produces NaN
  values in the mask- and label tensors. These are now substituted for zeroes.
  This means that they will always be masked out, and so the label will not
  matter anyway.


## [v0.5.2] - 2021-11-24
### Fixed
- Now converting masks to long tensors, which is required for them to be used
  as indexing tensors in PyTorch.

### Changed
- Now only dumping dataset once while adding embeddings, where previously it
  dumped the dataset after adding embeddings to each node type. This is done to
  add embeddings faster, as the dumping of the dataset can take quite a long
  time.
- Now blanket catching all errors when processing images and articles, as there
  were too many edge cases.


## [v0.5.1] - 2021-11-09
### Fixed
- When encountering HTTP status 401 (unauthorized) during rehydration, we skip
  that batch of tweets.
- Image relations were extracted incorrectly, due to a wrong treatment of the
  images coming directly from the tweets via the `media_key` identifier, and
  the images coming from URLs present in the tweets themselves. Both are now
  correctly included in a uniform fashion.
- Datatypes are now only set for a given node if the node is included in the
  dataset. For instance, datatypes for the article features are only set if
  `include_articles == True`.


## [v0.5.0] - 2021-11-08
### Added
- The `Claim` nodes now have `language`, `keywords`, `cluster_keywords` and
  `cluster` attributes.
- Now sets datatypes for all the dataframes, to reduce memory usage.

### Fixed
- Updated `README` to a single zip file, rather than stating that the dataset
  is saved as a bunch of CSV files.
- Fixed image embedding shape from (1, 768) to (768,).
- Article embeddings are now computed correctly.
- Catch `IndexError` and `LocationParseError` when processing images.

### Changed
- Now dumping files incrementally rather than keeping all of them in memory, to
  avoid out-of-memory issues when saving the dataset.
- Dataset `size` argument now defaults to 'small', rather than 'large'.
- Updated the dataset. This is still not the final version: timelines of users
  are currently missing.
- Now storing the dataset in a zip file of Pickle files instead of HDF. This is
  because of HDF requiring extra installation, and there being maximal storage
  requirements in the dataframes when storing as HDF. The resulting zip file of
  Pickle files is stored with protocol 4, making it compatible with Python 3.4
  and newer. Further, the dataset being downloaded has been heavily compressed,
  taking up a quarter of the disk space compared to the previous CSV approach.
  When the dataset has been downloaded it will be converted to a less
  compressed version, taking up more space but making loading and saving much
  faster.

### Removed
- Disabled `numexpr`, `transformers` and `bs4` logging.


## [v0.4.0] - 2021-10-26
### Fixed
- All embeddings are now extracted from the pooler output, corresponding to the
  `[CLS]` tag.
- Ensured that train/val/test masks are boolean tensors when exporting to DGL,
  as opposed to binary integers.
- Content embeddings for articles were not aggregated per chunk, but now a mean
  is taken across all content chunks.
- Assign zero embeddings to user descriptions if they are not available.

### Changed
- The DGL graph returned by the `to_dgl` method now returns a bidirectional
  graph.
- The `verbose` argument of `MuminDataset` now defaults to `True`.
- Now storing the dataset as a single HDF file instead of a zipped folder of
  CSV files, primarily because data types are being preserved in this way, and
  that HDF is a binary format supported by Pandas which can handle
  multidimensional ndarrays as entries in a dataframe.
- The default models used to embed texts and images are now `xlm-roberta-base`
  and `google/vit-base-patch16-224-in21k`.

### Removed
- Removed the `poll` and `place` nodes, as they were too few to matter.
- Removed the `(:User)-[:HAS_PINNED]->(:Tweet)` relation, as there were too few
  of them to matter.


## [v0.3.1] - 2021-10-19
### Fixed
- Fixed the shape of the user description embeddings.


## [v0.3.0] - 2021-10-18
### Fixed
- Now catches `SSLError` and `OSError` when processing images.
- Now catches `ReadTimeoutError` when processing articles.
- The `(:Tweet)-[:MENTIONS]->(:User)` was missing in the dataset. It has now
  been added back in.
- Added tokenizer truncation when adding node embeddings.
- Fixed an issue with embedding user descriptions when the description is not
  available.

### Changed
- Changed the download link to the dataset, which now fetches the dataset from
  a specific commit, enabling proper dataset versioning.
- Changed the timeout parameter when downloading images from five seconds to
  ten seconds.
- Now processing 50 articles and images on each worker, compared to the
  previous 5.
- When loading in an existing dataset, auxilliaries and islands are removed.
  This ensures that `to_dgl` works properly.

### Removed
- Removed the review warning from the `README` and when initialising the
  dataset. The dataset is still not complete, in the sense that we will add
  retweets and timelines, but we will instead just keep versioning the dataset
  until we have included these extra features.


## [v0.2.0] - 2021-10-12
### Added
- Added claim embeddings to Claim nodes, being the transformer embeddings of
  the claims translated to English, as described in the paper.
- Added train/val/test split to claim nodes. When exporting to DGL using the
  `to_dgl` method, the Claim and Tweet nodes will have `train_mask`, `val_mask`
  and `test_mask` attributes that can be used to control loss and metric
  calculation. These are consistent, meaning that tweets connected to claims
  will always belong to the same split.
- Added labels to both Tweet and Claim nodes.

### Fixed
- Properly embeds reviewers of claims in case a claim has been reviewed by
  multiple reviewers.
- Load claim embeddings properly.
- Catches `TooManyRequests` exception when extracting images.
- Load dataset CSVs with Python engine, as the C engine caused errors.
- Disable tokenizer parallelism, which caused warning messages during
  rehydration of tweets.
- Ensure proper quoting of strings when dumping dataset to CSVs.
- Enable truncation of strings before tokenizing, when embedding texts.
- Convert masks to integers, which caused an issue when exporting to a DGL
  graph.
- Bug when computing reviewer embeddings for claims.
- Now properly shows `compiled=True` when printing the dataset, after
  compilation.

### Changed
- Changed disclaimer about review period.


## [v0.1.4] - 2021-09-13
### Fixed
- Include `(:User)-[:POSTED]->(:Reply)` in the dataset, extracted from the
  rehydrated reply and quote tweets.


## [v0.1.3] - 2021-09-13
### Fixed
- Compilation error when including images.
- Only include videos if they are present in the dataset.
- Ensure that article embeddings can properly be converted to PyTorch tensors
  when exporting to DGL.


## [v0.1.2] - 2021-09-13
### Fixed
- The replies were not reduced correctly when the `small` or `medium` variants
  of the dataset was compiled.
- The reply features were not filtered and renamed properly, to keep them
  consistent with the tweet nodes.
- Users without any description now gets assigned a zero vector as their
  description embedding.
- If a relation does not have any node pairs then do not try to create a
  corresponding DGL relation.
- Reset `nodes` and `rels` attributes when loading dataset.
- Add embeddings for `Reply` nodes.


## [v0.1.1] - 2021-09-08
### Changed
- Changed installation instructions in readme to `pip install mumin`.


## [v0.1.0] - 2021-09-07
### Added
- First release, including a `MuminDataset` class that can compile the dataset
  dump the compiled dataset to local `csv` files, and export it as a `dgl`
  graph.
