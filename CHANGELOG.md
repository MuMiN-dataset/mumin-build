# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


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
