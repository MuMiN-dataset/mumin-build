# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


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
