# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
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
