# Image Reconstruction from Sparsely Sampled *k*-space Data

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE.md)
[![ORGANIZATION](https://img.shields.io/badge/Microsoft%20Research-0078d4?style=flat&logo=microsoft&logoColor=white)](https://www.microsoft.com/en-us/research/)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

The reconstructor module attempts to reconstruct an inverse fourier transform from sparsely sampled $k$-space data. Based on prior work, we implement both [variational network (VarNet)](https://openreview.net/forum?id=eAkOp9Oet5y) and [UNet](https://arxiv.org/abs/2105.06460) approaches to the reconstruction problem.

## Dataset and Data Transforms

The [`ReconstructorDataset`](data/dataset.py) class is a subclass of `torch.utils.data.Dataset` that is used to read in and store both singlecoil and multicoil data samples. The [`ReconstructorDataTransform`](data/transform.py) class masks training and validation data before passing a sample into the model. Specifically, here are the following tunable parameters of interest:

  - `center_fractions` : fraction of the center of $k$-space to always have sampled. This is to ensure that a subset of low-frequency data is always sampled.
  - `center_crop` : an optional size to crop the two-dimensional $k$-space data to. If center crop is anything other than the default `(640, 368)` maximum size, then the reconstruction target is appropriately adjusted.
  - `min_lines_acquired` : minimum number of unmasked $k$-space lines for any particular slice.

More broadly, the data transform prepares fully-sampled training and validation datasets by applying a sparsely sampled $k$-space mask to feed into the reconstructor neural network. The `min_lines_acquired` parameter affords the option to tune the minimum number of $k$-space lines that must be acquired by a scanner first before a reconstruction is attempted. Because the sparse sampling mask is already provided as part of the training and challenge fastMRI datasets, no mask is generated in the data transform. Only Cartesian line sampling is implemented to remain consistent with the fastMRI dataset.

The [`DataModule`](pl_modules/data_module.py) supports batch sizes greater than 1 with a custom `collate_fn()` implementation to handle variable *k*-space sizes within the same batch. Specifically, `collate_fn()` crops all *k*-space datasets within a particular batch to the minimum 2D acquired *k*-space dimension of the batch before stacking them into a single batch `torch.Tensor`.

## Model Architecture

TODO

## Results

TODO

## Installation

Please follow the installation instructions from this [parent README](../README.md). To train and test this model, navigate to the `reconstructor` directory and run the `main.py` program.

```
cd reconstructor 
python main.py <data_folder> --optional_arguments
```

To do model inference using our trained reconstruction model, navigate to the `reconstructor` directory and run the `infer.py` program.

```
cd reconstructor
python infer.py <data_folder_or_file> --optional_arguments
```

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## License

This repository is MIT licensed (see [LICENSE](../LICENSE.md)).