# Image Reconstruction from Sparse $k$-space Data

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE.md)
[![ORGANIZATION](https://img.shields.io/badge/Microsoft-0078d4?style=flat&logo=microsoft&logoColor=white)](https://www.microsoft.com/en-us/research/)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

The reconstructor module attempts to reconstruct an inverse fourier transform from sparsely sampled $k$-space data. Based on prior work, we implement both [variational network (VarNet)](https://openreview.net/forum?id=eAkOp9Oet5y) and [UNet](https://arxiv.org/abs/2105.06460) approaches to the reconstruction problem.

## Dataset and Data Transforms

TODO

## Model Architecture

TODO

## Installation

Please follow the installation instructions from this [parent README](../README.md). To train and test this model, navigate to the `reconstructor` directory and run the `main.py` program.

```
cd reconstructor 
python main.py
```

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## License

This repository is MIT licensed (see [LICENSE](../LICENSE.md)).