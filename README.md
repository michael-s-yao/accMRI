# A Practical Framework for Accelerated MRI

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![ORGANIZATION](https://img.shields.io/badge/Microsoft%20Research-0078d4?style=flat&logo=microsoft&logoColor=white)](https://www.microsoft.com/en-us/research/)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

In this work, we provide a practical framework for accelerated MRI imaging that builds on previous work to learn the following three tasks:

  1. [`reconstructor`](reconstructor): Image reconstruction from sparsely sampled $k$-space data with co-trained sensitivity map estimation.
  2. [`sampler`](sampler): Active $k$-space sampling to guide sequential MRI acquisition using reinforcement learning.
  3. [`discriminator`](discriminator): $k$-space fidelity quantification to inform subsequent *re*-sampling of previously acquired $k$-space data.

Our model is (1) agnostic to the input $k$-space sampling pattern, (2) built with the time constants and healthcare costs of clinically-relevant MRI data acquisition and network speeds in mind, and (3) considerant of real-world sources of noise and data corruption. Altogether, our framework can intelligently inform sequential acquision steps while reconstructing the best possible image from the currently available data, and is a step forward towards clinical applicability of accelerated MRI.

We also introduce a sandbox environment for simulating real-time accelerated MRI in future clinical settings (see [`simul`](simul)). 

## Installation

To install and run our code, first clone the `accMRI` repository.

```
git clone https://github.com/michael-s-yao/accMRI
cd accMRI
```

Next, create a virtual environment and install the relevant dependencies.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Download and organize the fastMRI single- and/or multi- coil data from the [fastMRI website](https://fastmri.med.nyu.edu/). For example the knee datasets should be organized as follows:

```
└── data
    ├── knee       
        ├── knee_multicoil_train
        ├── knee_multicoil_val
        ├── knee_multicoil_test
        ├── knee_singlecoil_train
        ├── knee_singlecoil_val
        ├── knee_singlecoil_test
```

Note that the parent directory `data` is the same as the `data` directory in this repository. For the purposes of testing our models, we note that the fastMRI `test` datasets can only be used for evaluating our reconstructor. Other models, such as those used for active $k$-space sampling and $k$-space fidelity quantification, use a subset of the associated `train` datasets for testing. Each of the relevant modules can be trained and tested by navigating to the associated directory and running the child `main.py` program:

```
python main.py
```

Each module also has an associated `README.md` file for reference.

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## License

This repository is MIT licensed (see [LICENSE](LICENSE.md)). Portions of the code in this repository is adapted from the following sources:

  1. [fastMRI repository](https://github.com/facebookresearch/fastMRI), a collaborative research project between Facebook AI Research (FAIR) and NYU Langone Health.
  2. [active-mri-acquisition](https://github.com/facebookresearch/active-mri-acquisition), a reinforcement learning environment to facilitate research on active MRI acquisition from Facebook AI Research (FAIR).
  3. [SeqMRI](https://github.com/tianweiy/SeqMRI), based on the 2021 paper *End-to-End Sequential Sampling and Reconstruction for MR Imaging* by Tianwei Yin and colleagues.
