# Accelerated MRI Simulation Environment

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE.md)
[![ORGANIZATION](https://img.shields.io/badge/Microsoft-0078d4?style=flat&logo=microsoft&logoColor=white)](https://www.microsoft.com/en-us/research/)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

This is a sandbox environment for simulating real-time accelerated MRI. The environment features two agents on parallel processes: (1) a simulated MRI scanner that 'acquires' requested $k$-space data and randomly introduces patient rigid motion, $k$-space spikes, RF contamination, and other potential sources of noise, and (2) a simulated compute cluster that reconstructs images from the data acquired by the MRI scanner and queues requests for additional $k$-space line acquisitions.

## Dataset and Data Transforms

TODO

## Model Architecture

TODO

## Installation

Please follow the installation instructions from this [parent README](../README.md). This environment makes use of the [`reconstructor`](../reconstructor), [`sampler`](../sampler), and [`discriminator`](../discriminator) models within this repository. To run the simulation, navigate to the [`simul`](../simul) directory and run the `main.py` program.

```
cd simul 
python main.py
```

For an illustrative, toy two-agent example, we also provide `toy.py` as well:

```
python toy.py
```

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## License

This repository is MIT licensed (see [LICENSE](../LICENSE.md)).
