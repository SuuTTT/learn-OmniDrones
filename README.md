![Visualization of OmniDrones](docs/source/_static/visualization.jpg)

---

# OmniDrones

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-2022.2.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


*OmniDrones* is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. Built on [Nvidia Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), *OmniDrones* features highly efficient and flxeible simulation that can be adopted for various research purposes. We also provide a suite of benchmark tasks and algorithm baselines to provide preliminary results for subsequent works.

For usage and more details, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/).

## Announcement 2023-09-25

The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. As the next version of 
Isaac Sim (2023.1.0) is expected to bring susbtantial changes but is not yet available, the APIs and usage 
of **OmniDrones** are subject to change. We will try our best to keep the documentation up-to-date.


## Citation

Please cite [this paper](https://arxiv.org/abs/2309.12825) if you use *OmniDrones* in your work:

```
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control}, 
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```


## Ackowledgement

Some of the abstractions and implementation was heavily inspired by [Isaac Orbit](https://github.com/NVIDIA-Omniverse/Orbit).