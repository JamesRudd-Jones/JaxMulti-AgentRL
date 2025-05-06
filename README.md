# Jax Multi-Agent RL

The beginnings of a benchmark suite of Multi-Agent Reinforcement Learning.

This is not the first Jax based package, however we found that existing approachs didn't fully meet our requirements typically focusing on centralised methodologies. 
This package focuses on decentralised approaches and enables easy comparisons between algorithms especailly when working in sub-domains such as Cooperative RL/Mixed Motive RL/Opponent Modelling/Opponent Shaping.
However, we aim to keep it quite general and provide a consistent location for many of the SOTA methods today, enabling much easier benchmarking of new algorithmic improvements as well as provide a useful testbed for practitioners.
The focus is on MARL but Single-Agent RL approaches are easily incorporated using a wrapper.
We are very open to suggestions and contributions so please get in touch!


### Why Jax?

Jax has become almost a staple in Model-Free RL due to the extreme potential of computational speed-ups, in part due to the vectorisation of environments enabling both RL agent and environment to be run together on a GPU. 

|    Algorithm     |                                                         Reference                                                         |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------:|
| PPO |                           [Paper]([https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf](https://arxiv.org/pdf/1707.06347))                           |
|       DDPG        | [Paper](https://arxiv.org/pdf/1509.02971) |
|       IDQN       |      [Paper](https://arxiv.org/pdf/1312.5602v1)       |
|      ERSAC       |                [Paper](https://arxiv.org/pdf/2302.09339)                |
|      VLITE       |[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6a6e010edde1b8f2812f558b67a1974e-Paper-Conference.pdf)                |
|      MFOS       |                [Paper](https://proceedings.mlr.press/v162/lu22d/lu22d.pdf)                |
|      MELIBA       |                [Paper](https://arxiv.org/pdf/2101.03864)                |
|      ROMMEO       |                [Paper](https://arxiv.org/pdf/1905.08087)                |
|      QMIX       |                [Paper](https://arxiv.org/pdf/1803.11485)                |
|      PR2       |                [Paper](https://arxiv.org/pdf/1901.09207)                |


Currently implemented environments:

We predominantly use the [JaxMARL](https://github.com/FLAIROx/JaxMARL) environments but with a wrapper can also use [Gymnax](https://github.com/RobertTLange/gymnax).
Further environments that are amenable to this setup are from [Pax](https://github.com/ucl-dark/pax).

Environments can be added to the folder and eventually will be easy to incorporate alongside environments from other packages.
We are still working on this, currently extra functions are required in the utils for each environment and so will need to be edited if you add your own.


## Basic Usage

This is a multi-file implementation as we think it enables easier comparisons than single-file implementations, but we try and keep most of the coniditional logic abstracted to the unique 'agent' files, rather than housing this type of logic in the shared running files!

This means at it's core all that needs to be edited when adding new algorithms is the specific agent implementation folder. 
New algorithms can be added in the 'agent' folder with their own directory/module stating the algorithm name. This must match the file name. The file itself has an agent class that imports from 'agent_base.py' and follows the naming convention of '\<Algorithm Name\>Agent'.

We use [Ml Collections](https://github.com/google/ml_collections) for Configs as it allows easy use with [ABSL](https://github.com/abseil/abseil-py) flags when running on a cluster for hyperparameter sweeps. There is a general config file and each agent must have it's own config file.

The number of agents can be set in the main config file and the specific agent types are similarly set there. This enables any number of combinations of decenteralised agents working (or competing) together in an environment. Setting there to be only 1 agent reduces this down to SARL, it is as easy as that!


## Installation

This is a Python package that should be installed into a virtual environment. Start by cloning this repo from Github:

git clone https://github.com/JamesRudd-Jones/JaxMulti-AgentRL.git

The package can then be installed into a virtual environment by adding it as a local dependency. We recommend [PDM](https://pdm-project.org/en/latest/) or [Poetry](https://python-poetry.org/).


## Contributing

We actively welcome contributions!

Please get in touch if you want to add an environment or algorithm, or have any questions regarding the implementations.
We also welcome any feedback regarding documentation!

For bug reports use Github's issues to track bugs, open a new issue and we will endeavour to get it fixed soon! 


## Future Roadmap

- Add CTDE functionality, to easily switch for example between MAPPO and IPPO.
- Enable an easy toggle for the use of an image based versus a flatenned state representation for all agents
- Similarly, enable agents to arbitrarily decide if they require sequential modelling or not (e.g. through the use of an RNN)
- Clean up the memory state to house all possible tracked values
- Add agent configs to the wandb call
