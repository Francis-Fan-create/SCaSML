# ScaML-Experiment

ScaML Solver for high dimensional PDE solving
-------


### Quick start

* Clone the repo and create a [conda environment](https://anthonyhu.github.io/python-environment) by running: `conda env create`.
* Run training with `python run_training.py --config experiments/cifar.yml`. This will download CIFAR10 data in a new
folder `./cifar10/dataset` and save the experiment outputs in `./cifar10/experiments/`.

For a new project, create a new trainer class in the `trainers` folder and implement the abstract methods of the general
`Trainer` class. See `trainers/trainer_cifar.py` for a detailed example.
