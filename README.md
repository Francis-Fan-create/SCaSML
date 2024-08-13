# Simulation-Calibrated Scientific Machine Learning (SCaSML)

SCaSML is a novel approach for solving high-dimensional partial differential equations (PDEs) that combines the dimension insensitivity of neural networks with the transparent, unbiased, and physical estimation from simulation solvers.

## Abstract

Addressing high-dimensional PDEs has long been challenging due to the 'curse of dimensionality'. Deep learning methods, while promising, often suffer from bias, non-transparency, and insufficient physics embedding. To address these, we propose Simulation Calibrated Scientific Machine Learning (SCaSML), integrating neural network's dimension insensitivity with transparent, unbiased and physical estimation from simulation solvers. 

We train Physics-Informed Neural Networks (PINNs) to provide initial solution estimates, whose errors, governed by a new PDE, are corrected via Monte Carlo solvers using the Feynman-Kac and Elworthy-Bismut-Li formulae. We prove SCaSML's rate improvement for two state-of-the-art solvers, quadrature multilevel Picard and full-history multilevel Picard. 

Numerical experiments on various high-dimensional PDEs, including the Hamilton-Jacobi-Bellman and Diffusion-Reaction equations, etc., confirm our theoretical results in terms of accuracy. This method advances fields like economics, finance, operations research, and physics by comprehensively considering all involved elements as agents, assets, and resources.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/francis-fan-create/scasml.git 
cd scasml
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To set up a new SCaSML solver for specific equations, follow these steps:

1. Configure your equation in `equations/equations.py` using the `Explicit_Solution_Example` class as a guide.

2. In the `models/` directory, create a new `.py` file and define your network structure, taking `FNN.py` as an example. Ensure to set the `self.regularizer` parameter.

3. In the `optimizers/` directory, create a new `.py` file to customize your training process, following the `Adam_LBFGS.py` template.

4. To use test methods other than `tests/NormalSphere.py`, create a new one in the `tests/` directory using the `NormalSphere.py` format.

5. Under `results/`, create a new folder named after your equation. Open a new `.md` file, paste the wandb key for your project related to this equation, rename the file to `wandbkey.md`, and save it in `results/(your equation)/`.

6. Copy `experiment_run.py` (Adam_LBFGS training) from `results/Explicit_Solution_Example/(certain dimension)/` to `results/(your equation)/(certain dimension)/`, replacing all "Explicit_Solution_Example" with your equation's name.

7. Review lines 39-42 to enable or disable wandb online logging.

8. Run the experiment:

```bash
python results/(your equation)/(certain dimension)/experiment_run.py
```

9. View your results in the `results/(your equation)/(certain dimension)/` folder and on wandb
10. (Optional) Repeat steps 6-9 for `further_training.py` after the weights of `experiment_run.py` are generated to get further trained weights using L_inf training. 
10. Replace `results/` by `results_full_history/` to replace the quadrature MLP solver by the full_history MLP solver.
11. **Caveat:** We highly recommend users run 6-11 steps under debug mode since some parameters are dependent on the equation itself and might need to be changed. However, we have put enough annotation on these lines, which can be easily fixed. (This feature will be fixed in the following versions.)
12. If you still have problems, please submit it to **Issues**.

## Project Structure

- `equations/`: Contains equation definitions
- `models/`: Neural network model definitions
- `optimizers/`: Custom optimization algorithms
- `tests/`: Test methods for evaluating the solver
- `results/`: Experiment results and wandb keys
- `solvers/`: Implementation of SCaSML and other solvers

## Contributing

Contributions to SCaSML are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## Citation



## Contact

**Maintainer&First Author:**

Zexi Fan: 2200010816@stu.pku.edu.cn

**Collaborators:**

Yan Sun: yansun414@gmail.com

Yiping Lu: yiping.lu@northwestern.edu














































