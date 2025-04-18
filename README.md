# Simulation-Calibrated Scientific Machine Learning (SCaSML)

SCaSML is a novel approach for solving high-dimensional partial differential equations (PDEs) that combines the dimension insensitivity of neural networks with the transparent, unbiased, and physical estimation from simulation solvers, leveraging inference time computation for scalable and refined predictions. This repo is a PINN implementation of this algorithm, another Gaussian Process implementation is on https://github.com/Francis-Fan-create/SCaSML_GP.git .

## Abstract

High-dimensional partial differential equations (PDEs) pose significant computational challenges across fields ranging from quantum chemistry to economics and finance. Although scientific machine learning (SciML) techniques offer approximate solutions, they often suffer from bias and neglect crucial physical insights. Inspired by inference-time scaling strategies in language models, we propose Simulation-Calibrated Scientific Machine Learning (SCaSML), a physics-informed framework that dynamically refines and debiases the SCiML predictions during inference by enforcing the physical laws. SCaSML leverages derived new physical laws that quantifies systematic errors and employs Monte Carlo solvers based on the Feynman–Kac and Elworthy–Bismut–Li formulas to dynamically correct the prediction. Both numerical and theoretical analysis confirms enhanced convergence rates via compute-optimal inference methods. Our numerical experiments demonstrate that SCaSML reduces errors by 20–50\% compared to the base surrogate model, establishing it as the first algorithm to refine approximated solutions to high-dimensional PDE during inference.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/francis-fan-create/scasml.git 
cd scasml
```

2. Pip install related repos:

```bash
pip install -r requirements.txt
```
## Custom Usage

To set up a new SCaSML solver for specific equations, follow these steps:

1. Configure your equation in `equations/equations.py` using the `Grad_Dependent_Nonlinear` class as a guide.

2. In the `optimizers/` directory, create a new `.py` file to customize your training process, following the `Adam.py` template.

3. To use test methods other than `tests/SimpleUniform.py`, create a new one in the `tests/` directory using the `SimpleUniform.py` format.

4. Copy `experiment_run.py` (Adam training) from `results/Grad_Dependent_Nonlinear/(certain dimension)/` to `results/(your equation)/(certain dimension)/`, replacing all "Grad_Dependent_Nonlinear" with your equation's name.

5. Review beginning lines to enable or disable wandb online logging.

6. Run the experiment:

```bash
python results/(your equation)/(certain dimension)/experiment_run.py
```

9. View your results in the `results/(your equation)/(certain dimension)/` folder and on wandb

10. Replace `results/` by `results_full_history/` to replace the quadrature MLP solver by the full_history MLP solver.
    
11. If you still have problems, please submit it to **Issues**.

## Project Structure

- `equations/`: Contains equation definitions
- `utils/`: Some decorators to log callings of functions(No longer supported)
- `optimizers/`: Custom optimization algorithms
- `tests/`: Test methods for evaluating the solver
- `results/`: Experiment results for quadrature MLP
- `result_full_history/`: Experiment results for full history MLP
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

Zexi Fan: fanzexi_francis@stu.pku.edu.cn

**Collaborators:**

Yan Sun: yansun414@gmail.com

Yiping Lu: yiping.lu@northwestern.edu













































