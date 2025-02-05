# Simulation-Calibrated Scientific Machine Learning (SCaSML)

SCaSML is a novel approach for solving high-dimensional partial differential equations (PDEs) that combines the dimension insensitivity of neural networks with the transparent, unbiased, and physical estimation from simulation solvers, leveragin inference time computation for scalable and refined predictions. This repo is a PINN implementation of this algorithm, another Gaussian Kernel Regression implementation is on https://github.com/Francis-Fan-create/SCaSML_GP.git .

## Abstract

High-dimensional partial differential equations (PDEs) are fundamental to many scientific and engineering applications yet remain notoriously difficult to solve due to the "curse of dimensionality." While scientific machine learning (SciML) methods offer promising avenues for approximation, they frequently suffer from bias, non-transparency, and insufficient incorporation of physical laws. Inspired by breakthroughs in language models that leverage inference time computation for scalable and refined predictions, we introduce Simulation Calibrated Scientific Machine Learning (SCaSML). This framework marries the dimension-insensitive capabilities of surrogate models with the transparent, physics-based accuracy of simulation solvers. In SCaSML, surrogate models such as Physics-Informed Neural Networks (PINNs) and Gaussian Process are first employed to generate initial solution estimates. The systematic errors in these estimates—governed by a newly derived PDE—are subsequently corrected during inference time using Monte Carlo solvers based on the Feynman-Kac and Elworthy-Bismut-Li formula. Our extensive numerical experiments demonstrate the scalability of inference time corrections over broad solution domains for various high-dimensional PDEs, including the Hamilton-Jacobi-Bellman equation and other gradient-dependent semilinear equations. This work underscores the critical role of physics--informed inference time scaling in enhancing predictive performance, thereby significantly advancing the reliability and interpretability of high-dimensional PDE solvers in scientific computing.

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
## Basic Usage

You can run experiments for Grad_Dependent_Nonlinear_Equation and Linear_HJB by the following script:

1. Run the experiment for 20d and 40d:

```bash
chmod +x low_dim_group.sh
./low_dim_group.sh
```

2. Run the experiment for 60d and 80d:

```bash
chmod +x high_dim_group.sh
./high_dim_group.sh
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

Zexi Fan: 2200010816@stu.pku.edu.cn

**Collaborators:**

Yan Sun: yansun414@gmail.com

Yiping Lu: yiping.lu@northwestern.edu














































