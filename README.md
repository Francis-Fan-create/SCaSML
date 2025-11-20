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

## Basic Usage
Find `experiment_run.py` under `results/(example equation)/(certain dimension)/` or `results_full_history/(your equation)/(certain dimension)/`

## Custom Usage

To set up a new SCaSML solver for specific equations, follow these steps:

1.  **Configure Equation:** Define your equation in `equations/equations.py`. Use the `Grad_Dependent_Nonlinear` class as a template.
2.  **Customize Training:** In the `optimizers/` directory, create a new Python file for your custom training process. Follow the structure of `Adam.py`.
3.  **Define Test Method:** If you need a test method different from `tests/SimpleUniform.py`, create a new one in the `tests/` directory, using `SimpleUniform.py` as a guide.
4.  **Set Up Experiment Script:** Copy `experiment_run.py` from `results/Grad_Dependent_Nonlinear/(certain dimension)/` to `results/(your equation)/(certain dimension)/`. Replace all occurrences of "Grad_Dependent_Nonlinear" with the name of your equation, and add your customized test in it.
5.  **Configure Logging:** Review the beginning lines of your new `experiment_run.py` to enable or disable Weights & Biases (wandb) online logging.
6.  **Run Experiment:** Execute the script from your terminal:
    ```bash
    python results/(your equation)/(certain dimension)/experiment_run.py
    ```
7.  **View Results:** Check the output in the `results/(your equation)/(certain dimension)/` folder and on the wandb dashboard (if enabled).
8.  **Use Full History Solver (Optional):** To use the full history MLP solver instead of the quadrature MLP solver, replace `results/` with `results_full_history/` in the paths mentioned above (steps 4, 6, 7).
9.  **Troubleshooting:** If you encounter issues, please submit them to the project's **Issues** page on GitHub.
10. **Caveat:** Sometimes DeepXDE cannot find the jax backend in the first run. When such an error occurs, please relaunch the program again. 

## Project Structure

- `equations/`: Contains equation definitions
- `utils/`: Includes logging and plotting tools
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

If you find SCaSML useful in your research, please consider citing the following paper:

Zexi Fan, Yan Sun, Shihao Yang, Yiping Lu. (2025). *Physics-Informed Inference Time Scaling via Simulation-Calibrated Scientific Machine Learning*. arXiv preprint arXiv:2504.16172. [https://arxiv.org/abs/2504.16172](https://arxiv.org/abs/2504.16172)

## Contact

**Maintainer&First Author:**

Zexi Fan: fanzexi_francis@stu.pku.edu.cn

**Collaborators:**

Yan Sun: yansun414@gmail.com

Yiping Lu: yiping.lu@northwestern.edu













































