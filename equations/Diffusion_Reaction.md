# Diffusion_Reaction

We solve the following [PDE]([[1907.08272\] Weak Adversarial Networks for High-dimensional Partial Differential Equations](https://arxiv.org/abs/1907.08272))(labeled as Eq(21) in the original paper, and we reverse the sign of the solution)

$$
\frac{\partial u}{\partial t}+\Delta_x u(x,t)+(u^2+(\pi^{2}-2)\sin{(\frac{\pi}{2}x_{1})}\cos(\frac{\pi}{2}x_{2})e^{-t}-4\sin^{2}{(\frac{\pi}{2}x_{1})}\cos(\frac{\pi}{2}x_{2})e^{-2t})=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose terminal condition is given by


$$
u(x,T)=g(x):=2\operatorname{sin}(\frac{\pi}{2}x_{1})\operatorname{cos}(\frac{\pi}{2}x_{2})e^{-T},
$$

with Dirichlet boundary constraint.

The nonlinear term is given by
$$
F(u,z)(x,t)=u^2+(\pi^{2}-2)\sin{(\frac{\pi}{2}x_{1})}\cos(\frac{\pi}{2}x_{2})e^{-t}-4\sin^{2}{(\frac{\pi}{2}x_{1})}\cos(\frac{\pi}{2}x_{2})e^{-2t}
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=2\operatorname{sin}(\frac{\pi}{2}x_{1})\operatorname{cos}(\frac{\pi}{2}x_{2})e^{-t}.
$$

which is our target in this section.

Specifically, we consider the problem for

$$
d=20, 40, 60, 80,\mu=0, \sigma=\sqrt{2}, D=[-0.5,0.5]^{20,40,60,80}, s=0, T=0.5
$$

