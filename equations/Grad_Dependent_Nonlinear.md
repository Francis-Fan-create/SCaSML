# Grad_Dependent_Nonlinear

We solve the following [PDE](https://arxiv.org/abs/1708.03223):

$$
\frac{\partial u}{\partial t}+(\sigma^2u(x,t)-\frac 1d -\frac{\sigma_0^2}{2})div_x u(x,t)+\frac {\sigma_0^2}2 \Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$

whose terminal condition is given by

$$
u(x,T)=g(x):=\frac{\exp(T+\sum\limits_{i=1}^d x_i)}{1+\exp(T+\sum\limits_{i=1}^d x_i)}
$$

without any boundary constraints.



Then nonlinear term is given by
$$
F(u,z)(x,t)=\sigma_0 u\sum_i z
$$


This PDE has an explicit solution at time $t$:
$$
u(x,t)=\frac{\exp(t+\sum\limits_{i=1}^d x_i)}{1+\exp(t+\sum\limits_{i=1}^d x_i)}
$$

which is our target in this section.



Specifically, we consider the problem for

$$
d=20, 40, 60, 80, \mu=-(\frac1d+\frac{\sigma_0^2}{2}), \sigma_=0.25, D=[-0.5,0.5]^{20,40,60,80}, s=0, T=0.5
$$
